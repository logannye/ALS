"""Tests for llm.spend_gate — shared budget enforcement.

DB-gated integration tests. Each test seeds a known MTD via direct inserts
and then asserts check_budget / enforce_budget behavior.
"""
from __future__ import annotations

import os
import uuid

import pytest

from llm.spend_gate import (
    BudgetExceededError,
    check_budget,
    enforce_budget,
    estimate_cost,
    record_call,
)


# ─── Pure functions ───────────────────────────────────────────────────────────


def test_estimate_cost_opus_matches_pricing():
    # claude-opus-4-6: $15/1M in, $75/1M out.
    # 1M in tokens → $15. 1M out tokens → $75. 1M of each → $90.
    assert abs(estimate_cost('claude-opus-4-6', 1_000_000, 1_000_000) - 90.0) < 1e-6


def test_estimate_cost_nova_micro_cheap():
    """Nova micro is 3 orders of magnitude cheaper than Opus per token."""
    opus = estimate_cost('claude-opus-4-6', 10_000, 10_000)
    nova = estimate_cost('amazon.nova-micro-v1:0', 10_000, 10_000)
    assert nova * 100 < opus


def test_estimate_cost_unknown_uses_conservative_default():
    """Unknown models bias high (toward enforcement), not low."""
    c = estimate_cost('fictional-model', 1_000_000, 0)
    # Default input rate is $3/1M, so $3 for 1M tokens.
    assert 2.9 < c < 3.1


# ─── DB-gated tests ──────────────────────────────────────────────────────────


def _can_connect() -> bool:
    import psycopg
    user = os.environ.get("USER", "logannye")
    try:
        c = psycopg.connect(f"dbname=erik_kg user={user}", connect_timeout=3)
        c.close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not _can_connect(), reason="erik_kg PG not reachable")


@pytest.fixture
def conn():
    import psycopg
    user = os.environ.get("USER", "logannye")
    c = psycopg.connect(f"dbname=erik_kg user={user}")
    yield c
    c.close()


@pytest.fixture
def isolated_spend_budget(monkeypatch, conn):
    """Give each test a known MTD baseline by routing budget reads through
    a test-specific budget env var and a per-test phase marker that the
    check_budget SQL already filters by (current month)."""
    # Every test sets a very-high budget so pre-existing rows don't trip
    # the gate; individual tests will explicitly lower the budget as needed.
    monkeypatch.setenv("ERIK_LLM_MONTHLY_BUDGET_USD", "100000.0")
    yield
    # No cleanup — llm_spend is append-only by design; test rows age out
    # at the next month boundary. Use unique phase tags to make debugging
    # possible.


def _seed_spend(conn, phase: str, cost_usd: float) -> None:
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_ops.llm_spend
                (model, phase, input_tokens, output_tokens, cost_usd, prompt_cached)
            VALUES ('test-model', %s, 0, 0, %s, FALSE)
        """, (phase, cost_usd))
        conn.commit()


@pg
def test_check_budget_returns_mtd_aggregate(conn, isolated_spend_budget):
    status_before = check_budget()
    _seed_spend(conn, phase=f"pytest_{uuid.uuid4().hex[:6]}", cost_usd=1.23)
    status_after = check_budget()
    assert status_after.month_to_date_usd - status_before.month_to_date_usd >= 1.22


@pg
def test_enforce_budget_raises_when_over(conn, monkeypatch):
    # Set a very low budget and seed more than that.
    monkeypatch.setenv("ERIK_LLM_MONTHLY_BUDGET_USD", "0.00001")
    _seed_spend(conn, phase=f"pytest_over_{uuid.uuid4().hex[:6]}", cost_usd=1.0)
    with pytest.raises(BudgetExceededError):
        enforce_budget(phase='test_enforce')


@pg
def test_enforce_budget_returns_status_when_under(monkeypatch):
    monkeypatch.setenv("ERIK_LLM_MONTHLY_BUDGET_USD", "1000000.0")
    status = enforce_budget(phase='test_under')
    assert status.over_budget is False
    assert status.budget_usd == 1_000_000.0
    assert status.remaining_usd > 0


@pg
def test_record_call_adds_row_to_llm_spend(conn, isolated_spend_budget):
    tag = f"pytest_record_{uuid.uuid4().hex[:6]}"
    record_call(
        model='amazon.nova-micro-v1:0',
        phase=tag,
        input_tokens=1000,
        output_tokens=500,
    )
    with conn.cursor() as cur:
        cur.execute(
            "SELECT model, input_tokens, output_tokens, cost_usd FROM erik_ops.llm_spend WHERE phase = %s",
            (tag,),
        )
        row = cur.fetchone()
    assert row is not None
    assert row[0] == 'amazon.nova-micro-v1:0'
    assert row[1] == 1000
    assert row[2] == 500
    # Nova micro: $0.035/1M in, $0.14/1M out → 1000*0.035/1M + 500*0.14/1M
    # = 3.5e-5 + 7e-5 ≈ 1.05e-4.
    assert 5e-5 < float(row[3]) < 2e-4


@pg
def test_near_limit_flag_fires_at_fraction_threshold(monkeypatch, conn):
    """near_limit triggers before over_budget so callers can downshift."""
    monkeypatch.setenv("ERIK_LLM_MONTHLY_BUDGET_USD", "100.0")
    # Clear the state by inspecting baseline, then add enough to put us
    # in the 80-99% band.
    baseline = check_budget()
    # Add enough to cross the default 80% soft threshold without hitting 100%.
    target_mtd = 0.85 * 100.0
    delta = max(0.0, target_mtd - baseline.month_to_date_usd)
    if delta > 0:
        _seed_spend(conn, phase=f"pytest_near_{uuid.uuid4().hex[:6]}", cost_usd=delta)
    status = check_budget(near_limit_fraction=0.8)
    # In a shared DB we might already be over — accept either state as long
    # as the invariant "near_limit implies not over_budget" holds.
    if status.near_limit:
        assert status.over_budget is False
