"""Shared LLM spend gate — one source of truth for budget enforcement.

Why this exists:
  The ClaudeClient has had a budget check baked in since it was added,
  but BedrockLLM — which is now the primary inference path after the
  Bedrock-only migration — had zero enforcement. A bug in the research
  loop could burn an unbounded amount of money overnight because Nova
  calls went straight to AWS with no ceiling.

  This module moves budget enforcement to a single place and makes every
  LLM caller in the system route through it. New provider? Call
  ``spend_gate.check_budget()`` before the API call and
  ``spend_gate.record_call()`` after.

Schema dependency:
  Reads and writes ``erik_ops.llm_spend`` (created in tcg_schema.sql).
  Does not assume the caller has set up connection pooling — it uses
  db.pool.get_connection, same as every other daemon.

Pricing:
  Kept here rather than inside each client so that adjusting a model's
  price is a one-file change. All prices are USD per 1M tokens.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

from db.pool import get_connection

logger = logging.getLogger(__name__)


# ─── Pricing (USD per 1M tokens) ──────────────────────────────────────────────
#
# Anthropic Claude prices tracked separately from AWS Bedrock Nova prices.
# Update when a contract / tier changes. Cost of being wrong is small —
# the gate over-enforces budget (by using a conservative estimate) rather
# than under-enforcing.

_COST_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic direct API
    'claude-opus-4-6':       (15.0, 75.0),
    'claude-sonnet-4-6':     (3.0,  15.0),
    'claude-opus-4-7':       (15.0, 75.0),
    # AWS Bedrock Nova
    'amazon.nova-micro-v1:0': (0.035, 0.14),
    'amazon.nova-pro-v1:0':   (0.8,   3.2),
    'amazon.nova-lite-v1:0':  (0.06,  0.24),
}

# Fallback (conservative — biased high) for unknown models.
_DEFAULT_COST_PER_M = (3.0, 15.0)


# ─── Public API ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BudgetStatus:
    month_to_date_usd: float
    budget_usd: float
    over_budget: bool
    remaining_usd: float
    # True when remaining is below a caller-configurable soft threshold.
    # Callers may use this to downshift to cheaper models before the hard cap.
    near_limit: bool


def _effective_budget_usd() -> float:
    """Read the budget from the environment; fall back to a hard-coded
    ceiling so a missing env var cannot translate to "unlimited spend".

    ERIK_LLM_MONTHLY_BUDGET_USD overrides the hard default. The fallback
    of $150/month is deliberately conservative — a research loop that
    genuinely needs more should require an explicit env-var flip, not
    accidentally slip through."""
    raw = os.environ.get("ERIK_LLM_MONTHLY_BUDGET_USD")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 150.0


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a single call at the listed per-M-token rates."""
    in_rate, out_rate = _COST_PER_M_TOKENS.get(model, _DEFAULT_COST_PER_M)
    return (in_rate * input_tokens + out_rate * output_tokens) / 1_000_000.0


def month_to_date_usd() -> float:
    """Sum erik_ops.llm_spend.cost_usd for the current calendar month.

    Returns 0.0 on any DB error — fail-open for availability. Enforcement
    happens separately in check_budget(), which fails closed when possible.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COALESCE(SUM(cost_usd), 0.0)
                      FROM erik_ops.llm_spend
                     WHERE created_at > date_trunc('month', now())
                """)
                return float(cur.fetchone()[0] or 0.0)
    except Exception as e:
        logger.warning("spend_gate: mtd lookup failed — returning 0: %s", e)
        return 0.0


def check_budget(near_limit_fraction: float = 0.8) -> BudgetStatus:
    """Return current budget state without mutating anything.

    Args:
        near_limit_fraction: 0.0-1.0; BudgetStatus.near_limit fires when
          month-to-date spend crosses this fraction of the budget.

    The caller decides what to do with over_budget=True — typically,
    raise or return an "LLM refused" sentinel rather than calling the API.
    """
    mtd = month_to_date_usd()
    budget = _effective_budget_usd()
    remaining = max(0.0, budget - mtd)
    over = mtd >= budget
    near = (not over) and mtd >= near_limit_fraction * budget
    return BudgetStatus(
        month_to_date_usd=mtd,
        budget_usd=budget,
        over_budget=over,
        remaining_usd=remaining,
        near_limit=near,
    )


def record_call(
    model: str,
    phase: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: Optional[float] = None,
    prompt_cached: bool = False,
) -> None:
    """Log a call to erik_ops.llm_spend. Never raises.

    Estimates cost if not provided. Spend logging must never block a
    research-loop step, so exceptions are swallowed and logged.
    """
    if cost_usd is None:
        cost_usd = estimate_cost(model, input_tokens, output_tokens)
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_ops.llm_spend
                        (model, phase, input_tokens, output_tokens,
                         cost_usd, prompt_cached)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (model, phase, input_tokens, output_tokens,
                      cost_usd, prompt_cached))
                conn.commit()
    except Exception as e:
        logger.warning("spend_gate: failed to record call: %s", e)


class BudgetExceededError(Exception):
    """Raised by enforce_budget() when month-to-date >= configured budget."""


def enforce_budget(phase: str = 'unknown') -> BudgetStatus:
    """Return the current budget status; raise BudgetExceededError when over.

    Use this at the top of any LLM call path that should hard-stop when
    the monthly budget is exhausted. Safer than the {'budget_exceeded': True}
    sentinel pattern because it can't be silently ignored — an exception
    propagates up and the research loop's step wraps in its error handler.
    """
    status = check_budget()
    if status.over_budget:
        raise BudgetExceededError(
            f"monthly LLM budget exhausted (phase={phase}, "
            f"mtd=${status.month_to_date_usd:.2f}, "
            f"budget=${status.budget_usd:.2f})"
        )
    return status
