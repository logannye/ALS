"""Tests for SCMWriter.submit_effect_update.

DB-gated integration tests. Cover the core invariants:
  * Round-trip: submit → active edge has effect_mean populated.
  * Event log: an 'effect_updated' scm_write_log row is emitted.
  * Silent skip on inactive edges (stale update).
  * Silent skip on nonexistent edges.
"""
from __future__ import annotations

import os
import time

import pytest

from world_model.scm_writer import (
    EffectDistribution,
    EffectUpdateRequest,
    SCMWriterLockContention,
    get_scm_writer,
    shutdown_scm_writer,
)


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
def running_writer():
    import world_model.scm_writer as m
    m._singleton = None
    writer = get_scm_writer()
    try:
        writer.start()
    except SCMWriterLockContention:
        pytest.skip("SCMWriter lock contended")
    yield writer
    shutdown_scm_writer(timeout=5.0)


def _seed_null_effect_edge(conn, status: str = 'active') -> int:
    """Insert a pair of nodes + an scm_edge with effect_mean=NULL."""
    import uuid
    tag = uuid.uuid4().hex[:8]
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'treatment', 'compound') RETURNING id""",
                    (f"test:ef:compound:{tag}",))
        src = cur.fetchone()[0]
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'covariate', 'gene') RETURNING id""",
                    (f"test:ef:gene:{tag}",))
        tgt = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, %s)
            RETURNING id
        """, (src, tgt, status))
        eid = int(cur.fetchone()[0])
        conn.commit()
    return eid


def _wait_for_effect(conn, edge_id: int, timeout: float = 5.0) -> tuple | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT effect_mean, effect_std, effect_scale FROM erik_ops.scm_edges WHERE id = %s",
                (edge_id,),
            )
            row = cur.fetchone()
        if row and row[0] is not None:
            return row
        time.sleep(0.1)
    return None


@pg
def test_submit_effect_update_populates_effect_on_active_edge(conn, running_writer):
    edge_id = _seed_null_effect_edge(conn, status='active')
    running_writer.submit_effect_update(EffectUpdateRequest(
        scm_edge_id=edge_id,
        effect=EffectDistribution(
            mean=2.5, std=0.3, ci_lower=2.2, ci_upper=2.8,
            scale='ic50_log_nm',
        ),
        source='pytest',
        daemon_source='pytest',
    ))
    row = _wait_for_effect(conn, edge_id)
    assert row is not None
    assert abs(float(row[0]) - 2.5) < 1e-6
    assert abs(float(row[1]) - 0.3) < 1e-6
    assert row[2] == 'ic50_log_nm'


@pg
def test_submit_effect_update_emits_write_log_event(conn, running_writer):
    edge_id = _seed_null_effect_edge(conn, status='active')
    running_writer.submit_effect_update(EffectUpdateRequest(
        scm_edge_id=edge_id,
        effect=EffectDistribution(mean=3.0, std=0.4, scale='ic50_log_nm'),
        source='pytest',
    ))
    _wait_for_effect(conn, edge_id)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT operation, target_id, payload
              FROM erik_ops.scm_write_log
             WHERE target_id = %s
               AND operation = 'effect_updated'
             ORDER BY id DESC LIMIT 1
        """, (edge_id,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == 'effect_updated'
    payload = row[2] if isinstance(row[2], dict) else {}
    assert payload.get('source') == 'pytest'
    assert payload.get('effect_scale') == 'ic50_log_nm'


@pg
def test_submit_effect_update_skips_inactive_edge(conn, running_writer):
    """Edges that are superseded between candidate-fetch and update must not mutate."""
    edge_id = _seed_null_effect_edge(conn, status='superseded')
    running_writer.submit_effect_update(EffectUpdateRequest(
        scm_edge_id=edge_id,
        effect=EffectDistribution(mean=2.5, std=0.3, scale='ic50_log_nm'),
        source='pytest',
    ))
    # Give the writer thread time to process.
    time.sleep(0.5)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT effect_mean FROM erik_ops.scm_edges WHERE id = %s", (edge_id,)
        )
        assert cur.fetchone()[0] is None  # still NULL — inactive edge left alone
        cur.execute("""
            SELECT COUNT(*) FROM erik_ops.scm_write_log
             WHERE target_id = %s AND operation = 'effect_updated'
        """, (edge_id,))
        assert cur.fetchone()[0] == 0


@pg
def test_submit_effect_update_rejects_zero_edge_id(running_writer):
    """submit path must raise; don't enqueue a nonsensical update."""
    from world_model.scm_writer import SCMWriterError
    with pytest.raises(SCMWriterError):
        running_writer.submit_effect_update(EffectUpdateRequest(
            scm_edge_id=0,
            effect=EffectDistribution(mean=2.5, std=0.3, scale='ic50_log_nm'),
        ))


@pg
def test_submit_effect_update_nonexistent_edge_is_silent_skip(conn, running_writer):
    """Submitting for an edge that was deleted (or never existed) must not crash
    the writer. The daemon recovers and continues."""
    running_writer.submit_effect_update(EffectUpdateRequest(
        scm_edge_id=2_000_000_000,  # almost certainly nonexistent
        effect=EffectDistribution(mean=2.5, std=0.3, scale='ic50_log_nm'),
        source='pytest',
    ))
    time.sleep(0.3)
    stats = running_writer.stats()
    # Writer must still be running — no uncaught exceptions killed the thread.
    assert stats['running'] is True
