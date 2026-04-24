"""Integration tests for CompoundRefutationDaemon.

Focused on the event-loop invariants:
  * Cursor advances monotonically after each cycle.
  * Running twice on the same scm_write_log range does not double-apply.
  * Events for non-intervention edges are no-ops.
  * Events whose refuted edge doesn't exist anymore are no-ops.
"""
from __future__ import annotations

import json
import os
import uuid

import pytest

from daemons.compound_refutation_daemon import CompoundRefutationDaemon


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


def _seed_scm_edge(conn, is_candidate: bool, status: str = 'superseded') -> tuple[int, str, str]:
    """Insert a pair of nodes + a superseded/invalidated scm_edge. Returns (edge_id, src_entity, tgt_entity)."""
    tag = uuid.uuid4().hex[:8]
    src = f"test:cr:compound:{tag}"
    tgt = f"test:cr:mechanism:{tag}"
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'treatment', 'compound') RETURNING id""", (src,))
        src_node = cur.fetchone()[0]
        cur.execute("""INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
                       VALUES (%s, 'covariate', 'mechanism') RETURNING id""", (tgt,))
        tgt_node = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence,
                 is_intervention_candidate, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, %s, %s)
            RETURNING id
        """, (src_node, tgt_node, is_candidate, status))
        eid = int(cur.fetchone()[0])
        conn.commit()
    return eid, src, tgt


def _emit_write_log(conn, operation: str, edge_id: int,
                    superseded_by: int | None = None) -> int:
    payload = {'superseded_by': superseded_by} if superseded_by else {}
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_ops.scm_write_log(operation, target_id, daemon, payload)
            VALUES (%s, %s, 'pytest', %s)
            RETURNING id
        """, (operation, edge_id, json.dumps(payload)))
        log_id = int(cur.fetchone()[0])
        conn.commit()
    return log_id


def _seed_downstream_tcg_edge(conn, src_entity: str, tgt_entity: str,
                              initial_confidence: float = 0.8) -> str:
    tag = uuid.uuid4().hex[:8]
    # tcg_nodes and tcg_edges keyed on string IDs; entity names serve as IDs in tests.
    with conn.cursor() as cur:
        for ent in (src_entity, tgt_entity):
            cur.execute("""
                INSERT INTO erik_core.tcg_nodes(id, entity_type, name)
                VALUES (%s, 'gene', %s) ON CONFLICT (id) DO NOTHING
            """, (ent, ent))
        edge_id = f"test:tcge:{tag}"
        cur.execute("""
            INSERT INTO erik_core.tcg_edges
                (id, source_id, target_id, edge_type, confidence)
            VALUES (%s, %s, %s, 'causal', %s)
        """, (edge_id, src_entity, tgt_entity, initial_confidence))
        conn.commit()
    return edge_id


@pg
def test_non_intervention_edge_produces_empty_proposal(conn):
    """write_log event on a non-intervention edge = empty proposal (no row inserted)."""
    edge_id, _, _ = _seed_scm_edge(conn, is_candidate=False)
    _emit_write_log(conn, 'edge_superseded', edge_id)

    daemon = CompoundRefutationDaemon()
    stats = daemon.run_once()
    assert stats['events_processed'] >= 1
    assert stats['applied'] == 0
    # No propagation_events row was inserted for this edge.
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM erik_ops.propagation_events WHERE refuted_scm_edge_id = %s", (edge_id,))
        assert cur.fetchone()[0] == 0


@pg
def test_cursor_advances_monotonically(conn):
    """After a cycle, propagation_cursor.last_write_log_id ≥ highest processed id."""
    # Seed two edges and write-log rows.
    e1, _, _ = _seed_scm_edge(conn, is_candidate=False)
    e2, _, _ = _seed_scm_edge(conn, is_candidate=False)
    l1 = _emit_write_log(conn, 'edge_superseded', e1)
    l2 = _emit_write_log(conn, 'edge_superseded', e2)

    daemon = CompoundRefutationDaemon()
    stats = daemon.run_once()
    assert stats['cursor_after'] >= max(l1, l2)

    # Second call sees zero new events.
    stats2 = daemon.run_once()
    assert stats2['events_processed'] == 0
    assert stats2['cursor_after'] >= stats['cursor_after']


@pg
def test_idempotent_replay_does_not_double_apply(conn):
    """Running the daemon twice in quick succession on the same event must not double-mutate."""
    edge_id, src_ent, tgt_ent = _seed_scm_edge(conn, is_candidate=True)
    tcg_edge = _seed_downstream_tcg_edge(conn, src_ent, tgt_ent, initial_confidence=0.8)
    _emit_write_log(conn, 'edge_superseded', edge_id, superseded_by=999)

    daemon = CompoundRefutationDaemon()
    s1 = daemon.run_once()
    s2 = daemon.run_once()
    # First cycle processes, second sees nothing new.
    assert s1['applied'] >= 1
    assert s2['events_processed'] == 0

    # tcg_edge confidence was clamped to 0.2 on first apply and is unchanged after second.
    with conn.cursor() as cur:
        cur.execute("SELECT confidence FROM erik_core.tcg_edges WHERE id = %s", (tcg_edge,))
        conf_after = float(cur.fetchone()[0])
    assert conf_after <= 0.2


@pg
def test_intervention_candidate_refutation_mutates_downstream_tcg_edge(conn):
    edge_id, src_ent, tgt_ent = _seed_scm_edge(conn, is_candidate=True)
    tcg_edge = _seed_downstream_tcg_edge(conn, src_ent, tgt_ent, initial_confidence=0.95)
    _emit_write_log(conn, 'edge_superseded', edge_id, superseded_by=42)

    daemon = CompoundRefutationDaemon()
    stats = daemon.run_once()
    assert stats['applied'] >= 1

    with conn.cursor() as cur:
        cur.execute("SELECT confidence FROM erik_core.tcg_edges WHERE id = %s", (tcg_edge,))
        row = cur.fetchone()
    assert float(row[0]) <= 0.2

    # An applied propagation_events row must exist.
    with conn.cursor() as cur:
        cur.execute("""
            SELECT status, rule_kind FROM erik_ops.propagation_events
             WHERE refuted_scm_edge_id = %s
             ORDER BY id DESC LIMIT 1
        """, (edge_id,))
        row = cur.fetchone()
    assert row is not None
    assert row[0] == 'applied'
    assert row[1] == 'R4_compound_mechanism_refute'
