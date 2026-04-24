"""Tests for propagation_rules — R4 compound-mechanism-refute.

Split into:
  * Pure-function tests for ``r4_propose`` (no DB).
  * Integration tests for ``apply_r4`` / ``rollback_event`` /
    ``find_downstream_citations`` (require PG; skipped if unavailable).
"""
from __future__ import annotations

import os
import pytest

from world_model.propagation_rules import (
    DownstreamObject,
    PropagationProposal,
    R4_MAX_BLAST_RADIUS,
    RefutationContext,
    apply_r4,
    find_downstream_citations,
    r4_propose,
    rollback_event,
)


# ─── r4_propose — pure function tests ────────────────────────────────────────


def _ctx(was_candidate: bool = True, op: str = 'edge_superseded',
         superseding: int | None = 999) -> RefutationContext:
    return RefutationContext(
        write_log_id=100,
        refuted_scm_edge_id=42,
        refuted_source_entity_id="compound:riluzole",
        refuted_target_entity_id="mechanism:glutamate_excitotoxicity",
        was_intervention_candidate=was_candidate,
        superseding_edge_id=superseding,
        operation=op,
    )


def _ds(object_id: str, kind: str = 'tcg_hypothesis',
        row: dict | None = None) -> DownstreamObject:
    return DownstreamObject(
        object_id=object_id,
        object_kind=kind,
        cite_reason='test',
        current_row=row or {'id': object_id, 'status': 'proposed', 'confidence': 0.5},
    )


def test_r4_no_op_when_not_intervention_candidate():
    ctx = _ctx(was_candidate=False)
    assert r4_propose(ctx, [_ds('h:1')]) is None


def test_r4_no_op_when_no_downstream():
    assert r4_propose(_ctx(), []) is None


def test_r4_produces_proposal_for_minimum_case():
    p = r4_propose(_ctx(), [_ds('h:1')])
    assert p is not None
    assert p.rule_kind == 'R4_compound_mechanism_refute'
    assert p.affected_object_ids == ['h:1']
    assert p.affected_object_types == ['tcg_hypothesis']
    assert p.source_write_log_id == 100
    assert p.refuted_scm_edge_id == 42
    assert 'h:1' in p.applied_change['before']
    assert p.truncated_at is None


def test_r4_blast_radius_is_capped():
    """Blast radius is bounded at R4_MAX_BLAST_RADIUS; truncated_at records the full size."""
    big = [_ds(f'h:{i}') for i in range(R4_MAX_BLAST_RADIUS + 5)]
    p = r4_propose(_ctx(), big)
    assert p is not None
    assert len(p.affected_object_ids) == R4_MAX_BLAST_RADIUS
    assert p.truncated_at == R4_MAX_BLAST_RADIUS + 5


def test_r4_proposal_is_deterministic_under_iteration_order():
    """r4_propose sorts its input so identical sets produce identical proposals."""
    items = [_ds('h:3'), _ds('h:1', 'tcg_edge'), _ds('h:2')]
    reversed_items = list(reversed(items))
    p1 = r4_propose(_ctx(), items)
    p2 = r4_propose(_ctx(), reversed_items)
    assert p1.affected_object_ids == p2.affected_object_ids
    assert p1.affected_object_types == p2.affected_object_types


def test_r4_reason_mentions_superseding_edge_when_present():
    p = r4_propose(_ctx(superseding=999), [_ds('h:1')])
    assert "superseded by 999" in p.reason


def test_r4_reason_handles_invalidation_op():
    ctx = _ctx(op='edge_invalidated', superseding=None)
    p = r4_propose(ctx, [_ds('h:1')])
    assert "edge_invalidated" in p.reason
    assert "superseded by" not in p.reason


def test_r4_captures_before_snapshots():
    row = {'id': 'h:7', 'status': 'supported', 'confidence': 0.85,
           'supporting_path': ['e:1', 'e:2']}
    p = r4_propose(_ctx(), [_ds('h:7', row=row)])
    assert p.applied_change['before']['h:7']['row'] == row


# ─── Integration tests (PG-gated) ────────────────────────────────────────────


def _can_connect_pg() -> bool:
    import psycopg
    user = os.environ.get("USER", "logannye")
    try:
        c = psycopg.connect(f"dbname=erik_kg user={user}", connect_timeout=3)
        c.close()
        return True
    except Exception:
        return False


pg = pytest.mark.skipif(not _can_connect_pg(), reason="erik_kg PG not reachable")


@pytest.fixture
def conn():
    import psycopg
    user = os.environ.get("USER", "logannye")
    c = psycopg.connect(f"dbname=erik_kg user={user}")
    yield c
    c.close()


def _seed_refuted_edge(conn) -> tuple[int, str, str]:
    """Insert an scm_node pair + a superseded scm_edge + a write-log row.

    Returns (scm_edge_id, src_entity_id, tgt_entity_id).
    """
    import uuid
    src_entity = f"test:compound:{uuid.uuid4().hex[:8]}"
    tgt_entity = f"test:mechanism:{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
            VALUES (%s, 'treatment', 'compound') RETURNING id
        """, (src_entity,))
        src_node = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_nodes(entity_id, node_class, als_role)
            VALUES (%s, 'covariate', 'mechanism') RETURNING id
        """, (tgt_entity,))
        tgt_node = cur.fetchone()[0]
        cur.execute("""
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 identification_algorithm, identification_confidence,
                 is_intervention_candidate, status)
            VALUES (%s, %s, 'causal', 'rct', 0.9, TRUE, 'superseded')
            RETURNING id
        """, (src_node, tgt_node))
        edge_id = cur.fetchone()[0]
        conn.commit()
    return int(edge_id), src_entity, tgt_entity


@pg
def test_apply_r4_transitions_to_applied_and_mutates_hypothesis(conn):
    import uuid, json
    edge_id, src_ent, tgt_ent = _seed_refuted_edge(conn)
    hyp_id = f"test:hyp:{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_core.tcg_hypotheses(id, hypothesis, status, confidence)
            VALUES (%s, 'test', 'supported', 0.85)
        """, (hyp_id,))
        # Insert a proposal row directly.
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, source_write_log_id, refuted_scm_edge_id,
                 affected_object_ids, affected_object_types,
                 status, applied_change, reason, daemon)
            VALUES ('R4_compound_mechanism_refute', NULL, %s,
                    %s, %s, 'proposed', %s, 'test', 'pytest')
            RETURNING id
        """, (edge_id, [hyp_id], ['tcg_hypothesis'],
              json.dumps({'before': {hyp_id: {'kind': 'tcg_hypothesis',
                                              'row': {'status': 'supported', 'confidence': 0.85}}}})))
        event_id = cur.fetchone()[0]
        conn.commit()

    result = apply_r4(conn, event_id)
    assert result['applied'] is True
    assert result['affected'] == 1

    # Assert mutations.
    with conn.cursor() as cur:
        cur.execute("SELECT status, confidence FROM erik_core.tcg_hypotheses WHERE id = %s",
                    (hyp_id,))
        row = cur.fetchone()
        assert row[0] == 'refuted_by_propagation'
        assert float(row[1]) <= 0.2
        cur.execute("SELECT status FROM erik_ops.propagation_events WHERE id = %s",
                    (event_id,))
        assert cur.fetchone()[0] == 'applied'


@pg
def test_apply_r4_rejects_stale_when_edge_reactivated(conn):
    """If the refuted edge has been re-promoted to 'active' between propose and
    apply, apply_r4 must reject rather than mutate downstream objects."""
    import uuid, json
    edge_id, _, _ = _seed_refuted_edge(conn)
    # Re-activate the edge (simulates it being re-promoted).
    with conn.cursor() as cur:
        cur.execute("UPDATE erik_ops.scm_edges SET status='active' WHERE id=%s", (edge_id,))
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, source_write_log_id, refuted_scm_edge_id,
                 affected_object_ids, affected_object_types,
                 status, applied_change, reason, daemon)
            VALUES ('R4_compound_mechanism_refute', NULL, %s,
                    %s, %s, 'proposed', %s, 'test', 'pytest')
            RETURNING id
        """, (edge_id, ['h:x'], ['tcg_hypothesis'], json.dumps({'before': {}})))
        event_id = cur.fetchone()[0]
        conn.commit()

    result = apply_r4(conn, event_id)
    assert result['applied'] is False
    assert result['reason'] == 'refuted_edge_reactivated'
    with conn.cursor() as cur:
        cur.execute("SELECT status FROM erik_ops.propagation_events WHERE id = %s",
                    (event_id,))
        assert cur.fetchone()[0] == 'rejected'


@pg
def test_rollback_restores_before_state(conn):
    import uuid, json
    edge_id, _, _ = _seed_refuted_edge(conn)
    hyp_id = f"test:hyp:{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_core.tcg_hypotheses(id, hypothesis, status, confidence)
            VALUES (%s, 'test', 'supported', 0.75)
        """, (hyp_id,))
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, source_write_log_id, refuted_scm_edge_id,
                 affected_object_ids, affected_object_types,
                 status, applied_change, reason, daemon)
            VALUES ('R4_compound_mechanism_refute', NULL, %s,
                    %s, %s, 'proposed', %s, 'test', 'pytest')
            RETURNING id
        """, (edge_id, [hyp_id], ['tcg_hypothesis'],
              json.dumps({'before': {hyp_id: {'kind': 'tcg_hypothesis',
                                              'row': {'status': 'supported', 'confidence': 0.75}}}})))
        event_id = cur.fetchone()[0]
        conn.commit()

    apply_r4(conn, event_id)
    # State after apply: refuted.
    with conn.cursor() as cur:
        cur.execute("SELECT status, confidence FROM erik_core.tcg_hypotheses WHERE id = %s",
                    (hyp_id,))
        r = cur.fetchone()
        assert r[0] == 'refuted_by_propagation'

    result = rollback_event(conn, event_id)
    assert result['rolled_back'] is True
    with conn.cursor() as cur:
        cur.execute("SELECT status, confidence FROM erik_core.tcg_hypotheses WHERE id = %s",
                    (hyp_id,))
        r = cur.fetchone()
        assert r[0] == 'supported'
        assert abs(float(r[1]) - 0.75) < 1e-6
        # Original event is now rolled_back.
        cur.execute("SELECT status, rollback_lineage_id FROM erik_ops.propagation_events WHERE id = %s",
                    (event_id,))
        orig = cur.fetchone()
        assert orig[0] == 'rolled_back'
        assert orig[1] is not None


@pg
def test_rollback_idempotent(conn):
    """Rolling back an already-rolled-back event is a no-op."""
    import uuid, json
    edge_id, _, _ = _seed_refuted_edge(conn)
    hyp_id = f"test:hyp:{uuid.uuid4().hex[:8]}"
    with conn.cursor() as cur:
        cur.execute("INSERT INTO erik_core.tcg_hypotheses(id, hypothesis, status, confidence) VALUES (%s, 't', 'supported', 0.75)", (hyp_id,))
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, source_write_log_id, refuted_scm_edge_id,
                 affected_object_ids, affected_object_types,
                 status, applied_change, reason, daemon)
            VALUES ('R4_compound_mechanism_refute', NULL, %s, %s, %s, 'proposed', %s, 'test', 'pytest')
            RETURNING id
        """, (edge_id, [hyp_id], ['tcg_hypothesis'],
              json.dumps({'before': {hyp_id: {'kind': 'tcg_hypothesis',
                                              'row': {'status': 'supported', 'confidence': 0.75}}}})))
        event_id = cur.fetchone()[0]
        conn.commit()
    apply_r4(conn, event_id)
    first = rollback_event(conn, event_id)
    second = rollback_event(conn, event_id)
    assert first['rolled_back'] is True
    assert second['rolled_back'] is False
    assert second['reason'] == 'already_rolled_back'


@pg
def test_append_only_trigger_forbids_delete(conn):
    """The schema trigger must block DELETE of any propagation_events row."""
    import json, psycopg
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, affected_object_ids, affected_object_types,
                 status, applied_change, reason, daemon)
            VALUES ('R4_compound_mechanism_refute', '{}', '{}', 'proposed', %s, 't', 'pytest')
            RETURNING id
        """, (json.dumps({'before': {}}),))
        ev_id = cur.fetchone()[0]
        conn.commit()

    with pytest.raises(psycopg.errors.RaiseException):
        with conn.cursor() as cur:
            cur.execute("DELETE FROM erik_ops.propagation_events WHERE id = %s", (ev_id,))
    conn.rollback()


@pg
def test_append_only_trigger_forbids_bad_transitions(conn):
    """proposed → rolled_back is forbidden (only applied → rolled_back is legal)."""
    import json, psycopg
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, affected_object_ids, affected_object_types,
                 status, applied_change, reason, daemon)
            VALUES ('R4_compound_mechanism_refute', '{}', '{}', 'proposed', %s, 't', 'pytest')
            RETURNING id
        """, (json.dumps({'before': {}}),))
        ev_id = cur.fetchone()[0]
        conn.commit()

    with pytest.raises(psycopg.errors.RaiseException):
        with conn.cursor() as cur:
            cur.execute("UPDATE erik_ops.propagation_events SET status='rolled_back' WHERE id = %s", (ev_id,))
    conn.rollback()
