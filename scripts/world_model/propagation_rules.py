"""Propagation rules — deterministic refutation-propagation logic.

Ported from Galen Track B M.5 (2026-04-18). Erik's Week 1 implements a
single rule, ``R4_compound_mechanism_refute``, which handles the most
dangerous silent-failure pattern for a compound-discovery engine:

  When an intervention-candidate ``scm_edge`` is superseded or invalidated
  (i.e. its identification_confidence has been beaten by stronger evidence,
  or its ground has been refuted by a higher-tier algorithm), every
  downstream compound dossier, TCG hypothesis, or TCG edge that cited
  that scm_edge must be flagged as contested — otherwise Erik's protocol
  could keep recommending a compound resting on biology the system has
  already refuted.

Design invariants:

  * **Pure-function rule body.** ``R4.propose`` takes read-only inputs and
    returns a ``PropagationProposal``. Zero DB writes; zero side effects.
    The applier below owns all writes.

  * **Bounded blast radius.** Each proposal caps ``affected_object_ids``
    at ``R4_MAX_BLAST_RADIUS`` so a single refutation at a highly-shared
    node cannot cascade into a system-wide lockup.

  * **Before-snapshot capture.** The applier reads the current row state
    for every affected object and embeds it in ``applied_change.before``
    as the anchor for rollback. Rollback restores these snapshots in a
    single transaction.

  * **Stale-check at apply time.** Between propose() and apply(), the SCM
    may have moved again. Apply re-reads the refuted edge's status and
    refuses to mutate anything if the edge has already been re-promoted.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)


# Hard cap on objects deprecated per single propagation event. Galen's
# equivalent is 20 for cancer compounds; Erik is a one-patient system so
# 10 is already generous.
R4_MAX_BLAST_RADIUS: int = 10


# ─── Types ────────────────────────────────────────────────────────────────────


@dataclass
class RefutationContext:
    """Read-only inputs to R4 derived from an scm_write_log row.

    The propagation daemon assembles one of these per ``edge_superseded``
    or ``edge_invalidated`` event whose payload indicates the edge was
    an intervention candidate.
    """
    write_log_id: int
    refuted_scm_edge_id: int
    refuted_source_entity_id: str
    refuted_target_entity_id: str
    was_intervention_candidate: bool
    superseding_edge_id: Optional[int]
    operation: str                       # 'edge_superseded' | 'edge_invalidated'


@dataclass
class DownstreamObject:
    """A concrete reference to a TCG-layer object that cites the refuted edge.

    ``object_kind`` ∈ {'tcg_hypothesis', 'tcg_edge', 'intervention'}.
    ``cite_reason`` is a short human-readable string (for the event log).
    """
    object_id: str
    object_kind: str
    cite_reason: str
    current_row: dict[str, Any]      # pre-change full row for rollback


@dataclass
class PropagationProposal:
    """The output of a single rule's propose() step.

    Proposals are persisted as rows with status='proposed'. The applier
    only mutates the affected objects when it transitions the row to
    status='applied'.
    """
    rule_kind: str
    source_write_log_id: int
    refuted_scm_edge_id: int
    affected_object_ids: list[str]
    affected_object_types: list[str]
    reason: str
    applied_change: dict[str, Any] = field(default_factory=dict)
    # Blast radius was truncated? (Recorded for telemetry / audit.)
    truncated_at: Optional[int] = None


# ─── R4: compound-mechanism-refute ────────────────────────────────────────────


def r4_propose(
    ctx: RefutationContext,
    downstream: Sequence[DownstreamObject],
) -> Optional[PropagationProposal]:
    """Propose deprecation of every TCG object citing the refuted edge.

    Returns None when there's nothing to do:
      * The source edge was never an intervention candidate — R4 skips,
        R1/R2/R3 may still apply (but Erik's Week 1 ports only R4).
      * No downstream objects cite the refuted edge.

    Otherwise returns a PropagationProposal with ``affected_object_ids``
    truncated to at most ``R4_MAX_BLAST_RADIUS``.
    """
    if not ctx.was_intervention_candidate:
        return None
    if not downstream:
        return None

    # Truncate blast radius deterministically. We sort by (object_kind, object_id)
    # so the same (ctx, downstream-set) always produces the same proposal
    # even when the caller's iteration order isn't stable.
    sorted_ds = sorted(downstream, key=lambda d: (d.object_kind, d.object_id))
    truncated_at: Optional[int] = None
    if len(sorted_ds) > R4_MAX_BLAST_RADIUS:
        truncated_at = len(sorted_ds)
        sorted_ds = sorted_ds[:R4_MAX_BLAST_RADIUS]

    before_map: dict[str, Any] = {
        d.object_id: {
            'kind': d.object_kind,
            'cite_reason': d.cite_reason,
            'row': d.current_row,
        } for d in sorted_ds
    }

    reason_suffix = f" (truncated from {truncated_at})" if truncated_at else ""
    return PropagationProposal(
        rule_kind='R4_compound_mechanism_refute',
        source_write_log_id=ctx.write_log_id,
        refuted_scm_edge_id=ctx.refuted_scm_edge_id,
        affected_object_ids=[d.object_id for d in sorted_ds],
        affected_object_types=[d.object_kind for d in sorted_ds],
        reason=(
            f"scm_edge {ctx.refuted_scm_edge_id} "
            f"({ctx.refuted_source_entity_id} → {ctx.refuted_target_entity_id}) "
            f"was {ctx.operation}"
            + (f", superseded by {ctx.superseding_edge_id}"
               if ctx.superseding_edge_id else "")
            + f"; deprecating {len(sorted_ds)} downstream citations"
            + reason_suffix
        ),
        applied_change={'before': before_map},
        truncated_at=truncated_at,
    )


# ─── Appliers ─────────────────────────────────────────────────────────────────
#
# The applier owns all DB writes. It is invoked once per proposal and must
# be idempotent under replay (the propagation daemon's cursor table means
# replay should never happen in practice, but defensive idempotence is
# cheap and removes one class of weekend-at-4am bug).


def apply_r4(conn, proposal_row_id: int) -> dict[str, Any]:
    """Apply an R4 proposal transactionally.

    Preconditions:
      * A row exists in erik_ops.propagation_events with id=proposal_row_id
        and status='proposed' and rule_kind='R4_compound_mechanism_refute'.

    Postconditions on success:
      * For each affected object, a deprecation mutation has been written.
      * The propagation_events row has status='applied' and applied_at set.

    Stale-check: if the refuted edge has since become 'active' again
    (re-promoted by a new identification), we instead transition the
    proposal to 'rejected' and do nothing — preserving the audit trail
    without firing a now-stale deprecation.

    Returns a stats dict: {applied: bool, affected: int, reason: str}.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT rule_kind, status, refuted_scm_edge_id, affected_object_ids,
                   affected_object_types, applied_change
              FROM erik_ops.propagation_events
             WHERE id = %s
             FOR UPDATE
        """, (proposal_row_id,))
        row = cur.fetchone()
        if row is None:
            return {'applied': False, 'affected': 0, 'reason': 'not_found'}
        rule_kind, status, refuted_edge_id, object_ids, object_types, applied_change = row

        if rule_kind != 'R4_compound_mechanism_refute':
            return {'applied': False, 'affected': 0, 'reason': 'wrong_rule_kind'}
        if status != 'proposed':
            return {'applied': False, 'affected': 0, 'reason': f'already_{status}'}

        # Stale check: if the refuted edge is active again, reject.
        cur.execute(
            "SELECT status FROM erik_ops.scm_edges WHERE id = %s",
            (refuted_edge_id,),
        )
        edge_row = cur.fetchone()
        edge_is_stale = bool(edge_row and edge_row[0] == 'active')
        if edge_is_stale:
            cur.execute("""
                UPDATE erik_ops.propagation_events
                   SET status = 'rejected'
                 WHERE id = %s
            """, (proposal_row_id,))
            conn.commit()
            return {'applied': False, 'affected': 0, 'reason': 'refuted_edge_reactivated'}

        # Apply mutations per object kind. All in one transaction so a
        # failure of any mutation rolls back the entire R4 event.
        affected = 0
        applied_change = applied_change or {'before': {}}
        before_map: dict[str, Any] = applied_change.get('before', {})

        for object_id, kind in zip(object_ids or [], object_types or []):
            if kind == 'tcg_hypothesis':
                cur.execute("""
                    UPDATE erik_core.tcg_hypotheses
                       SET status = 'refuted_by_propagation',
                           confidence = LEAST(confidence, 0.2),
                           updated_at = NOW()
                     WHERE id = %s
                """, (object_id,))
                affected += cur.rowcount
            elif kind == 'tcg_edge':
                cur.execute("""
                    UPDATE erik_core.tcg_edges
                       SET confidence = LEAST(confidence, 0.2),
                           updated_at = NOW()
                     WHERE id = %s
                """, (object_id,))
                affected += cur.rowcount
            elif kind == 'intervention':
                cur.execute("""
                    UPDATE erik_core.objects
                       SET status = 'contested_by_propagation',
                           updated_at = NOW()
                     WHERE id = %s
                       AND type = 'Intervention'
                """, (object_id,))
                affected += cur.rowcount
            else:
                logger.warning("apply_r4: unknown object kind %s (id=%s)", kind, object_id)

        cur.execute("""
            UPDATE erik_ops.propagation_events
               SET status = 'applied',
                   applied_at = NOW()
             WHERE id = %s
        """, (proposal_row_id,))
        conn.commit()

    return {'applied': True, 'affected': affected, 'reason': 'ok',
            'before_size': len(before_map)}


def rollback_event(conn, event_id: int) -> dict[str, Any]:
    """Restore before-state for a previously applied propagation event.

    Writes a new 'rolled_back' event linked via rollback_lineage_id to the
    original, then transitions the original to 'rolled_back'. Idempotent:
    rolling back an already-rolled-back event is a no-op.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT status, rule_kind, affected_object_ids, affected_object_types,
                   applied_change, refuted_scm_edge_id, source_write_log_id
              FROM erik_ops.propagation_events
             WHERE id = %s
             FOR UPDATE
        """, (event_id,))
        row = cur.fetchone()
        if row is None:
            return {'rolled_back': False, 'affected': 0, 'reason': 'not_found'}
        status, rule_kind, object_ids, object_types, applied_change, refuted_id, src_log_id = row
        if status == 'rolled_back':
            return {'rolled_back': False, 'affected': 0, 'reason': 'already_rolled_back'}
        if status != 'applied':
            return {'rolled_back': False, 'affected': 0, 'reason': f'cannot_rollback_{status}'}
        before_map = (applied_change or {}).get('before', {}) or {}

        affected = 0
        for object_id, kind in zip(object_ids or [], object_types or []):
            snap = before_map.get(object_id)
            if snap is None:
                continue
            row_before = snap.get('row', {})
            if kind == 'tcg_hypothesis':
                prior_status = row_before.get('status', 'proposed')
                prior_conf = row_before.get('confidence', 0.1)
                cur.execute("""
                    UPDATE erik_core.tcg_hypotheses
                       SET status = %s, confidence = %s, updated_at = NOW()
                     WHERE id = %s
                """, (prior_status, prior_conf, object_id))
                affected += cur.rowcount
            elif kind == 'tcg_edge':
                prior_conf = row_before.get('confidence', 0.1)
                cur.execute("""
                    UPDATE erik_core.tcg_edges
                       SET confidence = %s, updated_at = NOW()
                     WHERE id = %s
                """, (prior_conf, object_id))
                affected += cur.rowcount
            elif kind == 'intervention':
                prior_status = row_before.get('status', 'active')
                cur.execute("""
                    UPDATE erik_core.objects
                       SET status = %s, updated_at = NOW()
                     WHERE id = %s AND type = 'Intervention'
                """, (prior_status, object_id))
                affected += cur.rowcount

        # Write the rollback lineage event.
        cur.execute("""
            INSERT INTO erik_ops.propagation_events
                (rule_kind, source_write_log_id, refuted_scm_edge_id,
                 affected_object_ids, affected_object_types,
                 status, applied_change, reason,
                 rollback_lineage_id, daemon)
            VALUES (%s, %s, %s, %s, %s,
                    'applied', %s, %s, %s, 'rollback_cli')
            RETURNING id
        """, (
            rule_kind, src_log_id, refuted_id,
            object_ids, object_types,
            json.dumps({'rollback_of': event_id, 'before': before_map}),
            f'rollback of propagation_events id={event_id}',
            event_id,
        ))
        rollback_id = cur.fetchone()[0]
        # Transition the original to rolled_back.
        cur.execute("""
            UPDATE erik_ops.propagation_events
               SET status = 'rolled_back',
                   rolled_back_at = NOW(),
                   rollback_lineage_id = %s
             WHERE id = %s
        """, (rollback_id, event_id))
        conn.commit()

    return {'rolled_back': True, 'affected': affected, 'rollback_event_id': rollback_id, 'reason': 'ok'}


# ─── Downstream-citation queries ──────────────────────────────────────────────
#
# The propagation daemon uses these to build the ``downstream`` list before
# calling r4_propose. They live here (not in the daemon) so tests can
# assert the citation semantics without spinning up a full daemon.


def find_downstream_citations(
    conn,
    refuted_scm_edge_id: int,
    refuted_source_entity: str,
    refuted_target_entity: str,
    limit: int = R4_MAX_BLAST_RADIUS * 2,   # over-fetch so truncation is visible
) -> list[DownstreamObject]:
    """Find TCG-layer objects that cite the refuted scm_edge.

    Week 1 matching strategy (conservative):
      1. ``tcg_hypotheses`` whose supporting_path array contains any
         ``tcg_edges.id`` that references the same source/target nodes.
      2. ``tcg_edges`` that match the same source_id → target_id entity
         pair as the refuted scm_edge.
      3. ``erik_core.objects`` type='Intervention' whose body.targets array
         contains the refuted target entity name.

    We deliberately over-match (prefer false-positive flags over
    false-negative "silent refuted biology") and let the blast-radius cap
    truncate. Future work: switch to strict citation via scm_edge_id once
    downstream objects start storing that foreign key.
    """
    out: list[DownstreamObject] = []
    with conn.cursor() as cur:
        # (2) TCG edges at the same entity pair.
        cur.execute("""
            SELECT id, confidence, evidence_ids, source_id, target_id
              FROM erik_core.tcg_edges
             WHERE source_id = %s AND target_id = %s
             LIMIT %s
        """, (refuted_source_entity, refuted_target_entity, limit))
        matched_tcg_edge_ids: list[str] = []
        for r in cur.fetchall():
            matched_tcg_edge_ids.append(r[0])
            out.append(DownstreamObject(
                object_id=r[0],
                object_kind='tcg_edge',
                cite_reason=f'same src/tgt as scm_edge {refuted_scm_edge_id}',
                current_row={
                    'id': r[0], 'confidence': float(r[1] or 0.0),
                    'evidence_ids': list(r[2] or []),
                    'source_id': r[3], 'target_id': r[4],
                },
            ))

        # (1) Hypotheses whose supporting_path cites any matched tcg_edge.
        if matched_tcg_edge_ids:
            cur.execute("""
                SELECT id, status, confidence, supporting_path
                  FROM erik_core.tcg_hypotheses
                 WHERE supporting_path && %s
                 LIMIT %s
            """, (matched_tcg_edge_ids, limit))
            for r in cur.fetchall():
                out.append(DownstreamObject(
                    object_id=r[0],
                    object_kind='tcg_hypothesis',
                    cite_reason=f'supporting_path includes refuted tcg_edge',
                    current_row={
                        'id': r[0], 'status': r[1],
                        'confidence': float(r[2] or 0.0),
                        'supporting_path': list(r[3] or []),
                    },
                ))

        # (3) Interventions targeting the refuted target entity.
        cur.execute("""
            SELECT id, status, body
              FROM erik_core.objects
             WHERE type = 'Intervention'
               AND status = 'active'
               AND (body->'targets')::text ILIKE %s
             LIMIT %s
        """, (f'%{refuted_target_entity}%', limit))
        for r in cur.fetchall():
            body = r[2] if isinstance(r[2], dict) else {}
            out.append(DownstreamObject(
                object_id=r[0],
                object_kind='intervention',
                cite_reason=f'targets include {refuted_target_entity}',
                current_row={
                    'id': r[0], 'status': r[1],
                    'name': (body or {}).get('name'),
                    'targets': (body or {}).get('targets', []),
                },
            ))

    return out
