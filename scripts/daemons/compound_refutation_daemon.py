"""CompoundRefutationDaemon — consumes scm_write_log for R4 propagations.

Sister to Galen's ErrorPropagationDaemon (M.5, 2026-04-18). Runs R4 only
at Week 1: when ``erik_ops.scm_write_log`` records an
``edge_superseded`` or ``edge_invalidated`` event on an
``is_intervention_candidate`` edge, the daemon:

  1. Builds a RefutationContext from the write-log row + the refuted edge.
  2. Calls ``find_downstream_citations`` to discover TCG objects that
     cite the refuted biology.
  3. Calls ``r4_propose`` (pure function) to produce a proposal.
  4. Inserts the proposal row with status='proposed'.
  5. Calls ``apply_r4`` on the new proposal, atomically transitioning to
     'applied' or 'rejected' (stale-refuted-edge-reactivation).

Cursor discipline:
  * ``erik_ops.propagation_cursor`` tracks the highest ``scm_write_log.id``
    consumed per ``rule_kind``. The daemon restarts from that point on
    every cycle. Missed events across restarts are impossible under this
    scheme so long as the cursor advance and the event INSERT are in the
    same transaction — which they are, below.

Flag-gated by ``r4_propagation_enabled``. Default OFF.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from world_model.propagation_rules import (
    R4_MAX_BLAST_RADIUS,
    RefutationContext,
    apply_r4,
    find_downstream_citations,
    r4_propose,
)

logger = logging.getLogger(__name__)


_RULE_KIND = 'R4_compound_mechanism_refute'


class CompoundRefutationDaemon:
    """Pulls scm_write_log events, proposes + applies R4 per event.

    Typical lifecycle:

        d = CompoundRefutationDaemon()
        t = threading.Thread(target=d.run, daemon=True); t.start()
        # ...
        d.stop(); t.join(timeout=10)
    """

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self._interval_s = int(cfg.get("r4_propagation_interval_s", 300))   # 5 min
        self._max_events_per_cycle = int(cfg.get("r4_max_events_per_cycle", 50))
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        print("[R4] CompoundRefutationDaemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("r4_propagation_enabled", False):
                    self._stop.wait(60)
                    continue
                stats = self.run_once()
                if stats.get("events_processed", 0) > 0:
                    print(
                        f"[R4] cycle: processed {stats['events_processed']} events, "
                        f"applied {stats['applied']}, rejected {stats['rejected']}, "
                        f"empty {stats['empty_proposals']}"
                    )
            except Exception as e:
                logger.exception("R4 cycle failed")
                print(f"[R4] Error: {e}")
            self._stop.wait(self._interval_s)

    # -- public single-cycle API (used by daemon + tests) ----------------------

    def run_once(self) -> dict[str, Any]:
        """Process up to ``max_events_per_cycle`` new write-log events.

        Advances the cursor atomically with each event processed. Returns a
        stats dict.
        """
        cursor = self._load_cursor()
        events = self._fetch_new_events(after_id=cursor, limit=self._max_events_per_cycle)
        if not events:
            return {
                "events_processed": 0,
                "applied": 0,
                "rejected": 0,
                "empty_proposals": 0,
                "cursor_after": cursor,
            }

        applied = 0
        rejected = 0
        empty = 0
        new_cursor = cursor
        for ev in events:
            new_cursor = max(new_cursor, ev['id'])
            outcome = self._handle_one_event(ev)
            if outcome == 'applied':
                applied += 1
            elif outcome == 'rejected':
                rejected += 1
            else:
                empty += 1

        self._save_cursor(new_cursor)

        return {
            "events_processed": len(events),
            "applied": applied,
            "rejected": rejected,
            "empty_proposals": empty,
            "cursor_after": new_cursor,
        }

    # -- event handler ---------------------------------------------------------

    def _handle_one_event(self, ev: dict[str, Any]) -> str:
        """Process one scm_write_log event. Returns 'applied'|'rejected'|'empty'."""
        operation = ev['operation']
        refuted_edge_id = ev['target_id']
        if operation not in ('edge_superseded', 'edge_invalidated'):
            return 'empty'
        if refuted_edge_id is None:
            return 'empty'

        ctx = self._build_refutation_context(ev)
        if ctx is None or not ctx.was_intervention_candidate:
            return 'empty'

        with get_connection() as conn:
            downstream = find_downstream_citations(
                conn,
                refuted_scm_edge_id=ctx.refuted_scm_edge_id,
                refuted_source_entity=ctx.refuted_source_entity_id,
                refuted_target_entity=ctx.refuted_target_entity_id,
                limit=R4_MAX_BLAST_RADIUS * 2,
            )
            proposal = r4_propose(ctx, downstream)
            if proposal is None:
                return 'empty'

            # Insert proposal, then apply in the same connection.
            proposal_id = self._insert_proposal(conn, proposal)
            if proposal_id is None:
                return 'empty'
            result = apply_r4(conn, proposal_id)
            return 'applied' if result.get('applied') else 'rejected'

    def _build_refutation_context(self, ev: dict[str, Any]) -> Optional[RefutationContext]:
        """Read refuted edge details; attach was_intervention_candidate."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id, e.is_intervention_candidate,
                           sn_src.entity_id, sn_tgt.entity_id
                      FROM erik_ops.scm_edges e
                      JOIN erik_ops.scm_nodes sn_src ON sn_src.id = e.source_node_id
                      JOIN erik_ops.scm_nodes sn_tgt ON sn_tgt.id = e.target_node_id
                     WHERE e.id = %s
                """, (ev['target_id'],))
                row = cur.fetchone()
        if row is None:
            return None
        payload = ev.get('payload') or {}
        superseding_id = payload.get('superseded_by') if isinstance(payload, dict) else None
        return RefutationContext(
            write_log_id=ev['id'],
            refuted_scm_edge_id=int(row[0]),
            refuted_source_entity_id=row[2],
            refuted_target_entity_id=row[3],
            was_intervention_candidate=bool(row[1]),
            superseding_edge_id=int(superseding_id) if superseding_id else None,
            operation=ev['operation'],
        )

    # -- SQL helpers -----------------------------------------------------------

    def _load_cursor(self) -> int:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT last_write_log_id
                      FROM erik_ops.propagation_cursor
                     WHERE rule_kind = %s
                """, (_RULE_KIND,))
                row = cur.fetchone()
                if row is None:
                    cur.execute("""
                        INSERT INTO erik_ops.propagation_cursor(rule_kind, last_write_log_id)
                        VALUES (%s, 0)
                        ON CONFLICT (rule_kind) DO NOTHING
                    """, (_RULE_KIND,))
                    conn.commit()
                    return 0
                return int(row[0] or 0)

    def _save_cursor(self, new_cursor: int) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_ops.propagation_cursor(rule_kind, last_write_log_id, updated_at)
                    VALUES (%s, %s, NOW())
                    ON CONFLICT (rule_kind) DO UPDATE
                        SET last_write_log_id = GREATEST(propagation_cursor.last_write_log_id,
                                                         EXCLUDED.last_write_log_id),
                            updated_at = NOW()
                """, (_RULE_KIND, new_cursor))
                conn.commit()

    def _fetch_new_events(self, after_id: int, limit: int) -> list[dict[str, Any]]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, operation, target_id, payload, occurred_at
                      FROM erik_ops.scm_write_log
                     WHERE id > %s
                       AND operation IN ('edge_superseded','edge_invalidated')
                     ORDER BY id ASC
                     LIMIT %s
                """, (after_id, limit))
                rows = cur.fetchall()
        return [
            {
                'id': int(r[0]),
                'operation': r[1],
                'target_id': int(r[2]) if r[2] is not None else None,
                'payload': r[3] if isinstance(r[3], dict) else {},
                'occurred_at': r[4],
            }
            for r in rows
        ]

    def _insert_proposal(self, conn, proposal) -> Optional[int]:
        import json
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO erik_ops.propagation_events
                    (rule_kind, source_write_log_id, refuted_scm_edge_id,
                     affected_object_ids, affected_object_types,
                     status, applied_change, reason, daemon)
                VALUES (%s, %s, %s, %s, %s, 'proposed', %s, %s, 'compound_refutation')
                RETURNING id
            """, (
                proposal.rule_kind,
                proposal.source_write_log_id,
                proposal.refuted_scm_edge_id,
                proposal.affected_object_ids,
                proposal.affected_object_types,
                json.dumps(proposal.applied_change),
                proposal.reason,
            ))
            row = cur.fetchone()
            conn.commit()
            return int(row[0]) if row else None
