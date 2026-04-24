"""SCMWriter — single-writer coordinator for erik_ops SCM tables.

Ported from Galen Phase 1 (2026-04-16). Adapted for Erik's compound-discovery
mission — carries three Erik-specific responsibilities that Galen's doesn't:

1.  Tags nodes with als_role ('gene'|'mechanism'|'pathway'|'compound'|...)
    and propagates druggability_prior from the KG entity.
2.  Flags scm_edges.is_intervention_candidate when source_node is druggable
    AND target_node is a mechanism/pathway the system is treating as an
    intervention point for Erik's protocol.
3.  Emits 'intervention_flagged' events to scm_write_log so the compound
    daemon can pick up newly-identified intervention points without polling
    the whole scm_edges table.

Why a single writer (inherited from Galen rationale):
    SCM identifications form audit chains via superseded_by. Concurrent writers
    can corrupt those chains. The scm_edge_creation reward signal polls
    scm_write_log; a duplicate edge_created event would double-reward and
    invite reward hacking.

    Single-writer enforcement uses a PG advisory lock keyed on a fixed bigint
    (ERIK_SCM_WRITER_LOCK_KEY below — distinct from Galen's key so both
    systems can run concurrently on different databases without contention).
    Released automatically when the dedicated connection closes.

Sharding: not needed at Erik's write rate (target <100/day until Phase 2).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import psycopg
except ImportError:
    psycopg = None  # type: ignore

logger = logging.getLogger(__name__)


# ─── Constants ────────────────────────────────────────────────────────────────

# Distinct from Galen's 7_020_426_100 so both systems can run side-by-side
# against shared infra without advisory-lock collision.
ERIK_SCM_WRITER_LOCK_KEY = 7_020_426_101

# Algorithm strength ranking for supersession. Higher = stronger identification.
# Ties broken by confidence. Adapted for ALS:
#   rct                     — gold standard
#   crispr_interventional   — direct genetic manipulation in cells
#   als_mouse_model         — interventional but cross-species
#   ipsc_motor_neuron_assay — patient-derived cells, high translational value
#   patient_organoid        — ditto; scored just below iPSC for consistency
#   scm_counterfactual      — do-calculus on our own SCM (self-consistent)
#   replicated_experiment   — two or more independent observational replications
#   experimental_assay      — single experimental
#   galen_scm               — cross-disease causal evidence inherited from Galen
#   cmap_signature_match    — pattern-matching, not mechanistic
#   regression_discontinuity / iv / frontdoor_adj — observational causal methods
#   pc_algorithm            — weakest; untestable assumptions
_ALGORITHM_STRENGTH: dict[str, int] = {
    'rct':                      10,
    'crispr_interventional':     9,
    'als_mouse_model':           8,
    'ipsc_motor_neuron_assay':   8,
    'patient_organoid':          7,
    'scm_counterfactual':        7,
    'replicated_experiment':     6,
    'experimental_assay':        5,
    'galen_scm':                 5,
    'cmap_signature_match':      4,
    'regression_discontinuity':  4,
    'iv':                        3,
    'frontdoor_adj':             3,
    'docking_simulation':        2,
    'pc_algorithm':              2,
}

_SUPERSEDE_CONFIDENCE_DELTA = 0.05
_QUEUE_POLL_TIMEOUT_S = 1.0
_DEFAULT_QUEUE_MAX = 10_000
_RETRY_BACKOFF_S: list[float] = [1.0, 2.0, 4.0]


# ─── Exceptions ───────────────────────────────────────────────────────────────

class SCMWriterError(Exception):
    """Base error for SCMWriter."""


class SCMWriterLockContention(SCMWriterError):
    """Another writer holds the advisory lock."""


class SCMWriterNotRunning(SCMWriterError):
    """submit_* called while writer is stopped."""


class QueueFullError(SCMWriterError):
    """In-process queue is at capacity; caller should back off."""


# ─── Request types ────────────────────────────────────────────────────────────

@dataclass
class EffectDistribution:
    """Bundle of effect estimates on a single scale."""
    mean: Optional[float] = None
    std: Optional[float] = None
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None
    scale: Optional[str] = None


@dataclass
class IdentificationRequest:
    """A request to identify (or re-identify) a causal edge.

    source_entity_id / target_entity_id reference erik_core.entities.id.
    """
    source_entity_id: str
    target_entity_id: str
    edge_kind: str                       # 'causal' | 'confounding' | 'mediating'
    algorithm: str
    confidence: float
    source_node_class: str = 'covariate'
    target_node_class: str = 'covariate'
    effect: Optional[EffectDistribution] = None
    evidence_refs: list[str] = field(default_factory=list)
    derived_from_rel_ids: list[str] = field(default_factory=list)
    adjustment_set_id: Optional[int] = None
    transport_population: Optional[str] = None
    transport_conditions: dict[str, Any] = field(default_factory=dict)
    source_als_role: Optional[str] = None
    target_als_role: Optional[str] = None
    source_druggability: Optional[float] = None
    target_druggability: Optional[float] = None
    is_intervention_candidate: bool = False
    trace: dict[str, Any] = field(default_factory=dict)
    daemon_source: str = 'unknown'


@dataclass
class EffectUpdateRequest:
    """Augment an existing scm_edge with quantitative effect data.

    Distinct from IdentificationRequest because this is **not** a
    re-identification: the edge's identification_algorithm and
    identification_confidence stay unchanged. Only effect_mean /
    effect_std / effect_ci_lower / effect_ci_upper / effect_scale are
    overwritten. This is the pathway for the QuantitativeEffectEnricher
    daemon to fill in numbers for edges that were promoted by bootstrap
    without quantitative data.

    Idempotent: submitting the same update twice produces the same row
    state (last-writer-wins). Preserves audit via scm_write_log
    'effect_updated' event.
    """
    scm_edge_id: int
    effect: "EffectDistribution"
    source: str = 'unknown'              # human-readable data source tag
    daemon_source: str = 'unknown'
    trace: dict[str, Any] = field(default_factory=dict)


@dataclass
class CFTraceRequest:
    """A stored counterfactual trace anchored to a specific edge."""
    edge_id: int
    query_id: str
    abduction_state: dict[str, Any]
    intervention_do: dict[str, Any]
    factual_outcome: dict[str, Any]
    counterfactual_outcome: dict[str, Any]
    regret: Optional[float] = None
    patient_snapshot_id: Optional[str] = None
    runtime_ms: Optional[int] = None
    daemon_source: str = 'unknown'


@dataclass
class _EnqueuedWork:
    kind: str  # 'identification' | 'cf_trace' | 'reverification'
    payload: Any
    tracking_id: str
    enqueued_at: float


# ─── SCMWriter ────────────────────────────────────────────────────────────────

class SCMWriter:
    """Coordinates all writes to erik_ops.scm_* tables from a single thread.

    Typical usage (in daemon code)::

        writer = get_scm_writer()
        writer.start()                           # idempotent
        tracking_id = writer.submit_identification(req)
        # ... later, monitor via writer.stats() ...
        writer.stop(timeout=10.0)
    """

    def __init__(
        self,
        conninfo_factory,
        queue_max: int = _DEFAULT_QUEUE_MAX,
        daemon_name: str = 'scm_writer',
    ) -> None:
        self._conninfo_factory = conninfo_factory
        self._queue: queue.Queue[_EnqueuedWork] = queue.Queue(maxsize=queue_max)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = threading.Event()
        self._lock_conn: Optional["psycopg.Connection"] = None
        self._daemon_name = daemon_name
        self._stats_lock = threading.Lock()
        self._stats: dict[str, Any] = {
            'identifications_processed': 0,
            'edges_created': 0,
            'edges_superseded': 0,
            'edges_rejected_weaker': 0,
            'cf_traces_processed': 0,
            'errors': 0,
            'started_at': None,
        }

    # -- lifecycle -------------------------------------------------------------

    def start(self) -> None:
        """Acquire the advisory lock and start the worker thread.

        Raises SCMWriterLockContention if another process already holds the
        advisory lock.
        """
        if self._running.is_set():
            return
        if psycopg is None:
            raise SCMWriterError("psycopg not installed")

        conn = psycopg.connect(self._conninfo_factory(), autocommit=True)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (ERIK_SCM_WRITER_LOCK_KEY,))
            row = cur.fetchone()
            acquired = bool(row and row[0])
        if not acquired:
            conn.close()
            raise SCMWriterLockContention(
                f"ERIK_SCM_WRITER_LOCK_KEY={ERIK_SCM_WRITER_LOCK_KEY} is held by another process"
            )
        self._lock_conn = conn

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name='erik-scm-writer', daemon=True,
        )
        with self._stats_lock:
            self._stats['started_at'] = time.time()
        self._thread.start()
        self._running.set()
        logger.info("SCMWriter started (lock=%s)", ERIK_SCM_WRITER_LOCK_KEY)

    def stop(self, timeout: float = 10.0) -> None:
        if not self._running.is_set():
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._lock_conn is not None:
            try:
                self._lock_conn.close()
            finally:
                self._lock_conn = None
        self._running.clear()
        logger.info("SCMWriter stopped")

    # -- submit API (non-blocking) ---------------------------------------------

    def submit_identification(self, req: IdentificationRequest) -> str:
        self._require_running()
        _validate_identification_request(req)
        tracking_id = str(uuid.uuid4())
        try:
            self._queue.put_nowait(_EnqueuedWork(
                kind='identification',
                payload=req,
                tracking_id=tracking_id,
                enqueued_at=time.time(),
            ))
        except queue.Full as exc:
            raise QueueFullError("SCMWriter queue at capacity") from exc
        return tracking_id

    def submit_cf_trace(self, req: CFTraceRequest) -> str:
        self._require_running()
        tracking_id = str(uuid.uuid4())
        try:
            self._queue.put_nowait(_EnqueuedWork(
                kind='cf_trace',
                payload=req,
                tracking_id=tracking_id,
                enqueued_at=time.time(),
            ))
        except queue.Full as exc:
            raise QueueFullError("SCMWriter queue at capacity") from exc
        return tracking_id

    def submit_effect_update(self, req: EffectUpdateRequest) -> str:
        """Enqueue a quantitative effect update on an existing edge.

        The edge must already exist and be status='active'; otherwise the
        update is dropped with an error counter bump (no exception from
        the submit path — this is a non-blocking queue API).
        """
        self._require_running()
        if req.scm_edge_id <= 0:
            raise SCMWriterError("scm_edge_id must be positive")
        tracking_id = str(uuid.uuid4())
        try:
            self._queue.put_nowait(_EnqueuedWork(
                kind='effect_update',
                payload=req,
                tracking_id=tracking_id,
                enqueued_at=time.time(),
            ))
        except queue.Full as exc:
            raise QueueFullError("SCMWriter queue at capacity") from exc
        return tracking_id

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            snap = dict(self._stats)
        snap['queue_depth'] = self._queue.qsize()
        snap['running'] = self._running.is_set()
        return snap

    # -- worker ----------------------------------------------------------------

    def _require_running(self) -> None:
        if not self._running.is_set():
            raise SCMWriterNotRunning("SCMWriter is not running; call start()")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                work = self._queue.get(timeout=_QUEUE_POLL_TIMEOUT_S)
            except queue.Empty:
                continue
            try:
                self._process(work)
            except Exception as exc:  # noqa: BLE001
                with self._stats_lock:
                    self._stats['errors'] += 1
                logger.exception("SCMWriter: failed to process %s: %s", work.kind, exc)
        # Drain: best-effort during shutdown.
        while not self._queue.empty():
            try:
                work = self._queue.get_nowait()
                self._process(work)
            except queue.Empty:
                break
            except Exception:  # noqa: BLE001
                logger.exception("SCMWriter: drain error")

    def _process(self, work: _EnqueuedWork) -> None:
        if work.kind == 'identification':
            self._process_identification(work.payload, work.tracking_id)
        elif work.kind == 'cf_trace':
            self._process_cf_trace(work.payload, work.tracking_id)
        elif work.kind == 'effect_update':
            self._process_effect_update(work.payload, work.tracking_id)
        else:
            logger.warning("SCMWriter: unknown work kind %s", work.kind)

    def _process_effect_update(self, req: EffectUpdateRequest, tracking_id: str) -> None:
        """Apply an effect-only update to an existing active edge.

        Silently skips edges that aren't active (stale update — the edge
        may have been superseded between read + write). Stats:
        effects_updated or effects_skipped_inactive.
        """
        eff = req.effect
        with psycopg.connect(self._conninfo_factory()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status FROM erik_ops.scm_edges WHERE id = %s FOR UPDATE",
                    (req.scm_edge_id,),
                )
                row = cur.fetchone()
                if row is None:
                    with self._stats_lock:
                        self._stats['errors'] += 1
                    logger.warning("effect_update: edge %s not found", req.scm_edge_id)
                    return
                if row[0] != 'active':
                    with self._stats_lock:
                        self._stats.setdefault('effects_skipped_inactive', 0)
                        self._stats['effects_skipped_inactive'] += 1
                    return
                cur.execute(
                    """
                    UPDATE erik_ops.scm_edges
                       SET effect_mean     = %s,
                           effect_std      = %s,
                           effect_ci_lower = %s,
                           effect_ci_upper = %s,
                           effect_scale    = COALESCE(%s, effect_scale),
                           updated_at      = NOW()
                     WHERE id = %s
                    """,
                    (
                        eff.mean, eff.std, eff.ci_lower, eff.ci_upper,
                        eff.scale, req.scm_edge_id,
                    ),
                )
                self._emit_write_log(cur, 'effect_updated', req.scm_edge_id, {
                    'source': req.source,
                    'effect_scale': eff.scale,
                    'effect_mean': eff.mean,
                    'effect_std': eff.std,
                    'tracking_id': tracking_id,
                    'trace': req.trace,
                })
                conn.commit()
                with self._stats_lock:
                    self._stats.setdefault('effects_updated', 0)
                    self._stats['effects_updated'] += 1

    def _process_identification(self, req: IdentificationRequest, tracking_id: str) -> None:
        last_exc: Optional[Exception] = None
        for backoff in [0.0, *_RETRY_BACKOFF_S]:
            if backoff > 0:
                time.sleep(backoff)
            try:
                self._do_process_identification(req, tracking_id)
                with self._stats_lock:
                    self._stats['identifications_processed'] += 1
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
        raise SCMWriterError(f"identification failed after retries: {last_exc}") from last_exc

    def _do_process_identification(self, req: IdentificationRequest, tracking_id: str) -> None:
        t0 = time.time()
        with psycopg.connect(self._conninfo_factory()) as conn:
            with conn.cursor() as cur:
                src_id = self._get_or_create_node(
                    cur, req.source_entity_id, req.source_node_class,
                    req.source_als_role, req.source_druggability,
                )
                tgt_id = self._get_or_create_node(
                    cur, req.target_entity_id, req.target_node_class,
                    req.target_als_role, req.target_druggability,
                )
                existing = self._get_active_edge(cur, src_id, tgt_id, req.adjustment_set_id)
                outcome, edge_id = _decide_disposition(existing, req, _ALGORITHM_STRENGTH)

                if outcome == 'rejected_weaker':
                    self._insert_identification(
                        cur, None, src_id, tgt_id, req,
                        outcome='rejected_weaker',
                        runtime_ms=int((time.time() - t0) * 1000),
                    )
                    conn.commit()
                    with self._stats_lock:
                        self._stats['edges_rejected_weaker'] += 1
                    return

                if outcome == 'superseded':
                    assert existing is not None
                    new_edge_id = self._insert_edge(cur, src_id, tgt_id, req)
                    self._set_superseded_by(cur, existing['id'], new_edge_id)
                    self._insert_identification(
                        cur, new_edge_id, src_id, tgt_id, req,
                        outcome='identified',
                        runtime_ms=int((time.time() - t0) * 1000),
                    )
                    self._patch_relationships_scm_edge_id(
                        cur, req.derived_from_rel_ids, new_edge_id,
                    )
                    self._emit_write_log(cur, 'edge_superseded', existing['id'], {
                        'superseded_by': new_edge_id,
                        'reason': 'stronger_identification',
                    })
                    self._emit_write_log(cur, 'edge_created', new_edge_id, {
                        'source_node_id': src_id, 'target_node_id': tgt_id,
                        'algorithm': req.algorithm, 'confidence': req.confidence,
                        'is_intervention_candidate': req.is_intervention_candidate,
                    })
                    if req.is_intervention_candidate:
                        self._emit_write_log(cur, 'intervention_flagged', new_edge_id, {
                            'source_entity_id': req.source_entity_id,
                            'target_entity_id': req.target_entity_id,
                        })
                    conn.commit()
                    with self._stats_lock:
                        self._stats['edges_created'] += 1
                        self._stats['edges_superseded'] += 1
                    return

                # outcome == 'created'
                new_edge_id = self._insert_edge(cur, src_id, tgt_id, req)
                self._insert_identification(
                    cur, new_edge_id, src_id, tgt_id, req,
                    outcome='identified',
                    runtime_ms=int((time.time() - t0) * 1000),
                )
                self._patch_relationships_scm_edge_id(
                    cur, req.derived_from_rel_ids, new_edge_id,
                )
                self._emit_write_log(cur, 'edge_created', new_edge_id, {
                    'source_node_id': src_id, 'target_node_id': tgt_id,
                    'algorithm': req.algorithm, 'confidence': req.confidence,
                    'is_intervention_candidate': req.is_intervention_candidate,
                })
                if req.is_intervention_candidate:
                    self._emit_write_log(cur, 'intervention_flagged', new_edge_id, {
                        'source_entity_id': req.source_entity_id,
                        'target_entity_id': req.target_entity_id,
                    })
                conn.commit()
                with self._stats_lock:
                    self._stats['edges_created'] += 1

    def _process_cf_trace(self, req: CFTraceRequest, tracking_id: str) -> None:
        with psycopg.connect(self._conninfo_factory()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO erik_ops.scm_cf_traces
                        (edge_id, query_id, abduction_state, intervention_do,
                         factual_outcome, counterfactual_outcome, regret,
                         patient_snapshot_id, runtime_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (query_id) DO NOTHING
                    RETURNING id
                    """,
                    (
                        req.edge_id, req.query_id,
                        json.dumps(req.abduction_state),
                        json.dumps(req.intervention_do),
                        json.dumps(req.factual_outcome),
                        json.dumps(req.counterfactual_outcome),
                        req.regret, req.patient_snapshot_id, req.runtime_ms,
                    ),
                )
                row = cur.fetchone()
                cf_id = row[0] if row else None
                if cf_id is not None:
                    self._emit_write_log(cur, 'cf_computed', cf_id, {
                        'edge_id': req.edge_id,
                        'patient_snapshot_id': req.patient_snapshot_id,
                    })
                conn.commit()
                with self._stats_lock:
                    self._stats['cf_traces_processed'] += 1

    # -- SQL helpers -----------------------------------------------------------

    def _get_or_create_node(
        self, cur, entity_id: str, node_class: str,
        als_role: Optional[str], druggability: Optional[float],
    ) -> int:
        cur.execute(
            "SELECT id FROM erik_ops.scm_nodes WHERE entity_id = %s",
            (entity_id,),
        )
        row = cur.fetchone()
        if row is not None:
            return int(row[0])
        cur.execute(
            """
            INSERT INTO erik_ops.scm_nodes
                (entity_id, node_class, als_role, druggability_prior)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (entity_id, node_class, als_role, druggability),
        )
        return int(cur.fetchone()[0])

    def _get_active_edge(
        self, cur, src_id: int, tgt_id: int, adj_set_id: Optional[int],
    ) -> Optional[dict[str, Any]]:
        cur.execute(
            """
            SELECT id, identification_algorithm, identification_confidence
              FROM erik_ops.scm_edges
             WHERE source_node_id = %s
               AND target_node_id = %s
               AND COALESCE(adjustment_set_id, 0) = COALESCE(%s, 0)
               AND status = 'active'
            """,
            (src_id, tgt_id, adj_set_id),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return {
            'id': int(row[0]),
            'algorithm': row[1],
            'confidence': float(row[2]) if row[2] is not None else 0.0,
        }

    def _insert_edge(self, cur, src_id: int, tgt_id: int, req: IdentificationRequest) -> int:
        eff = req.effect or EffectDistribution()
        cur.execute(
            """
            INSERT INTO erik_ops.scm_edges
                (source_node_id, target_node_id, edge_kind,
                 effect_mean, effect_std, effect_ci_lower, effect_ci_upper, effect_scale,
                 identification_algorithm, identification_confidence,
                 adjustment_set_id, derived_from_rel_ids,
                 transport_population, transport_conditions,
                 is_intervention_candidate, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                src_id, tgt_id, req.edge_kind,
                eff.mean, eff.std, eff.ci_lower, eff.ci_upper, eff.scale,
                req.algorithm, req.confidence,
                req.adjustment_set_id, list(req.derived_from_rel_ids),
                req.transport_population, json.dumps(req.transport_conditions),
                req.is_intervention_candidate, json.dumps(req.trace),
            ),
        )
        return int(cur.fetchone()[0])

    def _set_superseded_by(self, cur, old_id: int, new_id: int) -> None:
        cur.execute(
            """
            UPDATE erik_ops.scm_edges
               SET status = 'superseded', superseded_by = %s, updated_at = NOW()
             WHERE id = %s
            """,
            (new_id, old_id),
        )

    def _insert_identification(
        self, cur, edge_id: Optional[int],
        src_id: int, tgt_id: int, req: IdentificationRequest,
        outcome: str, runtime_ms: int,
    ) -> None:
        cur.execute(
            """
            INSERT INTO erik_ops.scm_identifications
                (edge_id, source_node_id, target_node_id, algorithm,
                 outcome, confidence, evidence_refs, trace, runtime_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                edge_id, src_id, tgt_id, req.algorithm, outcome,
                req.confidence, list(req.evidence_refs),
                json.dumps(req.trace), runtime_ms,
            ),
        )

    def _patch_relationships_scm_edge_id(self, cur, rel_ids: list[str], edge_id: int) -> None:
        if not rel_ids:
            return
        cur.execute(
            """
            UPDATE erik_core.relationships
               SET scm_edge_id = %s
             WHERE id = ANY(%s)
               AND (scm_edge_id IS NULL OR scm_edge_id != %s)
            """,
            (edge_id, list(rel_ids), edge_id),
        )

    def _emit_write_log(self, cur, operation: str, target_id: Optional[int], payload: dict) -> None:
        cur.execute(
            """
            INSERT INTO erik_ops.scm_write_log
                (operation, target_id, daemon, payload)
            VALUES (%s, %s, %s, %s)
            """,
            (operation, target_id, self._daemon_name, json.dumps(payload)),
        )


# ─── Pure helpers ─────────────────────────────────────────────────────────────

def _decide_disposition(
    existing: Optional[dict[str, Any]],
    req: IdentificationRequest,
    strength_rank: dict[str, int],
) -> tuple[str, Optional[int]]:
    """Return ('created'|'superseded'|'rejected_weaker', existing_edge_id_or_None).

    Rules:
      - No existing active edge → 'created'.
      - New algorithm strictly stronger → 'superseded'.
      - Same algorithm, new confidence ≥ old + DELTA → 'superseded'.
      - Otherwise → 'rejected_weaker'.
    """
    if existing is None:
        return ('created', None)
    old_rank = strength_rank.get(existing['algorithm'], 0)
    new_rank = strength_rank.get(req.algorithm, 0)
    if new_rank > old_rank:
        return ('superseded', existing['id'])
    if new_rank < old_rank:
        return ('rejected_weaker', existing['id'])
    # Same strength — require confidence delta.
    if req.confidence >= existing['confidence'] + _SUPERSEDE_CONFIDENCE_DELTA:
        return ('superseded', existing['id'])
    return ('rejected_weaker', existing['id'])


def _validate_identification_request(req: IdentificationRequest) -> None:
    if req.edge_kind not in ('causal', 'confounding', 'mediating'):
        raise SCMWriterError(f"invalid edge_kind: {req.edge_kind}")
    if not (0.0 <= req.confidence <= 1.0):
        raise SCMWriterError(f"confidence out of range: {req.confidence}")
    if req.algorithm not in _ALGORITHM_STRENGTH:
        raise SCMWriterError(f"unknown algorithm: {req.algorithm}")
    if not req.source_entity_id or not req.target_entity_id:
        raise SCMWriterError("source/target entity_id required")
    if req.source_entity_id == req.target_entity_id:
        raise SCMWriterError("self-loop not permitted")


# ─── Module-level singleton ────────────────────────────────────────────────────

_singleton_lock = threading.Lock()
_singleton: Optional[SCMWriter] = None


def get_scm_writer(conninfo_factory=None, queue_max: int = _DEFAULT_QUEUE_MAX) -> SCMWriter:
    """Return the shared SCMWriter, creating it on first call.

    conninfo_factory defaults to db.pool's standard factory.
    """
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                if conninfo_factory is None:
                    from db.pool import _make_conninfo
                    conninfo_factory = _make_conninfo
                _singleton = SCMWriter(conninfo_factory=conninfo_factory, queue_max=queue_max)
    return _singleton


def shutdown_scm_writer(timeout: float = 10.0) -> None:
    global _singleton
    if _singleton is not None:
        _singleton.stop(timeout=timeout)
        _singleton = None
