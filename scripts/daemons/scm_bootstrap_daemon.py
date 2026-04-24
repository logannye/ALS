"""SCM Bootstrap Daemon — one-shot pass promoting high-confidence
``erik_core.relationships`` rows into ``erik_ops.scm_edges`` via SCMWriter.

Ported from Galen Phase 1 bootstrap (2026-04-16). Erik-specific behavior:

  * The evidence-type → identification-algorithm map below encodes what the
    ALS literature actually provides — iPSC motor-neuron assays, ALS mouse
    models, patient organoids — rather than Galen's cancer-focused set.

  * `is_intervention_candidate` is set when source entity has
    ``druggability_prior >= 0.5`` AND target is a mechanism / pathway /
    biomarker / clinical_endpoint. This is the flag the CompoundDaemon
    reads via scm_write_log to pick up new intervention points without
    polling the whole edges table.

  * Resumable: scm_bootstrap_progress.last_rel_id is updated per evidence
    type. On restart the daemon continues where it left off.

  * Idempotent: SCMWriter's supersession logic handles duplicate submits
    (same edge + same algorithm + same-or-lower confidence → rejected_weaker).

Flag-gated by ``scm_bootstrap_enabled``. Default off.
"""
from __future__ import annotations

import datetime as _dt
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

_EPOCH_FLOOR = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)

from config.loader import ConfigLoader
from db.pool import get_connection
from world_model.scm_writer import (
    EffectDistribution,
    IdentificationRequest,
    QueueFullError,
    get_scm_writer,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Evidence-type → identification-algorithm map.
# Conservative defaults — only map sources that we are confident map cleanly
# to a causal-identification algorithm. Sources not in the map stay in
# erik_core.relationships and are never promoted.
# -----------------------------------------------------------------------------

_EVIDENCE_TYPE_TO_ALGORITHM: dict[str, str] = {
    # Experimental / clinical gold tier
    'clinical_trial':             'rct',
    'clinicaltrials':             'rct',
    'clinicaltrials.gov':         'rct',
    # Cross-disease causal evidence inherited from Galen's SCM
    'galen_scm':                  'galen_scm',
    'galen_kg':                   'galen_scm',
    # Interventional cell / animal models (ALS-relevant)
    'ipsc_motor_neuron':          'ipsc_motor_neuron_assay',
    'als_mouse_model':            'als_mouse_model',
    'patient_organoid':           'patient_organoid',
    # CMap / docking — pattern-matching, lowest tier but still promotable
    'cmap':                       'cmap_signature_match',
    'docking':                    'docking_simulation',
    # Replicated observational — requires ≥2 sources; mapper below upgrades.
    'replicated_experiment':      'replicated_experiment',
    'pubmed':                     'replicated_experiment',  # only when ≥2 sources
    'biorxiv':                    'replicated_experiment',  # only when ≥2 sources
}

# Synthetic types produced by inference that must never auto-promote to
# scm_edges — they are not grounded in experimental evidence. Keeps the
# SCM substrate credible.
_EVIDENCE_TYPE_DENYLIST: frozenset[str] = frozenset({
    'inferred_from_evidence',
    'inferred_chain',
    'unknown',
    'manual_seed',
})


_ALS_ROLE_KEYWORDS: dict[str, list[str]] = {
    'gene':               ['gene', 'tardbp', 'sod1', 'fus', 'c9orf72', 'sigmar1',
                           'stmn2', 'unc13a', 'atxn2', 'optn', 'tbk1', 'nek1',
                           'vcp', 'pfn1', 'mtor', 'slc1a2', 'csf1r'],
    'compound':           ['drug', 'compound', 'riluzole', 'edaravone', 'rapamycin',
                           'tofersen', 'masitinib', 'ibudilast', 'phenylbutyrate'],
    'mechanism':          ['aggregation', 'proteostasis', 'autophagy', 'apoptosis',
                           'excitotoxicity', 'neuroinflammation', 'rna metabolism'],
    'pathway':            ['pathway', 'signaling', 'cascade', 'mtor signaling',
                           'ubiquitin proteasome'],
    'biomarker':          ['neurofilament', 'nfl', 'biomarker'],
    'clinical_endpoint':  ['alsfrs-r', 'alsfrs_r', 'survival', 'fvc', 'respiratory'],
}


def _infer_als_role(entity_name: str, entity_type: Optional[str]) -> str:
    """Infer als_role from entity name + declared type.

    Used only during bootstrap; on the live path the CompoundDaemon sets
    this explicitly. Returns 'other' when nothing matches — non-destructive.
    """
    if entity_type:
        et = entity_type.lower()
        if et in {'gene', 'protein'}:
            return 'gene'
        if et in {'drug', 'compound', 'intervention'}:
            return 'compound'
        if et in {'pathway'}:
            return 'pathway'
        if et in {'biomarker'}:
            return 'biomarker'
    nm = (entity_name or '').lower()
    for role, kws in _ALS_ROLE_KEYWORDS.items():
        if any(kw in nm for kw in kws):
            return role
    return 'other'


_INTERVENTION_TARGET_ROLES = frozenset({'mechanism', 'pathway', 'biomarker', 'clinical_endpoint'})


@dataclass
class _RelRow:
    rel_id: str
    created_at: Any          # datetime.datetime, kept as Any to avoid import churn
    source_entity_id: str
    target_entity_id: str
    source_entity_name: str
    target_entity_name: str
    source_entity_type: Optional[str]
    target_entity_type: Optional[str]
    source_druggability: Optional[float]
    target_druggability: Optional[float]
    confidence: float
    source_refs: list[str]
    evidence_type: str
    scm_edge_id: Optional[int]


class SCMBootstrapDaemon:
    """Resumable one-shot promotion of erik_core.relationships → scm_edges.

    Typical lifecycle (driven from run_loop.py):

        daemon = SCMBootstrapDaemon()
        t = threading.Thread(target=daemon.run, daemon=True)
        t.start()
        # ... on shutdown:
        daemon.stop()
        t.join(timeout=10)
    """

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("scm_bootstrap_interval_s", 30)
        self._min_conf = float(cfg.get("scm_bootstrap_min_confidence", 0.85))
        self._batch_size = int(cfg.get("scm_bootstrap_batch_size", 100))
        self._druggable_floor = float(cfg.get("scm_bootstrap_druggable_floor", 0.5))
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        """Main daemon loop.

        Runs forever while ``scm_bootstrap_enabled`` is true. When nothing
        is available this cycle, sleeps ``_interval_s`` and retries — new
        evidence can arrive at any time.
        """
        logger.info("SCMBootstrapDaemon started (min_conf=%.2f)", self._min_conf)
        print(f"[SCM-BOOTSTRAP] Daemon started (min_conf={self._min_conf:.2f})")
        empty_cycles = 0
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("scm_bootstrap_enabled", False):
                    self._stop.wait(60)
                    continue
                did_work = self.process_one_batch()
                if did_work:
                    empty_cycles = 0
                    self._stop.wait(self._interval_s)
                else:
                    # Exponential back-off on empty cycles, capped at 5 min,
                    # so an idle system doesn't hammer the DB.
                    empty_cycles += 1
                    sleep_s = min(self._interval_s * max(1, empty_cycles), 300)
                    self._stop.wait(sleep_s)
            except Exception as e:
                logger.exception("SCMBootstrapDaemon cycle failed")
                print(f"[SCM-BOOTSTRAP] Error: {e}")
                self._stop.wait(self._interval_s)

    # -- batch processing ------------------------------------------------------

    def process_one_batch(self) -> bool:
        """Promote up to ``batch_size`` rows from one evidence type.

        Returns True when work was done, False when nothing was available
        this cycle. **Does not persist a "complete" flag** — new evidence
        can arrive at any time, so every cycle re-scans using
        ``last_rel_id`` as the resume point.
        """
        evidence_type = self._pick_evidence_type()
        if evidence_type is None:
            return False

        progress = self._load_progress(evidence_type)
        rows = self._fetch_batch(evidence_type, progress.get("last_created_at"))
        if not rows:
            self._touch_progress(evidence_type)
            return False

        self._mark_running(evidence_type)
        writer = get_scm_writer()
        created = 0
        rejected = 0
        last_id = progress.get("last_rel_id")
        last_ts = progress.get("last_created_at")
        for row in rows:
            last_id = row.rel_id
            last_ts = row.created_at
            try:
                submitted = self._submit(writer, row, evidence_type)
            except QueueFullError:
                logger.warning("SCMWriter queue full; backing off this cycle")
                break
            if submitted:
                created += 1
            else:
                rejected += 1

        self._update_progress(
            evidence_type=evidence_type,
            last_rel_id=last_id,
            last_created_at=last_ts,
            processed_delta=len(rows),
            created_delta=created,
            rejected_delta=rejected,
        )
        print(
            f"[SCM-BOOTSTRAP] {evidence_type}: +{created} submitted, "
            f"{rejected} skipped (last_created_at={last_ts})"
        )
        return True

    def _submit(
        self,
        writer,
        row: _RelRow,
        evidence_type: str,
    ) -> bool:
        """Convert a _RelRow into an IdentificationRequest and submit it.

        Returns True when submitted, False when skipped (pre-submit checks).
        Post-submit disposition (created / superseded / rejected_weaker) is
        owned by SCMWriter.
        """
        if row.scm_edge_id is not None:
            # Already promoted — never re-enqueue.
            return False
        if row.confidence < self._min_conf:
            return False
        if row.source_entity_id == row.target_entity_id:
            return False

        algo = _EVIDENCE_TYPE_TO_ALGORITHM.get(evidence_type)
        # Observational-replicated fallback: any evidence type with ≥2 source_refs.
        if algo is None and len(row.source_refs) >= 2:
            algo = 'replicated_experiment'
        if algo is None:
            return False

        src_role = _infer_als_role(row.source_entity_name, row.source_entity_type)
        tgt_role = _infer_als_role(row.target_entity_name, row.target_entity_type)

        # Intervention-candidate heuristic: druggable source → non-compound target.
        src_drug = row.source_druggability or 0.0
        intervention = (
            src_role in {'compound', 'gene'}
            and src_drug >= self._druggable_floor
            and tgt_role in _INTERVENTION_TARGET_ROLES
        )

        req = IdentificationRequest(
            source_entity_id=row.source_entity_id,
            target_entity_id=row.target_entity_id,
            edge_kind='causal',
            algorithm=algo,
            confidence=row.confidence,
            source_node_class=('treatment' if src_role == 'compound' else 'covariate'),
            target_node_class=('outcome' if tgt_role == 'clinical_endpoint' else 'covariate'),
            effect=EffectDistribution(),
            evidence_refs=row.source_refs,
            derived_from_rel_ids=[row.rel_id],
            source_als_role=src_role,
            target_als_role=tgt_role,
            source_druggability=row.source_druggability,
            target_druggability=row.target_druggability,
            is_intervention_candidate=intervention,
            trace={'bootstrap_evidence_type': evidence_type},
            daemon_source='scm_bootstrap',
        )
        writer.submit_identification(req)
        return True

    # -- SQL helpers -----------------------------------------------------------

    def _pick_evidence_type(self) -> Optional[str]:
        """Rotate to an evidence type with potentially-unprocessed rows.

        Strategy (in order):
          1. Discover + seed any newly-seen evidence types in relationships
             (skipping ``_EVIDENCE_TYPE_DENYLIST``).
          2. Among promotable types in ``_EVIDENCE_TYPE_TO_ALGORITHM``, prefer
             'pending' > 'running' > 'idle', then oldest-updated_at first.
             This rotation ensures that one slow evidence type doesn't
             starve another — and an 'idle' type gets re-checked next cycle.
        """
        # Step 1: discover.
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT LOWER(COALESCE(evidence_type, 'unknown')) AS et
                      FROM erik_core.relationships
                     WHERE scm_edge_id IS NULL
                """)
                discovered = [r[0] for r in cur.fetchall()
                              if r[0] and r[0] not in _EVIDENCE_TYPE_DENYLIST]
                for et in discovered:
                    cur.execute("""
                        INSERT INTO erik_ops.scm_bootstrap_progress(evidence_type)
                        VALUES (%s)
                        ON CONFLICT (evidence_type) DO NOTHING
                    """, (et,))
                conn.commit()

        # Step 2: pick with preference for promotable (mapped) types, then
        # by status priority, then by staleness.
        allowed = [et for et in discovered if et in _EVIDENCE_TYPE_TO_ALGORITHM]
        if not allowed:
            # No mapped types discovered — still fetch any seeded progress row
            # so the daemon can at least complete its idle cycle.
            allowed = discovered
        if not allowed:
            return None

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT evidence_type
                      FROM erik_ops.scm_bootstrap_progress
                     WHERE evidence_type = ANY(%s)
                     ORDER BY CASE status
                                WHEN 'pending' THEN 0
                                WHEN 'running' THEN 1
                                WHEN 'idle'    THEN 2
                                ELSE 3 END ASC,
                              updated_at ASC NULLS FIRST
                     LIMIT 1
                """, (allowed,))
                row = cur.fetchone()
        return row[0] if row else (allowed[0] if allowed else None)

    def _load_progress(self, evidence_type: str) -> dict[str, Any]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT last_rel_id, last_created_at, processed_count,
                           edges_created, edges_rejected, status
                      FROM erik_ops.scm_bootstrap_progress
                     WHERE evidence_type = %s
                """, (evidence_type,))
                row = cur.fetchone()
        if not row:
            return {"last_rel_id": None, "last_created_at": None,
                    "processed_count": 0, "edges_created": 0,
                    "edges_rejected": 0, "status": "pending"}
        return {"last_rel_id": row[0], "last_created_at": row[1],
                "processed_count": row[2], "edges_created": row[3],
                "edges_rejected": row[4], "status": row[5]}

    def _mark_running(self, evidence_type: str) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.scm_bootstrap_progress
                       SET status = 'running', started_at = COALESCE(started_at, NOW()),
                           updated_at = NOW()
                     WHERE evidence_type = %s
                """, (evidence_type,))
                conn.commit()

    def _touch_progress(self, evidence_type: str) -> None:
        """Bump updated_at after a no-op fetch so rotation advances next cycle."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.scm_bootstrap_progress
                       SET updated_at = NOW()
                     WHERE evidence_type = %s
                """, (evidence_type,))
                conn.commit()

    def _update_progress(
        self,
        evidence_type: str,
        last_rel_id: Optional[str],
        last_created_at: Any,
        processed_delta: int,
        created_delta: int,
        rejected_delta: int,
    ) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.scm_bootstrap_progress
                       SET last_rel_id = %s,
                           last_created_at = GREATEST(last_created_at, %s),
                           processed_count = processed_count + %s,
                           edges_created = edges_created + %s,
                           edges_rejected = edges_rejected + %s,
                           updated_at = NOW()
                     WHERE evidence_type = %s
                """, (last_rel_id, last_created_at, processed_delta,
                      created_delta, rejected_delta, evidence_type))
                conn.commit()

    def _fetch_batch(
        self,
        evidence_type: str,
        last_created_at: Any,
    ) -> list[_RelRow]:
        """Fetch up to ``batch_size`` candidates for this evidence type.

        Resume cursor: ``last_created_at`` from scm_bootstrap_progress.
        erik_core.relationships.id is TEXT and *not* monotonic under
        UUID-style IDs, so ordering/filtering by id drops rows whose IDs
        happen to sort before earlier-processed IDs. created_at is the
        invariant that's actually monotonic.
        """
        query = """
            SELECT r.id,
                   r.created_at,
                   r.source_id,
                   r.target_id,
                   COALESCE(se.name::text, se.id) AS src_name,
                   COALESCE(te.name::text, te.id) AS tgt_name,
                   se.entity_type AS src_type,
                   te.entity_type AS tgt_type,
                   NULLIF(se.properties->>'druggability_prior', '')::float AS src_drug,
                   NULLIF(te.properties->>'druggability_prior', '')::float AS tgt_drug,
                   r.confidence,
                   r.sources,
                   LOWER(COALESCE(r.evidence_type, 'unknown')) AS ev_type,
                   r.scm_edge_id
              FROM erik_core.relationships r
              LEFT JOIN erik_core.entities se ON se.id = r.source_id
              LEFT JOIN erik_core.entities te ON te.id = r.target_id
             WHERE r.scm_edge_id IS NULL
               AND r.confidence >= %s
               AND r.created_at > %s
               AND LOWER(COALESCE(r.evidence_type, 'unknown')) = %s
             ORDER BY r.created_at ASC, r.id ASC
             LIMIT %s
        """
        rows: list[_RelRow] = []
        floor = last_created_at if last_created_at is not None else _EPOCH_FLOOR
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (
                    self._min_conf,
                    floor,
                    evidence_type,
                    self._batch_size,
                ))
                for r in cur.fetchall():
                    raw_sources = r[11]
                    source_refs: list[str] = []
                    if isinstance(raw_sources, list):
                        source_refs = [str(s) for s in raw_sources if s]
                    rows.append(_RelRow(
                        rel_id=str(r[0]),
                        created_at=r[1],
                        source_entity_id=str(r[2]),
                        target_entity_id=str(r[3]),
                        source_entity_name=r[4] or str(r[2]),
                        target_entity_name=r[5] or str(r[3]),
                        source_entity_type=r[6],
                        target_entity_type=r[7],
                        source_druggability=r[8],
                        target_druggability=r[9],
                        confidence=float(r[10] or 0.0),
                        source_refs=source_refs,
                        evidence_type=r[12] or 'unknown',
                        scm_edge_id=r[13],
                    ))
        return rows

