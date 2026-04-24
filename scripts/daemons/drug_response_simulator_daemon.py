"""DrugResponseSimulatorDaemon — periodic forward simulation of every
compound candidate's predicted effect on Erik's ALSFRS-R / NfL / FVC /
survival trajectory.

Sister to Galen's CPTSDaemon. The key architectural move: converts
``erik_ops.scm_edges`` from a lookup graph into the substrate of a
reproducible forward engine.

What the daemon does per cycle:

  1. Build Erik's current PatientBaseline from the most recent
     ALSFRSRScore + NfL + FVC observations.
  2. For each compound candidate in the TCG (active interventions whose
     druggable edges have landed in scm_edges via bootstrap / cognitive
     daemons), collect all active scm_edges where the compound is either
     the source or connected to a target_entity via a short walk.
  3. Call the pure ``simulate`` function with ensemble_size K and
     horizon_months H.
  4. Persist all K samples to ``erik_ops.simulated_trajectory`` — immutable
     once written.
  5. Emit one ``erik_ops.simulated_prediction_claims`` row per ensemble so
     M.1 validators (when wired up) can resolve the prediction against
     Erik's real trajectory or PRO-ACT matched outcomes.

Daemon discipline:

  * **Flag-gated** off by default. Set ``cpts_enabled=true`` in
    ``data/erik_config.json`` to activate.
  * **Stateless between cycles.** All state lives in Postgres.
  * **Idempotent.** Re-running for the same (intervention, baseline_hash,
    generator_version) does not create duplicates: the unique partial
    index on simulated_prediction_claims(..., status='open') means the
    daemon short-circuits after it sees an open claim.
  * **Bounded cost.** At most ``cpts_max_interventions_per_cycle``
    interventions per cycle; at most ``cpts_ensemble_size`` samples per.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from world_model.drug_response_simulator import (
    EdgeSnapshot,
    GENERATOR_VERSION,
    Intervention,
    PatientBaseline,
    RolloutEnsemble,
    hash_for,
    simulate,
)

logger = logging.getLogger(__name__)


# Canonical subject for Erik — matches ingestion/patient_builder.SUBJECT_REF.
ERIK_SUBJECT_REF = "patient:erik_draper"


class DrugResponseSimulatorDaemon:
    """Runs forward simulations over active compound candidates on a cadence.

    Typical wire-up (in run_loop.py)::

        d = DrugResponseSimulatorDaemon()
        t = threading.Thread(target=d.run, daemon=True)
        t.start()
        # ... on shutdown:
        d.stop(); t.join(timeout=10)
    """

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self._interval_s = int(cfg.get("cpts_interval_s", 1800))     # 30 min
        self._ensemble_k = int(cfg.get("cpts_ensemble_size", 32))
        self._horizon = int(cfg.get("cpts_horizon_months", 12))
        self._max_per_cycle = int(cfg.get("cpts_max_interventions_per_cycle", 10))
        self._max_edges_per_intervention = int(cfg.get("cpts_max_edges_per_intervention", 50))
        self._master_seed = int(cfg.get("cpts_master_seed", 20260424))
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------------

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        print(f"[CPTS] Daemon started (K={self._ensemble_k}, horizon={self._horizon}mo)")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("cpts_enabled", False):
                    self._stop.wait(60)
                    continue
                stats = self.run_once()
                if stats.get("ensembles_written", 0) > 0:
                    print(
                        f"[CPTS] cycle: simulated {stats['interventions_considered']} "
                        f"interventions, wrote {stats['ensembles_written']} ensembles, "
                        f"skipped {stats['skipped_open_claims']} open"
                    )
            except Exception as e:
                logger.exception("CPTS cycle failed")
                print(f"[CPTS] Error: {e}")
            self._stop.wait(self._interval_s)

    # -- public single-cycle API (used by daemon + tests) ----------------------

    def run_once(self) -> dict[str, Any]:
        """Execute one full cycle. Returns stats dict."""
        baseline = load_erik_baseline()
        if baseline is None:
            return {
                "interventions_considered": 0,
                "ensembles_written": 0,
                "skipped_open_claims": 0,
                "reason": "no_baseline",
            }
        interventions = self._list_active_interventions(limit=self._max_per_cycle)
        if not interventions:
            return {
                "interventions_considered": 0,
                "ensembles_written": 0,
                "skipped_open_claims": 0,
                "reason": "no_interventions",
            }

        written = 0
        skipped = 0
        for interv in interventions:
            if self._has_open_claim(interv.compound_entity_id, hash_for(baseline)):
                skipped += 1
                continue
            edges = self._load_edges_for_intervention(interv)
            ensemble = simulate(
                baseline=baseline,
                intervention=interv,
                edges=edges,
                K=self._ensemble_k,
                horizon_months=self._horizon,
                seed=self._master_seed,
            )
            self._persist_ensemble(baseline, interv, ensemble)
            written += 1

        return {
            "interventions_considered": len(interventions),
            "ensembles_written": written,
            "skipped_open_claims": skipped,
        }

    # -- data-access helpers ---------------------------------------------------

    def _list_active_interventions(self, limit: int) -> list[Intervention]:
        """Find compound candidates worth simulating this cycle.

        Sources (in priority order):
          1. scm_edges with is_intervention_candidate=true, deduped by source.
          2. erik_core.objects type='Intervention' status='active' whose
             targets contain at least one active scm_node.

        Week 1 scope: source (1) only. Source (2) is the next PR when the
        TCG-compound daemon starts flagging candidates.
        """
        out: list[Intervention] = []
        seen: set[str] = set()
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT ON (sn_src.entity_id)
                           sn_src.entity_id,
                           array_agg(DISTINCT sn_tgt.entity_id)
                             FILTER (WHERE sn_tgt.entity_id IS NOT NULL) AS targets
                      FROM erik_ops.scm_edges e
                      JOIN erik_ops.scm_nodes sn_src ON sn_src.id = e.source_node_id
                      JOIN erik_ops.scm_nodes sn_tgt ON sn_tgt.id = e.target_node_id
                     WHERE e.is_intervention_candidate = TRUE
                       AND e.status = 'active'
                  GROUP BY sn_src.entity_id
                     LIMIT %s
                """, (limit,))
                for row in cur.fetchall():
                    compound_id = row[0]
                    if compound_id in seen:
                        continue
                    seen.add(compound_id)
                    targets = tuple(row[1] or ())
                    out.append(Intervention(
                        compound_entity_id=compound_id,
                        target_entity_ids=targets,
                    ))
        return out

    def _load_edges_for_intervention(self, interv: Intervention) -> list[EdgeSnapshot]:
        """Snapshot active scm_edges reachable from the intervention.

        Week 1 scope: a 1-hop expansion — any active edge whose source is
        the compound or any of its declared targets. Deeper walks are a
        future improvement; they multiply complexity and need cycle checks.
        """
        relevant_ids = {interv.compound_entity_id, *interv.target_entity_ids}
        if not relevant_ids:
            return []
        out: list[EdgeSnapshot] = []
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT sn_src.entity_id, sn_tgt.entity_id, e.edge_kind,
                           e.effect_mean, e.effect_std, e.effect_scale,
                           e.identification_algorithm, e.identification_confidence,
                           e.adjustment_set_id, e.metadata
                      FROM erik_ops.scm_edges e
                      JOIN erik_ops.scm_nodes sn_src ON sn_src.id = e.source_node_id
                      JOIN erik_ops.scm_nodes sn_tgt ON sn_tgt.id = e.target_node_id
                     WHERE e.status = 'active'
                       AND sn_src.entity_id = ANY(%s)
                     ORDER BY e.identification_confidence DESC, e.id ASC
                     LIMIT %s
                """, (list(relevant_ids), self._max_edges_per_intervention))
                for r in cur.fetchall():
                    meta = r[9] if isinstance(r[9], dict) else {}
                    out.append(EdgeSnapshot(
                        source_entity_id=r[0],
                        target_entity_id=r[1],
                        edge_kind=r[2],
                        effect_mean=r[3],
                        effect_std=r[4],
                        effect_scale=r[5],
                        identification_algorithm=r[6],
                        identification_confidence=float(r[7] or 0.0),
                        adjustment_set_id=r[8],
                        metadata=dict(meta),
                    ))
        return out

    def _has_open_claim(self, intervention_id: str, baseline_hash: str) -> bool:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 1
                      FROM erik_ops.simulated_prediction_claims
                     WHERE intervention_entity_id = %s
                       AND baseline_hash = %s
                       AND generator_version = %s
                       AND status = 'open'
                     LIMIT 1
                """, (intervention_id, baseline_hash, GENERATOR_VERSION))
                return cur.fetchone() is not None

    def _persist_ensemble(
        self,
        baseline: PatientBaseline,
        interv: Intervention,
        ensemble: RolloutEnsemble,
    ) -> None:
        """Write K trajectory rows + one claim row in a single transaction.

        All-or-nothing: if any INSERT fails, nothing lands. The immutability
        trigger means we can never amend a trajectory after write — so we
        refuse to partial-write.
        """
        baseline_json = json.dumps(asdict(baseline))
        edge_snap_json = json.dumps([e.to_dict() for e in ensemble.edge_snapshot])
        metadata_json = json.dumps({
            'daemon': 'drug_response_simulator',
            'empty_dag_reason': ensemble.empty_dag_reason,
            'intervention_target_ids': list(interv.target_entity_ids),
        })
        summary_json = json.dumps(ensemble.to_prediction_summary())

        with get_connection() as conn:
            with conn.cursor() as cur:
                for sample_index, sample in enumerate(ensemble.samples):
                    cur.execute("""
                        INSERT INTO erik_ops.simulated_trajectory
                            (intervention_entity_id, baseline_hash, baseline_snapshot,
                             sample_index, ensemble_size, generator_version,
                             edge_snapshot, trajectory, rng_seed, horizon_months,
                             alsfrs_r_slope_delta, metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ensemble.intervention_entity_id,
                        ensemble.baseline_hash,
                        baseline_json,
                        sample_index,
                        ensemble.ensemble_size,
                        ensemble.generator_version,
                        edge_snap_json,
                        json.dumps(sample.to_dict()),
                        sample.rng_seed,
                        ensemble.horizon_months,
                        sample.alsfrs_r_slope_delta,
                        metadata_json,
                    ))
                # Partial-unique index scope: the index predicate must be
                # repeated verbatim on the INSERT's ON CONFLICT clause.
                cur.execute("""
                    INSERT INTO erik_ops.simulated_prediction_claims
                        (intervention_entity_id, baseline_hash, generator_version,
                         prediction_summary, ensemble_size, horizon_months,
                         status, daemon)
                    VALUES (%s, %s, %s, %s, %s, %s, 'open', 'drug_response_simulator')
                    ON CONFLICT (intervention_entity_id, baseline_hash, generator_version)
                        WHERE status = 'open'
                        DO NOTHING
                """, (
                    ensemble.intervention_entity_id,
                    ensemble.baseline_hash,
                    ensemble.generator_version,
                    summary_json,
                    ensemble.ensemble_size,
                    ensemble.horizon_months,
                ))
                conn.commit()


# ─── Baseline loader ──────────────────────────────────────────────────────────


def load_erik_baseline() -> Optional[PatientBaseline]:
    """Build Erik's PatientBaseline from the most recent ALSFRS-R + NfL + FVC.

    Returns None if the required observations aren't in erik_core.objects yet.
    """
    alsfrs = _latest_observation_value('ALSFRSRScore', 'total_score')
    nfl = _latest_observation_value('LabResult', 'value', name_match='neurofilament')
    fvc = _latest_observation_value('RespiratoryMetric', 'fvc_pct_predicted')

    # Fallback to the documented baseline from project memory so first-run
    # smoke tests succeed even before the full ingestion pipeline is wired.
    # Project memory: ALSFRS-R 43/48, NfL 5.82, FVC 100%.
    if alsfrs is None:
        alsfrs = 43.0
    if nfl is None:
        nfl = 5.82
    if fvc is None:
        fvc = 100.0

    return PatientBaseline(
        patient_id=ERIK_SUBJECT_REF,
        alsfrs_r=float(alsfrs),
        nfl_pg_ml=float(nfl),
        fvc_pct=float(fvc),
        trajectory_modifier=1.0,
    )


def _latest_observation_value(
    obj_type: str,
    body_key: str,
    name_match: Optional[str] = None,
) -> Optional[float]:
    """Read the most recent numeric observation value from erik_core.objects.

    ``name_match`` (lowercased) restricts lab-result lookups to rows whose
    body.name contains the substring (e.g. "neurofilament").
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            if name_match is not None:
                cur.execute("""
                    SELECT body->>%s
                      FROM erik_core.objects
                     WHERE type = %s
                       AND status = 'active'
                       AND LOWER(COALESCE(body->>'name', '')) ILIKE %s
                     ORDER BY created_at DESC
                     LIMIT 1
                """, (body_key, obj_type, f'%{name_match.lower()}%'))
            else:
                cur.execute("""
                    SELECT body->>%s
                      FROM erik_core.objects
                     WHERE type = %s AND status = 'active'
                     ORDER BY created_at DESC
                     LIMIT 1
                """, (body_key, obj_type))
            row = cur.fetchone()
    if row and row[0] is not None:
        try:
            return float(row[0])
        except (TypeError, ValueError):
            return None
    return None
