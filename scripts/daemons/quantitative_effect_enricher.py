"""QuantitativeEffectEnricher — fill ``effect_mean`` / ``effect_std`` on
``scm_edges`` that the bootstrap daemon wrote without quantitative data.

Motivation: ``SCMBootstrapDaemon`` promotes high-confidence relationships
to ``erik_ops.scm_edges`` based on their identification algorithm but
does not know the effect size. ``DrugResponseSimulator`` returns zero
slope delta for any edge whose ``effect_mean IS NULL``. Without the
enricher, CPTS is architecturally complete and empirically silent.

Week 1 source scope:
  * **ChEMBL local DB** — pChEMBL for (compound, target) pairs where
    source als_role='compound' and target als_role='gene'.
  * Future sources: PRO-ACT cohort-matched slopes, ClinicalTrials.gov
    outcomes, CMap signature connectivity.

Design:
  * Pull candidate edges (effect_mean IS NULL AND status='active' AND
    edge_kind='causal' AND source.als_role='compound' AND
    target.als_role='gene').
  * For each, extract compound name + gene symbol from the entity_id,
    call ``chembl_effect_mapper.map_compound_target_to_effect``, and
    submit an ``EffectUpdateRequest`` to SCMWriter.
  * Skip candidates whose names don't resolve — never write spurious data.

Flag-gated by ``effect_enricher_enabled`` (default OFF). Safe to turn on
after SCM bootstrap has produced meaningful edge count.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from world_model.chembl_effect_mapper import (
    DEFAULT_CHEMBL_DB,
    DEFAULT_UNIPROT_TSV,
    map_compound_target_to_effect,
)
from world_model.scm_writer import (
    EffectUpdateRequest,
    QueueFullError,
    get_scm_writer,
)

logger = logging.getLogger(__name__)


def _entity_id_tail(entity_id: str) -> str:
    """Extract the human-readable name from an entity id like 'compound:riluzole'."""
    if ':' in entity_id:
        return entity_id.rsplit(':', 1)[-1]
    return entity_id


class QuantitativeEffectEnricher:
    """Batch-enriches NULL-effect scm_edges from ChEMBL.

    Lifecycle mirrors the other daemons — run() loops on a cadence,
    run_once() is the testable unit.
    """

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self._interval_s = int(cfg.get("effect_enricher_interval_s", 600))    # 10 min
        self._batch_size = int(cfg.get("effect_enricher_batch_size", 50))
        self._chembl_db = cfg.get("computation_chembl_path", DEFAULT_CHEMBL_DB) or DEFAULT_CHEMBL_DB
        self._uniprot_tsv = cfg.get("uniprot_data_path", DEFAULT_UNIPROT_TSV) or DEFAULT_UNIPROT_TSV
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        print(f"[EFFECT-ENRICHER] Daemon started (chembl={self._chembl_db})")
        empty_cycles = 0
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("effect_enricher_enabled", False):
                    self._stop.wait(60)
                    continue
                stats = self.run_once()
                if stats.get("updated", 0) > 0:
                    empty_cycles = 0
                    print(
                        f"[EFFECT-ENRICHER] cycle: considered={stats['considered']}, "
                        f"resolved={stats['resolved']}, updated={stats['updated']}, "
                        f"skipped_no_mapping={stats['skipped_no_mapping']}, "
                        f"skipped_wrong_roles={stats['skipped_wrong_roles']}"
                    )
                    self._stop.wait(self._interval_s)
                else:
                    empty_cycles += 1
                    self._stop.wait(min(self._interval_s * max(1, empty_cycles), 1800))
            except Exception as e:
                logger.exception("effect_enricher cycle failed")
                print(f"[EFFECT-ENRICHER] Error: {e}")
                self._stop.wait(self._interval_s)

    # -- single-cycle API ------------------------------------------------------

    def run_once(self) -> dict[str, Any]:
        candidates = self._fetch_candidates(self._batch_size)
        stats = {
            "considered": len(candidates),
            "resolved": 0,
            "updated": 0,
            "skipped_no_mapping": 0,
            "skipped_wrong_roles": 0,
        }
        if not candidates:
            return stats

        writer = get_scm_writer()
        for edge_id, src_entity, tgt_entity, src_role, tgt_role in candidates:
            if src_role != 'compound' or tgt_role != 'gene':
                stats["skipped_wrong_roles"] += 1
                continue
            compound_name = _entity_id_tail(src_entity)
            gene_symbol = _entity_id_tail(tgt_entity)
            try:
                result = map_compound_target_to_effect(
                    compound_name=compound_name,
                    gene_symbol=gene_symbol,
                    chembl_db=self._chembl_db,
                    uniprot_tsv=self._uniprot_tsv,
                )
            except Exception as e:
                logger.exception("mapper failed for edge=%s (%s→%s): %s",
                                 edge_id, src_entity, tgt_entity, e)
                stats["skipped_no_mapping"] += 1
                continue
            if result is None:
                stats["skipped_no_mapping"] += 1
                continue
            stats["resolved"] += 1
            try:
                writer.submit_effect_update(EffectUpdateRequest(
                    scm_edge_id=edge_id,
                    effect=result.effect,
                    source='chembl_binding_assay',
                    daemon_source='effect_enricher',
                    trace={
                        'compound_chembl_id': result.compound_chembl_id,
                        'target_uniprot': result.target_uniprot,
                        'n_activities': result.n_activities,
                        'dominant_activity_type': result.dominant_activity_type,
                    },
                ))
                stats["updated"] += 1
            except QueueFullError:
                logger.warning("SCMWriter queue full — enricher backing off")
                break

        return stats

    # -- SQL -------------------------------------------------------------------

    def _fetch_candidates(self, limit: int) -> list[tuple[int, str, str, str, str]]:
        """Return NULL-effect active causal edges worth trying to enrich.

        Returns (edge_id, source_entity_id, target_entity_id,
                 source_als_role, target_als_role) tuples.

        Filters:
          * effect_mean IS NULL — never re-enrich.
          * edge_kind = 'causal' — mediating / confounding edges aren't
            the simulator's forward path.
          * als_role pair compound → gene — Week 1 ChEMBL-only scope.
          * Ordered by identification_confidence DESC so we prioritise
            high-confidence edges (where CPTS will most trust the output).
        """
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT e.id,
                           sn_src.entity_id,
                           sn_tgt.entity_id,
                           sn_src.als_role,
                           sn_tgt.als_role
                      FROM erik_ops.scm_edges e
                      JOIN erik_ops.scm_nodes sn_src ON sn_src.id = e.source_node_id
                      JOIN erik_ops.scm_nodes sn_tgt ON sn_tgt.id = e.target_node_id
                     WHERE e.effect_mean IS NULL
                       AND e.status = 'active'
                       AND e.edge_kind = 'causal'
                       AND sn_src.als_role = 'compound'
                       AND sn_tgt.als_role = 'gene'
                     ORDER BY e.identification_confidence DESC, e.id ASC
                     LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
        return [
            (int(r[0]), r[1], r[2], r[3], r[4])
            for r in rows
        ]
