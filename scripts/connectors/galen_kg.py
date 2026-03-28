"""GalenKGConnector — cross-references ALS entities against Galen's cancer KG.

Galen's knowledge graph (galen_kg database) contains 731K+ entities and 6.5M+
relationships covering cancer biology. Many pathways overlap with ALS:
autophagy/mTOR, HDAC, oxidative stress, neuroinflammation, ubiquitin-proteasome.

This connector performs read-only queries against galen_kg to find
cross-disease knowledge relevant to Erik's ALS treatment.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from connectors.base import BaseConnector, ConnectorResult

ALS_CROSS_REFERENCE_GENES = [
    "SOD1", "TARDBP", "FUS", "C9orf72", "STMN2", "UNC13A",
    "SIGMAR1", "SLC1A2", "MTOR", "CSF1R", "OPTN", "TBK1", "NEK1",
]


class GalenKGConnector(BaseConnector):
    """Query Galen's cancer KG for ALS-relevant cross-disease knowledge."""

    def __init__(self, store: Any = None, database: str = "galen_kg"):
        self._store = store
        self._database = database
        self._user = os.environ.get("USER", "logannye")

    def fetch(self, *, genes: Optional[list[str]] = None, **kwargs) -> ConnectorResult:
        """Query galen_kg for relationships involving the specified genes.

        Returns a ConnectorResult with evidence_items_added count.

        Uses UNION instead of OR to allow Postgres to use the index on
        entities.name for each branch, avoiding a full sequential scan
        of the 6.5M-row relationships table.
        """
        genes = genes or ALS_CROSS_REFERENCE_GENES[:3]
        result = ConnectorResult()

        rows: list[tuple] = []
        try:
            import psycopg
            conn = psycopg.connect(
                f"dbname={self._database} user={self._user}",
                connect_timeout=10,
                options="-c statement_timeout=30000 -c work_mem=16MB",
            )
            try:
                with conn.cursor() as cur:
                    placeholders = ",".join(["%s"] * len(genes))
                    # UNION approach: each branch hits the entities.name index
                    # instead of a full table scan with OR.
                    cur.execute(f"""
                        SELECT DISTINCT source_name, source_type,
                               relationship_type, target_name, target_type
                        FROM (
                            (SELECT e1.name AS source_name,
                                    e1.entity_type AS source_type,
                                    r.relationship_type,
                                    e2.name AS target_name,
                                    e2.entity_type AS target_type
                             FROM entities e1
                             JOIN relationships r ON r.source_id = e1.id
                             JOIN entities e2 ON r.target_id = e2.id
                             WHERE e1.name IN ({placeholders})
                             LIMIT 50)

                            UNION ALL

                            (SELECT e1.name AS source_name,
                                    e1.entity_type AS source_type,
                                    r.relationship_type,
                                    e2.name AS target_name,
                                    e2.entity_type AS target_type
                             FROM entities e2
                             JOIN relationships r ON r.target_id = e2.id
                             JOIN entities e1 ON r.source_id = e1.id
                             WHERE e2.name IN ({placeholders})
                             LIMIT 50)
                        ) sub
                        LIMIT 50
                    """, genes + genes)
                    rows = cur.fetchall()
            finally:
                conn.close()
        except Exception as e:
            result.errors.append(f"GalenKG query failed: {e}")
            return result

        # Build evidence items and store them
        evidence_added = 0
        for source_name, source_type, rel_type, target_name, target_type in rows:
            claim = (
                f"[Galen cross-reference] {source_name} ({source_type}) "
                f"{rel_type} {target_name} ({target_type})"
            )
            if self._store is not None and evidence_added < 20:
                try:
                    from ontology.base import Provenance
                    from ontology.enums import (
                        EvidenceDirection,
                        EvidenceStrength,
                        SourceSystem,
                    )
                    from ontology.evidence import EvidenceItem

                    evi_id = (
                        f"evi:galen:{source_name}_{target_name}"
                        .lower().replace(" ", "_")
                    )
                    evi = EvidenceItem(
                        id=evi_id,
                        claim=claim,
                        direction=EvidenceDirection.supports,
                        strength=EvidenceStrength.preclinical,
                        provenance=Provenance(
                            source_system=SourceSystem.database,
                            source_artifact_id=f"galen_kg:{source_name}",
                            asserted_by="galen_cross_reference",
                        ),
                        body={
                            "source_name": source_name,
                            "target_name": target_name,
                            "relationship_type": rel_type,
                            "provenance_source_system": "galen_cross_reference",
                        },
                    )
                    self._store.upsert_object(evi)
                    evidence_added += 1
                except Exception:
                    pass

        result.evidence_items_added = evidence_added
        return result
