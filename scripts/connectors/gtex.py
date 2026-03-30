"""GTEx (Genotype-Tissue Expression) connector for ALS target gene expression.

Queries the local GTEx v10 SQLite database to determine expression levels
of ALS target genes across brain and neural tissues. Answers: "Is this gene
expressed in the tissues where Erik's motor neurons are dying?"

Database: /Volumes/Databank/databases/gtex_v10.db
Tables: gene_expression (gene_symbol, tissue_id, median_tpm),
        tissue_dictionary (tissue_id, tissue_name)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

# Neural tissues most relevant to ALS
_ALS_RELEVANT_TISSUES = [
    "brain spinal cord cervical c-1",  # Primary — motor neurons here
    "brain cortex",
    "brain frontal cortex ba9",
    "brain cerebellum",
    "brain hypothalamus",
    "nerve tibial",
]

_DEFAULT_DB_PATH = "/Volumes/Databank/databases/gtex_v10.db"


class GTExConnector(BaseConnector):
    """Query GTEx for tissue expression of ALS target genes."""

    def __init__(self, store: Any = None, db_path: str = _DEFAULT_DB_PATH):
        self._store = store
        self._db_path = db_path

    def fetch(self, *, gene: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not gene:
            return result
        if not Path(self._db_path).exists():
            result.errors.append(f"GTEx DB not found: {self._db_path}")
            return result

        try:
            conn = sqlite3.connect(f"file:{self._db_path}?mode=ro", uri=True)
            cur = conn.cursor()

            cur.execute("""
                SELECT td.tissue_name, ge.median_tpm
                FROM gene_expression ge
                JOIN tissue_dictionary td ON ge.tissue_id = td.tissue_id
                WHERE ge.gene_symbol = ?
                ORDER BY ge.median_tpm DESC
            """, (gene,))
            rows = cur.fetchall()
            conn.close()

            if not rows:
                return result

            # Find expression in ALS-relevant neural tissues
            neural_expr = [(t, tpm) for t, tpm in rows if any(at in t for at in _ALS_RELEVANT_TISSUES)]
            spinal_expr = [(t, tpm) for t, tpm in rows if "spinal cord" in t]
            max_tissue, max_tpm = rows[0]
            spinal_tpm = spinal_expr[0][1] if spinal_expr else 0.0

            # Build evidence
            neural_str = ", ".join(f"{t}: {tpm:.1f} TPM" for t, tpm in neural_expr[:4])
            claim = (
                f"{gene} expression in neural tissues (GTEx v10): {neural_str}. "
                f"Spinal cord C1: {spinal_tpm:.1f} TPM. "
                f"Highest expression: {max_tissue} ({max_tpm:.1f} TPM). "
            )
            if spinal_tpm > 10:
                claim += f"HIGHLY EXPRESSED in spinal cord — validates {gene} as ALS-relevant target."
            elif spinal_tpm > 1:
                claim += f"Moderately expressed in spinal cord."
            else:
                claim += f"Low spinal cord expression — may limit {gene} as direct ALS target."

            evi = BaseEnvelope(
                id=f"evi:gtex_{gene.lower()}_neural_expression",
                type="EvidenceItem",
                status="active",
                body={
                    "claim": claim,
                    "source": "gtex_v10",
                    "gene": gene,
                    "spinal_cord_tpm": spinal_tpm,
                    "max_tissue": max_tissue,
                    "max_tpm": max_tpm,
                    "neural_tissues": {t: tpm for t, tpm in neural_expr},
                    "evidence_strength": "strong",
                    "pch_layer": 1,
                    "protocol_layer": "root_cause_suppression",
                    "experiment_type": "gene_expression",
                },
            )

            if self._store:
                self._store.upsert_object(evi)
                result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"GTEx query failed for {gene}: {e}")

        return result
