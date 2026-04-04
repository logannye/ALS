"""gnomAD constraint connector — gene-level LoF/missense constraint metrics.

Queries the local gnomAD v4.1 constraint metrics TSV to determine how
constrained (intolerant to mutation) an ALS target gene is. Answers:
"Is this gene under strong selection against loss-of-function or missense
mutations?" — critical for interpreting variant pathogenicity.

Database: /Volumes/Databank/databases/gnomad/gnomad_v4.1_constraint_metrics.tsv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_PATH = "/Volumes/Databank/databases/gnomad/gnomad_v4.1_constraint_metrics.tsv"


def _safe_float(val: Any) -> float | None:
    """Convert a value to float, returning None for NA/empty/invalid."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.upper() in ("NA", "NAN", "N/A", ".", "-"):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


class GnomADConnector(BaseConnector):
    """Query local gnomAD v4.1 constraint metrics for ALS target genes."""

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH, **kwargs):
        self._store = store
        self._file_path = file_path

    def fetch(self, *, gene: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not gene:
            return result
        if not Path(self._file_path).exists():
            result.errors.append(f"gnomAD file not found: {self._file_path}")
            return result

        try:
            row_data = self._search_gene(gene)
            if not row_data:
                return result

            # Extract key constraint metrics
            pli = _safe_float(row_data.get("lof_hc_lc.pLI"))
            loeuf = _safe_float(row_data.get("lof.oe_ci.upper"))
            mis_z = _safe_float(row_data.get("mis.z_score"))
            syn_z = _safe_float(row_data.get("syn.z_score"))
            lof_oe = _safe_float(row_data.get("lof.oe"))
            mis_oe = _safe_float(row_data.get("mis.oe"))
            lof_obs = _safe_float(row_data.get("lof.obs"))
            lof_exp = _safe_float(row_data.get("lof.exp"))

            # Build constraint interpretation
            parts: list[str] = [f"gnomAD v4.1 constraint for {gene}:"]

            # pLI interpretation
            if pli is not None:
                if pli > 0.9:
                    parts.append(
                        f"pLI={pli:.3f} — HIGHLY CONSTRAINED — extremely "
                        f"intolerant to LoF mutations"
                    )
                elif pli > 0.5:
                    parts.append(f"pLI={pli:.3f} — Moderately constrained")
                else:
                    parts.append(f"pLI={pli:.3f} — LoF-tolerant")

            # LOEUF interpretation
            if loeuf is not None:
                if loeuf < 0.35:
                    parts.append(
                        f"LOEUF={loeuf:.3f} — confirms strong LoF intolerance"
                    )
                else:
                    parts.append(f"LOEUF={loeuf:.3f}")

            # Missense constraint
            if mis_z is not None:
                if mis_z > 3.09:
                    parts.append(
                        f"mis_z={mis_z:.2f} — missense-constrained "
                        f"(significant depletion of missense variants)"
                    )
                else:
                    parts.append(f"mis_z={mis_z:.2f}")

            # Observed/expected LoF
            if lof_obs is not None and lof_exp is not None and lof_exp > 0:
                parts.append(
                    f"LoF observed/expected: {int(lof_obs)}/{lof_exp:.1f} "
                    f"(o/e={lof_oe:.3f})" if lof_oe is not None
                    else f"LoF observed/expected: {int(lof_obs)}/{lof_exp:.1f}"
                )

            claim = " ".join(parts) + "."

            evi = BaseEnvelope(
                id=f"evi:gnomad:{gene.lower()}_constraint",
                type="EvidenceItem",
                status="active",
                body={
                    "claim": claim,
                    "source": "gnomad_v4.1",
                    "gene": gene,
                    "pLI": pli,
                    "LOEUF": loeuf,
                    "mis_z": mis_z,
                    "syn_z": syn_z,
                    "lof_oe": lof_oe,
                    "mis_oe": mis_oe,
                    "lof_obs": lof_obs,
                    "lof_exp": lof_exp,
                    "transcript": row_data.get("transcript", ""),
                    "evidence_strength": "strong",
                    "pch_layer": 1,
                    "data_source": "gnomad",
                },
            )

            if self._store:
                self._store.upsert_object(evi)
            result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"gnomAD query failed for {gene}: {e}")

        return result

    def _search_gene(self, gene: str) -> dict | None:
        """Search the gnomAD constraint TSV for the given gene symbol.

        Returns the first matching row (preferring MANE Select transcript)
        as a dict of column_name -> value, or None if not found.
        """
        gene_upper = gene.upper()
        best: dict | None = None

        with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if row.get("gene", "").upper() != gene_upper:
                    continue
                # Prefer MANE Select transcript
                if row.get("mane_select", "").lower() == "true":
                    return row
                if best is None:
                    best = row

        return best
