"""SpliceAI connector — splice-altering variant predictions for ALS genes.

Queries per-gene SpliceAI TSV files for delta score predictions that indicate
splice-altering variants. Critical for UNC13A and STMN2 cryptic splice site
targets — two of Erik's top drug design priorities.

Directory: /Volumes/Databank/databases/spliceai/
Files: spliceai_{GENE}.tsv
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_DIR = "/Volumes/Databank/databases/spliceai"
_HIGH_THRESHOLD = 0.5
_MODERATE_THRESHOLD = 0.2

_ALS_SPLICE_GENES = {
    "UNC13A": {"chr": "chr19", "importance": "Cryptic exon inclusion upon TDP-43 loss — top drug target"},
    "STMN2": {"chr": "chr8", "importance": "Cryptic exon — loss impairs axon regeneration"},
    "TARDBP": {"chr": "chr1", "importance": "TDP-43 regulates splicing of UNC13A/STMN2"},
    "FUS": {"chr": "chr16", "importance": "RNA splicing regulator"},
    "SOD1": {"chr": "chr21", "importance": "Familial ALS gene"},
    "C9orf72": {"chr": "chr9", "importance": "Repeat expansion affects RNA processing"},
    "KIF5A": {"chr": "chr12", "importance": "C-terminal splice mutations cause ALS"},
    "HNRNPA1": {"chr": "chr12", "importance": "RNA splicing/stress granules"},
}

_SCORE_COLUMNS = ("DS_AG", "DS_AL", "DS_DG", "DS_DL")


def _safe_float(val: Any) -> float:
    """Convert a value to float, returning 0.0 on error."""
    if val is None:
        return 0.0
    s = str(val).strip()
    if not s or s.upper() in ("NA", "NAN", "N/A", ".", "-"):
        return 0.0
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0


class SpliceAIConnector(BaseConnector):
    """Query local SpliceAI per-gene TSV files for splice-altering variants."""

    def __init__(self, store: Any = None, data_dir: str = _DEFAULT_DIR, **kwargs):
        self._store = store
        self._data_dir = data_dir

    def fetch(self, *, gene: str = "", variant: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not gene:
            return result

        # Gracefully handle missing data directory (data not yet downloaded)
        if not Path(self._data_dir).exists():
            return result

        # Locate the TSV file
        tsv_path = self._find_gene_file(gene)
        if tsv_path is None:
            return result

        try:
            variants = self._parse_tsv(tsv_path)
        except Exception as e:
            result.errors.append(f"SpliceAI parse failed for {gene}: {e}")
            return result

        if not variants:
            return result

        # Classify variants
        high_impact = [v for v in variants if v["max_ds"] >= _HIGH_THRESHOLD]
        moderate_impact = [
            v for v in variants
            if _MODERATE_THRESHOLD <= v["max_ds"] < _HIGH_THRESHOLD
        ]

        # Sort by max_ds descending
        variants.sort(key=lambda v: v["max_ds"], reverse=True)
        high_impact.sort(key=lambda v: v["max_ds"], reverse=True)

        # ALS relevance context
        gene_upper = gene.upper()
        gene_info = _ALS_SPLICE_GENES.get(gene_upper, {})
        als_context = gene_info.get("importance", "")

        # Build summary evidence item
        claim_parts = [
            f"SpliceAI predicts {len(high_impact)} high-impact "
            f"(DS>={_HIGH_THRESHOLD}) and {len(moderate_impact)} moderate-impact "
            f"(DS>={_MODERATE_THRESHOLD}) splice-altering variants in {gene_upper} "
            f"({len(variants)} total scored).",
        ]
        if als_context:
            claim_parts.append(f"ALS relevance: {als_context}.")

        top_variants = variants[:10]
        top_variant_summaries = []
        for v in top_variants:
            top_variant_summaries.append({
                "chrom": v.get("CHROM", ""),
                "pos": v.get("POS", ""),
                "ref": v.get("REF", ""),
                "alt": v.get("ALT", ""),
                "max_ds": round(v["max_ds"], 4),
                "ds_ag": round(v["DS_AG"], 4),
                "ds_al": round(v["DS_AL"], 4),
                "ds_dg": round(v["DS_DG"], 4),
                "ds_dl": round(v["DS_DL"], 4),
            })

        summary_evi = BaseEnvelope(
            id=f"evi:spliceai:{gene_upper.lower()}_summary",
            type="EvidenceItem",
            status="active",
            body={
                "claim": " ".join(claim_parts),
                "source": "spliceai",
                "gene": gene_upper,
                "n_variants": len(variants),
                "n_high_impact": len(high_impact),
                "n_moderate_impact": len(moderate_impact),
                "top_variants": top_variant_summaries,
                "als_context": als_context,
                "evidence_strength": "strong" if high_impact else "moderate",
                "pch_layer": 2,
                "protocol_layer": "pathology_reversal",
            },
        )

        if self._store:
            self._store.upsert_object(summary_evi)
        result.evidence_items_added += 1

        # Build individual evidence for top 5 high-impact variants
        for v in high_impact[:5]:
            try:
                var_id = (
                    f"{v.get('CHROM', '')}_{v.get('POS', '')}_"
                    f"{v.get('REF', '')}_{v.get('ALT', '')}"
                ).replace(":", "_").replace("-", "_")

                var_claim = (
                    f"{gene_upper} variant {v.get('CHROM', '')}:{v.get('POS', '')} "
                    f"{v.get('REF', '')}>{v.get('ALT', '')} has SpliceAI max delta "
                    f"score {v['max_ds']:.3f} (AG={v['DS_AG']:.3f}, AL={v['DS_AL']:.3f}, "
                    f"DG={v['DS_DG']:.3f}, DL={v['DS_DL']:.3f}) — HIGH impact "
                    f"splice-altering variant."
                )

                var_evi = BaseEnvelope(
                    id=f"evi:spliceai:{gene_upper.lower()}_{var_id}",
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": var_claim,
                        "source": "spliceai",
                        "gene": gene_upper,
                        "chrom": v.get("CHROM", ""),
                        "pos": v.get("POS", ""),
                        "ref": v.get("REF", ""),
                        "alt": v.get("ALT", ""),
                        "max_ds": round(v["max_ds"], 4),
                        "ds_ag": round(v["DS_AG"], 4),
                        "ds_al": round(v["DS_AL"], 4),
                        "ds_dg": round(v["DS_DG"], 4),
                        "ds_dl": round(v["DS_DL"], 4),
                        "impact": "high",
                        "evidence_strength": "strong",
                        "pch_layer": 2,
                        "protocol_layer": "pathology_reversal",
                    },
                )

                if self._store:
                    self._store.upsert_object(var_evi)
                result.evidence_items_added += 1
            except Exception:
                pass

        return result

    def _find_gene_file(self, gene: str) -> str | None:
        """Locate the SpliceAI TSV file for the given gene."""
        gene_upper = gene.upper()

        # Try canonical filename first
        canonical = os.path.join(self._data_dir, f"spliceai_{gene_upper}.tsv")
        if os.path.exists(canonical):
            return canonical

        # Fallback: search for any file containing the gene name
        try:
            for fn in os.listdir(self._data_dir):
                if gene_upper in fn.upper() and fn.endswith(".tsv"):
                    return os.path.join(self._data_dir, fn)
        except OSError:
            pass

        return None

    def _parse_tsv(self, tsv_path: str) -> list[dict]:
        """Parse a SpliceAI TSV file and return scored variant dicts."""
        variants: list[dict] = []

        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                scores = {col: _safe_float(row.get(col)) for col in _SCORE_COLUMNS}
                max_ds = max(scores.values())

                entry = {
                    "CHROM": row.get("CHROM", ""),
                    "POS": row.get("POS", ""),
                    "REF": row.get("REF", ""),
                    "ALT": row.get("ALT", ""),
                    "SYMBOL": row.get("SYMBOL", ""),
                    "DS_AG": scores["DS_AG"],
                    "DS_AL": scores["DS_AL"],
                    "DS_DG": scores["DS_DG"],
                    "DS_DL": scores["DS_DL"],
                    "max_ds": max_ds,
                }
                variants.append(entry)

        return variants
