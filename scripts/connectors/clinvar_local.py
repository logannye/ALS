"""ClinVar local connector — pathogenicity lookup from variant_summary.txt.

Queries the local ClinVar variant_summary file (3.8GB TSV) for pathogenicity
classifications of ALS-associated gene variants. Answers: "Is this variant
pathogenic, benign, or VUS?" — critical for interpreting Erik's Invitae results.

Database: /Volumes/Databank/databases/clinvar/variant_summary.txt
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_PATH = "/Volumes/Databank/databases/clinvar/variant_summary.txt"

# ALS-related phenotype keywords for filtering
_ALS_PHENOTYPES = frozenset({
    "amyotrophic lateral sclerosis",
    "frontotemporal dementia",
    "motor neuron disease",
    "als",
    "ftd",
    "spinal muscular atrophy",
})


class ClinVarLocalConnector(BaseConnector):
    """Query local ClinVar variant_summary for ALS gene pathogenicity."""

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH):
        self._store = store
        self._file_path = file_path

    def fetch(self, *, gene: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not gene:
            return result
        if not Path(self._file_path).exists():
            result.errors.append(f"ClinVar file not found: {self._file_path}")
            return result

        try:
            variants = self._search_gene(gene)
            if not variants:
                return result

            # Summary evidence item
            pathogenic = [v for v in variants if "pathogenic" in v["significance"].lower()]
            benign = [v for v in variants if "benign" in v["significance"].lower()]
            vus = [v for v in variants if "uncertain" in v["significance"].lower()]

            claim = (
                f"ClinVar has {len(variants)} variants for {gene}: "
                f"{len(pathogenic)} pathogenic/likely pathogenic, "
                f"{len(benign)} benign/likely benign, "
                f"{len(vus)} VUS. "
            )
            if pathogenic:
                top = pathogenic[:3]
                claim += f"Key pathogenic variants: {', '.join(v['name'][:30] for v in top)}. "

            evi = BaseEnvelope(
                id=f"evi:clinvar_{gene.lower()}_summary",
                type="EvidenceItem",
                status="active",
                body={
                    "claim": claim,
                    "source": "clinvar",
                    "gene": gene,
                    "n_pathogenic": len(pathogenic),
                    "n_benign": len(benign),
                    "n_vus": len(vus),
                    "n_total": len(variants),
                    "pathogenic_variants": [v["name"][:50] for v in pathogenic[:10]],
                    "evidence_strength": "strong",
                    "pch_layer": 1,
                    "protocol_layer": "root_cause_suppression",
                },
            )

            if self._store:
                self._store.upsert_object(evi)
                result.evidence_items_added += 1

            # Individual evidence for pathogenic variants (up to 5)
            for var in pathogenic[:5]:
                var_evi = BaseEnvelope(
                    id=f"evi:clinvar_{gene.lower()}_{var['allele_id']}",
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": (
                            f"{gene} variant {var['name'][:50]} classified as "
                            f"{var['significance']} in ClinVar. "
                            f"Phenotype: {var.get('phenotype', 'unspecified')[:60]}."
                        ),
                        "source": "clinvar",
                        "gene": gene,
                        "variant_name": var["name"][:80],
                        "clinical_significance": var["significance"],
                        "allele_id": var["allele_id"],
                        "phenotype": var.get("phenotype", ""),
                        "evidence_strength": "strong",
                        "pch_layer": 1,
                        "protocol_layer": "root_cause_suppression",
                    },
                )
                if self._store:
                    try:
                        self._store.upsert_object(var_evi)
                        result.evidence_items_added += 1
                    except Exception:
                        pass

        except Exception as e:
            result.errors.append(f"ClinVar query failed for {gene}: {e}")

        return result

    def _search_gene(self, gene: str, max_results: int = 200) -> list[dict]:
        """Scan variant_summary.txt for variants in the given gene.

        The file is large (~3.8GB) so we stream-read and filter by GeneSymbol column.
        """
        variants: list[dict] = []
        gene_upper = gene.upper()

        with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)

            # Find column indices
            try:
                gene_idx = header.index("GeneSymbol")
                name_idx = header.index("Name")
                sig_idx = header.index("ClinicalSignificance")
                allele_idx = header.index("#AlleleID")
                pheno_idx = header.index("PhenotypeList") if "PhenotypeList" in header else -1
            except ValueError:
                return variants

            for row in reader:
                if len(row) <= max(gene_idx, name_idx, sig_idx, allele_idx):
                    continue
                if row[gene_idx].upper() != gene_upper:
                    continue

                var = {
                    "allele_id": row[allele_idx],
                    "name": row[name_idx],
                    "significance": row[sig_idx],
                    "phenotype": row[pheno_idx] if pheno_idx >= 0 and pheno_idx < len(row) else "",
                }
                variants.append(var)
                if len(variants) >= max_results:
                    break

        return variants
