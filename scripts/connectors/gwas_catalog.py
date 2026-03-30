"""GWAS Catalog connector — genome-wide association study results for ALS risk loci.

Queries the local GWAS Catalog associations TSV for ALS-associated SNPs,
risk loci, and mapped genes. Critical for interpreting Erik's genetic results
and understanding population-level ALS risk architecture.

File: /Volumes/Databank/databases/gwas/gwas-catalog-download-associations-alt-full.tsv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_PATH = "/Volumes/Databank/databases/gwas/gwas-catalog-download-associations-alt-full.tsv"

_ALS_TRAITS = frozenset({
    "amyotrophic lateral sclerosis",
    "frontotemporal dementia",
    "motor neuron disease",
    "als",
    "als/ftd",
    "age at onset of amyotrophic lateral sclerosis",
    "amyotrophic lateral sclerosis survival",
})


class GWASCatalogConnector(BaseConnector):
    """Query GWAS Catalog for ALS-associated genetic loci."""

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH):
        self._store = store
        self._file_path = file_path

    def fetch(self, *, gene: str = "", trait: str = "amyotrophic lateral sclerosis", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not Path(self._file_path).exists():
            result.errors.append(f"GWAS Catalog not found: {self._file_path}")
            return result

        try:
            hits = self._search(gene=gene, trait=trait)
            if not hits:
                return result

            # Summary evidence
            genes_found = set()
            for h in hits:
                for g in h.get("mapped_genes", "").split(","):
                    g = g.strip()
                    if g and g != "-":
                        genes_found.add(g)

            snps = [h.get("snp", "") for h in hits if h.get("snp")]
            claim = (
                f"GWAS Catalog: {len(hits)} ALS-associated loci"
                + (f" involving {gene}" if gene else "")
                + f". Mapped genes: {', '.join(sorted(genes_found)[:10])}. "
                f"Top SNPs: {', '.join(snps[:5])}."
            )

            evi_id = f"evi:gwas_als_{gene.lower()}" if gene else "evi:gwas_als_summary"
            evi = BaseEnvelope(
                id=evi_id,
                type="EvidenceItem",
                status="active",
                body={
                    "claim": claim,
                    "source": "gwas_catalog",
                    "gene": gene,
                    "n_loci": len(hits),
                    "mapped_genes": sorted(genes_found),
                    "snps": snps[:10],
                    "evidence_strength": "strong",
                    "pch_layer": 1,
                    "protocol_layer": "root_cause_suppression",
                },
            )

            if self._store:
                self._store.upsert_object(evi)
                result.evidence_items_added += 1

            # Individual locus evidence for top hits
            for h in hits[:5]:
                snp = h.get("snp", "")
                if not snp:
                    continue
                locus_evi = BaseEnvelope(
                    id=f"evi:gwas_{snp.lower().replace(' ', '_')}",
                    type="EvidenceItem",
                    status="active",
                    body={
                        "claim": (
                            f"GWAS locus {snp} at chr{h.get('chr', '?')}:{h.get('pos', '?')} "
                            f"associated with {h.get('trait', 'ALS')}. "
                            f"Mapped gene: {h.get('mapped_genes', 'unknown')}. "
                            f"p-value: {h.get('p_value', 'N/A')}. "
                            f"OR/Beta: {h.get('or_beta', 'N/A')}."
                        ),
                        "source": "gwas_catalog",
                        "snp": snp,
                        "chr": h.get("chr", ""),
                        "position": h.get("pos", ""),
                        "mapped_genes": h.get("mapped_genes", ""),
                        "p_value": h.get("p_value", ""),
                        "or_beta": h.get("or_beta", ""),
                        "trait": h.get("trait", ""),
                        "evidence_strength": "strong",
                        "pch_layer": 1,
                        "protocol_layer": "root_cause_suppression",
                    },
                )
                if self._store:
                    try:
                        self._store.upsert_object(locus_evi)
                        result.evidence_items_added += 1
                    except Exception:
                        pass

        except Exception as e:
            result.errors.append(f"GWAS search failed: {e}")

        return result

    def _search(self, gene: str = "", trait: str = "", max_results: int = 100) -> list[dict]:
        hits: list[dict] = []

        with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                disease_trait = (row.get("DISEASE/TRAIT") or "").lower()

                # Filter by ALS-related traits
                if not any(t in disease_trait for t in _ALS_TRAITS):
                    continue

                # Filter by gene if specified
                if gene:
                    mapped = (row.get("MAPPED_GENE") or "") + " " + (row.get("REPORTED GENE(S)") or "")
                    if gene.upper() not in mapped.upper():
                        continue

                hits.append({
                    "snp": row.get("SNPS", ""),
                    "chr": row.get("CHR_ID", ""),
                    "pos": row.get("CHR_POS", ""),
                    "mapped_genes": row.get("MAPPED_GENE", ""),
                    "reported_genes": row.get("REPORTED GENE(S)", ""),
                    "trait": row.get("DISEASE/TRAIT", ""),
                    "p_value": row.get("P-VALUE", ""),
                    "or_beta": row.get("OR or BETA", ""),
                    "pubmed": row.get("PUBMEDID", ""),
                })

                if len(hits) >= max_results:
                    break

        return hits
