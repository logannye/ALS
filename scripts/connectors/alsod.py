"""ALSoD (ALS Online Genetics Database) connector.

Fetches ALS gene variant data from alsod.ac.uk — the authoritative curated
database of ALS genetic mutations, maintained by King's College London.

Data includes:
- 160+ ALS-associated genes with evidence categorization
- Variant positions, HGVS nomenclature, patient counts, publication counts
- Gene evidence categories: Definitive, Strong, Moderate, Clinical modifier, Tenuous

URL pattern: https://alsod.ac.uk/output/gene.php/{GENE_SYMBOL}
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope


# Complete ALSoD gene catalog with evidence tiers (scraped from alsod.ac.uk main page)
ALSOD_GENES: dict[str, str] = {
    # Definitive ALS genes (17)
    "ANXA11": "definitive", "C9orf72": "definitive", "CHCHD10": "definitive",
    "EPHA4": "definitive", "FUS": "definitive", "HNRNPA1": "definitive",
    "KIF5A": "definitive", "NEK1": "definitive", "OPTN": "definitive",
    "PFN1": "definitive", "SOD1": "definitive", "TARDBP": "definitive",
    "TBK1": "definitive", "UBQLN2": "definitive", "UNC13A": "definitive",
    "VAPB": "definitive", "VCP": "definitive",
    # Strong evidence (6)
    "ATXN1": "strong", "CCNF": "strong", "CFAP410": "strong",
    "HFE": "strong", "NIPA1": "strong", "SCFD1": "strong",
    "TUBA4A": "strong",
    # Moderate evidence (selected — most relevant to Erik)
    "ANG": "moderate", "CHMP2B": "moderate", "DNAJC7": "moderate",
    "ERBB4": "moderate", "FIG4": "moderate", "GLE1": "moderate",
    "SARM1": "moderate", "SMN1": "moderate", "SQSTM1": "moderate",
    # Clinical modifiers
    "ATXN2": "clinical_modifier", "CAMTA1": "clinical_modifier",
    "ENAH": "clinical_modifier",
}

# Genes most relevant to Erik's case (sporadic TDP-43 likely, pending genetics)
ERIK_PRIORITY_GENES = [
    "TARDBP", "C9orf72", "SOD1", "FUS", "UNC13A", "NEK1", "TBK1",
    "OPTN", "ANXA11", "CHCHD10", "KIF5A", "TUBA4A", "SQSTM1",
    "ATXN2", "HNRNPA1",
]

_BASE_URL = "https://alsod.ac.uk/output/gene.php"


class ALSoDConnector(BaseConnector):
    """Connector for the ALS Online Genetics Database (ALSoD).

    Fetches gene variant data for ALS-associated genes, producing
    evidence items that capture variant counts, evidence tier, and
    clinical relevance to Erik's case.
    """

    def __init__(self, store: Any = None):
        self._store = store

    def fetch(self, *, gene: str = "", **kwargs) -> ConnectorResult:
        """Fetch variant data for an ALS gene from ALSoD.

        If gene is empty, rotates through ERIK_PRIORITY_GENES.
        """
        result = ConnectorResult()

        if not gene:
            # Default: pick from priority genes based on some rotation
            step = kwargs.get("step", 0)
            gene = ERIK_PRIORITY_GENES[step % len(ERIK_PRIORITY_GENES)]

        url = f"{_BASE_URL}/{gene}"
        try:
            import requests
            response = self._retry_with_backoff(
                requests.get, url, timeout=self.REQUEST_TIMEOUT,
                headers={"User-Agent": "Erik-ALS-Research/1.0 (academic research)"},
            )
            if response.status_code != 200:
                result.errors.append(f"ALSoD returned {response.status_code} for {gene}")
                return result

            # Parse the HTML response for variant data
            html = response.text
            variants = self._parse_gene_page(html, gene)
            evidence_tier = ALSOD_GENES.get(gene, "unknown")

            if variants:
                # Create a summary evidence item for this gene
                evi = self._build_gene_evidence(gene, variants, evidence_tier)
                if self._store and evi:
                    try:
                        self._store.upsert_object(evi)
                        result.evidence_items_added += 1
                    except Exception:
                        pass

                # Create individual variant evidence items for high-count variants
                for var in variants:
                    if var.get("patient_count", 0) >= 2:
                        var_evi = self._build_variant_evidence(gene, var, evidence_tier)
                        if self._store and var_evi:
                            try:
                                self._store.upsert_object(var_evi)
                                result.evidence_items_added += 1
                            except Exception:
                                pass

        except Exception as e:
            result.errors.append(f"ALSoD fetch failed for {gene}: {str(e)}")

        return result

    def _parse_gene_page(self, html: str, gene: str) -> list[dict]:
        """Parse variant data from an ALSoD gene page HTML.

        Extracts variant mnemonics, positions, patient counts, and publication counts
        from the HTML table structure.
        """
        variants: list[dict] = []

        # Extract variant rows — look for patterns like "A5T" followed by numbers
        # ALSoD uses table rows with: mnemonic, chr_position, hgvs, pubmed_ids, pub_count, patient_count
        # We parse conservatively — extracting what we can from the HTML

        # Pattern: variant mnemonics (e.g., A5T, L145F, D91A, G93A)
        # These follow amino acid substitution patterns: [A-Z]\d+[A-Z*]
        mnemonic_pattern = re.compile(r'\b([A-Z]\d{1,4}[A-Z*](?:fsX\d+)?)\b')
        mnemonics = mnemonic_pattern.findall(html)

        # Patient count pattern — look for numbers near "patient" context
        # In ALSoD tables, patient counts appear as integers
        patient_pattern = re.compile(r'(\d+)\s*patient')
        pub_pattern = re.compile(r'(\d+)\s*publication')

        # HGVS patterns (c.XXX>Y)
        hgvs_pattern = re.compile(r'(c\.\d+[ACGT]>[ACGT]|c\.\d+(?:del|ins|dup)[A-Z]*)')
        hgvs_matches = hgvs_pattern.findall(html)

        # Build variant records from mnemonics (deduplicated)
        seen = set()
        for mnemonic in mnemonics:
            if mnemonic in seen:
                continue
            seen.add(mnemonic)

            # Try to find associated data near this mnemonic
            var = {
                "mnemonic": mnemonic,
                "gene": gene,
                "patient_count": 0,
                "publication_count": 0,
            }

            # Look for this mnemonic in context to find counts
            idx = html.find(mnemonic)
            if idx >= 0:
                context = html[idx:idx + 500]
                # Try to extract numbers following the mnemonic
                nums = re.findall(r'>(\d+)<', context)
                if len(nums) >= 2:
                    var["publication_count"] = int(nums[-2]) if nums[-2].isdigit() else 0
                    var["patient_count"] = int(nums[-1]) if nums[-1].isdigit() else 0

            variants.append(var)

        return variants

    def _build_gene_evidence(
        self, gene: str, variants: list[dict], evidence_tier: str,
    ) -> Optional[BaseEnvelope]:
        """Build a summary evidence item for a gene's ALSoD entry."""
        total_patients = sum(v.get("patient_count", 0) for v in variants)
        total_pubs = sum(v.get("publication_count", 0) for v in variants)
        top_variants = sorted(variants, key=lambda v: v.get("patient_count", 0), reverse=True)[:5]
        top_str = ", ".join(f"{v['mnemonic']} ({v.get('patient_count', 0)} patients)" for v in top_variants if v.get("patient_count", 0) > 0)

        # Map evidence tier to strength
        strength_map = {
            "definitive": "strong", "strong": "strong",
            "moderate": "moderate", "clinical_modifier": "moderate",
            "tenuous": "emerging", "unknown": "emerging",
        }

        claim = (
            f"ALSoD catalogs {len(variants)} variants in {gene} across {total_patients} patients "
            f"and {total_pubs} publications. Evidence tier: {evidence_tier}. "
        )
        if top_str:
            claim += f"Most frequent variants: {top_str}."

        # Relevance to Erik
        if gene == "TARDBP":
            claim += " TARDBP encodes TDP-43 — Erik's working subtype hypothesis (sporadic TDP-43, posterior 0.65)."
        elif gene == "C9orf72":
            claim += " C9orf72 repeat expansion is the most common genetic cause of ALS — relevant given Erik's mother's Alzheimer's."
        elif gene == "SOD1":
            claim += " SOD1 mutations are the second most common fALS cause — tofersen (approved) targets this gene."

        return BaseEnvelope(
            id=f"evi:alsod_{gene.lower()}_summary",
            type="EvidenceItem",
            status="active",
            body={
                "claim": claim,
                "source": "alsod",
                "gene": gene,
                "evidence_tier": evidence_tier,
                "n_variants": len(variants),
                "n_patients": total_patients,
                "n_publications": total_pubs,
                "top_variants": [v["mnemonic"] for v in top_variants[:5]],
                "evidence_strength": strength_map.get(evidence_tier, "emerging"),
                "pch_layer": 1,
                "protocol_layer": "root_cause_suppression",
                "data_source": "alsod",
            },
        )

    def _build_variant_evidence(
        self, gene: str, variant: dict, evidence_tier: str,
    ) -> Optional[BaseEnvelope]:
        """Build an evidence item for a specific high-frequency variant."""
        mnemonic = variant["mnemonic"]
        patients = variant.get("patient_count", 0)
        pubs = variant.get("publication_count", 0)

        return BaseEnvelope(
            id=f"evi:alsod_{gene.lower()}_{mnemonic.lower()}",
            type="EvidenceItem",
            status="active",
            body={
                "claim": (
                    f"{gene} variant {mnemonic} reported in {patients} ALS patients "
                    f"across {pubs} publications (ALSoD). "
                    f"Gene evidence tier: {evidence_tier}."
                ),
                "source": "alsod",
                "gene": gene,
                "variant": mnemonic,
                "patient_count": patients,
                "publication_count": pubs,
                "evidence_tier": evidence_tier,
                "evidence_strength": "strong" if patients >= 5 else "moderate",
                "pch_layer": 1,
                "protocol_layer": "root_cause_suppression",
                "data_source": "alsod",
            },
        )
