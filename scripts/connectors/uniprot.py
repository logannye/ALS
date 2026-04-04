"""UniProt SwissProt connector — protein function, PTMs, disease, domains.

Queries the local UniProt/SwissProt human proteome TSV for comprehensive
protein annotations. Produces up to 4 evidence items per protein covering
function, post-translational modifications, disease associations, and
structural domains.

Database: /Volumes/Databank/databases/uniprot/uniprot_human_swissprot.tsv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

# UniProt TSV contains very large fields (Natural variant, Modified residue)
# that exceed Python's default csv field size limit of 131072 bytes.
csv.field_size_limit(sys.maxsize)

_DEFAULT_PATH = "/Volumes/Databank/databases/uniprot/uniprot_human_swissprot.tsv"


class UniProtConnector(BaseConnector):
    """Query local UniProt/SwissProt TSV for protein annotations."""

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH, **kwargs):
        self._store = store
        self._file_path = file_path

    def fetch(
        self, *, gene: str = "", accession: str = "", uniprot: str = "", **kwargs
    ) -> ConnectorResult:
        result = ConnectorResult()

        # Factory compatibility: uniprot kwarg maps to accession
        if uniprot and not accession:
            accession = uniprot

        if not gene and not accession:
            return result
        if not Path(self._file_path).exists():
            result.errors.append(f"UniProt file not found: {self._file_path}")
            return result

        try:
            row_data = self._search(gene=gene, accession=accession)
            if not row_data:
                return result

            acc = row_data.get("Entry", "")
            protein_name = row_data.get("Protein names", "")
            gene_names = row_data.get("Gene Names", "")
            primary_gene = gene_names.split()[0] if gene_names else gene
            function = row_data.get("Function [CC]", "")
            location = row_data.get("Subcellular location [CC]", "")
            ptm = row_data.get("Post-translational modification", "")
            disease = row_data.get("Involvement in disease", "")
            domains = row_data.get("Domain [FT]", "")
            binding = row_data.get("Binding site", "")
            modified_residue = row_data.get("Modified residue", "")
            pathway = row_data.get("Pathway", "")
            interacts = row_data.get("Interacts with", "")
            length = row_data.get("Length", "")

            # 1. Function evidence (always produced)
            func_parts = [f"{primary_gene} ({acc}): {protein_name[:120]}."]
            if function:
                func_parts.append(f"Function: {function[:300]}")
            if location:
                func_parts.append(f"Localization: {location[:200]}")
            if pathway:
                func_parts.append(f"Pathway: {pathway[:200]}")
            if interacts:
                func_parts.append(f"Interacts with: {interacts[:150]}")

            self._upsert_item(
                result,
                item_id=f"evi:uniprot:{acc.lower()}_function",
                claim=" ".join(func_parts),
                body_extra={
                    "protein_name": protein_name[:200],
                    "gene": primary_gene,
                    "accession": acc,
                    "length": length,
                    "function": function[:500],
                    "subcellular_location": location[:300],
                    "pathway": pathway[:300],
                    "interacts_with": interacts[:300],
                    "pch_layer": 1,
                },
            )

            # 2. PTM evidence (only if PTM or modified residue data exists)
            ptm_text = ptm or modified_residue
            if ptm_text:
                ptm_parts = [
                    f"{primary_gene} ({acc}) post-translational modifications:"
                ]
                if ptm:
                    ptm_parts.append(f"PTM: {ptm[:300]}")
                if modified_residue:
                    ptm_parts.append(f"Modified residues: {modified_residue[:300]}")

                self._upsert_item(
                    result,
                    item_id=f"evi:uniprot:{acc.lower()}_ptm",
                    claim=" ".join(ptm_parts),
                    body_extra={
                        "gene": primary_gene,
                        "accession": acc,
                        "ptm": ptm[:500],
                        "modified_residue": modified_residue[:500],
                        "pch_layer": 2,
                    },
                )

            # 3. Disease evidence (only if disease data exists)
            if disease:
                self._upsert_item(
                    result,
                    item_id=f"evi:uniprot:{acc.lower()}_disease",
                    claim=(
                        f"{primary_gene} ({acc}) disease associations: "
                        f"{disease[:400]}"
                    ),
                    body_extra={
                        "gene": primary_gene,
                        "accession": acc,
                        "disease": disease[:500],
                        "pch_layer": 1,
                    },
                )

            # 4. Structure evidence (only if domain or binding site data exists)
            structure_text = domains or binding
            if structure_text:
                struct_parts = [
                    f"{primary_gene} ({acc}) structural features:"
                ]
                if domains:
                    struct_parts.append(f"Domains: {domains[:300]}")
                if binding:
                    struct_parts.append(f"Binding sites: {binding[:300]}")

                self._upsert_item(
                    result,
                    item_id=f"evi:uniprot:{acc.lower()}_structure",
                    claim=" ".join(struct_parts),
                    body_extra={
                        "gene": primary_gene,
                        "accession": acc,
                        "domains": domains[:500],
                        "binding_sites": binding[:500],
                        "pch_layer": 1,
                    },
                )

        except Exception as e:
            result.errors.append(f"UniProt query failed for gene={gene} acc={accession}: {e}")

        return result

    def _upsert_item(
        self,
        result: ConnectorResult,
        *,
        item_id: str,
        claim: str,
        body_extra: dict,
    ) -> None:
        """Create and upsert a single evidence envelope. Failures are isolated."""
        body = {
            "claim": claim,
            "source": "uniprot_swissprot",
            "evidence_strength": "strong",
            "data_source": "uniprot",
            **body_extra,
        }
        evi = BaseEnvelope(
            id=item_id,
            type="EvidenceItem",
            status="active",
            body=body,
        )
        if self._store:
            try:
                self._store.upsert_object(evi)
            except Exception:
                pass
        result.evidence_items_added += 1

    def _search(self, *, gene: str = "", accession: str = "") -> dict | None:
        """Search the UniProt TSV by gene symbol or accession.

        Gene matching: case-insensitive match on the first token in
        'Gene Names' column. Accession matching: exact match on 'Entry'.
        """
        gene_upper = gene.upper() if gene else ""
        acc_upper = accession.upper() if accession else ""

        with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                # Match by accession
                if acc_upper and row.get("Entry", "").upper() == acc_upper:
                    return row
                # Match by gene symbol (first token in Gene Names)
                if gene_upper:
                    gene_names = row.get("Gene Names", "")
                    first_token = gene_names.split()[0].upper() if gene_names else ""
                    if first_token == gene_upper:
                        return row

        return None
