"""Human Protein Atlas connector — protein localization and expression in tissues.

Queries the local HPA proteinatlas.tsv for protein-level data on ALS targets.
Answers: "Where is this protein expressed? What cell types? What is its
subcellular localization?" — validates drug targets at the protein level.

File: /Volumes/Databank/databases/hpa/proteinatlas.tsv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_DEFAULT_PATH = "/Volumes/Databank/databases/hpa/proteinatlas.tsv"


class HPAConnector(BaseConnector):
    """Query Human Protein Atlas for protein expression in tissues."""

    def __init__(self, store: Any = None, file_path: str = _DEFAULT_PATH):
        self._store = store
        self._file_path = file_path

    def fetch(self, *, gene: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()

        if not gene:
            return result
        if not Path(self._file_path).exists():
            result.errors.append(f"HPA file not found: {self._file_path}")
            return result

        try:
            row = self._search_gene(gene)
            if not row:
                return result

            # Extract key fields
            protein_class = row.get("Protein class", "")
            bio_process = row.get("Biological process", "")
            mol_function = row.get("Molecular function", "")
            disease = row.get("Disease involvement", "")
            evidence = row.get("Evidence", "")
            subcellular = row.get("Subcellular location", "")
            chromosome = row.get("Chromosome", "")
            uniprot = row.get("Uniprot", "")

            claim = f"Human Protein Atlas: {gene}"
            if protein_class:
                claim += f" is classified as {protein_class}."
            if bio_process:
                claim += f" Biological process: {bio_process[:80]}."
            if disease:
                claim += f" Disease involvement: {disease[:80]}."
            if subcellular:
                claim += f" Subcellular location: {subcellular[:60]}."
            if evidence:
                claim += f" Evidence level: {evidence}."

            evi = BaseEnvelope(
                id=f"evi:hpa_{gene.lower()}_protein_atlas",
                type="EvidenceItem",
                status="active",
                body={
                    "claim": claim,
                    "source": "human_protein_atlas",
                    "gene": gene,
                    "uniprot": uniprot,
                    "protein_class": protein_class,
                    "biological_process": bio_process,
                    "molecular_function": mol_function,
                    "disease_involvement": disease,
                    "subcellular_location": subcellular,
                    "evidence_level": evidence,
                    "evidence_strength": "strong",
                    "pch_layer": 1,
                    "protocol_layer": "root_cause_suppression",
                },
            )

            if self._store:
                self._store.upsert_object(evi)
                result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"HPA query failed for {gene}: {e}")

        return result

    def _search_gene(self, gene: str) -> dict | None:
        """Search proteinatlas.tsv for a gene by symbol."""
        gene_upper = gene.upper()

        with open(self._file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                if (row.get("Gene") or "").upper() == gene_upper:
                    return row

        return None
