"""DrugBank local connector — drug vocabulary and target links.

Uses DrugBank's CC0 open data files (vocabulary + target-UniProt links) for
drug identification and target mapping. Enables drug repurposing by finding
all drugs that target ALS-relevant proteins.

Files: /Volumes/Databank/databases/drugbank/drugbank_vocabulary.csv
       /Volumes/Databank/databases/drugbank/drugbank_target_links.csv
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_VOCAB_PATH = "/Volumes/Databank/databases/drugbank/drugbank_vocabulary.csv"
_TARGET_LINKS_PATH = "/Volumes/Databank/databases/drugbank/drugbank_target_links.csv"

# UniProt IDs for Erik's 16 ALS targets
_ALS_TARGET_UNIPROTS = {
    "Q13148": "TARDBP", "P00441": "SOD1", "P35637": "FUS",
    "Q96LT7": "C9orf72", "Q93045": "STMN2", "Q99720": "SIGMAR1",
    "P43004": "SLC1A2", "P42345": "MTOR", "P07333": "CSF1R",
    "Q9UHD2": "TBK1", "P01031": "C5", "P23560": "BDNF",
    "P39905": "GDNF", "Q96CV9": "OPTN", "Q96PY6": "NEK1",
}


class DrugBankLocalConnector(BaseConnector):
    """Query DrugBank open data for drug-target relationships."""

    def __init__(self, store: Any = None):
        self._store = store
        self._vocab: dict[str, dict] = {}  # drugbank_id -> {name, cas, ...}
        self._target_links: dict[str, list[str]] = {}  # uniprot -> [drugbank_ids]
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return

        # Load vocabulary
        if Path(_VOCAB_PATH).exists():
            with open(_VOCAB_PATH, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dbid = row.get("DrugBank ID", "")
                    if dbid:
                        self._vocab[dbid] = {
                            "name": row.get("Common name", ""),
                            "cas": row.get("CAS", ""),
                            "unii": row.get("UNII", ""),
                            "synonyms": row.get("Synonyms", ""),
                        }

        # Load target links (UniProt -> DrugBank IDs)
        if Path(_TARGET_LINKS_PATH).exists():
            with open(_TARGET_LINKS_PATH, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    uniprot = row.get("UniProt ID", "")
                    dbid = row.get("DrugBank ID", "")
                    if uniprot and dbid:
                        if uniprot not in self._target_links:
                            self._target_links[uniprot] = []
                        self._target_links[uniprot].append(dbid)

        self._loaded = True

    def fetch(self, *, gene: str = "", uniprot: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        self._load()

        if not self._vocab:
            result.errors.append("DrugBank vocabulary not loaded")
            return result

        # Find UniProt for this gene from ALS targets
        if not uniprot and gene:
            for uid, gname in _ALS_TARGET_UNIPROTS.items():
                if gname.upper() == gene.upper():
                    uniprot = uid
                    break

        if not uniprot:
            return result

        # Find all drugs targeting this protein
        drug_ids = self._target_links.get(uniprot, [])
        if not drug_ids:
            return result

        gene_name = _ALS_TARGET_UNIPROTS.get(uniprot, gene)
        drug_names = []
        for dbid in drug_ids:
            info = self._vocab.get(dbid, {})
            name = info.get("name", dbid)
            if name:
                drug_names.append(name)

        claim = (
            f"DrugBank: {len(drug_ids)} drugs target {gene_name} ({uniprot}): "
            f"{', '.join(drug_names[:10])}"
            + (f" (+{len(drug_names) - 10} more)" if len(drug_names) > 10 else "")
            + ". These are drug repurposing candidates for this ALS target."
        )

        evi = BaseEnvelope(
            id=f"evi:drugbank_{gene_name.lower()}_targets",
            type="EvidenceItem",
            status="active",
            body={
                "claim": claim,
                "source": "drugbank",
                "gene": gene_name,
                "uniprot": uniprot,
                "n_drugs": len(drug_ids),
                "drugs": drug_names[:20],
                "drugbank_ids": drug_ids[:20],
                "evidence_strength": "strong",
                "pch_layer": 2,
                "protocol_layer": "root_cause_suppression",
            },
        )

        if self._store:
            self._store.upsert_object(evi)
            result.evidence_items_added += 1

        return result
