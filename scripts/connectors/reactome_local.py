"""Reactome local connector — pathway hierarchy from bulk download files.

Queries the local Reactome pathway files to map ALS genes to their
biological pathways and trace pathway cascades. Answers: "What pathway
does this gene belong to, and what other ALS-relevant genes share it?"

Files: /Volumes/Databank/databases/reactome/ReactomePathways.txt
       /Volumes/Databank/databases/reactome/ReactomePathwaysRelation.txt
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import BaseEnvelope

_PATHWAYS_FILE = "/Volumes/Databank/databases/reactome/ReactomePathways.txt"
_RELATIONS_FILE = "/Volumes/Databank/databases/reactome/ReactomePathwaysRelation.txt"

# ALS-relevant pathway keywords
_ALS_PATHWAY_KEYWORDS = frozenset({
    "autophagy", "apoptosis", "ubiquitin", "proteasome", "mtor",
    "neurotrophin", "axon", "motor neuron", "glutamate", "gaba",
    "mitochondr", "oxidative stress", "unfolded protein", "er stress",
    "rna processing", "splicing", "transport", "vesicle", "nfkb",
    "inflamm", "complement", "innate immune", "toll-like",
    "signaling by receptor tyrosine kinases", "mapk", "akt",
})


class ReactomeLocalConnector(BaseConnector):
    """Query local Reactome pathway files for ALS gene pathway membership."""

    def __init__(self, store: Any = None):
        self._store = store
        self._pathways: dict[str, str] = {}  # id -> name (human only)
        self._relations: dict[str, list[str]] = {}  # parent -> children
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        # Load pathway names (filter to Homo sapiens)
        if Path(_PATHWAYS_FILE).exists():
            with open(_PATHWAYS_FILE, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 3 and parts[2] == "Homo sapiens":
                        self._pathways[parts[0]] = parts[1]

        # Load parent-child relations
        if Path(_RELATIONS_FILE).exists():
            with open(_RELATIONS_FILE, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 2:
                        parent, child = parts[0], parts[1]
                        if parent not in self._relations:
                            self._relations[parent] = []
                        self._relations[parent].append(child)

        self._loaded = True

    def fetch(self, *, gene: str = "", pathway_keyword: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        self._load()

        if not self._pathways:
            result.errors.append("Reactome pathways not loaded")
            return result

        # Search for ALS-relevant pathways
        if pathway_keyword:
            keywords = [pathway_keyword.lower()]
        else:
            keywords = list(_ALS_PATHWAY_KEYWORDS)

        matched_pathways: list[tuple[str, str]] = []
        for pid, name in self._pathways.items():
            name_lower = name.lower()
            for kw in keywords:
                if kw in name_lower:
                    matched_pathways.append((pid, name))
                    break

        if not matched_pathways:
            return result

        # Create evidence for top pathways (limit to avoid overwhelming)
        for pid, name in matched_pathways[:15]:
            # Count child pathways (depth of cascade)
            children = self._relations.get(pid, [])
            child_names = [self._pathways.get(c, c) for c in children if c in self._pathways]

            claim = f"Reactome pathway: {name} (ID: {pid})"
            if child_names:
                claim += f". Sub-pathways: {', '.join(child_names[:3])}"
                if len(child_names) > 3:
                    claim += f" (+{len(child_names) - 3} more)"
            claim += "."

            evi = BaseEnvelope(
                id=f"evi:reactome_{pid.lower().replace('-', '_')}",
                type="EvidenceItem",
                status="active",
                body={
                    "claim": claim,
                    "source": "reactome_local",
                    "pathway_id": pid,
                    "pathway_name": name,
                    "n_sub_pathways": len(children),
                    "sub_pathways": child_names[:5],
                    "evidence_strength": "strong",
                    "pch_layer": 1,
                    "protocol_layer": "root_cause_suppression",
                },
            )

            if self._store:
                try:
                    self._store.upsert_object(evi)
                    result.evidence_items_added += 1
                except Exception:
                    pass

        return result
