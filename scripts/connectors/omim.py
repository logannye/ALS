# scripts/connectors/omim.py
"""OMIM connector — gene-phenotype mapping for ALS subtype refinement."""
from __future__ import annotations
import requests
from typing import Optional
from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

BASE_URL = "https://api.omim.org/api"

ALS_MIM_NUMBERS = {"SOD1": 105400, "TARDBP": 612069, "FUS": 608030, "C9orf72": 614260, "OPTN": 613435, "TBK1": 616795, "NEK1": 617892}

def _parse_entry_response(raw: dict) -> list[dict]:
    entries = []
    for ew in raw.get("omim", {}).get("entryList", []):
        entry = ew.get("entry", {})
        mim = entry.get("mimNumber", 0)
        title = entry.get("titles", {}).get("preferredTitle", "")
        for pmw in entry.get("geneMap", {}).get("phenotypeMapList", []):
            pm = pmw.get("phenotypeMap", {})
            entries.append({"mim_number": mim, "title": title, "gene_symbols": pm.get("geneSymbols", ""), "inheritance": pm.get("phenotypeInheritance", ""), "phenotype": pm.get("phenotype", "")})
        if not entry.get("geneMap", {}).get("phenotypeMapList"):
            entries.append({"mim_number": mim, "title": title, "gene_symbols": "", "inheritance": "", "phenotype": ""})
    return entries

class OMIMConnector(BaseConnector):
    def __init__(self, evidence_store=None, api_key: Optional[str] = None):
        self._store = evidence_store
        self._api_key = api_key

    def fetch(self, *, gene: str = "", mim_number: Optional[int] = None, **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        mim = mim_number or ALS_MIM_NUMBERS.get(gene)
        if not mim:
            result.errors.append(f"No known OMIM MIM number for gene: {gene}")
            return result
        if not self._api_key:
            result.errors.append("OMIM API key required (set omim_api_key in config)")
            return result
        try:
            params = {"mimNumber": mim, "include": "geneMap", "format": "json", "apiKey": self._api_key}
            resp = self._retry_with_backoff(requests.get, f"{BASE_URL}/entry", params=params, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            for entry in _parse_entry_response(resp.json()):
                item = self._build_evidence_item(**entry)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"OMIM error: {e}")
        return result

    def _build_evidence_item(self, mim_number, title, gene_symbols, inheritance, phenotype):
        return EvidenceItem(
            id=f"evi:omim:{mim_number}",
            claim=f"OMIM: {title} — {phenotype} ({inheritance})",
            direction=EvidenceDirection.supports, strength=EvidenceStrength.strong,
            source_refs=[f"omim:{mim_number}"],
            provenance=Provenance(source_system=SourceSystem.database, asserted_by="omim_connector", source_artifact_id=str(mim_number)),
            body={"mim_number": mim_number, "title": title, "gene_symbols": gene_symbols, "inheritance": inheritance, "phenotype": phenotype, "pch_layer": 1, "data_source": "omim"},
        )
