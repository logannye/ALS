# scripts/connectors/pharmgkb.py
"""PharmGKB connector — pharmacogenomics and drug safety."""
from __future__ import annotations
import requests
from typing import Optional
from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

BASE_URL = "https://api.pharmgkb.org/v1"

def _parse_drug_response(raw: dict) -> dict:
    data = raw.get("data", {})
    return {"pharmgkb_id": data.get("id", ""), "name": data.get("name", ""), "generic_names": data.get("genericNames", [])}

def _parse_clinical_annotations(raw: dict) -> list[dict]:
    results = []
    for ann in raw.get("data", []):
        gene_info = ann.get("gene", {}) or {}
        drug_info = ann.get("drug", {}) or {}
        cat_info = ann.get("phenotypeCategory", {}) or {}
        results.append({"annotation_id": ann.get("id", ""), "gene": gene_info.get("symbol", ""), "drug": drug_info.get("name", ""), "category": cat_info.get("term", ""), "level": ann.get("level", ""), "summary": ann.get("summary", "")})
    return results

class PharmGKBConnector(BaseConnector):
    def __init__(self, evidence_store=None, base_url: str = BASE_URL):
        self._store = evidence_store
        self._base_url = base_url

    def fetch(self, *, drug_name: str = "", gene: str = "", **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        if drug_name:
            return self._fetch_drug_annotations(drug_name, result)
        elif gene:
            return self._fetch_gene_annotations(gene, result)
        result.errors.append("drug_name or gene required")
        return result

    def _fetch_drug_annotations(self, drug_name, result):
        try:
            resp = self._retry_with_backoff(requests.get, f"{self._base_url}/data/drug", params={"name": drug_name}, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            drug_info = _parse_drug_response(resp.json())
            if drug_info["pharmgkb_id"]:
                ann_resp = self._retry_with_backoff(requests.get, f"{self._base_url}/data/clinicalAnnotation", params={"location.drugs.name": drug_name}, timeout=self.REQUEST_TIMEOUT)
                ann_resp.raise_for_status()
                for ann in _parse_clinical_annotations(ann_resp.json()):
                    item = self._build_evidence_item(drug_info["pharmgkb_id"], drug_name, ann["gene"], ann["summary"], ann["level"], ann["category"])
                    if self._store:
                        self._store.upsert_evidence_item(item)
                    result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"PharmGKB error: {e}")
        return result

    def _fetch_gene_annotations(self, gene, result):
        try:
            resp = self._retry_with_backoff(requests.get, f"{self._base_url}/data/clinicalAnnotation", params={"location.genes.symbol": gene}, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            for ann in _parse_clinical_annotations(resp.json()):
                item = self._build_evidence_item(f"gene_{gene}", ann["drug"], gene, ann["summary"], ann["level"], ann["category"])
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"PharmGKB error: {e}")
        return result

    def _build_evidence_item(self, pharmgkb_id, drug_name, gene, annotation, level, category):
        if level in ("1A", "1B"):
            strength = EvidenceStrength.strong
        elif level in ("2A", "2B"):
            strength = EvidenceStrength.moderate
        else:
            strength = EvidenceStrength.emerging
        return EvidenceItem(
            id=f"evi:pharmgkb:{pharmgkb_id}_{gene}",
            claim=f"PharmGKB: {drug_name} - {gene} ({category}) — Level {level}",
            direction=EvidenceDirection.supports, strength=strength,
            source_refs=[f"pharmgkb:{pharmgkb_id}"],
            provenance=Provenance(source_system=SourceSystem.database, asserted_by="pharmgkb_connector", source_artifact_id=pharmgkb_id),
            body={"pharmgkb_id": pharmgkb_id, "drug_name": drug_name, "gene": gene, "annotation": annotation, "level": level, "category": category, "pch_layer": 2, "data_source": "pharmgkb"},
        )
