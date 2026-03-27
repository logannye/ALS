# scripts/connectors/clinvar.py
"""ClinVar connector — genetic variant pathogenicity via NCBI E-utilities."""
from __future__ import annotations
import xml.etree.ElementTree as ET
import requests
from typing import Optional
from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

def _parse_variant_xml(xml_str: str, gene: str) -> list[dict]:
    if not xml_str or not xml_str.strip():
        return []
    results = []
    try:
        root = ET.fromstring(xml_str)
        for doc_sum in root.findall(".//DocumentSummary"):
            variation_id = doc_sum.get("uid", "")
            title = doc_sum.findtext("title", "")
            clinical_sig = doc_sum.findtext("clinical_significance/description", "")
            review_status = doc_sum.findtext("clinical_significance/review_status", "")
            if gene and title and gene.lower() not in title.lower():
                continue
            results.append({"variation_id": variation_id, "variant_name": title, "gene": gene,
                          "clinical_significance": clinical_sig, "review_status": review_status})
    except ET.ParseError:
        pass
    return results

class ClinVarConnector(BaseConnector):
    TOOL = "erik_als"
    EMAIL = "research@galenhealth.ai"

    def __init__(self, evidence_store=None, api_key: Optional[str] = None):
        self._store = evidence_store
        self._api_key = api_key

    def fetch(self, *, gene: str = "", max_results: int = 20, **kwargs) -> ConnectorResult:
        result = ConnectorResult()
        if not gene:
            result.errors.append("gene symbol required")
            return result
        try:
            params = {"db": "clinvar", "retmode": "json", "retmax": max_results,
                      "term": f"{gene}[gene] AND ALS[disease]", "tool": self.TOOL, "email": self.EMAIL}
            if self._api_key:
                params["api_key"] = self._api_key
            resp = self._retry_with_backoff(requests.get, f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            id_list = resp.json().get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return result
            s_params = {"db": "clinvar", "retmode": "xml", "id": ",".join(id_list), "tool": self.TOOL, "email": self.EMAIL}
            if self._api_key:
                s_params["api_key"] = self._api_key
            s_resp = self._retry_with_backoff(requests.get, f"{EUTILS_BASE}/esummary.fcgi", params=s_params, timeout=self.REQUEST_TIMEOUT)
            s_resp.raise_for_status()
            variants = _parse_variant_xml(s_resp.text, gene)
            for var in variants:
                item = self._build_evidence_item(**var)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"ClinVar error: {e}")
        return result

    def _build_evidence_item(self, variation_id, variant_name, gene, clinical_significance, review_status):
        sig_lower = clinical_significance.lower()
        if "pathogenic" in sig_lower and "uncertain" not in sig_lower:
            strength, direction = EvidenceStrength.strong, EvidenceDirection.supports
        elif "benign" in sig_lower:
            strength, direction = EvidenceStrength.strong, EvidenceDirection.refutes
        else:
            strength, direction = EvidenceStrength.emerging, EvidenceDirection.insufficient
        return EvidenceItem(
            id=f"evi:clinvar:{variation_id}",
            claim=f"ClinVar: {variant_name} — {clinical_significance}",
            direction=direction, strength=strength, source_refs=[f"clinvar:{variation_id}"],
            provenance=Provenance(source_system=SourceSystem.database, asserted_by="clinvar_connector", source_artifact_id=str(variation_id)),
            body={"variation_id": variation_id, "variant_name": variant_name, "gene": gene,
                  "clinical_significance": clinical_significance, "review_status": review_status, "pch_layer": 1, "data_source": "clinvar"},
        )
