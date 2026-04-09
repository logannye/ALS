"""PubMedConnector — fetches literature evidence from NCBI E-utilities.

Uses ESearch to find PMIDs and EFetch to retrieve article XML.
Parses PubmedArticle elements into EvidenceItem objects and upserts to DB.
"""
from __future__ import annotations

import logging
import time
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import (
    EvidenceDirection,
    EvidenceStrength,
    ProtocolLayer,
    SourceSystem,
)
from ontology.evidence import EvidenceItem

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Publication type → modality mapping
# ---------------------------------------------------------------------------

PUB_TYPE_MODALITY: dict[str, str] = {
    "Randomized Controlled Trial": "randomized_controlled_trial",
    "Clinical Trial": "clinical_trial",
    "Clinical Trial, Phase I": "clinical_trial",
    "Clinical Trial, Phase II": "clinical_trial",
    "Clinical Trial, Phase III": "clinical_trial",
    "Clinical Trial, Phase IV": "clinical_trial",
    "Meta-Analysis": "meta_analysis",
    "Systematic Review": "systematic_review",
    "Review": "review",
    "Case Reports": "case_report",
    "Observational Study": "observational_study",
    "Comparative Study": "comparative_study",
    "Multicenter Study": "multicenter_study",
    "Letter": "letter",
    "Editorial": "editorial",
    "Comment": "comment",
}

# ---------------------------------------------------------------------------
# Modality → evidence strength mapping
# ---------------------------------------------------------------------------

_MODALITY_TO_STRENGTH: dict[str, str] = {
    "randomized_controlled_trial": "strong",
    "meta_analysis": "strong",
    "systematic_review": "strong",
    "clinical_trial": "moderate",
    "multicenter_study": "moderate",
    "comparative_study": "moderate",
    "observational_study": "emerging",
    "review": "emerging",
    "case_report": "preclinical",
    "letter": "preclinical",
    "editorial": "preclinical",
    "comment": "preclinical",
}


def _infer_strength_from_modality(modality: str) -> str:
    """Map publication modality to evidence strength tier."""
    return _MODALITY_TO_STRENGTH.get(modality, "unknown")


# ---------------------------------------------------------------------------
# Free function: parse a single PubmedArticle XML element
# ---------------------------------------------------------------------------

def _parse_pubmed_article(article_el: ET.Element) -> EvidenceItem:
    """Parse a ``<PubmedArticle>`` XML element into an EvidenceItem.

    Parameters
    ----------
    article_el:
        An ``xml.etree.ElementTree.Element`` with tag ``PubmedArticle``.

    Returns
    -------
    EvidenceItem with fields populated from the article metadata.
    """
    citation = article_el.find("MedlineCitation")
    pmid = citation.findtext("PMID", default="").strip()
    article = citation.find("Article")

    # Title
    title = article.findtext("ArticleTitle", default="").strip()

    # Journal
    journal_el = article.find("Journal")
    journal = journal_el.findtext("Title", default="").strip() if journal_el is not None else ""

    # Abstract (first 500 chars)
    abstract_el = article.find("Abstract")
    abstract = ""
    if abstract_el is not None:
        parts = []
        for at in abstract_el.findall("AbstractText"):
            if at.text:
                parts.append(at.text.strip())
        abstract = " ".join(parts)[:500]

    # Publication type → modality
    modality = "other"
    pub_type_list = article.find("PublicationTypeList")
    if pub_type_list is not None:
        for pt in pub_type_list.findall("PublicationType"):
            pt_text = (pt.text or "").strip()
            if pt_text in PUB_TYPE_MODALITY:
                modality = PUB_TYPE_MODALITY[pt_text]
                break

    # Infer evidence strength from publication modality
    inferred_strength = _infer_strength_from_modality(modality)
    strength_enum = getattr(EvidenceStrength, inferred_strength, EvidenceStrength.unknown)

    return EvidenceItem(
        id=f"evi:pubmed:{pmid}",
        claim=title,
        direction=EvidenceDirection.insufficient,
        strength=strength_enum,
        source_refs=[f"pmid:{pmid}"],
        provenance=Provenance(
            source_system=SourceSystem.literature,
            asserted_by="pubmed_connector",
            source_artifact_id=pmid,
        ),
        body={
            "protocol_layer": "",
            "mechanism_target": "",
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "pch_layer": 1,
            "modality": modality,
            "evidence_strength": inferred_strength,
            "abstract": abstract,
            "journal": journal,
        },
    )


# ---------------------------------------------------------------------------
# PubMedConnector
# ---------------------------------------------------------------------------

class PubMedConnector(BaseConnector):
    """Connector for NCBI PubMed E-utilities API.

    Fetches literature evidence, parses XML results, and upserts
    EvidenceItem objects into the Erik evidence store.
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    TOOL = "erik_als_engine"
    EMAIL = "logan@galenhealth.ai"
    RATE_LIMIT_SECONDS = 0.35  # default without API key

    # Curated queries per protocol layer
    LAYER_QUERIES: dict[ProtocolLayer, str] = {
        ProtocolLayer.root_cause_suppression: (
            "(ALS OR amyotrophic lateral sclerosis) AND "
            "(TDP-43 OR SOD1 OR C9orf72 OR FUS) AND (therapy OR treatment)"
        ),
        ProtocolLayer.pathology_reversal: (
            "(ALS OR amyotrophic lateral sclerosis) AND "
            "(neuroinflammation OR excitotoxicity OR oxidative stress) AND "
            "(reversal OR neuroprotection)"
        ),
        ProtocolLayer.circuit_stabilization: (
            "(ALS OR amyotrophic lateral sclerosis) AND "
            "(motor neuron OR neuromuscular junction) AND "
            "(stabilization OR preservation)"
        ),
        ProtocolLayer.regeneration_reinnervation: (
            "(ALS OR amyotrophic lateral sclerosis) AND "
            "(regeneration OR reinnervation OR stem cell OR axonal growth)"
        ),
        ProtocolLayer.adaptive_maintenance: (
            "(ALS OR amyotrophic lateral sclerosis) AND "
            "(riluzole OR edaravone OR supportive care OR ventilation OR nutrition)"
        ),
    }

    def __init__(self, *, api_key: Optional[str] = None, store=None):
        self._api_key = api_key
        self._store = store
        if api_key:
            self.RATE_LIMIT_SECONDS = 0.11

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch(self, *, query: str = "", max_results: int = 50, **kwargs) -> ConnectorResult:
        """ESearch for PMIDs → EFetch XML → parse into EvidenceItems → upsert."""
        result = ConnectorResult()
        try:
            pmids = self._esearch(query, max_results)
        except Exception as e:
            result.errors.append(f"ESearch failed: {e}")
            return result
        if not pmids:
            return result
        return self._fetch_and_upsert(pmids, result)

    def fetch_by_pmids(self, pmids: list[str]) -> ConnectorResult:
        """Direct EFetch by PMIDs (no search), parse, upsert."""
        result = ConnectorResult()
        if not pmids:
            return result
        return self._fetch_and_upsert(pmids, result)

    def fetch_als_treatment_updates(self) -> ConnectorResult:
        """Run 5 curated queries (one per protocol layer), max 20 each."""
        combined = ConnectorResult()
        for layer, query in self.LAYER_QUERIES.items():
            layer_result = self.fetch(query=query, max_results=20)
            combined.evidence_items_added += layer_result.evidence_items_added
            combined.skipped_duplicates += layer_result.skipped_duplicates
            combined.errors.extend(layer_result.errors)
        return combined

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _base_params(self) -> dict:
        """Return common E-utilities query parameters."""
        params = {"tool": self.TOOL, "email": self.EMAIL}
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Call ESearch and return a list of PMIDs."""
        url = f"{self.BASE_URL}/esearch.fcgi"
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
        }
        resp = self._retry_with_backoff(
            requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("esearchresult", {}).get("idlist", [])

    def _efetch(self, pmids: list[str]) -> list[ET.Element]:
        """Call EFetch for a list of PMIDs and return PubmedArticle elements."""
        url = f"{self.BASE_URL}/efetch.fcgi"
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
        }
        resp = self._retry_with_backoff(
            requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        return root.findall("PubmedArticle")

    def _fetch_and_upsert(self, pmids: list[str], result: ConnectorResult) -> ConnectorResult:
        """EFetch PMIDs, parse articles, and upsert to store."""
        time.sleep(self.RATE_LIMIT_SECONDS)
        try:
            articles = self._efetch(pmids)
        except Exception as e:
            result.errors.append(f"EFetch failed: {e}")
            return result

        for article_el in articles:
            try:
                item = _parse_pubmed_article(article_el)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                pmid_el = article_el.find(".//PMID")
                pmid = pmid_el.text if pmid_el is not None else "unknown"
                result.errors.append(f"Parse/upsert failed for PMID {pmid}: {e}")

        return result
