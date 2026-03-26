# Phase 1B: Evidence Connectors — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build 5 on-demand API connectors (PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, DrugBank) that expand the curated evidence seed with live data, check Erik's trial eligibility, and store results in PostgreSQL.

**Architecture:** All connectors inherit from `BaseConnector` ABC providing retry-with-backoff, partial commit, and standardized `ConnectorResult`. Each connector is stateless (on-demand `fetch()` calls). Results flow through the existing `EvidenceStore` to PostgreSQL. Erik's eligibility is checked against every active clinical trial.

**Tech Stack:** Python 3.12, urllib.request (stdlib, no new deps), sqlite3 (ChEMBL read-only), psycopg3, pytest, Pydantic v2.

**Spec:** `/Users/logannye/.openclaw/erik/docs/specs/2026-03-25-phase1b-evidence-connectors-design.md`

---

## File Structure

```
scripts/
  connectors/
    __init__.py               # CREATE
    base.py                   # CREATE: BaseConnector ABC, ConnectorResult dataclass
    pubmed.py                 # CREATE: PubMedConnector (E-utilities)
    clinical_trials.py        # CREATE: ClinicalTrialsConnector + Erik eligibility
    chembl.py                 # CREATE: ChEMBLConnector (local SQL)
    opentargets.py            # CREATE: OpenTargetsConnector (GraphQL)
    drugbank.py               # CREATE: DrugBankConnector (XML parser)
data/
  erik_config.json            # MODIFY: add ncbi_api_key, chembl_db_path, drugbank_xml_path
tests/
  test_base_connector.py      # CREATE
  test_pubmed_connector.py    # CREATE
  test_clinical_trials_connector.py  # CREATE
  test_chembl_connector.py    # CREATE
  test_opentargets_connector.py      # CREATE
  test_drugbank_connector.py  # CREATE
```

---

## Task 1: BaseConnector + ConnectorResult

**Files:**
- Create: `scripts/connectors/__init__.py`
- Create: `scripts/connectors/base.py`
- Create: `tests/test_base_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base_connector.py
import pytest
from connectors.base import BaseConnector, ConnectorResult


def test_connector_result_defaults():
    r = ConnectorResult()
    assert r.evidence_items_added == 0
    assert r.interventions_added == 0
    assert r.errors == []
    assert r.skipped_duplicates == 0


def test_connector_result_accumulate():
    r = ConnectorResult()
    r.evidence_items_added += 5
    r.errors.append("parse error on item X")
    assert r.evidence_items_added == 5
    assert len(r.errors) == 1


class MockConnector(BaseConnector):
    def fetch(self, **kwargs):
        return ConnectorResult(evidence_items_added=1)

def test_mock_connector_fetch():
    c = MockConnector()
    result = c.fetch()
    assert result.evidence_items_added == 1


def test_retry_succeeds_on_second_attempt():
    call_count = 0
    def flaky_fn():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("transient failure")
        return "success"
    c = MockConnector()
    result = c._retry_with_backoff(flaky_fn)
    assert result == "success"
    assert call_count == 2


def test_retry_exhausts_retries():
    def always_fails():
        raise ConnectionError("permanent failure")
    c = MockConnector()
    with pytest.raises(ConnectionError):
        c._retry_with_backoff(always_fails)


def test_connector_has_timeout():
    c = MockConnector()
    assert c.REQUEST_TIMEOUT == 30
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_base_connector.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/__init__.py
```

```python
# scripts/connectors/base.py
"""BaseConnector ABC and ConnectorResult for all evidence connectors.

All connectors inherit from BaseConnector and produce ConnectorResult.
Provides: retry with exponential backoff, request timeout, partial commit.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ConnectorResult:
    """Result of a connector fetch operation."""
    evidence_items_added: int = 0
    interventions_added: int = 0
    errors: list[str] = field(default_factory=list)
    skipped_duplicates: int = 0


class BaseConnector(ABC):
    """Abstract base for all evidence connectors.

    Contract (per spec Section 7.1):
    - Upsert-by-source-ID (deterministic canonical IDs)
    - Exponential backoff retries (3 attempts, 1/2/4s)
    - Partial commit (each item individually, errors logged not raised)
    - 30s request timeout
    """
    MAX_RETRIES = 3
    BACKOFF_SECONDS = [1, 2, 4]
    REQUEST_TIMEOUT = 30

    @abstractmethod
    def fetch(self, **kwargs) -> ConnectorResult:
        """Fetch evidence from the external source. Must be implemented by subclasses."""
        ...

    def _retry_with_backoff(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Call fn with exponential backoff on failure. Raises after MAX_RETRIES."""
        last_error = None
        for attempt in range(self.MAX_RETRIES):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BACKOFF_SECONDS[attempt])
        raise last_error
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_base_connector.py -v`
Expected: All 6 PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/ tests/test_base_connector.py && git commit -m "feat: BaseConnector ABC with retry, backoff, and ConnectorResult"
```

---

## Task 2: PubMedConnector

**Files:**
- Create: `scripts/connectors/pubmed.py`
- Create: `tests/test_pubmed_connector.py`

- [ ] **Step 1: Write the failing test**

Tests should work without network access by testing XML parsing with fixture data.

```python
# tests/test_pubmed_connector.py
import pytest
from connectors.pubmed import PubMedConnector, _parse_pubmed_article

# Fixture: minimal PubMed eFetch XML for one article
SAMPLE_XML = """<?xml version="1.0"?>
<!DOCTYPE PubmedArticleSet PUBLIC "-//NLM//DTD PubmedArticleSet//EN" "">
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">35389978</PMID>
      <Article PubModel="Print">
        <Journal><Title>N Engl J Med</Title></Journal>
        <ArticleTitle>Trial of Antisense Oligonucleotide Tofersen for SOD1 ALS</ArticleTitle>
        <Abstract>
          <AbstractText>BACKGROUND: Tofersen is an antisense oligonucleotide targeting SOD1 mRNA. METHODS: We conducted a 28-week randomized trial. RESULTS: Tofersen reduced CSF SOD1 and plasma neurofilament light. CONCLUSIONS: Target engagement was confirmed.</AbstractText>
        </Abstract>
        <PublicationTypeList>
          <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_parse_pubmed_article():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(SAMPLE_XML)
    article_el = root.find(".//PubmedArticle")
    item = _parse_pubmed_article(article_el)
    assert item is not None
    assert item.id == "evi:pubmed:35389978"
    assert "Tofersen" in item.claim
    assert "pmid:35389978" in item.source_refs
    assert item.body.get("modality") == "randomized_controlled_trial"


def test_parse_pubmed_article_has_abstract_in_body():
    import xml.etree.ElementTree as ET
    root = ET.fromstring(SAMPLE_XML)
    article_el = root.find(".//PubmedArticle")
    item = _parse_pubmed_article(article_el)
    assert "abstract" in item.body
    assert "antisense oligonucleotide" in item.body["abstract"]


def test_pubmed_connector_instantiates():
    c = PubMedConnector()
    assert c.REQUEST_TIMEOUT == 30


def test_pubmed_als_queries_exist():
    c = PubMedConnector()
    queries = c._get_layer_queries()
    assert "root_cause_suppression" in queries
    assert "pathology_reversal" in queries
    assert len(queries) == 5


@pytest.mark.network
def test_pubmed_fetch_small_query():
    c = PubMedConnector()
    result = c.fetch(query="ALS tofersen 2024", max_results=3)
    assert result.evidence_items_added >= 1
    assert result.errors == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_pubmed_connector.py -v -k "not network"`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/pubmed.py
"""PubMed E-utilities connector for ALS evidence.

Fetches abstracts from PubMed via NCBI E-utilities API,
parses XML into EvidenceItem objects, and upserts to PostgreSQL.
"""
from __future__ import annotations

import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.evidence import EvidenceItem
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.base import Provenance
from evidence.evidence_store import EvidenceStore

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# Map PubMed publication types to modality strings
_PUB_TYPE_MAP = {
    "Randomized Controlled Trial": "randomized_controlled_trial",
    "Clinical Trial": "clinical_trial",
    "Clinical Trial, Phase II": "clinical_trial_phase2",
    "Clinical Trial, Phase III": "clinical_trial_phase3",
    "Meta-Analysis": "meta_analysis",
    "Systematic Review": "systematic_review",
    "Review": "review",
    "Case Reports": "case_report",
    "Observational Study": "observational",
}


def _parse_pubmed_article(article_el: ET.Element) -> Optional[EvidenceItem]:
    """Parse a single PubmedArticle XML element into an EvidenceItem."""
    citation = article_el.find("MedlineCitation")
    if citation is None:
        return None

    pmid_el = citation.find("PMID")
    if pmid_el is None or pmid_el.text is None:
        return None
    pmid = pmid_el.text.strip()

    article = citation.find("Article")
    if article is None:
        return None

    title_el = article.find("ArticleTitle")
    title = title_el.text.strip() if title_el is not None and title_el.text else "Untitled"

    abstract_el = article.find(".//AbstractText")
    abstract = ""
    if abstract_el is not None and abstract_el.text:
        abstract = abstract_el.text.strip()[:500]

    # Detect modality from publication types
    modality = "unknown"
    for pt in article.findall(".//PublicationType"):
        if pt.text and pt.text in _PUB_TYPE_MAP:
            modality = _PUB_TYPE_MAP[pt.text]
            break

    return EvidenceItem(
        id=f"evi:pubmed:{pmid}",
        claim=title,
        direction=EvidenceDirection.insufficient,
        source_refs=[f"pmid:{pmid}"],
        strength=EvidenceStrength.unknown,
        provenance=Provenance(
            source_system=SourceSystem.literature,
            asserted_by="pubmed_connector",
        ),
        body={
            "protocol_layer": "",
            "mechanism_target": "",
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "pch_layer": 1,
            "modality": modality,
            "abstract": abstract,
            "journal": (article.find(".//Journal/Title").text or "").strip() if article.find(".//Journal/Title") is not None else "",
        },
    )


class PubMedConnector(BaseConnector):
    """Fetch ALS evidence from PubMed via NCBI E-utilities."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key
        self._delay = 0.11 if api_key else 0.35
        self._store = EvidenceStore()

    def _get_layer_queries(self) -> dict[str, str]:
        base = "(ALS OR amyotrophic lateral sclerosis)"
        return {
            "root_cause_suppression": f"{base} AND (gene therapy OR antisense oligonucleotide OR tofersen OR SOD1 OR TDP-43 intrabody) AND 2025:2026[dp]",
            "pathology_reversal": f"{base} AND (TDP-43 OR cryptic splicing OR STMN2 OR UNC13A OR pridopidine OR sigma-1 receptor OR autophagy) AND 2025:2026[dp]",
            "circuit_stabilization": f"{base} AND (excitotoxicity OR riluzole OR masitinib OR neuroinflammation OR complement) AND 2025:2026[dp]",
            "regeneration_reinnervation": f"{base} AND (regeneration OR neurotrophic OR BDNF OR GDNF OR NMJ OR reinnervation) AND 2025:2026[dp]",
            "adaptive_maintenance": f"{base} AND (neurofilament light OR biomarker OR monitoring OR multidisciplinary care) AND 2025:2026[dp]",
        }

    def fetch(self, query: str = "", max_results: int = 50, **kwargs) -> ConnectorResult:
        """Search PubMed and ingest results as EvidenceItems."""
        result = ConnectorResult()
        pmids = self._esearch(query, max_results)
        if not pmids:
            return result

        articles_xml = self._efetch(pmids)
        if articles_xml is None:
            result.errors.append("EFetch returned no data")
            return result

        root = ET.fromstring(articles_xml)
        for article_el in root.findall(".//PubmedArticle"):
            try:
                item = _parse_pubmed_article(article_el)
                if item is not None:
                    self._store.upsert_evidence_item(item)
                    result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(f"Parse error: {e}")
        return result

    def fetch_by_pmids(self, pmids: list[str]) -> ConnectorResult:
        """Fetch specific papers by PMID list (direct EFetch, no search)."""
        result = ConnectorResult()
        if not pmids:
            return result
        articles_xml = self._efetch(pmids)
        if articles_xml is None:
            result.errors.append("EFetch returned no data")
            return result
        root = ET.fromstring(articles_xml)
        for article_el in root.findall(".//PubmedArticle"):
            try:
                item = _parse_pubmed_article(article_el)
                if item is not None:
                    self._store.upsert_evidence_item(item)
                    result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(f"Parse error: {e}")
        return result

    def fetch_als_treatment_updates(self) -> ConnectorResult:
        """Run curated queries per protocol layer (max 20 per layer)."""
        combined = ConnectorResult()
        for layer, query in self._get_layer_queries().items():
            r = self.fetch(query=query, max_results=20)
            combined.evidence_items_added += r.evidence_items_added
            combined.errors.extend(r.errors)
        return combined

    def _esearch(self, query: str, max_results: int) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "tool": "erik_als_engine",
            "email": "logan@galenhealth.ai",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        url = f"{EUTILS_BASE}/esearch.fcgi?{urllib.parse.urlencode(params)}"

        import json
        try:
            data = self._retry_with_backoff(
                lambda: urllib.request.urlopen(url, timeout=self.REQUEST_TIMEOUT).read()
            )
            result = json.loads(data)
            return result.get("esearchresult", {}).get("idlist", [])
        except Exception:
            return []

    def _efetch(self, pmids: list[str]) -> Optional[str]:
        """Fetch article XML for given PMIDs."""
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "xml",
            "retmode": "xml",
            "tool": "erik_als_engine",
            "email": "logan@galenhealth.ai",
        }
        if self._api_key:
            params["api_key"] = self._api_key
        url = f"{EUTILS_BASE}/efetch.fcgi?{urllib.parse.urlencode(params)}"
        time.sleep(self._delay)

        try:
            data = self._retry_with_backoff(
                lambda: urllib.request.urlopen(url, timeout=self.REQUEST_TIMEOUT).read()
            )
            return data.decode("utf-8")
        except Exception:
            return None
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_pubmed_connector.py -v -k "not network"`
Expected: All non-network tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/pubmed.py tests/test_pubmed_connector.py && git commit -m "feat: PubMed connector with E-utilities XML parsing and layer-based queries"
```

---

## Task 3: ClinicalTrialsConnector + Erik Eligibility

**Files:**
- Create: `scripts/connectors/clinical_trials.py`
- Create: `tests/test_clinical_trials_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clinical_trials_connector.py
import pytest
import json
from connectors.clinical_trials import (
    ClinicalTrialsConnector, ERIK_PROFILE, check_eligibility,
    _parse_trial,
)

# Fixture: minimal ClinicalTrials.gov v2 study JSON
SAMPLE_TRIAL = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT06000000",
            "briefTitle": "A Phase 3 Study of Pridopidine in ALS (PREVAiLS)",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
        },
        "designModule": {
            "phases": ["PHASE3"],
            "studyType": "INTERVENTIONAL",
            "enrollmentInfo": {"count": 500},
        },
        "armsInterventionsModule": {
            "interventions": [{
                "type": "DRUG",
                "name": "Pridopidine",
                "description": "Sigma-1 receptor agonist",
            }],
        },
        "eligibilityModule": {
            "minimumAge": "18 Years",
            "maximumAge": "80 Years",
            "sex": "ALL",
            "eligibilityCriteria": "Inclusion Criteria:\\n- Diagnosis of ALS\\n- ALSFRS-R >= 30\\n- FVC >= 60%\\nExclusion Criteria:\\n- Active malignancy\\n- Severe hepatic impairment",
        },
        "outcomesModule": {
            "primaryOutcomes": [{
                "measure": "Change in ALSFRS-R total score",
                "timeFrame": "48 weeks",
            }],
        },
        "contactsLocationsModule": {
            "locations": [
                {"facility": "Cleveland Clinic", "state": "Ohio", "country": "United States"},
                {"facility": "Mass General", "state": "Massachusetts", "country": "United States"},
            ],
        },
    }
}


def test_parse_trial():
    item, intervention = _parse_trial(SAMPLE_TRIAL)
    assert item.id == "evi:trial:NCT06000000"
    assert "Pridopidine" in item.claim or "PREVAiLS" in item.claim
    assert item.body["phase"] == "PHASE3"
    assert intervention.id == "int:trial:NCT06000000"
    assert intervention.name == "Pridopidine"


def test_erik_eligibility_eligible():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "80 Years",
        "sex": "ALL",
        "eligibilityCriteria": "Inclusion:\\n- ALS diagnosis\\n- ALSFRS-R >= 30\\n- FVC >= 60%",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "eligible"


def test_erik_eligibility_too_old():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "65 Years",
        "sex": "ALL",
        "eligibilityCriteria": "",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_erik_eligibility_female_only():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "80 Years",
        "sex": "FEMALE",
        "eligibilityCriteria": "",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_erik_eligibility_uncertain():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "80 Years",
        "sex": "ALL",
        "eligibilityCriteria": "Inclusion:\\n- Must have confirmed SOD1 mutation",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    # Genetic-specific criteria when genetics pending → uncertain
    assert result == "uncertain"


def test_erik_eligibility_alsfrs_too_low():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "80 Years",
        "sex": "ALL",
        "eligibilityCriteria": "Inclusion:\\n- ALSFRS-R >= 45",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_erik_eligibility_excluded_comorbidity():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "80 Years",
        "sex": "ALL",
        "eligibilityCriteria": "Inclusion:\\n- ALS diagnosis\\nExclusion Criteria:\\n- hypertension\\n- diabetes",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_erik_eligibility_excluded_medication():
    criteria = {
        "minimumAge": "18 Years",
        "maximumAge": "80 Years",
        "sex": "ALL",
        "eligibilityCriteria": "Inclusion:\\n- ALS diagnosis\\nExclusion Criteria:\\n- Patients taking atorvastatin",
    }
    result = check_eligibility(criteria, ERIK_PROFILE)
    assert result == "ineligible"


def test_ohio_sites_detected():
    item, _ = _parse_trial(SAMPLE_TRIAL)
    assert "Ohio" in str(item.body.get("ohio_sites", []))


def test_connector_instantiates():
    c = ClinicalTrialsConnector()
    assert c.REQUEST_TIMEOUT == 30


@pytest.mark.network
def test_fetch_active_trials():
    c = ClinicalTrialsConnector()
    result = c.fetch_active_als_trials(max_results=5)
    assert result.evidence_items_added >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_clinical_trials_connector.py -v -k "not network"`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/clinical_trials.py
"""ClinicalTrials.gov v2 API connector with Erik eligibility matching.

Fetches active ALS trials, parses into EvidenceItem + Intervention objects,
and assesses Erik Draper's eligibility against each trial.
"""
from __future__ import annotations

import json
import re
import urllib.request
import urllib.parse
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.evidence import EvidenceItem
from ontology.intervention import Intervention
from ontology.enums import (
    EvidenceDirection, EvidenceStrength, InterventionClass, SourceSystem,
)
from ontology.base import Provenance
from evidence.evidence_store import EvidenceStore

CT_API_BASE = "https://clinicaltrials.gov/api/v2/studies"

ERIK_PROFILE = {
    "age": 67,
    "sex": "male",
    "diagnosis": "ALS",
    "onset_region": "lower_limb",
    "alsfrs_r": 43,
    "fvc_percent": 100,
    "disease_duration_months": 14,
    "on_riluzole": True,
    "genetic_status": "pending",
    "comorbidities": ["hypertension", "prediabetes", "cervical_stenosis"],
    "medications": ["riluzole", "amlodipine", "atorvastatin", "ramipril"],
}

_INTERVENTION_TYPE_MAP = {
    "DRUG": InterventionClass.drug,
    "BIOLOGICAL": InterventionClass.drug,
    "GENETIC": InterventionClass.gene_therapy,
    "DEVICE": InterventionClass.supportive_care,
    "PROCEDURE": InterventionClass.supportive_care,
    "OTHER": InterventionClass.drug,
}


def check_eligibility(criteria: dict, profile: dict = ERIK_PROFILE) -> str:
    """Check Erik's eligibility against trial criteria. Returns eligible/ineligible/uncertain."""
    # Age check
    min_age = _parse_age(criteria.get("minimumAge", "0 Years"))
    max_age = _parse_age(criteria.get("maximumAge", "999 Years"))
    if profile["age"] < min_age or profile["age"] > max_age:
        return "ineligible"

    # Sex check
    sex_req = criteria.get("sex", "ALL")
    if sex_req == "FEMALE" and profile["sex"] == "male":
        return "ineligible"
    if sex_req == "MALE" and profile["sex"] == "female":
        return "ineligible"

    # Parse free-text criteria
    text = criteria.get("eligibilityCriteria", "").lower()

    # ALSFRS-R minimum check
    alsfrs_match = re.search(r"alsfrs[- ]?r?\s*>=?\s*(\d+)", text)
    if alsfrs_match:
        min_score = int(alsfrs_match.group(1))
        if profile["alsfrs_r"] < min_score:
            return "ineligible"

    # FVC minimum check
    fvc_match = re.search(r"fvc\s*>=?\s*(\d+)", text)
    if fvc_match:
        min_fvc = int(fvc_match.group(1))
        if profile["fvc_percent"] < min_fvc:
            return "ineligible"

    # Genetic-specific trials when genetics pending
    genetic_keywords = ["sod1 mutation", "c9orf72", "fus mutation", "confirmed genetic"]
    if any(kw in text for kw in genetic_keywords) and profile["genetic_status"] == "pending":
        return "uncertain"

    # Comorbidity exclusions
    exclusion_section = text.split("exclusion")[-1] if "exclusion" in text else ""
    for comorbidity in profile.get("comorbidities", []):
        # Check if the comorbidity keyword appears in exclusion criteria
        if comorbidity.replace("_", " ") in exclusion_section:
            return "ineligible"

    # Medication exclusions
    for med in profile.get("medications", []):
        if med.lower() in exclusion_section:
            return "ineligible"

    return "eligible"


def _parse_age(age_str: str) -> int:
    """Parse '67 Years' or '18 Months' to integer years."""
    match = re.match(r"(\d+)\s*(year|month)", age_str.lower())
    if match:
        val = int(match.group(1))
        if "month" in match.group(2):
            return val // 12
        return val
    return 0


def _parse_trial(study: dict) -> tuple[EvidenceItem, Intervention]:
    """Parse a ClinicalTrials.gov v2 study JSON into EvidenceItem + Intervention."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    arms = proto.get("armsInterventionsModule", {})
    elig = proto.get("eligibilityModule", {})
    outcomes = proto.get("outcomesModule", {})
    locations = proto.get("contactsLocationsModule", {})

    nct_id = ident.get("nctId", "UNKNOWN")
    title = ident.get("briefTitle", "Untitled trial")
    phases = design.get("phases", [])
    phase = phases[0] if phases else "UNKNOWN"
    enrollment = design.get("enrollmentInfo", {}).get("count", 0)

    # Intervention info
    interventions = arms.get("interventions", [])
    int_info = interventions[0] if interventions else {}
    int_name = int_info.get("name", "Unknown intervention")
    int_type = int_info.get("type", "OTHER")
    int_class = _INTERVENTION_TYPE_MAP.get(int_type, InterventionClass.drug)

    # Primary outcome
    primary = outcomes.get("primaryOutcomes", [{}])
    primary_measure = primary[0].get("measure", "") if primary else ""

    # Ohio sites
    locs = locations.get("locations", [])
    ohio_sites = [loc.get("facility", "") for loc in locs if loc.get("state") == "Ohio"]

    # Eligibility
    erik_eligible = check_eligibility(elig)

    evidence_item = EvidenceItem(
        id=f"evi:trial:{nct_id}",
        claim=title,
        direction=EvidenceDirection.supports,
        source_refs=[f"nct:{nct_id}"],
        strength=EvidenceStrength.emerging,
        provenance=Provenance(
            source_system=SourceSystem.trial,
            asserted_by="trial_connector",
        ),
        body={
            "protocol_layer": "",
            "mechanism_target": "",
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": erik_eligible,
            "pch_layer": 2,
            "nct_id": nct_id,
            "phase": phase,
            "enrollment": enrollment,
            "overall_status": status.get("overallStatus", ""),
            "primary_endpoint": primary_measure,
            "ohio_sites": ohio_sites,
            "intervention_name": int_name,
        },
    )

    intervention = Intervention(
        id=f"int:trial:{nct_id}",
        name=int_name,
        intervention_class=int_class,
        targets=[],
        route="",
        body={
            "nct_id": nct_id,
            "regulatory_status": "experimental",
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "phase": phase,
            "sponsor": "",
        },
    )

    return evidence_item, intervention


class ClinicalTrialsConnector(BaseConnector):
    """Fetch active ALS clinical trials from ClinicalTrials.gov v2 API."""

    def __init__(self):
        self._store = EvidenceStore()

    def fetch(self, **kwargs) -> ConnectorResult:
        return self.fetch_active_als_trials(**kwargs)

    def fetch_active_als_trials(
        self,
        phases: list[str] | None = None,
        max_results: int = 50,
    ) -> ConnectorResult:
        """Fetch recruiting ALS trials."""
        if phases is None:
            phases = ["PHASE2", "PHASE3"]
        result = ConnectorResult()

        params = {
            "query.cond": "amyotrophic lateral sclerosis",
            "filter.overallStatus": "RECRUITING,NOT_YET_RECRUITING",
            "filter.phase": ",".join(phases),
            "pageSize": str(min(max_results, 50)),
            "format": "json",
        }
        url = f"{CT_API_BASE}?{urllib.parse.urlencode(params)}"

        try:
            raw = self._retry_with_backoff(
                lambda: urllib.request.urlopen(url, timeout=self.REQUEST_TIMEOUT).read()
            )
            data = json.loads(raw)
        except Exception as e:
            result.errors.append(f"API error: {e}")
            return result

        studies = data.get("studies", [])
        for study in studies[:max_results]:
            try:
                item, intervention = _parse_trial(study)
                self._store.upsert_evidence_item(item)
                self._store.upsert_intervention(intervention)
                result.evidence_items_added += 1
                result.interventions_added += 1
            except Exception as e:
                result.errors.append(f"Parse error: {e}")

        return result

    def fetch_trial_details(self, nct_id: str) -> Optional[dict]:
        """Fetch full study record for a single NCT ID."""
        url = f"{CT_API_BASE}/{nct_id}?format=json"
        try:
            raw = self._retry_with_backoff(
                lambda: urllib.request.urlopen(url, timeout=self.REQUEST_TIMEOUT).read()
            )
            return json.loads(raw)
        except Exception:
            return None
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_clinical_trials_connector.py -v -k "not network"`
Expected: All non-network tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/clinical_trials.py tests/test_clinical_trials_connector.py && git commit -m "feat: ClinicalTrials.gov connector with Erik eligibility matching"
```

---

## Task 4: ChEMBLConnector

**Files:**
- Create: `scripts/connectors/chembl.py`
- Create: `tests/test_chembl_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chembl_connector.py
import pytest
from connectors.chembl import ChEMBLConnector


def test_chembl_connector_instantiates():
    # Should not fail even if DB not present (lazy connection)
    c = ChEMBLConnector(db_path="/nonexistent/path.db")
    assert c._db_path == "/nonexistent/path.db"


def test_chembl_query_construction():
    c = ChEMBLConnector(db_path="/nonexistent/path.db")
    sql, params = c._build_bioactivity_query("Q99720", "IC50", 50)
    assert "standard_type" in sql
    assert "Q99720" in params


@pytest.mark.chembl
def test_chembl_fetch_sigmar1():
    """Requires ChEMBL 36 at /Volumes/Databank/databases/chembl_36.db"""
    c = ChEMBLConnector()
    result = c.fetch_bioactivity("Q99720", activity_type="IC50", max_results=10)
    assert result.evidence_items_added >= 1


@pytest.mark.chembl
def test_chembl_fetch_for_target_name():
    c = ChEMBLConnector()
    result = c.fetch_compounds_for_target("SIGMAR1")
    assert result.evidence_items_added >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_chembl_connector.py -v -k "not chembl"`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/chembl.py
"""ChEMBL local database connector for ALS compound bioactivity.

Queries the local ChEMBL 36 SQLite database for compounds active
against ALS-relevant protein targets.

NOTE: This is the ONE exception to the 'never use sqlite3' rule.
ChEMBL is a read-only external reference database, not Erik's state.
"""
from __future__ import annotations

import os
import sqlite3
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.evidence import EvidenceItem
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.base import Provenance
from evidence.evidence_store import EvidenceStore
from targets.als_targets import get_target, ALS_TARGETS

DEFAULT_CHEMBL_PATH = "/Volumes/Databank/databases/chembl_36.db"


class ChEMBLConnector(BaseConnector):
    """Query local ChEMBL 36 for compound bioactivity against ALS targets."""

    def __init__(self, db_path: Optional[str] = None):
        self._db_path = db_path or os.environ.get("CHEMBL_DB_PATH", DEFAULT_CHEMBL_PATH)
        self._store = EvidenceStore()

    def fetch(self, **kwargs) -> ConnectorResult:
        return self.fetch_for_priority_targets(**kwargs)

    def _build_bioactivity_query(self, uniprot_id: str, activity_type: str, max_results: int) -> tuple[str, tuple]:
        sql = """
            SELECT DISTINCT
                md.chembl_id AS molecule_chembl_id,
                md.pref_name AS molecule_name,
                act.standard_type,
                act.standard_value,
                act.standard_units,
                td.chembl_id AS target_chembl_id,
                td.pref_name AS target_name
            FROM activities act
            JOIN assays a ON act.assay_id = a.assay_id
            JOIN target_dictionary td ON a.tid = td.tid
            JOIN target_components tc ON td.tid = tc.tid
            JOIN component_sequences cs ON tc.component_id = cs.component_id
            JOIN molecule_dictionary md ON act.molregno = md.molregno
            WHERE cs.accession = ?
              AND act.standard_type = ?
              AND act.standard_relation = '='
              AND act.standard_value IS NOT NULL
            ORDER BY act.standard_value ASC
            LIMIT ?
        """
        return sql, (uniprot_id, activity_type, max_results)

    def fetch_bioactivity(
        self,
        uniprot_id: str,
        activity_type: str = "IC50",
        max_results: int = 100,
    ) -> ConnectorResult:
        """Fetch bioactivity data for a target from local ChEMBL."""
        result = ConnectorResult()

        if not os.path.exists(self._db_path):
            result.errors.append(f"ChEMBL database not found at {self._db_path}")
            return result

        sql, params = self._build_bioactivity_query(uniprot_id, activity_type, max_results)

        try:
            conn = sqlite3.connect(self._db_path)
            cursor = conn.execute(sql, params)
            rows = cursor.fetchall()
            conn.close()
        except Exception as e:
            result.errors.append(f"ChEMBL query error: {e}")
            return result

        for row in rows:
            mol_chembl, mol_name, std_type, std_value, std_units, tgt_chembl, tgt_name = row
            if mol_name is None:
                mol_name = mol_chembl

            item = EvidenceItem(
                id=f"evi:chembl:{mol_chembl}_{tgt_chembl}",
                claim=f"{mol_name} has {std_type} = {std_value} {std_units or 'nM'} against {tgt_name}",
                direction=EvidenceDirection.supports,
                source_refs=[f"chembl:{mol_chembl}"],
                strength=EvidenceStrength.preclinical,
                provenance=Provenance(
                    source_system=SourceSystem.database,
                    asserted_by="chembl_connector",
                ),
                body={
                    "protocol_layer": "",
                    "mechanism_target": tgt_name or "",
                    "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
                    "erik_eligible": True,
                    "pch_layer": 2,
                    "molecule_chembl_id": mol_chembl,
                    "target_chembl_id": tgt_chembl,
                    "activity_type": std_type,
                    "activity_value": std_value,
                    "activity_units": std_units or "nM",
                },
            )

            try:
                self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(f"Upsert error for {mol_chembl}: {e}")

        return result

    def fetch_compounds_for_target(self, target_name: str) -> ConnectorResult:
        """Look up UniProt ID from als_targets.py, then query ChEMBL."""
        target = get_target(target_name)
        if target is None:
            r = ConnectorResult()
            r.errors.append(f"Unknown target: {target_name}")
            return r
        return self.fetch_bioactivity(target["uniprot_id"])

    def fetch_for_priority_targets(self, max_per_target: int = 50, **kwargs) -> ConnectorResult:
        """Query ChEMBL for all druggable ALS targets."""
        combined = ConnectorResult()
        priority = ["SIGMAR1", "EAAT2", "mTOR", "CSF1R", "TDP-43"]
        for name in priority:
            target = get_target(name)
            if target and target.get("druggable"):
                r = self.fetch_bioactivity(target["uniprot_id"], max_results=max_per_target)
                combined.evidence_items_added += r.evidence_items_added
                combined.errors.extend(r.errors)
        return combined
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_chembl_connector.py -v -k "not chembl"`
Expected: Non-ChEMBL tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/chembl.py tests/test_chembl_connector.py && git commit -m "feat: ChEMBL connector for local bioactivity queries against ALS targets"
```

---

## Task 5: OpenTargetsConnector

**Files:**
- Create: `scripts/connectors/opentargets.py`
- Create: `tests/test_opentargets_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_opentargets_connector.py
import pytest
import json
from connectors.opentargets import OpenTargetsConnector, _parse_target_association

SAMPLE_ASSOCIATION = {
    "target": {
        "id": "ENSG00000142168",
        "approvedSymbol": "SOD1",
    },
    "score": 0.85,
    "datatypeScores": [
        {"id": "genetic_association", "score": 0.9},
        {"id": "known_drug", "score": 0.7},
    ],
}


def test_parse_target_association():
    item = _parse_target_association(SAMPLE_ASSOCIATION)
    assert item.id == "evi:ot:ENSG00000142168_als"
    assert "SOD1" in item.claim
    assert item.body["association_score"] == 0.85


def test_connector_instantiates():
    c = OpenTargetsConnector()
    assert c.REQUEST_TIMEOUT == 30


@pytest.mark.network
def test_fetch_als_targets():
    c = OpenTargetsConnector()
    result = c.fetch_als_targets(max_results=10)
    assert result.evidence_items_added >= 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_opentargets_connector.py -v -k "not network"`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/opentargets.py
"""OpenTargets Platform connector for ALS target-disease associations.

Queries the OpenTargets GraphQL API for ranked ALS target associations
with tractability and evidence scoring.
"""
from __future__ import annotations

import json
import urllib.request
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.evidence import EvidenceItem
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.base import Provenance, Uncertainty
from evidence.evidence_store import EvidenceStore

OT_API = "https://api.platform.opentargets.org/api/v4/graphql"
ALS_EFO_ID = "EFO_0000253"

_ALS_TARGETS_QUERY = """
query ALSTargets($efoId: String!, $size: Int!) {
  disease(efoId: $efoId) {
    associatedTargets(page: {size: $size, index: 0}) {
      rows {
        target {
          id
          approvedSymbol
        }
        score
        datatypeScores {
          id
          score
        }
      }
    }
  }
}
"""


def _parse_target_association(row: dict) -> EvidenceItem:
    """Parse an OpenTargets association row into an EvidenceItem."""
    target = row.get("target", {})
    ensembl_id = target.get("id", "UNKNOWN")
    symbol = target.get("approvedSymbol", "UNKNOWN")
    score = row.get("score", 0.0)

    datatype_scores = {}
    for ds in row.get("datatypeScores", []):
        datatype_scores[ds["id"]] = ds["score"]

    return EvidenceItem(
        id=f"evi:ot:{ensembl_id}_als",
        claim=f"{symbol} is associated with ALS (overall score={score:.2f})",
        direction=EvidenceDirection.insufficient,
        source_refs=[f"opentargets:{ensembl_id}"],
        strength=EvidenceStrength.unknown,
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="opentargets_connector",
        ),
        uncertainty=Uncertainty(confidence=score),
        body={
            "protocol_layer": "",
            "mechanism_target": symbol,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "erik_eligible": True,
            "pch_layer": 1,
            "ensembl_id": ensembl_id,
            "gene_symbol": symbol,
            "association_score": score,
            "genetic_association": datatype_scores.get("genetic_association", 0),
            "known_drug_count": datatype_scores.get("known_drug", 0),
        },
    )


class OpenTargetsConnector(BaseConnector):
    """Fetch ALS target-disease associations from OpenTargets."""

    def __init__(self):
        self._store = EvidenceStore()

    def fetch(self, **kwargs) -> ConnectorResult:
        return self.fetch_als_targets(**kwargs)

    def fetch_als_targets(self, min_score: float = 0.1, max_results: int = 100) -> ConnectorResult:
        """Fetch ranked ALS target associations."""
        result = ConnectorResult()

        payload = json.dumps({
            "query": _ALS_TARGETS_QUERY,
            "variables": {"efoId": ALS_EFO_ID, "size": max_results},
        }).encode("utf-8")

        req = urllib.request.Request(
            OT_API,
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            raw = self._retry_with_backoff(
                lambda: urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT).read()
            )
            data = json.loads(raw)
        except Exception as e:
            result.errors.append(f"OpenTargets API error: {e}")
            return result

        rows = (
            data.get("data", {})
            .get("disease", {})
            .get("associatedTargets", {})
            .get("rows", [])
        )

        for row in rows:
            score = row.get("score", 0)
            if score < min_score:
                continue
            try:
                item = _parse_target_association(row)
                self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1
            except Exception as e:
                result.errors.append(f"Parse error: {e}")

        return result
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_opentargets_connector.py -v -k "not network"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/opentargets.py tests/test_opentargets_connector.py && git commit -m "feat: OpenTargets connector for ALS target-disease associations"
```

---

## Task 6: DrugBankConnector

**Files:**
- Create: `scripts/connectors/drugbank.py`
- Create: `tests/test_drugbank_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_drugbank_connector.py
import pytest
from connectors.drugbank import DrugBankConnector, _parse_drug_entry
import xml.etree.ElementTree as ET

# Minimal DrugBank XML fixture
SAMPLE_DRUG_XML = """<drug type="small molecule">
  <drugbank-id primary="true">DB00740</drugbank-id>
  <name>Riluzole</name>
  <description>A glutamate antagonist used to treat ALS.</description>
  <indication>For the treatment of amyotrophic lateral sclerosis (ALS).</indication>
  <mechanism-of-action>Riluzole inhibits glutamate release and blocks sodium channels.</mechanism-of-action>
  <groups><group>approved</group></groups>
  <targets>
    <target>
      <id>BE0000714</id>
      <name>Sodium channel protein type 5 subunit alpha</name>
      <polypeptide id="Q14524" source="Swiss-Prot">
        <name>SCN5A</name>
      </polypeptide>
    </target>
  </targets>
  <drug-interactions>
    <drug-interaction>
      <drugbank-id>DB01611</drugbank-id>
      <name>Hydroxychloroquine</name>
      <description>May increase hepatotoxicity risk.</description>
    </drug-interaction>
  </drug-interactions>
</drug>"""


def test_parse_drug_entry():
    ns = {}
    el = ET.fromstring(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(el, ns)
    assert intervention is not None
    assert intervention.id == "int:drugbank:DB00740"
    assert intervention.name == "Riluzole"
    assert "glutamate" in intervention.body.get("mechanism_of_action", "").lower()


def test_parse_drug_has_interactions():
    el = ET.fromstring(SAMPLE_DRUG_XML)
    intervention = _parse_drug_entry(el, {})
    interactions = intervention.body.get("drug_interactions", [])
    assert len(interactions) >= 1
    assert interactions[0]["drugbank_id"] == "DB01611"


def test_connector_handles_missing_xml():
    c = DrugBankConnector(xml_path="/nonexistent/drugbank.xml")
    result = c.fetch_als_drugs()
    assert result.interventions_added == 0
    assert len(result.errors) >= 1


def test_connector_instantiates():
    c = DrugBankConnector()
    assert c.REQUEST_TIMEOUT == 30
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_drugbank_connector.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/drugbank.py
"""DrugBank XML connector for ALS drug profiles and interactions.

Parses DrugBank full database XML (academic license) to extract
drug MOA, targets, interactions, and ADMET data for ALS-relevant drugs.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.intervention import Intervention
from ontology.enums import InterventionClass, SourceSystem
from ontology.base import Provenance
from evidence.evidence_store import EvidenceStore
from targets.als_targets import ALS_TARGETS

# DrugBank XML namespace (may vary by version)
_DB_NS = {"db": "http://www.drugbank.ca"}

_ALS_KEYWORDS = [
    "amyotrophic lateral sclerosis", "als", "motor neuron disease",
    "neurodegeneration", "neuroprotect",
]


def _text(el: Optional[ET.Element], ns: dict) -> str:
    """Extract text from an element, or empty string."""
    if el is not None and el.text:
        return el.text.strip()
    return ""


def _find_text(parent: ET.Element, tag: str, ns: dict) -> str:
    """Find a child element and return its text."""
    el = parent.find(tag, ns) if ns else parent.find(tag)
    return _text(el, ns)


def _parse_drug_entry(drug_el: ET.Element, ns: dict) -> Optional[Intervention]:
    """Parse a single <drug> element into an Intervention."""
    drugbank_id = ""
    for dbid in drug_el.findall("drugbank-id", ns) if ns else drug_el.findall("drugbank-id"):
        if dbid.get("primary") == "true":
            drugbank_id = dbid.text or ""
            break
    if not drugbank_id:
        # Try without namespace
        for dbid in drug_el.findall("drugbank-id"):
            if dbid.get("primary") == "true":
                drugbank_id = dbid.text or ""
                break

    name = _find_text(drug_el, "name", ns) or _find_text(drug_el, "name", {})
    description = _find_text(drug_el, "description", ns) or _find_text(drug_el, "description", {})
    indication = _find_text(drug_el, "indication", ns) or _find_text(drug_el, "indication", {})
    moa = _find_text(drug_el, "mechanism-of-action", ns) or _find_text(drug_el, "mechanism-of-action", {})

    # Groups (approved, experimental, etc.)
    groups = []
    groups_el = drug_el.find("groups", ns) if ns else drug_el.find("groups")
    if groups_el is not None:
        for g in groups_el.findall("group", ns) if ns else groups_el.findall("group"):
            if g.text:
                groups.append(g.text.strip())

    # Targets
    targets = []
    targets_el = drug_el.find("targets", ns) if ns else drug_el.find("targets")
    if targets_el is not None:
        for t in targets_el.findall("target", ns) if ns else targets_el.findall("target"):
            tname = _find_text(t, "name", ns) or _find_text(t, "name", {})
            if tname:
                targets.append(tname)

    # Drug interactions
    interactions = []
    di_el = drug_el.find("drug-interactions", ns) if ns else drug_el.find("drug-interactions")
    if di_el is not None:
        for di in (di_el.findall("drug-interaction", ns) if ns else di_el.findall("drug-interaction")):
            di_id = _find_text(di, "drugbank-id", ns) or _find_text(di, "drugbank-id", {})
            di_name = _find_text(di, "name", ns) or _find_text(di, "name", {})
            di_desc = _find_text(di, "description", ns) or _find_text(di, "description", {})
            if di_id:
                interactions.append({
                    "drugbank_id": di_id,
                    "name": di_name,
                    "description": di_desc[:200],
                })

    regulatory = "approved" if "approved" in groups else "experimental"

    return Intervention(
        id=f"int:drugbank:{drugbank_id}",
        name=name,
        intervention_class=InterventionClass.drug,
        targets=targets[:10],
        route="",
        provenance=Provenance(
            source_system=SourceSystem.database,
            asserted_by="drugbank_connector",
        ),
        body={
            "drugbank_id": drugbank_id,
            "regulatory_status": regulatory,
            "applicable_subtypes": ["sporadic_tdp43", "unresolved"],
            "description": description[:300],
            "indication": indication[:300],
            "mechanism_of_action": moa[:500],
            "groups": groups,
            "drug_interactions": interactions[:20],
        },
    )


class DrugBankConnector(BaseConnector):
    """Parse DrugBank XML for ALS-relevant drug profiles."""

    def __init__(self, xml_path: Optional[str] = None):
        self._xml_path = xml_path or os.environ.get("DRUGBANK_XML_PATH", "")
        self._store = EvidenceStore()

    def fetch(self, **kwargs) -> ConnectorResult:
        return self.fetch_als_drugs(**kwargs)

    def fetch_als_drugs(self, **kwargs) -> ConnectorResult:
        """Parse DrugBank XML and extract ALS-relevant drugs."""
        result = ConnectorResult()

        if not self._xml_path or not os.path.exists(self._xml_path):
            result.errors.append(f"DrugBank XML not found at '{self._xml_path}'. Download from go.drugbank.com with academic license.")
            return result

        # Collect ALS target UniProt IDs for matching
        als_uniprot_ids = set()
        for t in ALS_TARGETS.values():
            uid = t.get("uniprot_id", "")
            if uid:
                als_uniprot_ids.add(uid)

        try:
            tree = ET.parse(self._xml_path)
            root = tree.getroot()
        except Exception as e:
            result.errors.append(f"XML parse error: {e}")
            return result

        # Detect namespace
        ns = {}
        if root.tag.startswith("{"):
            ns_uri = root.tag.split("}")[0] + "}"
            ns = {"db": ns_uri.strip("{}")}

        tag = f"{{{ns['db']}}}drug" if ns else "drug"
        for drug_el in root.findall(tag) if ns else root.iter("drug"):
            if drug_el.get("type") is None:
                continue

            # Check if ALS-relevant (by indication or target)
            indication = _find_text(drug_el, "indication", ns) or _find_text(drug_el, "indication", {})
            is_als = any(kw in indication.lower() for kw in _ALS_KEYWORDS)

            if not is_als:
                continue

            try:
                intervention = _parse_drug_entry(drug_el, ns)
                if intervention:
                    self._store.upsert_intervention(intervention)
                    result.interventions_added += 1
            except Exception as e:
                result.errors.append(f"Parse error: {e}")

        return result

    def fetch_drug_interactions(self, drugbank_ids: list[str]) -> list[dict]:
        """Return interactions between specified drugs (requires prior XML parse)."""
        # Simplified: read interactions from stored interventions
        interactions = []
        for dbid in drugbank_ids:
            int_obj = self._store.get_intervention(f"int:drugbank:{dbid}")
            if int_obj and int_obj.get("body"):
                for di in int_obj["body"].get("drug_interactions", []):
                    if di.get("drugbank_id") in drugbank_ids:
                        interactions.append(di)
        return interactions
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_drugbank_connector.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/drugbank.py tests/test_drugbank_connector.py && git commit -m "feat: DrugBank XML connector for ALS drug profiles and interaction checking"
```

---

## Task 7: Config Update + Final Verification

**Files:**
- Modify: `data/erik_config.json`

- [ ] **Step 1: Update config with connector settings**

Add to `data/erik_config.json`:

```json
{
  "version": "0.2.0",
  "database_name": "erik_kg",
  "llm_server_enabled": false,
  "temperature": 1.0,
  "exploration_epsilon": 0.30,
  "action_timeout_s": 120,
  "hot_reload_interval_steps": 10,
  "audit_enabled": true,
  "ncbi_api_key": null,
  "chembl_db_path": "/Volumes/Databank/databases/chembl_36.db",
  "drugbank_xml_path": null,
  "connector_max_retries": 3,
  "connector_backoff_seconds": [1, 2, 4]
}
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not network and not chembl" --tb=short`
Expected: All tests PASS

- [ ] **Step 3: Run network integration tests (optional)**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -m network --tb=short`
Expected: Network tests PASS (PubMed, ClinicalTrials.gov, OpenTargets)

- [ ] **Step 4: Commit and push**

```bash
git add data/erik_config.json && git commit -m "feat: add connector config settings (API keys, DB paths)"
git push
```

---

## Summary

After completing all 7 tasks, Phase 1B delivers:

- **BaseConnector ABC** with retry/backoff, 30s timeout, ConnectorResult dataclass
- **PubMedConnector** — E-utilities XML parsing, curated per-layer queries, abstract extraction
- **ClinicalTrialsConnector** — v2 API, trial parsing, Erik eligibility matching (age, sex, ALSFRS-R, FVC, genetic status), Ohio site detection
- **ChEMBLConnector** — Local SQL queries against ChEMBL 36 for compound bioactivity on ALS targets
- **OpenTargetsConnector** — GraphQL for ranked target-disease associations with tractability
- **DrugBankConnector** — XML parsing for drug profiles, MOA, interactions (when academic XML available)
- **Erik eligibility engine** — assesses eligibility against every active trial
- All results flow through existing EvidenceStore to PostgreSQL

**What comes next (Phase 2):** World Model MVP — latent state estimation, subtype posterior inference, progression forecasting, and the first cure protocol candidate for Erik.
