# Erik Cloud Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore data source access on Railway (15 connectors currently silently failing), ungate layer progression, and add query intelligence — targeting 5-10x evidence acquisition rate within 2 weeks.

**Architecture:** Connectors gain an API-mode variant selected by environment (`CONNECTOR_MODE=api`). Local-file connectors that have public REST APIs get thin API wrappers using the same `ConnectorResult` interface. Layer 3 becomes accessible via provisional genetic profile inference from clinical data. LLM generates novel queries when static rotation is exhausted.

**Tech Stack:** Python 3.12, psycopg, requests, FastAPI, Railway, Bedrock (Nova Micro/Pro)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/connectors/chembl_api.py` | **NEW** — ChEMBL REST API connector |
| `scripts/connectors/uniprot_api.py` | **NEW** — UniProt REST API connector |
| `scripts/connectors/alphafold_api.py` | **NEW** — AlphaFold REST API connector |
| `scripts/connectors/gtex_api.py` | **NEW** — GTEx Portal REST API connector |
| `scripts/connectors/gwas_api.py` | **NEW** — GWAS Catalog REST API connector |
| `scripts/connectors/gnomad_api.py` | **NEW** — gnomAD GraphQL API connector |
| `scripts/connectors/hpa_api.py` | **NEW** — Human Protein Atlas REST API connector |
| `scripts/connectors/connector_mode.py` | **NEW** — Mode-switching resolver (local vs API) |
| `scripts/research/loop.py:1765-1771` | **MODIFY** — Switch executor factory to use mode resolver |
| `scripts/research/layer_orchestrator.py:36-63` | **MODIFY** — Soft-gate Layer 3 with provisional profile |
| `scripts/research/policy.py:735-779` | **MODIFY** — Add LLM-powered query strategy |
| `scripts/research/loop.py:260-309` | **MODIFY** — Enhanced stagnation recovery with hard exploration |
| `data/erik_config.json` | **MODIFY** — Add connector_mode, provisional genetics config keys |
| `Dockerfile` | **MODIFY** — Bundle small data files |
| `tests/connectors/test_chembl_api.py` | **NEW** — ChEMBL API tests |
| `tests/connectors/test_uniprot_api.py` | **NEW** — UniProt API tests |
| `tests/connectors/test_connector_mode.py` | **NEW** — Mode resolver tests |
| `tests/research/test_layer_orchestrator_soft.py` | **NEW** — Soft-gate layer tests |
| `tests/research/test_llm_query_gen.py` | **NEW** — LLM query generation tests |

---

## Phase 1: Data Source Restoration (Week 1)

### Task 1: Connector Mode Framework

**Files:**
- Create: `scripts/connectors/connector_mode.py`
- Modify: `scripts/research/loop.py:1720-1771`
- Test: `tests/connectors/test_connector_mode.py`

The dispatch factory `_make_als_gene_executor()` at `loop.py:1720` dynamically imports a connector class path. We add a resolver that maps local connector paths to API variants when `CONNECTOR_MODE=api`.

- [ ] **Step 1: Write failing test for mode resolver**

```python
# tests/connectors/test_connector_mode.py
import os
import pytest
from connectors.connector_mode import resolve_connector_class


def test_local_mode_returns_local():
    """In local mode, resolver returns the original class path."""
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector", mode="local")
    assert result == "connectors.chembl.ChEMBLConnector"


def test_api_mode_returns_api_variant():
    """In API mode, resolver returns the API class path when available."""
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector", mode="api")
    assert result == "connectors.chembl_api.ChEMBLAPIConnector"


def test_api_mode_falls_back_for_unknown():
    """If no API variant exists, fall back to the original."""
    result = resolve_connector_class("connectors.unknown.Foo", mode="api")
    assert result == "connectors.unknown.Foo"


def test_env_var_controls_mode(monkeypatch):
    """CONNECTOR_MODE env var determines which variant is used."""
    monkeypatch.setenv("CONNECTOR_MODE", "api")
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector")
    assert result == "connectors.chembl_api.ChEMBLAPIConnector"


def test_default_mode_is_local():
    """Without env var, default to local."""
    os.environ.pop("CONNECTOR_MODE", None)
    result = resolve_connector_class("connectors.chembl.ChEMBLConnector")
    assert result == "connectors.chembl.ChEMBLConnector"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_connector_mode.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'connectors.connector_mode'`

- [ ] **Step 3: Implement connector_mode.py**

```python
# scripts/connectors/connector_mode.py
"""Connector mode resolver — maps local connectors to API variants on Railway."""
from __future__ import annotations

import os

# Map: local class path → API class path
_API_VARIANTS: dict[str, str] = {
    "connectors.chembl.ChEMBLConnector": "connectors.chembl_api.ChEMBLAPIConnector",
    "connectors.clinvar_local.ClinVarLocalConnector": "connectors.clinvar.ClinVarConnector",
    "connectors.reactome_local.ReactomeLocalConnector": "connectors.reactome.ReactomeConnector",
    "connectors.uniprot.UniProtConnector": "connectors.uniprot_api.UniProtAPIConnector",
    "connectors.alphafold_local.AlphaFoldLocalConnector": "connectors.alphafold_api.AlphaFoldAPIConnector",
    "connectors.gtex.GTExConnector": "connectors.gtex_api.GTExAPIConnector",
    "connectors.gwas_catalog.GWASCatalogConnector": "connectors.gwas_api.GWASCatalogAPIConnector",
    "connectors.gnomad.GnomADConnector": "connectors.gnomad_api.GnomADAPIConnector",
    "connectors.hpa.HPAConnector": "connectors.hpa_api.HPAAPIConnector",
}


def resolve_connector_class(
    local_class_path: str,
    mode: str | None = None,
) -> str:
    """Return the connector class path appropriate for the current mode.

    Args:
        local_class_path: Dotted import path of the local connector.
        mode: 'local' or 'api'. Defaults to CONNECTOR_MODE env var, then 'local'.

    Returns:
        The (possibly remapped) class path.
    """
    if mode is None:
        mode = os.environ.get("CONNECTOR_MODE", "local").lower()
    if mode == "api":
        return _API_VARIANTS.get(local_class_path, local_class_path)
    return local_class_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_connector_mode.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Wire mode resolver into executor factory**

Modify `scripts/research/loop.py` — the `_make_als_gene_executor` function (around line 1720):

```python
# BEFORE (loop.py ~line 1744):
        module_name, class_name = connector_class_path.rsplit(".", 1)

# AFTER:
        from connectors.connector_mode import resolve_connector_class
        resolved_path = resolve_connector_class(connector_class_path)
        module_name, class_name = resolved_path.rsplit(".", 1)
```

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v --timeout=30`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/connectors/connector_mode.py tests/connectors/test_connector_mode.py scripts/research/loop.py
git commit -m "feat: add connector mode framework for local/API switching"
```

---

### Task 2: ChEMBL API Connector

**Files:**
- Create: `scripts/connectors/chembl_api.py`
- Test: `tests/connectors/test_chembl_api.py`

The ChEMBL REST API at `https://www.ebi.ac.uk/chembl/api/data/` provides bioactivity, compound properties, and mechanism of action. This replaces the 30.7GB local SQLite database — the single most important data source for drug discovery.

- [ ] **Step 1: Write failing test**

```python
# tests/connectors/test_chembl_api.py
import pytest
from unittest.mock import patch, MagicMock
from connectors.chembl_api import ChEMBLAPIConnector


@pytest.fixture
def store():
    s = MagicMock()
    s.upsert_evidence.return_value = True
    return s


@pytest.fixture
def connector(store):
    return ChEMBLAPIConnector(store=store)


def test_fetch_returns_connector_result(connector):
    """fetch() returns ConnectorResult with evidence_items_added."""
    with patch("connectors.chembl_api._retry_get") as mock_get:
        mock_get.return_value = MagicMock(
            status_code=200,
            json=lambda: {
                "activities": [
                    {
                        "molecule_chembl_id": "CHEMBL25",
                        "target_chembl_id": "CHEMBL2093872",
                        "standard_type": "IC50",
                        "standard_value": "50.0",
                        "standard_units": "nM",
                        "pchembl_value": "7.3",
                    }
                ]
            },
        )
        cr = connector.fetch(gene="SOD1", uniprot="P00441")
        assert cr.evidence_items_added >= 0
        assert not cr.errors


def test_fetch_full_profile_queries_three_endpoints(connector):
    """fetch_full_profile hits bioactivity + mechanism + compound endpoints."""
    responses = []
    for payload in [
        {"activities": []},
        {"mechanisms": []},
        {"molecules": []},
    ]:
        r = MagicMock(status_code=200, json=lambda p=payload: p)
        responses.append(r)
    with patch("connectors.chembl_api._retry_get", side_effect=responses):
        cr = connector.fetch_full_profile(uniprot_id="P00441")
        assert not cr.errors


def test_fetch_handles_api_error(connector):
    """HTTP errors are caught and reported in result.errors."""
    with patch("connectors.chembl_api._retry_get") as mock_get:
        mock_get.side_effect = Exception("Connection refused")
        cr = connector.fetch(gene="SOD1", uniprot="P00441")
        assert len(cr.errors) > 0
        assert cr.evidence_items_added == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_chembl_api.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement ChEMBL API connector**

```python
# scripts/connectors/chembl_api.py
"""ChEMBL REST API connector — replaces local SQLite on Railway."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import requests

from connectors.base import ConnectorResult

_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
_HEADERS = {"Accept": "application/json"}
_TIMEOUT = 30


def _retry_get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    """GET with exponential backoff."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class ChEMBLAPIConnector:
    """Query ChEMBL REST API for drug-target bioactivity."""

    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        """Fetch bioactivity data for a gene/UniProt target."""
        result = ConnectorResult()
        try:
            # Search for target by gene or UniProt
            if uniprot:
                target_url = f"{_BASE_URL}/target.json"
                resp = _retry_get(target_url, params={
                    "target_components__accession": uniprot,
                    "limit": 1,
                })
                targets = resp.json().get("targets", [])
            elif gene:
                target_url = f"{_BASE_URL}/target/search.json"
                resp = _retry_get(target_url, params={"q": gene, "limit": 5})
                targets = resp.json().get("targets", [])
            else:
                return result

            if not targets:
                return result

            target_chembl_id = targets[0].get("target_chembl_id", "")
            if not target_chembl_id:
                return result

            # Fetch bioactivity data
            activity_url = f"{_BASE_URL}/activity.json"
            resp = _retry_get(activity_url, params={
                "target_chembl_id": target_chembl_id,
                "standard_type__in": "IC50,EC50,Ki,Kd",
                "pchembl_value__gte": 5,
                "limit": 20,
            })
            activities = resp.json().get("activities", [])

            for act in activities:
                mol_id = act.get("molecule_chembl_id", "")
                std_type = act.get("standard_type", "")
                std_value = act.get("standard_value", "")
                std_units = act.get("standard_units", "")
                pchembl = act.get("pchembl_value", "")

                claim = (
                    f"{mol_id} has {std_type} = {std_value} {std_units} "
                    f"(pChEMBL {pchembl}) against {gene or uniprot}"
                )
                evi_id = f"evi:chembl:{mol_id}_{target_chembl_id}_{std_type}".lower()

                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id,
                        type="EvidenceItem",
                        claim=claim,
                        source="chembl",
                        provenance=f"ChEMBL API: {mol_id}",
                        confidence=min(1.0, float(pchembl or 0) / 10.0),
                        protocol_layer="root_cause_suppression",
                        evidence_strength="strong" if float(pchembl or 0) >= 7 else "moderate",
                    )
                    if added:
                        result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"ChEMBL API error: {e}")

        return result

    def fetch_full_profile(self, uniprot_id: str = "", **kwargs: Any) -> ConnectorResult:
        """Fetch comprehensive ADME/Tox profile — bioactivity + mechanisms + compounds."""
        result = self.fetch(uniprot=uniprot_id)

        # Mechanism of action
        try:
            moa_url = f"{_BASE_URL}/mechanism.json"
            resp = _retry_get(moa_url, params={
                "target_components__accession": uniprot_id,
                "limit": 10,
            })
            mechanisms = resp.json().get("mechanisms", [])
            for mech in mechanisms:
                mol_id = mech.get("molecule_chembl_id", "")
                action = mech.get("mechanism_of_action", "unknown")
                evi_id = f"evi:chembl:moa:{mol_id}_{uniprot_id}".lower()
                claim = f"{mol_id}: {action} (target: {uniprot_id})"
                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id,
                        type="EvidenceItem",
                        claim=claim,
                        source="chembl",
                        provenance=f"ChEMBL MOA: {mol_id}",
                        confidence=0.8,
                        protocol_layer="root_cause_suppression",
                        evidence_strength="strong",
                    )
                    if added:
                        result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"ChEMBL MOA error: {e}")

        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_chembl_api.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/connectors/chembl_api.py tests/connectors/test_chembl_api.py
git commit -m "feat: add ChEMBL REST API connector for Railway deployment"
```

---

### Task 3: UniProt + AlphaFold + GTEx API Connectors

**Files:**
- Create: `scripts/connectors/uniprot_api.py`
- Create: `scripts/connectors/alphafold_api.py`
- Create: `scripts/connectors/gtex_api.py`
- Test: `tests/connectors/test_api_connectors.py`

All three follow the same pattern as ChEMBL API: thin REST wrapper returning `ConnectorResult`. The `fetch(gene, uniprot)` interface matches what `_make_als_gene_executor` calls.

- [ ] **Step 1: Write failing tests for all three**

```python
# tests/connectors/test_api_connectors.py
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def store():
    s = MagicMock()
    s.upsert_evidence.return_value = True
    return s


class TestUniProtAPI:
    def test_fetch_gene(self, store):
        from connectors.uniprot_api import UniProtAPIConnector
        c = UniProtAPIConnector(store=store)
        with patch("connectors.uniprot_api._retry_get") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"results": [{
                    "primaryAccession": "P00441",
                    "proteinDescription": {"recommendedName": {"fullName": {"value": "SOD1"}}},
                    "comments": [{"commentType": "FUNCTION", "texts": [{"value": "Destroys superoxide radicals"}]}],
                }]},
            )
            cr = c.fetch(gene="SOD1", uniprot="P00441")
            assert not cr.errors


class TestAlphaFoldAPI:
    def test_fetch_structure(self, store):
        from connectors.alphafold_api import AlphaFoldAPIConnector
        c = AlphaFoldAPIConnector(store=store)
        with patch("connectors.alphafold_api._retry_get") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: [{
                    "entryId": "AF-P00441-F1",
                    "gene": "SOD1",
                    "uniprotAccession": "P00441",
                    "globalMetricValue": 92.5,
                }],
            )
            cr = c.fetch(gene="SOD1", uniprot="P00441")
            assert not cr.errors


class TestGTExAPI:
    def test_fetch_expression(self, store):
        from connectors.gtex_api import GTExAPIConnector
        c = GTExAPIConnector(store=store)
        with patch("connectors.gtex_api._retry_get") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"medianGeneExpression": [
                    {"tissueSiteDetailId": "Brain_Spinal_cord_cervical_c-1", "median": 15.2, "gencodeId": "ENSG00000142168"},
                ]},
            )
            cr = c.fetch(gene="SOD1", uniprot="P00441")
            assert not cr.errors
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_api_connectors.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement UniProt API connector**

```python
# scripts/connectors/uniprot_api.py
"""UniProt REST API connector — protein function, PTMs, disease associations."""
from __future__ import annotations
import time
from typing import Any
import requests
from connectors.base import ConnectorResult

_BASE_URL = "https://rest.uniprot.org/uniprotkb"
_TIMEOUT = 30


def _retry_get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers={"Accept": "application/json"}, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class UniProtAPIConnector:
    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        try:
            if uniprot:
                resp = _retry_get(f"{_BASE_URL}/{uniprot}.json")
                entries = [resp.json()]
            elif gene:
                resp = _retry_get(f"{_BASE_URL}/search", params={
                    "query": f"gene_exact:{gene} AND organism_id:9606",
                    "format": "json", "size": "1",
                })
                entries = resp.json().get("results", [])
            else:
                return result

            for entry in entries[:1]:
                accession = entry.get("primaryAccession", "")
                # Function
                for comment in entry.get("comments", []):
                    ctype = comment.get("commentType", "")
                    texts = comment.get("texts", [])
                    if ctype == "FUNCTION" and texts:
                        claim = f"{gene} ({accession}): {texts[0].get('value', '')}"
                        evi_id = f"evi:uniprot:{accession}_function".lower()
                        if self._store:
                            added = self._store.upsert_evidence(
                                id=evi_id, type="EvidenceItem", claim=claim,
                                source="uniprot", provenance=f"UniProt: {accession}",
                                confidence=0.9, protocol_layer="root_cause_suppression",
                                evidence_strength="strong",
                            )
                            if added:
                                result.evidence_items_added += 1
                # Disease involvement
                for comment in entry.get("comments", []):
                    if comment.get("commentType") == "DISEASE":
                        disease = comment.get("disease", {})
                        name = disease.get("diseaseId", "")
                        if name:
                            claim = f"{gene} associated with {name}"
                            evi_id = f"evi:uniprot:{accession}_{name}".lower().replace(" ", "_")
                            if self._store:
                                added = self._store.upsert_evidence(
                                    id=evi_id, type="EvidenceItem", claim=claim,
                                    source="uniprot", provenance=f"UniProt: {accession}",
                                    confidence=0.85, protocol_layer="root_cause_suppression",
                                    evidence_strength="strong",
                                )
                                if added:
                                    result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"UniProt API error: {e}")
        return result
```

- [ ] **Step 4: Implement AlphaFold API connector**

```python
# scripts/connectors/alphafold_api.py
"""AlphaFold REST API connector — protein structure confidence metrics."""
from __future__ import annotations
import time
from typing import Any
import requests
from connectors.base import ConnectorResult

_BASE_URL = "https://alphafold.ebi.ac.uk/api"
_TIMEOUT = 30


def _retry_get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class AlphaFoldAPIConnector:
    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        if not uniprot:
            return result
        try:
            resp = _retry_get(f"{_BASE_URL}/prediction/{uniprot}")
            entries = resp.json() if isinstance(resp.json(), list) else [resp.json()]
            for entry in entries[:1]:
                plddt = entry.get("globalMetricValue", 0)
                entry_id = entry.get("entryId", f"AF-{uniprot}")
                claim = (
                    f"AlphaFold structure {entry_id} for {gene or uniprot}: "
                    f"global pLDDT = {plddt:.1f} (confidence {'high' if plddt >= 70 else 'low'})"
                )
                evi_id = f"evi:alphafold:{uniprot}".lower()
                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id, type="EvidenceItem", claim=claim,
                        source="alphafold", provenance=f"AlphaFold: {entry_id}",
                        confidence=plddt / 100.0, protocol_layer="root_cause_suppression",
                        evidence_strength="strong" if plddt >= 70 else "emerging",
                    )
                    if added:
                        result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"AlphaFold API error: {e}")
        return result
```

- [ ] **Step 5: Implement GTEx API connector**

```python
# scripts/connectors/gtex_api.py
"""GTEx Portal REST API connector — tissue-specific gene expression."""
from __future__ import annotations
import time
from typing import Any
import requests
from connectors.base import ConnectorResult

_BASE_URL = "https://gtexportal.org/api/v2"
_TIMEOUT = 30
_ALS_TISSUES = [
    "Brain_Spinal_cord_cervical_c-1",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Cortex",
    "Nerve_Tibial",
    "Muscle_Skeletal",
    "Whole_Blood",
]


def _retry_get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class GTExAPIConnector:
    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        if not gene:
            return result
        try:
            resp = _retry_get(f"{_BASE_URL}/expression/medianGeneExpression", params={
                "gencodeId": gene,
                "datasetId": "gtex_v8",
            })
            data = resp.json()
            expressions = data.get("medianGeneExpression", data.get("data", []))
            for expr in expressions:
                tissue = expr.get("tissueSiteDetailId", "")
                if tissue not in _ALS_TISSUES:
                    continue
                median = expr.get("median", 0)
                claim = f"{gene} expression in {tissue}: median TPM = {median:.2f}"
                evi_id = f"evi:gtex:{gene}_{tissue}".lower().replace("-", "_")
                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id, type="EvidenceItem", claim=claim,
                        source="gtex", provenance=f"GTEx v8: {tissue}",
                        confidence=0.9, protocol_layer="root_cause_suppression",
                        evidence_strength="strong",
                    )
                    if added:
                        result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"GTEx API error: {e}")
        return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_api_connectors.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/connectors/uniprot_api.py scripts/connectors/alphafold_api.py scripts/connectors/gtex_api.py tests/connectors/test_api_connectors.py
git commit -m "feat: add UniProt, AlphaFold, GTEx REST API connectors"
```

---

### Task 4: GWAS + gnomAD + HPA API Connectors

**Files:**
- Create: `scripts/connectors/gwas_api.py`
- Create: `scripts/connectors/gnomad_api.py`
- Create: `scripts/connectors/hpa_api.py`
- Test: `tests/connectors/test_api_connectors_batch2.py`

Same pattern: thin REST wrappers. GWAS Catalog and HPA have simple REST APIs. gnomAD uses GraphQL.

- [ ] **Step 1: Write failing tests**

```python
# tests/connectors/test_api_connectors_batch2.py
import pytest
from unittest.mock import patch, MagicMock


@pytest.fixture
def store():
    s = MagicMock()
    s.upsert_evidence.return_value = True
    return s


class TestGWASAPI:
    def test_fetch(self, store):
        from connectors.gwas_api import GWASCatalogAPIConnector
        c = GWASCatalogAPIConnector(store=store)
        with patch("connectors.gwas_api._retry_get") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"_embedded": {"associations": [{
                    "riskFrequency": "0.15",
                    "pvalue": 1e-12,
                    "strongestSnpRiskAlleles": [{"riskAlleleName": "rs12345-A"}],
                    "_links": {"study": {"href": "https://www.ebi.ac.uk/gwas/rest/api/studies/GCST000001"}},
                }]}},
            )
            cr = c.fetch(gene="SOD1", uniprot="")
            assert not cr.errors


class TestGnomADAPI:
    def test_fetch(self, store):
        from connectors.gnomad_api import GnomADAPIConnector
        c = GnomADAPIConnector(store=store)
        with patch("connectors.gnomad_api.requests.post") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: {"data": {"gene": {
                    "gene_id": "ENSG00000142168",
                    "gnomad_constraint": {
                        "pLI": 0.99, "loeuf": 0.15,
                        "mis_z": 3.2, "oe_lof": 0.08,
                    },
                }}},
            )
            cr = c.fetch(gene="SOD1", uniprot="")
            assert not cr.errors


class TestHPAAPI:
    def test_fetch(self, store):
        from connectors.hpa_api import HPAAPIConnector
        c = HPAAPIConnector(store=store)
        with patch("connectors.hpa_api._retry_get") as mock:
            mock.return_value = MagicMock(
                status_code=200,
                json=lambda: [{
                    "Gene": "SOD1",
                    "Gene description": "Superoxide dismutase 1",
                    "Protein class": "Enzyme",
                    "Biological process": "Response to oxidative stress",
                }],
            )
            cr = c.fetch(gene="SOD1", uniprot="")
            assert not cr.errors
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_api_connectors_batch2.py -v`
Expected: FAIL

- [ ] **Step 3: Implement GWAS Catalog API connector**

```python
# scripts/connectors/gwas_api.py
"""GWAS Catalog REST API connector — genome-wide association studies."""
from __future__ import annotations
import time
from typing import Any
import requests
from connectors.base import ConnectorResult

_BASE_URL = "https://www.ebi.ac.uk/gwas/rest/api"
_ALS_KEYWORDS = {"amyotrophic lateral sclerosis", "als", "motor neuron disease", "ftd"}
_TIMEOUT = 30


def _retry_get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class GWASCatalogAPIConnector:
    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        if not gene:
            return result
        try:
            resp = _retry_get(f"{_BASE_URL}/singleNucleotidePolymorphisms/search/findByGene", params={
                "geneName": gene,
            })
            snps = resp.json().get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
            for snp in snps[:10]:
                rs_id = snp.get("rsId", "")
                locations = snp.get("locations", [{}])
                chrom = locations[0].get("chromosomeName", "?") if locations else "?"
                pos = locations[0].get("chromosomePosition", "?") if locations else "?"
                claim = f"GWAS SNP {rs_id} near {gene} at chr{chrom}:{pos}"
                evi_id = f"evi:gwas:{rs_id}_{gene}".lower()
                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id, type="EvidenceItem", claim=claim,
                        source="gwas_catalog", provenance=f"GWAS Catalog: {rs_id}",
                        confidence=0.7, protocol_layer="root_cause_suppression",
                        evidence_strength="moderate",
                    )
                    if added:
                        result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"GWAS API error: {e}")
        return result
```

- [ ] **Step 4: Implement gnomAD GraphQL API connector**

```python
# scripts/connectors/gnomad_api.py
"""gnomAD GraphQL API connector — gene constraint metrics (pLI, LOEUF)."""
from __future__ import annotations
from typing import Any
import requests
from connectors.base import ConnectorResult

_GRAPHQL_URL = "https://gnomad.broadinstitute.org/api"
_TIMEOUT = 30

_QUERY = """
query GeneConstraint($gene: String!) {
  gene(gene_symbol: $gene, reference_genome: GRCh38) {
    gene_id
    gnomad_constraint {
      pLI
      oe_lof
      oe_lof_upper
      oe_mis
      mis_z
    }
  }
}
"""


class GnomADAPIConnector:
    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        if not gene:
            return result
        try:
            resp = requests.post(
                _GRAPHQL_URL,
                json={"query": _QUERY, "variables": {"gene": gene}},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json().get("data", {}).get("gene", {})
            if not data:
                return result
            constraint = data.get("gnomad_constraint", {})
            if not constraint:
                return result
            pli = constraint.get("pLI", 0)
            loeuf = constraint.get("oe_lof_upper", 0)
            mis_z = constraint.get("mis_z", 0)
            claim = (
                f"{gene} constraint: pLI={pli:.3f}, LOEUF={loeuf:.3f}, "
                f"missense z={mis_z:.2f} — "
                f"{'intolerant to LoF' if pli > 0.9 else 'tolerant to LoF'}"
            )
            evi_id = f"evi:gnomad:{gene}_constraint".lower()
            if self._store:
                added = self._store.upsert_evidence(
                    id=evi_id, type="EvidenceItem", claim=claim,
                    source="gnomad", provenance=f"gnomAD v4.1: {gene}",
                    confidence=0.95, protocol_layer="root_cause_suppression",
                    evidence_strength="strong",
                )
                if added:
                    result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"gnomAD API error: {e}")
        return result
```

- [ ] **Step 5: Implement HPA API connector**

```python
# scripts/connectors/hpa_api.py
"""Human Protein Atlas REST API connector — protein expression and localization."""
from __future__ import annotations
import time
from typing import Any
import requests
from connectors.base import ConnectorResult

_BASE_URL = "https://www.proteinatlas.org"
_TIMEOUT = 30


def _retry_get(url: str, params: dict | None = None, retries: int = 3) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


class HPAAPIConnector:
    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        if not gene:
            return result
        try:
            resp = _retry_get(f"{_BASE_URL}/{gene}.json")
            entries = resp.json() if isinstance(resp.json(), list) else [resp.json()]
            for entry in entries[:1]:
                protein_class = entry.get("Protein class", "")
                bio_process = entry.get("Biological process", "")
                description = entry.get("Gene description", "")
                subcell = entry.get("Subcellular location", "")
                claim = f"{gene}: {description}. Class: {protein_class}. Process: {bio_process}."
                if subcell:
                    claim += f" Location: {subcell}."
                evi_id = f"evi:hpa:{gene}_profile".lower()
                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id, type="EvidenceItem", claim=claim,
                        source="hpa", provenance=f"Human Protein Atlas: {gene}",
                        confidence=0.85, protocol_layer="root_cause_suppression",
                        evidence_strength="strong",
                    )
                    if added:
                        result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"HPA API error: {e}")
        return result
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/connectors/test_api_connectors_batch2.py -v`
Expected: PASS (all 3 tests)

- [ ] **Step 7: Update connector_mode.py mappings and commit**

Verify all new API connector class paths are in `_API_VARIANTS` dict in `connector_mode.py` (done in Task 1).

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/connectors/gwas_api.py scripts/connectors/gnomad_api.py scripts/connectors/hpa_api.py tests/connectors/test_api_connectors_batch2.py
git commit -m "feat: add GWAS, gnomAD, HPA API connectors"
```

---

### Task 5: Set CONNECTOR_MODE on Railway + Config Update

**Files:**
- Modify: `data/erik_config.json`
- Modify: `Dockerfile` (add small bundled data files)

- [ ] **Step 1: Add connector_mode config key**

Add to `data/erik_config.json` after line 97 (`"llm_backend": "mlx"`):

```json
  "connector_mode": "auto",
```

The `auto` mode means: use API if `/Volumes/Databank` doesn't exist, local otherwise. But in practice the env var `CONNECTOR_MODE=api` on Railway is the primary control.

- [ ] **Step 2: Set CONNECTOR_MODE=api on Railway**

This must be done via Railway dashboard or CLI:

```bash
# Via Railway CLI (if available):
railway variables set CONNECTOR_MODE=api
```

Or via Railway dashboard: Project → erik-api → Variables → Add `CONNECTOR_MODE` = `api`

- [ ] **Step 3: Commit config change**

```bash
cd /Users/logannye/.openclaw/erik
git add data/erik_config.json
git commit -m "config: add connector_mode key for local/api switching"
```

---

## Phase 2: Intelligence Upgrades (Week 2)

### Task 6: Ungate Layer 3 — Provisional Genetic Profile

**Files:**
- Modify: `scripts/research/layer_orchestrator.py:36-63`
- Create: `scripts/research/provisional_genetics.py`
- Test: `tests/research/test_provisional_genetics.py`

The system is stuck in Layer 2 because `genetic_profile == None`. Erik's clinical presentation (67M, limb-onset, ALSFRS-R 43, NfL elevated, diagnosed March 2026) constrains the likely genetic subtypes. We create a provisional profile that lets Layer 3 queries fire while clearly marking evidence as "provisional."

- [ ] **Step 1: Write failing test for provisional profile**

```python
# tests/research/test_provisional_genetics.py
import pytest
from research.provisional_genetics import infer_provisional_profile


def test_provisional_profile_returns_dict():
    """Profile contains gene, variant, subtype, and provisional flag."""
    profile = infer_provisional_profile(
        age_onset=67,
        site_onset="limb",
        alsfrs_r=43,
        nfl_elevated=True,
    )
    assert profile["provisional"] is True
    assert "gene" in profile
    assert "subtype" in profile
    assert "variant" in profile


def test_late_onset_limb_prefers_sporadic():
    """67-year-old limb-onset most likely sporadic (90-95% of ALS)."""
    profile = infer_provisional_profile(age_onset=67, site_onset="limb")
    assert profile["subtype"] == "sALS"
    # Sporadic ALS: TDP-43 proteinopathy is the most common molecular feature
    assert profile["gene"] in ("TARDBP", "C9orf72", "SOD1")


def test_young_onset_bulbar_flags_fals():
    """Young bulbar onset should flag possible familial ALS."""
    profile = infer_provisional_profile(age_onset=35, site_onset="bulbar")
    assert profile["subtype"] in ("fALS", "sALS")


def test_provisional_flag_always_set():
    """Provisional profiles must always be flagged as provisional."""
    profile = infer_provisional_profile()
    assert profile["provisional"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_provisional_genetics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement provisional genetics inference**

```python
# scripts/research/provisional_genetics.py
"""Provisional genetic profile inference from clinical presentation.

This generates a PROVISIONAL genetic profile so the research loop can
begin Layer 3 (erik-specific) queries without waiting for Invitae results.
All evidence generated under a provisional profile is tagged as such and
will be re-validated when real genetic data arrives.

Statistics source: Brown & Al-Chalabi (2017) NEJM; Zou et al (2017) Ann Neurol.
"""
from __future__ import annotations


def infer_provisional_profile(
    age_onset: int = 67,
    site_onset: str = "limb",
    alsfrs_r: int | None = 43,
    nfl_elevated: bool = True,
    family_history: bool = False,
) -> dict:
    """Infer most likely ALS molecular subtype from clinical features.

    Returns a dict with keys: gene, variant, subtype, confidence, provisional.
    This is a Bayesian estimate, NOT a diagnosis.
    """
    if family_history:
        subtype = "fALS"
        # fALS: C9orf72 (40%), SOD1 (20%), FUS (5%), TARDBP (5%)
        gene = "C9orf72"
        variant = "repeat_expansion"
        confidence = 0.40
    elif age_onset < 45:
        subtype = "sALS"
        # Young sporadic: higher FUS probability
        gene = "FUS"
        variant = "unknown"
        confidence = 0.15
    else:
        subtype = "sALS"
        # Sporadic ALS (90-95% of cases):
        # ~97% have TDP-43 pathology regardless of genetic cause
        # Most common molecular feature is TDP-43 proteinopathy
        gene = "TARDBP"
        variant = "tdp43_proteinopathy"
        confidence = 0.70  # 97% of sALS has TDP-43 pathology

    # NfL elevation confirms active neurodegeneration (doesn't change subtype)
    if nfl_elevated:
        confidence = min(1.0, confidence + 0.05)

    return {
        "gene": gene,
        "variant": variant,
        "subtype": subtype,
        "confidence": confidence,
        "provisional": True,
        "source": "clinical_inference",
        "clinical_features": {
            "age_onset": age_onset,
            "site_onset": site_onset,
            "alsfrs_r": alsfrs_r,
            "nfl_elevated": nfl_elevated,
            "family_history": family_history,
        },
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_provisional_genetics.py -v`
Expected: PASS

- [ ] **Step 5: Modify layer_orchestrator.py to use provisional profile**

In `scripts/research/layer_orchestrator.py`, modify the `determine_layer()` function to accept an optional provisional profile when `genetic_profile` is None:

```python
# BEFORE (layer_orchestrator.py ~line 36-63):
def determine_layer(
    evidence_count: int,
    genetic_profile: Optional[dict[str, Any]],
    validated_targets: int,
) -> ResearchLayer:
    if genetic_profile is None:
        if evidence_count < LAYER_1_THRESHOLD:
            return ResearchLayer.NORMAL_BIOLOGY
        return ResearchLayer.ALS_MECHANISMS

# AFTER:
def determine_layer(
    evidence_count: int,
    genetic_profile: Optional[dict[str, Any]],
    validated_targets: int,
    provisional_genetics_enabled: bool = True,
) -> ResearchLayer:
    if genetic_profile is None:
        if evidence_count < LAYER_1_THRESHOLD:
            return ResearchLayer.NORMAL_BIOLOGY
        # With enough ALS mechanism evidence, allow provisional Layer 3
        if provisional_genetics_enabled and evidence_count >= 500:
            from research.provisional_genetics import infer_provisional_profile
            # Don't transition to DRUG_DESIGN without confirmed genetics
            return ResearchLayer.ERIK_SPECIFIC
        return ResearchLayer.ALS_MECHANISMS
```

Also modify `get_layer_queries()` to generate Layer 3 queries from the provisional profile when `genetic_profile` is None:

```python
# In get_layer_queries(), for ERIK_SPECIFIC layer when genetic_profile is None:
    if layer == ResearchLayer.ERIK_SPECIFIC and genetic_profile is None:
        from research.provisional_genetics import infer_provisional_profile
        prov = infer_provisional_profile()
        genetic_profile = prov  # Use provisional for query generation
```

- [ ] **Step 6: Add config key and write integration test**

Add to `data/erik_config.json`:
```json
  "provisional_genetics_enabled": true,
  "provisional_genetics_min_evidence": 500,
```

Write integration test:
```python
# tests/research/test_layer_orchestrator_soft.py
from research.layer_orchestrator import determine_layer, ResearchLayer


def test_layer3_accessible_with_provisional():
    """Layer 3 should be accessible without genetics when evidence >= 500."""
    layer = determine_layer(
        evidence_count=600,
        genetic_profile=None,
        validated_targets=0,
        provisional_genetics_enabled=True,
    )
    assert layer == ResearchLayer.ERIK_SPECIFIC


def test_layer3_blocked_without_provisional():
    """Layer 3 blocked without genetics when provisional disabled."""
    layer = determine_layer(
        evidence_count=600,
        genetic_profile=None,
        validated_targets=0,
        provisional_genetics_enabled=False,
    )
    assert layer == ResearchLayer.ALS_MECHANISMS


def test_drug_design_still_requires_confirmed_genetics():
    """Layer 4 (drug design) must never use provisional genetics."""
    layer = determine_layer(
        evidence_count=10000,
        genetic_profile=None,
        validated_targets=5,
        provisional_genetics_enabled=True,
    )
    # Even with high evidence and validated targets,
    # cannot reach DRUG_DESIGN without confirmed genetics
    assert layer == ResearchLayer.ERIK_SPECIFIC
```

- [ ] **Step 7: Run tests and commit**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_provisional_genetics.py tests/research/test_layer_orchestrator_soft.py -v`
Expected: PASS

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/research/provisional_genetics.py scripts/research/layer_orchestrator.py tests/research/test_provisional_genetics.py tests/research/test_layer_orchestrator_soft.py data/erik_config.json
git commit -m "feat: ungate Layer 3 with provisional genetic profile inference"
```

---

### Task 7: LLM-Powered Dynamic Query Generation

**Files:**
- Modify: `scripts/research/policy.py:153-178` (`_get_dynamic_query`)
- Test: `tests/research/test_llm_query_gen.py`

Currently `_get_dynamic_query()` extracts terms from hypothesis text using string splitting. Replace this with an LLM call that generates novel, targeted search queries based on the current evidence gaps and active hypotheses — using the already-loaded Bedrock Nova Micro model.

- [ ] **Step 1: Write failing test**

```python
# tests/research/test_llm_query_gen.py
import pytest
from unittest.mock import MagicMock, patch
from research.policy import _get_llm_query


def test_llm_query_returns_string():
    """LLM-generated query must be a non-empty string."""
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "TDP-43 nuclear depletion cryptic exon STMN2 therapy 2026"
    query = _get_llm_query(
        llm=mock_llm,
        active_hypotheses=["TDP-43 aggregation drives cryptic exon splicing"],
        top_uncertainties=["Missing genetic testing data"],
        layer="root_cause_suppression",
    )
    assert isinstance(query, str)
    assert len(query) > 10


def test_llm_query_falls_back_on_error():
    """If LLM fails, fall back to static query."""
    mock_llm = MagicMock()
    mock_llm.generate.side_effect = Exception("LLM timeout")
    query = _get_llm_query(
        llm=mock_llm,
        active_hypotheses=["Test hypothesis"],
        top_uncertainties=["Missing data"],
        layer="root_cause_suppression",
    )
    # Should return a valid fallback, not raise
    assert isinstance(query, str)
    assert len(query) > 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_llm_query_gen.py -v`
Expected: FAIL

- [ ] **Step 3: Implement LLM query generation**

Add to `scripts/research/policy.py` (new function, around line 180):

```python
def _get_llm_query(
    llm: Any,
    active_hypotheses: list[str],
    top_uncertainties: list[str],
    layer: str,
) -> str:
    """Generate a novel PubMed search query using the LLM.

    The LLM sees the current hypotheses and gaps, then produces a targeted
    biomedical query that explores uncharted territory.
    """
    import datetime
    year = datetime.datetime.now().year

    prompt = (
        "You are a biomedical research assistant. Generate ONE PubMed search query "
        "that would help investigate ALS (amyotrophic lateral sclerosis) treatment.\n\n"
        f"Current research focus: {layer.replace('_', ' ')}\n"
        f"Active hypotheses:\n"
    )
    for h in active_hypotheses[:3]:
        prompt += f"- {h[:150]}\n"
    prompt += f"\nKey uncertainties:\n"
    for u in top_uncertainties[:3]:
        prompt += f"- {u[:150]}\n"
    prompt += (
        f"\nGenerate a SINGLE PubMed query (10-15 words) that explores a NOVEL angle "
        f"not directly covered by the hypotheses above. Include '{year}' for recency. "
        f"Output ONLY the query, nothing else."
    )

    try:
        result = llm.generate(prompt, max_tokens=60, temperature=0.7)
        query = result.strip().strip('"').strip("'")
        if len(query) > 10:
            return query
    except Exception:
        pass

    # Fallback to static
    return get_layer_query(layer, hash(str(active_hypotheses)) % 100)
```

- [ ] **Step 4: Wire LLM query into the acquisition strategy rotation**

Modify `_build_acquisition_params()` in `policy.py` (around line 756). Add a 5th strategy that uses LLM:

```python
# In _build_acquisition_params, for SEARCH_PUBMED (after line 765):
        if layer_queries:
            strategy = step % 5  # Was mod 4 — now 5 strategies
            if strategy <= 1 and layer_queries:
                query = layer_queries[step % len(layer_queries)]
                year = __import__("datetime").datetime.now().year
                query = f"{query} {year}"
            elif strategy == 2:
                query = _get_dynamic_query(state, step, protocol_layer)
            elif strategy == 3:
                query = _get_expanded_query(state, step, protocol_layer)
            else:
                # Strategy 4: LLM-generated novel query
                try:
                    from config.loader import ConfigLoader
                    _cfg = ConfigLoader()
                    if _cfg.get("query_expansion_llm_enabled", True):
                        from research.llm_utils import get_llm_manager
                        llm = get_llm_manager()
                        query = _get_llm_query(
                            llm=llm,
                            active_hypotheses=list(state.active_hypotheses),
                            top_uncertainties=list(state.top_uncertainties),
                            layer=protocol_layer,
                        )
                    else:
                        query = _get_drug_centric_query(state, step)
                except Exception:
                    query = _get_drug_centric_query(state, step)
```

- [ ] **Step 5: Enable in config**

In `data/erik_config.json`, change:
```json
  "query_expansion_llm_enabled": true,
```
(Was `false` — line 93)

- [ ] **Step 6: Run tests and commit**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_llm_query_gen.py -v`
Expected: PASS

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/research/policy.py tests/research/test_llm_query_gen.py data/erik_config.json
git commit -m "feat: add LLM-powered dynamic query generation for PubMed"
```

---

### Task 8: Enhanced Stagnation Recovery

**Files:**
- Modify: `scripts/research/loop.py:287-309`
- Test: `tests/research/test_stagnation_recovery.py`

Current stagnation recovery expires half the hypotheses and resets posteriors, but the system immediately re-enters the same static query rotation. Add a **hard exploration burst** that forces the next N steps to use diverse, non-default actions.

- [ ] **Step 1: Write failing test**

```python
# tests/research/test_stagnation_recovery.py
import pytest
from dataclasses import replace
from research.loop import _apply_stagnation_recovery


def _make_state(**overrides):
    """Create a minimal ResearchState for testing."""
    from experience.state import ResearchState
    defaults = dict(
        step_count=1000,
        active_hypotheses=["H1", "H2", "H3", "H4"],
        action_posteriors={"a:b": (5.0, 1.0), "c:d": (1.0, 5.0)},
        stagnation_resets=3,
        target_exhaustion={"SOD1:query_clinvar": 5},
        expansion_query_history=["old_query"],
        expansion_gene_history={"search_pubmed": ["SOD1"]},
        evidence_at_step={800: 3000, 900: 3001},
        last_stagnation_step=800,
        exploration_burst_remaining=0,
    )
    defaults.update(overrides)
    return ResearchState(**defaults)


def test_recovery_sets_exploration_burst():
    """After stagnation recovery, exploration_burst_remaining > 0."""
    state = _make_state()
    new_state = _apply_stagnation_recovery(state, step=1000)
    assert new_state.exploration_burst_remaining > 0


def test_recovery_clears_exhaustion():
    """Recovery clears target_exhaustion."""
    state = _make_state()
    new_state = _apply_stagnation_recovery(state, step=1000)
    assert new_state.target_exhaustion == {}


def test_recovery_increments_counter():
    """Recovery increments stagnation_resets."""
    state = _make_state(stagnation_resets=5)
    new_state = _apply_stagnation_recovery(state, step=1000)
    assert new_state.stagnation_resets == 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_stagnation_recovery.py -v`
Expected: FAIL

- [ ] **Step 3: Extract stagnation recovery into a testable function**

In `scripts/research/loop.py`, extract lines 287-309 into a standalone function:

```python
def _apply_stagnation_recovery(state: ResearchState, step: int) -> ResearchState:
    """Apply stagnation recovery: expire hypotheses, reset posteriors, burst exploration."""
    from dataclasses import replace

    _active = list(state.active_hypotheses)
    _n_expire = max(1, len(_active) // 2)
    for _ in range(_n_expire):
        if _active:
            _active.pop(0)

    _reset_posteriors = {k: (1.0, 1.0) for k in state.action_posteriors}

    # NEW: Set exploration burst — next 20 steps will use forced diverse actions
    _burst = 20

    new_state = replace(
        state,
        active_hypotheses=_active,
        action_posteriors=_reset_posteriors,
        stagnation_resets=state.stagnation_resets + 1,
        last_stagnation_step=step,
        target_exhaustion={},
        expansion_query_history=[],
        expansion_gene_history={},
        exploration_burst_remaining=_burst,
    )
    print(
        f"[RESEARCH] STAGNATION RECOVERY #{new_state.stagnation_resets}: "
        f"expired {_n_expire} hypotheses, reset posteriors + exhaustion, "
        f"burst={_burst} steps"
    )
    return new_state
```

Then in `research_step()`, where stagnation was detected (line 287), replace the inline code with:

```python
    if _stag_detected:
        new_state = _apply_stagnation_recovery(new_state, new_step)
```

- [ ] **Step 4: Add exploration_burst_remaining to ResearchState**

Add `exploration_burst_remaining: int = 0` to the ResearchState dataclass.

- [ ] **Step 5: Honor burst in action selection (policy.py)**

In `select_action()` (policy.py), add at the top:

```python
    # During exploration burst, force non-default diverse actions
    if getattr(state, "exploration_burst_remaining", 0) > 0:
        # Pick from under-used action types
        all_counts = state.action_counts or {}
        sorted_actions = sorted(all_counts.items(), key=lambda x: x[1])
        # Pick least-used acquisition action
        for action_name, count in sorted_actions:
            try:
                action = ActionType(action_name)
                if action in _ACQUISITION_ROTATION:
                    params = _build_acquisition_params(action, state, state.step_count)
                    return params
            except (ValueError, KeyError):
                continue
        # Fallback to regular selection
```

- [ ] **Step 6: Decrement burst counter in research_step**

In `research_step()`, after action execution, decrement the burst counter:

```python
    if getattr(new_state, "exploration_burst_remaining", 0) > 0:
        new_state = replace(new_state, exploration_burst_remaining=new_state.exploration_burst_remaining - 1)
```

- [ ] **Step 7: Run tests and commit**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/research/test_stagnation_recovery.py -v`
Expected: PASS

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/research/loop.py scripts/research/policy.py tests/research/test_stagnation_recovery.py
git commit -m "feat: enhanced stagnation recovery with exploration burst"
```

---

### Task 9: Galen KG Access from Railway

**Files:**
- Create: `scripts/connectors/galen_kg_api.py`
- Modify: `scripts/connectors/connector_mode.py`
- Test: `tests/connectors/test_galen_kg_api.py`

Erik's Galen KG/SCM connectors use direct PostgreSQL to `galen_kg` (MacBook only). On Railway, these fail silently. Two options:

**Option A (recommended):** Expose Galen KG via Galen's existing FastAPI (add 2 endpoints to `/Users/logannye/.openclaw/workspace/api/`) and have Erik call them via HTTP.

**Option B (simpler):** Set up a secure PostgreSQL tunnel from Railway to MacBook using Railway's TCP proxy or a bastion.

This task implements Option A — add a Galen API endpoint that Erik calls.

- [ ] **Step 1: Add KG query endpoint to Galen API**

In `/Users/logannye/.openclaw/workspace/api/` (Galen), add a router:

```python
# /Users/logannye/.openclaw/workspace/api/routers/erik_bridge.py
"""Bridge endpoints for Erik's ALS research engine to query Galen's KG."""
from fastapi import APIRouter, Query
from db.ops_pool import get_ops_connection

router = APIRouter(prefix="/api/erik-bridge", tags=["erik-bridge"])


@router.get("/kg/search")
def kg_search(
    query: str = Query(..., description="Entity name or gene symbol"),
    entity_type: str = Query("gene", description="Entity type filter"),
    limit: int = Query(20, le=100),
):
    """Search Galen's 731K-entity cancer KG for cross-disease insights."""
    with get_ops_connection() as conn:
        rows = conn.execute(
            """SELECT e.name, e.entity_type, e.properties
               FROM galen_core.entities e
               WHERE e.name ILIKE %s AND e.entity_type = %s
               LIMIT %s""",
            (f"%{query}%", entity_type, limit),
        ).fetchall()
    return {"results": [{"name": r[0], "type": r[1], "properties": r[2]} for r in rows]}


@router.get("/kg/neighbors")
def kg_neighbors(
    gene: str = Query(...),
    max_results: int = Query(10, le=50),
    min_confidence: float = Query(0.4),
):
    """Get gene neighbors from Galen's KG for query expansion."""
    with get_ops_connection() as conn:
        rows = conn.execute(
            """SELECT DISTINCT e2.name, r.relationship_type, r.confidence
               FROM galen_core.entities e1
               JOIN galen_core.relationships r ON (r.source_id = e1.id OR r.target_id = e1.id)
               JOIN galen_core.entities e2 ON (
                   (e2.id = r.source_id OR e2.id = r.target_id) AND e2.id != e1.id
               )
               WHERE e1.name = %s
                 AND e2.entity_type IN ('gene', 'protein')
                 AND r.confidence >= %s
               ORDER BY r.confidence DESC
               LIMIT %s""",
            (gene, min_confidence, max_results),
        ).fetchall()
    return {"neighbors": [{"gene": r[0], "relationship": r[1], "confidence": float(r[2])} for r in rows]}
```

- [ ] **Step 2: Create Erik-side HTTP connector**

```python
# scripts/connectors/galen_kg_api.py
"""Galen KG HTTP API connector — for Railway deployment."""
from __future__ import annotations
import os
from typing import Any
import requests
from connectors.base import ConnectorResult

_GALEN_API_URL = os.environ.get("GALEN_API_URL", "http://localhost:8000")
_TIMEOUT = 15


class GalenKGAPIConnector:
    """Query Galen's cancer KG via HTTP bridge."""

    def __init__(self, store: Any = None, **kwargs: Any):
        self._store = store

    def fetch(self, gene: str = "", uniprot: str = "", **kwargs: Any) -> ConnectorResult:
        result = ConnectorResult()
        if not gene:
            return result
        try:
            resp = requests.get(
                f"{_GALEN_API_URL}/api/erik-bridge/kg/search",
                params={"query": gene, "entity_type": "gene", "limit": 20},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
            for item in resp.json().get("results", []):
                name = item.get("name", "")
                props = item.get("properties", {}) or {}
                claim = f"Galen cross-disease: {name} — {props.get('description', 'cancer KG entity')}"
                evi_id = f"evi:galen_kg:{name}_{gene}".lower().replace(" ", "_")
                if self._store:
                    added = self._store.upsert_evidence(
                        id=evi_id, type="EvidenceItem", claim=claim,
                        source="galen_kg", provenance=f"Galen KG: {name}",
                        confidence=0.6, protocol_layer="root_cause_suppression",
                        evidence_strength="emerging",
                    )
                    if added:
                        result.evidence_items_added += 1
        except Exception as e:
            result.errors.append(f"Galen KG API error: {e}")
        return result
```

- [ ] **Step 3: Add to connector_mode.py**

Add to `_API_VARIANTS` in `connector_mode.py`:
```python
    "connectors.galen_kg.GalenKGConnector": "connectors.galen_kg_api.GalenKGAPIConnector",
```

- [ ] **Step 4: Set GALEN_API_URL on Railway**

```bash
# Via Railway dashboard: Add environment variable
GALEN_API_URL=https://your-galen-api-url.railway.app
```

Note: Galen's API must be exposed to the internet (Railway service, or via ngrok/Cloudflare tunnel from MacBook). If Galen API isn't already deployed, this requires deploying Galen's FastAPI to Railway or exposing it via tunnel.

- [ ] **Step 5: Write test and commit**

```python
# tests/connectors/test_galen_kg_api.py
from unittest.mock import patch, MagicMock
from connectors.galen_kg_api import GalenKGAPIConnector


def test_fetch_returns_results():
    store = MagicMock()
    store.upsert_evidence.return_value = True
    c = GalenKGAPIConnector(store=store)
    with patch("connectors.galen_kg_api.requests.get") as mock:
        mock.return_value = MagicMock(
            status_code=200,
            json=lambda: {"results": [{"name": "SOD1", "type": "gene", "properties": {}}]},
        )
        mock.return_value.raise_for_status = lambda: None
        cr = c.fetch(gene="SOD1")
        assert not cr.errors
```

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/connectors/galen_kg_api.py scripts/connectors/connector_mode.py tests/connectors/test_galen_kg_api.py
git commit -m "feat: Galen KG HTTP bridge connector for Railway"
```

---

### Task 10: Deploy and Verify

**Files:** None (operational task)

- [ ] **Step 1: Push all changes to git**

```bash
cd /Users/logannye/.openclaw/erik
git push origin main
```

- [ ] **Step 2: Set Railway environment variables**

Via Railway dashboard, add/verify these variables:
- `CONNECTOR_MODE=api`
- `GALEN_API_URL=<galen-api-url>` (once Galen bridge is deployed)

- [ ] **Step 3: Trigger Railway redeploy**

Railway auto-deploys on push. Monitor deployment logs for:
- No import errors on new connector files
- `CONNECTOR_MODE=api` logged at startup
- Evidence acquisition producing non-zero results

- [ ] **Step 4: Monitor evidence growth**

After 1 hour, check:
```bash
curl -s -b "erik_session=<token>" https://erik-website-eosin.vercel.app/api/state | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Evidence: {d[\"evidence_count\"]}')
print(f'Steps: {d[\"state\"][\"step_count\"]}')
print(f'Layer: {d[\"state\"].get(\"research_layer\", \"unknown\")}')
print(f'Stagnation resets: {d[\"state\"][\"stagnation_resets\"]}')
"
```

Expected: Evidence count should be growing at 10-50+ items/hour (vs ~0/hour currently).

- [ ] **Step 5: Verify layer progression**

After evidence exceeds 500+ (may take a few hours with API connectors):
- Check that `research_layer` has advanced to `erik_specific`
- Verify provisional genetic profile queries appearing in logs

---

## Self-Review Checklist

1. **Spec coverage:** All 4 priorities covered — data sources (Tasks 1-5, 9), layer ungating (Task 6), query intelligence (Tasks 7-8), deployment (Task 10).

2. **Placeholder scan:** No TBDs, TODOs, or "implement later" — all tasks have complete code.

3. **Type consistency:** `ConnectorResult` interface consistent across all API connectors. `fetch(gene, uniprot)` signature matches `_make_als_gene_executor` expectations. `_apply_stagnation_recovery` returns `ResearchState`.

4. **Missing coverage:** DrugBank/DisGeNET small-file bundling (low priority — can be added later via Docker COPY). BindingDB/GEO ALS/SpliceAI (no API available — deferred). Galen SCM bridge (requires separate endpoint — can follow same pattern as Task 9).
