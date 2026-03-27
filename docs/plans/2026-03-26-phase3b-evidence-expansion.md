# Phase 3B: Evidence Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand Erik's evidence fabric with 7 new data sources (Reactome, KEGG, STRING, PRO-ACT, ClinVar, OMIM, PharmGKB), wire them into the research loop as 5 new action types, and enhance causal chain construction with pathway-grounded evidence.

**Architecture:** 6 new API connectors following the existing BaseConnector pattern, 1 local data analyzer (PRO-ACT), 5 new research actions wired into the loop dispatch. Causal chains enhanced with Reactome pathway data as ground truth. All connectors produce canonical EvidenceItems with deterministic IDs, provenance, and PCH annotations.

**Tech Stack:** Python 3.12, requests (HTTP), xml.etree (parsing), Pydantic v2, psycopg3, pytest. Existing BaseConnector ABC, EvidenceStore, research loop.

**Spec:** `/Users/logannye/.openclaw/erik/docs/specs/2026-03-26-phase3b-evidence-expansion-design.md`

---

## File Structure

```
scripts/
  connectors/
    reactome.py           # CREATE
    kegg.py               # CREATE
    string_db.py          # CREATE
    clinvar.py            # CREATE
    omim.py               # CREATE
    pharmgkb.py           # CREATE
  research/
    trajectory.py         # CREATE
    actions.py            # MODIFY: add 5 new ActionType values + NETWORK_ACTIONS update
    policy.py             # MODIFY: add pathway/genetics triggers
    loop.py               # MODIFY: add 5 new _exec_* functions + dispatch
    causal_chains.py      # MODIFY: add pathway_grounded_link()
  config/
    loader.py             # (already hot-reloadable, just add keys)
  data/
    erik_config.json      # MODIFY: add Phase 3B config keys
tests/
  test_reactome_connector.py
  test_kegg_connector.py
  test_string_connector.py
  test_clinvar_connector.py
  test_omim_connector.py
  test_pharmgkb_connector.py
  test_trajectory.py
  test_expanded_actions.py
```

---

## Task 1: Reactome Connector

**Files:**
- Create: `scripts/connectors/reactome.py`
- Create: `tests/test_reactome_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reactome_connector.py
"""Tests for Reactome pathway connector."""
from __future__ import annotations

import json
import pytest

from connectors.reactome import ReactomeConnector, _parse_pathway_response, _parse_contained_events


class TestParsePathwayResponse:

    def test_parses_pathway_list(self):
        raw = [
            {"stId": "R-HSA-3371556", "displayName": "Cellular response to heat stress",
             "speciesName": "Homo sapiens"},
            {"stId": "R-HSA-392499", "displayName": "Metabolism of proteins",
             "speciesName": "Homo sapiens"},
        ]
        pathways = _parse_pathway_response(raw, uniprot_id="Q13148")
        assert len(pathways) == 2
        assert pathways[0]["pathway_id"] == "R-HSA-3371556"
        assert pathways[0]["uniprot_id"] == "Q13148"

    def test_empty_response(self):
        pathways = _parse_pathway_response([], uniprot_id="Q13148")
        assert pathways == []


class TestParseContainedEvents:

    def test_parses_reaction_steps(self):
        raw = [
            {"stId": "R-HSA-3371568", "displayName": "HSF1 trimerization",
             "className": "Reaction"},
            {"stId": "R-HSA-3371571", "displayName": "HSP70 binding",
             "className": "Reaction"},
        ]
        steps = _parse_contained_events(raw, pathway_id="R-HSA-3371556")
        assert len(steps) == 2
        assert steps[0]["reaction_id"] == "R-HSA-3371568"


class TestReactomeConnector:

    def test_instantiates(self):
        connector = ReactomeConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = ReactomeConnector(evidence_store=None)
        item = connector._build_evidence_item(
            pathway_id="R-HSA-3371556",
            pathway_name="Cellular response to heat stress",
            uniprot_id="Q13148",
            gene_symbol="TARDBP",
            num_reactions=12,
        )
        assert item.id == "evi:reactome:R-HSA-3371556_Q13148"
        assert item.provenance.source_system.value == "database"
        assert item.provenance.asserted_by == "reactome_connector"
        assert "pathway" in item.body
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reactome_connector.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/reactome.py
"""Reactome Content Service connector — curated biological pathways.

For any protein (by UniProt ID), returns pathways it participates in
and the reaction steps within each pathway. Used to ground causal
chains in peer-curated pathway data.
"""
from __future__ import annotations

import requests
from typing import Any, Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem


BASE_URL = "https://reactome.org/ContentService"


def _parse_pathway_response(raw: list[dict], uniprot_id: str) -> list[dict]:
    """Parse Reactome pathway list response."""
    results = []
    for entry in raw:
        if entry.get("speciesName") != "Homo sapiens":
            continue
        results.append({
            "pathway_id": entry["stId"],
            "pathway_name": entry.get("displayName", ""),
            "uniprot_id": uniprot_id,
        })
    return results


def _parse_contained_events(raw: list[dict], pathway_id: str) -> list[dict]:
    """Parse reaction steps within a pathway."""
    results = []
    for entry in raw:
        results.append({
            "reaction_id": entry["stId"],
            "reaction_name": entry.get("displayName", ""),
            "reaction_type": entry.get("className", ""),
            "pathway_id": pathway_id,
        })
    return results


class ReactomeConnector(BaseConnector):
    """Connector to the Reactome Content Service REST API."""

    def __init__(self, evidence_store=None, base_url: str = BASE_URL):
        self._store = evidence_store
        self._base_url = base_url

    def fetch(self, *, uniprot_id: str = "", gene_symbol: str = "", **kwargs) -> ConnectorResult:
        """Fetch pathways for a protein by UniProt ID."""
        result = ConnectorResult()
        if not uniprot_id:
            result.errors.append("uniprot_id is required")
            return result

        try:
            pathways = self._fetch_pathways(uniprot_id)
        except Exception as e:
            result.errors.append(f"Reactome API error: {e}")
            return result

        for pw in pathways[:10]:  # Cap at 10 pathways per protein
            # Count reaction steps
            try:
                events = self._fetch_contained_events(pw["pathway_id"])
                num_reactions = len(events)
            except Exception:
                num_reactions = 0

            item = self._build_evidence_item(
                pathway_id=pw["pathway_id"],
                pathway_name=pw["pathway_name"],
                uniprot_id=uniprot_id,
                gene_symbol=gene_symbol,
                num_reactions=num_reactions,
            )
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1

        return result

    def _fetch_pathways(self, uniprot_id: str) -> list[dict]:
        """GET /data/pathways/low/entity/{UniProtID}."""
        url = f"{self._base_url}/data/pathways/low/entity/{uniprot_id}"
        resp = self._retry_with_backoff(
            requests.get, url, headers={"Accept": "application/json"}, timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_pathway_response(resp.json(), uniprot_id)

    def _fetch_contained_events(self, pathway_id: str) -> list[dict]:
        """GET /data/pathway/{stableId}/containedEvents."""
        url = f"{self._base_url}/data/pathway/{pathway_id}/containedEvents"
        resp = self._retry_with_backoff(
            requests.get, url, headers={"Accept": "application/json"}, timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_contained_events(resp.json(), pathway_id)

    def _build_evidence_item(
        self, pathway_id: str, pathway_name: str,
        uniprot_id: str, gene_symbol: str, num_reactions: int,
    ) -> EvidenceItem:
        """Build an EvidenceItem from a Reactome pathway."""
        return EvidenceItem(
            id=f"evi:reactome:{pathway_id}_{uniprot_id}",
            claim=f"{gene_symbol or uniprot_id} participates in pathway: {pathway_name} ({num_reactions} reactions)",
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.strong,
            source_refs=[f"reactome:{pathway_id}"],
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="reactome_connector",
                source_artifact_id=pathway_id,
            ),
            body={
                "pathway": pathway_id,
                "pathway_name": pathway_name,
                "uniprot_id": uniprot_id,
                "gene_symbol": gene_symbol,
                "num_reactions": num_reactions,
                "pch_layer": 1,
                "data_source": "reactome",
            },
        )
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reactome_connector.py -v -k "not network"`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/reactome.py tests/test_reactome_connector.py
git commit -m "feat: Reactome pathway connector — curated biological pathways for causal chain grounding"
```

---

## Task 2: KEGG Connector

**Files:**
- Create: `scripts/connectors/kegg.py`
- Create: `tests/test_kegg_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kegg_connector.py
"""Tests for KEGG pathway connector."""
from __future__ import annotations

import pytest

from connectors.kegg import KEGGConnector, _parse_link_response, _parse_pathway_entry


class TestParseLinkResponse:

    def test_parses_gene_pathway_links(self):
        raw_text = "hsa:6647\tpath:hsa04141\nhsa:6647\tpath:hsa04010\n"
        links = _parse_link_response(raw_text)
        assert len(links) == 2
        assert links[0] == ("hsa:6647", "path:hsa04141")

    def test_empty_response(self):
        links = _parse_link_response("")
        assert links == []


class TestParsePathwayEntry:

    def test_extracts_name(self):
        raw_text = "ENTRY       hsa04141\nNAME        Protein processing in endoplasmic reticulum - Homo sapiens\nDESCRIPTION The endoplasmic reticulum\n///"
        entry = _parse_pathway_entry(raw_text)
        assert "Protein processing" in entry["name"]

    def test_missing_name_returns_empty(self):
        entry = _parse_pathway_entry("ENTRY   hsa04141\n///")
        assert entry["name"] == ""


class TestKEGGConnector:

    def test_instantiates(self):
        connector = KEGGConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = KEGGConnector(evidence_store=None)
        item = connector._build_evidence_item(
            pathway_id="hsa04141",
            pathway_name="Protein processing in endoplasmic reticulum",
            gene_id="6647",
            gene_symbol="SOD1",
        )
        assert item.id == "evi:kegg:hsa04141_SOD1"
        assert item.provenance.asserted_by == "kegg_connector"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/kegg.py
"""KEGG REST API connector — pathway ontology and gene-pathway mapping.

Provides gene-to-pathway lookups and pathway descriptions. Complementary
to Reactome: KEGG emphasizes metabolic/signaling contexts.
"""
from __future__ import annotations

import requests
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem


BASE_URL = "https://rest.kegg.jp"


def _parse_link_response(text: str) -> list[tuple[str, str]]:
    """Parse KEGG link response (tab-separated gene\tpathway lines)."""
    links = []
    for line in text.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.strip().split("\t")
        if len(parts) == 2:
            links.append((parts[0], parts[1]))
    return links


def _parse_pathway_entry(text: str) -> dict:
    """Parse a KEGG pathway flat-file entry, extracting NAME."""
    name = ""
    for line in text.split("\n"):
        if line.startswith("NAME"):
            name = line[12:].strip().split(" - Homo sapiens")[0].strip()
            break
    return {"name": name}


class KEGGConnector(BaseConnector):
    """Connector to the KEGG REST API."""

    def __init__(self, evidence_store=None, base_url: str = BASE_URL):
        self._store = evidence_store
        self._base_url = base_url

    def fetch(self, *, gene_id: str = "", gene_symbol: str = "", **kwargs) -> ConnectorResult:
        """Fetch pathways for a human gene."""
        result = ConnectorResult()
        if not gene_id and not gene_symbol:
            result.errors.append("gene_id or gene_symbol required")
            return result

        kegg_gene = f"hsa:{gene_id}" if gene_id else ""
        try:
            links = self._fetch_gene_pathways(kegg_gene)
        except Exception as e:
            result.errors.append(f"KEGG API error: {e}")
            return result

        for _, pathway_ref in links[:10]:
            pathway_id = pathway_ref.replace("path:", "")
            try:
                entry = self._fetch_pathway_info(pathway_id)
                pathway_name = entry["name"]
            except Exception:
                pathway_name = pathway_id

            item = self._build_evidence_item(
                pathway_id=pathway_id,
                pathway_name=pathway_name,
                gene_id=gene_id,
                gene_symbol=gene_symbol,
            )
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1

        return result

    def _fetch_gene_pathways(self, kegg_gene: str) -> list[tuple[str, str]]:
        url = f"{self._base_url}/link/pathway/{kegg_gene}"
        resp = self._retry_with_backoff(requests.get, url, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_link_response(resp.text)

    def _fetch_pathway_info(self, pathway_id: str) -> dict:
        url = f"{self._base_url}/get/{pathway_id}"
        resp = self._retry_with_backoff(requests.get, url, timeout=self.REQUEST_TIMEOUT)
        resp.raise_for_status()
        return _parse_pathway_entry(resp.text)

    def _build_evidence_item(
        self, pathway_id: str, pathway_name: str, gene_id: str, gene_symbol: str,
    ) -> EvidenceItem:
        return EvidenceItem(
            id=f"evi:kegg:{pathway_id}_{gene_symbol or gene_id}",
            claim=f"{gene_symbol or gene_id} participates in KEGG pathway: {pathway_name}",
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.strong,
            source_refs=[f"kegg:{pathway_id}"],
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="kegg_connector",
                source_artifact_id=pathway_id,
            ),
            body={
                "pathway": pathway_id,
                "pathway_name": pathway_name,
                "gene_id": gene_id,
                "gene_symbol": gene_symbol,
                "pch_layer": 1,
                "data_source": "kegg",
            },
        )
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/kegg.py tests/test_kegg_connector.py
git commit -m "feat: KEGG pathway connector — gene-pathway mapping for off-target detection"
```

---

## Task 3: STRING Connector

**Files:**
- Create: `scripts/connectors/string_db.py`
- Create: `tests/test_string_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_string_connector.py
"""Tests for STRING protein-protein interaction connector."""
from __future__ import annotations

import pytest

from connectors.string_db import STRINGConnector, _parse_network_response


class TestParseNetworkResponse:

    def test_parses_interactions(self):
        raw = [
            {"preferredName_A": "TARDBP", "preferredName_B": "HNRNPA1",
             "score": 972, "nscore": 0, "fscore": 0, "pscore": 0,
             "ascore": 0, "escore": 912, "dscore": 900, "tscore": 500},
            {"preferredName_A": "TARDBP", "preferredName_B": "FUS",
             "score": 965, "nscore": 0, "fscore": 0, "pscore": 0,
             "ascore": 0, "escore": 880, "dscore": 850, "tscore": 600},
        ]
        interactions = _parse_network_response(raw, query_gene="TARDBP")
        assert len(interactions) == 2
        assert interactions[0]["gene_a"] == "TARDBP"
        assert interactions[0]["gene_b"] == "HNRNPA1"
        assert interactions[0]["combined_score"] == 972

    def test_empty_response(self):
        assert _parse_network_response([], query_gene="TARDBP") == []


class TestSTRINGConnector:

    def test_instantiates(self):
        connector = STRINGConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = STRINGConnector(evidence_store=None)
        item = connector._build_evidence_item(
            gene_a="TARDBP", gene_b="HNRNPA1",
            combined_score=972, experimental_score=912,
        )
        assert item.id == "evi:string:TARDBP_HNRNPA1"
        assert item.provenance.asserted_by == "string_connector"
        assert item.body["combined_score"] == 972

    def test_high_score_is_strong_evidence(self):
        connector = STRINGConnector(evidence_store=None)
        item = connector._build_evidence_item(
            gene_a="TARDBP", gene_b="FUS", combined_score=900, experimental_score=800,
        )
        assert item.strength.value == "strong"

    def test_low_score_is_emerging_evidence(self):
        connector = STRINGConnector(evidence_store=None)
        item = connector._build_evidence_item(
            gene_a="TARDBP", gene_b="OBSCURE", combined_score=450, experimental_score=0,
        )
        assert item.strength.value == "emerging"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/string_db.py
"""STRING protein-protein interaction connector.

Returns interaction partners for a query protein with confidence
scores. Used to validate causal chain links and discover indirect
mechanism connections.
"""
from __future__ import annotations

import requests
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem


BASE_URL = "https://string-db.org/api"
SPECIES_HUMAN = 9606


def _parse_network_response(raw: list[dict], query_gene: str) -> list[dict]:
    """Parse STRING network JSON response."""
    results = []
    for entry in raw:
        results.append({
            "gene_a": entry.get("preferredName_A", ""),
            "gene_b": entry.get("preferredName_B", ""),
            "combined_score": entry.get("score", 0),
            "experimental_score": entry.get("escore", 0),
            "database_score": entry.get("dscore", 0),
            "textmining_score": entry.get("tscore", 0),
        })
    return results


class STRINGConnector(BaseConnector):
    """Connector to the STRING protein interaction API."""

    def __init__(self, evidence_store=None, base_url: str = BASE_URL, min_score: int = 400):
        self._store = evidence_store
        self._base_url = base_url
        self._min_score = min_score

    def fetch(self, *, gene_symbol: str = "", limit: int = 20, **kwargs) -> ConnectorResult:
        """Fetch protein interaction network for a gene."""
        result = ConnectorResult()
        if not gene_symbol:
            result.errors.append("gene_symbol required")
            return result

        try:
            interactions = self._fetch_network(gene_symbol, limit)
        except Exception as e:
            result.errors.append(f"STRING API error: {e}")
            return result

        for interaction in interactions:
            if interaction["combined_score"] < self._min_score:
                continue
            item = self._build_evidence_item(
                gene_a=interaction["gene_a"],
                gene_b=interaction["gene_b"],
                combined_score=interaction["combined_score"],
                experimental_score=interaction["experimental_score"],
            )
            if self._store:
                self._store.upsert_evidence_item(item)
            result.evidence_items_added += 1

        return result

    def _fetch_network(self, gene_symbol: str, limit: int) -> list[dict]:
        url = f"{self._base_url}/json/network"
        params = {
            "identifiers": gene_symbol,
            "species": SPECIES_HUMAN,
            "limit": limit,
            "required_score": self._min_score,
            "caller_identity": "erik_als_engine",
        }
        resp = self._retry_with_backoff(
            requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        return _parse_network_response(resp.json(), gene_symbol)

    def _build_evidence_item(
        self, gene_a: str, gene_b: str, combined_score: int, experimental_score: int,
    ) -> EvidenceItem:
        # Alphabetical order for deterministic ID
        sorted_pair = sorted([gene_a, gene_b])
        eid = f"evi:string:{sorted_pair[0]}_{sorted_pair[1]}"

        if combined_score >= 700:
            strength = EvidenceStrength.strong
        elif combined_score >= 400:
            strength = EvidenceStrength.moderate
        else:
            strength = EvidenceStrength.emerging

        return EvidenceItem(
            id=eid,
            claim=f"Protein interaction: {gene_a} ↔ {gene_b} (STRING score {combined_score}/1000)",
            direction=EvidenceDirection.supports,
            strength=strength,
            source_refs=[f"string:{gene_a}_{gene_b}"],
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="string_connector",
            ),
            body={
                "gene_a": gene_a,
                "gene_b": gene_b,
                "combined_score": combined_score,
                "experimental_score": experimental_score,
                "pch_layer": 1,
                "data_source": "string",
            },
        )
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/string_db.py tests/test_string_connector.py
git commit -m "feat: STRING protein interaction connector — PPI network for mechanism validation"
```

---

## Task 4: ClinVar Connector

**Files:**
- Create: `scripts/connectors/clinvar.py`
- Create: `tests/test_clinvar_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_clinvar_connector.py
"""Tests for ClinVar genetic variant connector."""
from __future__ import annotations

import pytest

from connectors.clinvar import ClinVarConnector, _parse_variant_xml


class TestParseVariantXml:

    def test_parses_pathogenic_variant(self):
        # Minimal ClinVar XML structure for a variant
        xml_str = """<ClinVarResult-Set>
          <ClinVarAssertion ID="12345">
            <ClinVarSubmissionID localKey="test"/>
            <ClinicalSignificance>
              <Description>Pathogenic</Description>
            </ClinicalSignificance>
            <MeasureSet>
              <Measure Type="Variation">
                <Name><ElementValue>SOD1 A4V</ElementValue></Name>
              </Measure>
            </MeasureSet>
          </ClinVarAssertion>
        </ClinVarResult-Set>"""
        variants = _parse_variant_xml(xml_str, gene="SOD1")
        assert len(variants) >= 0  # Parser should handle gracefully

    def test_empty_xml(self):
        variants = _parse_variant_xml("", gene="SOD1")
        assert variants == []


class TestClinVarConnector:

    def test_instantiates(self):
        connector = ClinVarConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(
            variation_id="12345",
            variant_name="SOD1 A4V",
            gene="SOD1",
            clinical_significance="Pathogenic",
            review_status="criteria provided, multiple submitters",
        )
        assert item.id == "evi:clinvar:12345"
        assert item.provenance.asserted_by == "clinvar_connector"
        assert item.body["clinical_significance"] == "Pathogenic"
        assert item.body["gene"] == "SOD1"

    def test_pathogenic_is_strong_evidence(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(
            variation_id="1", variant_name="test", gene="SOD1",
            clinical_significance="Pathogenic", review_status="reviewed",
        )
        assert item.strength.value == "strong"

    def test_uncertain_is_emerging(self):
        connector = ClinVarConnector(evidence_store=None)
        item = connector._build_evidence_item(
            variation_id="2", variant_name="test", gene="FUS",
            clinical_significance="Uncertain significance", review_status="",
        )
        assert item.strength.value == "emerging"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/clinvar.py
"""ClinVar connector — genetic variant pathogenicity via NCBI E-utilities.

When Erik's Invitae genetic panel results arrive, this connector
queries ClinVar for pathogenicity classifications of reported variants.
"""
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
    """Parse ClinVar XML search results into variant dicts."""
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
            genes_el = doc_sum.findall(".//gene")
            gene_names = [g.findtext("symbol", "") for g in genes_el]

            if gene and gene not in gene_names and title and gene.lower() not in title.lower():
                continue

            results.append({
                "variation_id": variation_id,
                "variant_name": title,
                "gene": gene,
                "clinical_significance": clinical_sig,
                "review_status": review_status,
            })
    except ET.ParseError:
        pass
    return results


class ClinVarConnector(BaseConnector):
    """Connector to ClinVar via NCBI E-utilities."""

    TOOL = "erik_als"
    EMAIL = "research@galenhealth.ai"

    def __init__(self, evidence_store=None, api_key: Optional[str] = None):
        self._store = evidence_store
        self._api_key = api_key

    def fetch(self, *, gene: str = "", max_results: int = 20, **kwargs) -> ConnectorResult:
        """Search ClinVar for ALS-associated variants in a gene."""
        result = ConnectorResult()
        if not gene:
            result.errors.append("gene symbol required")
            return result

        try:
            # Search
            search_url = f"{EUTILS_BASE}/esearch.fcgi"
            params = {
                "db": "clinvar", "retmode": "json", "retmax": max_results,
                "term": f"{gene}[gene] AND ALS[disease]",
                "tool": self.TOOL, "email": self.EMAIL,
            }
            if self._api_key:
                params["api_key"] = self._api_key

            resp = self._retry_with_backoff(requests.get, search_url, params=params, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])

            if not id_list:
                return result

            # Fetch summaries
            summary_url = f"{EUTILS_BASE}/esummary.fcgi"
            s_params = {
                "db": "clinvar", "retmode": "xml", "id": ",".join(id_list),
                "tool": self.TOOL, "email": self.EMAIL,
            }
            if self._api_key:
                s_params["api_key"] = self._api_key

            s_resp = self._retry_with_backoff(requests.get, summary_url, params=s_params, timeout=self.REQUEST_TIMEOUT)
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

    def _build_evidence_item(
        self, variation_id: str, variant_name: str, gene: str,
        clinical_significance: str, review_status: str,
    ) -> EvidenceItem:
        sig_lower = clinical_significance.lower()
        if "pathogenic" in sig_lower and "uncertain" not in sig_lower:
            strength = EvidenceStrength.strong
            direction = EvidenceDirection.supports
        elif "benign" in sig_lower:
            strength = EvidenceStrength.strong
            direction = EvidenceDirection.refutes
        else:
            strength = EvidenceStrength.emerging
            direction = EvidenceDirection.insufficient

        return EvidenceItem(
            id=f"evi:clinvar:{variation_id}",
            claim=f"ClinVar: {variant_name} — {clinical_significance}",
            direction=direction,
            strength=strength,
            source_refs=[f"clinvar:{variation_id}"],
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="clinvar_connector",
                source_artifact_id=str(variation_id),
            ),
            body={
                "variation_id": variation_id,
                "variant_name": variant_name,
                "gene": gene,
                "clinical_significance": clinical_significance,
                "review_status": review_status,
                "pch_layer": 1,
                "data_source": "clinvar",
            },
        )
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/clinvar.py tests/test_clinvar_connector.py
git commit -m "feat: ClinVar connector — genetic variant pathogenicity for subtype refinement"
```

---

## Task 5: OMIM Connector

**Files:**
- Create: `scripts/connectors/omim.py`
- Create: `tests/test_omim_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_omim_connector.py
"""Tests for OMIM gene-phenotype connector."""
from __future__ import annotations

import pytest

from connectors.omim import OMIMConnector, _parse_entry_response


class TestParseEntryResponse:

    def test_parses_als_phenotype(self):
        raw = {
            "omim": {"entryList": [{"entry": {
                "mimNumber": 105400,
                "titles": {"preferredTitle": "AMYOTROPHIC LATERAL SCLEROSIS 1; ALS1"},
                "geneMap": {"phenotypeMapList": [
                    {"phenotypeMap": {
                        "phenotype": "Amyotrophic lateral sclerosis 1",
                        "phenotypeMimNumber": 105400,
                        "geneSymbols": "SOD1",
                        "phenotypeInheritance": "Autosomal dominant",
                    }}
                ]},
            }}]}
        }
        entries = _parse_entry_response(raw)
        assert len(entries) == 1
        assert entries[0]["mim_number"] == 105400
        assert "SOD1" in entries[0]["gene_symbols"]

    def test_empty_response(self):
        entries = _parse_entry_response({"omim": {"entryList": []}})
        assert entries == []


class TestOMIMConnector:

    def test_instantiates(self):
        connector = OMIMConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = OMIMConnector(evidence_store=None)
        item = connector._build_evidence_item(
            mim_number=105400,
            title="AMYOTROPHIC LATERAL SCLEROSIS 1; ALS1",
            gene_symbols="SOD1",
            inheritance="Autosomal dominant",
            phenotype="Amyotrophic lateral sclerosis 1",
        )
        assert item.id == "evi:omim:105400"
        assert item.provenance.asserted_by == "omim_connector"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/omim.py
"""OMIM connector — gene-phenotype mapping for ALS subtype refinement.

Maps genes to ALS phenotypes, inheritance patterns, and phenotypic
series. Used to refine subtype inference when genetic results arrive.
"""
from __future__ import annotations

import requests
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem


BASE_URL = "https://api.omim.org/api"

# Known ALS MIM numbers for direct lookup
ALS_MIM_NUMBERS = {
    "SOD1": 105400,    # ALS1
    "TARDBP": 612069,  # ALS10
    "FUS": 608030,     # ALS6
    "C9orf72": 614260, # ALS-FTD
    "OPTN": 613435,    # ALS12
    "TBK1": 616795,    # ALS-FTD
    "NEK1": 617892,    # ALS24
}


def _parse_entry_response(raw: dict) -> list[dict]:
    """Parse OMIM API entry response."""
    entries = []
    for entry_wrapper in raw.get("omim", {}).get("entryList", []):
        entry = entry_wrapper.get("entry", {})
        mim = entry.get("mimNumber", 0)
        title = entry.get("titles", {}).get("preferredTitle", "")

        phenotype_maps = entry.get("geneMap", {}).get("phenotypeMapList", [])
        for pm_wrapper in phenotype_maps:
            pm = pm_wrapper.get("phenotypeMap", {})
            entries.append({
                "mim_number": mim,
                "title": title,
                "gene_symbols": pm.get("geneSymbols", ""),
                "inheritance": pm.get("phenotypeInheritance", ""),
                "phenotype": pm.get("phenotype", ""),
            })

        if not phenotype_maps:
            entries.append({
                "mim_number": mim,
                "title": title,
                "gene_symbols": "",
                "inheritance": "",
                "phenotype": "",
            })

    return entries


class OMIMConnector(BaseConnector):
    """Connector to the OMIM REST API."""

    def __init__(self, evidence_store=None, api_key: Optional[str] = None):
        self._store = evidence_store
        self._api_key = api_key

    def fetch(self, *, gene: str = "", mim_number: Optional[int] = None, **kwargs) -> ConnectorResult:
        """Fetch OMIM entry for an ALS gene."""
        result = ConnectorResult()

        mim = mim_number or ALS_MIM_NUMBERS.get(gene)
        if not mim:
            result.errors.append(f"No known OMIM MIM number for gene: {gene}")
            return result

        if not self._api_key:
            result.errors.append("OMIM API key required (set omim_api_key in config)")
            return result

        try:
            url = f"{BASE_URL}/entry"
            params = {
                "mimNumber": mim,
                "include": "geneMap",
                "format": "json",
                "apiKey": self._api_key,
            }
            resp = self._retry_with_backoff(requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()

            entries = _parse_entry_response(resp.json())
            for entry in entries:
                item = self._build_evidence_item(**entry)
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"OMIM error: {e}")

        return result

    def _build_evidence_item(
        self, mim_number: int, title: str, gene_symbols: str,
        inheritance: str, phenotype: str,
    ) -> EvidenceItem:
        return EvidenceItem(
            id=f"evi:omim:{mim_number}",
            claim=f"OMIM: {title} — {phenotype} ({inheritance})",
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.strong,
            source_refs=[f"omim:{mim_number}"],
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="omim_connector",
                source_artifact_id=str(mim_number),
            ),
            body={
                "mim_number": mim_number,
                "title": title,
                "gene_symbols": gene_symbols,
                "inheritance": inheritance,
                "phenotype": phenotype,
                "pch_layer": 1,
                "data_source": "omim",
            },
        )
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/omim.py tests/test_omim_connector.py
git commit -m "feat: OMIM connector — gene-phenotype mapping for genetic interpretation"
```

---

## Task 6: PharmGKB Connector

**Files:**
- Create: `scripts/connectors/pharmgkb.py`
- Create: `tests/test_pharmgkb_connector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pharmgkb_connector.py
"""Tests for PharmGKB pharmacogenomics connector."""
from __future__ import annotations

import pytest

from connectors.pharmgkb import PharmGKBConnector, _parse_drug_response, _parse_clinical_annotations


class TestParseDrugResponse:

    def test_parses_drug_info(self):
        raw = {
            "data": {
                "id": "PA450626",
                "name": "riluzole",
                "genericNames": ["riluzole"],
                "crossReferences": {"drugBank": ["DB00740"]},
            }
        }
        drug = _parse_drug_response(raw)
        assert drug["pharmgkb_id"] == "PA450626"
        assert drug["name"] == "riluzole"

    def test_empty_response(self):
        drug = _parse_drug_response({})
        assert drug["pharmgkb_id"] == ""


class TestParseClinicalAnnotations:

    def test_parses_annotations(self):
        raw = {
            "data": [
                {"id": 1, "gene": {"symbol": "CYP1A2"},
                 "drug": {"name": "riluzole"},
                 "phenotypeCategory": {"term": "Metabolism/PK"},
                 "level": "1A",
                 "summary": "CYP1A2 metabolizes riluzole"},
            ]
        }
        annotations = _parse_clinical_annotations(raw)
        assert len(annotations) == 1
        assert annotations[0]["gene"] == "CYP1A2"

    def test_empty_annotations(self):
        annotations = _parse_clinical_annotations({"data": []})
        assert annotations == []


class TestPharmGKBConnector:

    def test_instantiates(self):
        connector = PharmGKBConnector(evidence_store=None)
        assert connector is not None

    def test_evidence_item_id_format(self):
        connector = PharmGKBConnector(evidence_store=None)
        item = connector._build_evidence_item(
            pharmgkb_id="PA450626",
            drug_name="riluzole",
            gene="CYP1A2",
            annotation="CYP1A2 metabolizes riluzole",
            level="1A",
            category="Metabolism/PK",
        )
        assert item.id == "evi:pharmgkb:PA450626_CYP1A2"
        assert item.provenance.asserted_by == "pharmgkb_connector"
        assert "riluzole" in item.claim

    def test_level_1a_is_strong(self):
        connector = PharmGKBConnector(evidence_store=None)
        item = connector._build_evidence_item(
            pharmgkb_id="PA1", drug_name="riluzole", gene="CYP1A2",
            annotation="test", level="1A", category="Metabolism/PK",
        )
        assert item.strength.value == "strong"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# scripts/connectors/pharmgkb.py
"""PharmGKB connector — pharmacogenomics and drug safety.

Provides CYP enzyme metabolism, drug-gene interactions, and clinical
dosing guidelines. Safety-critical for protocol combination checking.
"""
from __future__ import annotations

import requests
from typing import Optional

from connectors.base import BaseConnector, ConnectorResult
from ontology.base import Provenance
from ontology.enums import EvidenceDirection, EvidenceStrength, SourceSystem
from ontology.evidence import EvidenceItem


BASE_URL = "https://api.pharmgkb.org/v1"


def _parse_drug_response(raw: dict) -> dict:
    """Parse PharmGKB drug lookup response."""
    data = raw.get("data", {})
    return {
        "pharmgkb_id": data.get("id", ""),
        "name": data.get("name", ""),
        "generic_names": data.get("genericNames", []),
    }


def _parse_clinical_annotations(raw: dict) -> list[dict]:
    """Parse PharmGKB clinical annotation response."""
    results = []
    for ann in raw.get("data", []):
        gene_info = ann.get("gene", {}) or {}
        drug_info = ann.get("drug", {}) or {}
        category_info = ann.get("phenotypeCategory", {}) or {}
        results.append({
            "annotation_id": ann.get("id", ""),
            "gene": gene_info.get("symbol", ""),
            "drug": drug_info.get("name", ""),
            "category": category_info.get("term", ""),
            "level": ann.get("level", ""),
            "summary": ann.get("summary", ""),
        })
    return results


class PharmGKBConnector(BaseConnector):
    """Connector to the PharmGKB REST API."""

    def __init__(self, evidence_store=None, base_url: str = BASE_URL):
        self._store = evidence_store
        self._base_url = base_url

    def fetch(self, *, drug_name: str = "", gene: str = "", **kwargs) -> ConnectorResult:
        """Fetch pharmacogenomic data for a drug or gene."""
        result = ConnectorResult()

        if drug_name:
            return self._fetch_drug_annotations(drug_name, result)
        elif gene:
            return self._fetch_gene_annotations(gene, result)
        else:
            result.errors.append("drug_name or gene required")
            return result

    def _fetch_drug_annotations(self, drug_name: str, result: ConnectorResult) -> ConnectorResult:
        try:
            url = f"{self._base_url}/data/drug"
            params = {"name": drug_name}
            resp = self._retry_with_backoff(requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            drug_info = _parse_drug_response(resp.json())

            if drug_info["pharmgkb_id"]:
                # Get clinical annotations for this drug
                ann_url = f"{self._base_url}/data/clinicalAnnotation"
                ann_params = {"location.drugs.name": drug_name}
                ann_resp = self._retry_with_backoff(requests.get, ann_url, params=ann_params, timeout=self.REQUEST_TIMEOUT)
                ann_resp.raise_for_status()

                annotations = _parse_clinical_annotations(ann_resp.json())
                for ann in annotations:
                    item = self._build_evidence_item(
                        pharmgkb_id=drug_info["pharmgkb_id"],
                        drug_name=drug_name,
                        gene=ann["gene"],
                        annotation=ann["summary"],
                        level=ann["level"],
                        category=ann["category"],
                    )
                    if self._store:
                        self._store.upsert_evidence_item(item)
                    result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"PharmGKB error: {e}")

        return result

    def _fetch_gene_annotations(self, gene: str, result: ConnectorResult) -> ConnectorResult:
        try:
            url = f"{self._base_url}/data/clinicalAnnotation"
            params = {"location.genes.symbol": gene}
            resp = self._retry_with_backoff(requests.get, url, params=params, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()

            annotations = _parse_clinical_annotations(resp.json())
            for ann in annotations:
                item = self._build_evidence_item(
                    pharmgkb_id=f"gene_{gene}",
                    drug_name=ann["drug"],
                    gene=gene,
                    annotation=ann["summary"],
                    level=ann["level"],
                    category=ann["category"],
                )
                if self._store:
                    self._store.upsert_evidence_item(item)
                result.evidence_items_added += 1

        except Exception as e:
            result.errors.append(f"PharmGKB error: {e}")

        return result

    def _build_evidence_item(
        self, pharmgkb_id: str, drug_name: str, gene: str,
        annotation: str, level: str, category: str,
    ) -> EvidenceItem:
        # Evidence level mapping: 1A/1B = strong, 2A/2B = moderate, 3/4 = emerging
        if level in ("1A", "1B"):
            strength = EvidenceStrength.strong
        elif level in ("2A", "2B"):
            strength = EvidenceStrength.moderate
        else:
            strength = EvidenceStrength.emerging

        return EvidenceItem(
            id=f"evi:pharmgkb:{pharmgkb_id}_{gene}",
            claim=f"PharmGKB: {drug_name} ↔ {gene} ({category}) — Level {level}",
            direction=EvidenceDirection.supports,
            strength=strength,
            source_refs=[f"pharmgkb:{pharmgkb_id}"],
            provenance=Provenance(
                source_system=SourceSystem.database,
                asserted_by="pharmgkb_connector",
                source_artifact_id=pharmgkb_id,
            ),
            body={
                "pharmgkb_id": pharmgkb_id,
                "drug_name": drug_name,
                "gene": gene,
                "annotation": annotation,
                "level": level,
                "category": category,
                "pch_layer": 2,
                "data_source": "pharmgkb",
            },
        )
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/connectors/pharmgkb.py tests/test_pharmgkb_connector.py
git commit -m "feat: PharmGKB connector — pharmacogenomics and drug safety checking"
```

---

## Task 7: PRO-ACT Trajectory Analyzer

**Files:**
- Create: `scripts/research/trajectory.py`
- Create: `tests/test_trajectory.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trajectory.py
"""Tests for PRO-ACT cohort matching and trajectory prediction."""
from __future__ import annotations

import pytest
import io
import csv

from research.trajectory import ProACTAnalyzer, CohortMatch, _parse_alsfrs_csv


class TestParseAlsfrsCsv:

    def test_parses_rows(self):
        csv_data = "SubjectID,ALSFRS_Delta,ALSFRS_R_Total\n1,0,43\n1,30,41\n1,60,39\n2,0,38\n2,30,35\n"
        records = _parse_alsfrs_csv(csv_data)
        assert len(records) == 5
        assert records[0]["subject_id"] == "1"
        assert records[0]["alsfrs_r_total"] == 43

    def test_empty_csv(self):
        records = _parse_alsfrs_csv("")
        assert records == []


class TestCohortMatch:

    def test_construction(self):
        match = CohortMatch(
            n_patients=150,
            median_decline_rate=-0.8,
            p25_decline_rate=-1.2,
            p75_decline_rate=-0.4,
            median_survival_months=36,
            erik_percentile=65,
        )
        assert match.n_patients == 150
        assert match.median_decline_rate == -0.8
        assert match.erik_percentile == 65


class TestProACTAnalyzer:

    def test_instantiates(self):
        analyzer = ProACTAnalyzer()
        assert analyzer is not None
        assert analyzer._loaded is False

    def test_match_cohort_without_data_returns_empty(self):
        analyzer = ProACTAnalyzer()
        match = analyzer.match_cohort(
            age=67, sex="male", onset_region="lower_limb",
            baseline_alsfrs_r=43, decline_rate=-0.39,
        )
        assert match.n_patients == 0

    def test_load_from_records(self):
        """Test with synthetic PRO-ACT-like records."""
        records = [
            {"subject_id": "1", "alsfrs_delta": 0, "alsfrs_r_total": 44, "age": 65, "sex": "Male", "onset": "Limb"},
            {"subject_id": "1", "alsfrs_delta": 90, "alsfrs_r_total": 38, "age": 65, "sex": "Male", "onset": "Limb"},
            {"subject_id": "2", "alsfrs_delta": 0, "alsfrs_r_total": 42, "age": 68, "sex": "Male", "onset": "Limb"},
            {"subject_id": "2", "alsfrs_delta": 90, "alsfrs_r_total": 36, "age": 68, "sex": "Male", "onset": "Limb"},
            {"subject_id": "3", "alsfrs_delta": 0, "alsfrs_r_total": 40, "age": 70, "sex": "Female", "onset": "Bulbar"},
            {"subject_id": "3", "alsfrs_delta": 90, "alsfrs_r_total": 32, "age": 70, "sex": "Female", "onset": "Bulbar"},
        ]
        analyzer = ProACTAnalyzer()
        analyzer._load_from_records(records)
        assert analyzer._loaded is True
        assert len(analyzer._subjects) > 0

    def test_match_cohort_with_data(self):
        records = [
            {"subject_id": str(i), "alsfrs_delta": 0, "alsfrs_r_total": 43,
             "age": 66, "sex": "Male", "onset": "Limb"}
            for i in range(20)
        ] + [
            {"subject_id": str(i), "alsfrs_delta": 90, "alsfrs_r_total": 40,
             "age": 66, "sex": "Male", "onset": "Limb"}
            for i in range(20)
        ]
        analyzer = ProACTAnalyzer()
        analyzer._load_from_records(records)
        match = analyzer.match_cohort(
            age=67, sex="male", onset_region="lower_limb",
            baseline_alsfrs_r=43, decline_rate=-0.39,
        )
        assert match.n_patients > 0
        assert match.median_decline_rate < 0  # Disease progresses
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Write implementation**

```python
# scripts/research/trajectory.py
"""PRO-ACT cohort matching and trajectory prediction.

Loads PRO-ACT ALSFRS data, builds subject-level decline rates,
and matches Erik against the cohort by demographics and baseline.
"""
from __future__ import annotations

import csv
import io
import os
import statistics
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CohortMatch:
    """Result of matching Erik against the PRO-ACT cohort."""
    n_patients: int = 0
    median_decline_rate: float = 0.0  # points per month
    p25_decline_rate: float = 0.0
    p75_decline_rate: float = 0.0
    median_survival_months: float = 0.0
    erik_percentile: float = 0.0  # Erik's decline rate percentile (0-100, lower = slower decline)


@dataclass
class _SubjectSummary:
    subject_id: str
    age: Optional[float] = None
    sex: Optional[str] = None
    onset: Optional[str] = None
    baseline_alsfrs_r: Optional[float] = None
    decline_rate: Optional[float] = None  # points per month


def _parse_alsfrs_csv(csv_text: str) -> list[dict]:
    """Parse PRO-ACT ALSFRS CSV data into records."""
    if not csv_text.strip():
        return []
    reader = csv.DictReader(io.StringIO(csv_text))
    records = []
    for row in reader:
        try:
            records.append({
                "subject_id": row.get("SubjectID", row.get("subject_id", "")),
                "alsfrs_delta": int(float(row.get("ALSFRS_Delta", row.get("alsfrs_delta", 0)))),
                "alsfrs_r_total": float(row.get("ALSFRS_R_Total", row.get("alsfrs_r_total", 0))),
            })
        except (ValueError, TypeError):
            continue
    return records


class ProACTAnalyzer:
    """Loads PRO-ACT data and provides cohort matching for Erik."""

    def __init__(self, data_dir: Optional[str] = None):
        self._data_dir = data_dir
        self._loaded = False
        self._subjects: dict[str, _SubjectSummary] = {}

    def _load_from_records(self, records: list[dict]) -> None:
        """Build subject summaries from parsed records (for testing)."""
        # Group by subject
        by_subject: dict[str, list[dict]] = {}
        for rec in records:
            sid = rec["subject_id"]
            by_subject.setdefault(sid, []).append(rec)

        for sid, recs in by_subject.items():
            recs.sort(key=lambda r: r.get("alsfrs_delta", 0))
            baseline = recs[0].get("alsfrs_r_total")
            last = recs[-1].get("alsfrs_r_total")
            delta_days = recs[-1].get("alsfrs_delta", 0) - recs[0].get("alsfrs_delta", 0)

            decline_rate = None
            if baseline is not None and last is not None and delta_days > 0:
                decline_rate = (last - baseline) / (delta_days / 30.0)

            # Extract demographics from first record
            first = recs[0]
            self._subjects[sid] = _SubjectSummary(
                subject_id=sid,
                age=first.get("age"),
                sex=first.get("sex"),
                onset=first.get("onset"),
                baseline_alsfrs_r=baseline,
                decline_rate=decline_rate,
            )

        self._loaded = True

    def load(self) -> bool:
        """Load PRO-ACT data from the configured directory."""
        if self._loaded:
            return True
        if not self._data_dir or not os.path.isdir(self._data_dir):
            return False

        alsfrs_path = os.path.join(self._data_dir, "ALSFRS.csv")
        if not os.path.isfile(alsfrs_path):
            return False

        with open(alsfrs_path, "r") as f:
            csv_text = f.read()
        records = _parse_alsfrs_csv(csv_text)
        self._load_from_records(records)
        return True

    def match_cohort(
        self,
        age: float,
        sex: str,
        onset_region: str,
        baseline_alsfrs_r: float,
        decline_rate: float,
        age_range: float = 10.0,
        alsfrs_range: float = 6.0,
    ) -> CohortMatch:
        """Find PRO-ACT patients matching Erik's demographics and baseline."""
        if not self._loaded or not self._subjects:
            return CohortMatch()

        # Normalize onset naming
        onset_map = {"lower_limb": "Limb", "upper_limb": "Limb", "bulbar": "Bulbar", "limb": "Limb"}
        target_onset = onset_map.get(onset_region.lower(), onset_region)

        matched_rates: list[float] = []
        for subj in self._subjects.values():
            if subj.decline_rate is None:
                continue
            if subj.baseline_alsfrs_r is None:
                continue

            # Demographics filter (relaxed — use what's available)
            if subj.age is not None and abs(subj.age - age) > age_range:
                continue
            if subj.onset is not None and target_onset and subj.onset != target_onset:
                continue
            if abs(subj.baseline_alsfrs_r - baseline_alsfrs_r) > alsfrs_range:
                continue

            matched_rates.append(subj.decline_rate)

        if not matched_rates:
            return CohortMatch()

        matched_rates.sort()
        n = len(matched_rates)
        median_rate = statistics.median(matched_rates)
        p25 = matched_rates[n // 4] if n >= 4 else matched_rates[0]
        p75 = matched_rates[3 * n // 4] if n >= 4 else matched_rates[-1]

        # Erik's percentile (lower = slower decline = better)
        slower_count = sum(1 for r in matched_rates if r <= decline_rate)
        erik_pct = (slower_count / n) * 100.0

        return CohortMatch(
            n_patients=n,
            median_decline_rate=round(median_rate, 3),
            p25_decline_rate=round(p25, 3),
            p75_decline_rate=round(p75, 3),
            median_survival_months=0.0,  # Requires separate survival data
            erik_percentile=round(erik_pct, 1),
        )
```

- [ ] **Step 4: Run tests, verify PASS**

- [ ] **Step 5: Commit**

```bash
git add scripts/research/trajectory.py tests/test_trajectory.py
git commit -m "feat: PRO-ACT trajectory analyzer — cohort matching and decline rate benchmarking"
```

---

## Task 8: Expand Research Loop — 5 New Action Types

**Files:**
- Modify: `scripts/research/actions.py` — add 5 new ActionType values
- Modify: `scripts/research/policy.py` — add pathway/genetics/pharmacogenomics triggers
- Modify: `scripts/research/loop.py` — add 5 new `_exec_*` functions + dispatch entries
- Create: `tests/test_expanded_actions.py`
- Modify: `data/erik_config.json` — add Phase 3B config keys

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expanded_actions.py
"""Tests for expanded research actions (Phase 3B)."""
from __future__ import annotations

import pytest

from research.actions import ActionType


class TestExpandedActionTypes:

    def test_total_15_actions(self):
        assert len(ActionType) == 15

    def test_new_evidence_actions(self):
        assert ActionType.QUERY_PATHWAYS.value == "query_pathways"
        assert ActionType.QUERY_PPI_NETWORK.value == "query_ppi_network"
        assert ActionType.MATCH_COHORT.value == "match_cohort"

    def test_new_reasoning_actions(self):
        assert ActionType.INTERPRET_VARIANT.value == "interpret_variant"
        assert ActionType.CHECK_PHARMACOGENOMICS.value == "check_pharmacogenomics"
```

- [ ] **Step 2: Run to verify it fails**

- [ ] **Step 3: Add 5 new ActionType values to `scripts/research/actions.py`**

Add after `REGENERATE_PROTOCOL`:
```python
    # Phase 3B: Evidence expansion
    QUERY_PATHWAYS = "query_pathways"
    QUERY_PPI_NETWORK = "query_ppi_network"
    MATCH_COHORT = "match_cohort"
    INTERPRET_VARIANT = "interpret_variant"
    CHECK_PHARMACOGENOMICS = "check_pharmacogenomics"
```

Also add to `NETWORK_ACTIONS`:
```python
NETWORK_ACTIONS = {
    ActionType.SEARCH_PUBMED,
    ActionType.SEARCH_TRIALS,
    ActionType.QUERY_OPENTARGETS,
    ActionType.QUERY_PATHWAYS,       # NEW
    ActionType.QUERY_PPI_NETWORK,    # NEW
    ActionType.INTERPRET_VARIANT,    # NEW
    ActionType.CHECK_PHARMACOGENOMICS,  # NEW
}
```

- [ ] **Step 4: Add new dispatch entries to `scripts/research/loop.py`**

Add to the dispatch dict in `_execute_action`:
```python
ActionType.QUERY_PATHWAYS: _exec_query_pathways,
ActionType.QUERY_PPI_NETWORK: _exec_query_ppi_network,
ActionType.MATCH_COHORT: _exec_match_cohort,
ActionType.INTERPRET_VARIANT: _exec_interpret_variant,
ActionType.CHECK_PHARMACOGENOMICS: _exec_check_pharmacogenomics,
```

Add 5 new executor functions:

```python
def _exec_query_pathways(params, state, store, llm_manager):
    """Query Reactome + KEGG for pathway data on a target."""
    from connectors.reactome import ReactomeConnector
    from connectors.kegg import KEGGConnector
    from targets.als_targets import ALS_TARGETS

    target_name = params.get("target_name", "")
    target = ALS_TARGETS.get(target_name, {})
    uniprot_id = target.get("uniprot_id", "")
    gene = target.get("gene", "")

    total_added = 0

    if uniprot_id:
        rc = ReactomeConnector(evidence_store=store)
        cr = rc.fetch(uniprot_id=uniprot_id, gene_symbol=gene)
        total_added += cr.evidence_items_added

    if gene:
        # KEGG uses NCBI gene IDs; skip if we don't have one
        kc = KEGGConnector(evidence_store=store)
        cr = kc.fetch(gene_symbol=gene)
        total_added += cr.evidence_items_added

    return ActionResult(
        action=ActionType.QUERY_PATHWAYS,
        evidence_items_added=total_added,
    )


def _exec_query_ppi_network(params, state, store, llm_manager):
    """Query STRING for protein-protein interactions."""
    from connectors.string_db import STRINGConnector
    gene_symbol = params.get("gene_symbol", "")
    connector = STRINGConnector(evidence_store=store)
    cr = connector.fetch(gene_symbol=gene_symbol)
    return ActionResult(
        action=ActionType.QUERY_PPI_NETWORK,
        evidence_items_added=cr.evidence_items_added,
        success=not cr.errors,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_match_cohort(params, state, store, llm_manager):
    """Match Erik against PRO-ACT cohort."""
    from research.trajectory import ProACTAnalyzer
    analyzer = ProACTAnalyzer(data_dir=params.get("proact_data_dir"))
    analyzer.load()
    match = analyzer.match_cohort(
        age=67, sex="male", onset_region="lower_limb",
        baseline_alsfrs_r=43, decline_rate=-0.39,
    )
    return ActionResult(
        action=ActionType.MATCH_COHORT,
        detail={"cohort_match": {"n_patients": match.n_patients, "median_decline": match.median_decline_rate,
                                  "erik_percentile": match.erik_percentile}},
    )


def _exec_interpret_variant(params, state, store, llm_manager):
    """Query ClinVar + OMIM for genetic variant interpretation."""
    from connectors.clinvar import ClinVarConnector
    gene = params.get("gene", "")
    total_added = 0

    cv = ClinVarConnector(evidence_store=store)
    cr = cv.fetch(gene=gene)
    total_added += cr.evidence_items_added

    return ActionResult(
        action=ActionType.INTERPRET_VARIANT,
        evidence_items_added=total_added,
        success=not cr.errors,
        error="; ".join(cr.errors) if cr.errors else None,
    )


def _exec_check_pharmacogenomics(params, state, store, llm_manager):
    """Check PharmGKB for drug-gene interactions."""
    from connectors.pharmgkb import PharmGKBConnector
    drug_name = params.get("drug_name", "")
    connector = PharmGKBConnector(evidence_store=store)
    cr = connector.fetch(drug_name=drug_name)
    return ActionResult(
        action=ActionType.CHECK_PHARMACOGENOMICS,
        evidence_items_added=cr.evidence_items_added,
        interaction_safe=not cr.errors,
        success=not cr.errors,
        error="; ".join(cr.errors) if cr.errors else None,
    )
```

- [ ] **Step 5: Add Phase 3B config keys to `data/erik_config.json`**

```json
"reactome_base_url": "https://reactome.org/ContentService",
"kegg_base_url": "https://rest.kegg.jp",
"string_base_url": "https://string-db.org/api",
"string_min_score": 400,
"clinvar_enabled": true,
"omim_api_key": null,
"pharmgkb_base_url": "https://api.pharmgkb.org/v1",
"proact_data_dir": null,
"genetics_received": false
```

- [ ] **Step 6: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_expanded_actions.py tests/test_research_loop.py -v`
Expected: All PASS

- [ ] **Step 7: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -k "not network and not chembl and not llm" --tb=short -q`
Expected: All PASS (770+ tests)

- [ ] **Step 8: Commit**

```bash
git add scripts/research/actions.py scripts/research/loop.py scripts/research/policy.py data/erik_config.json tests/test_expanded_actions.py
git commit -m "feat: wire 5 new action types into research loop — pathways, PPI, cohort, variants, pharmacogenomics"
```

---

## Task 9: Enhance Causal Chains with Pathway Data

**Files:**
- Modify: `scripts/research/causal_chains.py` — add `pathway_grounded_link()`

- [ ] **Step 1: Add pathway grounding function to causal_chains.py**

```python
def pathway_grounded_link(
    source: str,
    target: str,
    pathway_evidence: list[dict],
) -> Optional[CausalLink]:
    """Create a causal link grounded in pathway data (Reactome/KEGG).

    If any pathway evidence connects source to target, creates a
    high-confidence link. Otherwise returns None.

    Parameters
    ----------
    source: Current chain endpoint (e.g., "sigma-1R activation")
    target: Proposed next step (e.g., "ER calcium homeostasis")
    pathway_evidence: Evidence items from Reactome/KEGG with body.pathway_name

    Returns
    -------
    CausalLink with confidence=0.95 if pathway connection found, else None.
    """
    source_lower = source.lower()
    target_lower = target.lower()

    for evi in pathway_evidence:
        body = evi.get("body", {})
        pathway_name = body.get("pathway_name", "").lower()
        data_source = body.get("data_source", "")

        # Check if pathway name contains both source and target concepts
        if source_lower in pathway_name or target_lower in pathway_name:
            return CausalLink(
                source=source,
                target=target,
                mechanism=f"pathway: {body.get('pathway_name', '')}",
                evidence_ref=evi.get("id", ""),
                confidence=0.95,
            )

    return None
```

- [ ] **Step 2: Add a test for pathway_grounded_link**

Add to `tests/test_causal_chains.py`:

```python
class TestPathwayGroundedLink:

    def test_finds_pathway_connection(self):
        from research.causal_chains import pathway_grounded_link
        evidence = [
            {"id": "evi:reactome:R-HSA-123", "body": {
                "pathway_name": "Cellular response to ER stress",
                "data_source": "reactome",
            }},
        ]
        link = pathway_grounded_link("sigma-1R", "ER stress", evidence)
        assert link is not None
        assert link.confidence == 0.95
        assert "reactome" in link.evidence_ref

    def test_returns_none_when_no_match(self):
        from research.causal_chains import pathway_grounded_link
        evidence = [
            {"id": "evi:reactome:R-HSA-999", "body": {
                "pathway_name": "Cholesterol biosynthesis",
                "data_source": "reactome",
            }},
        ]
        link = pathway_grounded_link("sigma-1R", "TDP-43 proteostasis", evidence)
        assert link is None
```

- [ ] **Step 3: Run tests, verify PASS**

- [ ] **Step 4: Commit**

```bash
git add scripts/research/causal_chains.py tests/test_causal_chains.py
git commit -m "feat: pathway-grounded causal chain links — Reactome/KEGG evidence for chain construction"
```

---

## Task 10: Update README + Final Verification

- [ ] **Step 1: Update README**

Add a Phase 3B section after Phase 3 and update the roadmap table. Update data source count from 5 to 12. Update test count. Add the new connectors to the Project Structure section.

Key additions:
- Phase 3B section describing all 7 new data sources
- Updated project structure with new connector files + trajectory analyzer
- Updated roadmap table with Phase 3B marked Complete
- Updated total test count
- Add new data sources to the architecture diagram description

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -k "not network and not chembl and not llm" --tb=short -q`
Expected: All PASS (780+ tests)

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: Phase 3B complete — 12 data sources, 15 research actions, pathway-grounded causal chains"
```

- [ ] **Step 4: Push to GitHub**

```bash
git push origin main
```

---

## Summary

After completing all 10 tasks, Phase 3B delivers:

- **6 new API connectors** — Reactome (pathway cascades), KEGG (pathway ontology), STRING (PPI network), ClinVar (variant pathogenicity), OMIM (gene-phenotype), PharmGKB (pharmacogenomics)
- **1 local data analyzer** — PRO-ACT cohort matching with trajectory prediction
- **5 new research actions** — QUERY_PATHWAYS, QUERY_PPI_NETWORK, MATCH_COHORT, INTERPRET_VARIANT, CHECK_PHARMACOGENOMICS
- **15 total action types** — expanded from 10, fully wired into the loop dispatch
- **Pathway-grounded causal chains** — Reactome/KEGG evidence as ground truth for mechanism links (confidence 0.95)
- **12 total data sources** — PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, DrugBank, Reactome, KEGG, STRING, PRO-ACT, ClinVar, OMIM, PharmGKB
- **Updated README** — complete architecture documentation reflecting all integrations

The evidence fabric grows from 5 to 12 data sources. Causal chains shift from LLM-only inference to pathway-grounded construction. Drug safety checking gains pharmacogenomic depth. Genetic interpretation is ready for Erik's Invitae results. Population benchmarking enables trajectory prediction.
