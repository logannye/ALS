# Phase 1A: Evidence Seed — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update ontology models, build ALS target definitions, create a curated evidence seed of ~110 evidence items and ~30 interventions organized by the 5 cure protocol layers, and store them in PostgreSQL.

**Architecture:** Extend existing Phase 0 ontology with `EvidenceStrength` enum and model changes. Define canonical ALS drug targets. Build an evidence store for PostgreSQL CRUD. Create curated seed data as JSON files (one per protocol layer + interventions + drug design targets). Build a seed loader that populates the database. All evidence items are tagged with protocol layer, mechanism target, applicable subtypes, Erik eligibility, and PCH level.

**Tech Stack:** Python 3.12, Pydantic v2, psycopg3, pytest, JSON seed files.

**Spec:** `/Users/logannye/.openclaw/erik/docs/specs/2026-03-25-phase1-evidence-fabric-design.md`

---

## File Structure

```
scripts/
  ontology/
    enums.py                   # MODIFY: add EvidenceStrength, extend InterventionClass
    evidence.py                # MODIFY: change strength type, add supersedes_ref + modality
    intervention.py            # MODIFY: add route default
  targets/
    __init__.py                # CREATE
    als_targets.py             # CREATE: canonical ALS drug target definitions
  evidence/
    __init__.py                # CREATE
    evidence_store.py          # CREATE: CRUD for EvidenceItem/Bundle/Intervention in PG
    seed_builder.py            # CREATE: load JSON seed files into evidence store
data/
  seed/
    interventions.json         # CREATE: ~30 ALS intervention objects
    layer_a_root_cause.json    # CREATE: ~25 evidence items
    layer_b_pathology.json     # CREATE: ~25 evidence items
    layer_c_circuit.json       # CREATE: ~20 evidence items
    layer_d_regeneration.json  # CREATE: ~15 evidence items
    layer_e_maintenance.json   # CREATE: ~15 evidence items
    drug_design_targets.json   # CREATE: ~10 computational drug design target defs
tests/
  test_evidence_strength.py    # CREATE
  test_als_targets.py          # CREATE
  test_evidence_store.py       # CREATE
  test_seed_builder.py         # CREATE
  test_seed_completeness.py    # CREATE: verify all protocol layers covered
```

---

## Task 1: Ontology Model Updates

**Files:**
- Modify: `scripts/ontology/enums.py` (add `EvidenceStrength`, extend `InterventionClass`)
- Modify: `scripts/ontology/evidence.py` (change `strength` type, add fields)
- Modify: `tests/test_evidence_models.py` (update `strength` from float to enum)
- Create: `tests/test_evidence_strength.py`

- [ ] **Step 1: Write the failing test for EvidenceStrength**

```python
# tests/test_evidence_strength.py
from ontology.enums import EvidenceStrength, InterventionClass


def test_evidence_strength_values():
    assert EvidenceStrength.strong.value == "strong"
    assert EvidenceStrength.moderate.value == "moderate"
    assert EvidenceStrength.EMERGING.value == "emerging"
    assert EvidenceStrength.PRECLINICAL.value == "preclinical"
    assert EvidenceStrength.unknown.value == "unknown"


def test_intervention_class_gene_therapy():
    assert InterventionClass.gene_therapy.value == "gene_therapy"


def test_intervention_class_cell_therapy():
    assert InterventionClass.cell_therapy.value == "cell_therapy"


def test_intervention_class_peptide():
    assert InterventionClass.peptide.value == "peptide"


def test_evidence_item_with_strength_enum():
    from ontology.evidence import EvidenceItem
    from ontology.enums import EvidenceDirection
    e = EvidenceItem(
        id="evi:test_strength",
        claim="Test claim",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.strong,
    )
    assert e.strength == EvidenceStrength.strong


def test_evidence_item_with_supersedes_ref():
    from ontology.evidence import EvidenceItem
    from ontology.enums import EvidenceDirection
    e = EvidenceItem(
        id="evi:v2",
        claim="Updated claim",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.moderate,
        supersedes_ref="evi:v1",
    )
    assert e.supersedes_ref == "evi:v1"


def test_evidence_item_modality():
    from ontology.evidence import EvidenceItem
    from ontology.enums import EvidenceDirection
    e = EvidenceItem(
        id="evi:test_modality",
        claim="RCT evidence",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.strong,
    )
    assert e.supersedes_ref is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_evidence_strength.py -v`
Expected: FAIL (EvidenceStrength not found)

- [ ] **Step 3: Add EvidenceStrength enum and extend InterventionClass**

In `scripts/ontology/enums.py`, add after `EvidenceDirection`:

```python
class EvidenceStrength(str, Enum):
    """Strength of evidence supporting a claim."""
    strong = "strong"
    moderate = "moderate"
    emerging = "emerging"
    preclinical = "preclinical"
    unknown = "unknown"
```

In `InterventionClass`, add three new members:

```python
    gene_therapy = "gene_therapy"
    cell_therapy = "cell_therapy"
    peptide = "peptide"
```

- [ ] **Step 4: Update EvidenceItem model**

In `scripts/ontology/evidence.py`, change:
- `strength: float` → `strength: EvidenceStrength = EvidenceStrength.unknown`
- Add `supersedes_ref: Optional[str] = None`
- Add import for `EvidenceStrength` and `Optional`

The updated EvidenceItem should be:

```python
from ontology.enums import EvidenceDirection, EvidenceStrength

class EvidenceItem(BaseEnvelope):
    type: str = Field(default="EvidenceItem", min_length=1)
    claim: str
    direction: EvidenceDirection
    source_refs: list[str] = Field(default_factory=list)
    strength: EvidenceStrength = EvidenceStrength.unknown
    supersedes_ref: Optional[str] = None
    notes: str = ""
```

- [ ] **Step 5: Add default to Intervention.route**

In `scripts/ontology/intervention.py`, change `route: str` to `route: str = ""` so interventions without a route don't fail validation.

- [ ] **Step 6: Update existing tests**

In `tests/test_evidence_models.py`, change all `strength=0.82` / `strength=0.0` / `strength=0.95` / `strength=0.5` to use `EvidenceStrength` enum values:
- `strength=0.82` → `strength=EvidenceStrength.strong`
- `strength=0.0` → `strength=EvidenceStrength.unknown`
- `strength=0.95` → `strength=EvidenceStrength.strong`
- `strength=0.5` → `strength=EvidenceStrength.moderate`

Remove or update `test_strength` that uses `pytest.approx(0.82)` — change to `assert e.strength == EvidenceStrength.strong`.

Add `from ontology.enums import EvidenceStrength` to the imports.

- [ ] **Step 7: Run ALL tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v`
Expected: All tests PASS (441 previous, adjusted, plus new)

- [ ] **Step 8: Commit**

```bash
git add scripts/ontology/enums.py scripts/ontology/evidence.py scripts/ontology/intervention.py tests/test_evidence_models.py tests/test_evidence_strength.py && git commit -m "feat: add EvidenceStrength enum, extend InterventionClass, update EvidenceItem model"
```

---

## Task 2: ALS Target Definitions

**Files:**
- Create: `scripts/targets/__init__.py`
- Create: `scripts/targets/als_targets.py`
- Create: `tests/test_als_targets.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_als_targets.py
from targets.als_targets import (
    ALS_TARGETS, get_target, get_targets_for_subtype,
    get_targets_for_protocol_layer,
)


def test_als_targets_not_empty():
    assert len(ALS_TARGETS) >= 15


def test_get_target_tdp43():
    t = get_target("TDP-43")
    assert t is not None
    assert t["uniprot_id"] == "Q13148"
    assert "sporadic_tdp43" in t["subtypes"]


def test_get_target_sod1():
    t = get_target("SOD1")
    assert t is not None
    assert t["uniprot_id"] == "P00441"


def test_get_target_unknown():
    assert get_target("NONEXISTENT") is None


def test_targets_for_sporadic_subtype():
    targets = get_targets_for_subtype("sporadic_tdp43")
    names = [t["name"] for t in targets]
    assert "TDP-43" in names
    assert "STMN2" in names
    assert "UNC13A" in names


def test_targets_for_root_cause_layer():
    targets = get_targets_for_protocol_layer("root_cause_suppression")
    names = [t["name"] for t in targets]
    assert "SOD1" in names or "TDP-43" in names


def test_every_target_has_required_fields():
    for name, target in ALS_TARGETS.items():
        assert "name" in target, f"{name} missing 'name'"
        assert "uniprot_id" in target or "gene_id" in target, f"{name} missing ID"
        assert "subtypes" in target, f"{name} missing 'subtypes'"
        assert "protocol_layers" in target, f"{name} missing 'protocol_layers'"
        assert "druggable" in target, f"{name} missing 'druggable'"
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_als_targets.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/targets/__init__.py
```

```python
# scripts/targets/als_targets.py
"""Canonical ALS drug target definitions.

Each target maps to: protein/gene identity, relevant subtypes,
protocol layers, druggability assessment, and key references.
"""
from __future__ import annotations
from typing import Optional

ALS_TARGETS: dict[str, dict] = {
    "TDP-43": {
        "name": "TDP-43",
        "gene": "TARDBP",
        "uniprot_id": "Q13148",
        "description": "RNA-binding protein; nuclear loss and cytoplasmic aggregation in ~97% of ALS",
        "subtypes": ["sporadic_tdp43", "tardbp", "c9orf72", "glia_amplified", "mixed"],
        "protocol_layers": ["root_cause_suppression", "pathology_reversal"],
        "druggable": True,
        "druggability_notes": "Intrabody (VTx-002), nuclear import enhancers, aggregation inhibitors",
        "pdb_ids": ["4BS2", "1WF0"],
        "alphafold_id": "AF-Q13148-F1",
    },
    "SOD1": {
        "name": "SOD1",
        "gene": "SOD1",
        "uniprot_id": "P00441",
        "description": "Cu/Zn superoxide dismutase; toxic gain-of-function mutations in ~2% of ALS",
        "subtypes": ["sod1"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": True,
        "druggability_notes": "ASO (tofersen), AAV gene therapy (Sodesta), small molecule stabilizers",
        "pdb_ids": ["2C9V"],
        "alphafold_id": "AF-P00441-F1",
    },
    "FUS": {
        "name": "FUS",
        "gene": "FUS",
        "uniprot_id": "P35637",
        "description": "RNA-binding protein; mutations cause aggressive juvenile-onset ALS",
        "subtypes": ["fus"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": True,
        "druggability_notes": "ASO (jacifusen/ulefnersen)",
        "pdb_ids": ["6GBM"],
        "alphafold_id": "AF-P35637-F1",
    },
    "C9orf72": {
        "name": "C9orf72",
        "gene": "C9orf72",
        "uniprot_id": "Q96LT7",
        "description": "GGGGCC repeat expansion; most common genetic cause (~40% familial, ~7% sporadic)",
        "subtypes": ["c9orf72"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": True,
        "druggability_notes": "ASO (next-gen after BIIB078), repeat RNA-targeting, DPR-specific",
        "pdb_ids": [],
        "alphafold_id": "AF-Q96LT7-F1",
    },
    "STMN2": {
        "name": "STMN2",
        "gene": "STMN2",
        "uniprot_id": "Q93045",
        "description": "Stathmin-2; critical for axonal maintenance. Cryptic exon inclusion under TDP-43 loss depletes STMN2.",
        "subtypes": ["sporadic_tdp43", "tardbp", "c9orf72"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": "ASO to block cryptic exon, gene therapy to restore",
    },
    "UNC13A": {
        "name": "UNC13A",
        "gene": "UNC13A",
        "uniprot_id": "Q9UPW8",
        "description": "Synaptic protein; cryptic exon inclusion under TDP-43 loss. ALS risk locus.",
        "subtypes": ["sporadic_tdp43", "tardbp", "c9orf72"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": "ASO to block cryptic exon inclusion",
    },
    "SIGMAR1": {
        "name": "Sigma-1 Receptor",
        "gene": "SIGMAR1",
        "uniprot_id": "Q99720",
        "description": "ER-mitochondria tethering chaperone; modulates calcium, proteostasis, and autophagy",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "fus", "tardbp", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": "Pridopidine (Phase 3 PREVAiLS), other S1R agonists",
        "pdb_ids": ["5HK1", "5HK2"],
    },
    "EAAT2": {
        "name": "EAAT2 / GLT-1",
        "gene": "SLC1A2",
        "uniprot_id": "P43004",
        "description": "Astrocytic glutamate transporter; loss in ALS contributes to excitotoxicity",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "fus", "tardbp", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["circuit_stabilization"],
        "druggable": True,
        "druggability_notes": "Riluzole (indirect), ceftriaxone (upregulator, failed Phase 3), LDN/OSU-0212320",
    },
    "BDNF": {
        "name": "BDNF",
        "gene": "BDNF",
        "uniprot_id": "P23560",
        "description": "Brain-derived neurotrophic factor; motor neuron survival and NMJ maintenance",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "fus", "tardbp", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["regeneration_reinnervation"],
        "druggable": True,
        "druggability_notes": "AAV-BDNF gene therapy, TrkB agonists, exercise-induced upregulation",
    },
    "GDNF": {
        "name": "GDNF",
        "gene": "GDNF",
        "uniprot_id": "P39905",
        "description": "Glial cell-derived neurotrophic factor; motor neuron survival and axonal regrowth",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "fus", "tardbp", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["regeneration_reinnervation"],
        "druggable": True,
        "druggability_notes": "AAV-GDNF gene therapy, encapsulated cell biodelivery",
    },
    "OPTN": {
        "name": "Optineurin",
        "gene": "OPTN",
        "uniprot_id": "Q96CV9",
        "description": "Autophagy receptor; mutations cause ALS via impaired autophagy/inflammation",
        "subtypes": ["sporadic_tdp43", "mixed"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": False,
        "druggability_notes": "No direct modulators; autophagy activators may compensate",
    },
    "TBK1": {
        "name": "TBK1",
        "gene": "TBK1",
        "uniprot_id": "Q9UHD2",
        "description": "Tank-binding kinase 1; links autophagy and innate immunity; ALS risk gene",
        "subtypes": ["sporadic_tdp43", "mixed"],
        "protocol_layers": ["pathology_reversal", "circuit_stabilization"],
        "druggable": True,
        "druggability_notes": "TBK1 activators (theoretical); pathway modulation via innate immunity",
    },
    "NEK1": {
        "name": "NEK1",
        "gene": "NEK1",
        "uniprot_id": "Q96PY6",
        "description": "NIMA-related kinase 1; DNA damage response and cilia function; common ALS risk gene",
        "subtypes": ["sporadic_tdp43", "mixed"],
        "protocol_layers": ["root_cause_suppression"],
        "druggable": False,
        "druggability_notes": "Loss-of-function mechanism; restoration strategies needed",
    },
    "C5_COMPLEMENT": {
        "name": "Complement C5",
        "gene": "C5",
        "uniprot_id": "P01031",
        "description": "Terminal complement pathway; neuroinflammation amplifier in ALS",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "glia_amplified"],
        "protocol_layers": ["circuit_stabilization"],
        "druggable": True,
        "druggability_notes": "Zilucoplan (C5 peptide inhibitor, HEALEY platform), eculizumab (off-label)",
    },
    "CSF1R": {
        "name": "CSF1R",
        "gene": "CSF1R",
        "uniprot_id": "P07333",
        "description": "Colony-stimulating factor 1 receptor; microglial activation and proliferation",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "glia_amplified"],
        "protocol_layers": ["circuit_stabilization"],
        "druggable": True,
        "druggability_notes": "Masitinib (tyrosine kinase inhibitor, Phase 3 ALS data), PLX5622",
    },
    "MTOR": {
        "name": "mTOR",
        "gene": "MTOR",
        "uniprot_id": "P42345",
        "description": "Mechanistic target of rapamycin; autophagy master regulator",
        "subtypes": ["sporadic_tdp43", "sod1", "c9orf72", "fus", "tardbp", "glia_amplified", "mixed", "unresolved"],
        "protocol_layers": ["pathology_reversal"],
        "druggable": True,
        "druggability_notes": "Rapamycin/everolimus (mTORC1 inhibitors), trehalose (mTOR-independent autophagy)",
    },
}


def get_target(name: str) -> Optional[dict]:
    """Return target definition by name, or None if not found."""
    return ALS_TARGETS.get(name)


def get_targets_for_subtype(subtype: str) -> list[dict]:
    """Return all targets relevant to the given ALS subtype."""
    return [t for t in ALS_TARGETS.values() if subtype in t["subtypes"]]


def get_targets_for_protocol_layer(layer: str) -> list[dict]:
    """Return all targets relevant to the given protocol layer."""
    return [t for t in ALS_TARGETS.values() if layer in t["protocol_layers"]]
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_als_targets.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/targets/ tests/test_als_targets.py && git commit -m "feat: canonical ALS drug target definitions (16 targets with druggability, subtypes, protocol layers)"
```

---

## Task 3: Evidence Store

**Files:**
- Create: `scripts/evidence/__init__.py`
- Create: `scripts/evidence/evidence_store.py`
- Create: `tests/test_evidence_store.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evidence_store.py
import json
import pytest
from ontology.evidence import EvidenceItem, EvidenceBundle
from ontology.intervention import Intervention
from ontology.enums import (
    EvidenceDirection, EvidenceStrength, InterventionClass, ProtocolLayer,
)


def test_store_and_retrieve_evidence_item(db_available):
    from evidence.evidence_store import EvidenceStore
    store = EvidenceStore()
    item = EvidenceItem(
        id="evi:test_store_001",
        claim="Test evidence claim",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.strong,
        source_refs=["pmid:12345"],
        body={"protocol_layer": "root_cause_suppression", "mechanism_target": "SOD1"},
    )
    store.upsert_evidence_item(item)
    retrieved = store.get_evidence_item("evi:test_store_001")
    assert retrieved is not None
    assert retrieved["claim"] == "Test evidence claim"
    assert retrieved["body"]["protocol_layer"] == "root_cause_suppression"
    # Cleanup
    store.delete("evi:test_store_001")


def test_store_and_retrieve_intervention(db_available):
    from evidence.evidence_store import EvidenceStore
    store = EvidenceStore()
    intervention = Intervention(
        id="int:test_riluzole",
        name="Riluzole",
        intervention_class=InterventionClass.drug,
        route="oral",
        targets=["EAAT2"],
        protocol_layer=ProtocolLayer.circuit_stabilization,
    )
    store.upsert_intervention(intervention)
    retrieved = store.get_intervention("int:test_riluzole")
    assert retrieved is not None
    assert retrieved["name"] == "Riluzole"
    store.delete("int:test_riluzole")


def test_query_by_protocol_layer(db_available):
    from evidence.evidence_store import EvidenceStore
    store = EvidenceStore()
    for i in range(3):
        store.upsert_evidence_item(EvidenceItem(
            id=f"evi:test_layer_{i}",
            claim=f"Claim {i}",
            direction=EvidenceDirection.supports,
            strength=EvidenceStrength.moderate,
            body={"protocol_layer": "pathology_reversal"},
        ))
    results = store.query_by_protocol_layer("pathology_reversal")
    assert len(results) >= 3
    for r in results:
        assert r["body"]["protocol_layer"] == "pathology_reversal"
    # Cleanup
    for i in range(3):
        store.delete(f"evi:test_layer_{i}")


def test_upsert_is_idempotent(db_available):
    from evidence.evidence_store import EvidenceStore
    store = EvidenceStore()
    item = EvidenceItem(
        id="evi:test_idempotent",
        claim="Original",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.strong,
    )
    store.upsert_evidence_item(item)
    store.upsert_evidence_item(item)  # Second call should not error
    retrieved = store.get_evidence_item("evi:test_idempotent")
    assert retrieved is not None
    store.delete("evi:test_idempotent")


def test_count_by_type(db_available):
    from evidence.evidence_store import EvidenceStore
    store = EvidenceStore()
    store.upsert_evidence_item(EvidenceItem(
        id="evi:test_count",
        claim="Count test",
        direction=EvidenceDirection.supports,
        strength=EvidenceStrength.unknown,
    ))
    count = store.count_by_type("EvidenceItem")
    assert count >= 1
    store.delete("evi:test_count")
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_evidence_store.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/evidence/__init__.py
```

```python
# scripts/evidence/evidence_store.py
"""CRUD operations for EvidenceItem, EvidenceBundle, and Intervention in PostgreSQL.

Uses the erik_core.objects table with JSONB body for flexible schema.
Supports upsert-by-ID, query-by-protocol-layer, and type-based counts.
"""
from __future__ import annotations

import json
from typing import Any, Optional

from db.pool import get_connection


class EvidenceStore:
    """Store and retrieve evidence objects in PostgreSQL."""

    def upsert_evidence_item(self, item) -> None:
        """Insert or update an EvidenceItem in erik_core.objects."""
        data = item.model_dump(mode="json")
        body = data.get("body", {})
        # Merge top-level evidence fields into body for query support
        body["claim"] = data.get("claim", "")
        body["direction"] = data.get("direction", "")
        body["strength"] = data.get("strength", "")
        body["source_refs"] = data.get("source_refs", [])
        body["supersedes_ref"] = data.get("supersedes_ref")
        body["notes"] = data.get("notes", "")

        with get_connection() as conn:
            conn.execute(
                """INSERT INTO erik_core.objects (id, type, status, body, provenance_source_system, confidence)
                   VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                   ON CONFLICT (id) DO UPDATE SET
                     body = EXCLUDED.body,
                     status = EXCLUDED.status,
                     updated_at = NOW()""",
                (
                    data["id"],
                    data["type"],
                    data.get("status", "active"),
                    json.dumps(body),
                    data.get("provenance", {}).get("source_system"),
                    data.get("uncertainty", {}).get("confidence"),
                ),
            )
            conn.commit()

    def upsert_intervention(self, intervention) -> None:
        """Insert or update an Intervention in erik_core.objects."""
        data = intervention.model_dump(mode="json")
        body = data.get("body", {})
        body["name"] = data.get("name", "")
        body["intervention_class"] = data.get("intervention_class", "")
        body["targets"] = data.get("targets", [])
        body["protocol_layer"] = data.get("protocol_layer")
        body["route"] = data.get("route", "")
        body["intended_effects"] = data.get("intended_effects", [])
        body["known_risks"] = data.get("known_risks", [])

        with get_connection() as conn:
            conn.execute(
                """INSERT INTO erik_core.objects (id, type, status, body)
                   VALUES (%s, %s, %s, %s::jsonb)
                   ON CONFLICT (id) DO UPDATE SET
                     body = EXCLUDED.body,
                     updated_at = NOW()""",
                (data["id"], data["type"], data.get("status", "active"), json.dumps(body)),
            )
            conn.commit()

    def get_evidence_item(self, item_id: str) -> Optional[dict]:
        """Retrieve an evidence item by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, type, status, body FROM erik_core.objects WHERE id = %s AND type = 'EvidenceItem'",
                (item_id,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "type": row[1], "status": row[2], "body": row[3],
                "claim": row[3].get("claim", ""), "direction": row[3].get("direction", ""),
                "strength": row[3].get("strength", "")}

    def get_intervention(self, int_id: str) -> Optional[dict]:
        """Retrieve an intervention by ID."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, type, status, body FROM erik_core.objects WHERE id = %s AND type = 'Intervention'",
                (int_id,),
            ).fetchone()
        if row is None:
            return None
        return {"id": row[0], "type": row[1], "status": row[2], "body": row[3],
                "name": row[3].get("name", "")}

    def query_by_protocol_layer(self, layer: str) -> list[dict]:
        """Return all evidence items for a given protocol layer."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT id, type, status, body FROM erik_core.objects
                   WHERE type = 'EvidenceItem' AND body->>'protocol_layer' = %s AND status = 'active'
                   ORDER BY id""",
                (layer,),
            ).fetchall()
        return [{"id": r[0], "type": r[1], "status": r[2], "body": r[3]} for r in rows]

    def query_by_mechanism_target(self, target: str) -> list[dict]:
        """Return all evidence items for a given mechanism target."""
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT id, type, status, body FROM erik_core.objects
                   WHERE type = 'EvidenceItem' AND body->>'mechanism_target' = %s AND status = 'active'
                   ORDER BY id""",
                (target,),
            ).fetchall()
        return [{"id": r[0], "type": r[1], "status": r[2], "body": r[3]} for r in rows]

    def count_by_type(self, obj_type: str) -> int:
        """Return count of objects of a given type."""
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM erik_core.objects WHERE type = %s AND status = 'active'",
                (obj_type,),
            ).fetchone()
        return row[0] if row else 0

    def delete(self, obj_id: str) -> None:
        """Delete an object by ID (for test cleanup)."""
        with get_connection() as conn:
            conn.execute("DELETE FROM erik_core.objects WHERE id = %s", (obj_id,))
            conn.commit()
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_evidence_store.py -v`
Expected: All PASS (or skip if DB not available)

- [ ] **Step 5: Commit**

```bash
git add scripts/evidence/ tests/test_evidence_store.py && git commit -m "feat: evidence store with PostgreSQL CRUD, upsert, and protocol layer queries"
```

---

## Task 4: Curated Intervention Seed

**Files:**
- Create: `data/seed/interventions.json`

This task creates the ~30 intervention objects as a JSON seed file. No code — pure data.

- [ ] **Step 1: Create the interventions seed file**

Write `data/seed/interventions.json` containing an array of intervention objects. Each object has: `id`, `name`, `intervention_class`, `targets`, `protocol_layer`, `route`, `intended_effects`, `known_risks`, and `body` with `regulatory_status`, `applicable_subtypes`, `dosing`, `drugbank_id`, `chembl_id`.

Include at minimum these interventions:

**Approved ALS therapies:**
- Riluzole (oral, anti-excitotoxicity, all subtypes)
- Edaravone IV and oral (free radical scavenger, all subtypes)
- Tofersen / QALSODY (ASO, SOD1 only)
- Sodesta (AAV gene therapy, SOD1 only)

**Phase 3 trials (highest priority):**
- Pridopidine (sigma-1R agonist, PREVAiLS, all subtypes)
- Masitinib (tyrosine kinase inhibitor, all subtypes)
- Jacifusen / ulefnersen (ASO, FUS only)

**Phase 1/2 (high interest):**
- VTx-002 (TDP-43 intrabody, PIONEER-ALS, all with TDP-43 pathology)
- ATLX-1282 (Alchemab/Lilly, mechanism TBD)

**Failed/withdrawn (for contradiction bundles):**
- AMX0035 / Relyvrio (withdrawn after PHOENIX Phase 3)
- BIIB078 (C9orf72 ASO, discontinued)
- Zilucoplan (C5 inhibitor, HEALEY negative primary)
- Verdiperstat (MPO inhibitor, HEALEY negative)
- DNL343 (eIF2B, HEALEY negative)
- Pridopidine HEALEY arm (negative at 24wk — contrasts with Phase 2 positive)

**Off-label / repurposing candidates:**
- Rapamycin/everolimus (mTOR inhibitor, autophagy)
- Memantine (NMDA antagonist)
- Perampanel (AMPA antagonist)
- Ibudilast (PDE inhibitor, neuroinflammation)
- Trehalose (mTOR-independent autophagy)

**Gene therapy / experimental:**
- AAV-BDNF (neurotrophic, preclinical)
- AAV-GDNF (neurotrophic, preclinical)
- STMN2 cryptic exon ASO (preclinical)
- UNC13A cryptic exon ASO (preclinical)

Each intervention must have `applicable_subtypes` and `regulatory_status` in its body.

- [ ] **Step 2: Commit**

```bash
mkdir -p data/seed && git add data/seed/interventions.json && git commit -m "feat: curated intervention seed — 25+ ALS therapies with mechanism, subtype, and regulatory data"
```

---

## Task 5: Curated Evidence Seed

**Files:**
- Create: `data/seed/layer_a_root_cause.json`
- Create: `data/seed/layer_b_pathology.json`
- Create: `data/seed/layer_c_circuit.json`
- Create: `data/seed/layer_d_regeneration.json`
- Create: `data/seed/layer_e_maintenance.json`
- Create: `data/seed/drug_design_targets.json`

This is the largest data task — ~110 evidence items. Each is a JSON object with: `id`, `claim`, `direction`, `source_refs`, `strength`, `body` (containing `protocol_layer`, `mechanism_target`, `intervention_ref`, `regulatory_status`, `applicable_subtypes`, `erik_eligible`, `pch_layer`, `trial_phase`, `modality`, `cohort_size`, `key_endpoints`).

- [ ] **Step 1: Create Layer A seed (root-cause suppression, ~25 items)**

Key items to include:
- Tofersen VALOR RCT data (Miller 2022 NEJM, pmid:35389978)
- Tofersen 3.5-year OLE (JAMA Neurol Dec 2025)
- Sodesta AAV approval and 1-year data
- BIIB078 failure analysis (Cell 2025)
- Jacifusen expanded access NfL data
- FUSION Phase 3 design
- VTx-002 TDP-43 intrabody (FDA Fast Track, PIONEER-ALS)
- TDP-43 aggregation biology (Neumann 2006, pmid:17023659)
- TDP-43 cryptic splicing (Ling 2015, PMC3094729)
- C9orf72 repeat RNA/DPR biology (Mori 2013, pmid:23393093)
- SOD1 toxic gain-of-function (Gurney 1994, pmid:8209258)
- AP-2 TDP-43 balance restoration (preclinical)

- [ ] **Step 2: Create Layer B seed (pathology reversal, ~25 items)**

Key items:
- UNC13A cryptic exon mechanism (Brown 2022, Ma 2022)
- STMN2 repression (Klim 2019, Melamed 2019)
- Pridopidine Phase 2 results (32% slowing)
- PREVAiLS Phase 3 design (500 patients)
- Sigma-1R biology (ER-mito tethering, calcium, proteostasis)
- Rapamycin in TDP-43 models
- TFEB activation strategies
- XBP1/UPR modulation (Hetz 2009)
- Stress granule LLPS biology (Patel 2015, Molliex 2015)
- Trehalose autophagy activation

- [ ] **Step 3: Create Layer C seed (circuit stabilization, ~20 items)**

Key items:
- Riluzole MOA and RCT data (Bensimon 1994, Lacomblez 1996)
- EAAT2/GLT-1 loss in ALS (Rothstein 1992, 1995)
- Masitinib Phase 3 survival subgroup
- Non-cell-autonomous toxicity (Boillee 2006, Di Giorgio 2008)
- Zilucoplan HEALEY (negative primary, biomarker question)
- Cortical hyperexcitability (Vucic 2008, Wainger 2014)
- Ibudilast neuroinflammation data
- Retigabine/Kv7 in iPSC models
- Perampanel AMPA antagonism case data

- [ ] **Step 4: Create Layer D + E seeds (regeneration ~15, maintenance ~15)**

Layer D: NMJ biology (Gould 2006), BDNF/GDNF delivery, axonal transport (KIF5A), NMJ reoccupation, iPSC motor neuron transplantation
Layer E: NfL monitoring (Benatar 2023), ALSFRS-R sensitivity, combination protocol design from oncology, respiratory monitoring, riluzole combination safety

- [ ] **Step 5: Create drug design targets seed**

```json
[
  {
    "id": "ddt:tdp43_rrm",
    "target_name": "TDP-43",
    "druggable_site": "RRM1/RRM2 RNA-binding domains",
    "strategy": "Prevent aberrant cytoplasmic aggregation or restore nuclear import",
    "pdb_ids": ["4BS2", "1WF0"],
    "alphafold_id": "AF-Q13148-F1",
    "compound_libraries": ["chembl_tardbp_actives", "zinc_rna_binding_modulators"],
    "notes": "RRM domains mediate RNA binding; disruption leads to splicing dysregulation"
  }
]
```

Include targets for: TDP-43 RRM domains, TDP-43 C-terminal aggregation region, UNC13A cryptic exon splice site (ASO design), STMN2 cryptic exon splice site, Sigma-1R binding pocket, SOD1 aggregation interface, stress granule condensation modulators.

**Note:** `drug_design_targets.json` is intentionally NOT loaded by the seed builder in Phase 1A. It is stored as structured input for the future computational drug design module (Phase 2+). The `test_seed_completeness.py` validates the file exists and has correct schema, but `load_seed()` does not process it.

- [ ] **Step 6: Commit all seed files**

```bash
git add data/seed/ && git commit -m "feat: curated evidence seed — 110 items across 5 protocol layers + drug design targets"
```

---

## Task 6: Seed Builder

**Files:**
- Create: `scripts/evidence/seed_builder.py`
- Create: `tests/test_seed_builder.py`
- Create: `tests/test_seed_completeness.py`

- [ ] **Step 1: Write the failing test for seed builder**

```python
# tests/test_seed_builder.py
import pytest


def test_load_interventions(db_available):
    from evidence.seed_builder import load_seed
    stats = load_seed()
    assert stats["interventions_loaded"] >= 20


def test_load_evidence_items(db_available):
    from evidence.seed_builder import load_seed
    stats = load_seed()
    assert stats["evidence_items_loaded"] >= 80


def test_all_layers_have_evidence(db_available):
    from evidence.seed_builder import load_seed
    from evidence.evidence_store import EvidenceStore
    load_seed()
    store = EvidenceStore()
    for layer in ["root_cause_suppression", "pathology_reversal",
                  "circuit_stabilization", "regeneration_reinnervation",
                  "adaptive_maintenance"]:
        items = store.query_by_protocol_layer(layer)
        assert len(items) >= 5, f"Layer {layer} has only {len(items)} items"


def test_seed_is_idempotent(db_available):
    from evidence.seed_builder import load_seed
    stats1 = load_seed()
    stats2 = load_seed()
    assert stats1["interventions_loaded"] == stats2["interventions_loaded"]
```

- [ ] **Step 2: Write the seed completeness test**

```python
# tests/test_seed_completeness.py
import json
from pathlib import Path

SEED_DIR = Path(__file__).parent.parent / "data" / "seed"


def test_all_seed_files_exist():
    expected = [
        "interventions.json",
        "layer_a_root_cause.json",
        "layer_b_pathology.json",
        "layer_c_circuit.json",
        "layer_d_regeneration.json",
        "layer_e_maintenance.json",
        "drug_design_targets.json",
    ]
    for f in expected:
        assert (SEED_DIR / f).exists(), f"Missing seed file: {f}"


def test_interventions_have_required_fields():
    data = json.loads((SEED_DIR / "interventions.json").read_text())
    for item in data:
        assert "id" in item, f"Missing id in intervention"
        assert "name" in item, f"Missing name in {item.get('id')}"
        assert "intervention_class" in item, f"Missing class in {item.get('id')}"
        assert "body" in item, f"Missing body in {item.get('id')}"
        assert "applicable_subtypes" in item["body"], f"Missing subtypes in {item.get('id')}"
        assert "regulatory_status" in item["body"], f"Missing regulatory in {item.get('id')}"


def test_evidence_items_have_required_fields():
    for layer_file in ["layer_a_root_cause.json", "layer_b_pathology.json",
                       "layer_c_circuit.json", "layer_d_regeneration.json",
                       "layer_e_maintenance.json"]:
        data = json.loads((SEED_DIR / layer_file).read_text())
        for item in data:
            assert "id" in item, f"Missing id in {layer_file}"
            assert "claim" in item, f"Missing claim in {item.get('id')}"
            assert "direction" in item, f"Missing direction in {item.get('id')}"
            assert "strength" in item, f"Missing strength in {item.get('id')}"
            body = item.get("body", {})
            assert "protocol_layer" in body, f"Missing protocol_layer in {item.get('id')}"
            assert "mechanism_target" in body, f"Missing mechanism_target in {item.get('id')}"
            assert "applicable_subtypes" in body, f"Missing subtypes in {item.get('id')}"
            assert "pch_layer" in body, f"Missing pch_layer in {item.get('id')}"


def test_erik_eligibility_assessed():
    """Every evidence item should have erik_eligible field."""
    for layer_file in ["layer_a_root_cause.json", "layer_b_pathology.json",
                       "layer_c_circuit.json", "layer_d_regeneration.json",
                       "layer_e_maintenance.json"]:
        data = json.loads((SEED_DIR / layer_file).read_text())
        for item in data:
            body = item.get("body", {})
            assert "erik_eligible" in body, f"Missing erik_eligible in {item.get('id')}"


def test_total_evidence_item_count():
    total = 0
    for layer_file in ["layer_a_root_cause.json", "layer_b_pathology.json",
                       "layer_c_circuit.json", "layer_d_regeneration.json",
                       "layer_e_maintenance.json"]:
        data = json.loads((SEED_DIR / layer_file).read_text())
        total += len(data)
    assert total >= 80, f"Only {total} evidence items — need at least 80"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_seed_builder.py tests/test_seed_completeness.py -v`
Expected: FAIL

- [ ] **Step 4: Write seed_builder.py**

```python
# scripts/evidence/seed_builder.py
"""Load curated evidence seed from JSON files into PostgreSQL.

Reads from data/seed/ directory and upserts all evidence items and
interventions into erik_core.objects via EvidenceStore.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ontology.evidence import EvidenceItem
from ontology.intervention import Intervention
from ontology.enums import (
    EvidenceDirection, EvidenceStrength, InterventionClass, ProtocolLayer,
)
from evidence.evidence_store import EvidenceStore

SEED_DIR = Path(__file__).parent.parent.parent / "data" / "seed"


def _parse_evidence_item(raw: dict) -> EvidenceItem:
    """Parse a raw JSON dict into an EvidenceItem."""
    return EvidenceItem(
        id=raw["id"],
        claim=raw["claim"],
        direction=EvidenceDirection(raw["direction"]),
        source_refs=raw.get("source_refs", []),
        strength=EvidenceStrength(raw["strength"]),
        supersedes_ref=raw.get("supersedes_ref"),
        notes=raw.get("notes", ""),
        body=raw.get("body", {}),
    )


def _parse_intervention(raw: dict) -> Intervention:
    """Parse a raw JSON dict into an Intervention."""
    return Intervention(
        id=raw["id"],
        name=raw["name"],
        intervention_class=InterventionClass(raw["intervention_class"]),
        targets=raw.get("targets", []),
        protocol_layer=ProtocolLayer(raw["protocol_layer"]) if raw.get("protocol_layer") else None,
        route=raw.get("route", ""),
        intended_effects=raw.get("intended_effects", []),
        known_risks=raw.get("known_risks", []),
        contraindications=raw.get("contraindications", []),
        body=raw.get("body", {}),
    )


def load_seed() -> dict[str, int]:
    """Load all seed files into the evidence store. Returns counts."""
    store = EvidenceStore()
    stats = {"interventions_loaded": 0, "evidence_items_loaded": 0}

    # Load interventions
    int_path = SEED_DIR / "interventions.json"
    if int_path.exists():
        raw_interventions = json.loads(int_path.read_text())
        for raw in raw_interventions:
            intervention = _parse_intervention(raw)
            store.upsert_intervention(intervention)
            stats["interventions_loaded"] += 1

    # Load evidence items from all layer files
    layer_files = [
        "layer_a_root_cause.json",
        "layer_b_pathology.json",
        "layer_c_circuit.json",
        "layer_d_regeneration.json",
        "layer_e_maintenance.json",
    ]
    for layer_file in layer_files:
        path = SEED_DIR / layer_file
        if path.exists():
            raw_items = json.loads(path.read_text())
            for raw in raw_items:
                item = _parse_evidence_item(raw)
                store.upsert_evidence_item(item)
                stats["evidence_items_loaded"] += 1

    return stats


if __name__ == "__main__":
    stats = load_seed()
    print(f"Loaded {stats['interventions_loaded']} interventions")
    print(f"Loaded {stats['evidence_items_loaded']} evidence items")
```

- [ ] **Step 5: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/evidence/seed_builder.py tests/test_seed_builder.py tests/test_seed_completeness.py && git commit -m "feat: seed builder loads curated evidence into PostgreSQL"
```

---

## Task 7: Final Verification

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify evidence counts in database**

```bash
cd /Users/logannye/.openclaw/erik && PYTHONPATH=scripts conda run -n erik-core python -c "
from evidence.seed_builder import load_seed
from evidence.evidence_store import EvidenceStore
stats = load_seed()
store = EvidenceStore()
print(f'Interventions: {stats[\"interventions_loaded\"]}')
print(f'Evidence items: {stats[\"evidence_items_loaded\"]}')
for layer in ['root_cause_suppression', 'pathology_reversal', 'circuit_stabilization', 'regeneration_reinnervation', 'adaptive_maintenance']:
    items = store.query_by_protocol_layer(layer)
    print(f'  {layer}: {len(items)} items')
print(f'Total objects in DB: {store.count_by_type(\"EvidenceItem\")} evidence + {store.count_by_type(\"Intervention\")} interventions')
"
```

- [ ] **Step 3: Push to GitHub**

```bash
cd /Users/logannye/.openclaw/erik && git push
```

---

## Summary

After completing all 7 tasks, Phase 1A delivers:

- **`EvidenceStrength` enum** with 5 levels (strong, moderate, emerging, preclinical, unknown)
- **Extended `InterventionClass`** with gene_therapy, cell_therapy, peptide
- **Updated `EvidenceItem`** with string-typed strength, supersedes_ref
- **16 canonical ALS drug targets** with UniProt IDs, druggability, subtype mapping
- **~30 intervention objects** covering approved, Phase 3, Phase 1/2, off-label, and experimental therapies
- **~110 curated evidence items** across all 5 protocol layers
- **~10 computational drug design targets** with PDB structures and compound libraries
- **Evidence store** with PostgreSQL CRUD, upsert, and protocol-layer queries
- **Seed builder** that loads JSON seed into database idempotently

**What comes next (Phase 1B):** Automated connectors (PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, DrugBank) to keep the evidence fabric current and expanding.
