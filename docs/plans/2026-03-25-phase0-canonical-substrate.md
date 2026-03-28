# Phase 0: Canonical Substrate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Erik's foundational data layer — PostgreSQL schema, Pydantic canonical models, ALS ontology, patient trajectory ingestion, and Erik Draper's clinical data as the first materialized disease state snapshot.

**Architecture:** Single-process Python application (like Galen) in a monorepo at `~/.openclaw/erik/`. PostgreSQL as single source of truth with two schemas: `erik_core` (canonical objects, KG, patient trajectories) and `erik_ops` (operational state, audit log, config). Pydantic v2 models replace the spec's Protobuf for pragmatic solo-dev velocity. All canonical objects conform to a shared base envelope with provenance, uncertainty, time, and privacy fields.

**Tech Stack:** Python 3.12 (conda `erik-core`), PostgreSQL 17 (shared instance with Galen), Pydantic v2, psycopg3, pytest, pgvector extension.

**Scope:** This plan covers the spec's Phase 0 (Canonical Substrate) and the first half of Phase 1 (Patient Ingestion). It produces the ontology, database schema, canonical models, and Erik Draper's structured clinical trajectory. It does NOT cover the world model, planner, RL loop, daemons, or cure protocol generation — those are separate plans.

**Deferred types:** The spec (section 8.3) mandates 29 canonical types. This plan implements 25 of them. The following 14 types are deferred to Phase 2 (World Model MVP) because they are latent state sub-components that only become meaningful once the state estimation machinery exists: `MotorSystemCompartment`, `MotorUnitState`, `CorticospinalState`, `LowerMotorNeuronState`, `MuscleIntegrityState`, `BulbarFunctionState`, `ProteostasisState`, `NucleocytoplasmicTransportState`, `MitochondrialState`, `AxonalTransportState`, `InflammatoryState`, `BiomarkerPanel`, `SubtypeHypothesis`, `ProtocolExecutionRecord`. Also deferred: `BranchEvaluation` (spec 30.26).

**Reference documents:**
- `/Users/logannye/Desktop/Erik/erik_als_technical_specification.md` (sections 4, 8, 10, 11, 23, 29, 30)
- `/Users/logannye/Desktop/Erik/Amyotrophic Lateral Sclerosis_ A First-Principles Mechanistic Synthesis.pdf`
- `/Users/logannye/Desktop/Erik/Open and Public Datasets for Amyotrophic Lateral Sclerosis.pdf`
- Galen codebase at `/Users/logannye/.openclaw/workspace/` (architectural reference)

---

## File Structure

```
/Users/logannye/.openclaw/erik/
├── scripts/
│   ├── __init__.py
│   ├── db/
│   │   ├── __init__.py
│   │   ├── core_schema.sql          # erik_core schema DDL (KG + patient + canonical objects)
│   │   ├── ops_schema.sql           # erik_ops schema DDL (audit, config, operational)
│   │   ├── pool.py                  # psycopg3 connection pool (shared)
│   │   └── migrate.py               # Schema migration runner
│   ├── ontology/
│   │   ├── __init__.py
│   │   ├── enums.py                 # All canonical enums (ObjectStatus, SubtypeClass, etc.)
│   │   ├── base.py                  # BaseEnvelope Pydantic model (shared by all objects)
│   │   ├── patient.py               # Patient, ALSTrajectory models
│   │   ├── observation.py           # Observation model (labs, EMG, MRI, spirometry, etc.)
│   │   ├── interpretation.py        # Interpretation, EtiologicDriverProfile models
│   │   ├── state.py                 # DiseaseStateSnapshot + all latent state factor models
│   │   ├── intervention.py          # Intervention model
│   │   ├── evidence.py              # EvidenceBundle, EvidenceItem models
│   │   ├── discovery.py             # MechanismHypothesis, ExperimentProposal models
│   │   ├── protocol.py              # CureProtocolCandidate, MonitoringPlan models
│   │   ├── meta.py                  # LearningEpisode, ErrorRecord, ImprovementProposal, Branch
│   │   ├── relations.py             # Typed relation vocabulary + guards
│   │   └── registry.py              # Type registry: name → model class mapping
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── lab_results.py           # Parse lab values with reference ranges
│   │   └── patient_builder.py       # Build Patient + ALSTrajectory from clinical data
│   ├── audit/
│   │   ├── __init__.py
│   │   └── event_log.py             # Append-only audit event logger
│   └── config/
│       ├── __init__.py
│       └── loader.py                # Hot-reloadable JSON config (Galen pattern)
├── data/
│   ├── erik_config.json             # Hot-reloadable config
│   └── erik_patient_data.json       # Erik Draper's structured clinical data (from ingestion)
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Shared fixtures (test DB, sample objects)
│   ├── test_enums.py
│   ├── test_base_envelope.py
│   ├── test_patient_models.py
│   ├── test_observation_models.py
│   ├── test_interpretation_models.py
│   ├── test_state_models.py
│   ├── test_evidence_models.py
│   ├── test_protocol_models.py
│   ├── test_meta_models.py
│   ├── test_relations.py
│   ├── test_registry.py
│   ├── test_db_schema.py
│   ├── test_db_pool.py
│   ├── test_audit_log.py
│   ├── test_config_loader.py
│   ├── test_ingestion_labs.py
│   ├── test_ingestion_patient.py
│   └── test_erik_trajectory.py
├── docs/
│   └── plans/
│       └── 2026-03-25-phase0-canonical-substrate.md  # This file
├── CLAUDE.md
├── pyproject.toml
└── README.md
```

Each file has one clear responsibility. The `ontology/` module owns all canonical types. The `db/` module owns schema and connection. The `ingestion/` module owns parsing clinical documents into canonical objects. The `audit/` module owns the immutable event log. Tests mirror the source structure.

---

## Task 1: Repository Scaffold + Environment

**Files:**
- Create: `/Users/logannye/.openclaw/erik/pyproject.toml`
- Create: `/Users/logannye/.openclaw/erik/CLAUDE.md`
- Create: `/Users/logannye/.openclaw/erik/scripts/__init__.py`
- Create: all `__init__.py` files for subpackages
- Create: `/Users/logannye/.openclaw/erik/tests/__init__.py`

- [ ] **Step 1: Create conda environment**

```bash
conda create -n erik-core python=3.12 -y
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[project]
name = "erik"
version = "0.1.0"
description = "Autonomous causal research and cure-protocol engine for ALS"
requires-python = ">=3.12"
dependencies = [
    "pydantic>=2.10",
    "psycopg[binary]>=3.2",
    "psycopg_pool>=3.2",
    "pytest>=8.0",
    "python-dateutil>=2.9",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["scripts"]
```

- [ ] **Step 3: Create CLAUDE.md**

```markdown
# Erik — ALS Causal Research Engine

## Project Structure
- `scripts/` — All Python source modules
- `scripts/ontology/` — Canonical Pydantic models (base envelope, ALS types, relations)
- `scripts/db/` — PostgreSQL schema and connection pool
- `scripts/ingestion/` — Clinical document parsing and patient trajectory building
- `scripts/audit/` — Immutable append-only event log
- `scripts/config/` — Hot-reloadable JSON config
- `tests/` — Pytest test suite (mirrors scripts/ structure)
- `data/` — Runtime config and structured patient data

## Key Conventions
- Python env: `/opt/homebrew/Caskroom/miniconda/base/envs/erik-core/bin/python`
- Database: PostgreSQL `erik_kg` with schemas `erik_core` and `erik_ops`
- All canonical objects inherit from `BaseEnvelope` (scripts/ontology/base.py)
- Entity IDs: `f"{type}:{name}".lower().replace(" ", "_")` (Galen convention)
- Import paths: `from ontology.base import BaseEnvelope` (scripts/ is on sys.path)
- Config file: `data/erik_config.json` (hot-reloaded, never restart for config changes)
- NEVER use sqlite3 — PostgreSQL is the single source of truth
- OBSERVATIONAL_RELATIONSHIP_TYPES must never be upgraded to L3 (causal)
- Tests: TDD — write failing test first, then minimal implementation
```

- [ ] **Step 4: Create all __init__.py files**

Create empty `__init__.py` in: `scripts/`, `scripts/db/`, `scripts/ontology/`, `scripts/ingestion/`, `scripts/audit/`, `scripts/config/`, `tests/`

- [ ] **Step 5: Install dependencies**

```bash
conda activate erik-core && pip install pydantic "psycopg[binary]" psycopg_pool pytest python-dateutil
```

- [ ] **Step 6: Initialize git repo**

```bash
cd /Users/logannye/.openclaw/erik && git init && git add -A && git commit -m "chore: initial repo scaffold with pyproject.toml and directory structure"
```

---

## Task 2: Canonical Enums

**Files:**
- Create: `scripts/ontology/enums.py`
- Test: `tests/test_enums.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_enums.py
from ontology.enums import (
    ObjectStatus, ConfidenceBand, PrivacyClass, ApprovalState,
    EvidenceDirection, ActionClass, ALSOnsetRegion, SubtypeClass,
    ProtocolLayer, ObservationKind, InterpretationKind, InterventionClass,
    PCHLayer, RelationCategory,
)


def test_object_status_values():
    assert ObjectStatus.ACTIVE.value == "active"
    assert ObjectStatus.SUPERSEDED.value == "superseded"
    assert ObjectStatus.DEPRECATED.value == "deprecated"
    assert ObjectStatus.DELETED_LOGICALLY.value == "deleted_logically"


def test_subtype_class_values():
    assert SubtypeClass.SOD1.value == "sod1"
    assert SubtypeClass.C9ORF72.value == "c9orf72"
    assert SubtypeClass.FUS.value == "fus"
    assert SubtypeClass.TARDBP.value == "tardbp"
    assert SubtypeClass.SPORADIC_TDP43.value == "sporadic_tdp43"
    assert SubtypeClass.GLIA_AMPLIFIED.value == "glia_amplified"
    assert SubtypeClass.MIXED.value == "mixed"
    assert SubtypeClass.UNRESOLVED.value == "unresolved"


def test_protocol_layer_ordering():
    layers = list(ProtocolLayer)
    names = [l.value for l in layers]
    assert names == [
        "root_cause_suppression",
        "pathology_reversal",
        "circuit_stabilization",
        "regeneration_reinnervation",
        "adaptive_maintenance",
    ]


def test_pch_layer_values():
    assert PCHLayer.L1_ASSOCIATIONAL.value == 1
    assert PCHLayer.L2_INTERVENTIONAL.value == 2
    assert PCHLayer.L3_COUNTERFACTUAL.value == 3


def test_als_onset_region_has_all_options():
    names = {r.value for r in ALSOnsetRegion}
    assert names == {"upper_limb", "lower_limb", "bulbar", "respiratory", "multifocal", "unknown"}


def test_observation_kind_includes_clinical_types():
    names = {k.value for k in ObservationKind}
    assert "lab_result" in names
    assert "emg_feature" in names
    assert "respiratory_metric" in names
    assert "genomic_result" in names
    assert "imaging_finding" in names
    assert "functional_score" in names
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_enums.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ontology/enums.py
"""Canonical enum definitions for the Erik ALS ontology.

Ref: Technical Specification sections 8.2, 30.1
"""
from enum import Enum, IntEnum


class ObjectStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DEPRECATED = "deprecated"
    DELETED_LOGICALLY = "deleted_logically"


class ConfidenceBand(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class PrivacyClass(str, Enum):
    PUBLIC = "public"
    RESTRICTED = "restricted"
    DEIDENTIFIED = "deidentified"
    PHI = "phi"


class ApprovalState(str, Enum):
    NOT_REQUIRED = "not_required"
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EvidenceDirection(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    MIXED = "mixed"
    INSUFFICIENT = "insufficient"


class ActionClass(str, Enum):
    READ_ONLY = "read_only"
    SIMULATION_ONLY = "simulation_only"
    RECOMMENDATION_GENERATION = "recommendation_generation"
    WORKFLOW_GENERATION = "workflow_generation"
    RESEARCH_EXECUTION = "research_execution"
    PROMOTION_CONTROL = "promotion_control"
    ROLLBACK_CONTROL = "rollback_control"


class ALSOnsetRegion(str, Enum):
    UPPER_LIMB = "upper_limb"
    LOWER_LIMB = "lower_limb"
    BULBAR = "bulbar"
    RESPIRATORY = "respiratory"
    MULTIFOCAL = "multifocal"
    UNKNOWN = "unknown"


class SubtypeClass(str, Enum):
    SOD1 = "sod1"
    C9ORF72 = "c9orf72"
    FUS = "fus"
    TARDBP = "tardbp"
    SPORADIC_TDP43 = "sporadic_tdp43"
    GLIA_AMPLIFIED = "glia_amplified"
    MIXED = "mixed"
    UNRESOLVED = "unresolved"


class ProtocolLayer(str, Enum):
    ROOT_CAUSE_SUPPRESSION = "root_cause_suppression"
    PATHOLOGY_REVERSAL = "pathology_reversal"
    CIRCUIT_STABILIZATION = "circuit_stabilization"
    REGENERATION_REINNERVATION = "regeneration_reinnervation"
    ADAPTIVE_MAINTENANCE = "adaptive_maintenance"


class PCHLayer(IntEnum):
    """Pearl Causal Hierarchy layers."""
    L1_ASSOCIATIONAL = 1
    L2_INTERVENTIONAL = 2
    L3_COUNTERFACTUAL = 3


class ObservationKind(str, Enum):
    LAB_RESULT = "lab_result"
    EMG_FEATURE = "emg_feature"
    RESPIRATORY_METRIC = "respiratory_metric"
    SPEECH_METRIC = "speech_metric"
    GENOMIC_RESULT = "genomic_result"
    IMAGING_FINDING = "imaging_finding"
    OMICS_MEASUREMENT = "omics_measurement"
    WORKFLOW_SIGNAL = "workflow_signal"
    FUNCTIONAL_SCORE = "functional_score"
    VITAL_SIGN = "vital_sign"
    WEIGHT_MEASUREMENT = "weight_measurement"
    MEDICATION_EVENT = "medication_event"
    PHYSICAL_EXAM_FINDING = "physical_exam_finding"


class InterpretationKind(str, Enum):
    DIAGNOSIS = "diagnosis"
    SUBTYPE_INFERENCE = "subtype_inference"
    PROGRESSION_ESTIMATE = "progression_estimate"
    REVERSIBILITY_ESTIMATE = "reversibility_estimate"
    TREATMENT_RESPONSE = "treatment_response"
    TARGET_ENGAGEMENT = "target_engagement"
    ELIGIBILITY_ASSESSMENT = "eligibility_assessment"
    RESPIRATORY_DECLINE_RISK = "respiratory_decline_risk"


class InterventionClass(str, Enum):
    DRUG = "drug"
    ASO = "aso"
    GENE_EDITING = "gene_editing"
    SMALL_MOLECULE = "small_molecule"
    GENE_SILENCING = "gene_silencing"
    SUPPORTIVE_CARE = "supportive_care"
    RESPIRATORY_SUPPORT = "respiratory_support"
    FEEDING_SUPPORT = "feeding_support"
    REHABILITATION = "rehabilitation"
    WORKFLOW_ACTION = "workflow_action"
    WET_LAB_PERTURBATION = "wet_lab_perturbation"
    TRIAL_ASSIGNMENT = "trial_assignment"


class RelationCategory(str, Enum):
    STRUCTURAL = "structural"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    EVIDENTIAL = "evidential"
    THERAPEUTIC = "therapeutic"
    GOVERNANCE = "governance"


class SourceSystem(str, Enum):
    EHR = "ehr"
    REGISTRY = "registry"
    LIMS = "lims"
    OMICS = "omics"
    TRIAL = "trial"
    MANUAL = "manual"
    MODEL = "model"
    WORKFLOW = "workflow"
    LITERATURE = "literature"
    DATABASE = "database"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_enums.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology/enums.py tests/test_enums.py && git commit -m "feat: canonical enum definitions for ALS ontology"
```

---

## Task 3: Base Object Envelope

**Files:**
- Create: `scripts/ontology/base.py`
- Test: `tests/test_base_envelope.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base_envelope.py
import json
from datetime import datetime, timezone
from ontology.base import (
    BaseEnvelope, TimeFields, Provenance, Uncertainty, Privacy,
)
from ontology.enums import ObjectStatus, PrivacyClass, ConfidenceBand, SourceSystem


def test_base_envelope_creation():
    obj = BaseEnvelope(
        id="test_001",
        type="TestType",
        body={"key": "value"},
    )
    assert obj.id == "test_001"
    assert obj.type == "TestType"
    assert obj.schema_version == "1.0"
    assert obj.status == ObjectStatus.ACTIVE
    assert obj.body == {"key": "value"}


def test_base_envelope_defaults():
    obj = BaseEnvelope(id="x", type="T", body={})
    assert obj.status == ObjectStatus.ACTIVE
    assert obj.privacy.classification == PrivacyClass.RESTRICTED
    assert obj.uncertainty.confidence is None
    assert obj.time.recorded_at is not None


def test_time_fields_auto_populate():
    tf = TimeFields()
    assert tf.recorded_at is not None
    assert isinstance(tf.recorded_at, datetime)


def test_provenance_required_fields():
    p = Provenance(
        source_system=SourceSystem.EHR,
        asserted_by="clinician_001",
    )
    assert p.source_system == SourceSystem.EHR
    assert p.source_artifact_id is None
    assert p.trace_id is None


def test_uncertainty_with_sources():
    u = Uncertainty(
        confidence=0.84,
        confidence_band=ConfidenceBand.HIGH,
        sources=["missing_biomarker", "subtype_ambiguity"],
    )
    assert u.confidence == 0.84
    assert len(u.sources) == 2


def test_base_envelope_json_roundtrip():
    obj = BaseEnvelope(
        id="rt_001",
        type="RoundTrip",
        body={"nested": {"data": [1, 2, 3]}},
        provenance=Provenance(
            source_system=SourceSystem.MODEL,
            asserted_by="erik_v0.1",
        ),
    )
    json_str = obj.model_dump_json()
    parsed = json.loads(json_str)
    assert parsed["id"] == "rt_001"
    assert parsed["provenance"]["source_system"] == "model"
    restored = BaseEnvelope.model_validate_json(json_str)
    assert restored.id == obj.id
    assert restored.body == obj.body


def test_base_envelope_rejects_empty_id():
    import pytest
    with pytest.raises(Exception):
        BaseEnvelope(id="", type="T", body={})


def test_base_envelope_rejects_empty_type():
    import pytest
    with pytest.raises(Exception):
        BaseEnvelope(id="x", type="", body={})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_base_envelope.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ontology/base.py
"""Base object envelope shared by all canonical Erik objects.

Ref: Technical Specification sections 4.1, 8.5
Every canonical object MUST conform to this stable base envelope.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field, field_validator

from ontology.enums import (
    ConfidenceBand,
    ObjectStatus,
    PrivacyClass,
    SourceSystem,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TimeFields(BaseModel):
    """Temporal coordinates for a canonical object."""
    observed_at: datetime | None = None
    effective_at: datetime | None = None
    recorded_at: datetime = Field(default_factory=_utcnow)
    valid_from: datetime | None = None
    valid_to: datetime | None = None


class Provenance(BaseModel):
    """Origin and trace information."""
    source_system: SourceSystem = SourceSystem.MANUAL
    source_artifact_id: str | None = None
    asserted_by: str | None = None
    trace_id: str | None = None


class Uncertainty(BaseModel):
    """Decomposed uncertainty attached to an object."""
    confidence: float | None = None
    confidence_band: ConfidenceBand | None = None
    sources: list[str] = Field(default_factory=list)


class Privacy(BaseModel):
    """Privacy classification."""
    classification: PrivacyClass = PrivacyClass.RESTRICTED


class BaseEnvelope(BaseModel):
    """Base envelope for all canonical Erik objects.

    Spec invariant: every clinically or discovery-consequential output
    MUST carry provenance, uncertainty, model version, trace ID, and
    approval state.
    """
    id: str = Field(..., min_length=1)
    type: str = Field(..., min_length=1)
    schema_version: str = "1.0"
    tenant_id: str = "erik_default"
    status: ObjectStatus = ObjectStatus.ACTIVE
    time: TimeFields = Field(default_factory=TimeFields)
    provenance: Provenance = Field(default_factory=Provenance)
    uncertainty: Uncertainty = Field(default_factory=Uncertainty)
    privacy: Privacy = Field(default_factory=Privacy)
    body: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def id_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("id must not be blank")
        return v

    @field_validator("type")
    @classmethod
    def type_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("type must not be blank")
        return v

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat()}}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_base_envelope.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology/base.py tests/test_base_envelope.py && git commit -m "feat: BaseEnvelope with time, provenance, uncertainty, privacy"
```

---

## Task 4: Patient + ALSTrajectory Models

**Files:**
- Create: `scripts/ontology/patient.py`
- Test: `tests/test_patient_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_patient_models.py
from datetime import date, datetime, timezone
from ontology.patient import Patient, ALSTrajectory, ALSFRSRScore
from ontology.enums import ALSOnsetRegion, PrivacyClass


def test_patient_creation():
    p = Patient(
        id="patient:erik_draper",
        patient_key="draper_erik_001",
        birth_year=1958,
        sex_at_birth="male",
        family_history_of_als=False,
        family_history_notes="Mother with Alzheimer's; maternal family AD history. No MND.",
    )
    assert p.type == "Patient"
    assert p.birth_year == 1958
    assert p.privacy.classification == PrivacyClass.PHI


def test_als_trajectory_creation():
    t = ALSTrajectory(
        id="traj:draper_001",
        patient_ref="patient:erik_draper",
        onset_date=date(2025, 1, 15),
        diagnosis_date=date(2026, 3, 6),
        onset_region=ALSOnsetRegion.LOWER_LIMB,
        episode_status="active",
    )
    assert t.type == "ALSTrajectory"
    assert t.onset_region == ALSOnsetRegion.LOWER_LIMB
    assert t.diagnosis_date == date(2026, 3, 6)
    assert t.linked_observation_refs == []


def test_alsfrs_r_score():
    score = ALSFRSRScore(
        speech=4, salivation=4, swallowing=4,
        handwriting=4, cutting_food=4, dressing_hygiene=3,
        turning_in_bed=3, walking=3, climbing_stairs=2,
        dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
        assessment_date=date(2026, 2, 6),
    )
    assert score.total == 43
    assert score.bulbar_subscore == 12
    assert score.fine_motor_subscore == 11
    assert score.gross_motor_subscore == 8
    assert score.respiratory_subscore == 12


def test_alsfrs_r_decline_rate():
    score = ALSFRSRScore(
        speech=4, salivation=4, swallowing=4,
        handwriting=4, cutting_food=4, dressing_hygiene=3,
        turning_in_bed=3, walking=3, climbing_stairs=2,
        dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
        assessment_date=date(2026, 2, 6),
    )
    onset = date(2025, 1, 15)
    rate = score.decline_rate_from_onset(onset)
    # (48 - 43) / 12.7 months ≈ -0.39 per month
    assert -0.50 < rate < -0.30


def test_trajectory_json_roundtrip():
    t = ALSTrajectory(
        id="traj:rt_001",
        patient_ref="patient:test",
        onset_date=date(2025, 1, 1),
        onset_region=ALSOnsetRegion.UPPER_LIMB,
    )
    json_str = t.model_dump_json()
    restored = ALSTrajectory.model_validate_json(json_str)
    assert restored.id == t.id
    assert restored.onset_date == t.onset_date
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_patient_models.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ontology/patient.py
"""Patient and ALSTrajectory canonical models.

Ref: Technical Specification sections 30.2, 30.3
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from ontology.base import BaseEnvelope, Privacy
from ontology.enums import ALSOnsetRegion, PrivacyClass


class ALSFRSRScore(BaseModel):
    """ALSFRS-R: 12-item functional rating scale (0-48, higher = better)."""
    # Bulbar (0-4 each)
    speech: int = Field(ge=0, le=4)
    salivation: int = Field(ge=0, le=4)
    swallowing: int = Field(ge=0, le=4)
    # Fine motor (0-4 each)
    handwriting: int = Field(ge=0, le=4)
    cutting_food: int = Field(ge=0, le=4)
    dressing_hygiene: int = Field(ge=0, le=4)
    # Gross motor (0-4 each)
    turning_in_bed: int = Field(ge=0, le=4)
    walking: int = Field(ge=0, le=4)
    climbing_stairs: int = Field(ge=0, le=4)
    # Respiratory (0-4 each)
    dyspnea: int = Field(ge=0, le=4)
    orthopnea: int = Field(ge=0, le=4)
    respiratory_insufficiency: int = Field(ge=0, le=4)

    assessment_date: date

    @computed_field
    @property
    def total(self) -> int:
        return sum([
            self.speech, self.salivation, self.swallowing,
            self.handwriting, self.cutting_food, self.dressing_hygiene,
            self.turning_in_bed, self.walking, self.climbing_stairs,
            self.dyspnea, self.orthopnea, self.respiratory_insufficiency,
        ])

    @computed_field
    @property
    def bulbar_subscore(self) -> int:
        return self.speech + self.salivation + self.swallowing

    @computed_field
    @property
    def fine_motor_subscore(self) -> int:
        return self.handwriting + self.cutting_food + self.dressing_hygiene

    @computed_field
    @property
    def gross_motor_subscore(self) -> int:
        return self.turning_in_bed + self.walking + self.climbing_stairs

    @computed_field
    @property
    def respiratory_subscore(self) -> int:
        return self.dyspnea + self.orthopnea + self.respiratory_insufficiency

    def decline_rate_from_onset(self, onset_date: date) -> float:
        """Points lost per month since onset. Negative = declining."""
        months = (self.assessment_date - onset_date).days / 30.44
        if months <= 0:
            return 0.0
        return -(48 - self.total) / months


class Patient(BaseEnvelope):
    """A person linked to one or more ALS trajectories. Always PHI."""
    type: str = "Patient"
    privacy: Privacy = Field(default_factory=lambda: Privacy(classification=PrivacyClass.PHI))

    # Body fields (flattened from spec's nested body for Pydantic ergonomics)
    patient_key: str
    birth_year: int
    sex_at_birth: str
    family_history_of_als: bool = False
    family_history_notes: str = ""
    consent_profiles: list[str] = Field(default_factory=list)
    preference_profile_ref: Optional[str] = None
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    comorbidities: list[str] = Field(default_factory=list)


class ALSTrajectory(BaseEnvelope):
    """Canonical longitudinal container for one ALS disease course."""
    type: str = "ALSTrajectory"
    privacy: Privacy = Field(default_factory=lambda: Privacy(classification=PrivacyClass.PHI))

    patient_ref: str
    onset_date: Optional[date] = None
    diagnosis_date: Optional[date] = None
    onset_region: ALSOnsetRegion = ALSOnsetRegion.UNKNOWN
    episode_status: str = "active"
    site_of_care_refs: list[str] = Field(default_factory=list)
    etiologic_driver_profile_ref: Optional[str] = None
    current_state_snapshot_ref: Optional[str] = None
    alsfrs_r_scores: list[ALSFRSRScore] = Field(default_factory=list)
    linked_observation_refs: list[str] = Field(default_factory=list)
    linked_intervention_refs: list[str] = Field(default_factory=list)
    linked_outcome_refs: list[str] = Field(default_factory=list)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_patient_models.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology/patient.py tests/test_patient_models.py && git commit -m "feat: Patient, ALSTrajectory, ALSFRSRScore models"
```

---

## Task 5: Observation Model

**Files:**
- Create: `scripts/ontology/observation.py`
- Test: `tests/test_observation_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_observation_models.py
from datetime import date, datetime, timezone
from ontology.observation import Observation, LabResult, EMGFinding, RespiratoryMetric, ImagingFinding, PhysicalExamFinding
from ontology.enums import ObservationKind, SourceSystem


def test_lab_result_creation():
    lab = LabResult(
        name="Neurofilament Light",
        value=5.82,
        unit="pg/mL",
        reference_low=0.0,
        reference_high=3.65,
        collection_date=date(2026, 2, 20),
    )
    assert lab.is_abnormal is True
    assert lab.is_high is True
    assert lab.is_low is False


def test_lab_result_normal():
    lab = LabResult(
        name="Creatine Kinase",
        value=200.0,
        unit="U/L",
        reference_low=51.0,
        reference_high=298.0,
        collection_date=date(2026, 2, 20),
    )
    assert lab.is_abnormal is False


def test_observation_with_lab():
    obs = Observation(
        id="obs:nfl_001",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.LAB_RESULT,
        name="Neurofilament Light, Plasma",
        lab_result=LabResult(
            name="Neurofilament Light",
            value=5.82,
            unit="pg/mL",
            reference_low=0.0,
            reference_high=3.65,
            collection_date=date(2026, 2, 20),
        ),
    )
    assert obs.type == "Observation"
    assert obs.lab_result.is_abnormal is True


def test_emg_finding():
    emg = EMGFinding(
        study_date=date(2026, 3, 6),
        summary="Widespread active and chronic motor axon loss changes supportive of ALS",
        regions_with_active_denervation=["left_leg"],
        regions_with_chronic_denervation=["left_leg"],
        supports_als=True,
    )
    assert emg.supports_als is True


def test_respiratory_metric():
    resp = RespiratoryMetric(
        measurement_date=date(2026, 3, 9),
        fvc_percent_predicted=100.0,
        fvc_liters_sitting=5.0,
        fvc_liters_supine=4.8,
    )
    assert resp.supine_drop_percent is not None
    assert 0 < resp.supine_drop_percent < 10


def test_observation_json_roundtrip():
    obs = Observation(
        id="obs:rt_001",
        subject_ref="traj:test",
        observation_kind=ObservationKind.VITAL_SIGN,
        name="Blood Pressure",
    )
    json_str = obs.model_dump_json()
    restored = Observation.model_validate_json(json_str)
    assert restored.id == obs.id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_observation_models.py -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/ontology/observation.py
"""Observation canonical model — raw or minimally transformed measured facts.

Ref: Technical Specification sections 8.2 (Observation namespace), 30.4
"""
from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, Field, computed_field

from ontology.base import BaseEnvelope
from ontology.enums import ObservationKind, SourceSystem


class LabResult(BaseModel):
    """A single lab value with reference range."""
    name: str
    value: float
    unit: str
    reference_low: Optional[float] = None
    reference_high: Optional[float] = None
    collection_date: date
    method: str = ""
    notes: str = ""

    @computed_field
    @property
    def is_high(self) -> bool:
        if self.reference_high is None:
            return False
        return self.value > self.reference_high

    @computed_field
    @property
    def is_low(self) -> bool:
        if self.reference_low is None:
            return False
        return self.value < self.reference_low

    @computed_field
    @property
    def is_abnormal(self) -> bool:
        return self.is_high or self.is_low


class EMGFinding(BaseModel):
    """Electromyography study results."""
    study_date: date
    summary: str
    performing_physician: str = ""
    regions_with_active_denervation: list[str] = Field(default_factory=list)
    regions_with_chronic_denervation: list[str] = Field(default_factory=list)
    regions_with_reinnervation: list[str] = Field(default_factory=list)
    fasciculation_potentials: list[str] = Field(default_factory=list)
    nerve_conduction_abnormalities: list[str] = Field(default_factory=list)
    supports_als: bool = False
    raw_report_ref: Optional[str] = None


class RespiratoryMetric(BaseModel):
    """Spirometry / respiratory function measurements."""
    measurement_date: date
    fvc_percent_predicted: Optional[float] = None
    fvc_liters_sitting: Optional[float] = None
    fvc_liters_supine: Optional[float] = None
    fev1_liters: Optional[float] = None
    snip: Optional[float] = None
    mip: Optional[float] = None
    notes: str = ""

    @computed_field
    @property
    def supine_drop_percent(self) -> Optional[float]:
        if self.fvc_liters_sitting and self.fvc_liters_supine and self.fvc_liters_sitting > 0:
            return ((self.fvc_liters_sitting - self.fvc_liters_supine) / self.fvc_liters_sitting) * 100
        return None


class ImagingFinding(BaseModel):
    """MRI, CT, or other imaging results."""
    study_date: date
    modality: str  # "mri_brain", "mri_cervical", "ct", etc.
    summary: str
    findings: list[str] = Field(default_factory=list)
    incidental_findings: list[str] = Field(default_factory=list)
    als_relevant: bool = False
    raw_report_ref: Optional[str] = None


class PhysicalExamFinding(BaseModel):
    """Structured physical/neurological examination finding."""
    exam_date: date
    category: str  # "motor", "reflex", "tone", "sensation", "gait"
    region: str
    finding: str
    laterality: str = "bilateral"  # "left", "right", "bilateral"
    value: Optional[str] = None  # e.g., "4+", "3+", "A2"
    notes: str = ""


class Observation(BaseEnvelope):
    """A raw or minimally transformed measured fact.

    Spec invariant: Observations are append-only. Observation and interpretation
    MUST remain separate object classes.
    """
    type: str = "Observation"

    subject_ref: str  # Reference to ALSTrajectory or Patient
    observation_kind: ObservationKind
    name: str
    measurement_method: str = ""
    specimen_or_context: str = ""
    source_ref: str = ""

    # Typed sub-objects (at most one populated per observation)
    lab_result: Optional[LabResult] = None
    emg_finding: Optional[EMGFinding] = None
    respiratory_metric: Optional[RespiratoryMetric] = None
    imaging_finding: Optional[ImagingFinding] = None
    physical_exam_finding: Optional[PhysicalExamFinding] = None

    # Generic value for simple observations
    value: Optional[float] = None
    value_str: Optional[str] = None
    unit: str = ""
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_observation_models.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology/observation.py tests/test_observation_models.py && git commit -m "feat: Observation model with LabResult, EMGFinding, RespiratoryMetric, ImagingFinding"
```

---

## Task 6: Interpretation + State Models

**Files:**
- Create: `scripts/ontology/interpretation.py`
- Create: `scripts/ontology/state.py`
- Test: `tests/test_interpretation_models.py`
- Test: `tests/test_state_models.py`

- [ ] **Step 1: Write the failing test for interpretations**

```python
# tests/test_interpretation_models.py
from ontology.interpretation import Interpretation, EtiologicDriverProfile
from ontology.enums import InterpretationKind, SubtypeClass


def test_interpretation_creation():
    i = Interpretation(
        id="interp:resp_001",
        subject_ref="traj:draper_001",
        interpretation_kind=InterpretationKind.RESPIRATORY_DECLINE_RISK,
        value="low",
        supporting_observation_refs=["obs:fvc_001"],
    )
    assert i.type == "Interpretation"
    assert i.supersedes_ref is None


def test_etiologic_driver_profile():
    edp = EtiologicDriverProfile(
        id="driver:draper_001",
        subject_ref="traj:draper_001",
        posterior={
            SubtypeClass.SOD1: 0.02,
            SubtypeClass.C9ORF72: 0.04,
            SubtypeClass.FUS: 0.01,
            SubtypeClass.TARDBP: 0.03,
            SubtypeClass.SPORADIC_TDP43: 0.73,
            SubtypeClass.GLIA_AMPLIFIED: 0.11,
            SubtypeClass.MIXED: 0.04,
            SubtypeClass.UNRESOLVED: 0.02,
        },
    )
    assert edp.type == "EtiologicDriverProfile"
    assert edp.dominant_subtype == SubtypeClass.SPORADIC_TDP43


def test_etiologic_driver_posterior_sums_to_one():
    edp = EtiologicDriverProfile(
        id="driver:test",
        subject_ref="traj:test",
        posterior={
            SubtypeClass.SOD1: 0.5,
            SubtypeClass.SPORADIC_TDP43: 0.5,
        },
    )
    assert abs(sum(edp.posterior.values()) - 1.0) < 0.01
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_interpretation_models.py -v`
Expected: FAIL

- [ ] **Step 3: Write interpretation implementation**

```python
# scripts/ontology/interpretation.py
"""Interpretation canonical model — model- or human-generated assessments.

Ref: Technical Specification sections 8.2 (Interpretation namespace), 30.5, 30.6
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field, computed_field

from ontology.base import BaseEnvelope
from ontology.enums import InterpretationKind, SubtypeClass


class Interpretation(BaseEnvelope):
    """A model- or human-generated interpretation over observations.

    Spec invariant: Interpretations MUST be versioned and supersedable.
    Observation and interpretation MUST remain separate object classes.
    """
    type: str = "Interpretation"

    subject_ref: str
    interpretation_kind: InterpretationKind
    value: str
    supporting_observation_refs: list[str] = Field(default_factory=list)
    evidence_bundle_ref: Optional[str] = None
    supersedes_ref: Optional[str] = None
    notes: str = ""


class EtiologicDriverProfile(BaseEnvelope):
    """Posterior belief distribution over primary ALS driver programs.

    Ref: Technical Specification section 30.6
    """
    type: str = "EtiologicDriverProfile"

    subject_ref: str
    posterior: dict[SubtypeClass, float] = Field(default_factory=dict)
    supporting_evidence_refs: list[str] = Field(default_factory=list)

    @computed_field
    @property
    def dominant_subtype(self) -> Optional[SubtypeClass]:
        if not self.posterior:
            return None
        return max(self.posterior, key=self.posterior.get)
```

- [ ] **Step 4: Run interpretation tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_interpretation_models.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Write the failing test for state models**

```python
# tests/test_state_models.py
from datetime import datetime, timezone
from ontology.state import (
    DiseaseStateSnapshot, TDP43FunctionalState, SplicingState,
    GlialState, NMJIntegrityState, RespiratoryReserveState,
    FunctionalState, ReversibilityWindowEstimate, UncertaintyState,
)


def test_disease_state_snapshot():
    snap = DiseaseStateSnapshot(
        id="state:draper_2026_03",
        subject_ref="traj:draper_001",
        as_of=datetime(2026, 3, 22, tzinfo=timezone.utc),
    )
    assert snap.type == "DiseaseStateSnapshot"


def test_tdp43_state():
    s = TDP43FunctionalState(
        id="tdp:draper_001",
        subject_ref="traj:draper_001",
        nuclear_function_score=0.31,
        cytoplasmic_pathology_probability=0.77,
        loss_of_function_probability=0.82,
    )
    assert s.type == "TDP43FunctionalState"
    assert s.loss_of_function_probability == 0.82


def test_functional_state():
    fs = FunctionalState(
        id="func:draper_001",
        subject_ref="traj:draper_001",
        alsfrs_r_total=43,
        bulbar_subscore=12,
        fine_motor_subscore=11,
        gross_motor_subscore=8,
        respiratory_subscore=12,
    )
    assert fs.type == "FunctionalState"
    assert fs.alsfrs_r_total == 43


def test_reversibility_window():
    rw = ReversibilityWindowEstimate(
        id="rev:draper_001",
        subject_ref="traj:draper_001",
        overall_reversibility_score=0.49,
        dominant_bottleneck="distal_denervation",
        estimated_time_sensitivity_days=90,
    )
    assert rw.dominant_bottleneck == "distal_denervation"


def test_uncertainty_state():
    us = UncertaintyState(
        id="unc:draper_001",
        subject_ref="traj:draper_001",
        subtype_ambiguity=0.18,
        missing_measurement_uncertainty=0.43,
        dominant_missing_measurements=["genetic_testing", "csf_biomarkers"],
    )
    assert len(us.dominant_missing_measurements) == 2
```

- [ ] **Step 6: Run state model tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_state_models.py -v`
Expected: FAIL with ModuleNotFoundError

- [ ] **Step 7: Write state model implementation**

```python
# scripts/ontology/state.py
"""Disease state models — latent factorization of ALS state at time t.

Ref: Technical Specification sections 9.1-9.7, 30.7-30.15

The normative patient latent state at time t is:
  x_t = [g_t, m_t, n_t, f_t, r_t, u_t]
where g=etiologic, m=molecular, n=neural, f=functional, r=reversibility, u=uncertainty.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from ontology.base import BaseEnvelope


class TDP43FunctionalState(BaseEnvelope):
    """Inferred TDP-43 functional status (part of m_t)."""
    type: str = "TDP43FunctionalState"
    subject_ref: str
    nuclear_function_score: float = 0.0
    cytoplasmic_pathology_probability: float = 0.0
    loss_of_function_probability: float = 0.0
    supporting_marker_refs: list[str] = Field(default_factory=list)
    dominant_uncertainties: list[str] = Field(default_factory=list)


class SplicingState(BaseEnvelope):
    """Cryptic splicing and downstream transcript integrity (part of m_t)."""
    type: str = "SplicingState"
    subject_ref: str
    cryptic_splicing_burden_score: float = 0.0
    stmn2_disruption_score: float = 0.0
    unc13a_disruption_score: float = 0.0
    other_target_scores: dict[str, float] = Field(default_factory=dict)
    source_assay_refs: list[str] = Field(default_factory=list)


class GlialState(BaseEnvelope):
    """Astrocytic, microglial, and inflammatory amplification state (part of m_t)."""
    type: str = "GlialState"
    subject_ref: str
    microglial_activation_score: float = 0.0
    astrocytic_toxicity_score: float = 0.0
    inflammatory_amplification_score: float = 0.0
    evidence_refs: list[str] = Field(default_factory=list)


class NMJIntegrityState(BaseEnvelope):
    """Distal denervation and reinnervation balance (part of n_t)."""
    type: str = "NMJIntegrityState"
    subject_ref: str
    estimated_nmj_occupancy: float = 0.0
    denervation_rate_score: float = 0.0
    reinnervation_capacity_score: float = 0.0
    supporting_refs: list[str] = Field(default_factory=list)


class RespiratoryReserveState(BaseEnvelope):
    """Present respiratory reserve and decline risk (part of n_t)."""
    type: str = "RespiratoryReserveState"
    subject_ref: str
    reserve_score: float = 0.0
    six_month_decline_risk: float = 0.0
    niv_transition_probability_6m: float = 0.0
    supporting_refs: list[str] = Field(default_factory=list)


class FunctionalState(BaseEnvelope):
    """Patient-level function at time t (f_t)."""
    type: str = "FunctionalState"
    subject_ref: str
    alsfrs_r_total: Optional[int] = None
    bulbar_subscore: Optional[int] = None
    fine_motor_subscore: Optional[int] = None
    gross_motor_subscore: Optional[int] = None
    respiratory_subscore: Optional[int] = None
    speech_function_score: Optional[float] = None
    swallow_function_score: Optional[float] = None
    mobility_score: Optional[float] = None
    weight_kg: Optional[float] = None


class ReversibilityWindowEstimate(BaseEnvelope):
    """Inferred salvageability of the current disease state (r_t).

    Spec: Erik's cure logic depends on estimating whether the current
    biological system remains recoverable.
    """
    type: str = "ReversibilityWindowEstimate"
    subject_ref: str
    overall_reversibility_score: float = 0.0
    molecular_correction_plausibility: float = 0.0
    nmj_recovery_plausibility: float = 0.0
    functional_recovery_plausibility: float = 0.0
    dominant_bottleneck: str = ""
    estimated_time_sensitivity_days: Optional[int] = None


class UncertaintyState(BaseEnvelope):
    """Decomposed uncertainty for a state snapshot (u_t).

    Spec: This state MUST be explicit and decomposed, not a single confidence scalar.
    """
    type: str = "UncertaintyState"
    subject_ref: str
    subtype_ambiguity: float = 0.0
    missing_measurement_uncertainty: float = 0.0
    model_form_uncertainty: float = 0.0
    intervention_effect_uncertainty: float = 0.0
    transportability_uncertainty: float = 0.0
    evidence_conflict_uncertainty: float = 0.0
    dominant_missing_measurements: list[str] = Field(default_factory=list)


class DiseaseStateSnapshot(BaseEnvelope):
    """Materialized latent state at a specific timepoint.

    Ref: Technical Specification section 30.7
    Composes all latent state factors: g_t, m_t, n_t, f_t, r_t, u_t.
    """
    type: str = "DiseaseStateSnapshot"
    subject_ref: str
    as_of: datetime

    etiologic_driver_profile_ref: Optional[str] = None
    molecular_state_refs: list[str] = Field(default_factory=list)
    compartment_state_refs: list[str] = Field(default_factory=list)
    functional_state_ref: Optional[str] = None
    reversibility_window_ref: Optional[str] = None
    uncertainty_ref: Optional[str] = None
```

- [ ] **Step 8: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_interpretation_models.py tests/test_state_models.py -v`
Expected: All 8 tests PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/ontology/interpretation.py scripts/ontology/state.py tests/test_interpretation_models.py tests/test_state_models.py && git commit -m "feat: Interpretation, EtiologicDriverProfile, and all latent state factor models"
```

---

## Task 7: Evidence + Protocol + Meta Models

**Files:**
- Create: `scripts/ontology/evidence.py`
- Create: `scripts/ontology/protocol.py`
- Create: `scripts/ontology/intervention.py`
- Create: `scripts/ontology/discovery.py`
- Create: `scripts/ontology/meta.py`
- Test: `tests/test_evidence_models.py`
- Test: `tests/test_protocol_models.py`
- Test: `tests/test_meta_models.py`

- [ ] **Step 1: Write tests for evidence, protocol, and meta models**

```python
# tests/test_evidence_models.py
from ontology.evidence import EvidenceBundle, EvidenceItem
from ontology.enums import EvidenceDirection


def test_evidence_bundle():
    eb = EvidenceBundle(
        id="evb:draper_001",
        subject_ref="traj:draper_001",
        topic="protocol_generation",
        evidence_item_refs=["evi_1", "evi_2"],
        coverage_score=0.77,
        grounding_score=0.82,
    )
    assert eb.type == "EvidenceBundle"


def test_evidence_item():
    ei = EvidenceItem(
        id="evi:riluzole_001",
        claim="Riluzole provides modest survival benefit in ALS",
        direction=EvidenceDirection.SUPPORTS,
        source_refs=["pmid:8302340"],
        strength="moderate",
    )
    assert ei.type == "EvidenceItem"
    assert ei.direction == EvidenceDirection.SUPPORTS
```

```python
# tests/test_protocol_models.py
from ontology.protocol import CureProtocolCandidate, ProtocolLayerEntry, MonitoringPlan
from ontology.enums import ProtocolLayer, ApprovalState


def test_cure_protocol_candidate():
    cpc = CureProtocolCandidate(
        id="proto:draper_001",
        subject_ref="traj:draper_001",
        objective="maximize_durable_disease_arrest_and_functional_recovery",
        layers=[
            ProtocolLayerEntry(
                layer=ProtocolLayer.ROOT_CAUSE_SUPPRESSION,
                intervention_refs=["int:riluzole"],
                start_offset_days=0,
            ),
            ProtocolLayerEntry(
                layer=ProtocolLayer.CIRCUIT_STABILIZATION,
                intervention_refs=["int:niv_monitoring"],
                start_offset_days=0,
            ),
        ],
        approval_state=ApprovalState.PENDING,
    )
    assert cpc.type == "CureProtocolCandidate"
    assert len(cpc.layers) == 2


def test_monitoring_plan():
    mp = MonitoringPlan(
        id="mon:draper_001",
        subject_ref="traj:draper_001",
        scheduled_checks=[
            {"day": 30, "measurements": ["alsfrs_r", "nfl", "fvc"]},
            {"day": 90, "measurements": ["emg", "alsfrs_r", "nfl", "fvc"]},
        ],
    )
    assert mp.type == "MonitoringPlan"
```

```python
# tests/test_meta_models.py
from ontology.meta import LearningEpisode, ErrorRecord, ImprovementProposal, Branch
from ontology.enums import ObjectStatus


def test_learning_episode():
    le = LearningEpisode(
        id="learn:001",
        subject_ref="traj:draper_001",
        trigger="protocol_outcome_deviation",
    )
    assert le.type == "LearningEpisode"


def test_error_record():
    er = ErrorRecord(
        id="err:001",
        category="wrong_causal_direction",
        severity="high",
        description="Planner over-weighted downstream biomarker.",
        affected_components=["world_model", "planner"],
    )
    assert er.type == "ErrorRecord"


def test_branch():
    b = Branch(
        id="branch:003",
        parent_model_ref="model_v0.1",
        branch_purpose="test_nmj_latent_variable_upgrade",
    )
    assert b.type == "Branch"
    assert b.deployment_rights == "none"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_evidence_models.py tests/test_protocol_models.py tests/test_meta_models.py -v`
Expected: FAIL

- [ ] **Step 3: Write all implementations**

```python
# scripts/ontology/evidence.py
"""Evidence canonical models.
Ref: Technical Specification sections 8.2 (Evidence namespace), 30.16
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field
from ontology.base import BaseEnvelope
from ontology.enums import EvidenceDirection


class EvidenceItem(BaseEnvelope):
    type: str = "EvidenceItem"
    claim: str
    direction: EvidenceDirection = EvidenceDirection.INSUFFICIENT
    source_refs: list[str] = Field(default_factory=list)
    strength: str = "unknown"
    notes: str = ""


class EvidenceBundle(BaseEnvelope):
    type: str = "EvidenceBundle"
    subject_ref: str
    topic: str
    evidence_item_refs: list[str] = Field(default_factory=list)
    contradiction_refs: list[str] = Field(default_factory=list)
    coverage_score: float = 0.0
    grounding_score: float = 0.0
```

```python
# scripts/ontology/intervention.py
"""Intervention canonical model.
Ref: Technical Specification section 30.19
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field
from ontology.base import BaseEnvelope
from ontology.enums import InterventionClass, ProtocolLayer


class Intervention(BaseEnvelope):
    type: str = "Intervention"
    name: str
    intervention_class: InterventionClass
    targets: list[str] = Field(default_factory=list)
    protocol_layer: Optional[ProtocolLayer] = None
    route: str = ""
    intended_effects: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
```

```python
# scripts/ontology/protocol.py
"""CureProtocolCandidate and MonitoringPlan.
Ref: Technical Specification sections 13.3, 30.20, 30.21
"""
from __future__ import annotations
from typing import Any, Optional
from pydantic import BaseModel, Field
from ontology.base import BaseEnvelope
from ontology.enums import ApprovalState, ProtocolLayer


class ProtocolLayerEntry(BaseModel):
    layer: ProtocolLayer
    intervention_refs: list[str] = Field(default_factory=list)
    start_offset_days: int = 0
    notes: str = ""


class CureProtocolCandidate(BaseEnvelope):
    """The main recommendation artifact.
    Spec: CureProtocolCandidate MUST be a first-class object.
    """
    type: str = "CureProtocolCandidate"
    subject_ref: str = ""
    objective: str = ""
    eligibility_constraints: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
    assumed_active_programs: list[str] = Field(default_factory=list)
    layers: list[ProtocolLayerEntry] = Field(default_factory=list)
    monitoring_plan_ref: Optional[str] = None
    expected_state_shift_summary: dict[str, float] = Field(default_factory=dict)
    dominant_failure_modes: list[str] = Field(default_factory=list)
    approval_state: ApprovalState = ApprovalState.PENDING
    required_approval_refs: list[str] = Field(default_factory=list)
    evidence_bundle_refs: list[str] = Field(default_factory=list)
    uncertainty_ref: Optional[str] = None


class MonitoringPlan(BaseEnvelope):
    type: str = "MonitoringPlan"
    subject_ref: str = ""
    scheduled_checks: list[dict[str, Any]] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    failure_triggers: list[str] = Field(default_factory=list)
```

```python
# scripts/ontology/discovery.py
"""Discovery object models.
Ref: Technical Specification sections 14.2, 30.17, 30.18
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field
from ontology.base import BaseEnvelope


class MechanismHypothesis(BaseEnvelope):
    type: str = "MechanismHypothesis"
    statement: str
    subject_scope: str = ""
    predicted_observables: list[str] = Field(default_factory=list)
    candidate_tests: list[str] = Field(default_factory=list)
    current_support_direction: str = "insufficient"


class ExperimentProposal(BaseEnvelope):
    type: str = "ExperimentProposal"
    objective: str
    modality: str = ""
    required_inputs: list[str] = Field(default_factory=list)
    expected_information_gain: float = 0.0
    estimated_cost_band: str = "unknown"
    estimated_duration_days: Optional[int] = None
    linked_hypothesis_refs: list[str] = Field(default_factory=list)
```

```python
# scripts/ontology/meta.py
"""Meta-loop models: LearningEpisode, ErrorRecord, ImprovementProposal, Branch.
Ref: Technical Specification sections 30.22-30.26
"""
from __future__ import annotations
from typing import Optional
from pydantic import Field
from ontology.base import BaseEnvelope


class LearningEpisode(BaseEnvelope):
    type: str = "LearningEpisode"
    subject_ref: str = ""
    trigger: str = ""
    state_snapshot_ref: Optional[str] = None
    protocol_ref: Optional[str] = None
    expected_outcome_ref: Optional[str] = None
    actual_outcome_ref: Optional[str] = None
    error_record_refs: list[str] = Field(default_factory=list)
    replay_trace_ref: Optional[str] = None


class ErrorRecord(BaseEnvelope):
    type: str = "ErrorRecord"
    category: str
    severity: str = "medium"
    description: str = ""
    affected_components: list[str] = Field(default_factory=list)
    candidate_root_causes: list[str] = Field(default_factory=list)


class ImprovementProposal(BaseEnvelope):
    type: str = "ImprovementProposal"
    proposal_kind: str = ""
    target_component: str = ""
    description: str = ""
    justification_refs: list[str] = Field(default_factory=list)
    evaluation_plan_ref: Optional[str] = None
    branch_ref: Optional[str] = None


class Branch(BaseEnvelope):
    type: str = "Branch"
    parent_model_ref: str = ""
    branch_purpose: str = ""
    created_from_snapshot_ref: Optional[str] = None
    deployment_rights: str = "none"
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_evidence_models.py tests/test_protocol_models.py tests/test_meta_models.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology/evidence.py scripts/ontology/intervention.py scripts/ontology/protocol.py scripts/ontology/discovery.py scripts/ontology/meta.py tests/test_evidence_models.py tests/test_protocol_models.py tests/test_meta_models.py && git commit -m "feat: Evidence, Intervention, Protocol, Discovery, and Meta-loop models"
```

---

## Task 8: Relation Vocabulary + Type Registry

**Files:**
- Create: `scripts/ontology/relations.py`
- Create: `scripts/ontology/registry.py`
- Test: `tests/test_relations.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_relations.py
from ontology.relations import (
    RELATION_TYPES, OBSERVATIONAL_RELATION_TYPES,
    is_observational, get_relation_category,
)


def test_core_relations_exist():
    for rel in ["causes", "contributes_to", "targets", "treats", "supports", "refutes"]:
        assert rel in RELATION_TYPES


def test_observational_types_never_upgrade():
    for rel in OBSERVATIONAL_RELATION_TYPES:
        assert is_observational(rel) is True


def test_causal_relations_not_observational():
    assert is_observational("causes") is False
    assert is_observational("contributes_to") is False


def test_relation_category():
    assert get_relation_category("causes") == "causal"
    assert get_relation_category("part_of") == "structural"
    assert get_relation_category("supports") == "evidential"
```

```python
# tests/test_registry.py
from ontology.registry import get_model_class, list_types


def test_get_model_class_patient():
    cls = get_model_class("Patient")
    assert cls is not None
    assert cls.__name__ == "Patient"


def test_get_model_class_observation():
    cls = get_model_class("Observation")
    assert cls.__name__ == "Observation"


def test_list_types_has_all_core():
    types = list_types()
    for t in ["Patient", "ALSTrajectory", "Observation", "Interpretation",
              "DiseaseStateSnapshot", "CureProtocolCandidate", "EvidenceBundle",
              "LearningEpisode", "ErrorRecord", "Branch"]:
        assert t in types, f"Missing type: {t}"


def test_unknown_type_returns_none():
    assert get_model_class("NonexistentType") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_relations.py tests/test_registry.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementations**

```python
# scripts/ontology/relations.py
"""Typed relation vocabulary for the Erik knowledge graph.

Ref: Technical Specification section 8.4
Spec: The graph and APIs MUST use typed relations.

CRITICAL: OBSERVATIONAL_RELATION_TYPES must NEVER be upgraded to L3 (counterfactual).
This guard must be checked in ALL PCH upgrade paths (same pattern as Galen).
"""

RELATION_TYPES: dict[str, dict] = {
    # Structural
    "has_part": {"category": "structural"},
    "part_of": {"category": "structural"},
    "subtype_of": {"category": "structural"},
    "instance_of": {"category": "structural"},
    "located_in": {"category": "structural"},
    "member_of": {"category": "structural"},
    "expressed_in": {"category": "structural"},
    # Causal
    "causes": {"category": "causal"},
    "contributes_to": {"category": "causal"},
    "amplifies": {"category": "causal"},
    "suppresses": {"category": "causal"},
    "confounds": {"category": "causal"},
    "modifies_risk_of": {"category": "causal"},
    # Observational
    "observed_in": {"category": "observational"},
    "measures": {"category": "observational"},
    "derived_from": {"category": "observational"},
    "associated_with": {"category": "observational"},
    "variant_in_gene": {"category": "observational"},
    # Temporal
    "precedes": {"category": "temporal"},
    "follows": {"category": "temporal"},
    # Evidential
    "asserts": {"category": "evidential"},
    "inferred_from": {"category": "evidential"},
    "supports": {"category": "evidential"},
    "refutes": {"category": "evidential"},
    "learned_from": {"category": "evidential"},
    "counterfactual_of": {"category": "evidential"},
    "supersedes": {"category": "evidential"},
    # Therapeutic
    "targets": {"category": "therapeutic"},
    "treats": {"category": "therapeutic"},
    "contraindicates": {"category": "therapeutic"},
    "eligible_for": {"category": "therapeutic"},
    "ineligible_for": {"category": "therapeutic"},
    "resulted_in": {"category": "therapeutic"},
    # Governance
    "constrained_by": {"category": "governance"},
    "optimizes_for": {"category": "governance"},
    "requires_approval_from": {"category": "governance"},
    "executed_by": {"category": "governance"},
    "evaluated_by": {"category": "governance"},
}

# These relation types represent structural/observational facts that must
# NEVER be promoted to L3 (counterfactual) in the Pearl Causal Hierarchy.
OBSERVATIONAL_RELATION_TYPES: frozenset[str] = frozenset({
    "variant_in_gene",
    "located_in",
    "member_of",
    "subtype_of",
    "instance_of",
    "observed_in",
    "has_part",
    "part_of",
    "expressed_in",
    "associated_with",
    "derived_from",
    "measures",
})


def is_observational(relation_type: str) -> bool:
    return relation_type in OBSERVATIONAL_RELATION_TYPES


def get_relation_category(relation_type: str) -> str:
    info = RELATION_TYPES.get(relation_type)
    if info is None:
        return "unknown"
    return info["category"]
```

```python
# scripts/ontology/registry.py
"""Type registry: maps canonical type names to Pydantic model classes.

Used for deserialization, validation, and schema introspection.
"""
from __future__ import annotations

from typing import Optional, Type

from ontology.base import BaseEnvelope
from ontology.patient import Patient, ALSTrajectory
from ontology.observation import Observation
from ontology.interpretation import Interpretation, EtiologicDriverProfile
from ontology.state import (
    DiseaseStateSnapshot, TDP43FunctionalState, SplicingState,
    GlialState, NMJIntegrityState, RespiratoryReserveState,
    FunctionalState, ReversibilityWindowEstimate, UncertaintyState,
)
from ontology.evidence import EvidenceBundle, EvidenceItem
from ontology.intervention import Intervention
from ontology.protocol import CureProtocolCandidate, MonitoringPlan
from ontology.discovery import MechanismHypothesis, ExperimentProposal
from ontology.meta import LearningEpisode, ErrorRecord, ImprovementProposal, Branch

_REGISTRY: dict[str, Type[BaseEnvelope]] = {
    "Patient": Patient,
    "ALSTrajectory": ALSTrajectory,
    "Observation": Observation,
    "Interpretation": Interpretation,
    "EtiologicDriverProfile": EtiologicDriverProfile,
    "DiseaseStateSnapshot": DiseaseStateSnapshot,
    "TDP43FunctionalState": TDP43FunctionalState,
    "SplicingState": SplicingState,
    "GlialState": GlialState,
    "NMJIntegrityState": NMJIntegrityState,
    "RespiratoryReserveState": RespiratoryReserveState,
    "FunctionalState": FunctionalState,
    "ReversibilityWindowEstimate": ReversibilityWindowEstimate,
    "UncertaintyState": UncertaintyState,
    "EvidenceBundle": EvidenceBundle,
    "EvidenceItem": EvidenceItem,
    "Intervention": Intervention,
    "CureProtocolCandidate": CureProtocolCandidate,
    "MonitoringPlan": MonitoringPlan,
    "MechanismHypothesis": MechanismHypothesis,
    "ExperimentProposal": ExperimentProposal,
    "LearningEpisode": LearningEpisode,
    "ErrorRecord": ErrorRecord,
    "ImprovementProposal": ImprovementProposal,
    "Branch": Branch,
}


def get_model_class(type_name: str) -> Optional[Type[BaseEnvelope]]:
    return _REGISTRY.get(type_name)


def list_types() -> list[str]:
    return list(_REGISTRY.keys())
```

- [ ] **Step 4: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_relations.py tests/test_registry.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/ontology/relations.py scripts/ontology/registry.py tests/test_relations.py tests/test_registry.py && git commit -m "feat: relation vocabulary with observational guards + type registry"
```

---

## Task 9: PostgreSQL Schema

**Files:**
- Create: `scripts/db/core_schema.sql`
- Create: `scripts/db/ops_schema.sql`
- Create: `scripts/db/pool.py`
- Create: `scripts/db/migrate.py`
- Test: `tests/conftest.py`
- Test: `tests/test_db_schema.py`
- Test: `tests/test_db_pool.py`

- [ ] **Step 1: Write core_schema.sql**

```sql
-- scripts/db/core_schema.sql
-- Erik ALS Engine — Core Schema (erik_core)
-- Canonical objects, knowledge graph, patient trajectories

CREATE SCHEMA IF NOT EXISTS erik_core;
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;

-- Canonical object store (all types in one table, JSONB body)
CREATE TABLE IF NOT EXISTS erik_core.objects (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT '1.0',
    tenant_id TEXT NOT NULL DEFAULT 'erik_default',
    status TEXT NOT NULL DEFAULT 'active',
    time_observed_at TIMESTAMPTZ,
    time_effective_at TIMESTAMPTZ,
    time_recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    time_valid_from TIMESTAMPTZ,
    time_valid_to TIMESTAMPTZ,
    provenance_source_system TEXT,
    provenance_artifact_id TEXT,
    provenance_asserted_by TEXT,
    provenance_trace_id TEXT,
    confidence REAL,
    privacy_class TEXT NOT NULL DEFAULT 'restricted',
    body JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_objects_type ON erik_core.objects(type);
CREATE INDEX IF NOT EXISTS idx_objects_status ON erik_core.objects(status);
CREATE INDEX IF NOT EXISTS idx_objects_body ON erik_core.objects USING gin(body);

-- Knowledge graph entities
CREATE TABLE IF NOT EXISTS erik_core.entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name CITEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence REAL DEFAULT 0.5,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    pch_layer INTEGER NOT NULL DEFAULT 1,
    evidence_type TEXT,
    provenance TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON erik_core.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON erik_core.entities(name);

-- Knowledge graph relationships
CREATE TABLE IF NOT EXISTS erik_core.relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES erik_core.entities(id),
    target_id TEXT NOT NULL REFERENCES erik_core.entities(id),
    relationship_type TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence REAL DEFAULT 0.5,
    evidence TEXT,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    pch_layer INTEGER NOT NULL DEFAULT 1,
    evidence_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rel_source ON erik_core.relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON erik_core.relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON erik_core.relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_pch ON erik_core.relationships(pch_layer);

-- Embedding store for semantic retrieval
CREATE TABLE IF NOT EXISTS erik_core.embeddings (
    id TEXT PRIMARY KEY,
    object_ref TEXT NOT NULL,
    object_type TEXT NOT NULL,
    embedding vector(384),
    text_content TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 2: Write ops_schema.sql**

```sql
-- scripts/db/ops_schema.sql
-- Erik ALS Engine — Operational Schema (erik_ops)
-- Audit log, config, operational state

CREATE SCHEMA IF NOT EXISTS erik_ops;

-- Immutable audit event log
CREATE TABLE IF NOT EXISTS erik_ops.audit_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    object_id TEXT,
    object_type TEXT,
    actor TEXT NOT NULL DEFAULT 'erik_system',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    trace_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_type ON erik_ops.audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_object ON erik_ops.audit_events(object_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON erik_ops.audit_events(created_at);

-- Config snapshots (for audit trail of config changes)
CREATE TABLE IF NOT EXISTS erik_ops.config_snapshots (
    id BIGSERIAL PRIMARY KEY,
    config JSONB NOT NULL,
    changed_keys TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

- [ ] **Step 3: Write pool.py**

```python
# scripts/db/pool.py
"""PostgreSQL connection pool for Erik.

Usage:
    from db.pool import get_connection
    with get_connection() as conn:
        conn.execute("SELECT 1")
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

import psycopg
from psycopg_pool import ConnectionPool

_DB_NAME = os.environ.get("ERIK_DB_NAME", "erik_kg")
_DB_HOST = os.environ.get("ERIK_DB_HOST", "")  # empty = Unix socket
_DB_USER = os.environ.get("ERIK_DB_USER", os.environ.get("USER", "logannye"))

_conninfo = f"dbname={_DB_NAME}"
if _DB_HOST:
    _conninfo += f" host={_DB_HOST}"
if _DB_USER:
    _conninfo += f" user={_DB_USER}"

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_conninfo,
            min_size=1,
            max_size=5,
            open=True,
        )
    return _pool


@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    pool = get_pool()
    with pool.connection() as conn:
        yield conn


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
```

- [ ] **Step 4: Write migrate.py**

```python
# scripts/db/migrate.py
"""Run schema migrations for Erik database."""
from __future__ import annotations

import os
from pathlib import Path

from db.pool import get_connection

_SCHEMA_DIR = Path(__file__).parent


def run_migrations() -> None:
    for sql_file in ["core_schema.sql", "ops_schema.sql"]:
        path = _SCHEMA_DIR / sql_file
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        sql = path.read_text()
        with get_connection() as conn:
            conn.execute(sql)
            conn.commit()
        print(f"Applied: {sql_file}")


if __name__ == "__main__":
    run_migrations()
    print("All migrations applied successfully.")
```

- [ ] **Step 5: Write test fixtures and DB tests**

```python
# tests/conftest.py
"""Shared test fixtures."""
import os
import pytest

# Use a test database if available, otherwise skip DB tests
TEST_DB = os.environ.get("ERIK_TEST_DB", "erik_kg")


@pytest.fixture
def db_available():
    """Skip test if database is not available."""
    try:
        import psycopg
        conn = psycopg.connect(f"dbname={TEST_DB}")
        conn.close()
        return True
    except Exception:
        pytest.skip("PostgreSQL database not available")
```

```python
# tests/test_db_schema.py
import pytest


def test_core_schema_sql_exists():
    from pathlib import Path
    path = Path("/Users/logannye/.openclaw/erik/scripts/db/core_schema.sql")
    assert path.exists()
    content = path.read_text()
    assert "erik_core.objects" in content
    assert "erik_core.entities" in content
    assert "erik_core.relationships" in content


def test_ops_schema_sql_exists():
    from pathlib import Path
    path = Path("/Users/logannye/.openclaw/erik/scripts/db/ops_schema.sql")
    assert path.exists()
    content = path.read_text()
    assert "erik_ops.audit_events" in content


def test_migrate_runs(db_available):
    from db.migrate import run_migrations
    run_migrations()  # Should be idempotent
```

```python
# tests/test_db_pool.py
import pytest


def test_pool_connection(db_available):
    from scripts.db.pool import get_connection, close_pool
    with get_connection() as conn:
        result = conn.execute("SELECT 1 AS val").fetchone()
        assert result[0] == 1
    close_pool()
```

- [ ] **Step 6: Create PostgreSQL database**

```bash
createdb erik_kg 2>/dev/null || echo "Database erik_kg already exists"
```

- [ ] **Step 7: Run schema migration**

```bash
cd /Users/logannye/.openclaw/erik && PYTHONPATH=scripts conda run -n erik-core python -m db.migrate
```

- [ ] **Step 8: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_db_schema.py tests/test_db_pool.py -v`
Expected: All 4 tests PASS (or skip if DB not available)

- [ ] **Step 9: Commit**

```bash
git add scripts/db/ tests/conftest.py tests/test_db_schema.py tests/test_db_pool.py && git commit -m "feat: PostgreSQL schema (erik_core + erik_ops) with connection pool and migrations"
```

---

## Task 10: Audit Event Log

**Files:**
- Create: `scripts/audit/event_log.py`
- Test: `tests/test_audit_log.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit_log.py
import pytest
from audit.event_log import AuditLogger, AuditEvent


def test_audit_event_creation():
    event = AuditEvent(
        event_type="ObjectCreated",
        object_id="patient:erik_draper",
        object_type="Patient",
        actor="erik_system",
        details={"source": "clinical_ingestion"},
    )
    assert event.event_type == "ObjectCreated"


def test_audit_logger_log(db_available):
    logger = AuditLogger()
    logger.log(
        event_type="ObjectCreated",
        object_id="test_001",
        object_type="TestType",
    )
    # Verify it was written
    events = logger.query(object_id="test_001")
    assert len(events) >= 1
    assert events[0].event_type == "ObjectCreated"
    # Cleanup
    logger.delete_test_events("test_001")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_audit_log.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/audit/event_log.py
"""Append-only audit event logger.

Ref: Technical Specification section 18.6
Every proposal, simulation, promotion, rollback, protocol version,
and override decision MUST be inspectable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from db.pool import get_connection


@dataclass
class AuditEvent:
    event_type: str
    object_id: Optional[str] = None
    object_type: Optional[str] = None
    actor: str = "erik_system"
    details: dict[str, Any] = field(default_factory=dict)
    trace_id: Optional[str] = None
    created_at: Optional[datetime] = None


class AuditLogger:
    """Immutable audit trail writer and reader."""

    def log(
        self,
        event_type: str,
        object_id: Optional[str] = None,
        object_type: Optional[str] = None,
        actor: str = "erik_system",
        details: Optional[dict] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        import json
        with get_connection() as conn:
            conn.execute(
                """INSERT INTO erik_ops.audit_events
                   (event_type, object_id, object_type, actor, details, trace_id)
                   VALUES (%s, %s, %s, %s, %s::jsonb, %s)""",
                (event_type, object_id, object_type, actor,
                 json.dumps(details or {}), trace_id),
            )
            conn.commit()

    def query(
        self,
        object_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        conditions = []
        params: list = []
        if object_id:
            conditions.append("object_id = %s")
            params.append(object_id)
        if event_type:
            conditions.append("event_type = %s")
            params.append(event_type)
        where = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        with get_connection() as conn:
            rows = conn.execute(
                f"""SELECT event_type, object_id, object_type, actor, details,
                           trace_id, created_at
                    FROM erik_ops.audit_events
                    WHERE {where}
                    ORDER BY created_at DESC
                    LIMIT %s""",
                params,
            ).fetchall()

        return [
            AuditEvent(
                event_type=r[0], object_id=r[1], object_type=r[2],
                actor=r[3], details=r[4] or {}, trace_id=r[5], created_at=r[6],
            )
            for r in rows
        ]

    def delete_test_events(self, object_id: str) -> None:
        """For test cleanup only."""
        with get_connection() as conn:
            conn.execute(
                "DELETE FROM erik_ops.audit_events WHERE object_id = %s",
                (object_id,),
            )
            conn.commit()
```

- [ ] **Step 4: Run test**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_audit_log.py -v`
Expected: PASS (or skip if DB not available)

- [ ] **Step 5: Commit**

```bash
git add scripts/audit/event_log.py tests/test_audit_log.py && git commit -m "feat: append-only audit event logger with query support"
```

---

## Task 11: Config Loader

**Files:**
- Create: `scripts/config/loader.py`
- Create: `data/erik_config.json`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_loader.py
import json
import tempfile
from pathlib import Path
from config.loader import ConfigLoader


def test_config_loader_reads_file():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"temperature": 1.0, "exploration_epsilon": 0.3}, f)
        f.flush()
        loader = ConfigLoader(f.name)
        assert loader.get("temperature") == 1.0
        assert loader.get("exploration_epsilon") == 0.3
    Path(f.name).unlink()


def test_config_loader_hot_reload():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "original"}, f)
        f.flush()
        loader = ConfigLoader(f.name)
        assert loader.get("key") == "original"
    # Modify file
    with open(f.name, "w") as f2:
        json.dump({"key": "updated"}, f2)
    loader.reload()
    assert loader.get("key") == "updated"
    Path(f.name).unlink()


def test_config_loader_default():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({}, f)
        f.flush()
        loader = ConfigLoader(f.name)
        assert loader.get("nonexistent", default=42) == 42
    Path(f.name).unlink()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_config_loader.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# scripts/config/loader.py
"""Hot-reloadable JSON config loader (Galen pattern).

Config is loaded once at startup, then reloaded periodically (every N steps)
without requiring a process restart.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class ConfigLoader:
    def __init__(self, path: str | None = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "data", "erik_config.json",
            )
        self._path = Path(path)
        self._data: dict[str, Any] = {}
        self._mtime: float = 0.0
        self.reload()

    def reload(self) -> bool:
        if not self._path.exists():
            return False
        mtime = self._path.stat().st_mtime
        if mtime == self._mtime:
            return False
        with open(self._path) as f:
            self._data = json.load(f)
        self._mtime = mtime
        return True

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def get_all(self) -> dict[str, Any]:
        return dict(self._data)

    def reload_if_changed(self) -> bool:
        if not self._path.exists():
            return False
        mtime = self._path.stat().st_mtime
        if mtime != self._mtime:
            return self.reload()
        return False
```

- [ ] **Step 4: Create initial config file**

```json
{
  "version": "0.1.0",
  "database_name": "erik_kg",
  "llm_server_enabled": false,
  "temperature": 1.0,
  "exploration_epsilon": 0.30,
  "action_timeout_s": 120,
  "hot_reload_interval_steps": 10,
  "audit_enabled": true
}
```

Write to: `data/erik_config.json`

- [ ] **Step 5: Run tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_config_loader.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add scripts/config/loader.py data/erik_config.json tests/test_config_loader.py && git commit -m "feat: hot-reloadable JSON config loader"
```

---

## Task 12: Ingest Erik Draper's Clinical Data

**Files:**
- Create: `scripts/ingestion/lab_results.py`
- Create: `scripts/ingestion/patient_builder.py`
- Create: `data/erik_patient_data.json`
- Test: `tests/test_ingestion_labs.py`
- Test: `tests/test_ingestion_patient.py`
- Test: `tests/test_erik_trajectory.py`

This is the culmination task — it takes everything built in Tasks 1-11 and produces the first real patient trajectory from Erik Draper's clinical records.

- [ ] **Step 1: Write the failing test for lab parsing**

```python
# tests/test_ingestion_labs.py
from ingestion.lab_results import parse_lab_panel


def test_parse_lab_panel():
    raw = [
        {"name": "Neurofilament Light", "value": 5.82, "unit": "pg/mL",
         "ref_low": 0.0, "ref_high": 3.65, "date": "2026-02-20"},
        {"name": "Creatine Kinase", "value": 200.0, "unit": "U/L",
         "ref_low": 51.0, "ref_high": 298.0, "date": "2026-02-20"},
    ]
    results = parse_lab_panel(raw, subject_ref="traj:draper_001")
    assert len(results) == 2
    assert results[0].lab_result.is_abnormal is True   # NfL elevated
    assert results[1].lab_result.is_abnormal is False   # CK normal
    assert results[0].observation_kind.value == "lab_result"
```

- [ ] **Step 2: Write the failing test for patient builder**

```python
# tests/test_ingestion_patient.py
from ingestion.patient_builder import build_erik_draper


def test_build_erik_draper():
    patient, trajectory, observations = build_erik_draper()

    # Patient checks
    assert patient.id == "patient:erik_draper"
    assert patient.birth_year == 1958
    assert patient.sex_at_birth == "male"
    assert patient.family_history_of_als is False

    # Trajectory checks
    assert trajectory.id == "traj:draper_001"
    assert trajectory.onset_region.value == "lower_limb"
    assert trajectory.diagnosis_date.year == 2026
    assert len(trajectory.alsfrs_r_scores) >= 1
    assert trajectory.alsfrs_r_scores[0].total == 43

    # Observations: should have labs, EMG, MRI, spirometry, exam findings
    obs_kinds = {o.observation_kind.value for o in observations}
    assert "lab_result" in obs_kinds
    assert "emg_feature" in obs_kinds
    assert "imaging_finding" in obs_kinds
    assert "respiratory_metric" in obs_kinds

    # NfL should be flagged abnormal
    nfl_obs = [o for o in observations if "neurofilament" in o.name.lower()]
    assert len(nfl_obs) == 1
    assert nfl_obs[0].lab_result.is_abnormal is True
```

- [ ] **Step 3: Write test for Erik's complete trajectory**

```python
# tests/test_erik_trajectory.py
from ingestion.patient_builder import build_erik_draper


def test_erik_observation_count():
    """Erik should have 30+ structured observations from his clinical records."""
    _, _, observations = build_erik_draper()
    assert len(observations) >= 30


def test_erik_alsfrs_r_decline():
    """Erik's ALSFRS-R is 43/48 with -0.39 points/month decline rate."""
    _, trajectory, _ = build_erik_draper()
    score = trajectory.alsfrs_r_scores[0]
    rate = score.decline_rate_from_onset(trajectory.onset_date)
    assert -0.50 < rate < -0.30


def test_erik_respiratory_preserved():
    """Erik's FVC is 100% predicted — respiratory function preserved."""
    _, _, observations = build_erik_draper()
    resp = [o for o in observations if o.observation_kind.value == "respiratory_metric"]
    assert len(resp) >= 1
    assert resp[0].respiratory_metric.fvc_percent_predicted == 100.0


def test_erik_emg_supports_als():
    """EMG findings are supportive of ALS."""
    _, _, observations = build_erik_draper()
    emg = [o for o in observations if o.observation_kind.value == "emg_feature"]
    assert len(emg) >= 1
    assert emg[0].emg_finding.supports_als is True


def test_erik_mri_brain_no_motor_pathway_changes():
    """Brain MRI shows no motor pathway signal changes typical for ALS."""
    _, _, observations = build_erik_draper()
    brain = [o for o in observations
             if o.observation_kind.value == "imaging_finding"
             and "brain" in o.name.lower()]
    assert len(brain) >= 1
    assert brain[0].imaging_finding.als_relevant is False
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_ingestion_labs.py tests/test_ingestion_patient.py tests/test_erik_trajectory.py -v`
Expected: FAIL

- [ ] **Step 5: Write lab_results.py**

```python
# scripts/ingestion/lab_results.py
"""Parse lab result data into Observation objects."""
from __future__ import annotations

from datetime import date
from typing import Any

from ontology.observation import LabResult, Observation
from ontology.enums import ObservationKind, SourceSystem
from ontology.base import Provenance


def parse_lab_panel(
    raw_labs: list[dict[str, Any]],
    subject_ref: str,
) -> list[Observation]:
    observations = []
    for lab in raw_labs:
        lab_result = LabResult(
            name=lab["name"],
            value=float(lab["value"]),
            unit=lab["unit"],
            reference_low=lab.get("ref_low"),
            reference_high=lab.get("ref_high"),
            collection_date=date.fromisoformat(lab["date"]),
        )
        obs = Observation(
            id=f"obs:lab:{lab['name'].lower().replace(' ', '_')}:{lab['date']}",
            subject_ref=subject_ref,
            observation_kind=ObservationKind.LAB_RESULT,
            name=lab["name"],
            lab_result=lab_result,
            provenance=Provenance(
                source_system=SourceSystem.EHR,
                asserted_by="cleveland_clinic",
            ),
        )
        observations.append(obs)
    return observations
```

- [ ] **Step 6: Write patient_builder.py**

This is a long file containing all of Erik Draper's structured clinical data. The implementation must faithfully encode every finding from the medical records.

```python
# scripts/ingestion/patient_builder.py
"""Build Erik Draper's Patient, ALSTrajectory, and Observations from clinical records.

Source: Cleveland Clinic records (Feb-Mar 2026), PCP records (Jun 2025).
All data extracted from PDFs in /Users/logannye/Desktop/Erik/.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

from ontology.patient import Patient, ALSTrajectory, ALSFRSRScore
from ontology.observation import (
    Observation, LabResult, EMGFinding, RespiratoryMetric,
    ImagingFinding, PhysicalExamFinding,
)
from ontology.enums import (
    ALSOnsetRegion, ObservationKind, PrivacyClass, SourceSystem,
)
from ontology.base import Provenance, Privacy
from ingestion.lab_results import parse_lab_panel


def _cc_provenance() -> Provenance:
    return Provenance(source_system=SourceSystem.EHR, asserted_by="cleveland_clinic_neuromuscular")


def _pcp_provenance() -> Provenance:
    return Provenance(source_system=SourceSystem.EHR, asserted_by="majetich_family_medicine")


def build_erik_draper() -> tuple[Patient, ALSTrajectory, list[Observation]]:
    patient = Patient(
        id="patient:erik_draper",
        patient_key="draper_erik_001",
        birth_year=1958,
        sex_at_birth="male",
        family_history_of_als=False,
        family_history_notes="Mother with Alzheimer's disease; maternal family AD. No family history of MND.",
        allergies=[],
        medications=[
            "amlodipine-atorvastatin 10-20mg daily",
            "ramipril 10mg daily",
            "riluzole (started 2026-03-06)",
            "vitamin C 500mg daily",
            "calcium carbonate-vitamin D3 daily",
            "glucosamine 2000mg daily",
            "lysine 500mg daily",
            "magnesium oxide 400mg daily",
            "multivitamin daily",
            "potassium gluconate 600mg daily",
        ],
        comorbidities=[
            "benign essential hypertension",
            "prediabetes (A1c 5.7%)",
            "elevated cholesterol (on statin)",
            "cervical spinal stenosis (C3-C7, no myelopathy)",
            "basal cell carcinoma history",
        ],
        consent_profiles=["research_participation", "genetic_testing"],
    )

    trajectory = ALSTrajectory(
        id="traj:draper_001",
        patient_ref="patient:erik_draper",
        onset_date=date(2025, 1, 15),
        diagnosis_date=date(2026, 3, 6),
        onset_region=ALSOnsetRegion.LOWER_LIMB,
        episode_status="active",
        site_of_care_refs=["site:cleveland_clinic_neuromuscular"],
        alsfrs_r_scores=[
            ALSFRSRScore(
                speech=4, salivation=4, swallowing=4,
                handwriting=4, cutting_food=4, dressing_hygiene=3,
                turning_in_bed=3, walking=3, climbing_stairs=2,
                dyspnea=4, orthopnea=4, respiratory_insufficiency=4,
                assessment_date=date(2026, 2, 6),
            ),
        ],
        provenance=_cc_provenance(),
    )

    observations: list[Observation] = []

    # --- Feb 20, 2026: ALS workup labs (ordered by Dr. Thakore) ---
    als_labs = [
        {"name": "Neurofilament Light, Plasma", "value": 5.82, "unit": "pg/mL", "ref_low": 0.0, "ref_high": 3.65, "date": "2026-02-20"},
        {"name": "Creatine Kinase", "value": 200.0, "unit": "U/L", "ref_low": 51.0, "ref_high": 298.0, "date": "2026-02-20"},
        {"name": "Vitamin B12", "value": 603.0, "unit": "pg/mL", "ref_low": 232.0, "ref_high": 1245.0, "date": "2026-02-20"},
        {"name": "Folate", "value": 11.2, "unit": "ng/mL", "ref_low": 4.7, "ref_high": None, "date": "2026-02-20"},
        {"name": "Copper, Blood", "value": 95.0, "unit": "ug/dL", "ref_low": 70.0, "ref_high": 140.0, "date": "2026-02-20"},
        {"name": "Sed Rate, Westergren", "value": 2.0, "unit": "mm/hr", "ref_low": 0.0, "ref_high": 15.0, "date": "2026-02-20"},
        {"name": "WBC", "value": 8.92, "unit": "k/uL", "ref_low": 3.7, "ref_high": 11.0, "date": "2026-02-20"},
        {"name": "Hemoglobin", "value": 15.2, "unit": "g/dL", "ref_low": 13.0, "ref_high": 17.0, "date": "2026-02-20"},
        {"name": "Hematocrit", "value": 46.5, "unit": "%", "ref_low": 39.0, "ref_high": 51.0, "date": "2026-02-20"},
        {"name": "Platelet Count", "value": 366.0, "unit": "k/uL", "ref_low": 150.0, "ref_high": 400.0, "date": "2026-02-20"},
        {"name": "Glucose", "value": 128.0, "unit": "mg/dL", "ref_low": 74.0, "ref_high": 99.0, "date": "2026-02-20"},
        {"name": "BUN", "value": 21.0, "unit": "mg/dL", "ref_low": 9.0, "ref_high": 24.0, "date": "2026-02-20"},
        {"name": "Creatinine", "value": 1.06, "unit": "mg/dL", "ref_low": 0.73, "ref_high": 1.22, "date": "2026-02-20"},
        {"name": "eGFR", "value": 77.0, "unit": "mL/min/1.73m2", "ref_low": 60.0, "ref_high": None, "date": "2026-02-20"},
        {"name": "Sodium", "value": 139.0, "unit": "mmol/L", "ref_low": 136.0, "ref_high": 144.0, "date": "2026-02-20"},
        {"name": "Potassium", "value": 4.3, "unit": "mmol/L", "ref_low": 3.7, "ref_high": 5.1, "date": "2026-02-20"},
        {"name": "AST", "value": 19.0, "unit": "U/L", "ref_low": 14.0, "ref_high": 40.0, "date": "2026-02-20"},
        {"name": "ALT", "value": 28.0, "unit": "U/L", "ref_low": 10.0, "ref_high": 54.0, "date": "2026-02-20"},
        {"name": "Albumin", "value": 4.5, "unit": "g/dL", "ref_low": 3.9, "ref_high": 4.9, "date": "2026-02-20"},
        {"name": "Calcium, Total", "value": 9.2, "unit": "mg/dL", "ref_low": 8.5, "ref_high": 10.2, "date": "2026-02-20"},
    ]
    observations.extend(parse_lab_panel(als_labs, subject_ref="traj:draper_001"))

    # --- Jun 9, 2025: Baseline labs (PCP wellness visit) ---
    baseline_labs = [
        {"name": "Hemoglobin A1c", "value": 5.7, "unit": "%", "ref_low": 4.3, "ref_high": 5.6, "date": "2025-06-09"},
        {"name": "TSH", "value": 1.38, "unit": "mIU/L", "ref_low": 0.27, "ref_high": 4.2, "date": "2025-06-09"},
        {"name": "Cholesterol, Total", "value": 143.0, "unit": "mg/dL", "ref_low": None, "ref_high": 200.0, "date": "2025-06-09"},
        {"name": "HDL Cholesterol", "value": 36.0, "unit": "mg/dL", "ref_low": 39.0, "ref_high": None, "date": "2025-06-09"},
        {"name": "LDL Cholesterol", "value": 93.0, "unit": "mg/dL", "ref_low": None, "ref_high": 100.0, "date": "2025-06-09"},
        {"name": "Triglyceride", "value": 72.0, "unit": "mg/dL", "ref_low": None, "ref_high": 150.0, "date": "2025-06-09"},
        {"name": "PSA", "value": 1.76, "unit": "ng/mL", "ref_low": None, "ref_high": 2.6, "date": "2025-06-09"},
    ]
    observations.extend(parse_lab_panel(baseline_labs, subject_ref="traj:draper_001"))

    # --- EMG: March 6, 2026 ---
    observations.append(Observation(
        id="obs:emg:2026_03_06",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.EMG_FEATURE,
        name="EMG ALS Workup",
        emg_finding=EMGFinding(
            study_date=date(2026, 3, 6),
            summary="Widespread active and chronic motor axon loss changes supportive of ALS",
            performing_physician="Georgette Dib, MD",
            regions_with_active_denervation=["left_leg"],
            regions_with_chronic_denervation=["left_leg", "bilateral_proximal_leg"],
            supports_als=True,
        ),
        provenance=_cc_provenance(),
    ))

    # --- EMG: June 25, 2025 (outside, Precision Orthopaedic) ---
    observations.append(Observation(
        id="obs:emg:2025_06_25",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.EMG_FEATURE,
        name="EMG Outside Facility",
        emg_finding=EMGFinding(
            study_date=date(2025, 6, 25),
            summary="Positive sharp waves in L vastus lateralis and tibialis anterior, reduced recruitment with polyphasic MUPs. Low CMAPs L peroneal (0.6mV) and posterior tibial (1.6mV). Interpreted as L4-L5 radiculopathy at the time.",
            performing_physician="Precision Orthopaedic Specialties",
            regions_with_active_denervation=["left_vastus_lateralis", "left_tibialis_anterior"],
            nerve_conduction_abnormalities=["L_peroneal_CMAP_0.6mV", "L_tibial_CMAP_1.6mV", "L_superficial_peroneal_SNAP_2.8uV"],
            supports_als=False,
        ),
        provenance=Provenance(source_system=SourceSystem.EHR, asserted_by="precision_orthopaedic"),
    ))

    # --- Spirometry: March 9, 2026 ---
    observations.append(Observation(
        id="obs:spirometry:2026_03_09",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.RESPIRATORY_METRIC,
        name="Spirometry Sitting and Supine",
        respiratory_metric=RespiratoryMetric(
            measurement_date=date(2026, 3, 9),
            fvc_percent_predicted=100.0,
            fvc_liters_sitting=5.0,
            fvc_liters_supine=4.8,
        ),
        provenance=_cc_provenance(),
    ))

    # --- MRI Brain: Feb 21, 2026 ---
    observations.append(Observation(
        id="obs:mri_brain:2026_02_21",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.IMAGING_FINDING,
        name="MRI Brain Without Contrast",
        imaging_finding=ImagingFinding(
            study_date=date(2026, 2, 21),
            modality="mri_brain",
            summary="No motor pathway signal changes typical for ALS. No acute infarct or hemorrhage. Mild generalized volume loss with microvascular changes.",
            findings=[
                "No confluent signal changes along corticospinal tracts",
                "No acute ischemic infarction or hemorrhage",
                "Scattered T2/FLAIR white matter changes (chronic microvascular)",
                "Mild generalized cerebral volume loss",
            ],
            incidental_findings=[
                "1.8x1.1cm lobulated lesion in inferior fourth ventricle (possible ependymoma — contrast MRI recommended)",
                "Cavum septum pellucidum/vergae",
            ],
            als_relevant=False,
        ),
        provenance=_cc_provenance(),
    ))

    # --- MRI Cervical Spine: Feb 21, 2026 ---
    observations.append(Observation(
        id="obs:mri_cervical:2026_02_21",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.IMAGING_FINDING,
        name="MRI Cervical Spine Without Contrast",
        imaging_finding=ImagingFinding(
            study_date=date(2026, 2, 21),
            modality="mri_cervical",
            summary="Normal cord signal intensity. No cord signal abnormality. Multi-level degenerative changes with up to severe foraminal stenoses C3-C7. No high-grade spinal canal stenosis.",
            findings=[
                "Normal cord signal — no myelopathy",
                "C3-C4: moderate-severe disc-osteophyte, mild canal stenosis",
                "C4-C5: severe disc-osteophyte, mild cord compression",
                "C5-C6: severe disc-osteophyte, severe bilateral foraminal stenoses",
                "C6-C7: moderate-severe disc-osteophyte, severe L foraminal stenosis",
            ],
            incidental_findings=[
                "Multinodular thyroid (up to 3cm on left) — ultrasound recommended",
            ],
            als_relevant=False,
        ),
        provenance=_cc_provenance(),
    ))

    # --- Neurological exam findings: Feb 6, 2026 ---
    exam_findings = [
        ("motor", "left_hip_flexion", "4+", "left"),
        ("motor", "left_knee_extension", "4", "left"),
        ("motor", "left_ankle_dorsiflexion", "4+", "left"),
        ("motor", "right_hip_flexion", "5-", "right"),
        ("motor", "right_knee_extension", "4+", "right"),
        ("reflex", "plantar_response", "upgoing", "bilateral"),
        ("reflex", "left_biceps", "3+", "left"),
        ("reflex", "right_achilles", "4", "right"),
        ("tone", "lower_extremity", "A2 (moderate increase)", "bilateral"),
        ("tone", "left_upper_extremity", "A1 (mild increase)", "left"),
        ("gait", "overall", "wide_base spastic; left foot drop; uses 2 hiking poles", "bilateral"),
        ("sensation", "vibration_left_foot", "absent at toe", "left"),
    ]
    for i, (cat, region, finding, lat) in enumerate(exam_findings):
        observations.append(Observation(
            id=f"obs:exam:2026_02_06:{i}",
            subject_ref="traj:draper_001",
            observation_kind=ObservationKind.PHYSICAL_EXAM_FINDING,
            name=f"Neurological Exam: {region}",
            physical_exam_finding=PhysicalExamFinding(
                exam_date=date(2026, 2, 6),
                category=cat,
                region=region,
                finding=finding,
                laterality=lat,
            ),
            provenance=_cc_provenance(),
        ))

    # --- Weight measurements ---
    from ontology.base import TimeFields
    observations.append(Observation(
        id="obs:weight:2025_06_09",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.WEIGHT_MEASUREMENT,
        name="Weight",
        value=117.0,
        unit="kg",
        provenance=_pcp_provenance(),
        time=TimeFields(observed_at=datetime(2025, 6, 9, tzinfo=timezone.utc)),
    ))
    observations.append(Observation(
        id="obs:weight:2026_02_06",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.WEIGHT_MEASUREMENT,
        name="Weight",
        value=111.1,
        unit="kg",
        provenance=_cc_provenance(),
    ))

    # --- Vital signs: Feb 6, 2026 ---
    observations.append(Observation(
        id="obs:vitals:2026_02_06",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.VITAL_SIGN,
        name="Vital Signs",
        value_str="BP 129/95, Pulse 95, SpO2 94%",
        provenance=_cc_provenance(),
    ))

    # --- Medication event: riluzole start ---
    observations.append(Observation(
        id="obs:med:riluzole_start",
        subject_ref="traj:draper_001",
        observation_kind=ObservationKind.MEDICATION_EVENT,
        name="Riluzole Initiated",
        value_str="riluzole started per Dr. Thakore recommendation",
        provenance=_cc_provenance(),
    ))

    # Link observations to trajectory
    trajectory.linked_observation_refs = [o.id for o in observations]

    return patient, trajectory, observations
```

- [ ] **Step 7: Run all tests**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_ingestion_labs.py tests/test_ingestion_patient.py tests/test_erik_trajectory.py -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/ingestion/ tests/test_ingestion_labs.py tests/test_ingestion_patient.py tests/test_erik_trajectory.py && git commit -m "feat: ingest Erik Draper's clinical data — Patient, ALSTrajectory, 40+ Observations"
```

---

## Task 13: Full Test Suite + Final Verification

- [ ] **Step 1: Run the complete test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v --tb=short`
Expected: All tests PASS (DB-dependent tests skip if no PG)

- [ ] **Step 2: Verify object count**

```python
# Quick verification script (run inline, not committed)
from ingestion.patient_builder import build_erik_draper
from ontology.registry import list_types
patient, traj, obs = build_erik_draper()
print(f"Patient: {patient.id}")
print(f"Trajectory: {traj.id}, onset={traj.onset_date}, dx={traj.diagnosis_date}")
print(f"ALSFRS-R: {traj.alsfrs_r_scores[0].total}/48")
print(f"Observations: {len(obs)}")
print(f"Registry types: {len(list_types())}")
```

- [ ] **Step 3: Final commit with all __init__.py files clean**

```bash
cd /Users/logannye/.openclaw/erik && git add -A && git status
```

Review staged files, then:

```bash
git commit -m "chore: Phase 0 complete — canonical substrate with Erik Draper's clinical trajectory"
```

---

## Summary

After completing all 13 tasks, Erik has:

- **25 canonical Pydantic models** conforming to the spec's base envelope
- **17 enum types** covering the ALS ontology
- **30+ typed relations** with observational guards (NEVER upgrade to L3)
- **PostgreSQL schema** (erik_core + erik_ops) with KG tables, audit log, and pgvector
- **Erik Draper's structured clinical data**: 40+ observations spanning labs (NfL, CK, CBC, CMP, lipids, thyroid, B12, folate, copper, ESR), EMG (2 studies), MRI (brain + cervical), spirometry, neurological exam, vitals, weight, medications
- **ALSFRS-R 43/48** with computed subscores and decline rate
- **Hot-reloadable config** system
- **Append-only audit log**
- **Type registry** for deserialization
- **Full test suite** with TDD throughout

**What comes next (separate plans):**
1. **Phase 1: Evidence Fabric** — Literature/trial/dataset ingestion, PubMed ALS connector, ChEMBL connector, evidence bundle builder
2. **Phase 2: World Model MVP** — Latent state estimator, subtype posterior, progression forecast, basic counterfactuals
3. **Phase 3: RL Loop** — Experience stream, action space, reward function, value function, Thompson sampling policy
4. **Phase 4: Cure Protocol Generation** — Planner, protocol candidate builder, abstention logic
