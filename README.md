# Erik

**Autonomous causal research and cure-protocol engine for ALS.**

Erik is a patient-specific causal machine learning system that builds a deep, mechanistic world model of amyotrophic lateral sclerosis (ALS) and leverages that intelligence to generate curative treatment protocol candidates for a single individual: **Erik Draper**, a 67-year-old male newly diagnosed with limb-onset ALS.

Named after the patient. Built by [Galen Health](https://galenhealth.ai).

---

## Mission

ALS is a progressive, ultimately fatal neurodegenerative disease with no cure. The few approved therapies provide modest survival benefit at best. Clinical trials have failed repeatedly due to patient heterogeneity, wrong-subgroup enrollment, insufficient target engagement, and weak translation from animal models.

Erik takes a different approach: build a causal world model of ALS biology grounded in the Pearl Causal Hierarchy (association, intervention, counterfactual), then use that model to reason about which interventions — alone or in combination — have the highest probability of arresting and reversing disease in one specific patient, given his unique etiologic drivers, molecular state, and clinical trajectory.

The system is designed to:

1. **Ingest and structure** clinical observations, biomarkers, genomics, imaging, and literature into a typed ontology with provenance and uncertainty
2. **Build a causal knowledge graph** of ALS mechanisms, drawing from ChEMBL, PubMed, ClinicalTrials.gov, PRO-ACT, and other ALS data sources
3. **Materialize a latent disease state** for Erik at each timepoint, factorized into etiologic, molecular, neural circuit, functional, reversibility, and uncertainty components
4. **Estimate subtype posterior** over ALS driver programs (SOD1, C9orf72, FUS, TARDBP, sporadic TDP-43, glia-amplified, mixed, unresolved)
5. **Generate cure protocol candidates** as layered interventions (root-cause suppression, pathology reversal, circuit stabilization, regeneration, adaptive maintenance)
6. **Learn autonomously** via reinforcement learning, grounded in measurable environmental effects

---

## Architecture

Erik runs as a single Python process on a MacBook Pro M4 Max (128GB), sharing hardware with its sibling project [Galen](https://github.com/logannye/galen) (a pan-cancer causal research engine). The architecture is inspired by Galen's battle-tested patterns, adapted for ALS-specific ontology and patient-centric reasoning.

```
                        Erik ALS Engine
                    ========================

  Clinical Records ──> Ingestion Pipeline ──> Canonical Objects
  (PDFs, labs, EMG)     (patient_builder)      (BaseEnvelope)
                                                     |
                                                     v
            ┌─────────────────────────────────────────────┐
            │           PostgreSQL (erik_kg)               │
            │  ┌──────────┐  ┌────────────┐  ┌─────────┐ │
            │  │erik_core  │  │erik_core   │  │erik_ops │ │
            │  │.objects   │  │.entities   │  │.audit   │ │
            │  │(canonical)│  │.relations  │  │.config  │ │
            │  └──────────┘  │(KG + PCH)  │  └─────────┘ │
            │                └────────────┘               │
            └─────────────────────────────────────────────┘
                         |                 |
                         v                 v
              World Model (Phase 2)   RL Loop (Phase 3)
                         |                 |
                         v                 v
              Cure Protocol Generation (Phase 4)
```

**Key design decisions:**
- **PostgreSQL as single source of truth** — no SQLite, no file-based state
- **Pydantic v2 canonical models** — every clinical or discovery object inherits from `BaseEnvelope` with provenance, uncertainty, time, and privacy fields
- **Pearl Causal Hierarchy** — relationships are tagged L1 (associational), L2 (interventional), or L3 (counterfactual). Observational relation types are guarded against spurious promotion to L3.
- **Hot-reloadable config** — ~10 parameters in JSON, reloaded without restart
- **Append-only audit log** — every mutation is traceable

---

## The Patient: Erik Draper

| Field | Value |
|-------|-------|
| Onset | January 2025 (left leg weakness, foot drop) |
| Diagnosis | March 6, 2026 (Cleveland Clinic, Dr. Thakore) |
| Onset region | Lower limb (left leg) |
| ALSFRS-R | 43/48 (Bulbar 12, Fine Motor 11, Gross Motor 8, Respiratory 12) |
| Decline rate | -0.39 points/month |
| NfL (plasma) | 5.82 pg/mL (elevated; ref 0-3.65) |
| FVC | 100% predicted (respiratory preserved) |
| EMG | Widespread active and chronic motor axon loss, supportive of ALS |
| MRI Brain | No motor pathway signal changes |
| MRI Cervical | Normal cord signal, no myelopathy |
| Genetic testing | Pending (Invitae saliva kit ordered) |
| Current treatment | Riluzole (started March 6, 2026) |
| Classification | Definite ALS by Gold Coast Criteria |

Erik's complete clinical trajectory is ingested as **51 structured observations** spanning labs (27), physical exam findings (14), EMG studies (2), MRI imaging (2), weight measurements (3), spirometry (1), vital signs (1), and medication events (1).

---

## Current Status: Phase 0 Complete

Phase 0 (Canonical Substrate) delivers the foundational data layer:

- **25 canonical Pydantic models** — Patient, ALSTrajectory, Observation (with LabResult, EMGFinding, RespiratoryMetric, ImagingFinding, PhysicalExamFinding), Interpretation, EtiologicDriverProfile, 9 latent state models (TDP-43, Splicing, Glial, NMJ, Respiratory, Functional, Reversibility, Uncertainty, DiseaseStateSnapshot), EvidenceBundle, EvidenceItem, Intervention, CureProtocolCandidate, MonitoringPlan, MechanismHypothesis, ExperimentProposal, LearningEpisode, ErrorRecord, ImprovementProposal, Branch
- **15 type-safe enums** — SubtypeClass, ProtocolLayer, PCHLayer, ObservationKind, ALSOnsetRegion, etc.
- **38 typed relations** with 12 observational guards (never promote to L3)
- **6 PostgreSQL tables** across 2 schemas (erik_core, erik_ops)
- **441 tests** passing in 0.19s
- **Erik Draper's full clinical trajectory** structured from real medical records

### Roadmap

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | Canonical Substrate | **Complete** | Ontology, schema, patient ingestion, Erik's data |
| 1 | Evidence Fabric | Planned | PubMed, ChEMBL, ClinicalTrials.gov connectors |
| 2 | World Model MVP | Planned | Latent state estimation, subtype posterior, progression forecast |
| 3 | RL Loop | Planned | Experience stream, action space, reward function, value function |
| 4 | Cure Protocol Generation | Planned | Planner, protocol builder, abstention logic |

---

## Project Structure

```
scripts/
  ontology/         # 25 canonical Pydantic models + enums + relations + registry
  db/               # PostgreSQL schema (DDL), connection pool, migrations
  ingestion/        # Clinical document parsing, patient trajectory builder
  audit/            # Append-only event logger
  config/           # Hot-reloadable JSON config
tests/              # 441 pytest tests mirroring scripts/ structure
data/               # Runtime config (erik_config.json)
docs/plans/         # Implementation plans
```

---

## Setup

### Prerequisites

- macOS with Apple Silicon (M4 Max recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- PostgreSQL 17+

### Installation

```bash
# Clone
git clone https://github.com/logannye/ALS.git
cd ALS

# Create conda environment
conda create -n erik-core python=3.12 -y
conda activate erik-core
pip install pydantic "psycopg[binary]" psycopg_pool pytest python-dateutil

# Create database
createdb erik_kg

# Run schema migrations
PYTHONPATH=scripts python -m db.migrate

# Verify
pytest tests/ -v
```

### Running Tests

```bash
conda activate erik-core
cd /path/to/ALS
pytest tests/ -v          # All 441 tests
pytest tests/ -v -k erik  # Just Erik trajectory tests
```

### Quick Verification

```python
from ingestion.patient_builder import build_erik_draper
patient, trajectory, observations = build_erik_draper()

print(f"Patient: {patient.patient_key}")
print(f"Onset: {trajectory.onset_date}")
print(f"Diagnosis: {trajectory.diagnosis_date}")
print(f"ALSFRS-R: {trajectory.alsfrs_r_scores[0].total}/48")
print(f"Observations: {len(observations)}")
```

---

## Philosophical Foundation

Erik follows the "Era of Experience" philosophy (Silver & Sutton, 2025):

> The agent learns primarily from its own interaction with the environment, not from human-curated labels or static datasets. Rewards are grounded in measurable effects — knowledge gained, predictions validated, gaps reduced, truth verified — not in proxy scores.

Applied to ALS: Erik doesn't memorize treatment guidelines. It builds a causal model from first principles, tests its understanding against real data, and generates protocol candidates that it can explain and uncertainty-bound. Every recommendation carries provenance, confidence, and explicit disclosure of what it doesn't know.

---

## References

**Disease biology:**
- Hardiman et al., 2017 — ALS overview (Nat Rev Dis Primers)
- Rosen et al., 1993 — SOD1 mutations in familial ALS
- DeJesus-Hernandez et al., 2011 — C9orf72 repeat expansion
- Neumann et al., 2006 — TDP-43 as disease protein in ALS/FTLD
- Brown et al., 2022; Ma et al., 2022 — UNC13A cryptic exon and TDP-43 mechanism

**Therapeutics:**
- Bensimon et al., 1994 — Riluzole RCT
- Miller et al., 2022 — VALOR tofersen RCT (SOD1-ALS)
- HEALEY platform trial publications (verdiperstat, zilucoplan, pridopidine)

**Data sources:**
- PRO-ACT (13K patient records, 38 trials)
- Answer ALS / Neuromine (multi-omic consortium)
- Project MinE (ALS WGS consortium)
- ALSoD, ClinVar, OMIM (gene/variant databases)

---

## License

Proprietary. All rights reserved by Galen Health.

## Contact

Logan Nye — [logan@galenhealth.ai](mailto:logan@galenhealth.ai)
