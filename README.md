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

## Current Status: Phase 1A Complete

### Phase 0: Canonical Substrate (Complete)

- **25 canonical Pydantic models** — Patient, ALSTrajectory, Observation, Interpretation, EtiologicDriverProfile, 9 latent state models, EvidenceBundle, EvidenceItem, Intervention, CureProtocolCandidate, MonitoringPlan, MechanismHypothesis, ExperimentProposal, LearningEpisode, ErrorRecord, ImprovementProposal, Branch
- **16 type-safe enums** — SubtypeClass, ProtocolLayer, PCHLayer, EvidenceStrength, ObservationKind, InterventionClass (with gene_therapy, cell_therapy, peptide), etc.
- **38 typed relations** with 12 observational guards (never promote to L3)
- **6 PostgreSQL tables** across 2 schemas (erik_core, erik_ops)
- **Erik Draper's full clinical trajectory** — 51 structured observations from real medical records

### Phase 1A: Evidence Seed (Complete)

Protocol-first evidence fabric organized around the 5 cure protocol layers:

- **93 curated evidence items** across all 5 protocol layers, each tagged with mechanism target, applicable subtypes, Erik eligibility, PCH level, and real PMIDs
- **25 intervention objects** covering approved therapies (riluzole, edaravone, tofersen, Sodesta), Phase 3 trials (pridopidine PREVAiLS, masitinib, jacifusen FUSION), Phase 1/2 (VTx-002 TDP-43 intrabody), failed/withdrawn (AMX0035, BIIB078, zilucoplan), off-label candidates (rapamycin, memantine, perampanel), and preclinical (STMN2/UNC13A ASOs, AAV-BDNF/GDNF)
- **16 canonical ALS drug targets** with UniProt IDs, druggability assessments, and subtype mapping (TDP-43, SOD1, FUS, C9orf72, STMN2, UNC13A, Sigma-1R, EAAT2, BDNF, GDNF, OPTN, TBK1, NEK1, C5, CSF1R, mTOR)
- **10 computational drug design targets** with PDB structures and compound library references
- **Evidence store** with PostgreSQL CRUD, upsert, and protocol-layer queries
- **514 tests** passing in 0.28s

**Evidence coverage by protocol layer:**

| Layer | Name | Evidence Items | Key Interventions |
|-------|------|---------------|-------------------|
| A | Root-cause suppression | 22 | Tofersen, Sodesta, VTx-002, jacifusen |
| B | Pathology reversal | 23 | Pridopidine, rapamycin, STMN2/UNC13A ASOs |
| C | Circuit stabilization | 19 | Riluzole, masitinib, ibudilast, perampanel |
| D | Regeneration | 14 | AAV-BDNF, AAV-GDNF, NMJ stabilization |
| E | Adaptive maintenance | 15 | NfL monitoring, MDC, respiratory surveillance |

### Roadmap

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | Canonical Substrate | **Complete** | Ontology, schema, patient ingestion, Erik's data |
| 1A | Evidence Seed | **Complete** | Curated evidence corpus, interventions, drug targets |
| 1B | Evidence Connectors | Planned | PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets APIs |
| 2 | World Model MVP | Planned | Latent state estimation, subtype posterior, progression forecast |
| 3 | RL Loop | Planned | Experience stream, action space, reward function, value function |
| 4 | Cure Protocol Generation | Planned | Planner, protocol builder, abstention logic |

---

## Project Structure

```
scripts/
  ontology/         # 25 canonical Pydantic models + 16 enums + relations + registry
  db/               # PostgreSQL schema (DDL), connection pool, migrations
  ingestion/        # Clinical document parsing, patient trajectory builder
  evidence/         # Evidence store (PostgreSQL CRUD) + seed builder
  targets/          # Canonical ALS drug target definitions (16 targets)
  audit/            # Append-only event logger
  config/           # Hot-reloadable JSON config
data/
  seed/             # Curated evidence seed (7 JSON files, 128 objects)
  erik_config.json  # Hot-reloadable runtime config
tests/              # 514 pytest tests mirroring scripts/ structure
docs/
  specs/            # Design specifications
  plans/            # Implementation plans
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
pytest tests/ -v              # All 514 tests
pytest tests/ -v -k erik      # Just Erik trajectory tests
pytest tests/ -v -k seed      # Evidence seed validation
```

### Quick Verification

```python
# Patient trajectory
from ingestion.patient_builder import build_erik_draper
patient, trajectory, observations = build_erik_draper()
print(f"Patient: {patient.patient_key}")
print(f"ALSFRS-R: {trajectory.alsfrs_r_scores[0].total}/48")
print(f"Observations: {len(observations)}")

# Evidence fabric
from evidence.seed_builder import load_seed
from evidence.evidence_store import EvidenceStore
stats = load_seed()
store = EvidenceStore()
print(f"Interventions: {stats['interventions_loaded']}")
print(f"Evidence items: {stats['evidence_items_loaded']}")
for layer in ['root_cause_suppression', 'pathology_reversal', 'circuit_stabilization',
              'regeneration_reinnervation', 'adaptive_maintenance']:
    print(f"  {layer}: {len(store.query_by_protocol_layer(layer))} items")
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
- Miller et al., 2025 — Tofersen 3.5-year OLE (JAMA Neurology)
- Pridopidine PREVAiLS Phase 3 (500 patients, enrolling Jan 2026)
- VTx-002 TDP-43 intrabody (FDA Fast Track Jan 2026, PIONEER-ALS Phase 1/2)
- HEALEY platform trial publications (verdiperstat, zilucoplan, pridopidine, CNM-Au8, DNL343)
- Jacifusen/ulefnersen FUSION Phase 3 (FUS-ALS, readout H2 2026)

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
