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
2. **Build a causal knowledge graph** of ALS mechanisms, drawing from 12 data sources (PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, DrugBank, Reactome, KEGG, STRING, PRO-ACT, ClinVar, OMIM, PharmGKB)
3. **Materialize a latent disease state** for Erik at each timepoint, factorized into etiologic, molecular, neural circuit, functional, reversibility, and uncertainty components
4. **Estimate subtype posterior** over ALS driver programs (SOD1, C9orf72, FUS, TARDBP, sporadic TDP-43, glia-amplified, mixed, unresolved)
5. **Generate and iteratively refine cure protocol candidates** as layered interventions (root-cause suppression, pathology reversal, circuit stabilization, regeneration, adaptive maintenance)
6. **Converge autonomously** on the optimal therapeutic strategy through hypothesis-driven evidence acquisition, causal chain deepening, and protocol regeneration

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
               ┌───────────────┐   ┌────────────────────┐
               │  World Model  │   │   Research Loop     │
               │  (6 stages)   │   │  (15 actions,       │
               │               │◄──│   12 data sources,  │
               │  State → Sub- │   │   hypothesis-driven │
               │  type → Score │   │   convergence)      │
               │  → Assemble → │   │                     │
               │  Counterfact  │   └────────────────────┘
               │  → Output     │             |
               └───────┬───────┘             |
                       |                     |
                       v                     v
              ┌──────────────────────────────────────┐
              │     CureProtocolCandidate             │
              │  (5 layers, evidence-grounded,        │
              │   uncertainty-bounded, approval-gated) │
              └──────────────────────────────────────┘
                                 |
                                 v
                    ACTION READINESS ASSESSMENT
                    (see "When to Act" below)
```

**Key design decisions:**
- **PostgreSQL as single source of truth** — no SQLite, no file-based state
- **Pydantic v2 canonical models** — every clinical or discovery object inherits from `BaseEnvelope` with provenance, uncertainty, time, and privacy fields
- **Pearl Causal Hierarchy** — relationships are tagged L1 (associational), L2 (interventional), or L3 (counterfactual). Observational relation types are guarded against spurious promotion to L3.
- **Single-threaded sequential execution** — no daemon threads (lesson from Galen's GIL contention); one action at a time, deterministic
- **Dual-LLM memory management** — 9B model stays loaded (4.7GB) for research; 35B loaded on demand for protocol generation then unloaded (~1.6s overhead)
- **Hot-reloadable config** — ~45 parameters in JSON, reloaded without restart
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

## How It Works: From Evidence to Cure Protocol

### Stage 1: Build (Complete)

The system infrastructure is fully operational:

- **Canonical Substrate** — 25 Pydantic models, 16 enums, 38 typed relations, PostgreSQL schema, Erik's 51 observations
- **Evidence Fabric** — 93 curated evidence items, 25 interventions, 16 drug targets, 10 computational design targets
- **12 Data Source Connectors** — PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, DrugBank, Reactome, KEGG, STRING, PRO-ACT, ClinVar, OMIM, PharmGKB
- **World Model Pipeline** — 6-stage evidence-grounded reasoning: state materialization → subtype inference → intervention scoring → protocol assembly → counterfactual verification → output
- **Research Loop** — 15 action types, uncertainty-directed policy, hypothesis system, causal chain construction, protocol convergence detection, episode logging

### Stage 2: Research (Running 24/7)

The autonomous research loop operates in two modes:

**Active research mode** (pre-convergence): A 15-action cycle driven by protocol gap analysis:

1. **Analyze gaps** — The intelligence module examines the current protocol to identify the weakest evidence link, shallowest causal chain, most uncertain layer, and missing measurements. Each gap is ranked by priority.
2. **Generate targeted hypotheses** — Instead of generic "generate a hypothesis" prompts, the LLM receives Erik's full clinical context, the specific gap being addressed, relevant evidence items, and a structured prompt asking for a testable claim with search terms and target genes.
3. **Search with purpose** — Hypothesis validation uses the hypothesis's own search terms (extracted by the LLM) to query PubMed, not generic queries. Target genes drive STRING and Reactome lookups.
4. **Deepen causal chains** — For each protocol intervention, build the full mechanism chain (drug → target → pathway → cellular effect → motor neuron survival) grounded in pathway databases.
5. **Regenerate protocol** — When 15+ new evidence items accumulate, re-run the full 6-stage pipeline with the expanded evidence fabric.
6. **Converge** — When top interventions stabilize across 3 consecutive regenerations.

**Deep research mode** (post-convergence): The system does NOT stop. It continuously expands the evidence fabric:

- Every 30 seconds: executes a research query (PubMed, ClinicalTrials, STRING PPI, Reactome pathways, PharmGKB safety)
- Every 3rd step: uses protocol gap analysis to pick the most impactful query instead of rotating through hardcoded searches
- When 15+ new evidence items accumulate: triggers re-convergence with an improved protocol
- Checks for genetic results and config changes on every cycle

### Stage 3: Converge and Refine

The system produces a **CureProtocolCandidate** — a 5-layer treatment strategy with:
- Evidence-grounded intervention selection per layer
- Full causal chains from mechanism to patient outcome
- Drug interaction safety validation (PharmGKB)
- Uncertainty disclosure and missing measurement recommendations
- Human approval gate (always `approval_state=pending`)
- The protocol is regenerated with deeper evidence each time the system re-converges

---

## When to Act: The Decision Framework

**The central tension:** ALS is progressive. Every day of inaction costs Erik motor neurons. But acting on a poorly-grounded protocol wastes time on ineffective interventions. The system resolves this tension with a principled readiness framework.

### Convergence Signals (system-assessed)

The system declares **protocol convergence** when:

| Signal | Threshold | Meaning |
|--------|-----------|---------|
| **Intervention stability** | Top intervention per layer unchanged across 3 consecutive regenerations | The system has explored thoroughly and keeps arriving at the same answer |
| **Evidence saturation** | New searches return <2 novel evidence items per cycle | The available literature has been exhausted for these targets |
| **Causal chain depth** | All top-3 interventions have chains of depth >= 5 | The mechanism from drug to patient outcome is well-grounded |
| **Hypothesis resolution** | >80% of generated hypotheses resolved (supported or refuted) | The system's questions about Erik's disease have been answered |

### Action Readiness Criteria (human-assessed)

Even after convergence, the protocol requires human judgment on:

| Criterion | Question | Who Decides |
|-----------|----------|-------------|
| **Clinical accessibility** | Can Erik actually access the top interventions? (approved drugs vs. trial enrollment vs. compassionate use) | Physician + patient |
| **Safety clearance** | Are drug interactions acceptable? Does Erik's comorbidity profile allow the combination? | Physician |
| **Genetic integration** | Have Invitae results arrived? Has the subtype posterior been updated? | System (auto-triggers `INTERPRET_VARIANT` when `genetics_received=true`) |
| **Timing urgency** | Is Erik's decline rate accelerating? Has ALSFRS-R dropped below a critical threshold? | Physician + PRO-ACT trajectory comparison |
| **Regulatory pathway** | Which interventions need IND applications, IRB approval, or compassionate use requests? | Physician + regulatory |

### The Decision Rule

**Act when:** The protocol has converged AND the top-layer interventions are clinically accessible AND drug safety is cleared AND genetic results have been integrated (or genetics are negative and sporadic TDP-43 is confirmed as dominant subtype).

**Do NOT wait for:**
- Perfect certainty (it will never arrive — all medicine operates under uncertainty)
- All 5 layers to have strong-evidence interventions (Layer D regeneration is preclinical by nature — this should not block Layer A-C action)
- A single "cure" (the protocol is a multi-layer combination strategy; each layer contributes partial benefit)

**Continue running the loop while:**
- Acting on accessible interventions (start riluzole optimization, enroll in eligible trials)
- New evidence is arriving (trial readouts, preprints, genetic results)
- Erik's clinical state changes (new measurements update the disease state snapshot)

### The Protocol is a Living Document

The converged protocol is not a final answer — it is the **best answer given current evidence**. The system continues to refine it as:

- **Genetic results arrive** — subtype posterior shifts, Layer A may restructure entirely
- **Trial results read out** — pridopidine PREVAiLS (H2 2026), VTx-002 PIONEER-ALS (ongoing), jacifusen FUSION (H2 2026) could dramatically change intervention scores
- **Erik's state changes** — new ALSFRS-R assessments, NfL measurements, respiratory tests update the disease state and timing urgency
- **New research publishes** — the loop discovers new evidence and regenerates

The system re-enters active research mode automatically when new data changes the protocol's top-3 interventions.

---

## Operational Status

**Erik is running 24/7** via macOS LaunchAgent (`ai.erik.researcher.plist`), coexisting with Galen on the same M4 Max 128GB.

### First Live Run Results (March 26, 2026)

| Metric | Result |
|--------|--------|
| Steps to convergence | **74** |
| Protocol versions generated | 4 |
| Evidence items acquired | 199 new (342 total) |
| Causal chains at depth 5 | **7/7** (all interventions fully traced) |
| Hypotheses generated | 10 |
| Hypotheses resolved | 2 |
| Action types exercised | 8 of 15 |
| Time to convergence | ~17 minutes |

### Current Mode: Monitoring

The system has converged on `proto:erik_draper_v1` and is now in monitoring mode, checking every 5 minutes for:
- **Genetic results** — set `genetics_received: true` in config to trigger reactivation
- **New evidence** — >20 new items in DB triggers active research
- **Config changes** — hot-reloaded on each monitoring cycle

### Build Phases

| Phase | Name | Status | Description |
|-------|------|--------|-------------|
| 0 | Canonical Substrate | **Complete** | Ontology, schema, patient ingestion, Erik's 51 observations |
| 1A | Evidence Seed | **Complete** | 93 curated evidence items, 25 interventions, 16 drug targets |
| 1B | Evidence Connectors | **Complete** | 5 original API connectors (PubMed, ClinicalTrials, ChEMBL, OpenTargets, DrugBank) |
| 2 | World Model Pipeline | **Complete** | 6-stage protocol generation: state → subtype → scoring → assembly → counterfactual → output |
| 3 | Autonomous Research Loop | **Complete** | 15-action hypothesis-driven loop with causal chains, convergence detection, episode logging |
| 3B | Evidence Expansion | **Complete** | 7 new data sources (Reactome, KEGG, STRING, PRO-ACT, ClinVar, OMIM, PharmGKB) |
| 4 | Live Execution | **Complete** | First convergence achieved — 24/7 LaunchAgent running |
| 5 | Clinical Translation | Next | Physician review, trial enrollment, compassionate use applications |

**811 tests passing.** System is fully operational with intelligent, gap-driven research and balanced action diversity.

---

## Project Structure

```
scripts/
  ontology/         # 25 canonical Pydantic models + 16 enums + relations + registry
  db/               # PostgreSQL schema (DDL), connection pool, migrations
  ingestion/        # Clinical document parsing, patient trajectory builder
  evidence/         # Evidence store (PostgreSQL CRUD) + seed builder
  connectors/       # 11 connectors (PubMed, ClinicalTrials, ChEMBL, OpenTargets, DrugBank,
                    #   Reactome, KEGG, STRING, ClinVar, OMIM, PharmGKB)
  targets/          # Canonical ALS drug target definitions (16 targets)
  llm/              # MLX LLM inference wrapper (generate, generate_json, lazy loading, unload)
  world_model/      # 6-stage cure protocol pipeline (state, subtype, scoring, assembly, CF, orchestrator)
    prompts/        # Evidence-grounded LLM prompt templates
  research/         # Autonomous research loop (15 actions, policy, rewards, hypotheses,
                    #   causal chains, convergence, trajectory, intelligence)
  run_loop.py       # 24/7 continuous entry point (LaunchAgent target)
  monitor.py        # Real-time terminal dashboard for loop progress
  audit/            # Append-only event logger
  config/           # Hot-reloadable JSON config
data/
  seed/             # Curated evidence seed (7 JSON files, 128 objects)
  erik_config.json  # Hot-reloadable runtime config (~45 keys)
logs/               # LaunchAgent log output (erik_research.log, erik_research.err)
tests/              # 811 pytest tests mirroring scripts/ structure
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
pip install pydantic "psycopg[binary]" psycopg_pool pytest python-dateutil requests mlx-lm

# Create database
createdb erik_kg

# Run schema migrations
PYTHONPATH=scripts python -m db.migrate

# Verify
pytest tests/ -v -k "not network and not chembl and not llm"
```

### Running Erik 24/7

```bash
# Start the continuous research loop (LaunchAgent — survives reboots)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/ai.erik.researcher.plist

# Watch progress in real time (separate terminal)
cd /Users/logannye/.openclaw/erik
PYTHONPATH=scripts /opt/homebrew/Caskroom/miniconda/base/envs/erik-core/bin/python scripts/monitor.py

# Tail the log file
tail -f /Users/logannye/.openclaw/erik/logs/erik_research.log

# Stop the loop
launchctl bootout gui/$(id -u)/ai.erik.researcher
```

### Manual Single Run

```bash
# Generate Erik's first cure protocol (single pass, no loop)
PYTHONPATH=scripts /opt/homebrew/Caskroom/miniconda/base/envs/erik-core/bin/python -c "
from world_model.protocol_generator import generate_cure_protocol
result = generate_cure_protocol(use_llm=True)
print(f'Protocol: {result[\"protocol\"].id}')
print(f'Layers: {len(result[\"protocol\"].layers)}')
"

# Run N steps of the research loop (foreground, live output)
PYTHONPATH=scripts /opt/homebrew/Caskroom/miniconda/base/envs/erik-core/bin/python -c "
from evidence.evidence_store import EvidenceStore
from research.dual_llm import DualLLMManager
from research.loop import run_research_loop
state = run_research_loop('traj:draper_001', EvidenceStore(), DualLLMManager(), max_steps=100)
print(f'Converged: {state.converged}, Steps: {state.step_count}, Evidence: {state.total_evidence_items}')
"
```

### When Genetic Results Arrive

```bash
# Edit config to trigger re-activation:
# In data/erik_config.json, change: "genetics_received": false → true
# Erik detects this within 5 minutes and re-enters active research,
# running ClinVar variant interpretation and regenerating the protocol
# with the updated subtype posterior.
```

---

## Philosophical Foundation

Erik follows the "Era of Experience" philosophy (Silver & Sutton, 2025):

> The agent learns primarily from its own interaction with the environment, not from human-curated labels or static datasets. Rewards are grounded in measurable effects — knowledge gained, predictions validated, gaps reduced, truth verified — not in proxy scores.

Applied to ALS: Erik doesn't memorize treatment guidelines. It builds a causal model from first principles, tests its understanding against real data, and generates protocol candidates that it can explain and uncertainty-bound. Every recommendation carries provenance, confidence, and explicit disclosure of what it doesn't know.

The system is designed with a clear ethical boundary: it generates recommendations, never decisions. The `approval_state=pending` gate ensures that every protocol is reviewed by a qualified physician before any action is taken on Erik's behalf. The system's job is to be the most informed, most rigorous, most honest research assistant possible — not to replace clinical judgment.

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
- Reactome, KEGG (curated biological pathways)
- STRING (protein-protein interaction network)
- PharmGKB (pharmacogenomics)
- ChEMBL 36 (compound bioactivity)

---

## License

Proprietary. All rights reserved by Galen Health.

## Contact

Logan Nye — [logan@galenhealth.ai](mailto:logan@galenhealth.ai)
