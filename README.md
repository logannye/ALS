# Erik

**Find or create the curative drug for Erik Draper's ALS.**

Named after the patient. Built by [Galen Health](https://galenhealth.ai).

---

## Grand Purpose

> **This system exists for one reason: to identify or computationally design the drug that will cure Erik Draper's amyotrophic lateral sclerosis.**

Erik Draper is a 67-year-old man diagnosed with ALS in March 2026. ALS is progressive and fatal. There is no cure. The approved therapies extend survival by months at best. Clinical trials fail repeatedly.

This system takes a fundamentally different approach. Instead of waiting for the pharmaceutical industry to find a general ALS drug through decade-long trial-and-error, Erik builds a deep causal understanding of the molecular biology of ALS, maps it to one specific patient's biology, and uses that understanding to either identify an existing compound or computationally design a new molecule that would halt or reverse the disease in this specific human being.

**Every design decision, every line of code, every architectural choice must be measured against this question:**

> *Does this bring us closer to identifying or creating the curative drug for Erik Draper's ALS?*

If the answer is no, it is not a priority. The system has a patient waiting.

---

## The Four Stages

The system progresses through four stages toward its goal:

### Stage 1: Causal Anatomy & Molecular Biology
Build a computable model of every entity and relationship involved in motor neuron survival and death — genes, proteins, pathways, post-translational modifications, expression profiles, binding affinities, structural conformations. Not just names and associations, but quantitative, mechanistic understanding grounded in the Pearl Causal Hierarchy (L1 association → L2 intervention → L3 counterfactual).

### Stage 2: Causal Pathophysiology — Beyond the Literature
Discover WHY motor neurons die. The answer is not fully in PubMed — if it were, someone would have found it. The system must use causal inference, computational experiments, and structured gap analysis to identify the missing links in the chain from molecular trigger → motor neuron death. This stage produces understanding that does not yet exist in the research literature.

### Stage 3: Precision Mapping to Erik Draper
Map the general causal model of ALS to Erik's specific biology. His genetics (pending Invitae results), his biomarkers (NfL 5.82 pg/mL, ALSFRS-R 43/48), his disease trajectory (-0.39 points/month), his comorbidities. Build a patient-specific molecular model that predicts which pathways are disrupted in HIS motor neurons and which molecular targets would be most effective for HIM.

### Stage 4: Drug Identification or Creation
Given precise molecular targets grounded in deep causal understanding of Erik's specific disease:
- **Screen existing compounds** (ChEMBL, DrugBank, BindingDB) for molecules that bind the target with sufficient affinity and cross the blood-brain barrier
- **Computationally design new molecules** if no adequate existing compound exists — using the 16 characterized drug design targets with PDB/AlphaFold structures, molecular docking, and generative chemistry
- **Predict ADMET properties** (absorption, distribution, metabolism, excretion, toxicity) and synthetic feasibility
- **Deliver actionable output** to Erik's care team: the specific molecule(s), the target(s), the evidence chain, and the access pathway (clinical trial enrollment, compassionate use, compounding, or de novo synthesis)

---

## Architecture

Erik runs 24/7 as a FastAPI application on Railway, with a Next.js family dashboard on Vercel. LLM inference uses Amazon Bedrock (Nova Micro for research, Nova Pro for protocol generation). PostgreSQL is the single source of truth. The architecture is inspired by its sibling project [Galen](https://github.com/logannye/galen) (a pan-cancer causal research engine), adapted for ALS-specific ontology and patient-centric drug discovery.

```
                    Erik ALS Drug Discovery Engine
                  ==================================

  Stage 1: CAUSAL ANATOMY                     Stage 2: PATHOPHYSIOLOGY
  ┌─────────────────────┐                     ┌─────────────────────────┐
  │  28 Data Sources     │                     │  Causal Gap Analysis     │
  │  (PubMed, ChEMBL,   │──── Evidence ────>  │  Structure Learning      │
  │  STRING, AlphaFold,  │     Acquisition     │  Hypothesis Generation   │
  │  BindingDB, DepMap,  │                     │  Computational Expts     │
  │  GDSC, Reactome...)  │                     │  Counterfactual Verify   │
  └─────────────────────┘                     └────────────┬────────────┘
                                                           │
              ┌─────────────────────────────────────────────┘
              │
              v
  Stage 3: PRECISION MAPPING                  Stage 4: DRUG DESIGN
  ┌─────────────────────────┐                 ┌──────────────────────────┐
  │  Erik's Genetics         │                 │  16 Drug Design Targets   │
  │  Erik's Biomarkers       │── Patient ──>   │  Molecular Docking        │
  │  Erik's Disease State    │   Model         │  De Novo Generation       │
  │  Subtype Posterior       │                 │  ADMET/BBB Prediction     │
  │  Molecular Twin          │                 │  Candidate Ranking        │
  └─────────────────────────┘                 └────────────┬─────────────┘
                                                           │
              ┌────────────────────────────────────────────┘
              │
              v
  ┌──────────────────────────────────────────────────────┐
  │         CureProtocolCandidate                         │
  │  5 layers: root_cause → pathology_reversal →          │
  │  circuit_stabilization → regeneration → maintenance   │
  │  + Designed molecules for each target                 │
  │  + Trial urgency scores                               │
  │  + Physician-ready evidence report                    │
  └──────────────────────────────────────────────────────┘
              │
              v
  ┌──────────────────────────────────────────────────────┐
  │  PostgreSQL (Railway)        │  Family Dashboard      │
  │  erik_core.objects           │  (Next.js + Vercel)    │
  │  erik_core.entities          │  Live research status   │
  │  erik_core.relationships     │  Protocol viewer        │
  │  erik_ops.audit              │  Evidence explorer      │
  │  erik_ops.research_state     │  Ask questions          │
  └──────────────────────────────│  Upload health data     │
                                 │  Trial tracker          │
  LLM: Amazon Bedrock            └──────────────────────────┘
  (Nova Micro + Nova Pro)
```
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
- **15 Data Source Connectors** — PubMed, ClinicalTrials.gov, ChEMBL, OpenTargets, DrugBank, Reactome, KEGG, STRING, PRO-ACT, ClinVar, OMIM, PharmGKB, Galen KG (cross-disease), bioRxiv/medRxiv (preprints), Galen SCM (causal graph)
- **World Model Pipeline** — 7-stage evidence-grounded reasoning: state materialization → subtype inference → intervention scoring → **combination synergy analysis** → protocol assembly → **adversarial counterfactual verification** → output
- **Research Loop** — 30 action types, Thompson sampling policy (with fixed-cycle fallback), hypothesis system, causal chain construction (depth 10), protocol convergence detection, episode logging, gap resolvability classification, clinical subtype posterior, **clinical trial eligibility matching**, **adversarial protocol verification**, **PRO-ACT trajectory matching**, **stagnation detection and recovery**

### Stage 2: Research (Running 24/7)

The autonomous research loop operates in two modes:

**Active research mode** (pre-convergence): A 15-action cycle driven by protocol gap analysis:

1. **Analyze gaps with resolvability classification** — The intelligence module examines the current protocol to identify the weakest evidence link, shallowest causal chain, most uncertain layer, and missing measurements. Each gap is classified as `computational` (resolvable by searching literature/databases) or `clinical_required` (requires a clinical test on the patient). Only computational gaps drive research actions; clinical gaps generate recommendations to the physician.
2. **Generate targeted hypotheses** — Instead of generic "generate a hypothesis" prompts, the LLM receives Erik's full clinical context, the specific gap being addressed, relevant evidence items, and a structured prompt asking for a testable claim with search terms and target genes.
3. **Search with purpose** — Hypothesis validation uses the hypothesis's own search terms (extracted by the LLM) to query PubMed, not generic queries. Target genes drive STRING and Reactome lookups. Cross-disease knowledge from the Galen cancer KG provides drug repurposing candidates via shared pathways (autophagy/mTOR, HDAC, oxidative stress).
4. **Deepen causal chains** — For each protocol intervention, build the full mechanism chain (drug → target → pathway → cellular effect → motor neuron survival) to depth 10, grounded in pathway databases.
5. **Regenerate protocol** — When uncertainty score drops meaningfully, re-run the full 6-stage pipeline with the expanded evidence fabric.
6. **Converge** — When top interventions stabilize across 3 consecutive regenerations and uncertainty score is stable.

**Deep research mode** (post-convergence): The system does NOT stop. It continuously expands the evidence fabric:

- Every 30 seconds: executes a research query (PubMed, ClinicalTrials, STRING PPI, Reactome pathways, PharmGKB safety, Galen KG cross-reference)
- Every 3rd step: uses protocol gap analysis to pick the most impactful *actionable* query — clinical-required gaps (e.g., genetic testing) are filtered out and logged as recommendations
- When uncertainty score drops meaningfully (>5% over a 5-step window): triggers re-convergence with an improved protocol
- Checks for genetic results and config changes on every cycle
- Periodically logs clinical recommendations for gaps that require physical tests (genetic testing, CSF biomarkers, etc.)

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
| **Causal chain depth** | All top-3 interventions have chains of depth >= 10 | The mechanism from drug to patient outcome is deeply grounded |
| **Uncertainty score** | Score stable below 0.3 across 5+ monitoring cycles | Evidence is well-distributed across all protocol layers |
| **Hypothesis resolution** | >80% of generated hypotheses resolved (supported or refuted) | The system's questions about Erik's disease have been answered |

### Action Readiness Criteria (human-assessed)

Even after convergence, the protocol requires human judgment on:

| Criterion | Question | Who Decides |
|-----------|----------|-------------|
| **Clinical accessibility** | Can Erik actually access the top interventions? (approved drugs vs. trial enrollment vs. compassionate use) | Physician + patient |
| **Safety clearance** | Are drug interactions acceptable? Does Erik's comorbidity profile allow the combination? | Physician |
| **Genetic integration** | Have Invitae results arrived? Has the subtype posterior been updated? The system computes a clinical subtype posterior from Erik's features (P(sporadic_tdp43) ≈ 0.65), reducing but not eliminating the value of genetic confirmation. | System (auto-triggers `INTERPRET_VARIANT` when `genetics_received=true`) |
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

## Phase 8: Production Deadlock Fix (March 30, 2026)

Five fixes to break Erik's hypothesis deadlock and restore evidence flow after production observation showed the loop stuck generating hypotheses without validating or expiring them.

### Thompson Path: VALIDATE_HYPOTHESIS Integration
The Thompson sampling path never included `VALIDATE_HYPOTHESIS` as a candidate action — only the fixed-cycle path could validate. This meant Thompson mode accumulated hypotheses to `max_active` (10) and then deadlocked: `GENERATE_HYPOTHESIS` was infeasible (at cap), but no path existed to resolve them. Fix: `VALIDATE_HYPOTHESIS` added to Thompson candidate list with feasibility guard (only when `active_hypotheses` is non-empty), plus `_build_thompson_params` handles parameter construction.

### Hypothesis Expiry at Max Active
`_maybe_expire_hypotheses` only expired based on validation-ratio (avg validations per hypothesis > threshold). But if no validations ever ran (Thompson deadlock above), the ratio was 0/N and expiry never triggered. Fix: force-expire the oldest hypothesis when `len(active_hypotheses) >= max_active`, breaking the deadlock. Thompson path now also calls `_maybe_expire_hypotheses` before action selection.

### Galen SCM Causal Depth Inflation
`query_galen_scm` reported `causal_depth_added=1` even when all evidence was duplicate (DB delta = 0). This inflated Thompson success signal and causal chain scores despite producing no new knowledge. Fix: when `_true_new == 0`, zero out `causal_depth_added` — no new evidence means no real causal depth gained.

### DrugBank Encoding Fix
DrugBank CSV files contain non-UTF-8 bytes (e.g., `0xb0` degree symbol in drug names). Both `_VOCAB_PATH` and `_TARGET_LINKS_PATH` reads crashed with `UnicodeDecodeError`. Fix: `errors="replace"` on both `open()` calls.

### CHALLENGE_INTERVENTION Executor
`ActionType.CHALLENGE_INTERVENTION` existed (from Phase 4C adversarial verification) but was missing from the dispatch table in `_execute_action()`. Any Thompson selection of this action returned "Unknown action". Fix: wired `_exec_challenge_intervention` into dispatch with full PubMed adversarial query pipeline and challenge count tracking in state.

### Evidence Stagnation Detection
New stagnation recovery mechanism: checkpoints total evidence count every 50 steps, and when growth falls below `stagnation_min_growth` (default 5) over a `stagnation_detection_window` (default 200 steps), the system expires half of active hypotheses and resets all Thompson posteriors to (1.0, 1.0). This breaks out of local minima where the loop is cycling through unproductive actions.

### Results
- 19 new tests covering all 5 fixes
- 1,212 total tests, 0 regressions
- 3 pre-existing external API test failures (ClinicalTrials.gov 404, OpenTargets schema change, stale action count assertion)

---

## Phase 7: Deep Causal Understanding (March 29, 2026)

Structural overhaul to move Erik from evidence accumulation to mechanistic reasoning grounded in real ALS biology. Evidence rate increased 78x (from ~2 items/hour to ~156 items/hour).

### Knowledge Graph Entity Extraction
`erik_core.entities` and `erik_core.relationships` were empty (0 rows) despite 1,700+ evidence items. New `entity_extractor.py` scans evidence body fields and claim text to extract genes, proteins, drugs, and mechanisms into a proper graph structure. Backfill produced 1,600+ entities and 145+ relationships. Runs incrementally after each evidence-producing step. Respects Pearl Causal Hierarchy constraints (observational relations never L3).

### 9 New Database Connectors (24 total)
| Database | Size | What It Provides |
|----------|------|------------------|
| **GTEx** (on disk) | 153MB | Gene expression across 45 tissues — validates ALS targets in spinal cord motor neurons |
| **ClinVar** (on disk) | 3.8GB | Variant pathogenicity classifications — critical for interpreting Erik's pending Invitae results |
| **GWAS Catalog** (downloaded) | 626MB | ALS-associated risk loci from genome-wide studies |
| **BindingDB** (downloaded) | 7.9GB | 3.2M experimental binding affinity measurements (Ki/IC50/Kd) for drug design |
| **Human Protein Atlas** (downloaded) | 37MB | Protein-level expression, subcellular localization, disease involvement |
| **DrugBank** (CC0 open data) | 1.1MB | Drug vocabulary + target-UniProt links for drug repurposing |
| **AlphaFold** (on disk) | 24K structures | 3D protein structures with pLDDT confidence for computational drug design |
| **Reactome** (on disk) | 23K pathways | Complete human pathway hierarchy for multi-step mechanism cascades |
| **ALSoD** (web API) | 160+ genes | ALS-specific gene variant data with evidence tiers |

### Computational Experiment Engine
New `als_computation_executor.py` runs in-silico experiments using ChEMBL (30.7GB), DepMap (409MB), and GDSC2 databases. Four experiment types: gene essentiality (CRISPR Chronos scores), drug sensitivity (LN_IC50), binding affinity (Tanimoto similarity), and drug interaction profiling. Produces quantitative, falsifiable evidence without clinical data.

### Targeted PubMed Query Strategy
3-strategy cycling: static query bank (40 queries) → dynamic hypothesis-derived terms → targeted queries from 16 ALS target gene definitions. Trials now rotate through all 7 protocol interventions instead of always searching one layer.

### Thompson SCM Exploitation Fix
`query_galen_scm` was consuming 70% of steps (reward=1.39 for causal depth even with 0 new evidence). Fix: `causal_depth_added` now requires actual new evidence, and depth-without-evidence gets 90% reward discount. SCM dropped from 70% to 2.3%.

### Hypothesis Validation Lifecycle
`resolve_hypothesis()` existed but was never called — all 1,408 hypotheses stuck at "generated". Now: `_exec_validate_hypothesis` updates hypothesis status in DB to "searching" or "supported" based on PubMed evidence found.

### Action Dominance Prevention
`DEEPEN_CAUSAL_CHAIN` silently fell back to `GENERATE_HYPOTHESIS` when all chains were at depth 10, bypassing the consecutive cap. New `_action_is_feasible()` pre-filters Thompson candidates to remove infeasible actions before sampling.

### Startup State Sanitization
New `_sanitize_resumed_state()` runs on every restart: flushes old-format hypothesis IDs, resets stale EMA values, resets Thompson posteriors, and uses DB count as evidence truth (was inflated from 9,111 to match the real 1,604).

### Results
- Evidence: 1,604 → 1,865+ (and growing at ~156 items/hour)
- KG: 0 → 1,600+ entities, 145+ relationships
- Action diversity: 20 different action types exercised per cycle
- Protocol regenerated 4+ times with new multi-source evidence
- 94 tests, 0 regressions

---

## Phase 6: Production Stall Recovery (March 29, 2026)

Five compounding bugs caused a complete evidence stall — 220 steps with zero new evidence, total frozen at 9,111. Diagnosed and resolved:

### Hypothesis Statement Deduplication
The LLM's "DO NOT DUPLICATE" context was receiving hypothesis hash IDs (`hyp:abc123`) instead of readable statements. The LLM could not interpret these, so it regenerated the same CK1δ/TDP-43 hypothesis 163 times in one day. Now: `active_hypotheses` stores full hypothesis statements, Jaccard threshold tightened from 0.60 to 0.45, pre-generation saturation check limits 3 hypotheses per gap type, and post-generation dedup rejects near-duplicates before storage. Config: `hypothesis_dedup_threshold`, `gap_same_type_max`.

### Gap Analysis Rotation
The `unvalidated_safety` gap had a hardcoded priority of 0.7 that never decayed — it was selected 2,159 times consecutively. The recency penalty mechanism existed in code but `state.last_gap_layers` was never updated (dead code). Now: `last_gap_layers` is tracked in the research step, recency penalty applies to all gap types including safety, and the sliding window is capped at 10 entries. After 2 safety selections, priority drops from 0.7 → 0.175 and sparse protocol layers overtake it. Config: `gap_recency_window`.

### Dynamic Query Expansion
The static query bank had only 20 queries (4 per layer) — all exhausted after 9,111 items. Now: 40 static queries (8 per layer) covering broader mechanisms (neuroinflammation, mitochondrial dysfunction, stress granules, NMJ preservation, metabolic intervention). PubMed and preprint searches alternate between static and dynamic queries, where dynamic queries extract biomedical terms from hypothesis statements for targeted evidence acquisition.

### Thompson Posterior Wiring
`_update_posteriors()` and `_apply_decay()` existed in `policy.py` but were never called from the research loop — posteriors stayed at (1.0, 1.0) forever, making Thompson sampling equivalent to random selection. Now: posteriors are updated after every step (success = evidence gained, hypothesis generated, chain deepened, or protocol regenerated; failure = none of the above), decay is applied at configured intervals, and posteriors are persisted in state.

### Safeguards
- 30 new tests covering all five fixes, backward compatibility, and state serialization
- All config keys are hot-reloadable (no restart required for tuning)
- Old-format hypothesis IDs in state are handled gracefully (backward compat)

---

## Phase 5: Research Loop Recovery (March 28, 2026)

Three systemic failures diagnosed and resolved:

### Eliminated Empty Steps (79% → ~25%)
Six of eight acquisition actions were consistently returning zero evidence — two had no executor implementation (`SEARCH_PREPRINTS`, `QUERY_GALEN_SCM`), four had silent connector failures. The dispatcher now logs all errors. A yield-aware skip mechanism uses EMA action values to bypass consistently-unproductive actions in favour of sources that produce evidence. Config: `yield_skip_min_count`, `yield_skip_threshold`.

### Broke Hypothesis Fixation (partially — completed in Phase 6)
Every hypothesis was an STMN2/TDP-43 variant because gap analysis deterministically picked the same minimum-evidence layer. Now: all sparse layers (< 30 evidence items) appear as gap candidates with evidence-inverse priority and a recency penalty (0.5^n halving per recent targeting). All hypothesis prompts inject "PRIOR HYPOTHESES — DO NOT DUPLICATE" context. Jaccard-similarity deduplication rejects hypotheses with > 60% keyword overlap with existing active hypotheses. The unused `research_hypothesis_max_active=10` config value is now enforced. **Note:** Phase 5 stored hypothesis IDs instead of statements in the dedup context, rendering it ineffective — fixed in Phase 6.

### Increased Evidence Utilization (9 → 30+ citations)
The protocol was citing only 9–11 evidence items from 7,590 collected because `max_per_layer=2` discarded all non-selected intervention evidence. Now: `max_per_layer` is configurable (raised to 3), protocol body includes `supporting_evidence_refs` from all scored interventions, protocol ID increments with version number, and the LLM scoring prompt instructs exhaustive citation.

### Structural Safeguards
Thompson sampling enabled by default (was implemented but disabled). Search queries now rotate across 4 variants per layer with year suffix for freshness, preventing query staleness.

---

## Phase 4C: Research Enhancements (March 27, 2026)

Seven new capabilities added to strengthen Erik's research and clinical translation:

### Clinical Trial Eligibility Matching
Every time the system searches ClinicalTrials.gov, it now computes Erik's precise eligibility for each trial — checking age, sex, ALSFRS-R, FVC, disease duration, riluzole status, and genetic requirements against structured and free-text criteria. Eligible trials are logged prominently and tracked in a persistent watchlist (`erik_ops.trial_watchlist`). The system errs toward "likely" over "no" — missing an eligible trial is worse than flagging one for physician review.

### PRO-ACT Trajectory Matching
Matches Erik's ALSFRS-R trajectory against ~13,000 historical ALS patients using dynamic time warping (DTW) alignment. Estimates median survival, 25th/75th percentile bounds, and intervention window closure times for each protocol layer. This transforms the protocol from "best interventions given current state" to "best interventions given where Erik is heading and how fast."

### bioRxiv/medRxiv Preprint Connector
Searches preprint servers for ALS research that hasn't yet been indexed by PubMed, eliminating a 6-12 month recency gap. All preprint evidence is clamped to `EvidenceStrength.emerging` and tagged `peer_reviewed: false` — preprints inform but never inflate intervention scores. When a preprint is later published and appears via PubMed, the preprint version is automatically superseded.

### Adversarial Protocol Verification
Actively searches for evidence that *contradicts* the protocol's top interventions — failed trials, harm signals, disputed mechanisms. Corrects the research loop's structural confirmation bias. If an intervention accumulates 3+ contradicting papers with no countervailing support, it's flagged as "contested" and the protocol discloses this with an alternative.

### Drug Combination Synergy Analysis
A new Stage 3B in the pipeline that analyzes pairwise interactions between protocol interventions. Detects pathway redundancy (>60% overlap in causal chains), pharmacodynamic antagonism (opposing effects on shared pathway nodes), and potential synergy. Antagonistic pairs trigger automatic substitution with the next-best intervention for that layer.

### Thompson Sampling Policy
Replaces the fixed 5-step action cycle with information-theoretic action selection. Each action type maintains a Beta posterior tracking its probability of producing nonzero evidence. Actions with high expected yield are selected more often; exhausted sources are naturally deprioritized. Includes decay (old observations matter less as evidence landscape changes) and a diversity floor (every action exercised at least once per 30 steps). Activatable via `thompson_policy_enabled` config flag; falls back to the fixed cycle when disabled.

### Galen SCM Integration
Queries Galen's structural causal model for cross-disease causal reasoning. Instead of just looking up entity relationships, Erik can now walk the L2/L3 causal graph downstream from drug targets (recursive CTE, depth 3) and assess pathway strength (how many causal edges Galen has accumulated for a pathway). A pathway with 347 L3 edges in Galen's cancer KG provides much stronger cross-disease evidence than one with 5.

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

### Current Mode: Active Research (Phase 7 — Deep Causal Understanding)

The system is running with 24 data connectors across 9 databases, a knowledge graph with 1,600+ entities, computational experiments (DepMap/GDSC/ChEMBL), and targeted gene-specific PubMed queries. Evidence is flowing at ~156 items/hour across 20 action types. Monitor with:

```bash
tail -f /Users/logannye/.openclaw/erik/logs/erik_research.log
```

The system checks every cycle for:
- **Genetic results** — set `genetics_received: true` in config to trigger reactivation
- **New evidence** — >10 new items triggers protocol regeneration
- **Config changes** — hot-reloaded every 10 steps

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
| 4B | Research Optimization | **Complete** | Evidence tracking fix, gap resolvability classification, connector repair, uncertainty-based convergence, Galen KG cross-reference, clinical subtype posterior, entity tagging |
| 4C | Research Enhancements | **Complete** | 7 new capabilities: clinical trial eligibility, PRO-ACT trajectory matching, bioRxiv preprints, adversarial verification, combination synergy, Thompson sampling, Galen SCM |
| 5 | Research Loop Recovery | **Complete** | Fixed 3 systemic failures: 79% empty steps → ~25%, hypothesis fixation → diversity, 0.12% evidence utilization → 30+ citations. Thompson enabled, yield-aware skip, hypothesis dedup, rotating queries. 29 new tests. |
| 6 | Production Stall Recovery | **Complete** | Fixed 5 compounding failures causing complete evidence stall (220 steps, 0 evidence). Hypothesis statement storage, gap recency penalty, dynamic queries, Thompson posterior wiring. 30 new tests. |
| 7 | Deep Causal Understanding | **Complete** | 9 new databases (GTEx, ClinVar, GWAS, BindingDB, HPA, DrugBank, AlphaFold, Reactome, ALSoD), KG entity extraction, computational experiments (DepMap/GDSC/ChEMBL), targeted queries, SCM fix, hypothesis validation. 78x evidence rate. 94 tests. |
| 8 | Clinical Translation | Next | Physician review, trial enrollment, compassionate use applications |

System is fully operational with 24 data connectors, knowledge graph (1,600+ entities, 145+ relationships), computational experiments (gene essentiality, drug sensitivity, binding affinity), 9 local databases, targeted gene-specific queries, intelligent gap-driven research, resolvability-aware gap classification, clinical subtype posterior, cross-disease knowledge transfer from Galen, uncertainty-based convergence, clinical trial eligibility matching, adversarial protocol verification, drug combination synergy analysis, PRO-ACT trajectory matching, Thompson sampling action selection, yield-aware action routing, hypothesis deduplication, dynamic query expansion, and configurable protocol assembly.

---

## Project Structure

```
scripts/
  ontology/         # 25 canonical Pydantic models + 16 enums + relations + registry
  db/               # PostgreSQL schema (DDL), connection pool, migrations
  ingestion/        # Clinical document parsing, patient trajectory builder
  evidence/         # Evidence store (PostgreSQL CRUD) + seed builder
  connectors/       # 24 connectors (PubMed, ClinicalTrials, ChEMBL, OpenTargets, DrugBank,
                    #   Reactome, KEGG, STRING, ClinVar, OMIM, PharmGKB, GalenKG, GalenSCM,
                    #   bioRxiv/medRxiv, PRO-ACT, ALSoD, GTEx, ClinVar-local, GWAS Catalog,
                    #   BindingDB, HPA, DrugBank-local, AlphaFold, Reactome-local)
  targets/          # Canonical ALS drug target definitions (16 targets)
  llm/              # MLX LLM inference wrapper (generate, generate_json, lazy loading, unload)
  world_model/      # 7-stage cure protocol pipeline (state, subtype, scoring, combination
                    #   analysis, assembly, adversarial counterfactual, orchestrator)
    prompts/        # Evidence-grounded LLM prompt templates
    trajectory_matcher.py  # PRO-ACT DTW matching + Kaplan-Meier survival estimation
    combination_analyzer.py  # Drug synergy/antagonism/redundancy detection
  knowledge_quality/ # KG entity extraction from evidence items
  executors/        # ALS computational experiments (DepMap, GDSC, ChEMBL binding)
  research/         # Autonomous research loop (28 actions, Thompson sampling policy,
                    #   rewards, hypotheses, causal chains, convergence, trajectory,
                    #   intelligence, gap resolvability, adversarial verification, eligibility)
    eligibility.py  # Clinical trial eligibility matching for Erik
    adversarial.py  # Adversarial protocol verification (contradiction search)
  run_loop.py       # 24/7 continuous entry point (LaunchAgent target)
  monitor.py        # Real-time terminal dashboard for loop progress
  audit/            # Append-only event logger
  config/           # Hot-reloadable JSON config
data/
  seed/             # Curated evidence seed (7 JSON files, 128 objects)
  erik_config.json  # Hot-reloadable runtime config (~75 keys)
logs/               # LaunchAgent log output (erik_research.log, erik_research.err)
tests/              # 1000+ pytest tests mirroring scripts/ structure
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
