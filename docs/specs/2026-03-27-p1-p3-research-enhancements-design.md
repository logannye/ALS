# P1-P3 Research Enhancements — Design Specification

**Date:** 2026-03-27
**Author:** Logan Nye + Claude
**Status:** Approved
**Approach:** Additive Integration (no pipeline restructuring, zero downtime)

---

## Overview

Seven enhancements across three priority tiers that transform Erik from a literature synthesis engine into a system capable of temporal reasoning, combination therapy design, adversarial self-verification, clinical trial matching, information-theoretic action selection, and cross-disease causal inference.

**Guiding constraint:** Erik runs 24/7 for a patient with progressive ALS. Every enhancement plugs into the existing architecture through established interfaces — new connectors extend `BaseConnector`, new action types extend `ActionType`, new pipeline stages slot into the existing 6-stage protocol generator. Each enhancement is independently activatable via hot-reload config flags.

**Implementation order (clinical urgency):**

1. P2c — Clinical trial eligibility matching
2. P1b — PRO-ACT trajectory matching
3. P1a — bioRxiv/medRxiv preprint connector
4. P2b — Adversarial protocol verification
5. P2a — Drug combination synergy modeling
6. P3a — Thompson sampling policy
7. P3b — Galen SCM integration

---

## Enhancement 1: bioRxiv/medRxiv Preprint Connector (P1a)

### Purpose

Eliminate the 6-12 month recency gap in Erik's literature awareness. ALS therapeutic research moves faster than PubMed indexing — VTx-002 preclinical data, STMN2 splice-switching results, and novel mechanism papers appear on preprint servers months before publication.

### Design

A new `BiorxivConnector` extending `BaseConnector`, mirroring the existing `PubMedConnector` pattern. Uses the bioRxiv API (`api.biorxiv.org/details/{server}/{interval}` for listing, search endpoint for keyword queries).

**Key design decisions:**

- **Evidence strength clamping:** Every `EvidenceItem` from this connector gets `strength=EvidenceStrength.emerging` (never `strong` or `moderate`), with `provenance.source_system=SourceSystem.literature` and body field `"peer_reviewed": false`. Preprints are not peer-reviewed and must not inflate intervention scores.

- **Deduplication against PubMed:** The connector stores the DOI in the evidence body. During upsert, if a PubMed-sourced evidence item with a matching DOI already exists, the preprint version is marked `status=ObjectStatus.superseded`. Uses the existing supersession mechanism.

- **Recency-weighted search:** Queries are date-filtered to the last 90 days by default (configurable). Older unpublished preprints are increasingly suspect.

### New Files

- `scripts/connectors/biorxiv.py`
- `tests/test_biorxiv_connector.py`

### Touches Existing

- `scripts/research/actions.py` — add `ActionType.SEARCH_PREPRINTS`
- `scripts/research/policy.py` — add `SEARCH_PREPRINTS` to `_ACQUISITION_ROTATION`
- `data/erik_config.json` — new keys

### New Action Type

`ActionType.SEARCH_PREPRINTS` — added to `_ACQUISITION_ROTATION` (see Appendix for final rotation). Queried roughly every 8th acquisition step.

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `biorxiv_enabled` | bool | `true` | Enable/disable preprint search |
| `biorxiv_lookback_days` | int | `90` | How far back to search |
| `biorxiv_max_per_query` | int | `15` | Max results per query |

### Tests

- Fetch parsing (XML/JSON → EvidenceItem)
- DOI deduplication against PubMed
- Strength clamping (never above `emerging`)
- Supersession logic when published version exists

---

## Enhancement 2: PRO-ACT Trajectory Matching (P1b)

### Purpose

Answer the most urgent clinical question: how fast is Erik's disease progressing relative to similar patients, and how much time remains in each intervention window? Transforms the protocol from static "best interventions now" to temporally-aware "best interventions given where Erik is heading."

### Design

Two components: a data ingestion connector and a trajectory matching module.

#### Component A — `ProactConnector` (data ingestion)

PRO-ACT contains ~13,000 ALS patient records across 38 clinical trials with longitudinal ALSFRS-R scores, demographics, labs, and survival outcomes. This is a static dataset (CSV download), not a live API.

- Load CSVs into PostgreSQL table `erik_ops.proact_trajectories` at startup.
- Schema: `patient_id, age_onset, sex, onset_region, alsfrs_r_total, alsfrs_r_subscores[4], fvc_percent, time_months, vital_status, survival_months`.
- Index on `(onset_region, age_onset)` for fast cohort matching.
- One-time idempotent ingestion. Connector's `fetch()` is a no-op after initial load.

#### Component B — `TrajectoryMatcher` (new module)

Given Erik's current profile (age 67, male, limb-onset, ALSFRS-R 43, decline rate -0.39/month, FVC 100%), the matcher:

1. **Cohort selection:** Filter PRO-ACT to patients matching onset region + age bracket (±5 years) + sex. Expected cohort: ~800-1,500 patients.

2. **Trajectory alignment:** Align each cohort patient's ALSFRS-R curve to Erik's current position using dynamic time warping (DTW) on subscores. Select top-K most similar trajectories (K=50).

3. **Survival estimation:** From matched cohort, compute Kaplan-Meier survival curves. Output: `median_months_remaining`, `p25_months`, `p75_months`.

4. **Intervention window estimation:** For each protocol layer, estimate the time window during which that intervention class is effective, based on when matched patients crossed functional thresholds:
   - Root cause suppression: effective while ALSFRS-R > 30
   - Regeneration: effective while ALSFRS-R > 35
   - Respiratory support: trigger when FVC < 70%
   - Thresholds are configurable.

### Pipeline Integration

- **Stage 1 (State Materialization):** `DiseaseStateSnapshot` gains field `trajectory_match: TrajectoryMatchResult` with survival estimate and window estimates.
- **Stage 3 (Intervention Scoring):** Scoring prompt extended with temporal context: "Erik's estimated median survival from current state is X months. Layer D window closes in approximately Y months."
- **Stage 4 (Protocol Assembly):** Window estimates adjust `DEFAULT_TIMING` — if regeneration window is closing faster than expected, move from day 21 to day 0.

### New Files

- `scripts/connectors/proact.py`
- `scripts/world_model/trajectory_matcher.py`
- `tests/test_proact_connector.py`
- `tests/test_trajectory_matcher.py`

### Touches Existing

- `scripts/world_model/state_materializer.py` — add trajectory match to snapshot
- `scripts/world_model/intervention_scorer.py` — extend scoring prompt
- `scripts/world_model/protocol_assembler.py` — adjust timing from windows
- `scripts/ontology/state.py` — add `TrajectoryMatchResult` model
- `scripts/research/actions.py` — add `ActionType.UPDATE_TRAJECTORY`
- `data/erik_config.json` — new keys
- DB schema — new `erik_ops.proact_trajectories` table

### New Action Type

`ActionType.UPDATE_TRAJECTORY` — event-driven (triggered when new clinical observations arrive or every 100 steps), not part of the regular acquisition rotation.

### Dependencies

- `scipy` (DTW computation)
- `lifelines` (Kaplan-Meier — lightweight, pure Python)

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `proact_enabled` | bool | `true` | Enable trajectory matching |
| `proact_data_dir` | str | `null` | Path to PRO-ACT CSV files |
| `proact_cohort_age_window` | int | `5` | ±years for cohort filtering |
| `proact_top_k_matches` | int | `50` | Number of matched trajectories |
| `trajectory_refresh_interval_steps` | int | `100` | Auto-refresh cadence |

### Tests

- Cohort filtering (age, sex, onset region)
- DTW alignment on synthetic trajectories
- Survival estimation on mock dataset
- Window threshold computation
- Idempotent CSV ingestion

---

## Enhancement 3: Drug Combination Synergy Modeling (P2a)

### Purpose

Replace independent intervention scoring + assembly with explicit combination reasoning. The current pipeline misses synergy (more-than-additive pairs), antagonism (pathway-level undermining), and redundancy (diminishing returns from same-mechanism pairs).

### Design

A new sub-stage between Stage 3 (Intervention Scoring) and Stage 4 (Protocol Assembly): **Stage 3B — Combination Analysis.**

Three analysis layers in order of reliability:

#### Layer 1 — Pathway Overlap Analysis (deterministic, no LLM)

Using existing `CausalChain` objects, compute pairwise pathway overlap between all top-3-per-layer interventions. Two interventions share a pathway if their chains contain overlapping intermediate nodes.

- **Overlap > 60%:** Flag as redundant. Assembler prefers diversifying.
- **Opposing terminal effects on shared node:** Flag as potentially antagonistic.
- **Convergent non-overlapping chains, compatible effects:** Flag as synergy candidate.

Graph operation on existing data structures. No external data needed.

#### Layer 2 — Literature-Grounded Interaction Evidence

For flagged pairs (or all top-1 × top-1 cross-layer pairs), search PubMed:
- Query: `"{drug_a}" AND "{drug_b}" AND (ALS OR "motor neuron" OR neurodegeneration) AND (combination OR synergy OR interaction)`
- Results stored as `EvidenceItem` with body field `"combination_pair": [drug_a_id, drug_b_id]`.
- Clinical/preclinical evidence overrides computational Layer 1 assessment.

Reuses existing `PubMedConnector` — no new connector.

#### Layer 3 — LLM-Assisted Mechanistic Reasoning (9B)

For the final 5-intervention combination, structured LLM query identifying antagonism, synergy, and redundancy.

Response parsed into:

```python
class InteractionFlag(BaseModel):
    intervention_a: str
    intervention_b: str
    interaction_type: Literal["synergy", "antagonism", "redundancy"]
    mechanism: str
    confidence: float  # 0-1
    cited_evidence: list[str]

class CombinationAnalysis(BaseModel):
    flags: list[InteractionFlag]
    overall_coherence: float  # 0-1
    suggested_substitutions: list[dict]
```

### Pipeline Integration

Stage 4 receives `CombinationAnalysis` alongside `InterventionScore` list:
- Antagonism (confidence > 0.7): replace lower-scoring intervention with layer's next-best.
- Redundancy: swap one of the pair for next-best in its layer.
- Synergy: recorded but doesn't trigger substitutions.

`CureProtocolCandidate` gains field: `combination_analysis: Optional[dict]`.

### New Files

- `scripts/world_model/combination_analyzer.py`
- `tests/test_combination_analyzer.py`

### Touches Existing

- `scripts/world_model/protocol_generator.py` — call Stage 3B between scoring and assembly
- `scripts/world_model/protocol_assembler.py` — accept and apply interaction flags
- `scripts/ontology/protocol.py` — add `combination_analysis` field
- `data/erik_config.json` — new keys

### No New Action Type

Runs as part of `REGENERATE_PROTOCOL` within the existing 6-stage pipeline.

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `combination_analysis_enabled` | bool | `true` | Enable combination analysis |
| `combination_antagonism_threshold` | float | `0.7` | Confidence threshold for antagonism swap |
| `combination_overlap_threshold` | float | `0.6` | Pathway overlap threshold for redundancy |

### Tests

- Pathway overlap on mock causal chains
- Substitution logic (antagonism triggers swap, synergy doesn't)
- LLM response parser
- Assembler respects interaction flags

---

## Enhancement 4: Adversarial Protocol Verification (P2b)

### Purpose

Correct the research loop's structural confirmation bias. Gap analysis finds weak spots and fills them — this produces uniformly moderate evidence but never tests whether core assumptions are wrong. Adversarial verification actively tries to disprove the protocol's top interventions.

### Design

Two components: a new action type for the research loop, and an extension to Stage 5 (Counterfactual Check).

#### Component A — `CHALLENGE_INTERVENTION` Action Type

1. **Target selection:** Pick the intervention with highest score and least contradictory evidence. Tracked via new `ResearchState` field: `challenge_counts: dict[str, int]`.

2. **Adversarial query generation (three types):**
   - Failure: `"{drug_name}" ALS (failed OR negative OR ineffective OR discontinued)`
   - Harm: `"{drug_name}" (neurotoxicity OR adverse OR "motor neuron" harm OR contraindicated)`
   - Mechanism dispute: `"{mechanism}" ALS (disputed OR disproven OR "no effect" OR insufficient)`

3. **Result classification (9B LLM):** Each paper classified as `contradicts`, `weakens`, `irrelevant`, or `context_dependent`.

4. **Evidence storage:** Contradicting items stored with `direction=EvidenceDirection.refutes` or `mixed`, body field `"challenged_intervention": intervention_id`.

5. **Contested threshold:** 3+ `contradicts` items with no matching `supports` of equal strength → intervention flagged as `contested`. Scorer applies penalty on next regeneration.

#### Policy Integration

Inserted at cycle position 3 (validation slot): validate hypotheses if active, *else challenge an intervention if all top interventions challenged < 2 times*, else acquire evidence. No new cycle step — fits within existing 5-step structure.

#### Component B — Pipeline Stage 5 Extension

`CounterfactualResult` gains:
```python
strongest_counterargument: str = ""
counterargument_strength: str = "none"  # none, weak, moderate, strong
```

If `counterargument_strength == "strong"` for any layer, protocol output includes explicit disclosure with next-best alternative.

`CureProtocolCandidate` gains field: `contested_layers: list[str]`.

#### Reward Signal

`CHALLENGE_INTERVENTION` applies 1.5x multiplier to `evidence_gain` reward when evidence direction is `refutes` or `mixed`. Implemented as conditional in action execution, not a change to the reward function.

### New Files

- `scripts/research/adversarial.py`
- `tests/test_adversarial.py`

### Touches Existing

- `scripts/research/actions.py` — add `ActionType.CHALLENGE_INTERVENTION`
- `scripts/research/policy.py` — insert challenge logic at position 3
- `scripts/research/state.py` — add `challenge_counts` field
- `scripts/world_model/counterfactual_check.py` — extend prompt and result model
- `scripts/ontology/protocol.py` — add `contested_layers` field
- `data/erik_config.json` — new keys

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `adversarial_verification_enabled` | bool | `true` | Enable adversarial challenges |
| `adversarial_max_challenges_per_intervention` | int | `3` | Max challenges per intervention |
| `adversarial_contested_threshold` | int | `3` | Contradicts items to flag contested |
| `adversarial_reward_multiplier` | float | `1.5` | Reward boost for contradictory evidence |

### Tests

- Adversarial query generation
- Classification parser
- Contested threshold logic
- Policy inserts challenges at position 3 when hypotheses empty
- Reward multiplier application
- Counterfactual prompt extension

---

## Enhancement 5: Clinical Trial Eligibility Matching (P2c)

### Purpose

Compute Erik's precise eligibility for every active ALS trial. If Erik qualifies for PREVAiLS or PIONEER-ALS, that matters more than any computational refinement. The system already searches ClinicalTrials.gov but doesn't evaluate whether Erik actually qualifies.

### Design

An extension to the existing `ClinicalTrialsConnector` plus a new eligibility computation module.

#### Component A — `EligibilityMatcher`

Takes a structured trial record (ClinicalTrials.gov API v2 `protocolSection.eligibilityModule`) and Erik's profile, returns:

```python
class EligibilityVerdict(BaseModel):
    trial_nct_id: str
    trial_title: str
    trial_phase: str
    intervention_name: str
    eligible: Literal["yes", "no", "likely", "pending_data"]
    blocking_criteria: list[str]
    pending_criteria: list[str]
    matching_criteria: list[str]
    protocol_alignment: float  # 0-1
    urgency: str  # enrolling_now, not_yet_recruiting, completed
    sites_near_erik: list[str]
```

**Two-tier eligibility computation (rule-based, no LLM):**

1. **Structured fields (deterministic):** Check age, sex, healthy volunteer status against `ERIK_PROFILE`.

2. **Free-text criteria (pattern-matched):** Regex extraction for:
   - ALSFRS-R thresholds → compare against Erik's 43
   - FVC thresholds → compare against Erik's 100%
   - Disease duration → compare against Erik's 14 months
   - Riluzole status → Erik is on riluzole
   - Genetic requirements → `pending_data` (genetics pending)
   - Exclusion comorbidities → check against Erik's comorbidities

   Unmatched criteria classified as `pending_data`, not silently dropped. System errs toward `likely` over `no` — false negatives (missing eligible trials) are worse than false positives.

3. **Protocol alignment:** Trial intervention compared against protocol's top interventions. High alignment confirms protocol direction; low alignment flags potential novel approaches.

#### Component B — Research Loop Integration

`SEARCH_TRIALS` action is extended: every trial returned also gets an eligibility verdict. Verdicts persisted as `"EligibilityVerdict"` objects in `erik_core.objects`.

Prominent log line for eligible trials:
```
[ERIK-TRIAL] ★ ELIGIBLE: NCT06012345 — PREVAiLS Phase 3 (pridopidine)
  Phase: 3 | Status: enrolling_now | Alignment: 0.87
  Pending: genetic_status | Site: Cleveland Clinic, OH
```

#### Component C — Trial Watchlist

New table `erik_ops.trial_watchlist`:
- Columns: `nct_id, title, eligible_status, last_checked, enrollment_status, protocol_alignment, sites, reviewed`
- Updated on every `SEARCH_TRIALS` action
- Detects enrollment status changes
- Tracks physician review status (`reviewed: bool`, manual)
- Surfaced through `scripts/monitor.py`

### New Files

- `scripts/research/eligibility.py`
- `tests/test_eligibility.py`

### Touches Existing

- `scripts/connectors/clinical_trials.py` — call eligibility matcher on results
- `scripts/db/` — migration for `erik_ops.trial_watchlist` table
- `data/erik_config.json` — new keys

### No New Action Type

Enhances existing `SEARCH_TRIALS`. Policy unchanged.

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `trial_eligibility_enabled` | bool | `true` | Enable eligibility matching |
| `trial_geographic_region` | str | `"Ohio"` | For site proximity matching |
| `trial_alsfrs_r_current` | int | `43` | Updated when new assessments arrive |
| `trial_fvc_current` | int | `100` | Updated when new assessments arrive |

### Tests

- Structured field matching (age, sex)
- Regex extraction for ALSFRS-R/FVC/duration thresholds
- `pending_data` classification for genetic criteria
- Protocol alignment scoring
- Watchlist upsert idempotency
- `ERIK_PROFILE` propagation

---

## Enhancement 6: Thompson Sampling Policy (P3a)

### Purpose

Replace the fixed 5-step cycle with information-theoretic action selection. The logs show long runs of `evidence=0 | reward=0.00` where the system queries exhausted sources. Thompson sampling selects the action with highest expected yield, eliminating wasted queries.

### Design

A replacement for the policy in `policy.py`, activated behind a config flag. Existing fixed-cycle policy preserved as fallback.

#### Core Mechanism — Thompson Sampling with Beta Posteriors

For each action type, maintain `Beta(α, β)` representing belief about probability of nonzero evidence yield:
- Success (`evidence > 0`): `α += 1`
- Failure (`evidence == 0`): `β += 1`
- Selection: sample from each action's Beta, select highest sample.

Posteriors stored in `ResearchState` as `action_posteriors: dict[str, tuple[float, float]]`, initialized to `(1.0, 1.0)` (uniform prior). Persisted to PostgreSQL.

#### Three Modifications Beyond Vanilla Thompson Sampling

1. **Context-sensitive priors:** Posterior is per `"{action_type}:{context_key}"` — e.g., `"search_pubmed:root_cause_suppression"`, `"query_galen_kg:TARDBP"`. Prevents exhaustion of one topic from penalizing a different topic for the same action type.

2. **Decay (every 50 steps):** `α *= 0.95, β *= 0.95`, floor `(1.0, 1.0)`. Prevents posteriors from concentrating so strongly that the system stops exploring actions whose yield profile has changed.

3. **Diversity floor:** Every action type exercised at least once per 30 steps. If starved, forced on next step regardless of posterior. Prevents permanent starvation of low-yield but important actions.

#### Policy Integration

Same function signature as existing `select_action()`:

```python
def select_action_thompson(state, regen_threshold, target_depth) -> (ActionType, dict)
```

Two correctness constraints preserved from fixed-cycle:
- Protocol regeneration preempts when `new_evidence_since_regen >= regen_threshold`
- Hypothesis validation capped at 2 consecutive

The module dispatches based on config:
```python
def select_action(state, **kwargs):
    if config.get("thompson_policy_enabled", False):
        return select_action_thompson(state, **kwargs)
    return select_action_cycle(state, **kwargs)
```

Parameter generation (which gene, which query) delegated to existing helper functions after Thompson selects the action type.

### New Files

None — extends `scripts/research/policy.py`

### Touches Existing

- `scripts/research/policy.py` — add `select_action_thompson`, rename existing to `select_action_cycle`
- `scripts/research/state.py` — add `action_posteriors` field
- `data/erik_config.json` — new keys
- `tests/test_research_policy.py` — new test cases

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `thompson_policy_enabled` | bool | `false` | Enable Thompson sampling (starts off) |
| `thompson_decay_interval` | int | `50` | Steps between decay applications |
| `thompson_decay_rate` | float | `0.95` | Multiplicative decay factor |
| `thompson_diversity_floor` | int | `30` | Max steps without exercising any action type |

### Tests

- Posterior updates (success → α, failure → β)
- Sampling produces expected exploration/exploitation balance
- Decay mechanics
- Diversity floor forcing
- Regeneration still preempts
- Fallback config flag
- State serialization of posteriors across restart

---

## Enhancement 7: Galen SCM Integration (P3b)

### Purpose

Upgrade the Galen bridge from relationship lookups to causal reasoning queries. Galen's SCM has 640K L3 edges and mature do-calculus capabilities. Erik should query the causal *consequences* of intervention on shared pathways (mTOR, autophagy, HDAC, oxidative stress), not just entity relationships.

### Design

A new connector querying Galen's causal graph via direct SQL against `galen_kg` PostgreSQL (same pattern as existing `GalenKGConnector`). No changes to Galen required.

#### Communication Approach

Direct SQL against Galen's causal tables. Erik cannot import Galen's Python modules (separate conda environment), but the causal graph data is in PostgreSQL. Erik replicates essential causal reasoning via SQL graph traversal.

#### What Erik Extracts

1. **L3 causal edges for shared targets:** `galen_kg.relationships` filtered to `pch_layer >= 2` where entities overlap ALS/cancer biology.

2. **Causal chain traversal:** Recursive CTE walking the L3 graph 3 hops downstream from a drug target to find predicted causal consequences.

3. **Pathway strength:** For a given pathway, count L2+ and L3 edges. A pathway with 347 L3 edges is far better understood than one with 5. Erik weights cross-disease confidence accordingly.

#### New Module: `scripts/connectors/galen_scm.py`

Separate from existing `galen_kg.py`. Three methods:

- `query_causal_downstream(target_gene, max_depth=3) -> list[CausalEdge]`
- `query_causal_upstream(effect, max_depth=3) -> list[CausalEdge]`
- `query_pathway_strength(pathway_name) -> PathwayStrength`

Returns lightweight dataclasses consumed by scoring and combination analysis — reasoning inputs, not stored as evidence:

```python
@dataclass
class CausalEdge:
    source: str
    target: str
    relationship_type: str
    pch_layer: int
    confidence: float

@dataclass
class PathwayStrength:
    pathway: str
    l2_edges: int
    l3_edges: int
    total_entities: int
    confidence: float  # l3 / (l2 + l3)
```

### Pipeline Integration

- **Stage 3 (Intervention Scoring):** Scorer receives `causal_context` for each intervention with shared-pathway targets: downstream chain + pathway strength. Prompt extended with cross-disease evidence.
- **Stage 3B (Combination Analysis):** Pathway overlap detection strengthened by Galen's causal edges — interventions that appear independent in ALS-only chains may converge in Galen's deeper graph.

### New Action Type

`ActionType.QUERY_GALEN_SCM` — replaces the second `SEARCH_PUBMED` slot in `_ACQUISITION_ROTATION` (see Appendix for final rotation). Queried roughly every 8th acquisition step.

### Connection Management

Same pattern as fixed `galen_kg.py`: raw `psycopg.connect()` with UNION optimization, `statement_timeout=30000`, `work_mem=16MB`. Connection opened per query, closed in `finally`.

### New Files

- `scripts/connectors/galen_scm.py`
- `tests/test_galen_scm_connector.py`

### Touches Existing

- `scripts/research/actions.py` — add `ActionType.QUERY_GALEN_SCM`
- `scripts/research/policy.py` — add to acquisition rotation
- `scripts/world_model/intervention_scorer.py` — consume causal context
- `data/erik_config.json` — new keys

### Config Keys (hot-reloadable)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `galen_scm_enabled` | bool | `true` | Enable SCM queries |
| `galen_scm_min_pch_layer` | int | `2` | Minimum PCH layer for edges |
| `galen_scm_max_chain_depth` | int | `3` | Max recursive CTE depth |
| `galen_scm_database` | str | `"galen_kg"` | Database name |

### Tests

- Recursive CTE on mock graph
- Pathway strength computation
- L1-only edge exclusion when `min_pch_layer=2`
- Connection cleanup
- Dataclass serialization
- Integration with scoring prompt construction

---

## Cross-Cutting Concerns

### State Machine Changes

`ResearchState` gains three new fields:
- `challenge_counts: dict[str, int]` — adversarial challenge tracking
- `action_posteriors: dict[str, tuple[float, float]]` — Thompson sampling posteriors
- `last_action_per_type: dict[str, int]` — for diversity floor enforcement

All default to empty dicts, so existing serialized state loads without migration.

### Database Schema Changes

Two new tables:
- `erik_ops.proact_trajectories` — PRO-ACT patient data (one-time load)
- `erik_ops.trial_watchlist` — clinical trial eligibility tracking

Both created via the existing migration mechanism in `scripts/db/`.

### Ontology Model Changes

- `DiseaseStateSnapshot` — add `trajectory_match: Optional[dict]`
- `CureProtocolCandidate` — add `combination_analysis: Optional[dict]`, `contested_layers: list[str]`
- `CounterfactualResult` — add `strongest_counterargument: str`, `counterargument_strength: str`

All new fields are `Optional` or have defaults, so existing objects remain valid.

### Dependencies

Two new Python packages:
- `scipy` — DTW computation for trajectory matching
- `lifelines` — Kaplan-Meier survival analysis

Both lightweight, pure Python / NumPy. No GPU dependencies.

### Config Summary

27 new hot-reloadable config keys across 7 enhancements. All have sensible defaults. Each enhancement can be independently disabled by setting its `_enabled` flag to `false`.

### Test Summary

7 new test files mirroring the 7 enhancements, following the existing `tests/` structure. TDD approach as specified in CLAUDE.md.

---

## File Impact Summary

| # | Enhancement | New Files | Modifies |
|---|------------|-----------|----------|
| P1a | bioRxiv | `connectors/biorxiv.py`, test | `actions.py`, `policy.py`, config |
| P1b | PRO-ACT | `connectors/proact.py`, `world_model/trajectory_matcher.py`, tests | `state_materializer.py`, `intervention_scorer.py`, `protocol_assembler.py`, `state.py` (ontology), config, DB |
| P2a | Combination | `world_model/combination_analyzer.py`, test | `protocol_generator.py`, `protocol_assembler.py`, `protocol.py`, config |
| P2b | Adversarial | `research/adversarial.py`, test | `actions.py`, `policy.py`, `state.py` (research), `counterfactual_check.py`, `protocol.py`, config |
| P2c | Eligibility | `research/eligibility.py`, test | `clinical_trials.py`, DB schema, config |
| P3a | Thompson | (extends `policy.py`) | `policy.py`, `state.py` (research), config, existing test |
| P3b | Galen SCM | `connectors/galen_scm.py`, test | `actions.py`, `policy.py`, `intervention_scorer.py`, config |

**Total: 8 new source files, 7 new test files, ~15 existing files modified.**

---

## Appendix: Final Acquisition Rotation

The current `_ACQUISITION_ROTATION` has 7 slots. After all enhancements:

```python
_ACQUISITION_ROTATION = [
    ActionType.SEARCH_PUBMED,           # existing
    ActionType.SEARCH_TRIALS,           # existing (now with eligibility matching)
    ActionType.QUERY_PATHWAYS,          # existing
    ActionType.QUERY_PPI_NETWORK,       # existing
    ActionType.CHECK_PHARMACOGENOMICS,  # existing
    ActionType.QUERY_GALEN_KG,          # existing
    ActionType.SEARCH_PREPRINTS,        # NEW (P1a)
    ActionType.QUERY_GALEN_SCM,         # NEW (P3b, replaces second SEARCH_PUBMED)
]
```

8 slots. Each acquisition action fires roughly every 8th acquisition step (every ~13th total step, given the 5-step cycle has 3 acquisition positions). This rotation only applies when `thompson_policy_enabled=false` (the fixed-cycle fallback). When Thompson sampling is active, it selects action types by posterior, not rotation.
