# Phase 2: World Model + Cure Protocol Generation — Design Specification

## 1. Purpose

Generate Erik Draper's first cure protocol candidate by materializing his disease state from clinical observations, inferring his most likely disease subtype, scoring all candidate interventions against evidence, assembling a multi-layer protocol, and stress-testing it with counterfactual verification.

This is the phase where Erik transitions from storing evidence to reasoning about it. The output is a concrete, actionable `CureProtocolCandidate` object with full provenance, uncertainty disclosure, and human approval gate.

## 2. Strategic Decisions

- **LLM-augmented causal reasoning**: Local Qwen LLM synthesizes mechanistic arguments from the evidence fabric. The LLM is an evidence synthesizer, not a knowledge source.
- **Evidence-grounded generation**: Every LLM claim must cite evidence item IDs from the fabric. Uncited claims are rejected. No training-data knowledge leaks.
- **Multi-pass refinement**: Single deliberative pass produces the protocol, then counterfactual verification stress-tests each layer. This architecture supports Phase 3 iterative refinement.
- **All subtypes covered**: Genetic testing pending — protocol includes conditional branches for each subtype, with sporadic TDP-43 as the working hypothesis.
- **Dual verification**: Critical claims that affect protocol ranking get a second LLM verification pass.

## 3. Pipeline Architecture

```
Stage 1: State Materialization
  Erik's 51 observations → x_t = [g_t, m_t, n_t, f_t, r_t, u_t]
  Mostly deterministic. LLM used only for reversibility estimate (r_t).

Stage 2: Subtype Inference
  State + evidence → EtiologicDriverProfile (posterior over 8 subtypes)
  LLM reasons over subtype-specific evidence with Erik's presentation.

Stage 3: Intervention Scoring
  For each of ~25 interventions: retrieve evidence → LLM synthesizes
  causal argument → score with citations → rank by relevance to Erik.

Stage 4: Protocol Assembly
  Select best intervention(s) per layer (A-E) → interaction check →
  sequence and time → assemble CureProtocolCandidate.

Stage 5: Counterfactual Verification
  For each layer: "What if we removed this?" → identifies weakest links,
  highest-uncertainty components, and most impactful missing measurements.

Stage 6: Protocol Output
  Final CureProtocolCandidate with uncertainty, provenance, approval gate.
```

Each stage produces canonical Pydantic objects stored in PostgreSQL. Every LLM call is evidence-grounded with citation-mandatory output.

## 4. Evidence-Grounded Reasoning Engine

### 4.1 Architecture

`ReasoningEngine` is the core LLM interface. It is NOT a chatbot. It is a structured evidence synthesizer with hard grounding constraints.

```python
class ReasoningEngine:
    def reason(self, prompt_template: str, evidence_items: list[dict],
               output_schema: type[BaseModel], verify_critical: bool = False) -> BaseModel:
        """Run evidence-grounded LLM reasoning.

        1. Serialize evidence items into prompt context
        2. Call LLM with structured output schema
        3. Parse and validate JSON response
        4. Validate all citations exist in input evidence set
        5. Strip uncited claims, log warnings
        6. If verify_critical: run second verification pass
        7. Return validated Pydantic model
        """
```

### 4.2 Grounding Constraints (hardcoded, non-configurable)

1. **Evidence-only context**: LLM receives ONLY evidence items from the fabric. System prompt explicitly prohibits introducing external knowledge.
2. **Citation-mandatory**: Every claim must cite `evi:*` or `int:*` IDs from the input set. This is enforced programmatically post-generation.
3. **Uncited claim rejection**: Claims without valid citations are stripped from output. A warning is logged with the rejected text for audit.
4. **Strength-aware prompting**: Evidence strength (strong/moderate/emerging/preclinical/unknown) and PCH level (L1/L2/L3) are included in the context so the LLM can weight appropriately. Prompt instructs: "A preclinical result cannot override a Phase 3 RCT finding."
5. **Structured JSON output only**: Every LLM call specifies an output schema. Raw text is never used.
6. **Dual verification**: For claims marked `verify_critical=True`, a second LLM call receives ONLY the cited evidence items and the claim, then assesses: "Is this claim fully supported by the provided evidence?" Disagreements are flagged as `contested`.

### 4.3 Citation Validation Algorithm

```python
def validate_citations(output: dict, valid_ids: set[str]) -> tuple[dict, list[str]]:
    """Validate all evidence citations in LLM output.

    Returns (cleaned_output, warnings).
    - Extracts all evi:* and int:* references from output
    - Any ID not in valid_ids → warning logged, claim containing it marked uncertain
    - Claims with zero valid citations → stripped entirely
    """
```

### 4.4 LLM Configuration

- **Model**: Qwen3.5-35B-A3B-mlx-lm-mxfp4 (standard) at `/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4`
- **Fallback**: Qwen3.5-9B-mlx-lm-4bit (fast) at `/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit`
- **Temperature**: 0.1 (near-deterministic for reproducibility)
- **Interface**: `mlx_lm.generate()` or Galen's LLM server if running (shared GPU)
- **Max tokens**: 500-2000 depending on stage
- **Timeout**: 120s per call

### 4.5 Prompt Templates

All prompts follow this structure:
```
SYSTEM: You are a structured evidence synthesizer for ALS research.
You reason ONLY from the evidence items provided below.
You MUST cite evidence item IDs (evi:* or int:*) for every claim.
You MUST NOT introduce information not present in the provided evidence.
Output ONLY valid JSON matching the schema provided.

EVIDENCE ITEMS:
{serialized evidence items as JSON array}

PATIENT STATE:
{Erik's current disease state as JSON}

TASK:
{stage-specific instruction}

OUTPUT SCHEMA:
{JSON schema for expected output}
```

## 5. Stage 1: State Materialization

### Input
- Erik's 51 observations from `build_erik_draper()`
- Existing `ALSFRSRScore`, lab results, EMG, MRI, spirometry, exam findings

### Output
- `FunctionalState` (f_t): directly from ALSFRS-R (43/48), subscores, FVC, weight
- `NMJIntegrityState` (part of n_t): from EMG (widespread denervation), exam (atrophy, weakness pattern)
- `RespiratoryReserveState` (part of n_t): from spirometry (FVC 100%), exam findings
- `UncertaintyState` (u_t): explicitly enumerate what's measured vs missing
- `ReversibilityWindowEstimate` (r_t): LLM-synthesized from disease duration, functional state, literature on reversibility
- `DiseaseStateSnapshot`: composition of all above

### Molecular State (m_t) — LLM-Inferred

Erik has no molecular assays (no transcriptomics, proteomics, or cryptic exon measurements). The molecular state `m_t` is therefore **inferred from evidence items** by the LLM, similar to `r_t`. The LLM receives:
- Erik's clinical presentation and inferred subtype
- Evidence items about TDP-43 pathology, cryptic splicing, proteostasis, stress granules, glial activation
- Population-level data (e.g., "97% of sporadic ALS has TDP-43 pathology")

And produces estimates for existing state models:
- `TDP43FunctionalState`: nuclear_function_score, cytoplasmic_pathology_probability, loss_of_function_probability
- `SplicingState`: cryptic_splicing_burden_score, stmn2_disruption_score, unc13a_disruption_score
- `GlialState`: microglial_activation_score, astrocytic_toxicity_score, inflammatory_amplification_score

These are explicitly flagged as **inferred from population data, not measured in Erik**. Each score includes a `dominant_uncertainties` annotation: `["inferred_from_literature_not_measured"]`. When Erik's genetic results or future biomarkers arrive, these estimates can be updated with measured data.

### Logic
Mostly deterministic mapping (no LLM for f_t, n_t, u_t). LLM used for m_t and r_t:

```python
def materialize_functional_state(trajectory, observations) -> FunctionalState:
    """Direct mapping from ALSFRS-R and clinical observations."""
    score = trajectory.alsfrs_r_scores[0]
    weight_obs = [o for o in observations if o.observation_kind == ObservationKind.weight_measurement]
    return FunctionalState(
        alsfrs_r_total=score.total,  # 43
        bulbar_subscore=score.bulbar_subscore,  # 12
        fine_motor_subscore=score.fine_motor_subscore,  # 11
        gross_motor_subscore=score.gross_motor_subscore,  # 8
        respiratory_subscore=score.respiratory_subscore,  # 12
        weight_kg=weight_obs[-1].value if weight_obs else None,
    )
```

LLM is used ONLY for `r_t` (reversibility):
- Input: disease duration (14 months), ALSFRS-R (43), FVC (100%), EMG findings, evidence items about reversibility windows
- Output: `ReversibilityWindowEstimate` with scored plausibilities

### Missing Measurements (explicit in u_t)
- Genetic testing (pending) — highest priority gap
- CSF biomarkers (NfL CSF, pNfH) — not collected
- Cryptic exon splicing signatures (UNC13A, STMN2) — research assay, not clinical
- TDP-43 in vivo measurement — no clinical assay exists
- Cortical excitability (TMS) — not performed
- Repeat expansion testing (C9orf72) — part of pending genetics

## 6. Stage 2: Subtype Inference

### Input
- Erik's materialized state
- Evidence items tagged with `applicable_subtypes`
- Erik's clinical presentation features

### Output
- `EtiologicDriverProfile` with posterior over 8 subtypes

### Logic
LLM receives Erik's presentation:
- Age 67, male, limb-onset (left leg), sporadic (no MND family history)
- Widespread EMG denervation + UMN signs (upgoing plantars, hyperreflexia, spasticity)
- NfL elevated (5.82, ref 0-3.65)
- No known genetic mutation (pending)
- Family history of Alzheimer's (mother) — possibly relevant for C9orf72/FTD spectrum

Plus evidence about subtype characteristics, and produces:
```json
{
  "posterior": {
    "sporadic_tdp43": 0.65,
    "c9orf72": 0.12,
    "glia_amplified": 0.08,
    "tardbp": 0.05,
    "sod1": 0.03,
    "fus": 0.02,
    "mixed": 0.03,
    "unresolved": 0.02
  },
  "reasoning": "Limb-onset sporadic presentation without family MND history... [citations]",
  "conditional_on_genetics": {
    "if_c9orf72_positive": {"c9orf72": 0.85, "sporadic_tdp43": 0.05, "...": "..."},
    "if_sod1_positive": {"sod1": 0.90, "...": "..."},
    "if_negative_panel": {"sporadic_tdp43": 0.75, "glia_amplified": 0.10, "...": "..."}
  },
  "cited_evidence": ["evi:genetic_testing_als", "evi:tdp43_disease_protein", "evi:nfl_elevated_als"]
}
```

The `conditional_on_genetics` field is critical — it pre-computes how the posterior shifts when genetics arrive, so the protocol can be immediately re-ranked.

**Post-processing**: LLM produces string keys (e.g., `"sporadic_tdp43"`). The subtype inference module maps these to `SubtypeClass` enum values before constructing the `EtiologicDriverProfile`. Invalid keys are logged and discarded.

### Abstention Logic

The planner MUST abstain (return a partial or empty protocol for a layer) when:
- No single subtype has posterior > 0.30 AND the layer is subtype-specific (Layer A) → abstain from Layer A, recommend genetic testing as next-best-action
- Zero eligible interventions exist for a layer → abstain, note gap
- All evidence for a layer is `strength=preclinical` or weaker → flag layer as "low confidence, preclinical only"
- Erik is ineligible for all candidate interventions in a layer → abstain

Abstention produces a `ProtocolLayerEntry` with `intervention_refs=[]` and `notes="ABSTENTION: {reason}"`.

## 7. Stage 3: Intervention Scoring

### Input
- Erik's state + subtype posterior
- All ~25 interventions + their associated evidence items
- ALS target definitions

### Output
- `InterventionScore` for each intervention

### Schema
```python
class InterventionScore(BaseModel):
    intervention_id: str
    intervention_name: str
    protocol_layer: str
    relevance_score: float  # 0-1, higher = more relevant for Erik
    mechanism_argument: str  # Evidence-cited reasoning
    evidence_strength: str  # overall strength of supporting evidence
    erik_eligible: bool | str  # true, false, or "pending_genetics"
    key_uncertainties: list[str]
    cited_evidence: list[str]  # must all be valid evi:* or int:* IDs
    contested_claims: list[str]  # claims that failed verification
```

### Logic
For each intervention:
1. Retrieve all evidence items where `body.intervention_ref` matches or `body.mechanism_target` matches the intervention's targets
2. Build prompt with intervention details, evidence items, Erik's state, and subtype
3. LLM produces `InterventionScore` with mandatory citations
4. Validate citations against input evidence IDs
5. For top-5 scoring interventions: run verification pass on mechanism_argument

### Scoring Criteria (embedded in prompt)
The LLM is instructed to score based on:
- **Mechanism relevance**: Does this intervention target an active disease program in Erik's inferred subtype?
- **Evidence quality**: Strong (RCT) > Moderate (Phase 2/observational) > Emerging (case series) > Preclinical
- **Erik eligibility**: Is Erik eligible (age, comorbidities, medications)?
- **Safety profile**: Known risks relative to Erik's comorbidities
- **Feasibility**: Route of administration, availability, regulatory status
- **Timing sensitivity**: How urgent is this intervention given Erik's reversibility window?

## 8. Stage 4: Protocol Assembly

### Input
- Ranked intervention scores
- Drug interaction data (from DrugBank connector)
- Erik's medication list

### Output
- `CureProtocolCandidate` with 5 layers

### Logic

1. **Per-layer selection**: For each protocol layer, select the highest-scoring eligible intervention(s). Multiple interventions per layer are allowed if they target different mechanisms.

2. **Interaction check**: For all selected drugs, query DrugBank interactions. Flag any concerning interactions. The LLM reasons about whether interactions are clinically significant given Erik's profile.

3. **Sequencing and timing**: The LLM proposes start_offset_days for each layer based on:
   - Urgency (root-cause suppression before regeneration)
   - Interaction windows (don't start two hepatotoxic drugs simultaneously)
   - Availability (approved drugs immediately, trial enrollment takes weeks)

4. **Conditional branches**: Since genetics are pending, the protocol includes conditional arms:
   - Default arm (sporadic TDP-43 assumed)
   - SOD1 arm (if genetics positive): add tofersen to Layer A
   - C9orf72 arm: adjust Layer A strategy
   - FUS arm: add jacifusen consideration

### Drug Interaction Check

Uses the existing `DrugBankConnector.fetch_drug_interactions(drugbank_ids)` method from `scripts/connectors/drugbank.py`. If DrugBank XML is not available, falls back to a basic interaction check using the `known_risks` and `contraindications` fields on `Intervention` objects in the evidence store.

### Trajectory Forecasting (Deferred to Phase 3)

The technical spec (Section 12.4) requires trajectory forecasting. Phase 2 produces a **static protocol** based on current evidence and Erik's current state. Dynamic trajectory simulation (predicting Erik's disease course under the proposed protocol over 6-12 months) is deferred to Phase 3, which adds the RL loop and temporal modeling. The Phase 2 protocol's `expected_state_shift_summary` provides qualitative expected outcomes, not quantitative trajectory forecasts.

### Layer Selection Priority
- Layer A: Highest-scoring root-cause intervention for dominant subtype
- Layer B: Highest-scoring pathology reversal (pridopidine is likely leader for sporadic)
- Layer C: Riluzole (already taking) + highest-scoring additional stabilizer
- Layer D: Best available regeneration strategy (even if preclinical)
- Layer E: Standard monitoring protocol (NfL, ALSFRS-R, FVC, respiratory)

## 9. Stage 5: Counterfactual Verification

### Input
- Assembled protocol
- Evidence items for each selected intervention

### Output
- Counterfactual analysis per layer
- Weakest-link identification
- Missing-measurement recommendations

### Logic
For each protocol layer:
1. Prompt: "If Layer X were removed from this protocol, what would be the expected impact on Erik's disease trajectory? Cite evidence."
2. LLM reasons about the causal necessity of each layer
3. Identifies: load-bearing layers (removal would significantly worsen prognosis) vs marginal layers (removal has uncertain impact)
4. Produces: `next_best_measurement` — what single measurement would most reduce uncertainty in this layer

### Output Schema
```python
class CounterfactualResult(BaseModel):
    layer: str
    removal_impact: str  # "high", "moderate", "low", "uncertain"
    reasoning: str  # Evidence-cited
    is_load_bearing: bool
    weakest_evidence: str  # Which cited evidence is most uncertain
    next_best_measurement: str  # What measurement would most reduce uncertainty
    cited_evidence: list[str]
```

## 10. Stage 6: Protocol Output

### Final Assembly
Combines all stages into one `CureProtocolCandidate`:

```python
CureProtocolCandidate(
    id="proto:erik_draper_v1",
    subject_ref="traj:draper_001",
    objective="maximize_durable_disease_arrest_and_functional_recovery",
    assumed_active_programs=["sporadic_tdp43_loss_of_function", "proteostasis_stress", "excitotoxicity"],
    layers=[...],  # 5 ProtocolLayerEntry objects
    expected_state_shift_summary={...},
    dominant_failure_modes=[...],
    approval_state=ApprovalState.pending,
    evidence_bundle_refs=[...],
    uncertainty_ref="unc:proto_erik_v1",
    body={
        "subtype_posterior": {...},
        "conditional_genetics_arms": {...},
        "counterfactual_analysis": [...],
        "intervention_scores": [...],
        "total_evidence_items_cited": N,
        "grounding_violations_caught": M,
        "verification_contested_claims": [...],
    }
)
```

### Human Approval Gate
The protocol is stored with `approval_state=pending`. It MUST be reviewed by a clinician before any action is taken. The output includes:
- Plain-language summary of the protocol rationale
- Explicit statement of dominant uncertainties
- List of missing measurements that would improve confidence
- All evidence citations traceable to source

## 11. File Structure

```
scripts/
  world_model/
    __init__.py
    reasoning_engine.py      # Evidence-grounded LLM with citation validation
    state_materializer.py    # Stage 1: observations → latent state
    subtype_inference.py     # Stage 2: state + evidence → subtype posterior
    intervention_scorer.py   # Stage 3: score each intervention for Erik
    protocol_assembler.py    # Stage 4: select + sequence + interaction check
    counterfactual_check.py  # Stage 5: stress-test each layer
    protocol_generator.py    # Stage 6: orchestrate full pipeline
    prompts/
      __init__.py
      templates.py           # All LLM prompt templates
  llm/
    __init__.py
    inference.py             # MLX LLM inference wrapper (generate, parse JSON)
tests/
  test_reasoning_engine.py
  test_state_materializer.py
  test_subtype_inference.py
  test_intervention_scorer.py
  test_protocol_assembler.py
  test_counterfactual_check.py
  test_protocol_generator.py
  test_llm_inference.py
```

## 12. LLM Inference Module

Since this is the first module in Erik that calls an LLM, we need a thin wrapper:

```python
class LLMInference:
    """Thin wrapper for local MLX LLM inference."""
    def __init__(self, model_path: str, max_tokens: int = 1000, temperature: float = 0.1): ...
    def generate(self, prompt: str) -> str: ...
    def generate_json(self, prompt: str, schema: type[BaseModel]) -> dict: ...
```

This wraps `mlx_lm` (already installed for Galen) or can call Galen's LLM server via HTTP if it's running. The reasoning engine uses this interface — it never calls the LLM directly.

## 13. Prerequisites

**Install `mlx-lm`** (LLM inference engine, same as Galen uses):
```bash
conda run -n erik-core pip install mlx-lm
```

**Add `query_by_intervention_ref()` to EvidenceStore**: Stage 3 needs to retrieve evidence items by `body.intervention_ref`. Add a new method to `scripts/evidence/evidence_store.py`:
```python
def query_by_intervention_ref(self, intervention_id: str) -> list[dict]:
    sql = """SELECT id, type, status, body FROM erik_core.objects
             WHERE type = 'EvidenceItem' AND status = 'active'
             AND body->>'intervention_ref' = %s"""
    return self._run_query(sql, (intervention_id,))
```

## 14. Config Additions

Add to `data/erik_config.json`:
```json
{
  "llm_model_path": "/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4",
  "llm_fallback_model_path": "/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit",
  "llm_temperature": 0.1,
  "llm_max_tokens_default": 1000,
  "llm_timeout_s": 120,
  "reasoning_verify_critical": true,
  "reasoning_strip_uncited": true
}
```

## 15. Testing Strategy

**Unit tests (no LLM, no DB):**
- State materializer: fixture observations → verify FunctionalState fields
- Citation validator: test with valid/invalid/mixed citations
- Protocol assembler: mock intervention scores → verify layer selection logic
- Counterfactual structure: verify output schema
- Prompt template rendering: verify evidence items serialize correctly

**Integration tests (with LLM, marked @pytest.mark.llm):**
- Reasoning engine: real LLM call with 3-5 evidence items → verify citations valid
- Full pipeline: run all 6 stages on Erik's real data → verify protocol produced
- Grounding test: intentionally provide empty evidence set → verify LLM abstains

**Grounding-specific tests:**
- Uncited claims are stripped (inject LLM response with fake citations)
- Hallucinated evidence IDs are caught
- Verification pass catches contradictions
- Output with zero valid citations returns abstention

## 16. Success Criteria

Phase 2 is complete when:
1. Erik's disease state is materialized from his 51 observations into all 6 latent factors
2. Subtype posterior is computed (sporadic TDP-43 dominant, with conditional genetics arms)
3. All ~25 interventions are scored with evidence-grounded, citation-validated reasoning
4. A `CureProtocolCandidate` is generated across all 5 layers with sequencing and timing
5. Drug interactions are checked for the selected combination
6. Counterfactual verification identifies weakest links and missing measurements
7. Every claim in the protocol cites evidence from the fabric (zero uncited claims)
8. Dual verification is run on critical claims (top-5 intervention arguments)
9. Protocol includes explicit uncertainty disclosure and human approval gate
10. Full provenance trail stored in PostgreSQL
11. All unit tests pass without LLM; integration tests pass with LLM
