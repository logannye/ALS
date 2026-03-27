# Phase 3: Autonomous Research Loop — Design Specification

## 1. Purpose

Phase 2 produces a single-pass cure protocol. Phase 3 makes Erik *intelligent* — an autonomous research loop that deepens causal understanding of ALS mechanisms specific to Erik Draper, systematically expands the evidence fabric, and iteratively refines the cure protocol until it converges on the optimal therapeutic strategy.

The output is not "a" protocol — it is **the best protocol the system can construct given all available evidence**, refined over hundreds of research cycles, with every claim traced to evidence and every uncertainty explicitly bounded.

## 2. Core Philosophy

Erik is not exploring a vast landscape like Galen. Erik has ONE patient, ONE disease, and a ticking clock. Every computational cycle must advance toward a cure for Erik Draper.

**Convergence, not exploration.** The loop should converge on optimal interventions, not wander. Exploration serves convergence — we explore only where uncertainty blocks the best protocol.

**Causal depth over breadth.** A deep understanding of WHY pridopidine might work for Erik (sigma-1R → ER-mitochondria calcium → TDP-43 proteostasis → motor neuron survival) is more valuable than knowing about 100 drugs at surface level.

**Evidence-grounded always.** Every hypothesis, every mechanism link, every protocol decision traces back to citable evidence. The LLM is a synthesizer, not an oracle.

## 3. Architecture Overview

```
                    ┌──────────────────────────────────┐
                    │     Research Loop (main loop)     │
                    │  ┌────────────────────────────┐   │
                    │  │  1. Assess Current State    │   │
                    │  │     - Protocol quality      │   │
                    │  │     - Evidence gaps          │   │
                    │  │     - Uncertainty map        │   │
                    │  └─────────┬──────────────────┘   │
                    │            │                       │
                    │  ┌─────────▼──────────────────┐   │
                    │  │  2. Select Research Action  │   │
                    │  │     - Hypothesis-driven     │   │
                    │  │     - Gap-directed          │   │
                    │  │     - Exploration budget     │   │
                    │  └─────────┬──────────────────┘   │
                    │            │                       │
                    │  ┌─────────▼──────────────────┐   │
                    │  │  3. Execute Action          │   │
                    │  │     - Query connectors      │   │
                    │  │     - Generate hypotheses    │   │
                    │  │     - Validate mechanisms    │   │
                    │  │     - Deepen causal chains   │   │
                    │  └─────────┬──────────────────┘   │
                    │            │                       │
                    │  ┌─────────▼──────────────────┐   │
                    │  │  4. Integrate & Learn       │   │
                    │  │     - Upsert new evidence   │   │
                    │  │     - Update causal graph   │   │
                    │  │     - Compute reward        │   │
                    │  │     - Update action values  │   │
                    │  └─────────┬──────────────────┘   │
                    │            │                       │
                    │  ┌─────────▼──────────────────┐   │
                    │  │  5. Regenerate Protocol     │   │
                    │  │     (every N steps or on    │   │
                    │  │      significant evidence)  │   │
                    │  └────────────────────────────┘   │
                    └──────────────────────────────────┘
```

Single Python process. No daemon threads (lesson from Galen: GIL contention kills throughput on M4). Sequential execution. Hot-reloadable config.

## 4. Strategic Decisions

### 4.1 No Daemon Architecture
Galen's v52-v88 history is a cautionary tale: 9+ daemon threads created GIL contention, MPS stalls, and lock storms. Erik runs ONE loop, ONE thread, sequential actions. Simpler, faster, debuggable.

### 4.2 Hypothesis-Driven Search, Not Random Exploration
Unlike Galen's broad Thompson sampling, Erik generates specific mechanistic hypotheses about Erik Draper's disease, then systematically searches for evidence. Each hypothesis is a `MechanismHypothesis` object stored in PostgreSQL.

### 4.3 Protocol Convergence as Primary Metric
The protocol is regenerated periodically. The system tracks protocol stability — when the top intervention selections and scores stop changing across cycles, the system has converged. This is the primary termination signal.

### 4.4 Causal Chain Depth
For each intervention in the protocol, the system builds a full causal chain from mechanism → target → pathway → cell biology → patient phenotype. Deeper chains with more evidence links produce higher-confidence recommendations.

### 4.5 LLM Budget Management
The 35B model uses ~17GB. Galen also uses it. To avoid OOM:
- Use the 9B fallback model for routine hypothesis generation
- Reserve the 35B model for critical reasoning (protocol generation, verification)
- Config-switchable: `llm_research_model` vs `llm_protocol_model`
- Each LLM call has a token budget and timeout

## 5. Research Actions (10 types)

### 5.1 Evidence Acquisition Actions

| Action | Description | Connector | Expected Yield |
|--------|-------------|-----------|----------------|
| `SEARCH_PUBMED` | Query PubMed for evidence on a specific mechanism/target/intervention | PubMed | New EvidenceItems |
| `SEARCH_TRIALS` | Find active clinical trials relevant to Erik | ClinicalTrials.gov | New Interventions + eligibility data |
| `QUERY_CHEMBL` | Get compound bioactivity data for a target | ChEMBL (local) | Quantitative binding/efficacy data |
| `QUERY_OPENTARGETS` | Get target-disease association scores | OpenTargets | Target validation strength |
| `CHECK_INTERACTIONS` | Drug-drug interaction safety check | DrugBank | Interaction flags for combinations |

### 5.2 Reasoning Actions

| Action | Description | Uses LLM | Expected Yield |
|--------|-------------|----------|----------------|
| `GENERATE_HYPOTHESIS` | Formulate a testable mechanistic hypothesis about Erik's disease | Yes (9B) | MechanismHypothesis object |
| `DEEPEN_CAUSAL_CHAIN` | For a protocol intervention, extend the causal reasoning depth | Yes (9B) | Extended mechanism chain with citations |
| `VALIDATE_HYPOTHESIS` | Search for evidence supporting/refuting a hypothesis | Yes (9B) + connectors | Updated hypothesis support direction |
| `SCORE_NEW_EVIDENCE` | Re-score an intervention given newly acquired evidence | Yes (9B) | Updated InterventionScore |
| `REGENERATE_PROTOCOL` | Re-run the full Phase 2 pipeline with current evidence fabric | Yes (35B) | New CureProtocolCandidate |

### 5.3 Action Selection Policy

**Uncertainty-directed with hypothesis guidance:**

1. After each protocol generation, identify the **top 3 uncertainty sources** in the current protocol (from `counterfactual_analysis` and `key_uncertainties`).
2. Generate hypotheses targeting those uncertainties.
3. Select actions that would validate or refute those hypotheses.
4. If no uncertainty-directed action is available, fall back to systematic evidence expansion (SEARCH_PUBMED by protocol layer rotation).

**Policy structure:**
```python
def select_action(state: ResearchState) -> ResearchAction:
    # 1. If pending hypotheses need validation → VALIDATE_HYPOTHESIS
    # 2. If causal chain is shallow for top intervention → DEEPEN_CAUSAL_CHAIN
    # 3. If high-uncertainty layer has sparse evidence → SEARCH_PUBMED/TRIALS
    # 4. If new evidence accumulated since last protocol → REGENERATE_PROTOCOL
    # 5. If no targeted action → systematic expansion by layer rotation
    # 6. If protocol has stabilized → GENERATE_HYPOTHESIS on remaining gaps
```

**Action value tracking:**
Simple exponential moving average of reward per action type. No neural value function needed — with 10 action types and clear reward signals, a running average suffices. This avoids the V(s) complexity that consumed months of Galen development.

## 6. Reward Signal (8 components)

| Component | Weight | Description |
|-----------|--------|-------------|
| `evidence_gain` | 3.0 | New unique evidence items integrated (diminishing returns per item) |
| `uncertainty_reduction` | 4.0 | Decrease in protocol uncertainty score (highest weight — this is what matters) |
| `protocol_improvement` | 3.5 | Improvement in top intervention scores after re-scoring |
| `hypothesis_resolution` | 2.5 | Hypothesis moved from "insufficient" to "supports"/"refutes" |
| `causal_depth` | 2.0 | Extension of causal chain length for protocol interventions |
| `interaction_safety` | 2.0 | Successful drug interaction check (no new contraindications) |
| `erik_eligibility` | 1.5 | Evidence that Erik is eligible for a scored intervention/trial |
| `convergence_bonus` | 1.0 | Bonus when protocol top-3 interventions remain stable across cycles |

**Reward formula:** Weighted sum, normalized. No differential TD — just immediate reward for each action. The value function is a running average per action type, not a neural network.

## 7. Research State

```python
class ResearchState:
    """Point-in-time snapshot of the research loop's knowledge."""

    # Current protocol
    current_protocol: CureProtocolCandidate
    protocol_version: int
    protocol_stable_cycles: int  # Consecutive cycles with no top-3 change

    # Evidence fabric metrics
    total_evidence_items: int
    evidence_by_layer: dict[str, int]  # Items per protocol layer
    evidence_by_strength: dict[str, int]  # Items per strength level

    # Hypothesis tracking
    active_hypotheses: list[MechanismHypothesis]
    resolved_hypotheses: int

    # Causal depth
    causal_chains: dict[str, int]  # intervention_id → chain depth

    # Uncertainty map
    top_uncertainties: list[str]
    missing_measurements: list[str]

    # Action history
    step_count: int
    action_values: dict[str, float]  # EMA reward per action type
    action_counts: dict[str, int]
    last_action: str
    last_reward: float

    # Convergence
    converged: bool
```

## 8. Causal Chain Model

For each intervention in the protocol, maintain a **causal chain** — a directed sequence of mechanism steps from intervention to patient outcome, each grounded in evidence.

```
Intervention: Pridopidine (sigma-1R agonist)
Chain:
  pridopidine → sigma-1R activation [evi:sigma1r_pridopidine_binding]
    → ER-mitochondria calcium homeostasis [evi:sigma1r_er_mito_calcium]
      → reduced ER stress [evi:sigma1r_er_stress_reduction]
        → improved TDP-43 proteostasis [evi:tdp43_proteostasis_er_stress]
          → reduced cryptic splicing [evi:tdp43_splicing_regulation]
            → motor neuron survival [evi:stmn2_motor_neuron_survival]
              → ALSFRS-R stabilization (predicted)
Depth: 7
Evidence links: 6
Weakest link: evi:tdp43_proteostasis_er_stress (strength: emerging)
```

The `DEEPEN_CAUSAL_CHAIN` action asks the LLM: "Given this intervention and its known mechanism, what is the next step in the causal chain toward motor neuron survival? Cite evidence." Each step is stored as a relationship in the causal chain, with the evidence item that supports it.

**Why this matters:** A deep causal chain with strong evidence at every link is what distinguishes a curative protocol from a hope-and-pray combination. If any link is weak or missing, the system knows exactly where to focus research.

## 9. Hypothesis System

### 9.1 Hypothesis Generation
The LLM generates hypotheses in three categories:

1. **Mechanism hypotheses**: "Erik's TDP-43 pathology may be driven by loss-of-nuclear-function rather than toxic-gain-of-cytoplasmic-function, because his NfL elevation is moderate and FVC is preserved" (cite evidence)
2. **Interaction hypotheses**: "Riluzole + pridopidine may have synergistic neuroprotection via independent sigma-1R and glutamate pathways" (cite evidence)
3. **Eligibility hypotheses**: "Erik may be eligible for VTx-002 PIONEER-ALS Phase 1/2 based on his ALSFRS-R, FVC, and Ohio location" (cite trial criteria)

### 9.2 Hypothesis Lifecycle
```
GENERATED → SEARCHING → SUPPORTED / REFUTED / INSUFFICIENT
```

Each hypothesis tracks:
- `statement`: The testable claim
- `evidence_for`: list of evidence IDs supporting
- `evidence_against`: list of evidence IDs refuting
- `current_support_direction`: supports / refutes / insufficient
- `impact_on_protocol`: which protocol layer/intervention would change if confirmed

### 9.3 Hypothesis-Action Bridge
When a hypothesis is generated, the system plans 1-3 concrete actions to test it:
- "Search PubMed for sigma-1R + TDP-43 proteostasis"
- "Query ChEMBL for pridopidine selectivity data"
- "Check ClinicalTrials.gov for VTx-002 eligibility criteria"

These become the next actions in the queue.

## 10. Protocol Convergence

### 10.1 When to Regenerate
The full Phase 2 pipeline (6 stages) is computationally expensive (~25 LLM calls with 35B model). Regenerate only when:
- `new_evidence_since_last_protocol >= 10` (meaningful new information)
- A hypothesis resolution changes the subtype posterior significantly
- A new intervention is discovered (from ClinicalTrials.gov or literature)
- Manual trigger via config

### 10.2 Convergence Detection
Track the top-3 interventions per layer across protocol versions:
```python
def is_converged(history: list[CureProtocolCandidate], window: int = 3) -> bool:
    """Protocol has converged when top interventions are stable for `window` consecutive regenerations."""
    if len(history) < window:
        return False
    recent = history[-window:]
    # Compare top intervention per layer across recent protocols
    for layer in ALL_LAYERS:
        tops = [get_top_intervention(p, layer) for p in recent]
        if len(set(tops)) > 1:
            return False
    return True
```

### 10.3 Post-Convergence
When the protocol converges, the system:
1. Produces a **final protocol report** with full causal chains, evidence summary, uncertainty disclosure
2. Identifies **next-best-measurement**: the single most valuable new data point (e.g., genetic testing results)
3. Enters a **monitoring mode**: checks for new clinical trials, publications, and guideline changes at lower frequency
4. Re-activates if new data arrives (genetics, new biomarkers, new trial results)

## 11. Learning Episodes

Each research step is logged as a `LearningEpisode`:
```python
LearningEpisode(
    subject_ref="traj:draper_001",
    trigger=f"step:{step_count}",
    state_snapshot_ref=f"research_state:{step_count}",
    protocol_ref=current_protocol.id if current_protocol else None,
    body={
        "action": action_type,
        "action_params": {...},
        "result_summary": "...",
        "evidence_items_added": N,
        "reward": R,
        "reward_components": {...},
    },
)
```

## 12. File Structure

```
scripts/
  research/
    __init__.py
    loop.py               # Main research loop (run_research_loop)
    state.py              # ResearchState dataclass
    actions.py            # 10 action types + action execution
    policy.py             # Action selection (uncertainty-directed + hypothesis-guided)
    rewards.py            # 8-component reward computation
    hypotheses.py         # Hypothesis generation, lifecycle, action planning
    causal_chains.py      # Causal chain construction and deepening
    convergence.py        # Protocol convergence detection + reporting
    episode_logger.py     # LearningEpisode persistence
  config/
    loader.py             # MODIFY: add Phase 3 config keys
tests/
  test_research_state.py
  test_research_actions.py
  test_research_policy.py
  test_research_rewards.py
  test_research_hypotheses.py
  test_causal_chains.py
  test_convergence.py
  test_episode_logger.py
  test_research_loop.py
```

## 13. Config Additions

```json
{
  "research_loop_enabled": true,
  "research_max_steps": 500,
  "research_protocol_regen_threshold": 10,
  "research_convergence_window": 3,
  "research_llm_model": "fallback",
  "research_protocol_llm_model": "primary",
  "research_step_timeout_s": 120,
  "research_hypothesis_max_active": 10,
  "research_causal_chain_target_depth": 5,
  "research_exploration_fraction": 0.15,
  "research_inter_step_pause_s": 1.0,
  "research_pubmed_max_per_query": 20,
  "research_trials_max_per_search": 30
}
```

**Memory management keys:**
- `research_llm_model`: `"fallback"` uses 9B (4.7GB) for routine actions; `"primary"` uses 35B (17GB)
- `research_protocol_llm_model`: `"primary"` uses 35B for protocol generation only
- `research_inter_step_pause_s`: GC breathing room between steps (lesson from Galen v52.0)

## 14. LLM Configuration

Two-tier model strategy:
- **Research model** (9B, 4.7GB): Hypothesis generation, causal chain extension, evidence scoring. Fast, cheap, good enough for iterative refinement.
- **Protocol model** (35B, 17GB): Full protocol regeneration. Only loaded when `REGENERATE_PROTOCOL` fires. Unloaded after use to free memory.

```python
class DualLLMManager:
    """Manages lazy loading/unloading of two LLM tiers."""
    def get_research_engine(self) -> ReasoningEngine:
        """Returns 9B-backed engine (stays loaded, ~4.7GB)."""
    def get_protocol_engine(self) -> ReasoningEngine:
        """Returns 35B-backed engine (loaded on demand, unloaded after)."""
    def unload_protocol_model(self) -> None:
        """Free 35B model memory: del model/tokenizer → gc.collect() → mlx.core.metal.clear_cache()."""
```

## 15. Testing Strategy

**Unit tests (no LLM, no DB, no network):**
- ResearchState construction and serialization
- Action selection policy with mock state
- Reward computation with mock results
- Hypothesis lifecycle state transitions
- Convergence detection with mock protocol histories
- Causal chain construction from mock data

**Integration tests (with LLM, marked @pytest.mark.llm):**
- Hypothesis generation produces valid MechanismHypothesis
- Causal chain deepening extends an existing chain
- Full loop runs 5 steps without error

**Network integration tests (marked @pytest.mark.network):**
- PubMed search returns and integrates evidence
- ClinicalTrials search finds relevant trials

## 16. Success Criteria

Phase 3 is complete when:

1. The research loop runs autonomously, selecting actions based on protocol uncertainty
2. Evidence fabric grows from 93 items to 200+ through systematic acquisition
3. At least 10 mechanistic hypotheses are generated and resolved
4. Causal chains reach depth ≥ 5 for the top 3 protocol interventions
5. Protocol regeneration produces measurably better protocols (higher scores, fewer uncertainties)
6. Convergence detection works — the loop identifies when the protocol stabilizes
7. A final protocol report is produced with full causal chains and evidence summary
8. All unit tests pass without LLM; integration tests pass with LLM
9. The system runs for 100+ steps without OOM on the M4 Max alongside Galen
