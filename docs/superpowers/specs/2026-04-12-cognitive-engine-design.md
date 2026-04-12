# Erik Cognitive Engine — Continuous Understanding Architecture

**Date:** 2026-04-12
**Author:** Logan Nye + Claude
**Status:** Approved design, pending implementation plan

## Problem Statement

Erik's research loop has stagnated at 87,660 steps with 4,306 evidence items. Recent steps produce zero new evidence because:

1. **Evidence deduplication saturates.** PostgreSQL `ON CONFLICT` upserts correctly deduplicate, but the loop treats evidence count as the sole progress signal. Once all publicly available facts about 28 ALS genes are found, reward drops to zero permanently.
2. **No synthesis layer.** The reasoning engine strips uncited claims, so the system cannot build multi-evidence insights that integrate what it already knows.
3. **No progressive understanding.** There is no representation of "how well do I understand this mechanism?" that deepens over time. Causal chains have a fixed max depth (15) and all 7 are maxed.
4. **Blind acquisition.** The 5-step cycle rotates through 25 action types regardless of what's needed. Gap analysis identifies unmeasurable clinical gaps (genetic testing, CSF biomarkers) and loops on them.
5. **Reward misalignment.** The reward function pays for new evidence items (finite) rather than understanding gains (infinite).

**Core insight:** Erik treats evidence accumulation as the goal, but evidence is finite. Understanding is infinite. The system needs to reward *thinking harder about facts it already has*, not just finding new facts.

## Design Philosophy

**Depth drives breadth.** Understanding identifies what's missing, and acquisition fills those gaps. A virtuous cycle that never stagnates because deeper understanding always reveals new questions.

This is a layered evolution (Approach 2), not a rewrite. The existing research loop becomes the acquisition engine. New background daemons handle integration, reasoning, and compound evaluation. Erik never stops running during the transition.

## 1. Therapeutic Causal Graph (TCG)

The central data structure representing Erik's understanding of ALS biology and how to cure it.

### What It Is

A directed graph in PostgreSQL where:
- **Nodes** are biological entities: genes, proteins, pathways, phenotypes, drug targets, compounds, clinical measurements
- **Edges** are directed mechanistic links: "TDP-43 aggregation -> STMN2 cryptic exon inclusion -> axon degeneration"
- Every edge carries confidence, evidence support, contradictions, open questions, intervention potential, and edge type

### Schema (new tables in `erik_core`)

```sql
-- TCG Nodes: biological entities in the ALS mechanistic model
CREATE TABLE erik_core.tcg_nodes (
    id TEXT PRIMARY KEY,                -- e.g. "protein:tdp-43", "pathway:autophagy"
    entity_type TEXT NOT NULL,          -- gene, protein, pathway, phenotype, compound, measurement
    name TEXT NOT NULL,
    description TEXT,
    pathway_cluster TEXT,               -- proteostasis, rna_metabolism, excitotoxicity,
                                        -- neuroinflammation, axonal_transport, mitochondrial,
                                        -- neuromuscular_junction, glial_biology
    druggability_score FLOAT DEFAULT 0, -- 0-1, updated by CompoundDaemon
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- TCG Edges: directed mechanistic links between entities
CREATE TABLE erik_core.tcg_edges (
    id TEXT PRIMARY KEY,                -- e.g. "edge:tdp43_agg->stmn2_splicing"
    source_id TEXT NOT NULL REFERENCES erik_core.tcg_nodes(id),
    target_id TEXT NOT NULL REFERENCES erik_core.tcg_nodes(id),
    edge_type TEXT NOT NULL,            -- activates, inhibits, upregulates, downregulates,
                                        -- causes, prevents, binds, transports
    confidence FLOAT DEFAULT 0.1,       -- 0-1, strengthened by IntegrationDaemon
    evidence_ids TEXT[] DEFAULT '{}',
    contradiction_ids TEXT[] DEFAULT '{}',
    open_questions TEXT[] DEFAULT '{}',
    intervention_potential JSONB DEFAULT '{}',
                                        -- {druggable: bool, compounds: [...], mechanism: "..."}
    last_reasoned_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_tcg_edges_source ON erik_core.tcg_edges(source_id);
CREATE INDEX idx_tcg_edges_target ON erik_core.tcg_edges(target_id);
CREATE INDEX idx_tcg_edges_confidence ON erik_core.tcg_edges(confidence);
CREATE INDEX idx_tcg_edges_cluster ON erik_core.tcg_edges(source_id, target_id)
    INCLUDE (confidence, edge_type);

-- TCG Hypotheses: therapeutic hypotheses with causal justification
CREATE TABLE erik_core.tcg_hypotheses (
    id TEXT PRIMARY KEY,
    hypothesis TEXT NOT NULL,
    supporting_path TEXT[] DEFAULT '{}', -- ordered list of edge IDs forming causal argument
    confidence FLOAT DEFAULT 0.1,
    status TEXT DEFAULT 'proposed',      -- proposed, under_investigation, supported, refuted, actionable
    generated_by TEXT,                   -- which phase/daemon generated this
    evidence_for TEXT[] DEFAULT '{}',
    evidence_against TEXT[] DEFAULT '{}',
    open_questions TEXT[] DEFAULT '{}',
    therapeutic_relevance FLOAT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_tcg_hypotheses_status ON erik_core.tcg_hypotheses(status);
CREATE INDEX idx_tcg_hypotheses_relevance ON erik_core.tcg_hypotheses(therapeutic_relevance DESC);
```

### Acquisition Queue

```sql
-- Acquisition Queue: TCG-driven targeted evidence requests
CREATE TABLE erik_ops.acquisition_queue (
    id SERIAL PRIMARY KEY,
    tcg_edge_id TEXT REFERENCES erik_core.tcg_edges(id),
    open_question TEXT NOT NULL,
    suggested_sources TEXT[] DEFAULT '{}',
    exhausted_sources TEXT[] DEFAULT '{}',
    priority FLOAT DEFAULT 0,           -- therapeutic_relevance * (1 - confidence)
    status TEXT DEFAULT 'pending',       -- pending, in_progress, answered, unanswerable
    created_by TEXT,                     -- integration, reasoning, compound
    created_at TIMESTAMPTZ DEFAULT now(),
    answered_at TIMESTAMPTZ
);

CREATE INDEX idx_acq_queue_priority ON erik_ops.acquisition_queue(priority DESC)
    WHERE status = 'pending';
```

### Activity Feed

```sql
-- Activity Feed: significant events across all phases
CREATE TABLE erik_ops.activity_feed (
    id SERIAL PRIMARY KEY,
    phase TEXT NOT NULL,                 -- acquisition, integration, reasoning, compound
    event_type TEXT NOT NULL,            -- edge_strengthened, contradiction_found,
                                         -- hypothesis_generated, compound_evaluated, etc.
    summary TEXT NOT NULL,               -- human-readable description
    details JSONB DEFAULT '{}',
    tcg_edge_id TEXT,
    tcg_hypothesis_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_activity_feed_created ON erik_ops.activity_feed(created_at DESC);
CREATE INDEX idx_activity_feed_phase ON erik_ops.activity_feed(phase, created_at DESC);
```

### LLM Spend Tracking

```sql
-- LLM Spend: cost tracking for Claude API budget control
CREATE TABLE erik_ops.llm_spend (
    id SERIAL PRIMARY KEY,
    model TEXT NOT NULL,                 -- claude-opus-4-6, claude-sonnet-4-6, bedrock-nova-micro, etc.
    phase TEXT NOT NULL,                 -- reasoning, compound, integration, acquisition
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd FLOAT,
    prompt_cached BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_llm_spend_month ON erik_ops.llm_spend(created_at)
    WHERE created_at > now() - INTERVAL '31 days';
```

## 2. Four Cognitive Phases

Each phase runs as an independent background daemon (thread), communicating through the TCG and PostgreSQL. Same pattern as Galen's daemon architecture.

### Phase 1 — Acquisition (existing loop, redirected)

**Timescale:** Every 1-3 seconds
**LLM tier:** Bedrock Nova Micro
**Module:** Existing `scripts/research/loop.py`, modified

**Behavior:**
1. Pop the highest-priority item from `erik_ops.acquisition_queue`
2. Translate the open question into a connector-specific query (LLM-assisted query generation)
3. Execute the appropriate connector based on `suggested_sources`
4. Store raw evidence in `erik_core.objects` (existing evidence store, unchanged)
5. Tag evidence with the TCG edge it was acquired to investigate
6. Mark queue item as `answered` (evidence found) or try next source / mark `unanswerable`

**Source selection mapping:**
| Question type | Suggested sources |
|--------------|-------------------|
| Mechanistic ("does X cause Y?") | PubMed, bioRxiv, Galen KG |
| Binding/affinity ("does X bind Y?") | ChEMBL, BindingDB, DrugBank |
| Expression ("is gene X expressed in motor neurons?") | GTEx, HPA, GEO |
| Genetic ("is variant X pathogenic?") | ClinVar, gnomAD, GWAS Catalog, ALSoD |
| Clinical ("is drug X in ALS trials?") | ClinicalTrials.gov, FAERS |
| Pathway ("what pathway includes X?") | Reactome, KEGG, STRING |
| Structural ("3D structure of X?") | AlphaFold, UniProt |

**Stagnation prevention:** The system never asks "what should I search?" generically. The TCG tells it exactly which edge is weak, what question to answer, and which data source to try. When all suggested sources for a question are exhausted, the item is marked `unanswerable` — not retried in a loop.

**Fallback:** When the acquisition queue is empty (all questions answered or unanswerable), Phase 1 enters a low-frequency exploration mode: pick a random TCG node, expand its neighborhood via PPI or pathway queries, look for entities not yet in the graph. This is slow but prevents total idleness.

**Config:**
```json
{
    "acquisition_enabled": true,
    "acquisition_interval_s": 2,
    "acquisition_min_confidence_target": 0.7,
    "acquisition_max_per_edge": 10,
    "acquisition_fallback_exploration": true
}
```

### Phase 2 — Integration (new daemon)

**Timescale:** Every 2-5 minutes
**LLM tier:** Bedrock Nova Pro
**Module:** `scripts/daemons/integration_daemon.py` (new)

**Behavior:**
1. Query `erik_core.objects` for evidence with `tcg_integrated = FALSE` (new column)
2. For each batch of unintegrated evidence:
   a. Extract entities and relationships mentioned in the evidence (LLM-assisted)
   b. Match to existing TCG nodes (fuzzy match + LLM disambiguation)
   c. For each relationship found:
      - **Edge exists:** Bayesian confidence update, append to `evidence_ids`
      - **Edge does not exist:** Create new edge, confidence proportional to evidence strength
      - **Evidence contradicts edge:** Append to `contradiction_ids`, flag for ReasoningDaemon
   d. For each new entity: create TCG node, infer pathway cluster, link to existing nodes
3. Re-score modified edges' `open_questions` — remove answered questions
4. Create `acquisition_queue` entries for new low-confidence edges
5. Mark evidence as `tcg_integrated = TRUE`
6. Write significant events to `erik_ops.activity_feed`

**Bayesian confidence update:**
```
posterior_confidence = (prior_confidence * prior_strength + evidence_strength) / (prior_strength + 1)
```
Where `prior_strength` grows with the number of supporting evidence items, providing natural diminishing returns — the 20th paper supporting an edge moves confidence less than the 2nd.

**Contradiction handling:** The IntegrationDaemon does not resolve contradictions. It flags them by adding to `contradiction_ids`. The ReasoningDaemon (Phase 3) handles resolution with deeper analysis. This prevents fast-but-shallow integration from making bad judgment calls.

**Config:**
```json
{
    "integration_enabled": true,
    "integration_interval_s": 180,
    "integration_batch_size": 50,
    "integration_confidence_prior_strength": 2.0
}
```

### Phase 3 — Reasoning (new daemon)

**Timescale:** Every 15-30 minutes
**LLM tier:** Claude API (Opus for deep analysis, Sonnet for lighter tasks)
**Module:** `scripts/daemons/reasoning_daemon.py` (new)

Three reasoning modes, rotated by configurable weights (default 0.5 / 0.3 / 0.2):

**Mode A — Edge Deepening (most frequent, uses Sonnet):**
1. Select edge with highest `therapeutic_relevance * (1 - confidence)` score
2. Gather all supporting and contradicting evidence
3. Prompt Claude: analyze causal link, assess mechanism, identify what would resolve uncertainty, check for confounders
4. Update edge confidence, open_questions, metadata
5. If Claude identifies an intermediate mechanism, split the edge through a new node — the graph gets more precise

**Mode B — Counterfactual Analysis (uses Opus):**
1. Select a hypothesis from `tcg_hypotheses` with status `under_investigation`
2. Trace full causal path through TCG
3. Prompt Claude: if we intervene at this node with this compound, trace every downstream consequence. Assess off-target effects, unintended pathway interactions.
4. Update hypothesis confidence, evidence_for/against
5. If analysis reveals new pathway interactions, create new TCG edges

**Mode C — Cross-Pathway Synthesis (least frequent, highest value, uses Opus):**
1. Select two pathway clusters that share few edges
2. Gather strongest evidence from each cluster
3. Prompt Claude: are there undiscovered mechanistic links between these clusters? Could intervening in one affect the other?
4. Discovered cross-pathway links become new TCG edges with low confidence — triggering acquisition and integration cycles

**Why this never stagnates:** Every reasoning cycle produces at least one of:
- Edge confidence change (up or down — both are progress)
- New edge or node (graph grows)
- New open question (drives acquisition)
- New therapeutic hypothesis (drives compound evaluation)
- Edge split into finer mechanism (graph gets more precise)

**Config:**
```json
{
    "reasoning_enabled": true,
    "reasoning_interval_s": 900,
    "reasoning_model_deep": "claude-opus-4-6",
    "reasoning_model_light": "claude-sonnet-4-6",
    "reasoning_mode_weights": [0.5, 0.3, 0.2],
    "reasoning_max_evidence_per_prompt": 30
}
```

### Phase 4 — Compound Evaluation (new daemon)

**Timescale:** Every 1-2 hours
**LLM tier:** Claude Sonnet
**Module:** `scripts/daemons/compound_daemon.py` (new)

**Behavior:**
1. Query `tcg_hypotheses` for hypotheses with status `supported` and high `therapeutic_relevance`
2. For each actionable hypothesis:
   a. Identify intervention points (which TCG nodes are druggable?)
   b. Query ChEMBL, DrugBank, BindingDB for compounds modulating those targets
   c. Score compounds on: target affinity, CNS penetrance (CNS_MPO), ADMET, drug-drug interactions with current protocol, clinical stage
   d. Run `molecule_generator.py` for novel compound design when no existing compounds score well (with SMILES deduplication to prevent the current re-generation bug)
3. Compare top candidates against current protocol
4. If protocol change warranted, generate structured recommendation with full TCG causal justification
5. Update `tcg_nodes.druggability_score` for evaluated targets
6. Write events to activity feed

**Protocol updates** are triggered by specific therapeutic insights from the graph, not by blind evidence accumulation thresholds.

**Config:**
```json
{
    "compound_enabled": true,
    "compound_interval_s": 3600,
    "compound_min_hypothesis_confidence": 0.6,
    "compound_max_candidates_per_target": 10
}
```

### Phase Coordination

No explicit orchestrator. The TCG is the shared blackboard:
- Phase 1 reads `acquisition_queue` (populated by Phases 2, 3, 4)
- Phase 2 reads unintegrated evidence (produced by Phase 1)
- Phase 3 reads low-confidence/high-relevance edges (updated by Phase 2)
- Phase 4 reads supported hypotheses (produced by Phase 3)

Each daemon reads what it needs, writes what it produces. No direct daemon-to-daemon communication.

**Starvation prevention:** If any phase has nothing to do, it shifts to speculative mode:
- Phase 1: random neighborhood exploration
- Phase 2: re-examine old evidence with updated graph context
- Phase 3: cross-pathway synthesis (Mode C) — always available
- Phase 4: adversarial protocol stress-testing

## 3. Reward System

### Phase-Specific Reward Components

**Phase 1 — Acquisition:**
| Component | Weight | Description |
|-----------|--------|-------------|
| `targeted_evidence_gain` | 2.0 | Evidence acquired for a specific TCG open question (3x untargeted) |
| `source_diversity` | 1.0 | New data source contributing to an edge for the first time |

**Phase 2 — Integration:**
| Component | Weight | Description |
|-----------|--------|-------------|
| `confidence_delta` | 4.0 | Sum of edge confidence increases in this cycle |
| `contradiction_discovered` | 2.0 | Evidence found contradicting an existing edge |
| `graph_growth` | 1.5 | New connected nodes or edges added to TCG |

**Phase 3 — Reasoning:**
| Component | Weight | Description |
|-----------|--------|-------------|
| `edge_resolved` | 5.0 | Edge driven to confidence >0.8 or refuted |
| `cross_pathway_link` | 5.0 | New connection between previously unconnected pathway clusters |
| `hypothesis_generated` | 3.0 | Novel therapeutic hypothesis with TCG causal path |
| `question_depth` | 3.0 | Resolving one question reveals a deeper question (edge split) |

**Phase 4 — Compound:**
| Component | Weight | Description |
|-----------|--------|-------------|
| `candidate_improvement` | 4.0 | New compound scores better than current best for a target |
| `protocol_justified_update` | 5.0 | Protocol change with full TCG causal justification |
| `coverage_increase` | 3.0 | Fraction of high-confidence edges covered by interventions increases |

### Why Reward Never Reaches Zero

Understanding is fractal. Every resolved edge reveals finer-grained questions. Every cross-pathway link creates new edges needing investigation. Every therapeutic hypothesis generates new questions about interactions, dosing, and off-target effects. The reward function always has positive gradient:

- Low edge confidence -> Phases 2/3 rewarded for raising it
- High edge confidence -> Phase 3 rewarded for discovering finer sub-mechanisms or cross-pathway links
- All edges high -> Phase 3 Mode C discovers new inter-cluster edges with low confidence -> cycle restarts
- Great compound found -> Phase 4 rewarded, compound's mechanism generates new edges to investigate

## 4. Global Progress Metrics

| Metric | Description | Start (est.) | Target |
|--------|-------------|-------------|--------|
| Graph Confidence | Mean confidence across all TCG edges | ~0.15 | 0.70+ |
| Therapeutic Coverage | % of high-confidence edges addressed by >= 1 intervention | ~20% | 80%+ |
| Pathway Completeness | % of clusters with >50 high-confidence edges | ~0% | 100% |
| Hypothesis Pipeline | Count by status: proposed / investigating / supported / actionable | 0 | 10+ actionable |
| Compound Pipeline | Drug candidates per intervention point, by eval stage | ~7 | 15+ evaluated |
| Protocol Score | coverage * mean_intervention_confidence * safety_score | ~0.3 | 0.8+ |
| Cross-Pathway Density | Inter-cluster edges / total edges | ~5% | 25%+ |
| Evidence Utilization | % of evidence items integrated into >= 1 TCG edge | ~0% | 95%+ |

### Convergence Redefined

Erik never converges. Two operational modes:

**Active mode:** At least one metric below target. All four phases at full speed.

**Deepening mode:** All primary metrics at target. Phases 1-2 slow down (longer intervals). Phase 3 shifts to cross-pathway synthesis and speculative reasoning. Phase 4 stress-tests the protocol. The system still improves, just on harder problems.

Transition to deepening mode triggers a confidence report to the family dashboard.

## 5. LLM Routing

### Three Tiers

| Tier | Model | Used By | Cost | Quality Need |
|------|-------|---------|------|-------------|
| Fast | Bedrock Nova Micro | Phase 1 (query gen, entity extraction) | ~$0.04/1M in | Low |
| Moderate | Bedrock Nova Pro | Phase 2 (entity disambiguation, edge mapping) | ~$0.20/1M in | Medium |
| Deep | Claude Opus / Sonnet | Phase 3 (reasoning), Phase 4 (evaluation) | Higher | Maximum |

### Claude API Integration

New module: `scripts/llm/claude_client.py`

Interface:
- `reason_about_edge(edge, supporting_evidence, contradicting_evidence) -> EdgeAnalysis`
- `counterfactual_analysis(hypothesis, causal_path, tcg_context) -> CounterfactualResult`
- `cross_pathway_synthesis(cluster_a_evidence, cluster_b_evidence) -> list[ProposedEdge]`
- `evaluate_compound(compound, target_edges, current_protocol) -> CompoundEvaluation`
- `justify_protocol_change(proposed_change, tcg_subgraph) -> ProtocolJustification`

Each method builds a purpose-specific prompt with relevant TCG subgraph context, all supporting/contradicting evidence, the specific question, and structured JSON output format.

**Prompt caching:** System prompt + TCG scaffold context in cacheable prefix. Specific question in variable suffix. Cuts cost and latency for repeated reasoning over similar subgraphs.

**Rate limiting:** `claude_max_opus_calls_per_hour` (default 30), `claude_max_sonnet_calls_per_hour` (default 60), `claude_monthly_budget_usd` (hard cap, daemon pauses when reached). All calls logged to `erik_ops.llm_spend`.

### Fallback Chain

1. Sonnet tasks -> Bedrock Nova Pro
2. Opus tasks -> Sonnet -> Nova Pro
3. All cloud unavailable -> local Qwen3.5-35B (when Galen isn't saturating GPU)
4. No LLM at all -> reasoning daemon pauses, acquisition and integration continue with structured data only

The system never fully stops.

### Config

```json
{
    "anthropic_api_key": "sk-ant-...",
    "claude_reasoning_model": "claude-opus-4-6",
    "claude_evaluation_model": "claude-sonnet-4-6",
    "claude_max_opus_calls_per_hour": 30,
    "claude_max_sonnet_calls_per_hour": 60,
    "claude_monthly_budget_usd": 100,
    "claude_prompt_cache_enabled": true,
    "claude_fallback_chain": ["claude-sonnet-4-6", "bedrock-nova-pro", "local-qwen"]
}
```

## 6. ALS Scaffold Seeding

One-time initialization via `scripts/tcg/seed_scaffold.py`.

### Three-Step Seeding

**Step 1 — Hardcoded ALS biology backbone (~200 nodes, ~600 edges):**

8 pathway clusters with major genes/proteins and known causal links:

1. **Proteostasis:** TDP-43, SOD1, FUS, VCP, ubiquitin-proteasome, autophagy/mTOR, chaperones
2. **RNA metabolism:** STMN2, UNC13A, cryptic exon splicing, stress granules, ATXN2
3. **Excitotoxicity:** glutamate, EAAT2/SLC1A2, AMPA/NMDA receptors, calcium homeostasis
4. **Neuroinflammation:** microglia, CSF1R, astrocytes, NF-kB, complement cascade
5. **Axonal transport:** dynein/kinesin, neurofilaments, TDP-43 transport role
6. **Mitochondrial:** CHCHD10, oxidative stress, bioenergetics, edaravone mechanism
7. **Neuromuscular junction:** denervation, reinnervation, muscle atrophy signals
8. **Glial biology:** oligodendrocytes, Schwann cells, non-cell-autonomous toxicity

Starting confidence:
- 0.7-0.9: well-established links (FDA-approved drug mechanisms, replicated genetic associations)
- 0.3-0.6: established-but-nuanced links (pathway interactions with known caveats)
- 0.1-0.2: hypothesized links (emerging research, single-study findings)

Every edge below 0.7 gets at least one seeded open question.

**Step 2 — Integrate existing evidence (~4,306 items):**

Run IntegrationDaemon in batch mode over all existing `erik_core.objects`. Maps 18 days of accumulated research onto the scaffold. Scaffold goes from textbook knowledge to evidence-grounded in a single pass.

**Step 3 — Galen KG enrichment:**

Query Galen's KG (700K+ entities) for ALS-relevant entities. Pull:
- Quantitative binding data (IC50, Ki, EC50)
- Expression levels from overlapping biology
- Drug mechanism-of-action edges
- Protein-protein interactions

Added as supplementary evidence on existing TCG edges. New edges created only for strong Galen evidence (L2+ PCH).

### Post-Seeding

The scaffold is a one-time seed. After initialization:
- Refuted edges get removed
- ReasoningDaemon-discovered edges are first-class citizens
- After a few thousand integration cycles, the scaffold is unrecognizable — reshaped by evidence

## 7. API and Dashboard Updates

### New API Endpoints

**TCG exploration:**
- `GET /api/graph/summary` — node/edge counts, mean confidence, pathway cluster breakdown
- `GET /api/graph/cluster/{cluster_name}` — all nodes and edges in a cluster with confidence
- `GET /api/graph/edge/{edge_id}` — full edge detail: evidence, contradictions, open questions
- `GET /api/graph/weakest?limit=10` — weakest high-relevance edges ("what Erik is investigating now")

**Hypothesis pipeline:**
- `GET /api/hypotheses` — all hypotheses with status, confidence, therapeutic relevance
- `GET /api/hypotheses/{id}` — full detail with causal path and evidence

**Progress dashboard:**
- `GET /api/progress` — all 8 global metrics with historical values for trend lines
- `GET /api/progress/phases` — phase status: last run, items processed, reward earned

**Activity feed (replaces broken existing endpoint):**
- `GET /api/activity` — unified feed of significant events across all phases

### Frontend Updates (Vercel)

1. **Progress dashboard** — 8 metrics as trend lines showing understanding deepening over time
2. **Live activity feed** — real-time scrolling feed of what Erik is doing (replaces empty page)
3. **Protocol page (enhanced)** — TCG causal chain justification for each intervention, not just relevance scores
4. **Graph explorer (new)** — interactive TCG visualization. Nodes colored by pathway cluster, edges colored by confidence. Click to see evidence and open questions.

## 8. New File Structure

```
scripts/
  tcg/
    __init__.py
    models.py              -- TCG node/edge/hypothesis dataclasses
    graph.py               -- TCG read/write operations, Bayesian updates
    seed_scaffold.py       -- one-time ALS biology scaffold seeder
    acquisition_queue.py   -- queue read/write operations
  daemons/
    __init__.py
    integration_daemon.py  -- Phase 2: evidence -> TCG integration
    reasoning_daemon.py    -- Phase 3: Claude-powered deep reasoning
    compound_daemon.py     -- Phase 4: drug candidate evaluation
  llm/
    claude_client.py       -- Claude API client with prompt caching, rate limiting, fallback
  api/routers/
    graph.py               -- TCG exploration endpoints
    hypotheses.py          -- hypothesis pipeline endpoints
    progress.py            -- progress metrics endpoints
```

Existing files modified:
- `scripts/research/loop.py` — acquisition driven by `acquisition_queue` instead of blind rotation
- `scripts/research/rewards.py` — new reward components per phase
- `scripts/research/convergence.py` — remove convergence, replace with active/deepening mode
- `scripts/run_loop.py` — start daemons alongside existing loop
- `scripts/db/migrate.py` — new tables
- `scripts/api/main.py` — mount new routers
- `data/erik_config.json` — new config keys for all phases and Claude API
- `erik_core.objects` — add `tcg_integrated BOOLEAN DEFAULT FALSE` column

## 9. The Structural Guarantee

Why this design guarantees continuous improvement:

1. **Understanding is the reward, not evidence.** The reward function pays for confidence increases, contradiction resolution, cross-pathway discovery, and hypothesis advancement. These never run out.

2. **The TCG drives acquisition.** The system never asks "what database should I query?" It asks "what don't I understand well enough?" and queries the specific source that can answer that question.

3. **Four phases create a virtuous cycle.** Acquisition feeds integration. Integration feeds reasoning. Reasoning generates new questions (feeding acquisition) and hypotheses (feeding compound evaluation). Compound evaluation reveals gaps (feeding reasoning). No phase can stagnate without creating work for another.

4. **Cross-pathway synthesis is the infinite frontier.** 8 pathway clusters with ~250 edges each means ~2,000 potential cross-pathway connections. The ReasoningDaemon discovers new interactions indefinitely.

5. **The graph gets more precise, not just bigger.** Splitting coarse edges into finer mechanism chains is unbounded. "TDP-43 aggregation -> neurodegeneration" can always be refined into more specific intermediate steps.
