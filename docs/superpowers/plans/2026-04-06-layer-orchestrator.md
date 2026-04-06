# Layer Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a layer-aware research orchestrator that determines which of the 4 research layers the system is in (normal biology → ALS mechanisms → Erik's pathophysiology → drug design) and constrains the action space accordingly.

**Architecture:** New `research/layer_orchestrator.py` module determines the current `ResearchLayer` from state signals. `ResearchState` gets a `research_layer` field. `policy.py` reads the current layer to select layer-appropriate query templates. `run_loop.py` calls the orchestrator each step to update the layer. Layer transitions are gated: Layer 1→2 by evidence threshold, Layer 2→3 by genetic profile upload, Layer 3→4 by causal target validation.

**Tech Stack:** Python 3.12, PostgreSQL, dataclasses, existing ConfigLoader

**Codebase:** `/Users/logannye/.openclaw/erik/` with `PYTHONPATH=scripts`
**Test command:** `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v`
**Conventions:** TDD, PostgreSQL only (never sqlite3), follow existing patterns in `research/` module

---

## File Map

| File | Change |
|------|--------|
| Create: `scripts/research/layer_orchestrator.py` | ResearchLayer enum + `determine_layer()` function + layer-specific query banks |
| Modify: `scripts/research/state.py` | Add `research_layer` field to ResearchState |
| Modify: `scripts/research/policy.py` | Use `research_layer` to constrain query selection |
| Modify: `scripts/run_loop.py` | Call orchestrator each step to update layer |
| Create: `tests/test_layer_orchestrator.py` | Unit tests for layer determination + transitions |

---

### Task 1: Define ResearchLayer enum and determine_layer()

**Files:**
- Create: `scripts/research/layer_orchestrator.py`
- Create: `tests/test_layer_orchestrator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_orchestrator.py`:

```python
"""Tests for the layer orchestrator — determines which research phase the system is in."""
import pytest
from research.layer_orchestrator import ResearchLayer, determine_layer


def test_layer_enum_has_four_values():
    assert len(ResearchLayer) == 4
    assert ResearchLayer.NORMAL_BIOLOGY.value == "normal_biology"
    assert ResearchLayer.ALS_MECHANISMS.value == "als_mechanisms"
    assert ResearchLayer.ERIK_SPECIFIC.value == "erik_specific"
    assert ResearchLayer.DRUG_DESIGN.value == "drug_design"


def test_fresh_start_is_normal_biology():
    """A brand-new system with zero evidence starts in Layer 1."""
    layer = determine_layer(evidence_count=0, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.NORMAL_BIOLOGY


def test_early_evidence_stays_normal_biology():
    """With few evidence items, still building the normal biology model."""
    layer = determine_layer(evidence_count=30, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.NORMAL_BIOLOGY


def test_sufficient_evidence_advances_to_als_mechanisms():
    """Once enough basic biology evidence exists, advance to Layer 2."""
    layer = determine_layer(evidence_count=200, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.ALS_MECHANISMS


def test_large_evidence_without_genetics_stays_als_mechanisms():
    """Even with lots of evidence, can't advance to Layer 3 without genetics."""
    layer = determine_layer(evidence_count=5000, genetic_profile=None, validated_targets=0)
    assert layer == ResearchLayer.ALS_MECHANISMS


def test_genetics_received_advances_to_erik_specific():
    """Once genetic profile is uploaded, advance to Layer 3."""
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    layer = determine_layer(evidence_count=500, genetic_profile=profile, validated_targets=0)
    assert layer == ResearchLayer.ERIK_SPECIFIC


def test_validated_targets_advances_to_drug_design():
    """Once causal targets are validated, advance to Layer 4."""
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    layer = determine_layer(evidence_count=1000, genetic_profile=profile, validated_targets=3)
    assert layer == ResearchLayer.DRUG_DESIGN


def test_genetics_with_low_evidence_still_advances():
    """Genetics trump evidence count for Layer 2→3 transition."""
    profile = {"gene": "C9orf72", "variant": "repeat_expansion", "subtype": "C9orf72"}
    layer = determine_layer(evidence_count=100, genetic_profile=profile, validated_targets=0)
    assert layer == ResearchLayer.ERIK_SPECIFIC


def test_validated_targets_without_genetics_stays_als_mechanisms():
    """Can't jump to drug design without genetics (even if targets exist from prior work)."""
    layer = determine_layer(evidence_count=2000, genetic_profile=None, validated_targets=5)
    assert layer == ResearchLayer.ALS_MECHANISMS
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_layer_orchestrator.py -v
```

Expected: `ModuleNotFoundError: No module named 'research.layer_orchestrator'`

- [ ] **Step 3: Implement the layer orchestrator**

Create `scripts/research/layer_orchestrator.py`:

```python
"""Layer orchestrator — determines the current research phase.

The research engine progresses through four layers, each building on
the previous one:

  Layer 1 (Normal Biology):    Model healthy motor neuron function
  Layer 2 (ALS Mechanisms):    Map how ALS disrupts normal biology
  Layer 3 (Erik's Case):       Narrow to Erik's specific pathways
                                (requires genetic testing results)
  Layer 4 (Drug Design):       Design/identify molecules targeting
                                Erik's validated causal targets

Transitions are gated:
  1 → 2:  Evidence count >= LAYER_1_THRESHOLD (basic biology mapped)
  2 → 3:  Genetic profile uploaded (non-None)
  3 → 4:  Validated causal targets >= LAYER_3_TARGET_THRESHOLD
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional


class ResearchLayer(Enum):
    NORMAL_BIOLOGY = "normal_biology"
    ALS_MECHANISMS = "als_mechanisms"
    ERIK_SPECIFIC = "erik_specific"
    DRUG_DESIGN = "drug_design"


# Evidence count thresholds for layer transitions
LAYER_1_THRESHOLD = 100  # Advance to Layer 2 after basic biology foundation
LAYER_3_TARGET_THRESHOLD = 2  # Advance to Layer 4 after validating causal targets


def determine_layer(
    evidence_count: int,
    genetic_profile: Optional[dict[str, Any]],
    validated_targets: int,
) -> ResearchLayer:
    """Determine the current research layer from state signals.

    Args:
        evidence_count: Total evidence items in the knowledge graph.
        genetic_profile: Erik's genetic testing results (None if not yet received).
            Expected keys: gene, variant, subtype.
        validated_targets: Number of causal targets with L2+ evidence linking
            them to Erik's specific disease mechanism.

    Returns:
        The current ResearchLayer.
    """
    # Gate: Can't do drug design without genetics
    if genetic_profile is None:
        if evidence_count < LAYER_1_THRESHOLD:
            return ResearchLayer.NORMAL_BIOLOGY
        return ResearchLayer.ALS_MECHANISMS

    # Genetics received — at least Layer 3
    if validated_targets >= LAYER_3_TARGET_THRESHOLD:
        return ResearchLayer.DRUG_DESIGN

    return ResearchLayer.ERIK_SPECIFIC


# -----------------------------------------------------------------------
# Layer-specific query templates
# -----------------------------------------------------------------------

LAYER_QUERIES: dict[ResearchLayer, list[str]] = {
    ResearchLayer.NORMAL_BIOLOGY: [
        "motor neuron survival signaling pathway BDNF GDNF",
        "neuromuscular junction formation maintenance agrin LRP4 MuSK",
        "proteostasis chaperone system neuron protein quality control",
        "axonal transport dynein kinesin motor neuron",
        "glutamate receptor signaling motor neuron AMPA NMDA homeostasis",
        "mitochondrial function electron transport chain neuron ATP",
        "RNA processing splicing regulation motor neuron TDP-43 FUS normal",
        "superoxide dismutase SOD1 normal function reactive oxygen species",
        "autophagy lysosome pathway neuron protein clearance",
        "neurotrophic factor signaling motor neuron IGF-1 VEGF CNTF",
        "upper motor neuron corticospinal tract normal physiology",
        "Schwann cell myelination peripheral nerve motor function",
    ],
    ResearchLayer.ALS_MECHANISMS: [
        "ALS TDP-43 aggregation pathological cascade motor neuron",
        "ALS SOD1 misfolding toxic gain of function mechanism",
        "ALS C9orf72 repeat expansion dipeptide repeat RNA foci",
        "ALS FUS mutation RNA processing defect mechanism",
        "ALS glutamate excitotoxicity EAAT2 motor neuron death",
        "ALS neuroinflammation microglia astrocyte activation",
        "ALS mitochondrial dysfunction oxidative stress",
        "ALS axonal transport disruption neurofilament accumulation",
        "ALS neuromuscular junction denervation dying back",
        "ALS protein aggregation stress granule pathology",
        "ALS cortical hyperexcitability upper motor neuron",
        "ALS cryptic exon splicing STMN2 UNC13A loss",
    ],
    # Layer 3 queries are generated dynamically from the genetic profile
    ResearchLayer.ERIK_SPECIFIC: [],
    # Layer 4 queries are generated dynamically from validated targets
    ResearchLayer.DRUG_DESIGN: [],
}


def get_layer_queries(
    layer: ResearchLayer,
    genetic_profile: Optional[dict[str, Any]] = None,
    validated_targets: Optional[list[str]] = None,
) -> list[str]:
    """Get query templates for the current research layer.

    For Layers 1-2, returns static query banks.
    For Layer 3, generates queries from the genetic profile.
    For Layer 4, generates queries from validated causal targets.
    """
    if layer in (ResearchLayer.NORMAL_BIOLOGY, ResearchLayer.ALS_MECHANISMS):
        return LAYER_QUERIES[layer]

    if layer == ResearchLayer.ERIK_SPECIFIC and genetic_profile:
        gene = genetic_profile.get("gene", "")
        variant = genetic_profile.get("variant", "")
        subtype = genetic_profile.get("subtype", "")
        return [
            f"{gene} ALS mutation mechanism motor neuron",
            f"{gene} {variant} functional impact pathogenesis",
            f"{subtype} ALS subtype disease progression",
            f"{gene} ALS causal pathway downstream effects",
            f"{gene} protein structure function loss mutation",
            f"{gene} ALS patient genotype phenotype correlation",
            f"{gene} motor neuron selective vulnerability mechanism",
            f"{gene} ALS biomarker disease monitoring",
            f"{subtype} ALS prognosis trajectory prediction",
            f"{gene} ALS therapeutic target druggable site",
        ]

    if layer == ResearchLayer.DRUG_DESIGN and validated_targets:
        queries = []
        for target in validated_targets:
            queries.extend([
                f"{target} drug binding affinity inhibitor",
                f"{target} small molecule modulator ALS",
                f"{target} structure activity relationship drug design",
                f"{target} ADMET pharmacokinetics blood brain barrier",
            ])
        return queries if queries else [
            "ALS drug target structure based design",
            "ALS small molecule therapeutic development",
        ]

    # Fallback
    return LAYER_QUERIES[ResearchLayer.ALS_MECHANISMS]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_layer_orchestrator.py -v
```

Expected: All 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/research/layer_orchestrator.py tests/test_layer_orchestrator.py
git commit -m "feat: add layer orchestrator — 4-phase research layer determination"
```

---

### Task 2: Add research_layer to ResearchState

**Files:**
- Modify: `scripts/research/state.py`
- Create: `tests/test_research_state_layer.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_research_state_layer.py`:

```python
"""Tests for ResearchState layer field."""
from research.state import ResearchState, initial_state


def test_initial_state_has_normal_biology_layer():
    state = initial_state(subject_ref="test")
    assert state.research_layer == "normal_biology"


def test_state_roundtrip_preserves_layer():
    state = initial_state(subject_ref="test")
    state_dict = state.to_dict()
    assert state_dict["research_layer"] == "normal_biology"
    restored = ResearchState.from_dict(state_dict)
    assert restored.research_layer == "normal_biology"


def test_state_roundtrip_with_erik_specific():
    state = initial_state(subject_ref="test")
    from dataclasses import replace
    state = replace(state, research_layer="erik_specific")
    state_dict = state.to_dict()
    restored = ResearchState.from_dict(state_dict)
    assert restored.research_layer == "erik_specific"


def test_genetic_profile_roundtrip():
    state = initial_state(subject_ref="test")
    from dataclasses import replace
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    state = replace(state, genetic_profile=profile)
    state_dict = state.to_dict()
    restored = ResearchState.from_dict(state_dict)
    assert restored.genetic_profile == profile
    assert restored.genetic_profile["gene"] == "SOD1"


def test_state_without_layer_field_defaults():
    """Old state dicts (pre-layer) should default to normal_biology."""
    old_dict = {"subject_ref": "test", "step_count": 100}
    state = ResearchState.from_dict(old_dict)
    assert state.research_layer == "normal_biology"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_research_state_layer.py -v
```

Expected: `AttributeError: ... has no attribute 'research_layer'`

- [ ] **Step 3: Add research_layer and genetic_profile fields to ResearchState**

In `scripts/research/state.py`, add two new fields to the `ResearchState` dataclass (after `expansion_gene_history`):

```python
    research_layer: str = "normal_biology"
    genetic_profile: dict[str, Any] | None = None
```

Add `research_layer` and `genetic_profile` to the `to_dict()` method (after the `expansion_gene_history` line):

```python
        "research_layer": self.research_layer,
        "genetic_profile": dict(self.genetic_profile) if self.genetic_profile else None,
```

No changes needed in `from_dict()` — the existing generic loop handles unknown-to-known field mapping, and the default `"normal_biology"` covers old state dicts.

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_research_state_layer.py -v
```

Expected: All 5 tests PASS.

- [ ] **Step 5: Run full test suite to verify no regressions**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not test_fetch_trial_details_real" 2>&1 | tail -5
```

Expected: All existing tests still pass (the pre-existing network test is excluded).

- [ ] **Step 6: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/research/state.py tests/test_research_state_layer.py
git commit -m "feat: add research_layer and genetic_profile fields to ResearchState"
```

---

### Task 3: Integrate orchestrator into run_loop.py

**Files:**
- Modify: `scripts/run_loop.py`

- [ ] **Step 1: Add orchestrator import and layer update call**

At the top of `run_loop.py`, add to the imports (after the existing `from research.state import ...` line):

```python
from research.layer_orchestrator import determine_layer
```

- [ ] **Step 2: Add layer update after each research step**

In the `main()` function, after the `state = research_step(...)` call (line ~431) and before `_persist_state(state, evidence_store)`, add:

```python
            # Update research layer based on current state
            new_layer = determine_layer(
                evidence_count=state.total_evidence_items,
                genetic_profile=state.genetic_profile,
                validated_targets=sum(1 for d in state.causal_chains.values() if d >= 3),
            )
            if new_layer.value != state.research_layer:
                print(f"[ERIK] ★ LAYER TRANSITION: {state.research_layer} → {new_layer.value}")
                state = replace(state, research_layer=new_layer.value)
```

- [ ] **Step 3: Add the same layer update in the monitoring cycle**

In the `_monitoring_cycle()` function, after `state = _deep_research_step(...)` (line ~295) and before the uncertainty computation, add:

```python
    # Update research layer
    new_layer = determine_layer(
        evidence_count=state.total_evidence_items,
        genetic_profile=state.genetic_profile,
        validated_targets=sum(1 for d in state.causal_chains.values() if d >= 3),
    )
    if new_layer.value != state.research_layer:
        print(f"[ERIK-MONITOR] ★ LAYER TRANSITION: {state.research_layer} → {new_layer.value}")
        state = replace(state, research_layer=new_layer.value)
```

- [ ] **Step 4: Add layer to the startup log**

In `main()`, update the startup print after state is loaded (line ~337) to include the layer:

Replace:
```python
        print(f"[ERIK] Resumed from DB: step={state.step_count}, "
              f"protocol_v={state.protocol_version}, "
              f"evidence={state.total_evidence_items}, "
              f"converged={state.converged}")
```

With:
```python
        print(f"[ERIK] Resumed from DB: step={state.step_count}, "
              f"protocol_v={state.protocol_version}, "
              f"evidence={state.total_evidence_items}, "
              f"converged={state.converged}, "
              f"layer={state.research_layer}")
```

- [ ] **Step 5: Verify the module imports cleanly**

```bash
cd /Users/logannye/.openclaw/erik && PYTHONPATH=scripts conda run -n erik-core python -c "from research.layer_orchestrator import determine_layer, ResearchLayer; print('OK:', list(ResearchLayer))"
```

Expected: `OK: [<ResearchLayer.NORMAL_BIOLOGY: 'normal_biology'>, ...]`

- [ ] **Step 6: Run full test suite**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not test_fetch_trial_details_real" 2>&1 | tail -5
```

Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/run_loop.py
git commit -m "feat: integrate layer orchestrator into research loop"
```

---

### Task 4: Wire layer-aware queries into policy.py

**Files:**
- Modify: `scripts/research/policy.py`
- Create: `tests/test_layer_policy.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_layer_policy.py`:

```python
"""Tests for layer-aware query selection in the research policy."""
from research.layer_orchestrator import ResearchLayer, get_layer_queries


def test_normal_biology_queries_dont_mention_als():
    """Layer 1 queries should be about normal biology, not ALS drugs."""
    queries = get_layer_queries(ResearchLayer.NORMAL_BIOLOGY)
    assert len(queries) >= 8
    for q in queries:
        assert "therapy" not in q.lower() or "normal" in q.lower(), \
            f"Layer 1 query should not be therapy-focused: {q}"


def test_als_mechanism_queries_mention_als():
    """Layer 2 queries should focus on ALS disease mechanisms."""
    queries = get_layer_queries(ResearchLayer.ALS_MECHANISMS)
    assert len(queries) >= 8
    for q in queries:
        assert "als" in q.lower(), f"Layer 2 query should mention ALS: {q}"


def test_erik_specific_queries_use_genetic_profile():
    """Layer 3 queries should reference Erik's specific gene/variant."""
    profile = {"gene": "SOD1", "variant": "G93A", "subtype": "SOD1_familial"}
    queries = get_layer_queries(ResearchLayer.ERIK_SPECIFIC, genetic_profile=profile)
    assert len(queries) >= 5
    assert any("SOD1" in q for q in queries), "Layer 3 queries should mention Erik's gene"
    assert any("G93A" in q for q in queries), "Layer 3 queries should mention Erik's variant"


def test_drug_design_queries_use_validated_targets():
    """Layer 4 queries should reference specific drug targets."""
    targets = ["SOD1", "EAAT2"]
    queries = get_layer_queries(ResearchLayer.DRUG_DESIGN, validated_targets=targets)
    assert len(queries) >= 4
    assert any("SOD1" in q for q in queries)
    assert any("EAAT2" in q for q in queries)
    assert any("binding" in q.lower() or "drug" in q.lower() for q in queries)


def test_erik_specific_without_profile_falls_back():
    """Layer 3 without a profile should fall back to Layer 2 queries."""
    queries = get_layer_queries(ResearchLayer.ERIK_SPECIFIC, genetic_profile=None)
    assert len(queries) >= 8  # Should get ALS_MECHANISMS fallback


def test_drug_design_without_targets_falls_back():
    """Layer 4 without targets should fall back to generic drug design queries."""
    queries = get_layer_queries(ResearchLayer.DRUG_DESIGN, validated_targets=None)
    assert len(queries) >= 2
```

- [ ] **Step 2: Run tests to verify they pass (queries are already implemented in Task 1)**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_layer_policy.py -v
```

Expected: All 6 tests PASS (the query functions were implemented in Task 1).

- [ ] **Step 3: Modify policy.py to use layer-aware queries for PubMed**

In `scripts/research/policy.py`, in the `_build_acquisition_params()` function, replace the `SEARCH_PUBMED` branch (lines ~735-750) with a layer-aware version.

Find:
```python
    if action == ActionType.SEARCH_PUBMED:
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        layer = ALL_LAYERS[layer_idx]
        # 5-strategy cycling: static → dynamic → targeted → expanded → drug-centric
        strategy = step % 5
        if strategy == 0:
            query = get_layer_query(layer, step)
        elif strategy == 1:
            query = _get_dynamic_query(state, step, layer)
        elif strategy == 2:
            query = _get_targeted_query(state, step)
        elif strategy == 3:
            query = _get_expanded_query(state, step, layer)
        else:
            query = _get_drug_centric_query(state, step)
        return action, build_action_params(action, query=query, protocol_layer=layer)
```

Replace with:
```python
    if action == ActionType.SEARCH_PUBMED:
        layer_idx = (step // _CYCLE_LENGTH) % len(ALL_LAYERS)
        protocol_layer = ALL_LAYERS[layer_idx]

        # Layer-aware query selection: use research layer to pick appropriate queries
        from research.layer_orchestrator import ResearchLayer, get_layer_queries
        try:
            research_layer = ResearchLayer(state.research_layer)
        except (ValueError, AttributeError):
            research_layer = ResearchLayer.ALS_MECHANISMS

        layer_queries = get_layer_queries(
            research_layer,
            genetic_profile=getattr(state, "genetic_profile", None),
            validated_targets=[
                k for k, v in state.causal_chains.items() if v >= 3
            ] if research_layer == ResearchLayer.DRUG_DESIGN else None,
        )

        if layer_queries:
            # Rotate through layer-specific queries, with occasional dynamic/expanded
            strategy = step % 4
            if strategy <= 1 and layer_queries:
                # Primary: layer-specific query bank
                query = layer_queries[step % len(layer_queries)]
                year = __import__("datetime").datetime.now().year
                query = f"{query} {year}"
            elif strategy == 2:
                query = _get_dynamic_query(state, step, protocol_layer)
            else:
                query = _get_expanded_query(state, step, protocol_layer)
        else:
            # Fallback to existing 5-strategy cycling
            strategy = step % 5
            if strategy == 0:
                query = get_layer_query(protocol_layer, step)
            elif strategy == 1:
                query = _get_dynamic_query(state, step, protocol_layer)
            elif strategy == 2:
                query = _get_targeted_query(state, step)
            elif strategy == 3:
                query = _get_expanded_query(state, step, protocol_layer)
            else:
                query = _get_drug_centric_query(state, step)
        return action, build_action_params(action, query=query, protocol_layer=protocol_layer)
```

- [ ] **Step 4: Apply the same layer-aware logic to SEARCH_PREPRINTS**

In the `SEARCH_PREPRINTS` branch of `_build_acquisition_params()`, replace the query generation block (after the `biorxiv_enabled` check) with the same layer-aware logic. Find the 5-strategy cycling block and replace it with:

```python
        # Layer-aware query selection (same as SEARCH_PUBMED)
        from research.layer_orchestrator import ResearchLayer, get_layer_queries
        try:
            research_layer = ResearchLayer(state.research_layer)
        except (ValueError, AttributeError):
            research_layer = ResearchLayer.ALS_MECHANISMS

        layer_queries = get_layer_queries(
            research_layer,
            genetic_profile=getattr(state, "genetic_profile", None),
        )
        if layer_queries:
            query = layer_queries[step % len(layer_queries)]
            year = __import__("datetime").datetime.now().year
            query = f"{query} {year}"
        else:
            query = get_layer_query(layer, step)
        return action, build_action_params(action, query=query, protocol_layer=layer)
```

- [ ] **Step 5: Run full test suite**

```bash
cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v -k "not test_fetch_trial_details_real" 2>&1 | tail -5
```

Expected: All tests pass including the new layer policy tests.

- [ ] **Step 6: Commit**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/research/policy.py tests/test_layer_policy.py
git commit -m "feat: wire layer-aware queries into PubMed/preprint action selection"
```

---

### Task 5: Add research_layer to /api/summary response and frontend tracker

**Files:**
- Modify: `scripts/api/routers/summary.py`
- Modify: (frontend) `/Users/logannye/erik-website/lib/types.ts`
- Modify: (frontend) `/Users/logannye/erik-website/app/(dashboard)/dashboard/page.tsx`

- [ ] **Step 1: Add research_layer to the summary API response**

In `scripts/api/routers/summary.py`, in the `_get_last_24h_stats()` function, after the `state` variable is extracted from research_state, add:

```python
        # Get research layer
        research_layer = state.get("research_layer", "normal_biology") if state else "normal_biology"
```

Add `"research_layer": research_layer` to the returned dict.

In the `get_summary()` endpoint function, add `"research_layer"` to the response dict:

```python
        "research_layer": stats.get("research_layer", "normal_biology"),
```

- [ ] **Step 2: Add research_layer to frontend DailySummary type**

In `/Users/logannye/erik-website/lib/types.ts`, add to the `DailySummary` interface:

```typescript
  research_layer: "normal_biology" | "als_mechanisms" | "erik_specific" | "drug_design";
```

- [ ] **Step 3: Update the frontend progress tracker to use research_layer from API**

In `/Users/logannye/erik-website/app/(dashboard)/dashboard/page.tsx`, update the `ResearchTracker` component to accept and use the `research_layer` from the summary instead of computing it client-side.

Replace the progress calculation logic inside `ResearchTracker` with:

```tsx
  // Use authoritative layer from the backend when available
  const backendLayer = summary?.research_layer;

  const layerIndex = backendLayer
    ? ["normal_biology", "als_mechanisms", "erik_specific", "drug_design"].indexOf(backendLayer)
    : evidenceCount < 100 ? 0 : 1;

  const currentStep = Math.max(0, layerIndex);

  // Continuous progress: base from layer + sub-progress from evidence within layer
  const layerBase = currentStep * 25;
  const withinLayerProgress = (() => {
    if (currentStep === 0) return Math.min(24, (evidenceCount / 100) * 24);
    if (currentStep === 1) {
      const log_p = Math.log10(Math.max(100, evidenceCount)) - Math.log10(100);
      const log_max = Math.log10(10000) - Math.log10(100);
      return Math.min(24, (log_p / log_max) * 24);
    }
    if (currentStep === 2) {
      const log_p = Math.log10(Math.max(100, evidenceCount)) - Math.log10(100);
      const log_max = Math.log10(20000) - Math.log10(100);
      return Math.min(24, (log_p / log_max) * 24);
    }
    return 12; // Layer 4: in progress
  })();

  const progress = Math.min(95, Math.max(1, layerBase + withinLayerProgress));
```

Remove the old `geneticTestingNeeded` logic and the old `progress` calculation.

Update the `isBlocked` logic for the Erik's Case step:

```tsx
const isBlocked = i === 2 && currentStep < 2;
```

- [ ] **Step 4: Verify frontend build**

```bash
cd /Users/logannye/erik-website && npm run build 2>&1 | tail -5
```

- [ ] **Step 5: Commit both backend and frontend**

```bash
cd /Users/logannye/.openclaw/erik
git add scripts/api/routers/summary.py
git commit -m "feat: expose research_layer in /api/summary response"

cd /Users/logannye/erik-website
git add lib/types.ts app/\(dashboard\)/dashboard/page.tsx
git commit -m "feat: use backend research_layer for progress tracker"
```

- [ ] **Step 6: Deploy both**

```bash
cd /Users/logannye/.openclaw/erik && railway up
cd /Users/logannye/erik-website && git push origin main && vercel deploy --prod --yes
```
