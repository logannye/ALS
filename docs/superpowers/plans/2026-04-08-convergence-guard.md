# Convergence Guard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent premature reconvergence after layer transitions by requiring a minimum active research duration before allowing convergence.

**Architecture:** Add `min_active_steps_remaining` field to ResearchState. When a layer transition or force_active_research triggers active mode, set this counter to N (default 200). The convergence check at run_loop.py:489 skips convergence while the counter is positive. Decremented each active step.

**Tech Stack:** Python 3.12, dataclass field addition, config key

---

### Task 1: Convergence Guard

**Files:**
- Modify: `scripts/research/state.py` — add `min_active_steps_remaining: int = 0`
- Modify: `scripts/run_loop.py:306-308` — set counter on layer transition
- Modify: `scripts/run_loop.py:289` — set counter on force_active_research
- Modify: `scripts/run_loop.py:484-493` — block convergence while counter > 0
- Modify: `scripts/run_loop.py:496` — decrement counter each step
- Modify: `data/erik_config.json` — add `min_active_steps_after_transition: 200`

- [ ] **Step 1: Add field to ResearchState**

In `scripts/research/state.py`, add alongside `exploration_burst_remaining`:

```python
    min_active_steps_remaining: int = 0
```

And in `to_dict()` if it exists.

- [ ] **Step 2: Set counter on layer transition (run_loop.py ~line 306)**

```python
    if new_layer.value != state.research_layer:
        print(f"[ERIK-MONITOR] ★ LAYER TRANSITION: {state.research_layer} → {new_layer.value}")
        _min_active = cfg.get("min_active_steps_after_transition", 200)
        print(f"[ERIK-MONITOR] Layer transition invalidates convergence — {_min_active} active steps required")
        state = replace(
            state,
            research_layer=new_layer.value,
            converged=False,
            protocol_stable_cycles=0,
            min_active_steps_remaining=_min_active,
        )
```

- [ ] **Step 3: Set counter on force_active_research (run_loop.py ~line 289)**

```python
    if cfg.get("force_active_research", False):
        _min_active = cfg.get("min_active_steps_after_transition", 200)
        print(f"[ERIK-MONITOR] force_active_research=true — {_min_active} active steps required")
        state = replace(state, converged=False, protocol_stable_cycles=0, min_active_steps_remaining=_min_active)
```

- [ ] **Step 4: Block convergence while counter > 0 (run_loop.py ~line 489)**

```python
            _min_remaining = getattr(state, "min_active_steps_remaining", 0)
            if (_stable and _quality) and not state.converged and _min_remaining <= 0:
                state = replace(state, converged=True)
                ...
            elif _min_remaining > 0:
                state = replace(state, min_active_steps_remaining=_min_remaining - 1)
```

- [ ] **Step 5: Add config key**

```json
  "min_active_steps_after_transition": 200,
```

- [ ] **Step 6: Commit and deploy**
