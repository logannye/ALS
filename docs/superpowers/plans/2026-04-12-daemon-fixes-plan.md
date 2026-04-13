# Daemon Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two bugs preventing the cognitive engine from deepening understanding: (1) IntegrationDaemon matches 0 edges because it requires both node names to appear in evidence text, (2) ReasoningDaemon re-analyzes the same edge every cycle because `therapeutic_priority()` stays highest even after confidence drops.

**Architecture:** Both fixes are isolated to their respective daemon files. Integration matching switches from "both endpoints in text" to "any node name in text → find all edges touching that node." Reasoning adds a `last_reasoned_at` cooldown that deprioritizes recently-analyzed edges.

**Tech Stack:** Python 3.12, PostgreSQL (psycopg3), pytest

---

## File Structure

- Modify: `scripts/daemons/integration_daemon.py` — replace `_extract_edge_matches` with entity-name matching
- Modify: `scripts/daemons/reasoning_daemon.py` — add cooldown to `_run_edge_deepening` edge selection
- Modify: `tests/test_integration_daemon.py` — add tests for new matching logic
- Modify: `tests/test_reasoning_daemon.py` — add test for cooldown rotation

---

## Task 1: Fix IntegrationDaemon Entity Matching

**Files:**
- Modify: `scripts/daemons/integration_daemon.py:91-118`
- Modify: `tests/test_integration_daemon.py`

The current `_extract_edge_matches` loads ALL edges, joins both node names, and checks if BOTH names appear in the evidence text. This fails because evidence like `"TARDBP variant classified as Pathogenic"` mentions one gene (TARDBP) but not the other end of any edge (e.g., "TDP-43 aggregation").

**New approach:** Build a lookup of `node_name → [edge_ids]` from the TCG. For each evidence item, find all node names that appear in the text, collect all edges touching those nodes, and return them. This matches when ANY node in an edge is mentioned — far more permissive and correct.

- [ ] **Step 1: Write failing test for entity-name matching**

Add to `tests/test_integration_daemon.py`:

```python
class TestEntityMatching:
    def test_single_node_match_finds_edges(self):
        """Evidence mentioning one node name should match edges containing that node."""
        from daemons.integration_daemon import _build_node_name_index, _match_evidence_to_edges

        # Simulate a node index: node names -> edge IDs
        node_index = {
            "tardbp": [("edge:tardbp->tdp43", 0.9), ("edge:tardbp->fus_interaction", 0.3)],
            "tdp-43": [("edge:tardbp->tdp43", 0.9), ("edge:tdp43->aggregation", 0.5)],
            "sod1": [("edge:sod1->misfolding", 0.7)],
        }

        evidence = {
            "id": "test:1",
            "body": {"claim": "TARDBP variant classified as Pathogenic in ClinVar"},
            "confidence": 0.8,
        }

        matches = _match_evidence_to_edges(evidence, node_index)
        edge_ids = [m[0] for m in matches]
        assert "edge:tardbp->tdp43" in edge_ids
        assert "edge:tardbp->fus_interaction" in edge_ids

    def test_case_insensitive_matching(self):
        from daemons.integration_daemon import _match_evidence_to_edges

        node_index = {
            "riluzole": [("edge:riluzole->eaat2", 0.8)],
        }
        evidence = {
            "id": "test:2",
            "body": {"claim": "Riluzole inhibits glutamate release"},
            "confidence": 0.7,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        assert len(matches) >= 1

    def test_no_match_returns_empty(self):
        from daemons.integration_daemon import _match_evidence_to_edges

        node_index = {
            "tardbp": [("edge:tardbp->tdp43", 0.9)],
        }
        evidence = {
            "id": "test:3",
            "body": {"claim": "Unrelated protein found in kidney tissue"},
            "confidence": 0.5,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        assert len(matches) == 0

    def test_deduplicates_edges(self):
        """If both nodes of an edge appear in text, the edge should appear once, not twice."""
        from daemons.integration_daemon import _match_evidence_to_edges

        node_index = {
            "tardbp": [("edge:tardbp->tdp43", 0.9)],
            "tdp-43": [("edge:tardbp->tdp43", 0.9)],
        }
        evidence = {
            "id": "test:4",
            "body": {"claim": "TARDBP encodes TDP-43 protein"},
            "confidence": 0.9,
        }
        matches = _match_evidence_to_edges(evidence, node_index)
        edge_ids = [m[0] for m in matches]
        assert edge_ids.count("edge:tardbp->tdp43") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_integration_daemon.py::TestEntityMatching -v`
Expected: FAIL — `cannot import name '_build_node_name_index'`

- [ ] **Step 3: Implement entity-name matching**

Replace the `_extract_edge_matches` method and add two new module-level functions in `scripts/daemons/integration_daemon.py`:

```python
def _build_node_name_index() -> dict[str, list[tuple[str, float]]]:
    """Build a lookup: lowercase node name -> list of (edge_id, edge_confidence).

    Each node maps to all edges where it appears as source or target.
    Cached per daemon cycle (called once per integrate_batch).
    """
    index: dict[str, list[tuple[str, float]]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT e.id, e.confidence, sn.name, tn.name
                FROM erik_core.tcg_edges e
                JOIN erik_core.tcg_nodes sn ON e.source_id = sn.id
                JOIN erik_core.tcg_nodes tn ON e.target_id = tn.id
            """)
            for edge_id, conf, src_name, tgt_name in cur.fetchall():
                entry = (edge_id, conf)
                for name in (src_name.lower(), tgt_name.lower()):
                    if name not in index:
                        index[name] = []
                    index[name].append(entry)
    return index


def _match_evidence_to_edges(
    evidence: dict,
    node_index: dict[str, list[tuple[str, float]]],
) -> list[tuple[str, float]]:
    """Match evidence to TCG edges by finding node names in evidence text.

    Returns deduplicated list of (edge_id, evidence_strength) tuples.
    """
    body = evidence.get("body", {})
    claim = body.get("claim", "")
    text = f"{claim} {body.get('notes', '')} {body.get('mechanism', '')}".lower()
    confidence = evidence.get("confidence") or 0.5

    seen_edges: set[str] = set()
    matches: list[tuple[str, float]] = []

    for node_name, edges in node_index.items():
        if node_name in text:
            for edge_id, _edge_conf in edges:
                if edge_id not in seen_edges:
                    seen_edges.add(edge_id)
                    matches.append((edge_id, confidence))

    return matches
```

Then update the `_extract_edge_matches` method in the `IntegrationDaemon` class to use these:

```python
    def _extract_edge_matches(self, evidence: dict, node_index: dict[str, list[tuple[str, float]]]) -> list[tuple[str, float]]:
        """Match evidence to TCG edges by entity name overlap.

        Returns list of (edge_id, evidence_strength) tuples.
        """
        return _match_evidence_to_edges(evidence, node_index)
```

And update `integrate_batch` to build the index once per batch and pass it through. Change line 130-131 from:

```python
        for ev in evidence_items:
            matches = self._extract_edge_matches(ev)
```

to:

```python
        node_index = _build_node_name_index()
        for ev in evidence_items:
            matches = self._extract_edge_matches(ev, node_index)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_integration_daemon.py -v`
Expected: All tests PASS (old tests + 4 new)

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/daemons/integration_daemon.py tests/test_integration_daemon.py && git commit -m "fix(integration): switch to entity-name matching — single node mention finds all connected edges"
```

---

## Task 2: Fix ReasoningDaemon Edge Rotation

**Files:**
- Modify: `scripts/daemons/reasoning_daemon.py:70-77`
- Modify: `tests/test_reasoning_daemon.py`

The current `_run_edge_deepening` selects `max(edges, key=lambda e: e.therapeutic_priority())`. The problem: `edge:vtx002->nuclear_depletion` has `therapeutic_relevance=0.8`, so even at confidence 0.08, its priority is `0.8 * 0.92 = 0.736` — higher than every other edge. It gets selected every cycle indefinitely.

**Fix:** After reasoning about an edge, `last_reasoned_at` is set. Use this as a cooldown — exclude edges reasoned about in the last N seconds (configurable, default 3600s = 1 hour). This forces the daemon to rotate through different edges.

- [ ] **Step 1: Write failing test for cooldown rotation**

Add to `tests/test_reasoning_daemon.py`:

```python
class TestEdgeCooldown:
    def test_recently_reasoned_edges_excluded(self):
        """Edges with last_reasoned_at within cooldown window should be skipped."""
        from daemons.reasoning_daemon import _filter_cooled_down
        from tcg.models import TCGEdge
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        edges = [
            TCGEdge(id="edge:a", source_id="x", target_id="y", edge_type="causes",
                    confidence=0.1, last_reasoned_at=now - timedelta(seconds=60),
                    intervention_potential={"therapeutic_relevance": 0.9}),
            TCGEdge(id="edge:b", source_id="x", target_id="z", edge_type="causes",
                    confidence=0.2,  # last_reasoned_at=None — never reasoned
                    intervention_potential={"therapeutic_relevance": 0.5}),
        ]

        filtered = _filter_cooled_down(edges, cooldown_s=3600)
        assert len(filtered) == 1
        assert filtered[0].id == "edge:b"

    def test_expired_cooldown_included(self):
        """Edges whose cooldown has expired should be included."""
        from daemons.reasoning_daemon import _filter_cooled_down
        from tcg.models import TCGEdge
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        edges = [
            TCGEdge(id="edge:a", source_id="x", target_id="y", edge_type="causes",
                    confidence=0.1, last_reasoned_at=now - timedelta(seconds=7200),
                    intervention_potential={"therapeutic_relevance": 0.9}),
        ]

        filtered = _filter_cooled_down(edges, cooldown_s=3600)
        assert len(filtered) == 1

    def test_all_cooled_down_returns_all(self):
        """If every edge is on cooldown, return all of them (fallback to avoid starvation)."""
        from daemons.reasoning_daemon import _filter_cooled_down
        from tcg.models import TCGEdge
        from datetime import datetime, timezone, timedelta

        now = datetime.now(timezone.utc)
        edges = [
            TCGEdge(id="edge:a", source_id="x", target_id="y", edge_type="causes",
                    confidence=0.1, last_reasoned_at=now - timedelta(seconds=60),
                    intervention_potential={"therapeutic_relevance": 0.9}),
            TCGEdge(id="edge:b", source_id="x", target_id="z", edge_type="causes",
                    confidence=0.2, last_reasoned_at=now - timedelta(seconds=120),
                    intervention_potential={"therapeutic_relevance": 0.5}),
        ]

        filtered = _filter_cooled_down(edges, cooldown_s=3600)
        # All on cooldown → return all (starvation prevention)
        assert len(filtered) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reasoning_daemon.py::TestEdgeCooldown -v`
Expected: FAIL — `cannot import name '_filter_cooled_down'`

- [ ] **Step 3: Implement cooldown filter**

Add this function to `scripts/daemons/reasoning_daemon.py` after the `_select_mode` function (around line 33):

```python
def _filter_cooled_down(
    edges: list[TCGEdge],
    cooldown_s: int = 3600,
) -> list[TCGEdge]:
    """Exclude edges that were reasoned about within the cooldown window.

    If ALL edges are on cooldown, returns the full list to prevent starvation.
    """
    now = datetime.now(timezone.utc)
    available = [
        e for e in edges
        if e.last_reasoned_at is None
        or (now - e.last_reasoned_at).total_seconds() > cooldown_s
    ]
    # Starvation prevention: if everything is on cooldown, return all
    return available if available else edges
```

Then update `_run_edge_deepening` (around line 72-77) to use it. Change:

```python
        edges = self._graph.get_weakest_edges(limit=20)
        if not edges:
            return {"mode": "edge_deepening", "action": "no_weak_edges"}

        # Pick the edge with highest therapeutic priority
        edge = max(edges, key=lambda e: e.therapeutic_priority())
```

to:

```python
        edges = self._graph.get_weakest_edges(limit=20)
        if not edges:
            return {"mode": "edge_deepening", "action": "no_weak_edges"}

        # Filter out recently-reasoned edges (cooldown)
        cfg = ConfigLoader()
        cooldown_s = cfg.get("reasoning_edge_cooldown_s", 3600)
        candidates = _filter_cooled_down(edges, cooldown_s=cooldown_s)

        # Pick the edge with highest therapeutic priority from available candidates
        edge = max(candidates, key=lambda e: e.therapeutic_priority())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reasoning_daemon.py -v`
Expected: All tests PASS (old tests + 3 new)

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/daemons/reasoning_daemon.py tests/test_reasoning_daemon.py && git commit -m "fix(reasoning): add edge cooldown — prevents re-analyzing the same edge every cycle"
```

---

## Task 3: Deploy and Verify

**Files:**
- None (deployment task)

- [ ] **Step 1: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_integration_daemon.py tests/test_reasoning_daemon.py -v`
Expected: All tests PASS

- [ ] **Step 2: Push and deploy**

```bash
cd /Users/logannye/.openclaw/erik && git push origin main && railway up --detach
```

- [ ] **Step 3: Verify new deployment**

Wait ~2 minutes, then:

```bash
railway deployment list 2>&1 | head -3
curl -s --max-time 10 https://erik-api-production.up.railway.app/health
```

Expected: `SUCCESS` deployment, low uptime

- [ ] **Step 4: Monitor daemon output**

```bash
railway logs -n 100 2>&1 | grep -iE "INTEGRATION|REASONING" | head -20
```

Expected:
- Integration: `Batch: 50 items, N edges` where N > 0 (was always 0 before)
- Reasoning: Different edge IDs across cycles (not always `vtx002->nuclear_depletion`)
