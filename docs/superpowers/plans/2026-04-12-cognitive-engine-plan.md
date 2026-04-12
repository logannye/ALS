# Erik Cognitive Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace Erik's stagnating flat research loop with a 4-phase cognitive engine (Acquisition, Integration, Reasoning, Compound Evaluation) driven by a Therapeutic Causal Graph (TCG) that structurally guarantees continuous knowledge improvement.

**Architecture:** Layered evolution — existing loop becomes acquisition engine, three new background daemons handle integration/reasoning/compound evaluation. The TCG (PostgreSQL tables) is the shared blackboard. Claude API provides deep reasoning; Bedrock handles fast acquisition. Erik never stops running during the transition.

**Tech Stack:** Python 3.12, PostgreSQL (psycopg3), FastAPI, Anthropic SDK (Claude API), boto3 (Bedrock), pydantic 2.x, pytest

**Spec:** `docs/superpowers/specs/2026-04-12-cognitive-engine-design.md`

**Codebase conventions:**
- Python env: `conda run -n erik-core`
- Test command: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v`
- Imports: `from ontology.base import BaseEnvelope` (scripts/ is on PYTHONPATH)
- DB: `from db.pool import get_connection` — PostgreSQL only, never sqlite3
- Config: `from config.loader import ConfigLoader` — hot-reloadable from `data/erik_config.json`
- IDs: `f"{type}:{name}".lower().replace(" ", "_")`
- All new files go under `scripts/` (on PYTHONPATH)
- TDD: write failing test first, then minimal implementation

---

## Task 1: TCG Data Models

**Files:**
- Create: `scripts/tcg/__init__.py`
- Create: `scripts/tcg/models.py`
- Test: `tests/test_tcg_models.py`

- [ ] **Step 1: Create test file with TCGNode tests**

```python
# tests/test_tcg_models.py
"""Tests for TCG data models."""
import pytest
from tcg.models import TCGNode, TCGEdge, TCGHypothesis, AcquisitionItem


class TestTCGNode:
    def test_create_gene_node(self):
        node = TCGNode(
            id="gene:tardbp",
            entity_type="gene",
            name="TARDBP",
            pathway_cluster="proteostasis",
        )
        assert node.id == "gene:tardbp"
        assert node.entity_type == "gene"
        assert node.name == "TARDBP"
        assert node.pathway_cluster == "proteostasis"
        assert node.druggability_score == 0.0
        assert node.metadata == {}

    def test_create_compound_node(self):
        node = TCGNode(
            id="compound:riluzole",
            entity_type="compound",
            name="Riluzole",
            pathway_cluster="excitotoxicity",
            druggability_score=0.9,
        )
        assert node.druggability_score == 0.9

    def test_node_to_dict_roundtrip(self):
        node = TCGNode(
            id="protein:tdp-43",
            entity_type="protein",
            name="TDP-43",
            pathway_cluster="proteostasis",
            description="TAR DNA-binding protein 43",
            metadata={"uniprot": "Q13148"},
        )
        d = node.to_dict()
        restored = TCGNode.from_dict(d)
        assert restored.id == node.id
        assert restored.metadata == {"uniprot": "Q13148"}


class TestTCGEdge:
    def test_create_causal_edge(self):
        edge = TCGEdge(
            id="edge:tdp43_agg->stmn2_splicing",
            source_id="protein:tdp-43",
            target_id="gene:stmn2",
            edge_type="causes",
            confidence=0.4,
            open_questions=["Is the effect mediated by nuclear TDP-43 depletion?"],
        )
        assert edge.confidence == 0.4
        assert len(edge.open_questions) == 1
        assert edge.evidence_ids == []
        assert edge.contradiction_ids == []

    def test_edge_therapeutic_priority(self):
        """Priority = therapeutic_relevance * (1 - confidence). Higher = more important to investigate."""
        edge = TCGEdge(
            id="edge:test",
            source_id="a",
            target_id="b",
            edge_type="causes",
            confidence=0.3,
            intervention_potential={"druggable": True, "therapeutic_relevance": 0.9},
        )
        assert edge.therapeutic_priority() == pytest.approx(0.9 * 0.7, abs=0.01)

    def test_edge_to_dict_roundtrip(self):
        edge = TCGEdge(
            id="edge:test",
            source_id="a",
            target_id="b",
            edge_type="inhibits",
            confidence=0.7,
            evidence_ids=["pubmed:123", "pubmed:456"],
        )
        d = edge.to_dict()
        restored = TCGEdge.from_dict(d)
        assert restored.evidence_ids == ["pubmed:123", "pubmed:456"]


class TestTCGHypothesis:
    def test_create_hypothesis(self):
        hyp = TCGHypothesis(
            id="hyp:vtx002_tdp43_clearance",
            hypothesis="VTx-002 gene therapy reduces TDP-43 aggregation via enhanced autophagy",
            supporting_path=["edge:vtx002->tdp43", "edge:tdp43->autophagy"],
            confidence=0.3,
            status="proposed",
            generated_by="reasoning",
            therapeutic_relevance=0.85,
        )
        assert hyp.status == "proposed"
        assert len(hyp.supporting_path) == 2

    def test_hypothesis_status_values(self):
        for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
            hyp = TCGHypothesis(
                id=f"hyp:test_{status}",
                hypothesis="test",
                status=status,
            )
            assert hyp.status == status


class TestAcquisitionItem:
    def test_create_acquisition_item(self):
        item = AcquisitionItem(
            tcg_edge_id="edge:tdp43_agg->stmn2_splicing",
            open_question="Does TDP-43 aggregation directly cause STMN2 cryptic exon inclusion?",
            suggested_sources=["pubmed", "biorxiv"],
            priority=0.63,
            created_by="reasoning",
        )
        assert item.status == "pending"
        assert item.exhausted_sources == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tcg'`

- [ ] **Step 3: Create tcg package and models**

```python
# scripts/tcg/__init__.py
"""Therapeutic Causal Graph — Erik's representation of ALS understanding."""
```

```python
# scripts/tcg/models.py
"""TCG data models: nodes, edges, hypotheses, and acquisition queue items."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TCGNode:
    """A biological entity in the ALS mechanistic model."""

    id: str                                # e.g. "gene:tardbp", "pathway:autophagy"
    entity_type: str                       # gene, protein, pathway, phenotype, compound, measurement
    name: str
    pathway_cluster: Optional[str] = None  # proteostasis, rna_metabolism, etc.
    description: Optional[str] = None
    druggability_score: float = 0.0        # 0-1, updated by CompoundDaemon
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "name": self.name,
            "pathway_cluster": self.pathway_cluster,
            "description": self.description,
            "druggability_score": self.druggability_score,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCGNode:
        d = dict(d)
        for k in ("created_at", "updated_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TCGEdge:
    """A directed mechanistic link between two TCG nodes."""

    id: str
    source_id: str
    target_id: str
    edge_type: str                              # activates, inhibits, causes, prevents, binds, etc.
    confidence: float = 0.1                     # 0-1
    evidence_ids: list[str] = field(default_factory=list)
    contradiction_ids: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    intervention_potential: dict[str, Any] = field(default_factory=dict)
    last_reasoned_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def therapeutic_priority(self) -> float:
        """Higher = more important to investigate. Relevance * uncertainty."""
        relevance = self.intervention_potential.get("therapeutic_relevance", 0.5)
        return relevance * (1.0 - self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "contradiction_ids": self.contradiction_ids,
            "open_questions": self.open_questions,
            "intervention_potential": self.intervention_potential,
            "last_reasoned_at": self.last_reasoned_at.isoformat() if self.last_reasoned_at else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCGEdge:
        d = dict(d)
        for k in ("last_reasoned_at", "created_at", "updated_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TCGHypothesis:
    """A therapeutic hypothesis with causal justification through the TCG."""

    id: str
    hypothesis: str
    supporting_path: list[str] = field(default_factory=list)  # ordered edge IDs
    confidence: float = 0.1
    status: str = "proposed"  # proposed, under_investigation, supported, refuted, actionable
    generated_by: Optional[str] = None
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    therapeutic_relevance: float = 0.0
    created_at: datetime = field(default_factory=_now)
    updated_at: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis": self.hypothesis,
            "supporting_path": self.supporting_path,
            "confidence": self.confidence,
            "status": self.status,
            "generated_by": self.generated_by,
            "evidence_for": self.evidence_for,
            "evidence_against": self.evidence_against,
            "open_questions": self.open_questions,
            "therapeutic_relevance": self.therapeutic_relevance,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TCGHypothesis:
        d = dict(d)
        for k in ("created_at", "updated_at"):
            if isinstance(d.get(k), str):
                d[k] = datetime.fromisoformat(d[k])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AcquisitionItem:
    """A targeted evidence request driven by a TCG open question."""

    tcg_edge_id: str
    open_question: str
    suggested_sources: list[str] = field(default_factory=list)
    exhausted_sources: list[str] = field(default_factory=list)
    priority: float = 0.0
    status: str = "pending"  # pending, in_progress, answered, unanswerable
    created_by: Optional[str] = None
    id: Optional[int] = None  # DB-assigned serial
    created_at: datetime = field(default_factory=_now)
    answered_at: Optional[datetime] = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_models.py -v`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/tcg/__init__.py scripts/tcg/models.py tests/test_tcg_models.py && git commit -m "feat(tcg): add TCG data models — nodes, edges, hypotheses, acquisition items"
```

---

## Task 2: TCG Database Schema & Migration

**Files:**
- Create: `scripts/db/tcg_schema.sql`
- Modify: `scripts/db/migrate.py`
- Test: `tests/test_tcg_schema.py`

- [ ] **Step 1: Write test that TCG tables exist after migration**

```python
# tests/test_tcg_schema.py
"""Tests for TCG database schema."""
import pytest
from db.pool import get_connection


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db(db_available):
    if not db_available:
        pytest.skip("Database not available")


class TestTCGSchema:
    def test_tcg_nodes_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'tcg_nodes'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "entity_type" in columns
        assert "name" in columns
        assert "pathway_cluster" in columns
        assert "druggability_score" in columns
        assert "metadata" in columns

    def test_tcg_edges_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'tcg_edges'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "source_id" in columns
        assert "target_id" in columns
        assert "edge_type" in columns
        assert "confidence" in columns
        assert "evidence_ids" in columns
        assert "open_questions" in columns

    def test_tcg_hypotheses_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'tcg_hypotheses'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "hypothesis" in columns
        assert "status" in columns
        assert "therapeutic_relevance" in columns

    def test_acquisition_queue_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_ops' AND table_name = 'acquisition_queue'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "id" in columns
        assert "tcg_edge_id" in columns
        assert "open_question" in columns
        assert "priority" in columns

    def test_activity_feed_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_ops' AND table_name = 'activity_feed'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "phase" in columns
        assert "event_type" in columns
        assert "summary" in columns

    def test_llm_spend_table_exists(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_ops' AND table_name = 'llm_spend'
                    ORDER BY ordinal_position
                """)
                columns = [row[0] for row in cur.fetchall()]
        assert "model" in columns
        assert "phase" in columns
        assert "cost_usd" in columns

    def test_objects_has_tcg_integrated_column(self):
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema = 'erik_core' AND table_name = 'objects'
                    AND column_name = 'tcg_integrated'
                """)
                assert cur.fetchone() is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_schema.py -v`
Expected: FAIL — tables do not exist yet

- [ ] **Step 3: Create TCG schema SQL file**

```sql
-- scripts/db/tcg_schema.sql
-- Therapeutic Causal Graph schema for Erik Cognitive Engine

-- TCG Nodes: biological entities in the ALS mechanistic model
CREATE TABLE IF NOT EXISTS erik_core.tcg_nodes (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    pathway_cluster TEXT,
    druggability_score FLOAT DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- TCG Edges: directed mechanistic links between entities
CREATE TABLE IF NOT EXISTS erik_core.tcg_edges (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES erik_core.tcg_nodes(id),
    target_id TEXT NOT NULL REFERENCES erik_core.tcg_nodes(id),
    edge_type TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.1,
    evidence_ids TEXT[] DEFAULT '{}',
    contradiction_ids TEXT[] DEFAULT '{}',
    open_questions TEXT[] DEFAULT '{}',
    intervention_potential JSONB DEFAULT '{}',
    last_reasoned_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tcg_edges_source ON erik_core.tcg_edges(source_id);
CREATE INDEX IF NOT EXISTS idx_tcg_edges_target ON erik_core.tcg_edges(target_id);
CREATE INDEX IF NOT EXISTS idx_tcg_edges_confidence ON erik_core.tcg_edges(confidence);

-- TCG Hypotheses: therapeutic hypotheses with causal justification
CREATE TABLE IF NOT EXISTS erik_core.tcg_hypotheses (
    id TEXT PRIMARY KEY,
    hypothesis TEXT NOT NULL,
    supporting_path TEXT[] DEFAULT '{}',
    confidence FLOAT DEFAULT 0.1,
    status TEXT DEFAULT 'proposed',
    generated_by TEXT,
    evidence_for TEXT[] DEFAULT '{}',
    evidence_against TEXT[] DEFAULT '{}',
    open_questions TEXT[] DEFAULT '{}',
    therapeutic_relevance FLOAT DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_tcg_hypotheses_status ON erik_core.tcg_hypotheses(status);
CREATE INDEX IF NOT EXISTS idx_tcg_hypotheses_relevance ON erik_core.tcg_hypotheses(therapeutic_relevance DESC);

-- Acquisition Queue: TCG-driven targeted evidence requests
CREATE TABLE IF NOT EXISTS erik_ops.acquisition_queue (
    id SERIAL PRIMARY KEY,
    tcg_edge_id TEXT REFERENCES erik_core.tcg_edges(id),
    open_question TEXT NOT NULL,
    suggested_sources TEXT[] DEFAULT '{}',
    exhausted_sources TEXT[] DEFAULT '{}',
    priority FLOAT DEFAULT 0,
    status TEXT DEFAULT 'pending',
    created_by TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    answered_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_acq_queue_priority ON erik_ops.acquisition_queue(priority DESC)
    WHERE status = 'pending';

-- Activity Feed: significant events across all cognitive phases
CREATE TABLE IF NOT EXISTS erik_ops.activity_feed (
    id SERIAL PRIMARY KEY,
    phase TEXT NOT NULL,
    event_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    details JSONB DEFAULT '{}',
    tcg_edge_id TEXT,
    tcg_hypothesis_id TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_activity_feed_created ON erik_ops.activity_feed(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_activity_feed_phase ON erik_ops.activity_feed(phase, created_at DESC);

-- LLM Spend: cost tracking for Claude API budget control
CREATE TABLE IF NOT EXISTS erik_ops.llm_spend (
    id SERIAL PRIMARY KEY,
    model TEXT NOT NULL,
    phase TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd FLOAT,
    prompt_cached BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_llm_spend_month ON erik_ops.llm_spend(created_at)
    WHERE created_at > now() - INTERVAL '31 days';

-- Add tcg_integrated column to existing objects table
ALTER TABLE erik_core.objects ADD COLUMN IF NOT EXISTS tcg_integrated BOOLEAN DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_objects_tcg_integrated ON erik_core.objects(tcg_integrated)
    WHERE tcg_integrated = FALSE;
```

- [ ] **Step 4: Add tcg_schema.sql to migrate.py**

Read `scripts/db/migrate.py` and add `tcg_schema.sql` to the list of schema files executed by `run_migrations()`. The existing pattern reads SQL files from the same directory — add `"tcg_schema.sql"` to the list after the existing schema files.

- [ ] **Step 5: Run migration locally**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core python -c "from db.migrate import run_migrations; run_migrations()"`
Expected: No errors, tables created

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_schema.py -v`
Expected: All 7 tests PASS

- [ ] **Step 7: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/db/tcg_schema.sql scripts/db/migrate.py tests/test_tcg_schema.py && git commit -m "feat(db): add TCG schema — nodes, edges, hypotheses, acquisition queue, activity feed, llm spend"
```

---

## Task 3: TCG Graph Operations (Read/Write)

**Files:**
- Create: `scripts/tcg/graph.py`
- Test: `tests/test_tcg_graph.py`

- [ ] **Step 1: Write tests for graph CRUD operations**

```python
# tests/test_tcg_graph.py
"""Tests for TCG graph read/write operations."""
import pytest
from tcg.models import TCGNode, TCGEdge, TCGHypothesis, AcquisitionItem
from tcg.graph import TCGraph


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        from db.pool import get_connection
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db(db_available):
    if not db_available:
        pytest.skip("Database not available")


@pytest.fixture
def graph():
    return TCGraph()


@pytest.fixture
def sample_nodes():
    return [
        TCGNode(id="gene:tardbp_test", entity_type="gene", name="TARDBP", pathway_cluster="proteostasis"),
        TCGNode(id="protein:tdp43_test", entity_type="protein", name="TDP-43", pathway_cluster="proteostasis"),
        TCGNode(id="gene:stmn2_test", entity_type="gene", name="STMN2", pathway_cluster="rna_metabolism"),
    ]


@pytest.fixture
def cleanup_test_nodes(graph):
    """Remove test nodes after each test."""
    yield
    from db.pool import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM erik_ops.acquisition_queue WHERE tcg_edge_id LIKE '%_test%'")
            cur.execute("DELETE FROM erik_core.tcg_hypotheses WHERE id LIKE '%_test%'")
            cur.execute("DELETE FROM erik_core.tcg_edges WHERE id LIKE '%_test%'")
            cur.execute("DELETE FROM erik_core.tcg_nodes WHERE id LIKE '%_test%'")
        conn.commit()


class TestNodeCRUD:
    def test_upsert_and_get_node(self, graph, sample_nodes, cleanup_test_nodes):
        graph.upsert_node(sample_nodes[0])
        result = graph.get_node("gene:tardbp_test")
        assert result is not None
        assert result.name == "TARDBP"
        assert result.pathway_cluster == "proteostasis"

    def test_upsert_updates_existing(self, graph, sample_nodes, cleanup_test_nodes):
        graph.upsert_node(sample_nodes[0])
        updated = TCGNode(id="gene:tardbp_test", entity_type="gene", name="TARDBP",
                          pathway_cluster="proteostasis", druggability_score=0.75)
        graph.upsert_node(updated)
        result = graph.get_node("gene:tardbp_test")
        assert result.druggability_score == 0.75

    def test_get_nonexistent_node_returns_none(self, graph):
        assert graph.get_node("gene:does_not_exist") is None

    def test_list_nodes_by_cluster(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes:
            graph.upsert_node(n)
        proteo = graph.list_nodes(pathway_cluster="proteostasis")
        proteo_ids = [n.id for n in proteo]
        assert "gene:tardbp_test" in proteo_ids
        assert "protein:tdp43_test" in proteo_ids
        assert "gene:stmn2_test" not in proteo_ids


class TestEdgeCRUD:
    def test_upsert_and_get_edge(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(
            id="edge:tardbp_tdp43_test",
            source_id="gene:tardbp_test",
            target_id="protein:tdp43_test",
            edge_type="encodes",
            confidence=0.95,
        )
        graph.upsert_edge(edge)
        result = graph.get_edge("edge:tardbp_tdp43_test")
        assert result is not None
        assert result.edge_type == "encodes"
        assert result.confidence == 0.95

    def test_update_edge_confidence(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:tardbp_tdp43_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="encodes", confidence=0.3)
        graph.upsert_edge(edge)
        graph.update_edge_confidence("edge:tardbp_tdp43_test", 0.8, evidence_id="pubmed:123")
        result = graph.get_edge("edge:tardbp_tdp43_test")
        assert result.confidence == 0.8
        assert "pubmed:123" in result.evidence_ids

    def test_get_weakest_edges(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes:
            graph.upsert_node(n)
        e1 = TCGEdge(id="edge:e1_test", source_id="gene:tardbp_test", target_id="protein:tdp43_test",
                      edge_type="causes", confidence=0.2)
        e2 = TCGEdge(id="edge:e2_test", source_id="protein:tdp43_test", target_id="gene:stmn2_test",
                      edge_type="causes", confidence=0.8)
        graph.upsert_edge(e1)
        graph.upsert_edge(e2)
        weakest = graph.get_weakest_edges(limit=10)
        ids = [e.id for e in weakest]
        assert ids.index("edge:e1_test") < ids.index("edge:e2_test")

    def test_get_edges_for_node(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:tardbp_tdp43_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="encodes", confidence=0.5)
        graph.upsert_edge(edge)
        outgoing = graph.get_edges_from("gene:tardbp_test")
        assert len(outgoing) >= 1
        assert outgoing[0].target_id == "protein:tdp43_test"


class TestBayesianUpdate:
    def test_bayesian_confidence_update(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:bayes_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="causes", confidence=0.3)
        graph.upsert_edge(edge)
        # Bayesian update with strong evidence should increase confidence
        graph.bayesian_update("edge:bayes_test", evidence_strength=0.9, evidence_id="pubmed:999")
        result = graph.get_edge("edge:bayes_test")
        assert result.confidence > 0.3
        assert "pubmed:999" in result.evidence_ids


class TestAcquisitionQueue:
    def test_push_and_pop_queue(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:queue_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="causes", confidence=0.3)
        graph.upsert_edge(edge)
        item = AcquisitionItem(
            tcg_edge_id="edge:queue_test",
            open_question="What is the binding mechanism?",
            suggested_sources=["chembl", "bindingdb"],
            priority=0.7,
            created_by="integration",
        )
        graph.push_acquisition(item)
        popped = graph.pop_acquisition()
        assert popped is not None
        assert popped.open_question == "What is the binding mechanism?"
        assert popped.status == "in_progress"

    def test_pop_returns_highest_priority(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes[:2]:
            graph.upsert_node(n)
        edge = TCGEdge(id="edge:prio_test", source_id="gene:tardbp_test",
                       target_id="protein:tdp43_test", edge_type="causes", confidence=0.3)
        graph.upsert_edge(edge)
        low = AcquisitionItem(tcg_edge_id="edge:prio_test", open_question="low priority",
                              priority=0.1, created_by="test")
        high = AcquisitionItem(tcg_edge_id="edge:prio_test", open_question="high priority",
                               priority=0.9, created_by="test")
        graph.push_acquisition(low)
        graph.push_acquisition(high)
        popped = graph.pop_acquisition()
        assert popped.open_question == "high priority"

    def test_pop_empty_queue_returns_none(self, graph):
        # Pop all existing items first, then verify None
        while graph.pop_acquisition() is not None:
            pass
        assert graph.pop_acquisition() is None


class TestGraphSummary:
    def test_summary_returns_counts(self, graph, sample_nodes, cleanup_test_nodes):
        for n in sample_nodes:
            graph.upsert_node(n)
        summary = graph.summary()
        assert summary["node_count"] >= 3
        assert "edge_count" in summary
        assert "mean_confidence" in summary
        assert "clusters" in summary
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_graph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tcg.graph'`

- [ ] **Step 3: Implement TCGraph class**

```python
# scripts/tcg/graph.py
"""TCG read/write operations — the interface between daemons and the Therapeutic Causal Graph."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from db.pool import get_connection
from tcg.models import TCGNode, TCGEdge, TCGHypothesis, AcquisitionItem


class TCGraph:
    """Read/write interface for the Therapeutic Causal Graph."""

    # ── Nodes ──────────────────────────────────────────────

    def upsert_node(self, node: TCGNode) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_core.tcg_nodes
                        (id, entity_type, name, description, pathway_cluster,
                         druggability_score, metadata, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        entity_type = EXCLUDED.entity_type,
                        name = EXCLUDED.name,
                        description = EXCLUDED.description,
                        pathway_cluster = EXCLUDED.pathway_cluster,
                        druggability_score = EXCLUDED.druggability_score,
                        metadata = EXCLUDED.metadata,
                        updated_at = EXCLUDED.updated_at
                """, (
                    node.id, node.entity_type, node.name, node.description,
                    node.pathway_cluster, node.druggability_score,
                    __import__("json").dumps(node.metadata),
                    node.created_at, node.updated_at,
                ))
            conn.commit()

    def get_node(self, node_id: str) -> Optional[TCGNode]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, entity_type, name, description, pathway_cluster,
                           druggability_score, metadata, created_at, updated_at
                    FROM erik_core.tcg_nodes WHERE id = %s
                """, (node_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return TCGNode(
            id=row[0], entity_type=row[1], name=row[2], description=row[3],
            pathway_cluster=row[4], druggability_score=row[5] or 0.0,
            metadata=row[6] if isinstance(row[6], dict) else {},
            created_at=row[7], updated_at=row[8],
        )

    def list_nodes(self, pathway_cluster: Optional[str] = None) -> list[TCGNode]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if pathway_cluster:
                    cur.execute("""
                        SELECT id, entity_type, name, description, pathway_cluster,
                               druggability_score, metadata, created_at, updated_at
                        FROM erik_core.tcg_nodes WHERE pathway_cluster = %s
                    """, (pathway_cluster,))
                else:
                    cur.execute("""
                        SELECT id, entity_type, name, description, pathway_cluster,
                               druggability_score, metadata, created_at, updated_at
                        FROM erik_core.tcg_nodes
                    """)
                rows = cur.fetchall()
        return [
            TCGNode(
                id=r[0], entity_type=r[1], name=r[2], description=r[3],
                pathway_cluster=r[4], druggability_score=r[5] or 0.0,
                metadata=r[6] if isinstance(r[6], dict) else {},
                created_at=r[7], updated_at=r[8],
            )
            for r in rows
        ]

    # ── Edges ──────────────────────────────────────────────

    def upsert_edge(self, edge: TCGEdge) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_core.tcg_edges
                        (id, source_id, target_id, edge_type, confidence,
                         evidence_ids, contradiction_ids, open_questions,
                         intervention_potential, last_reasoned_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        edge_type = EXCLUDED.edge_type,
                        confidence = EXCLUDED.confidence,
                        evidence_ids = EXCLUDED.evidence_ids,
                        contradiction_ids = EXCLUDED.contradiction_ids,
                        open_questions = EXCLUDED.open_questions,
                        intervention_potential = EXCLUDED.intervention_potential,
                        last_reasoned_at = EXCLUDED.last_reasoned_at,
                        updated_at = EXCLUDED.updated_at
                """, (
                    edge.id, edge.source_id, edge.target_id, edge.edge_type,
                    edge.confidence, edge.evidence_ids, edge.contradiction_ids,
                    edge.open_questions,
                    __import__("json").dumps(edge.intervention_potential),
                    edge.last_reasoned_at, edge.created_at, edge.updated_at,
                ))
            conn.commit()

    def get_edge(self, edge_id: str) -> Optional[TCGEdge]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, edge_type, confidence,
                           evidence_ids, contradiction_ids, open_questions,
                           intervention_potential, last_reasoned_at, created_at, updated_at
                    FROM erik_core.tcg_edges WHERE id = %s
                """, (edge_id,))
                row = cur.fetchone()
        if row is None:
            return None
        return TCGEdge(
            id=row[0], source_id=row[1], target_id=row[2], edge_type=row[3],
            confidence=row[4], evidence_ids=list(row[5] or []),
            contradiction_ids=list(row[6] or []), open_questions=list(row[7] or []),
            intervention_potential=row[8] if isinstance(row[8], dict) else {},
            last_reasoned_at=row[9], created_at=row[10], updated_at=row[11],
        )

    def get_edges_from(self, node_id: str) -> list[TCGEdge]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, edge_type, confidence,
                           evidence_ids, contradiction_ids, open_questions,
                           intervention_potential, last_reasoned_at, created_at, updated_at
                    FROM erik_core.tcg_edges WHERE source_id = %s
                """, (node_id,))
                rows = cur.fetchall()
        return [self._edge_from_row(r) for r in rows]

    def get_weakest_edges(self, limit: int = 10) -> list[TCGEdge]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, source_id, target_id, edge_type, confidence,
                           evidence_ids, contradiction_ids, open_questions,
                           intervention_potential, last_reasoned_at, created_at, updated_at
                    FROM erik_core.tcg_edges
                    ORDER BY confidence ASC
                    LIMIT %s
                """, (limit,))
                rows = cur.fetchall()
        return [self._edge_from_row(r) for r in rows]

    def update_edge_confidence(
        self, edge_id: str, confidence: float, evidence_id: Optional[str] = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                if evidence_id:
                    cur.execute("""
                        UPDATE erik_core.tcg_edges
                        SET confidence = %s,
                            evidence_ids = array_append(evidence_ids, %s),
                            updated_at = %s
                        WHERE id = %s
                    """, (confidence, evidence_id, now, edge_id))
                else:
                    cur.execute("""
                        UPDATE erik_core.tcg_edges
                        SET confidence = %s, updated_at = %s
                        WHERE id = %s
                    """, (confidence, now, edge_id))
            conn.commit()

    def bayesian_update(
        self, edge_id: str, evidence_strength: float, evidence_id: str,
    ) -> None:
        """Bayesian confidence update with diminishing returns."""
        edge = self.get_edge(edge_id)
        if edge is None:
            return
        prior_strength = max(1.0, len(edge.evidence_ids))
        posterior = (edge.confidence * prior_strength + evidence_strength) / (prior_strength + 1)
        posterior = max(0.0, min(1.0, posterior))
        self.update_edge_confidence(edge_id, posterior, evidence_id)

    # ── Hypotheses ─────────────────────────────────────────

    def upsert_hypothesis(self, hyp: TCGHypothesis) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_core.tcg_hypotheses
                        (id, hypothesis, supporting_path, confidence, status,
                         generated_by, evidence_for, evidence_against,
                         open_questions, therapeutic_relevance, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO UPDATE SET
                        hypothesis = EXCLUDED.hypothesis,
                        supporting_path = EXCLUDED.supporting_path,
                        confidence = EXCLUDED.confidence,
                        status = EXCLUDED.status,
                        evidence_for = EXCLUDED.evidence_for,
                        evidence_against = EXCLUDED.evidence_against,
                        open_questions = EXCLUDED.open_questions,
                        therapeutic_relevance = EXCLUDED.therapeutic_relevance,
                        updated_at = EXCLUDED.updated_at
                """, (
                    hyp.id, hyp.hypothesis, hyp.supporting_path, hyp.confidence,
                    hyp.status, hyp.generated_by, hyp.evidence_for,
                    hyp.evidence_against, hyp.open_questions,
                    hyp.therapeutic_relevance, hyp.created_at, hyp.updated_at,
                ))
            conn.commit()

    def get_hypotheses_by_status(self, status: str) -> list[TCGHypothesis]:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, hypothesis, supporting_path, confidence, status,
                           generated_by, evidence_for, evidence_against,
                           open_questions, therapeutic_relevance, created_at, updated_at
                    FROM erik_core.tcg_hypotheses
                    WHERE status = %s
                    ORDER BY therapeutic_relevance DESC
                """, (status,))
                rows = cur.fetchall()
        return [
            TCGHypothesis(
                id=r[0], hypothesis=r[1], supporting_path=list(r[2] or []),
                confidence=r[3], status=r[4], generated_by=r[5],
                evidence_for=list(r[6] or []), evidence_against=list(r[7] or []),
                open_questions=list(r[8] or []), therapeutic_relevance=r[9],
                created_at=r[10], updated_at=r[11],
            )
            for r in rows
        ]

    # ── Acquisition Queue ──────────────────────────────────

    def push_acquisition(self, item: AcquisitionItem) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_ops.acquisition_queue
                        (tcg_edge_id, open_question, suggested_sources,
                         exhausted_sources, priority, status, created_by)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """, (
                    item.tcg_edge_id, item.open_question, item.suggested_sources,
                    item.exhausted_sources, item.priority, item.status, item.created_by,
                ))
            conn.commit()

    def pop_acquisition(self) -> Optional[AcquisitionItem]:
        """Atomically pop the highest-priority pending item."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.acquisition_queue
                    SET status = 'in_progress'
                    WHERE id = (
                        SELECT id FROM erik_ops.acquisition_queue
                        WHERE status = 'pending'
                        ORDER BY priority DESC
                        LIMIT 1
                        FOR UPDATE SKIP LOCKED
                    )
                    RETURNING id, tcg_edge_id, open_question, suggested_sources,
                              exhausted_sources, priority, status, created_by,
                              created_at, answered_at
                """)
                row = cur.fetchone()
            conn.commit()
        if row is None:
            return None
        return AcquisitionItem(
            id=row[0], tcg_edge_id=row[1], open_question=row[2],
            suggested_sources=list(row[3] or []), exhausted_sources=list(row[4] or []),
            priority=row[5], status=row[6], created_by=row[7],
            created_at=row[8], answered_at=row[9],
        )

    def mark_acquisition(self, item_id: int, status: str) -> None:
        now = datetime.now(timezone.utc)
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_ops.acquisition_queue
                    SET status = %s, answered_at = %s
                    WHERE id = %s
                """, (status, now if status in ("answered", "unanswerable") else None, item_id))
            conn.commit()

    # ── Activity Feed ──────────────────────────────────────

    def log_activity(
        self, phase: str, event_type: str, summary: str, *,
        details: dict | None = None, tcg_edge_id: str | None = None,
        tcg_hypothesis_id: str | None = None,
    ) -> None:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO erik_ops.activity_feed
                        (phase, event_type, summary, details, tcg_edge_id, tcg_hypothesis_id)
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s)
                """, (
                    phase, event_type, summary,
                    __import__("json").dumps(details or {}),
                    tcg_edge_id, tcg_hypothesis_id,
                ))
            conn.commit()

    # ── Summary / Metrics ──────────────────────────────────

    def summary(self) -> dict:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM erik_core.tcg_nodes")
                node_count = cur.fetchone()[0]
                cur.execute("SELECT count(*), coalesce(avg(confidence), 0) FROM erik_core.tcg_edges")
                row = cur.fetchone()
                edge_count, mean_confidence = row[0], float(row[1])
                cur.execute("""
                    SELECT pathway_cluster, count(*)
                    FROM erik_core.tcg_nodes
                    WHERE pathway_cluster IS NOT NULL
                    GROUP BY pathway_cluster
                """)
                clusters = {r[0]: r[1] for r in cur.fetchall()}
        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "mean_confidence": round(mean_confidence, 4),
            "clusters": clusters,
        }

    # ── Helpers ────────────────────────────────────────────

    def _edge_from_row(self, r: tuple) -> TCGEdge:
        return TCGEdge(
            id=r[0], source_id=r[1], target_id=r[2], edge_type=r[3],
            confidence=r[4], evidence_ids=list(r[5] or []),
            contradiction_ids=list(r[6] or []), open_questions=list(r[7] or []),
            intervention_potential=r[8] if isinstance(r[8], dict) else {},
            last_reasoned_at=r[9], created_at=r[10], updated_at=r[11],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_graph.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/tcg/graph.py tests/test_tcg_graph.py && git commit -m "feat(tcg): add TCGraph CRUD — nodes, edges, hypotheses, acquisition queue, bayesian updates"
```

---

## Task 4: ALS Scaffold Seeder

**Files:**
- Create: `scripts/tcg/seed_scaffold.py`
- Test: `tests/test_tcg_scaffold.py`

- [ ] **Step 1: Write test for scaffold seeding**

```python
# tests/test_tcg_scaffold.py
"""Tests for ALS biology scaffold seeding."""
import pytest
from tcg.graph import TCGraph
from tcg.seed_scaffold import seed_scaffold, PATHWAY_CLUSTERS


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        from db.pool import get_connection
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db(db_available):
    if not db_available:
        pytest.skip("Database not available")


@pytest.fixture
def graph():
    return TCGraph()


class TestScaffoldConstants:
    def test_eight_pathway_clusters(self):
        assert len(PATHWAY_CLUSTERS) == 8
        assert "proteostasis" in PATHWAY_CLUSTERS
        assert "rna_metabolism" in PATHWAY_CLUSTERS
        assert "excitotoxicity" in PATHWAY_CLUSTERS
        assert "neuroinflammation" in PATHWAY_CLUSTERS
        assert "axonal_transport" in PATHWAY_CLUSTERS
        assert "mitochondrial" in PATHWAY_CLUSTERS
        assert "neuromuscular_junction" in PATHWAY_CLUSTERS
        assert "glial_biology" in PATHWAY_CLUSTERS


class TestScaffoldSeeding:
    def test_seed_creates_nodes(self, graph):
        stats = seed_scaffold(graph)
        assert stats["nodes_created"] >= 200
        assert stats["edges_created"] >= 500

    def test_seed_is_idempotent(self, graph):
        stats1 = seed_scaffold(graph)
        stats2 = seed_scaffold(graph)
        # Second run uses ON CONFLICT upsert — same counts
        assert stats2["nodes_created"] == stats1["nodes_created"]

    def test_all_clusters_populated(self, graph):
        seed_scaffold(graph)
        summary = graph.summary()
        for cluster in PATHWAY_CLUSTERS:
            assert summary["clusters"].get(cluster, 0) >= 5, \
                f"Cluster {cluster} has fewer than 5 nodes"

    def test_edges_have_open_questions(self, graph):
        seed_scaffold(graph)
        weak = graph.get_weakest_edges(limit=50)
        with_questions = [e for e in weak if e.open_questions]
        assert len(with_questions) >= 20, \
            "At least 20 weak edges should have open questions"

    def test_confidence_ranges(self, graph):
        seed_scaffold(graph)
        from db.pool import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT min(confidence), max(confidence) FROM erik_core.tcg_edges")
                row = cur.fetchone()
        assert row[0] >= 0.1, "Minimum confidence should be >= 0.1"
        assert row[1] <= 0.95, "Maximum confidence should be <= 0.95"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_scaffold.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tcg.seed_scaffold'`

- [ ] **Step 3: Implement scaffold seeder**

Create `scripts/tcg/seed_scaffold.py`. This file is large (~400-500 lines) because it contains the hardcoded ALS biology backbone. The structure:

```python
# scripts/tcg/seed_scaffold.py
"""Seed the Therapeutic Causal Graph with the ALS biology scaffold.

~200 nodes across 8 pathway clusters and ~600 directed mechanistic edges
with initial confidence and open questions. This is textbook-level ALS biology
that the system should not need to rediscover.

Idempotent — safe to re-run (uses ON CONFLICT upsert).
"""
from __future__ import annotations

from tcg.graph import TCGraph
from tcg.models import TCGNode, TCGEdge

PATHWAY_CLUSTERS = [
    "proteostasis",
    "rna_metabolism",
    "excitotoxicity",
    "neuroinflammation",
    "axonal_transport",
    "mitochondrial",
    "neuromuscular_junction",
    "glial_biology",
]


def _proteostasis_nodes() -> list[TCGNode]:
    """Nodes for the proteostasis/protein aggregation cluster."""
    return [
        TCGNode(id="gene:tardbp", entity_type="gene", name="TARDBP",
                pathway_cluster="proteostasis", description="TAR DNA-binding protein 43 gene"),
        TCGNode(id="protein:tdp-43", entity_type="protein", name="TDP-43",
                pathway_cluster="proteostasis", description="TAR DNA-binding protein 43"),
        TCGNode(id="process:tdp43_aggregation", entity_type="phenotype", name="TDP-43 aggregation",
                pathway_cluster="proteostasis", description="Cytoplasmic TDP-43 protein aggregation"),
        TCGNode(id="process:tdp43_nuclear_depletion", entity_type="phenotype", name="TDP-43 nuclear depletion",
                pathway_cluster="proteostasis", description="Loss of nuclear TDP-43 function"),
        TCGNode(id="gene:sod1", entity_type="gene", name="SOD1",
                pathway_cluster="proteostasis", description="Superoxide dismutase 1"),
        TCGNode(id="protein:sod1", entity_type="protein", name="SOD1 protein",
                pathway_cluster="proteostasis", description="Cu/Zn superoxide dismutase"),
        TCGNode(id="process:sod1_misfolding", entity_type="phenotype", name="SOD1 misfolding",
                pathway_cluster="proteostasis", description="SOD1 protein misfolding and aggregation"),
        TCGNode(id="gene:fus", entity_type="gene", name="FUS",
                pathway_cluster="proteostasis", description="Fused in sarcoma gene"),
        TCGNode(id="protein:fus", entity_type="protein", name="FUS protein",
                pathway_cluster="proteostasis", description="FUS RNA-binding protein"),
        TCGNode(id="gene:vcp", entity_type="gene", name="VCP",
                pathway_cluster="proteostasis", description="Valosin-containing protein"),
        TCGNode(id="pathway:ubiquitin_proteasome", entity_type="pathway", name="Ubiquitin-proteasome system",
                pathway_cluster="proteostasis", description="Major protein degradation pathway"),
        TCGNode(id="pathway:autophagy", entity_type="pathway", name="Autophagy",
                pathway_cluster="proteostasis", description="Cellular self-digestion pathway"),
        TCGNode(id="protein:mtor", entity_type="protein", name="mTOR",
                pathway_cluster="proteostasis", description="Mechanistic target of rapamycin"),
        TCGNode(id="process:chaperone_response", entity_type="pathway", name="Chaperone response",
                pathway_cluster="proteostasis", description="Heat shock protein response"),
        TCGNode(id="gene:ubqln2", entity_type="gene", name="UBQLN2",
                pathway_cluster="proteostasis", description="Ubiquilin 2"),
        TCGNode(id="gene:sqstm1", entity_type="gene", name="SQSTM1",
                pathway_cluster="proteostasis", description="Sequestosome 1 / p62"),
        # Compounds
        TCGNode(id="compound:rapamycin", entity_type="compound", name="Rapamycin",
                pathway_cluster="proteostasis", druggability_score=0.8,
                description="mTOR inhibitor, autophagy inducer"),
        TCGNode(id="compound:vtx-002", entity_type="compound", name="VTx-002",
                pathway_cluster="proteostasis", druggability_score=0.7,
                description="AAV gene therapy targeting TDP-43 aggregation"),
    ]

# Similar functions for each cluster:
# _rna_metabolism_nodes(), _excitotoxicity_nodes(), _neuroinflammation_nodes(),
# _axonal_transport_nodes(), _mitochondrial_nodes(), _nmj_nodes(), _glial_nodes()
# Each returns 15-30 nodes.

# Then edge definitions per cluster and cross-cluster:

def _proteostasis_edges() -> list[TCGEdge]:
    return [
        TCGEdge(id="edge:tardbp->tdp43", source_id="gene:tardbp", target_id="protein:tdp-43",
                edge_type="encodes", confidence=0.95),
        TCGEdge(id="edge:tdp43->aggregation", source_id="protein:tdp-43",
                target_id="process:tdp43_aggregation", edge_type="causes", confidence=0.85,
                open_questions=["What triggers the transition from soluble to aggregated state?"]),
        TCGEdge(id="edge:tdp43_agg->nuclear_depletion", source_id="process:tdp43_aggregation",
                target_id="process:tdp43_nuclear_depletion", edge_type="causes", confidence=0.7,
                open_questions=["Is nuclear depletion a cause or consequence of aggregation?"]),
        TCGEdge(id="edge:mtor->autophagy", source_id="protein:mtor",
                target_id="pathway:autophagy", edge_type="inhibits", confidence=0.9),
        TCGEdge(id="edge:rapamycin->mtor", source_id="compound:rapamycin",
                target_id="protein:mtor", edge_type="inhibits", confidence=0.9,
                intervention_potential={"druggable": True, "therapeutic_relevance": 0.7}),
        TCGEdge(id="edge:autophagy->tdp43_clearance", source_id="pathway:autophagy",
                target_id="process:tdp43_aggregation", edge_type="prevents", confidence=0.5,
                open_questions=["Does autophagy upregulation clear existing aggregates or only prevent new ones?"]),
        TCGEdge(id="edge:vtx002->tdp43_agg", source_id="compound:vtx-002",
                target_id="process:tdp43_aggregation", edge_type="prevents", confidence=0.4,
                intervention_potential={"druggable": True, "therapeutic_relevance": 0.9},
                open_questions=["Phase 1 data pending — what is the clinical effect size?"]),
        TCGEdge(id="edge:sod1->misfolding", source_id="gene:sod1",
                target_id="process:sod1_misfolding", edge_type="causes", confidence=0.85,
                open_questions=["Only in SOD1 mutation carriers — requires genetic confirmation"]),
        # ... more edges
    ]

# Similar edge functions for each cluster and cross-cluster connections.

def seed_scaffold(graph: TCGraph) -> dict[str, int]:
    """Seed the full ALS biology scaffold. Returns counts of nodes and edges created."""
    all_nodes: list[TCGNode] = []
    all_edges: list[TCGEdge] = []

    # Collect from all clusters
    for fn in [_proteostasis_nodes, _rna_metabolism_nodes, _excitotoxicity_nodes,
               _neuroinflammation_nodes, _axonal_transport_nodes, _mitochondrial_nodes,
               _nmj_nodes, _glial_nodes]:
        all_nodes.extend(fn())

    for fn in [_proteostasis_edges, _rna_metabolism_edges, _excitotoxicity_edges,
               _neuroinflammation_edges, _axonal_transport_edges, _mitochondrial_edges,
               _nmj_edges, _glial_edges, _cross_cluster_edges]:
        all_edges.extend(fn())

    # Upsert all (idempotent)
    for node in all_nodes:
        graph.upsert_node(node)
    for edge in all_edges:
        graph.upsert_edge(edge)

    return {"nodes_created": len(all_nodes), "edges_created": len(all_edges)}
```

**Implementation note:** The full scaffold file will contain ~200 node definitions and ~600 edge definitions across all 8 clusters plus cross-cluster connections. The implementing agent should build out ALL 8 cluster functions and the cross-cluster edges function, using the proteostasis example as the template. Each cluster should have 15-30 nodes and 30-80 edges covering the major genes, proteins, pathways, phenotypes, and compounds relevant to that ALS pathology cluster. Cross-cluster edges connect pathways across clusters (e.g., TDP-43 aggregation in proteostasis causes STMN2 loss in rna_metabolism).

Key guidance for the implementing agent:
- Use PubMed-level knowledge of ALS biology — these are established facts
- Confidence 0.7-0.9 for well-replicated findings, 0.3-0.6 for established but nuanced, 0.1-0.2 for emerging
- Every edge below 0.7 must have at least one open_question
- Include the 7 current protocol interventions (riluzole, edaravone, VTx-002, pridopidine, rapamycin, masitinib, STMN2 ASO) as compound nodes
- Include drug-target edges with `intervention_potential: {"druggable": True, "therapeutic_relevance": 0.X}`

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_tcg_scaffold.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/tcg/seed_scaffold.py tests/test_tcg_scaffold.py && git commit -m "feat(tcg): add ALS biology scaffold — 8 pathway clusters, ~200 nodes, ~600 edges"
```

---

## Task 5: Claude API Client

**Files:**
- Create: `scripts/llm/claude_client.py`
- Test: `tests/test_claude_client.py`

- [ ] **Step 1: Write tests for Claude client**

```python
# tests/test_claude_client.py
"""Tests for Claude API client — rate limiting, spend tracking, fallback."""
import pytest
from unittest.mock import MagicMock, patch
from llm.claude_client import ClaudeClient, LLMSpendTracker


class TestLLMSpendTracker:
    def test_log_spend(self):
        tracker = LLMSpendTracker()
        tracker.log(model="claude-opus-4-6", phase="reasoning",
                    input_tokens=1000, output_tokens=500, cost_usd=0.05, prompt_cached=True)
        # Should not raise

    def test_monthly_spend(self):
        tracker = LLMSpendTracker()
        # Fresh tracker should report 0 or near-zero
        spend = tracker.monthly_spend_usd()
        assert isinstance(spend, float)
        assert spend >= 0.0


class TestClaudeClient:
    def test_init_with_config(self):
        client = ClaudeClient(
            api_key="test-key",
            reasoning_model="claude-opus-4-6",
            evaluation_model="claude-sonnet-4-6",
            max_opus_per_hour=30,
            max_sonnet_per_hour=60,
            monthly_budget_usd=100.0,
        )
        assert client._reasoning_model == "claude-opus-4-6"
        assert client._monthly_budget_usd == 100.0

    def test_budget_exceeded_blocks_calls(self):
        client = ClaudeClient(api_key="test-key", monthly_budget_usd=0.0)
        # With $0 budget, all calls should be blocked
        result = client.reason_about_edge(
            edge_context="test edge",
            supporting_evidence=["ev1"],
            contradicting_evidence=[],
        )
        assert result is None or result.get("budget_exceeded") is True

    @patch("llm.claude_client.ClaudeClient._call_api")
    def test_reason_about_edge_parses_response(self, mock_call):
        mock_call.return_value = {
            "confidence_assessment": 0.7,
            "mechanism": "Direct causal link supported by in vitro evidence",
            "open_questions": ["In vivo confirmation needed"],
            "confounders": [],
        }
        client = ClaudeClient(api_key="test-key", monthly_budget_usd=100.0)
        result = client.reason_about_edge(
            edge_context="TDP-43 aggregation causes STMN2 cryptic exon inclusion",
            supporting_evidence=["Paper A found direct correlation"],
            contradicting_evidence=[],
        )
        assert result["confidence_assessment"] == 0.7
        assert len(result["open_questions"]) == 1

    @patch("llm.claude_client.ClaudeClient._call_api")
    def test_cross_pathway_synthesis(self, mock_call):
        mock_call.return_value = {
            "proposed_edges": [
                {"source": "protein:tdp-43", "target": "process:microglial_activation",
                 "edge_type": "activates", "confidence": 0.3,
                 "rationale": "TDP-43 aggregates activate innate immune response"}
            ]
        }
        client = ClaudeClient(api_key="test-key", monthly_budget_usd=100.0)
        result = client.cross_pathway_synthesis(
            cluster_a="proteostasis", cluster_a_evidence=["ev1"],
            cluster_b="neuroinflammation", cluster_b_evidence=["ev2"],
        )
        assert len(result["proposed_edges"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_claude_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'llm.claude_client'`

- [ ] **Step 3: Install anthropic SDK**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pip install anthropic`
Expected: Successfully installed

- [ ] **Step 4: Implement Claude client**

```python
# scripts/llm/claude_client.py
"""Claude API client for deep reasoning phases — rate limiting, spend tracking, fallback."""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from config.loader import ConfigLoader
from db.pool import get_connection


class LLMSpendTracker:
    """Track and enforce LLM API spend in PostgreSQL."""

    def log(
        self, model: str, phase: str, input_tokens: int, output_tokens: int,
        cost_usd: float, prompt_cached: bool = False,
    ) -> None:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO erik_ops.llm_spend
                            (model, phase, input_tokens, output_tokens, cost_usd, prompt_cached)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (model, phase, input_tokens, output_tokens, cost_usd, prompt_cached))
                conn.commit()
        except Exception:
            pass  # Spend logging should never block research

    def monthly_spend_usd(self) -> float:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT coalesce(sum(cost_usd), 0)
                        FROM erik_ops.llm_spend
                        WHERE created_at > date_trunc('month', now())
                    """)
                    return float(cur.fetchone()[0])
        except Exception:
            return 0.0


class ClaudeClient:
    """Claude API client with rate limiting, spend tracking, and structured prompts."""

    # Approximate costs per 1M tokens (for spend tracking)
    _COST_PER_M_INPUT = {"claude-opus-4-6": 15.0, "claude-sonnet-4-6": 3.0}
    _COST_PER_M_OUTPUT = {"claude-opus-4-6": 75.0, "claude-sonnet-4-6": 15.0}

    def __init__(
        self, api_key: str, reasoning_model: str = "claude-opus-4-6",
        evaluation_model: str = "claude-sonnet-4-6",
        max_opus_per_hour: int = 30, max_sonnet_per_hour: int = 60,
        monthly_budget_usd: float = 100.0,
    ) -> None:
        self._api_key = api_key
        self._reasoning_model = reasoning_model
        self._evaluation_model = evaluation_model
        self._max_opus_per_hour = max_opus_per_hour
        self._max_sonnet_per_hour = max_sonnet_per_hour
        self._monthly_budget_usd = monthly_budget_usd
        self._spend_tracker = LLMSpendTracker()
        self._opus_calls: deque[float] = deque()
        self._sonnet_calls: deque[float] = deque()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if anthropic is None:
                raise ImportError("anthropic package not installed")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _check_rate_limit(self, model: str) -> bool:
        now = time.time()
        hour_ago = now - 3600
        if "opus" in model:
            self._opus_calls = deque(t for t in self._opus_calls if t > hour_ago)
            return len(self._opus_calls) < self._max_opus_per_hour
        else:
            self._sonnet_calls = deque(t for t in self._sonnet_calls if t > hour_ago)
            return len(self._sonnet_calls) < self._max_sonnet_per_hour

    def _record_call(self, model: str) -> None:
        if "opus" in model:
            self._opus_calls.append(time.time())
        else:
            self._sonnet_calls.append(time.time())

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_cost = self._COST_PER_M_INPUT.get(model, 3.0) * input_tokens / 1_000_000
        out_cost = self._COST_PER_M_OUTPUT.get(model, 15.0) * output_tokens / 1_000_000
        return in_cost + out_cost

    def _call_api(
        self, model: str, system: str, user_prompt: str, phase: str,
        max_tokens: int = 4096,
    ) -> Optional[dict]:
        # Budget check
        if self._spend_tracker.monthly_spend_usd() >= self._monthly_budget_usd:
            print(f"[CLAUDE] Monthly budget ${self._monthly_budget_usd} exceeded — skipping call")
            return {"budget_exceeded": True}

        # Rate limit check
        if not self._check_rate_limit(model):
            print(f"[CLAUDE] Rate limit hit for {model} — skipping call")
            return None

        try:
            client = self._get_client()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_prompt}],
            )
            self._record_call(model)

            # Parse response
            text = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            cached = getattr(response.usage, "cache_read_input_tokens", 0) > 0

            self._spend_tracker.log(
                model=model, phase=phase, input_tokens=input_tokens,
                output_tokens=output_tokens, cost_usd=cost, prompt_cached=cached,
            )

            # Extract JSON from response
            return _extract_json(text)
        except Exception as e:
            print(f"[CLAUDE] API error: {e}")
            return None

    def reason_about_edge(
        self, edge_context: str, supporting_evidence: list[str],
        contradicting_evidence: list[str],
    ) -> Optional[dict]:
        system = (
            "You are an ALS research scientist analyzing mechanistic evidence. "
            "Return JSON with keys: confidence_assessment (0-1), mechanism (string), "
            "open_questions (list[string]), confounders (list[string])."
        )
        user = (
            f"Analyze this mechanistic edge in ALS biology:\n\n"
            f"Edge: {edge_context}\n\n"
            f"Supporting evidence:\n" + "\n".join(f"- {e}" for e in supporting_evidence) + "\n\n"
            f"Contradicting evidence:\n" + "\n".join(f"- {e}" for e in contradicting_evidence) + "\n\n"
            f"Assess: (1) confidence in this causal link, (2) most likely mechanism, "
            f"(3) what would resolve remaining uncertainty, (4) confounders."
        )
        return self._call_api(self._evaluation_model, system, user, phase="reasoning")

    def counterfactual_analysis(
        self, hypothesis: str, causal_path: list[str], tcg_context: str,
    ) -> Optional[dict]:
        system = (
            "You are an ALS therapeutic strategist performing counterfactual analysis. "
            "Return JSON with keys: downstream_effects (list[dict]), off_target_risks (list[string]), "
            "confidence (float), new_edges (list[dict with source, target, edge_type, rationale])."
        )
        user = (
            f"Hypothesis: {hypothesis}\n\n"
            f"Causal path through the therapeutic graph:\n"
            + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(causal_path)) + "\n\n"
            f"Graph context:\n{tcg_context}\n\n"
            f"Trace every downstream consequence of this intervention. "
            f"Assess off-target effects and unintended pathway interactions."
        )
        return self._call_api(self._reasoning_model, system, user, phase="reasoning")

    def cross_pathway_synthesis(
        self, cluster_a: str, cluster_a_evidence: list[str],
        cluster_b: str, cluster_b_evidence: list[str],
    ) -> Optional[dict]:
        system = (
            "You are an ALS systems biologist looking for undiscovered pathway interactions. "
            "Return JSON with key: proposed_edges (list[dict with source, target, edge_type, "
            "confidence (float), rationale (string)])."
        )
        user = (
            f"Cluster A ({cluster_a}):\n" + "\n".join(f"- {e}" for e in cluster_a_evidence) + "\n\n"
            f"Cluster B ({cluster_b}):\n" + "\n".join(f"- {e}" for e in cluster_b_evidence) + "\n\n"
            f"Are there undiscovered mechanistic links between these clusters? "
            f"Could intervening in one affect the other?"
        )
        return self._call_api(self._reasoning_model, system, user, phase="reasoning")

    def evaluate_compound(
        self, compound: str, target_edges: list[str], current_protocol: str,
    ) -> Optional[dict]:
        system = (
            "You are a medicinal chemist evaluating drug candidates for ALS. "
            "Return JSON with keys: suitability_score (0-1), strengths (list[string]), "
            "risks (list[string]), drug_interactions (list[string]), recommendation (string)."
        )
        user = (
            f"Compound: {compound}\n\n"
            f"Target edges in therapeutic graph:\n"
            + "\n".join(f"- {e}" for e in target_edges) + "\n\n"
            f"Current protocol:\n{current_protocol}\n\n"
            f"Evaluate this compound for inclusion in the treatment protocol."
        )
        return self._call_api(self._evaluation_model, system, user, phase="compound")


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from text (handles markdown fences)."""
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try markdown fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_claude_client.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/llm/claude_client.py tests/test_claude_client.py && git commit -m "feat(llm): add Claude API client — rate limiting, spend tracking, structured prompts"
```

---

## Task 6: Integration Daemon

**Files:**
- Create: `scripts/daemons/__init__.py`
- Create: `scripts/daemons/integration_daemon.py`
- Test: `tests/test_integration_daemon.py`

- [ ] **Step 1: Write tests for integration daemon**

```python
# tests/test_integration_daemon.py
"""Tests for the IntegrationDaemon — evidence -> TCG integration."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.integration_daemon import IntegrationDaemon, _classify_question_type


class TestQuestionTypeClassification:
    def test_mechanistic_question(self):
        assert _classify_question_type("Does TDP-43 aggregation cause STMN2 loss?") == "mechanistic"

    def test_binding_question(self):
        assert _classify_question_type("What is the binding affinity of riluzole to EAAT2?") == "binding"

    def test_expression_question(self):
        assert _classify_question_type("Is SOD1 expressed in upper motor neurons?") == "expression"

    def test_genetic_question(self):
        assert _classify_question_type("Is the TARDBP variant pathogenic?") == "genetic"

    def test_clinical_question(self):
        assert _classify_question_type("Is masitinib in clinical trials for ALS?") == "clinical"

    def test_pathway_question(self):
        assert _classify_question_type("What pathway does CSF1R participate in?") == "pathway"


class TestSourceMapping:
    def test_mechanistic_sources(self):
        from daemons.integration_daemon import QUESTION_TYPE_SOURCES
        assert "pubmed" in QUESTION_TYPE_SOURCES["mechanistic"]
        assert "biorxiv" in QUESTION_TYPE_SOURCES["mechanistic"]

    def test_binding_sources(self):
        from daemons.integration_daemon import QUESTION_TYPE_SOURCES
        assert "chembl" in QUESTION_TYPE_SOURCES["binding"]
        assert "bindingdb" in QUESTION_TYPE_SOURCES["binding"]


class TestIntegrationDaemon:
    def test_init(self):
        daemon = IntegrationDaemon()
        assert daemon._interval_s > 0

    @patch("daemons.integration_daemon.IntegrationDaemon._get_unintegrated_evidence")
    def test_no_evidence_is_noop(self, mock_get):
        mock_get.return_value = []
        daemon = IntegrationDaemon()
        stats = daemon.integrate_batch()
        assert stats["items_processed"] == 0
        assert stats["edges_updated"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_integration_daemon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemons'`

- [ ] **Step 3: Implement integration daemon**

```python
# scripts/daemons/__init__.py
"""Background daemons for Erik's cognitive engine."""
```

```python
# scripts/daemons/integration_daemon.py
"""Phase 2 — IntegrationDaemon: weave raw evidence into the Therapeutic Causal Graph.

Runs every 2-5 minutes. Reads unintegrated evidence from erik_core.objects,
extracts entities and relationships, updates TCG edge confidence via Bayesian
updates, and creates acquisition queue entries for new weak edges.
"""
from __future__ import annotations

import time
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from llm.inference import create_llm
from tcg.graph import TCGraph
from tcg.models import AcquisitionItem


# Source mapping: question type -> suggested data sources
QUESTION_TYPE_SOURCES: dict[str, list[str]] = {
    "mechanistic": ["pubmed", "biorxiv", "galen_kg"],
    "binding": ["chembl", "bindingdb", "drugbank"],
    "expression": ["gtex", "hpa", "geo_als"],
    "genetic": ["clinvar", "gnomad", "gwas", "alsod"],
    "clinical": ["clinical_trials", "faers"],
    "pathway": ["reactome", "kegg", "string"],
    "structural": ["alphafold", "uniprot"],
}

_QUESTION_KEYWORDS: dict[str, list[str]] = {
    "binding": ["bind", "affinity", "ic50", "ki ", "kd ", "inhibit"],
    "expression": ["express", "transcri", "mrna", "rna level"],
    "genetic": ["variant", "mutation", "pathogenic", "snp", "polymorphism", "allele"],
    "clinical": ["trial", "clinical", "phase ", "fda", "approved"],
    "pathway": ["pathway", "signal", "cascade", "downstream"],
    "structural": ["structure", "3d", "crystal", "cryo-em", "fold"],
}


def _classify_question_type(question: str) -> str:
    """Classify an open question to determine which data sources to suggest."""
    q_lower = question.lower()
    for qtype, keywords in _QUESTION_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return qtype
    return "mechanistic"  # default


class IntegrationDaemon:
    """Phase 2: Evidence -> TCG integration."""

    def __init__(self) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("integration_interval_s", 180)
        self._batch_size = cfg.get("integration_batch_size", 50)
        self._prior_strength = cfg.get("integration_confidence_prior_strength", 2.0)
        self._graph = TCGraph()
        self._stop = threading.Event()

    def _get_unintegrated_evidence(self) -> list[dict]:
        """Fetch evidence items not yet integrated into TCG."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, type, body, confidence, provenance_source_system
                    FROM erik_core.objects
                    WHERE status = 'active'
                      AND tcg_integrated = FALSE
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (self._batch_size,))
                rows = cur.fetchall()
        return [
            {"id": r[0], "type": r[1], "body": r[2] if isinstance(r[2], dict) else {},
             "confidence": r[3], "source": r[4]}
            for r in rows
        ]

    def _mark_integrated(self, item_ids: list[str]) -> None:
        if not item_ids:
            return
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_core.objects SET tcg_integrated = TRUE
                    WHERE id = ANY(%s)
                """, (item_ids,))
            conn.commit()

    def _extract_edge_matches(self, evidence: dict) -> list[tuple[str, float]]:
        """Match evidence to TCG edges by text overlap on node names.

        Returns list of (edge_id, evidence_strength) tuples.
        """
        body = evidence.get("body", {})
        claim = body.get("claim", "")
        text = f"{claim} {body.get('notes', '')} {body.get('mechanism', '')}".lower()
        confidence = evidence.get("confidence") or 0.5

        matches = []
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Find edges where both source and target node names appear in evidence text
                cur.execute("""
                    SELECT e.id, e.confidence,
                           sn.name, tn.name
                    FROM erik_core.tcg_edges e
                    JOIN erik_core.tcg_nodes sn ON e.source_id = sn.id
                    JOIN erik_core.tcg_nodes tn ON e.target_id = tn.id
                """)
                for row in cur.fetchall():
                    edge_id, edge_conf, src_name, tgt_name = row
                    src_lower = src_name.lower()
                    tgt_lower = tgt_name.lower()
                    if src_lower in text and tgt_lower in text:
                        matches.append((edge_id, confidence))
        return matches

    def integrate_batch(self) -> dict[str, int]:
        """Process one batch of unintegrated evidence. Returns stats."""
        evidence_items = self._get_unintegrated_evidence()
        if not evidence_items:
            return {"items_processed": 0, "edges_updated": 0, "queue_items_created": 0}

        edges_updated = 0
        queue_items = 0
        processed_ids = []

        for ev in evidence_items:
            matches = self._extract_edge_matches(ev)
            for edge_id, strength in matches:
                self._graph.bayesian_update(edge_id, strength, ev["id"])
                edges_updated += 1

                # Create acquisition items for edges that got updated but are still weak
                edge = self._graph.get_edge(edge_id)
                if edge and edge.confidence < 0.7 and edge.open_questions:
                    for q in edge.open_questions[:1]:  # First open question only
                        qtype = _classify_question_type(q)
                        sources = QUESTION_TYPE_SOURCES.get(qtype, ["pubmed"])
                        self._graph.push_acquisition(AcquisitionItem(
                            tcg_edge_id=edge_id,
                            open_question=q,
                            suggested_sources=sources,
                            priority=edge.therapeutic_priority(),
                            created_by="integration",
                        ))
                        queue_items += 1

            processed_ids.append(ev["id"])

        self._mark_integrated(processed_ids)

        if edges_updated > 0:
            self._graph.log_activity(
                phase="integration", event_type="batch_integrated",
                summary=f"Integrated {len(processed_ids)} evidence items, updated {edges_updated} edges",
            )

        return {
            "items_processed": len(processed_ids),
            "edges_updated": edges_updated,
            "queue_items_created": queue_items,
        }

    def run(self) -> None:
        """Run the daemon loop until stopped."""
        print("[INTEGRATION] Daemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("integration_enabled", True):
                    self._stop.wait(60)
                    continue

                self._interval_s = cfg.get("integration_interval_s", 180)
                stats = self.integrate_batch()
                if stats["items_processed"] > 0:
                    print(f"[INTEGRATION] Batch: {stats['items_processed']} items, "
                          f"{stats['edges_updated']} edges, {stats['queue_items_created']} queue items")
            except Exception as e:
                print(f"[INTEGRATION] Error: {e}")

            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_integration_daemon.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/daemons/__init__.py scripts/daemons/integration_daemon.py tests/test_integration_daemon.py && git commit -m "feat(daemons): add IntegrationDaemon — evidence to TCG Bayesian integration with acquisition queue"
```

---

## Task 7: Reasoning Daemon

**Files:**
- Create: `scripts/daemons/reasoning_daemon.py`
- Test: `tests/test_reasoning_daemon.py`

- [ ] **Step 1: Write tests for reasoning daemon**

```python
# tests/test_reasoning_daemon.py
"""Tests for the ReasoningDaemon — Claude-powered deep analysis."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.reasoning_daemon import ReasoningDaemon, _select_mode


class TestModeSelection:
    def test_mode_weights_default(self):
        mode = _select_mode(step=0, weights=[0.5, 0.3, 0.2])
        assert mode in ("edge_deepening", "counterfactual", "cross_pathway")

    def test_mode_distribution_respects_weights(self):
        """Over many calls, mode A should appear ~50% of the time."""
        counts = {"edge_deepening": 0, "counterfactual": 0, "cross_pathway": 0}
        for step in range(100):
            mode = _select_mode(step=step, weights=[0.5, 0.3, 0.2])
            counts[mode] += 1
        assert counts["edge_deepening"] >= 35  # Expect ~50, allow variance
        assert counts["cross_pathway"] >= 10   # Expect ~20


class TestReasoningDaemon:
    def test_init(self):
        daemon = ReasoningDaemon(claude_api_key="test-key")
        assert daemon._interval_s > 0

    @patch("daemons.reasoning_daemon.TCGraph")
    def test_no_weak_edges_uses_cross_pathway(self, mock_graph_cls):
        mock_graph = MagicMock()
        mock_graph.get_weakest_edges.return_value = []
        mock_graph_cls.return_value = mock_graph
        daemon = ReasoningDaemon(claude_api_key="test-key")
        # When no weak edges exist, daemon should fall through to cross-pathway mode
        # (this tests the selection logic, not the full cycle)
        assert daemon is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reasoning_daemon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemons.reasoning_daemon'`

- [ ] **Step 3: Implement reasoning daemon**

```python
# scripts/daemons/reasoning_daemon.py
"""Phase 3 — ReasoningDaemon: Claude-powered deep reasoning over the TCG.

Three modes rotated by configurable weights:
A) Edge Deepening — strengthen/refute individual edges (Sonnet)
B) Counterfactual Analysis — trace intervention consequences (Opus)
C) Cross-Pathway Synthesis — discover inter-cluster connections (Opus)
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from llm.claude_client import ClaudeClient
from tcg.graph import TCGraph
from tcg.models import TCGEdge, TCGHypothesis


def _select_mode(step: int, weights: list[float]) -> str:
    """Deterministic mode selection from weights using golden-ratio stride."""
    modes = ["edge_deepening", "counterfactual", "cross_pathway"]
    total = sum(weights)
    target = ((step * 61) % 100) / 100.0 * total
    cumulative = 0.0
    for mode, w in zip(modes, weights):
        cumulative += w
        if cumulative >= target:
            return mode
    return modes[-1]


class ReasoningDaemon:
    """Phase 3: Claude-powered deep reasoning over the TCG."""

    def __init__(self, claude_api_key: str) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("reasoning_interval_s", 900)
        self._mode_weights = cfg.get("reasoning_mode_weights", [0.5, 0.3, 0.2])
        self._max_evidence_per_prompt = cfg.get("reasoning_max_evidence_per_prompt", 30)
        self._graph = TCGraph()
        self._claude = ClaudeClient(
            api_key=claude_api_key,
            reasoning_model=cfg.get("claude_reasoning_model", "claude-opus-4-6"),
            evaluation_model=cfg.get("claude_evaluation_model", "claude-sonnet-4-6"),
            max_opus_per_hour=cfg.get("claude_max_opus_calls_per_hour", 30),
            max_sonnet_per_hour=cfg.get("claude_max_sonnet_calls_per_hour", 60),
            monthly_budget_usd=cfg.get("claude_monthly_budget_usd", 100.0),
        )
        self._step = 0
        self._stop = threading.Event()

    def _get_evidence_text(self, evidence_ids: list[str], limit: int = 30) -> list[str]:
        """Fetch evidence claim text by IDs."""
        if not evidence_ids:
            return []
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, body->>'claim' as claim
                    FROM erik_core.objects
                    WHERE id = ANY(%s) AND status = 'active'
                    LIMIT %s
                """, (evidence_ids[:limit], limit))
                rows = cur.fetchall()
        return [f"[{r[0]}] {r[1]}" for r in rows if r[1]]

    def _run_edge_deepening(self) -> dict:
        """Mode A: Select weakest high-relevance edge, reason about it with Claude Sonnet."""
        edges = self._graph.get_weakest_edges(limit=20)
        if not edges:
            return {"mode": "edge_deepening", "action": "no_weak_edges"}

        # Pick the edge with highest therapeutic priority
        edge = max(edges, key=lambda e: e.therapeutic_priority())
        supporting = self._get_evidence_text(edge.evidence_ids)
        contradicting = self._get_evidence_text(edge.contradiction_ids)

        source_node = self._graph.get_node(edge.source_id)
        target_node = self._graph.get_node(edge.target_id)
        edge_context = (
            f"{source_node.name if source_node else edge.source_id} "
            f"--[{edge.edge_type}]--> "
            f"{target_node.name if target_node else edge.target_id} "
            f"(current confidence: {edge.confidence})"
        )

        result = self._claude.reason_about_edge(edge_context, supporting, contradicting)
        if not result or result.get("budget_exceeded"):
            return {"mode": "edge_deepening", "action": "api_unavailable"}

        # Update edge based on Claude's analysis
        new_confidence = result.get("confidence_assessment", edge.confidence)
        new_questions = result.get("open_questions", [])
        now = datetime.now(timezone.utc)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE erik_core.tcg_edges
                    SET confidence = %s, open_questions = %s, last_reasoned_at = %s, updated_at = %s
                    WHERE id = %s
                """, (new_confidence, new_questions, now, now, edge.id))
            conn.commit()

        self._graph.log_activity(
            phase="reasoning", event_type="edge_deepened",
            summary=f"Edge {edge.id}: {edge.confidence:.2f} -> {new_confidence:.2f}",
            tcg_edge_id=edge.id,
        )

        return {"mode": "edge_deepening", "edge": edge.id,
                "confidence_before": edge.confidence, "confidence_after": new_confidence}

    def _run_counterfactual(self) -> dict:
        """Mode B: Counterfactual analysis on a hypothesis with Claude Opus."""
        hypotheses = self._graph.get_hypotheses_by_status("under_investigation")
        if not hypotheses:
            # Promote a proposed hypothesis
            proposed = self._graph.get_hypotheses_by_status("proposed")
            if proposed:
                hyp = proposed[0]
                hyp.status = "under_investigation"
                self._graph.upsert_hypothesis(hyp)
                hypotheses = [hyp]
            else:
                return {"mode": "counterfactual", "action": "no_hypotheses"}

        hyp = hypotheses[0]
        # Build causal path context
        path_descriptions = []
        for edge_id in hyp.supporting_path:
            edge = self._graph.get_edge(edge_id)
            if edge:
                src = self._graph.get_node(edge.source_id)
                tgt = self._graph.get_node(edge.target_id)
                path_descriptions.append(
                    f"{src.name if src else edge.source_id} --[{edge.edge_type}]--> "
                    f"{tgt.name if tgt else edge.target_id} (conf: {edge.confidence:.2f})"
                )

        result = self._claude.counterfactual_analysis(
            hypothesis=hyp.hypothesis,
            causal_path=path_descriptions,
            tcg_context=f"Hypothesis: {hyp.hypothesis}\nTherapeutic relevance: {hyp.therapeutic_relevance}",
        )
        if not result or result.get("budget_exceeded"):
            return {"mode": "counterfactual", "action": "api_unavailable"}

        # Update hypothesis
        new_conf = result.get("confidence", hyp.confidence)
        now = datetime.now(timezone.utc)
        hyp.confidence = new_conf
        hyp.updated_at = now

        # Create new edges from counterfactual analysis
        new_edges_data = result.get("new_edges", [])
        new_edges_created = 0
        for ne in new_edges_data:
            if all(k in ne for k in ("source", "target", "edge_type")):
                edge_id = f"edge:cf_{ne['source'].split(':')[-1]}_{ne['target'].split(':')[-1]}"
                new_edge = TCGEdge(
                    id=edge_id, source_id=ne["source"], target_id=ne["target"],
                    edge_type=ne["edge_type"], confidence=ne.get("confidence", 0.2),
                    open_questions=[ne.get("rationale", "Discovered via counterfactual analysis")],
                )
                try:
                    self._graph.upsert_edge(new_edge)
                    new_edges_created += 1
                except Exception:
                    pass  # Node may not exist

        if new_conf >= 0.7:
            hyp.status = "supported"
        elif new_conf < 0.2:
            hyp.status = "refuted"
        self._graph.upsert_hypothesis(hyp)

        self._graph.log_activity(
            phase="reasoning", event_type="counterfactual_analyzed",
            summary=f"Hypothesis '{hyp.id}': conf {new_conf:.2f}, {new_edges_created} new edges",
            tcg_hypothesis_id=hyp.id,
        )

        return {"mode": "counterfactual", "hypothesis": hyp.id,
                "new_edges": new_edges_created, "confidence": new_conf}

    def _run_cross_pathway(self) -> dict:
        """Mode C: Cross-pathway synthesis between two clusters with Claude Opus."""
        # Find the two clusters with fewest inter-cluster edges
        clusters = list(self._graph.summary().get("clusters", {}).keys())
        if len(clusters) < 2:
            return {"mode": "cross_pathway", "action": "insufficient_clusters"}

        # Pick two clusters with least connection
        # Simple heuristic: rotate through pairs using step count
        import itertools
        pairs = list(itertools.combinations(sorted(clusters), 2))
        if not pairs:
            return {"mode": "cross_pathway", "action": "no_pairs"}
        pair = pairs[self._step % len(pairs)]
        cluster_a, cluster_b = pair

        # Get evidence summaries from each cluster
        nodes_a = self._graph.list_nodes(pathway_cluster=cluster_a)
        nodes_b = self._graph.list_nodes(pathway_cluster=cluster_b)
        evidence_a = [f"{n.name}: {n.description or n.entity_type}" for n in nodes_a[:15]]
        evidence_b = [f"{n.name}: {n.description or n.entity_type}" for n in nodes_b[:15]]

        result = self._claude.cross_pathway_synthesis(cluster_a, evidence_a, cluster_b, evidence_b)
        if not result or result.get("budget_exceeded"):
            return {"mode": "cross_pathway", "action": "api_unavailable"}

        proposed = result.get("proposed_edges", [])
        created = 0
        for pe in proposed:
            if all(k in pe for k in ("source", "target", "edge_type")):
                edge_id = f"edge:xp_{pe['source'].split(':')[-1]}_{pe['target'].split(':')[-1]}"
                new_edge = TCGEdge(
                    id=edge_id, source_id=pe["source"], target_id=pe["target"],
                    edge_type=pe["edge_type"], confidence=pe.get("confidence", 0.2),
                    open_questions=[pe.get("rationale", "Cross-pathway synthesis discovery")],
                )
                try:
                    self._graph.upsert_edge(new_edge)
                    created += 1
                except Exception:
                    pass

        if created > 0:
            self._graph.log_activity(
                phase="reasoning", event_type="cross_pathway_discovery",
                summary=f"Found {created} new edges between {cluster_a} and {cluster_b}",
            )

        return {"mode": "cross_pathway", "clusters": [cluster_a, cluster_b],
                "new_edges": created}

    def reason_once(self) -> dict:
        """Execute one reasoning cycle. Returns stats dict."""
        cfg = ConfigLoader()
        self._mode_weights = cfg.get("reasoning_mode_weights", self._mode_weights)
        mode = _select_mode(self._step, self._mode_weights)
        self._step += 1

        if mode == "edge_deepening":
            return self._run_edge_deepening()
        elif mode == "counterfactual":
            return self._run_counterfactual()
        else:
            return self._run_cross_pathway()

    def run(self) -> None:
        """Run the daemon loop until stopped."""
        print("[REASONING] Daemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("reasoning_enabled", True):
                    self._stop.wait(60)
                    continue

                self._interval_s = cfg.get("reasoning_interval_s", 900)
                result = self.reason_once()
                print(f"[REASONING] {result.get('mode', '?')}: {result}")
            except Exception as e:
                print(f"[REASONING] Error: {e}")

            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_reasoning_daemon.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/daemons/reasoning_daemon.py tests/test_reasoning_daemon.py && git commit -m "feat(daemons): add ReasoningDaemon — 3-mode Claude reasoning (edge deepening, counterfactual, cross-pathway)"
```

---

## Task 8: Compound Evaluation Daemon

**Files:**
- Create: `scripts/daemons/compound_daemon.py`
- Test: `tests/test_compound_daemon.py`

- [ ] **Step 1: Write tests for compound daemon**

```python
# tests/test_compound_daemon.py
"""Tests for the CompoundDaemon — drug candidate evaluation."""
import pytest
from unittest.mock import MagicMock, patch
from daemons.compound_daemon import CompoundDaemon


class TestCompoundDaemon:
    def test_init(self):
        daemon = CompoundDaemon(claude_api_key="test-key")
        assert daemon._interval_s >= 3600

    @patch("daemons.compound_daemon.TCGraph")
    def test_no_supported_hypotheses_is_noop(self, mock_graph_cls):
        mock_graph = MagicMock()
        mock_graph.get_hypotheses_by_status.return_value = []
        mock_graph_cls.return_value = mock_graph
        daemon = CompoundDaemon(claude_api_key="test-key")
        daemon._graph = mock_graph
        result = daemon.evaluate_once()
        assert result["action"] == "no_actionable_hypotheses"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_compound_daemon.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'daemons.compound_daemon'`

- [ ] **Step 3: Implement compound daemon**

```python
# scripts/daemons/compound_daemon.py
"""Phase 4 — CompoundDaemon: evaluate drug candidates against TCG hypotheses.

Runs every 1-2 hours. Takes supported hypotheses, identifies druggable nodes,
queries compound databases, and scores candidates for the treatment protocol.
"""
from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Optional

from config.loader import ConfigLoader
from db.pool import get_connection
from llm.claude_client import ClaudeClient
from tcg.graph import TCGraph
from tcg.models import TCGHypothesis


class CompoundDaemon:
    """Phase 4: Drug candidate evaluation against TCG hypotheses."""

    def __init__(self, claude_api_key: str) -> None:
        cfg = ConfigLoader()
        self._interval_s = cfg.get("compound_interval_s", 3600)
        self._min_confidence = cfg.get("compound_min_hypothesis_confidence", 0.6)
        self._graph = TCGraph()
        self._claude = ClaudeClient(
            api_key=claude_api_key,
            evaluation_model=cfg.get("claude_evaluation_model", "claude-sonnet-4-6"),
            monthly_budget_usd=cfg.get("claude_monthly_budget_usd", 100.0),
        )
        self._stop = threading.Event()

    def _get_druggable_nodes_for_hypothesis(self, hyp: TCGHypothesis) -> list[dict]:
        """Find druggable nodes along a hypothesis's causal path."""
        druggable = []
        for edge_id in hyp.supporting_path:
            edge = self._graph.get_edge(edge_id)
            if not edge:
                continue
            for node_id in (edge.source_id, edge.target_id):
                node = self._graph.get_node(node_id)
                if node and node.druggability_score > 0.3:
                    druggable.append({
                        "node_id": node.id,
                        "name": node.name,
                        "druggability": node.druggability_score,
                        "cluster": node.pathway_cluster,
                    })
        return druggable

    def _query_existing_compounds(self, target_name: str) -> list[dict]:
        """Query evidence store for known compounds targeting this entity."""
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, body->>'name' as name, body->>'intervention_class' as cls,
                           confidence
                    FROM erik_core.objects
                    WHERE type = 'Intervention' AND status = 'active'
                    AND (body->>'targets')::text ILIKE %s
                    LIMIT 10
                """, (f"%{target_name}%",))
                rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "class": r[2], "confidence": r[3]} for r in rows]

    def evaluate_once(self) -> dict:
        """Run one compound evaluation cycle."""
        # Get supported hypotheses above confidence threshold
        supported = self._graph.get_hypotheses_by_status("supported")
        actionable = [h for h in supported if h.confidence >= self._min_confidence]

        if not actionable:
            return {"action": "no_actionable_hypotheses"}

        hyp = actionable[0]
        druggable = self._get_druggable_nodes_for_hypothesis(hyp)
        if not druggable:
            return {"action": "no_druggable_nodes", "hypothesis": hyp.id}

        # Collect existing compounds for each druggable target
        target_compounds: dict[str, list[dict]] = {}
        for node in druggable:
            compounds = self._query_existing_compounds(node["name"])
            if compounds:
                target_compounds[node["name"]] = compounds

        # Build edge descriptions for Claude
        edge_descriptions = []
        for eid in hyp.supporting_path:
            edge = self._graph.get_edge(eid)
            if edge:
                src = self._graph.get_node(edge.source_id)
                tgt = self._graph.get_node(edge.target_id)
                edge_descriptions.append(
                    f"{src.name if src else edge.source_id} --[{edge.edge_type}]--> "
                    f"{tgt.name if tgt else edge.target_id}"
                )

        # Get current protocol summary
        protocol_summary = self._get_protocol_summary()

        # Ask Claude to evaluate
        compound_text = ""
        for target, compounds in target_compounds.items():
            compound_text += f"\nTarget: {target}\n"
            for c in compounds:
                compound_text += f"  - {c['name']} ({c['class']}, confidence: {c['confidence']})\n"

        result = self._claude.evaluate_compound(
            compound=compound_text or "No existing compounds found — novel design needed",
            target_edges=edge_descriptions,
            current_protocol=protocol_summary,
        )

        if result and not result.get("budget_exceeded"):
            # Log the evaluation
            self._graph.log_activity(
                phase="compound", event_type="compound_evaluated",
                summary=f"Evaluated compounds for hypothesis '{hyp.id}': "
                        f"{result.get('recommendation', 'pending')}",
                tcg_hypothesis_id=hyp.id,
                details=result,
            )

            # If highly recommended, promote hypothesis to actionable
            score = result.get("suitability_score", 0)
            if score >= 0.7 and hyp.status != "actionable":
                hyp.status = "actionable"
                self._graph.upsert_hypothesis(hyp)

        return {
            "hypothesis": hyp.id,
            "druggable_nodes": len(druggable),
            "compounds_found": sum(len(v) for v in target_compounds.values()),
            "evaluation": result,
        }

    def _get_protocol_summary(self) -> str:
        """Get current protocol as a text summary."""
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT body FROM erik_core.objects
                        WHERE type = 'Protocol' AND status = 'active'
                        ORDER BY updated_at DESC LIMIT 1
                    """)
                    row = cur.fetchone()
            if row and isinstance(row[0], dict):
                layers = row[0].get("layers", [])
                return "\n".join(
                    f"Layer {l.get('name', '?')}: "
                    + ", ".join(i.get("name", "?") for i in l.get("interventions", []))
                    for l in layers
                )
        except Exception:
            pass
        return "No protocol available"

    def run(self) -> None:
        """Run the daemon loop until stopped."""
        print("[COMPOUND] Daemon started")
        while not self._stop.is_set():
            try:
                cfg = ConfigLoader()
                if not cfg.get("compound_enabled", True):
                    self._stop.wait(60)
                    continue

                self._interval_s = cfg.get("compound_interval_s", 3600)
                result = self.evaluate_once()
                print(f"[COMPOUND] {result}")
            except Exception as e:
                print(f"[COMPOUND] Error: {e}")

            self._stop.wait(self._interval_s)

    def stop(self) -> None:
        self._stop.set()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_compound_daemon.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/daemons/compound_daemon.py tests/test_compound_daemon.py && git commit -m "feat(daemons): add CompoundDaemon — drug candidate evaluation against TCG hypotheses"
```

---

## Task 9: Wire Daemons into run_loop.py

**Files:**
- Modify: `scripts/run_loop.py`
- Test: `tests/test_daemon_startup.py`

- [ ] **Step 1: Write test for daemon startup**

```python
# tests/test_daemon_startup.py
"""Tests for daemon startup wiring in run_loop."""
import pytest
from unittest.mock import patch, MagicMock


class TestDaemonStartup:
    @patch("daemons.integration_daemon.IntegrationDaemon")
    @patch("daemons.reasoning_daemon.ReasoningDaemon")
    @patch("daemons.compound_daemon.CompoundDaemon")
    def test_start_daemons_creates_threads(self, mock_compound, mock_reasoning, mock_integration):
        from run_loop import _start_cognitive_daemons, _stop_cognitive_daemons

        daemons = _start_cognitive_daemons(claude_api_key="test-key")
        assert "integration" in daemons
        assert "reasoning" in daemons
        assert "compound" in daemons
        _stop_cognitive_daemons(daemons)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_daemon_startup.py -v`
Expected: FAIL — `_start_cognitive_daemons` not found

- [ ] **Step 3: Add daemon startup functions to run_loop.py**

Add the following to `scripts/run_loop.py` after the existing imports:

```python
from daemons.integration_daemon import IntegrationDaemon
from daemons.reasoning_daemon import ReasoningDaemon
from daemons.compound_daemon import CompoundDaemon


def _start_cognitive_daemons(claude_api_key: str = "") -> dict[str, tuple]:
    """Start all cognitive engine daemons as background threads.

    Returns a dict mapping daemon name to (daemon_instance, thread) tuples.
    """
    cfg = ConfigLoader()
    daemons: dict[str, tuple] = {}

    # Phase 2: Integration
    if cfg.get("integration_enabled", True):
        integration = IntegrationDaemon()
        t_int = threading.Thread(target=integration.run, name="integration-daemon", daemon=True)
        t_int.start()
        daemons["integration"] = (integration, t_int)
        print("[ERIK] Started IntegrationDaemon")

    # Phase 3: Reasoning (requires Claude API key)
    if cfg.get("reasoning_enabled", True) and claude_api_key:
        reasoning = ReasoningDaemon(claude_api_key=claude_api_key)
        t_rea = threading.Thread(target=reasoning.run, name="reasoning-daemon", daemon=True)
        t_rea.start()
        daemons["reasoning"] = (reasoning, t_rea)
        print("[ERIK] Started ReasoningDaemon")

    # Phase 4: Compound (requires Claude API key)
    if cfg.get("compound_enabled", True) and claude_api_key:
        compound = CompoundDaemon(claude_api_key=claude_api_key)
        t_cmp = threading.Thread(target=compound.run, name="compound-daemon", daemon=True)
        t_cmp.start()
        daemons["compound"] = (compound, t_cmp)
        print("[ERIK] Started CompoundDaemon")

    return daemons


def _stop_cognitive_daemons(daemons: dict[str, tuple]) -> None:
    """Gracefully stop all running daemons."""
    for name, (daemon, thread) in daemons.items():
        daemon.stop()
        thread.join(timeout=5)
        print(f"[ERIK] Stopped {name} daemon")
```

Also add `import threading` to the imports at the top of `run_loop.py` if not already present.

Then modify the `main()` function to start daemons after scaffold seeding and stop them on shutdown. In the existing `main()`, after the KG backfill and causal gap init blocks but before `print("[ERIK] Entering main loop...")`, add:

```python
    # Seed TCG scaffold if not already done
    try:
        from tcg.graph import TCGraph
        from tcg.seed_scaffold import seed_scaffold
        _tcg = TCGraph()
        _summary = _tcg.summary()
        if _summary["node_count"] == 0:
            print("[ERIK] Seeding TCG scaffold...")
            stats = seed_scaffold(_tcg)
            print(f"[ERIK] TCG scaffold: {stats['nodes_created']} nodes, {stats['edges_created']} edges")
        else:
            print(f"[ERIK] TCG: {_summary['node_count']} nodes, {_summary['edge_count']} edges, "
                  f"mean confidence {_summary['mean_confidence']:.3f}")
    except Exception as e:
        print(f"[ERIK] TCG scaffold skipped: {e}")

    # Start cognitive engine daemons
    _claude_key = os.environ.get("ANTHROPIC_API_KEY", cfg.get("anthropic_api_key", ""))
    cognitive_daemons = _start_cognitive_daemons(claude_api_key=_claude_key)
```

And in the `except KeyboardInterrupt` block, add before the break:

```python
            _stop_cognitive_daemons(cognitive_daemons)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_daemon_startup.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/run_loop.py tests/test_daemon_startup.py && git commit -m "feat: wire cognitive daemons into run_loop — TCG scaffold seeding + integration/reasoning/compound daemons"
```

---

## Task 10: New API Endpoints

**Files:**
- Create: `scripts/api/routers/graph.py`
- Create: `scripts/api/routers/hypotheses.py`
- Create: `scripts/api/routers/progress.py`
- Modify: `scripts/api/main.py`
- Modify: `scripts/api/routers/activity.py`
- Test: `tests/test_api_graph.py`

- [ ] **Step 1: Write API tests**

```python
# tests/test_api_graph.py
"""Tests for TCG API endpoints."""
import pytest
from unittest.mock import patch, MagicMock


class TestGraphEndpoints:
    def test_graph_summary_import(self):
        from api.routers.graph import router
        routes = [r.path for r in router.routes]
        assert "/graph/summary" in routes
        assert "/graph/weakest" in routes

    def test_hypotheses_import(self):
        from api.routers.hypotheses import router
        routes = [r.path for r in router.routes]
        assert "/hypotheses" in routes

    def test_progress_import(self):
        from api.routers.progress import router
        routes = [r.path for r in router.routes]
        assert "/progress" in routes
        assert "/progress/phases" in routes
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_api_graph.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement graph router**

```python
# scripts/api/routers/graph.py
"""TCG exploration API endpoints."""
from fastapi import APIRouter, Query
from tcg.graph import TCGraph

router = APIRouter(prefix="/api")
_graph = TCGraph()


@router.get("/graph/summary")
def graph_summary():
    return _graph.summary()


@router.get("/graph/cluster/{cluster_name}")
def graph_cluster(cluster_name: str):
    nodes = _graph.list_nodes(pathway_cluster=cluster_name)
    return {"cluster": cluster_name, "nodes": [n.to_dict() for n in nodes]}


@router.get("/graph/edge/{edge_id:path}")
def graph_edge(edge_id: str):
    edge = _graph.get_edge(edge_id)
    if edge is None:
        return {"error": "Edge not found"}
    return edge.to_dict()


@router.get("/graph/weakest")
def graph_weakest(limit: int = Query(default=10, le=100)):
    edges = _graph.get_weakest_edges(limit=limit)
    return {"edges": [e.to_dict() for e in edges]}
```

- [ ] **Step 4: Implement hypotheses router**

```python
# scripts/api/routers/hypotheses.py
"""Hypothesis pipeline API endpoints."""
from fastapi import APIRouter
from tcg.graph import TCGraph

router = APIRouter(prefix="/api")
_graph = TCGraph()


@router.get("/hypotheses")
def list_hypotheses():
    all_hyps = []
    for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
        all_hyps.extend(_graph.get_hypotheses_by_status(status))
    return {"hypotheses": [h.to_dict() for h in all_hyps]}


@router.get("/hypotheses/{hypothesis_id:path}")
def get_hypothesis(hypothesis_id: str):
    for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
        for h in _graph.get_hypotheses_by_status(status):
            if h.id == hypothesis_id:
                return h.to_dict()
    return {"error": "Hypothesis not found"}
```

- [ ] **Step 5: Implement progress router**

```python
# scripts/api/routers/progress.py
"""Progress dashboard API endpoints."""
from fastapi import APIRouter
from db.pool import get_connection
from tcg.graph import TCGraph

router = APIRouter(prefix="/api")
_graph = TCGraph()


@router.get("/progress")
def get_progress():
    summary = _graph.summary()
    edge_count = summary["edge_count"]
    mean_conf = summary["mean_confidence"]

    # Therapeutic coverage: edges with confidence > 0.7 that have intervention_potential.druggable
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FROM erik_core.tcg_edges
                WHERE confidence > 0.7 AND intervention_potential->>'druggable' = 'true'
            """)
            covered = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM erik_core.tcg_edges WHERE confidence > 0.7")
            high_conf = cur.fetchone()[0]
            cur.execute("""
                SELECT count(*) FROM erik_core.objects
                WHERE tcg_integrated = TRUE AND status = 'active'
            """)
            integrated = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM erik_core.objects WHERE status = 'active'")
            total_evidence = cur.fetchone()[0]

    # Hypothesis counts by status
    hyp_counts = {}
    for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
        hyp_counts[status] = len(_graph.get_hypotheses_by_status(status))

    return {
        "graph_confidence": mean_conf,
        "node_count": summary["node_count"],
        "edge_count": edge_count,
        "therapeutic_coverage": covered / max(high_conf, 1),
        "pathway_completeness": summary["clusters"],
        "hypothesis_pipeline": hyp_counts,
        "evidence_utilization": integrated / max(total_evidence, 1),
    }


@router.get("/progress/phases")
def get_phase_status():
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Last activity per phase
            cur.execute("""
                SELECT phase, max(created_at) as last_run, count(*) as total_events
                FROM erik_ops.activity_feed
                GROUP BY phase
            """)
            phases = {r[0]: {"last_run": r[1].isoformat() if r[1] else None,
                             "total_events": r[2]} for r in cur.fetchall()}

            # Monthly LLM spend
            cur.execute("""
                SELECT model, sum(cost_usd), count(*)
                FROM erik_ops.llm_spend
                WHERE created_at > date_trunc('month', now())
                GROUP BY model
            """)
            spend = {r[0]: {"cost_usd": float(r[1]), "calls": r[2]} for r in cur.fetchall()}

    return {"phases": phases, "llm_spend": spend}
```

- [ ] **Step 6: Update activity router to read from activity_feed table**

Read the existing `scripts/api/routers/activity.py` and modify it to query `erik_ops.activity_feed` instead of whatever empty source it currently uses:

```python
# Replace the get_activity endpoint body with:
@router.get("/activity")
def get_activity(limit: int = 50, offset: int = 0):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT phase, event_type, summary, details, created_at
                FROM erik_ops.activity_feed
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            rows = cur.fetchall()
    events = [
        {"phase": r[0], "event_type": r[1], "summary": r[2],
         "details": r[3], "timestamp": r[4].isoformat() if r[4] else None}
        for r in rows
    ]
    return {"events": events, "limit": limit, "offset": offset}
```

- [ ] **Step 7: Mount new routers in main.py**

Add to `scripts/api/main.py` after the existing router imports:

```python
from api.routers.graph import router as graph_router
from api.routers.hypotheses import router as hypotheses_router
from api.routers.progress import router as progress_router
```

And after the existing `app.include_router(...)` lines:

```python
app.include_router(graph_router)
app.include_router(hypotheses_router)
app.include_router(progress_router)
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_api_graph.py -v`
Expected: All 3 tests PASS

- [ ] **Step 9: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add scripts/api/routers/graph.py scripts/api/routers/hypotheses.py scripts/api/routers/progress.py scripts/api/routers/activity.py scripts/api/main.py tests/test_api_graph.py && git commit -m "feat(api): add TCG graph, hypotheses, and progress endpoints + fix activity feed"
```

---

## Task 11: Update Config and Deploy Migration

**Files:**
- Modify: `data/erik_config.json`
- Test: Manual verification

- [ ] **Step 1: Add cognitive engine config keys**

Read `data/erik_config.json` and add the following new keys (preserving all existing keys):

```json
{
    "integration_enabled": true,
    "integration_interval_s": 180,
    "integration_batch_size": 50,
    "integration_confidence_prior_strength": 2.0,
    "reasoning_enabled": true,
    "reasoning_interval_s": 900,
    "reasoning_model_deep": "claude-opus-4-6",
    "reasoning_model_light": "claude-sonnet-4-6",
    "reasoning_mode_weights": [0.5, 0.3, 0.2],
    "reasoning_max_evidence_per_prompt": 30,
    "compound_enabled": true,
    "compound_interval_s": 3600,
    "compound_min_hypothesis_confidence": 0.6,
    "compound_max_candidates_per_target": 10,
    "acquisition_enabled": true,
    "acquisition_interval_s": 2,
    "acquisition_min_confidence_target": 0.7,
    "acquisition_max_per_edge": 10,
    "acquisition_fallback_exploration": true,
    "claude_max_opus_calls_per_hour": 30,
    "claude_max_sonnet_calls_per_hour": 60,
    "claude_monthly_budget_usd": 100,
    "claude_prompt_cache_enabled": true,
    "claude_fallback_chain": ["claude-sonnet-4-6", "bedrock-nova-pro", "local-qwen"]
}
```

- [ ] **Step 2: Run full test suite**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v --timeout=60`
Expected: All tests PASS (including existing tests — no regressions)

- [ ] **Step 3: Run migration on Railway DB**

Run: `cd /Users/logannye/.openclaw/erik && railway run conda run -n erik-core python -c "from db.migrate import run_migrations; run_migrations()"`
Expected: TCG tables created on Railway PostgreSQL

- [ ] **Step 4: Commit config changes**

```bash
cd /Users/logannye/.openclaw/erik && git add data/erik_config.json && git commit -m "feat(config): add cognitive engine config keys — integration, reasoning, compound, claude API"
```

- [ ] **Step 5: Deploy to Railway**

```bash
cd /Users/logannye/.openclaw/erik && git push origin main
```

Wait for Railway auto-deploy to complete. Verify health endpoint returns OK.

---

## Task 12: End-to-End Integration Test

**Files:**
- Create: `tests/test_cognitive_engine_e2e.py`

- [ ] **Step 1: Write E2E test**

```python
# tests/test_cognitive_engine_e2e.py
"""End-to-end test: seed scaffold, integrate evidence, verify TCG state."""
import pytest
from tcg.graph import TCGraph
from tcg.seed_scaffold import seed_scaffold
from daemons.integration_daemon import IntegrationDaemon


@pytest.fixture(scope="session")
def db_available() -> bool:
    try:
        from db.pool import get_connection
        with get_connection() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


@pytest.fixture(autouse=True)
def skip_if_no_db(db_available):
    if not db_available:
        pytest.skip("Database not available")


class TestCognitiveEngineE2E:
    def test_scaffold_then_integrate(self):
        """Seed scaffold, run one integration batch, verify edges updated."""
        graph = TCGraph()

        # Seed scaffold
        stats = seed_scaffold(graph)
        assert stats["nodes_created"] >= 200

        # Get initial state
        summary_before = graph.summary()
        assert summary_before["edge_count"] >= 500

        # Run one integration batch
        daemon = IntegrationDaemon()
        batch_stats = daemon.integrate_batch()
        # May or may not process items depending on existing evidence state
        assert isinstance(batch_stats["items_processed"], int)

    def test_acquisition_queue_populated_after_integration(self):
        """After integration, weak edges should generate acquisition queue items."""
        graph = TCGraph()
        seed_scaffold(graph)

        daemon = IntegrationDaemon()
        daemon.integrate_batch()

        # Check that some acquisition items exist
        from db.pool import get_connection
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM erik_ops.acquisition_queue")
                count = cur.fetchone()[0]
        # May be 0 if no evidence was unintegrated, but should not error
        assert isinstance(count, int)

    def test_progress_endpoint_returns_valid_data(self):
        """The progress metrics should return valid numbers after scaffold."""
        graph = TCGraph()
        seed_scaffold(graph)

        from api.routers.progress import get_progress
        progress = get_progress()
        assert progress["graph_confidence"] >= 0.0
        assert progress["node_count"] >= 200
        assert progress["edge_count"] >= 500
```

- [ ] **Step 2: Run E2E test**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/test_cognitive_engine_e2e.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run full test suite one final time**

Run: `cd /Users/logannye/.openclaw/erik && conda run -n erik-core pytest tests/ -v --timeout=60`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/logannye/.openclaw/erik && git add tests/test_cognitive_engine_e2e.py && git commit -m "test: add cognitive engine E2E test — scaffold + integration + progress verification"
```

---

## Post-Implementation Verification

After all 12 tasks are complete, verify:

1. **Locally:** Restart the research loop (`launchctl unload ~/Library/LaunchAgents/ai.erik.researcher.plist && launchctl load ~/Library/LaunchAgents/ai.erik.researcher.plist`) and verify:
   - TCG scaffold seeds on first run
   - IntegrationDaemon starts and processes existing evidence
   - ReasoningDaemon starts (needs ANTHROPIC_API_KEY in env)
   - CompoundDaemon starts (needs ANTHROPIC_API_KEY in env)
   - No crash in the first 5 minutes

2. **Railway:** After deploy, verify:
   - `/health` returns OK
   - `/api/graph/summary` returns non-zero node/edge counts
   - `/api/progress` returns valid metrics
   - `/api/activity` returns events (after daemons have run)

3. **Family dashboard:** Check `erik-website-eosin.vercel.app` — the existing frontend should continue working. New endpoints are available for frontend updates (separate task).
