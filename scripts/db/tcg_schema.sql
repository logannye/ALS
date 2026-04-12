-- scripts/db/tcg_schema.sql
-- Therapeutic Causal Graph schema for Erik Cognitive Engine

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

CREATE INDEX IF NOT EXISTS idx_llm_spend_created ON erik_ops.llm_spend(created_at DESC);

ALTER TABLE erik_core.objects ADD COLUMN IF NOT EXISTS tcg_integrated BOOLEAN DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS idx_objects_tcg_integrated ON erik_core.objects(tcg_integrated)
    WHERE tcg_integrated = FALSE;
