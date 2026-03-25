CREATE SCHEMA IF NOT EXISTS erik_core;
CREATE EXTENSION IF NOT EXISTS citext;
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS erik_core.objects (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    schema_version TEXT NOT NULL DEFAULT '1.0',
    tenant_id TEXT NOT NULL DEFAULT 'erik_default',
    status TEXT NOT NULL DEFAULT 'active',
    time_observed_at TIMESTAMPTZ,
    time_effective_at TIMESTAMPTZ,
    time_recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    time_valid_from TIMESTAMPTZ,
    time_valid_to TIMESTAMPTZ,
    provenance_source_system TEXT,
    provenance_artifact_id TEXT,
    provenance_asserted_by TEXT,
    provenance_trace_id TEXT,
    confidence REAL,
    privacy_class TEXT NOT NULL DEFAULT 'restricted',
    body JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_objects_type ON erik_core.objects(type);
CREATE INDEX IF NOT EXISTS idx_objects_status ON erik_core.objects(status);
CREATE INDEX IF NOT EXISTS idx_objects_body ON erik_core.objects USING gin(body);

CREATE TABLE IF NOT EXISTS erik_core.entities (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name CITEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence REAL DEFAULT 0.5,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    pch_layer INTEGER NOT NULL DEFAULT 1,
    evidence_type TEXT,
    provenance TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_entities_type ON erik_core.entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON erik_core.entities(name);

CREATE TABLE IF NOT EXISTS erik_core.relationships (
    id TEXT PRIMARY KEY,
    source_id TEXT NOT NULL REFERENCES erik_core.entities(id),
    target_id TEXT NOT NULL REFERENCES erik_core.entities(id),
    relationship_type TEXT NOT NULL,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence REAL DEFAULT 0.5,
    evidence TEXT,
    sources JSONB NOT NULL DEFAULT '[]'::jsonb,
    pch_layer INTEGER NOT NULL DEFAULT 1,
    evidence_type TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON erik_core.relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON erik_core.relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_type ON erik_core.relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_rel_pch ON erik_core.relationships(pch_layer);

CREATE TABLE IF NOT EXISTS erik_core.embeddings (
    id TEXT PRIMARY KEY,
    object_ref TEXT NOT NULL,
    object_type TEXT NOT NULL,
    embedding vector(384),
    text_content TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
