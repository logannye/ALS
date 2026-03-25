CREATE SCHEMA IF NOT EXISTS erik_ops;

CREATE TABLE IF NOT EXISTS erik_ops.audit_events (
    id BIGSERIAL PRIMARY KEY,
    event_type TEXT NOT NULL,
    object_id TEXT,
    object_type TEXT,
    actor TEXT NOT NULL DEFAULT 'erik_system',
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    trace_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_audit_type ON erik_ops.audit_events(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_object ON erik_ops.audit_events(object_id);
CREATE INDEX IF NOT EXISTS idx_audit_time ON erik_ops.audit_events(created_at);

CREATE TABLE IF NOT EXISTS erik_ops.config_snapshots (
    id BIGSERIAL PRIMARY KEY,
    config JSONB NOT NULL,
    changed_keys TEXT[],
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
