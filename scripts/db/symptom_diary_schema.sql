-- =============================================================================
-- Family symptom diary — "how is Erik feeling today?" reports from family.
--
-- Why append-only: these are first-person observations from family members.
-- Past reports are never edited — they're testimony. If a reporter made a
-- typo they write a correction as a new report.
--
-- mood is a 1-5 integer: 1 = very bad day, 5 = very good day. Deliberately
-- simple — we're optimising for one-tap reporting, not structured data.
--
-- note is a short free-text field (max 2 KB). The research loop can use
-- these as qualitative evidence when building/validating its patient model.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS erik_ops;

CREATE TABLE IF NOT EXISTS erik_ops.symptom_reports (
    id              BIGSERIAL PRIMARY KEY,
    reporter_name   TEXT NOT NULL DEFAULT 'family',
    mood            SMALLINT CHECK (mood IS NULL OR (mood BETWEEN 1 AND 5)),
    note            TEXT NOT NULL DEFAULT '' CHECK (LENGTH(note) <= 2048),
    -- Free-form tags so the backend can coarse-categorise without
    -- forcing the family into a dropdown. Populated by a simple
    -- keyword extractor on the ingest path.
    symptoms_mentioned TEXT[] NOT NULL DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS symptom_reports_created_idx
    ON erik_ops.symptom_reports(created_at DESC);
CREATE INDEX IF NOT EXISTS symptom_reports_reporter_idx
    ON erik_ops.symptom_reports(reporter_name, created_at DESC);

-- Append-only guard (same pattern as propagation_events).
CREATE OR REPLACE FUNCTION erik_ops.symptom_reports_append_only()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'symptom_reports is append-only; UPDATE forbidden';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'symptom_reports is append-only; DELETE forbidden';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS symptom_reports_append_only_trigger ON erik_ops.symptom_reports;
CREATE TRIGGER symptom_reports_append_only_trigger
    BEFORE UPDATE OR DELETE ON erik_ops.symptom_reports
    FOR EACH ROW EXECUTE FUNCTION erik_ops.symptom_reports_append_only();
