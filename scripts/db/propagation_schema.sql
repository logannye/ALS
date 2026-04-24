-- =============================================================================
-- Erik propagation_events — append-only refutation-propagation log.
-- Ported from Galen Track B M.5 (2026-04-18). Adapted for Erik's
-- compound-discovery mission: the sole consumer at Week 1 is R4
-- (compound-mechanism-refute), which deprecates CompoundDossiers and
-- tcg_hypotheses that cite an scm_edge after that edge gets superseded
-- or invalidated with lower-confidence alternative evidence.
--
-- Append-only invariants (enforced by trigger below):
--   * No UPDATE of status='applied' rows except by propagation_rollback.py
--     (which sets a specific rollback_lineage_id).
--   * No DELETE ever.
--   * applied_change captures a full 'before' snapshot so rollback is
--     deterministic.
--
-- Bounded blast radius:
--   affected_object_ids is a hard-capped array (default cap 20, enforced in
--   the Python applier, not here — we want the full intended blast radius
--   recorded even when the applier truncates it).
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS erik_ops;

CREATE TABLE IF NOT EXISTS erik_ops.propagation_events (
    id                      BIGSERIAL PRIMARY KEY,
    rule_kind               TEXT NOT NULL CHECK (rule_kind IN (
                              'R1_abstraction_deprecation',
                              'R2_insight_delta_revise',
                              'R3_tree_node_refute',
                              'R4_compound_mechanism_refute')),
    -- Source event in scm_write_log that triggered this propagation.
    source_write_log_id     BIGINT REFERENCES erik_ops.scm_write_log(id),
    -- What the source event refuted / changed.
    refuted_scm_edge_id     BIGINT REFERENCES erik_ops.scm_edges(id),
    -- Objects affected by propagation. Types:
    --   'tcg_hypothesis', 'tcg_edge', 'intervention', 'compound_dossier'.
    affected_object_ids     TEXT[] NOT NULL DEFAULT '{}',
    affected_object_types   TEXT[] NOT NULL DEFAULT '{}',
    -- Applier state-machine: proposed → applied → rolled_back.
    status                  TEXT NOT NULL DEFAULT 'proposed'
                              CHECK (status IN ('proposed','applied','rolled_back','rejected')),
    -- Full before-state so rollback is atomic. Keyed by affected_object_id.
    --   Shape: { object_id: { ...full pre-change row... }, ... }
    applied_change          JSONB DEFAULT '{}'::jsonb,
    -- Why-line for humans.
    reason                  TEXT,
    -- For rollback chaining: if a rollback event is created it links back here.
    rollback_lineage_id     BIGINT REFERENCES erik_ops.propagation_events(id),
    daemon                  TEXT NOT NULL DEFAULT 'compound_refutation',
    proposed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    applied_at              TIMESTAMPTZ,
    rolled_back_at          TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS propagation_events_source_idx
    ON erik_ops.propagation_events(source_write_log_id);
CREATE INDEX IF NOT EXISTS propagation_events_refuted_edge_idx
    ON erik_ops.propagation_events(refuted_scm_edge_id);
CREATE INDEX IF NOT EXISTS propagation_events_rule_idx
    ON erik_ops.propagation_events(rule_kind, proposed_at DESC);
CREATE INDEX IF NOT EXISTS propagation_events_status_idx
    ON erik_ops.propagation_events(status, proposed_at DESC);
CREATE INDEX IF NOT EXISTS propagation_events_lineage_idx
    ON erik_ops.propagation_events(rollback_lineage_id)
    WHERE rollback_lineage_id IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Append-only guard.
-- Permitted transitions (enforced here, not in Python):
--   proposed  → applied        (setting applied_at, applied_change)
--   proposed  → rejected
--   applied   → rolled_back    (setting rolled_back_at, rollback_lineage_id)
-- All other UPDATE/DELETE is forbidden. This is load-bearing for audit.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION erik_ops.propagation_events_append_only()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'propagation_events is append-only; DELETE forbidden';
    END IF;
    IF TG_OP = 'UPDATE' THEN
        -- Only status transitions defined above are permitted.
        IF OLD.status = 'applied' AND NEW.status = 'rolled_back' THEN
            RETURN NEW;
        END IF;
        IF OLD.status = 'proposed' AND NEW.status IN ('applied', 'rejected') THEN
            RETURN NEW;
        END IF;
        -- Identity update (no status change) is allowed only on timestamp columns.
        IF OLD.status = NEW.status
           AND OLD.rule_kind = NEW.rule_kind
           AND OLD.source_write_log_id IS NOT DISTINCT FROM NEW.source_write_log_id
           AND OLD.refuted_scm_edge_id IS NOT DISTINCT FROM NEW.refuted_scm_edge_id
           AND OLD.affected_object_ids = NEW.affected_object_ids
           AND OLD.applied_change::text = NEW.applied_change::text THEN
            RETURN NEW;
        END IF;
        RAISE EXCEPTION
            'propagation_events: forbidden status transition % → % (id=%)',
            OLD.status, NEW.status, OLD.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS propagation_events_append_only_trigger ON erik_ops.propagation_events;
CREATE TRIGGER propagation_events_append_only_trigger
    BEFORE UPDATE OR DELETE ON erik_ops.propagation_events
    FOR EACH ROW EXECUTE FUNCTION erik_ops.propagation_events_append_only();

-- -----------------------------------------------------------------------------
-- Cursor table: tracks the last scm_write_log.id the propagation daemon
-- consumed per rule_kind. Idempotent replay = safe restarts.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.propagation_cursor (
    rule_kind           TEXT PRIMARY KEY,
    last_write_log_id   BIGINT NOT NULL DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
