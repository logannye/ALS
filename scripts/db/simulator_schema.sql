-- =============================================================================
-- Erik DrugResponseSimulator — immutable rollout log.
-- Ported from Galen CPTS Week 1 (2026-04-24). Erik-specific adaptations:
--
--   * simulate() writes an ensemble (K samples) under one generator_version
--     per call. Each row = one rollout; regret / calibration is computed
--     across rows sharing (intervention_id, baseline_hash, generator_version).
--
--   * Anchored to Erik's baseline snapshot (ALSFRS-R, NfL, FVC at time of sim)
--     via baseline_hash so we can retroactively group rollouts that share a
--     clinical anchor.
--
--   * Append-only trigger + immutability invariant: simulated_trajectory must
--     never be updated or deleted. It is the ground-truth record of what the
--     system predicted at a point in time. M.1 prospective-prediction
--     resolution reads this table; rewriting it would be reward hacking.
--
--   * No FK to scm_edges.id because the simulator captures a DAG snapshot
--     at call time (edge_snapshot JSONB). This decouples retroactive edge
--     supersession from the sim's reproducibility.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS erik_ops;

CREATE TABLE IF NOT EXISTS erik_ops.simulated_trajectory (
    id                      BIGSERIAL PRIMARY KEY,
    -- The compound / intervention we simulated do()-ing.
    intervention_entity_id  TEXT NOT NULL,
    -- Hash of the Erik baseline used for this sim (so we can group rollouts
    -- that share a clinical anchor regardless of what changes later).
    baseline_hash           TEXT NOT NULL,
    baseline_snapshot       JSONB NOT NULL,
    -- The K index within this ensemble. 0 ≤ sample_index < K.
    sample_index            INTEGER NOT NULL,
    ensemble_size           INTEGER NOT NULL CHECK (ensemble_size > 0),
    -- Bump when the simulator implementation changes. Rollouts under
    -- different generator_versions live side-by-side and are NEVER mixed.
    generator_version       TEXT NOT NULL,
    -- The resolved DAG at sim time. Shape:
    --   [{source, target, edge_kind, effect_mean, effect_std, effect_scale, algo, conf}, ...]
    edge_snapshot           JSONB NOT NULL,
    -- The trajectory this sample produced. Shape:
    --   {"months": [0, 1, 3, 6, 12],
    --    "alsfrs_r":      [43.0, 42.1, 40.3, 38.2, 34.5],
    --    "nfl":           [5.82, 5.9, 6.1, 6.4, 7.0],
    --    "fvc":           [100.0, 99.2, 97.5, 94.0, 89.0],
    --    "survival_prob": [1.0, 1.0, 0.99, 0.97, 0.93]}
    trajectory              JSONB NOT NULL,
    -- Deterministic RNG seed that produced this sample. Replay = take the
    -- same seed + edge_snapshot + baseline_snapshot and get the same row.
    rng_seed                BIGINT NOT NULL,
    -- Horizon in months.
    horizon_months          INTEGER NOT NULL DEFAULT 12,
    -- Regret vs. "do nothing" baseline, in ALSFRS-R-slope units.
    -- Positive = the intervention is predicted to slow decline.
    alsfrs_r_slope_delta    DOUBLE PRECISION,
    -- Metadata: simulator config, daemon source, etc.
    metadata                JSONB DEFAULT '{}'::jsonb,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS sim_traj_intervention_idx
    ON erik_ops.simulated_trajectory(intervention_entity_id);
CREATE INDEX IF NOT EXISTS sim_traj_baseline_idx
    ON erik_ops.simulated_trajectory(baseline_hash);
CREATE INDEX IF NOT EXISTS sim_traj_gen_version_idx
    ON erik_ops.simulated_trajectory(generator_version);
CREATE INDEX IF NOT EXISTS sim_traj_ensemble_idx
    ON erik_ops.simulated_trajectory(intervention_entity_id, baseline_hash, generator_version);
CREATE INDEX IF NOT EXISTS sim_traj_computed_idx
    ON erik_ops.simulated_trajectory(computed_at DESC);

-- -----------------------------------------------------------------------------
-- Immutability guard — simulated_trajectory is the system's memory of what
-- it predicted. Rewriting it breaks calibration.
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION erik_ops.simulated_trajectory_immutable()
RETURNS trigger AS $$
BEGIN
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'simulated_trajectory is immutable; UPDATE forbidden';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'simulated_trajectory is immutable; DELETE forbidden';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS simulated_trajectory_immutable_trigger ON erik_ops.simulated_trajectory;
CREATE TRIGGER simulated_trajectory_immutable_trigger
    BEFORE UPDATE OR DELETE ON erik_ops.simulated_trajectory
    FOR EACH ROW EXECUTE FUNCTION erik_ops.simulated_trajectory_immutable();

-- -----------------------------------------------------------------------------
-- simulated_prediction_claims — Erik's M.1 analog (minimal slice for Week 1).
-- Each row is a prospective claim generated from an ensemble (one row per
-- simulate() call, not per sample). Resolution happens via Erik's own
-- trajectory (ingested through ingestion/patient_builder) or PRO-ACT cohort
-- matching. The validator daemon (not implemented in Week 1) will read these.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.simulated_prediction_claims (
    id                      BIGSERIAL PRIMARY KEY,
    intervention_entity_id  TEXT NOT NULL,
    baseline_hash           TEXT NOT NULL,
    generator_version       TEXT NOT NULL,
    -- Prediction summary across the K-sample ensemble.
    -- Shape: {"alsfrs_r_slope_delta_mean": X, "..._std": Y, "..._ci_lower": L, "..._ci_upper": U}
    prediction_summary      JSONB NOT NULL,
    ensemble_size           INTEGER NOT NULL,
    horizon_months          INTEGER NOT NULL,
    -- Status machine: open → resolved → expired.
    status                  TEXT NOT NULL DEFAULT 'open'
                              CHECK (status IN ('open','resolved','expired','superseded')),
    resolution_source       TEXT,  -- 'erik_trajectory' | 'proact_matched' | 'cf_backfill'
    resolution_value        JSONB,
    brier_score             DOUBLE PRECISION,
    daemon                  TEXT NOT NULL DEFAULT 'drug_response_simulator',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at             TIMESTAMPTZ,
    expires_at              TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS sim_pred_intervention_idx
    ON erik_ops.simulated_prediction_claims(intervention_entity_id);
CREATE INDEX IF NOT EXISTS sim_pred_status_idx
    ON erik_ops.simulated_prediction_claims(status, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS sim_pred_unique_open
    ON erik_ops.simulated_prediction_claims(intervention_entity_id, baseline_hash, generator_version)
    WHERE status = 'open';
