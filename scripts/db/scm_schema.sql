-- =============================================================================
-- Erik SCM Foundation — KG ⊥ SCM two-graph architecture (Phase 1 port)
-- Ported from Galen Phase 1 (2026-04-16). Adapted for ALS compound-discovery.
--
-- Why this split:
--   erik_core.relationships.pch_layer conflates corroboration-tier (how well
--   the observation is supported) with causal-identification-tier (can we
--   do-calculus this claim). That conflation is fine for knowledge accumulation
--   but dangerous for compound discovery: picking the intervention node in a
--   mechanism chain requires knowing whether each edge is observational or
--   identified-causal, and under which adjustment set.
--
-- Mission fit (Erik-specific, not in Galen):
--   scm_edges is the substrate for intervention-point selection. A compound's
--   mechanism-of-action claim is an scm_edge with source=compound-entity,
--   target=mechanism-node, edge_kind='causal', identification_algorithm chosen
--   from ALS-relevant algorithms (ipsc_motor_neuron_assay, als_mouse_model, etc.).
--   CompoundDossier (future table) will reference scm_edges.id to anchor every
--   claim about every candidate compound.
--
-- Invariants enforced by the schema (not Python):
--   1. Single active edge per (src, tgt, adjustment_set) — partial unique index.
--   2. Supersession chain via superseded_by — non-cyclic by construction
--      (SCMWriter checks before writing).
--   3. identification_confidence ∈ [0,1] — CHECK constraint.
--   4. edge_kind ⊥ node_class — nodes have semantic class, edges have graph role.
--
-- Migration strategy: additive-only. erik_core.relationships gains a
-- nullable scm_edge_id FK; pch_layer stays in place during the window.
-- Drop scheduled only after two weeks of parity validation.
-- =============================================================================

CREATE SCHEMA IF NOT EXISTS erik_ops;

-- -----------------------------------------------------------------------------
-- scm_nodes: causal variables in the SCM.
-- Node class captures role in causal queries (treatment / outcome / confounder
-- / mediator / instrument / covariate). Role is semantic, separate from the
-- graph position — a node may be a confounder in one query and a covariate in
-- another. Store the default role here; query-specific overrides go in
-- scm_adjustment_sets.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_nodes (
    id                  BIGSERIAL PRIMARY KEY,
    entity_id           TEXT NOT NULL UNIQUE,  -- FK-by-convention to erik_core.entities.id
    node_class          TEXT NOT NULL CHECK (node_class IN (
                           'treatment','outcome','covariate','confounder','mediator','instrument')),
    markov_blanket      BIGINT[] DEFAULT '{}',
    domain_constraints  JSONB DEFAULT '{}'::jsonb,
    -- Erik-specific: tags for compound-discovery relevance.
    -- als_role: one of 'gene','mechanism','pathway','compound','biomarker','clinical_endpoint'.
    -- druggability_prior: [0,1] estimate of how druggable this node is as an intervention point.
    als_role            TEXT CHECK (als_role IN (
                           'gene','mechanism','pathway','compound','biomarker','clinical_endpoint','other')),
    druggability_prior  DOUBLE PRECISION CHECK (druggability_prior BETWEEN 0 AND 1),
    metadata            JSONB DEFAULT '{}'::jsonb,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS scm_nodes_entity_id_idx ON erik_ops.scm_nodes(entity_id);
CREATE INDEX IF NOT EXISTS scm_nodes_class_idx     ON erik_ops.scm_nodes(node_class);
CREATE INDEX IF NOT EXISTS scm_nodes_als_role_idx  ON erik_ops.scm_nodes(als_role);
CREATE INDEX IF NOT EXISTS scm_nodes_druggable_idx ON erik_ops.scm_nodes(druggability_prior DESC NULLS LAST)
    WHERE als_role IN ('gene','mechanism','pathway');

-- -----------------------------------------------------------------------------
-- scm_adjustment_sets: which variables to condition on for a given query.
-- Stored separately so the same edge can be valid under multiple adjustment
-- sets (e.g. backdoor vs. frontdoor on the same pair).
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_adjustment_sets (
    id              BIGSERIAL PRIMARY KEY,
    set_kind        TEXT NOT NULL CHECK (set_kind IN (
                      'backdoor','frontdoor','iv','do_see','minimal_backdoor','empty')),
    variable_ids    BIGINT[] NOT NULL,
    minimality      BOOLEAN DEFAULT FALSE,
    validity_proof  JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS scm_adj_sets_variables_idx
    ON erik_ops.scm_adjustment_sets USING GIN (variable_ids);
CREATE INDEX IF NOT EXISTS scm_adj_sets_kind_idx ON erik_ops.scm_adjustment_sets(set_kind);

-- -----------------------------------------------------------------------------
-- scm_edges: identified causal (or confounding, or mediating) relationships.
-- This is the do-calculus-valid surface — distinct from erik_core.relationships
-- which accumulates observations of any strength.
--
-- Erik-specific extensions vs. Galen:
--   * effect_scale adds alsfrs_r_slope_delta, motor_neuron_survival_pct,
--     ec50_log_nm, target_occupancy_pct (compound-discovery units).
--   * identification_algorithm adds ipsc_motor_neuron_assay, als_mouse_model,
--     patient_organoid, docking_simulation, cmap_signature_match, galen_scm
--     (inherited from cross-disease causal evidence).
--   * is_intervention_candidate flags edges whose source node is a
--     druggable intervention point being considered for Erik's protocol.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_edges (
    id                          BIGSERIAL PRIMARY KEY,
    source_node_id              BIGINT NOT NULL REFERENCES erik_ops.scm_nodes(id),
    target_node_id              BIGINT NOT NULL REFERENCES erik_ops.scm_nodes(id),
    edge_kind                   TEXT NOT NULL CHECK (edge_kind IN ('causal','confounding','mediating')),
    effect_mean                 DOUBLE PRECISION,
    effect_std                  DOUBLE PRECISION,
    effect_ci_lower             DOUBLE PRECISION,
    effect_ci_upper             DOUBLE PRECISION,
    effect_scale                TEXT CHECK (effect_scale IN (
                                  'log_odds','hazard_ratio','ic50_log_nm','ec50_log_nm',
                                  'delta_continuous','risk_ratio','binding_affinity_kd',
                                  'alsfrs_r_slope_delta','motor_neuron_survival_pct',
                                  'target_occupancy_pct')),
    identification_algorithm    TEXT NOT NULL CHECK (identification_algorithm IN (
                                  'rct','crispr_interventional','pc_algorithm',
                                  'frontdoor_adj','iv','regression_discontinuity',
                                  'scm_counterfactual','experimental_assay',
                                  'replicated_experiment',
                                  'ipsc_motor_neuron_assay','als_mouse_model',
                                  'patient_organoid','docking_simulation',
                                  'cmap_signature_match','galen_scm')),
    identification_confidence   DOUBLE PRECISION NOT NULL
                                  CHECK (identification_confidence BETWEEN 0 AND 1),
    adjustment_set_id           BIGINT REFERENCES erik_ops.scm_adjustment_sets(id),
    derived_from_rel_ids        TEXT[] NOT NULL DEFAULT '{}',
    transport_population        TEXT,
    transport_conditions        JSONB DEFAULT '{}'::jsonb,
    is_intervention_candidate   BOOLEAN NOT NULL DEFAULT FALSE,
    identified_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reverified_at               TIMESTAMPTZ,
    expires_at                  TIMESTAMPTZ,
    superseded_by               BIGINT REFERENCES erik_ops.scm_edges(id),
    status                      TEXT NOT NULL DEFAULT 'active'
                                  CHECK (status IN ('active','superseded','invalidated','pending_reverify')),
    metadata                    JSONB DEFAULT '{}'::jsonb,
    created_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS scm_edges_unique_active
    ON erik_ops.scm_edges(source_node_id, target_node_id, COALESCE(adjustment_set_id, 0))
    WHERE status = 'active';
CREATE INDEX IF NOT EXISTS scm_edges_src_tgt_idx
    ON erik_ops.scm_edges(source_node_id, target_node_id) WHERE status='active';
CREATE INDEX IF NOT EXISTS scm_edges_algorithm_idx ON erik_ops.scm_edges(identification_algorithm);
CREATE INDEX IF NOT EXISTS scm_edges_expires_idx
    ON erik_ops.scm_edges(expires_at) WHERE status='active';
CREATE INDEX IF NOT EXISTS scm_edges_status_idx ON erik_ops.scm_edges(status);
CREATE INDEX IF NOT EXISTS scm_edges_identified_at_idx ON erik_ops.scm_edges(identified_at DESC);
CREATE INDEX IF NOT EXISTS scm_edges_intervention_idx
    ON erik_ops.scm_edges(is_intervention_candidate) WHERE is_intervention_candidate = TRUE AND status='active';

-- -----------------------------------------------------------------------------
-- scm_identifications: audit log of every identification attempt.
-- Successful attempts have edge_id set; unidentifiable/rejected attempts still
-- log here so we know we tried. Analytical source for calibrating
-- identification_confidence over time.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_identifications (
    id                      BIGSERIAL PRIMARY KEY,
    edge_id                 BIGINT REFERENCES erik_ops.scm_edges(id),
    source_node_id          BIGINT NOT NULL REFERENCES erik_ops.scm_nodes(id),
    target_node_id          BIGINT NOT NULL REFERENCES erik_ops.scm_nodes(id),
    algorithm               TEXT NOT NULL,
    outcome                 TEXT NOT NULL CHECK (outcome IN (
                              'identified','unidentifiable','pending',
                              'rejected_weaker','cycle_detected')),
    confidence              DOUBLE PRECISION,
    evidence_refs           TEXT[] DEFAULT '{}',
    trace                   JSONB DEFAULT '{}'::jsonb,
    ran_at                  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    runtime_ms              INTEGER
);
CREATE INDEX IF NOT EXISTS scm_idents_edge_idx    ON erik_ops.scm_identifications(edge_id);
CREATE INDEX IF NOT EXISTS scm_idents_src_tgt_idx ON erik_ops.scm_identifications(source_node_id, target_node_id);
CREATE INDEX IF NOT EXISTS scm_idents_outcome_idx ON erik_ops.scm_identifications(outcome);
CREATE INDEX IF NOT EXISTS scm_idents_ran_at_idx  ON erik_ops.scm_identifications(ran_at DESC);

-- -----------------------------------------------------------------------------
-- scm_cf_traces: stored counterfactual queries and their results.
-- For Erik, the most important CF query is the patient-specific one:
--   "Under do(compound=X), what is Erik's expected ALSFRS-R slope?"
-- The trace captures abduction state (Erik's current biomarkers), intervention,
-- factual baseline, counterfactual prediction, regret bound.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_cf_traces (
    id                      BIGSERIAL PRIMARY KEY,
    edge_id                 BIGINT NOT NULL REFERENCES erik_ops.scm_edges(id),
    query_id                TEXT NOT NULL UNIQUE,
    abduction_state         JSONB NOT NULL,
    intervention_do         JSONB NOT NULL,
    factual_outcome         JSONB NOT NULL,
    counterfactual_outcome  JSONB NOT NULL,
    regret                  DOUBLE PRECISION,
    -- Erik-specific: patient context (which snapshot of Erik's state this CF is anchored to).
    patient_snapshot_id     TEXT,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    runtime_ms              INTEGER
);
CREATE INDEX IF NOT EXISTS scm_cf_edge_idx      ON erik_ops.scm_cf_traces(edge_id);
CREATE INDEX IF NOT EXISTS scm_cf_computed_idx  ON erik_ops.scm_cf_traces(computed_at DESC);
CREATE INDEX IF NOT EXISTS scm_cf_patient_idx   ON erik_ops.scm_cf_traces(patient_snapshot_id);

-- -----------------------------------------------------------------------------
-- scm_write_log: append-only log of every SCM mutation.
-- Consumed by downstream daemons (error_propagation in M.5, prospective in
-- M.1, reward attribution in Phase 5). DO NOT truncate.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_write_log (
    id          BIGSERIAL PRIMARY KEY,
    operation   TEXT NOT NULL CHECK (operation IN (
                  'edge_created','edge_superseded','edge_invalidated',
                  'cf_computed','bootstrap_progress','reverification_run',
                  'intervention_flagged','intervention_unflagged')),
    target_id   BIGINT,
    daemon      TEXT NOT NULL,
    payload     JSONB DEFAULT '{}'::jsonb,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS scm_write_log_occurred_idx  ON erik_ops.scm_write_log(occurred_at DESC);
CREATE INDEX IF NOT EXISTS scm_write_log_operation_idx ON erik_ops.scm_write_log(operation);
CREATE INDEX IF NOT EXISTS scm_write_log_daemon_idx    ON erik_ops.scm_write_log(daemon);
CREATE INDEX IF NOT EXISTS scm_write_log_reward_idx
    ON erik_ops.scm_write_log(operation, occurred_at DESC)
    WHERE operation = 'edge_created';

-- -----------------------------------------------------------------------------
-- scm_bootstrap_progress: checkpoint table for the one-shot bootstrap pass
-- that promotes high-confidence erik_core.relationships into scm_edges.
-- Resumable — on restart the daemon picks up from last_rel_id.
-- -----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS erik_ops.scm_bootstrap_progress (
    id                BIGSERIAL PRIMARY KEY,
    evidence_type     TEXT NOT NULL,
    -- Legacy cursor retained for audit; new daemon code uses last_created_at
    -- because erik_core.relationships.id is TEXT and not monotonic under
    -- UUID-style IDs.
    last_rel_id       TEXT,
    last_created_at   TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01'::timestamptz,
    processed_count   BIGINT NOT NULL DEFAULT 0,
    edges_created     BIGINT NOT NULL DEFAULT 0,
    edges_rejected    BIGINT NOT NULL DEFAULT 0,
    status            TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending','running','complete','failed')),
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    error_message     TEXT,
    CONSTRAINT scm_bootstrap_unique_type UNIQUE (evidence_type)
);
-- Backfill for pre-existing installs where the column was added via ALTER.
ALTER TABLE erik_ops.scm_bootstrap_progress
    ADD COLUMN IF NOT EXISTS last_created_at TIMESTAMPTZ NOT NULL DEFAULT '1970-01-01'::timestamptz;

-- -----------------------------------------------------------------------------
-- Add scm_edge_id to erik_core.relationships for KG→SCM linkage.
-- Nullable; populated by SCMWriter when a relationship promotes to an edge.
-- -----------------------------------------------------------------------------
ALTER TABLE erik_core.relationships
    ADD COLUMN IF NOT EXISTS scm_edge_id BIGINT;
CREATE INDEX IF NOT EXISTS idx_rel_scm_edge ON erik_core.relationships(scm_edge_id)
    WHERE scm_edge_id IS NOT NULL;
