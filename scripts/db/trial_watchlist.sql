CREATE TABLE IF NOT EXISTS erik_ops.trial_watchlist (
    nct_id          TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    eligible_status TEXT NOT NULL DEFAULT 'likely',
    last_checked    TIMESTAMPTZ NOT NULL DEFAULT now(),
    enrollment_status TEXT NOT NULL DEFAULT '',
    phase           TEXT NOT NULL DEFAULT '',
    intervention_name TEXT NOT NULL DEFAULT '',
    protocol_alignment FLOAT NOT NULL DEFAULT 0.0,
    sites           JSONB NOT NULL DEFAULT '[]'::jsonb,
    reviewed        BOOLEAN NOT NULL DEFAULT false,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_watchlist_eligible
    ON erik_ops.trial_watchlist (eligible_status)
    WHERE eligible_status IN ('yes', 'likely', 'pending_data');
