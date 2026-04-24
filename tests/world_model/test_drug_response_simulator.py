"""Pure-function tests for DrugResponseSimulator.

No DB required. These are the load-bearing tests — if simulate() drifts
without the corresponding GENERATOR_VERSION bump, the entire calibration
trail in simulated_prediction_claims becomes meaningless.
"""
from __future__ import annotations

import pytest

from world_model.drug_response_simulator import (
    EdgeSnapshot,
    GENERATOR_VERSION,
    Intervention,
    PatientBaseline,
    hash_for,
    simulate,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def erik_baseline() -> PatientBaseline:
    return PatientBaseline(
        patient_id="patient:erik_draper",
        alsfrs_r=43.0,
        nfl_pg_ml=5.82,
        fvc_pct=100.0,
        trajectory_modifier=1.0,
    )


@pytest.fixture
def riluzole_intervention() -> Intervention:
    return Intervention(
        compound_entity_id="compound:riluzole",
        target_entity_ids=("mechanism:glutamate_excitotoxicity",),
        dose_descriptor="standard",
    )


def _rct_edge(source: str, target: str, mean: float, std: float = 0.05,
              scale: str = "alsfrs_r_slope_delta", conf: float = 0.9) -> EdgeSnapshot:
    return EdgeSnapshot(
        source_entity_id=source,
        target_entity_id=target,
        edge_kind="causal",
        effect_mean=mean,
        effect_std=std,
        effect_scale=scale,
        identification_algorithm="rct",
        identification_confidence=conf,
    )


# ─── Determinism ──────────────────────────────────────────────────────────────


def test_same_inputs_produce_identical_outputs(erik_baseline, riluzole_intervention):
    """Replay is byte-identical under the same seed + inputs."""
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.15)]
    a = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=42)
    b = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=42)
    assert a.alsfrs_r_slope_delta_mean == b.alsfrs_r_slope_delta_mean
    assert [s.alsfrs_r_slope_delta for s in a.samples] == \
           [s.alsfrs_r_slope_delta for s in b.samples]
    assert [s.rng_seed for s in a.samples] == [s.rng_seed for s in b.samples]


def test_different_seeds_produce_different_samples(erik_baseline, riluzole_intervention):
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.15, std=0.1)]
    a = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=1)
    b = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=2)
    deltas_a = [s.alsfrs_r_slope_delta for s in a.samples]
    deltas_b = [s.alsfrs_r_slope_delta for s in b.samples]
    assert deltas_a != deltas_b


def test_sample_seeds_are_derived_not_shared(erik_baseline, riluzole_intervention):
    """Each sample uses a distinct RNG seed — otherwise K samples would collapse to 1."""
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.15, std=0.1)]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=8, seed=0)
    seeds = [s.rng_seed for s in result.samples]
    assert len(set(seeds)) == len(seeds)


# ─── Empty / partial DAG ─────────────────────────────────────────────────────


def test_empty_edges_returns_natural_history(erik_baseline, riluzole_intervention):
    result = simulate(erik_baseline, riluzole_intervention, edges=[], K=4, seed=0)
    assert result.empty_dag_reason == 'no_edges_supplied'
    assert result.alsfrs_r_slope_delta_mean == 0.0
    # Trajectory still materializes — baseline decline continues.
    first = result.samples[0]
    assert first.alsfrs_r[0] == erik_baseline.alsfrs_r
    # At 12 months, ALSFRS-R has declined per natural history (~-1 pt/mo).
    assert first.alsfrs_r[-1] < erik_baseline.alsfrs_r
    assert first.alsfrs_r[-1] > 0.0


def test_edges_not_touching_intervention_flagged(erik_baseline, riluzole_intervention):
    """Edges on unrelated entities produce empty_dag_reason='no_edges_touch_intervention'."""
    unrelated = _rct_edge("gene:SOD1", "mechanism:protein_aggregation", 0.2)
    result = simulate(erik_baseline, riluzole_intervention, edges=[unrelated], K=4, seed=0)
    assert result.empty_dag_reason == 'no_edges_touch_intervention'
    assert result.alsfrs_r_slope_delta_mean == 0.0


# ─── Effect-scale handling ────────────────────────────────────────────────────


def test_direct_alsfrs_scale_moves_trajectory(erik_baseline, riluzole_intervention):
    """alsfrs_r_slope_delta edges directly slow decline."""
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.5, std=0.01)]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=42)
    # Mean slope delta should be close to 0.5 * identification_confidence (0.9) = 0.45,
    # clamped by per-edge max 1.5 (not triggered here).
    assert 0.35 <= result.alsfrs_r_slope_delta_mean <= 0.55


def test_motor_neuron_survival_pct_scales_into_slope(erik_baseline, riluzole_intervention):
    """motor_neuron_survival_pct contributes indirectly at factor 0.02 pts/mo per 1%."""
    edges = [_rct_edge(
        "compound:riluzole", "mechanism:glutamate_excitotoxicity",
        mean=10.0, std=0.01,  # +10% survival
        scale="motor_neuron_survival_pct",
    )]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=42)
    # 10% survival * 0.02 conversion * 0.9 conf ≈ 0.18 pts/mo
    assert 0.12 <= result.alsfrs_r_slope_delta_mean <= 0.25


def test_potency_scale_is_gate_not_contributor(erik_baseline, riluzole_intervention):
    """ec50/ic50 edges must not add a slope contribution (they are PD gates)."""
    edges = [_rct_edge(
        "compound:riluzole", "mechanism:glutamate_excitotoxicity",
        mean=-8.0, std=0.01,  # 10nM binding, strong
        scale="ec50_log_nm",
    )]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=4, seed=42)
    assert result.alsfrs_r_slope_delta_mean == 0.0
    # But the edge_contributions trace must record the potency_gate_only note.
    notes = [c.get('note') for c in result.samples[0].edge_contributions]
    assert 'potency_gate_only' in notes


def test_unhandled_scale_is_ignored(erik_baseline, riluzole_intervention):
    edges = [_rct_edge(
        "compound:riluzole", "mechanism:glutamate_excitotoxicity",
        mean=0.5, scale="some_future_scale",
    )]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=4, seed=42)
    assert result.alsfrs_r_slope_delta_mean == 0.0
    notes = [c.get('note') for c in result.samples[0].edge_contributions]
    assert 'unhandled_scale' in notes


# ─── Per-edge clamp ───────────────────────────────────────────────────────────


def test_outlier_edge_is_clamped(erik_baseline, riluzole_intervention):
    """A single preposterous effect can't swing the whole ensemble by >1.5 pts/mo."""
    edges = [_rct_edge(
        "compound:riluzole", "mechanism:glutamate_excitotoxicity",
        mean=100.0, std=0.01, conf=1.0,
    )]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=16, seed=42)
    # Clamp at 1.5 pts/mo regardless of raw sample size.
    assert result.alsfrs_r_slope_delta_mean <= 1.5 + 1e-6


# ─── Confidence attenuation ───────────────────────────────────────────────────


def test_low_confidence_edge_contributes_less(erik_baseline, riluzole_intervention):
    """An identical edge with lower identification_confidence produces smaller slope."""
    high = _rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.5, std=0.01, conf=0.9)
    low = _rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.5, std=0.01, conf=0.2)
    r_high = simulate(erik_baseline, riluzole_intervention, [high], K=16, seed=42)
    r_low = simulate(erik_baseline, riluzole_intervention, [low], K=16, seed=42)
    assert r_high.alsfrs_r_slope_delta_mean > r_low.alsfrs_r_slope_delta_mean


# ─── Baseline hashing ─────────────────────────────────────────────────────────


def test_baseline_hash_is_stable_under_value_roundtrip():
    a = PatientBaseline("p:x", 43.0, 5.82, 100.0)
    b = PatientBaseline("p:x", 43.0, 5.82, 100.0)
    assert hash_for(a) == hash_for(b)


def test_baseline_hash_differs_on_any_field_change():
    base = PatientBaseline("p:x", 43.0, 5.82, 100.0)
    assert hash_for(base) != hash_for(PatientBaseline("p:x", 42.9, 5.82, 100.0))
    assert hash_for(base) != hash_for(PatientBaseline("p:x", 43.0, 5.83, 100.0))
    assert hash_for(base) != hash_for(PatientBaseline("p:y", 43.0, 5.82, 100.0))


# ─── Sanity / schema invariants ───────────────────────────────────────────────


def test_trajectory_timepoints_always_include_zero(erik_baseline, riluzole_intervention):
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.15)]
    result = simulate(erik_baseline, riluzole_intervention, edges,
                      K=4, seed=0, timepoints_months=(3, 6, 12))
    first = result.samples[0]
    assert first.months[0] == 0  # always prepended


def test_alsfrsr_never_exceeds_48_nor_goes_negative(erik_baseline, riluzole_intervention):
    """Clamp invariants hold across horizons."""
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 5.0, std=0.01, conf=1.0)]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=4, seed=0, horizon_months=60)
    for s in result.samples:
        for v in s.alsfrs_r:
            assert 0.0 <= v <= 48.0


def test_survival_monotonically_nonincreasing(erik_baseline, riluzole_intervention):
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.2)]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=4, seed=0)
    for s in result.samples:
        for i in range(1, len(s.survival_prob)):
            assert s.survival_prob[i] <= s.survival_prob[i - 1] + 1e-9


def test_generator_version_carried_through(erik_baseline, riluzole_intervention):
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.15)]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=4, seed=0)
    assert result.generator_version == GENERATOR_VERSION


def test_k_must_be_positive(erik_baseline, riluzole_intervention):
    with pytest.raises(ValueError):
        simulate(erik_baseline, riluzole_intervention, edges=[], K=0)


def test_horizon_must_be_positive(erik_baseline, riluzole_intervention):
    with pytest.raises(ValueError):
        simulate(erik_baseline, riluzole_intervention, edges=[], K=4, horizon_months=0)


def test_ci_bounds_span_mean(erik_baseline, riluzole_intervention):
    edges = [_rct_edge("compound:riluzole", "mechanism:glutamate_excitotoxicity", 0.3, std=0.2)]
    result = simulate(erik_baseline, riluzole_intervention, edges, K=64, seed=0)
    assert result.alsfrs_r_slope_delta_ci_lower <= result.alsfrs_r_slope_delta_mean
    assert result.alsfrs_r_slope_delta_mean <= result.alsfrs_r_slope_delta_ci_upper
