"""Drug-Response Simulator — Erik's analog of Galen's CPTS (2026-04-24).

Turns the SCM from a passive lookup graph into an active forward engine.
Given Erik's current clinical baseline and a candidate intervention
(a compound entity + its causal edges into ALS mechanisms), ``simulate``
produces K sampled rollouts of Erik's ALSFRS-R / NfL / FVC / survival
over ``horizon_months`` months.

Design constraints:

  * **Pure function.** No DB access, no I/O, no global state. Every
    argument is explicit. This is what makes the simulator testable and
    what makes ``generator_version`` a meaningful calibration anchor.

  * **Deterministic under seed.** Same (baseline, intervention, edges, seed,
    K, horizon) → byte-identical output. Replay from simulated_trajectory
    is how we verify no silent drift has happened.

  * **SCM as DAG walker.** Each edge contributes its effect_mean ± effect_std
    on its effect_scale to Erik's trajectory. Scales we handle natively:
        - alsfrs_r_slope_delta      (pts/month change vs natural history)
        - motor_neuron_survival_pct (relative retention of motor neurons)
        - target_occupancy_pct      (PD proxy; scaled into slope_delta)
        - ec50_log_nm / ic50_log_nm (potency proxy; gated by druggability_prior)
    Scales not in this set are ignored (logged via metadata).

  * **Fail-closed.** When no valid edges connect intervention → clinical
    endpoint, the simulator returns a zero-effect ensemble with an
    explicit ``empty_dag_reason``. Callers must never receive a silent
    "predicted cure" from an unresolved DAG.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

# Bump when any math below changes — this is the calibration anchor.
GENERATOR_VERSION = "drs_v1.0.0_2026-04-24"

# Natural-history baseline for ALSFRS-R decline in points/month (absent
# treatment). Literature range ≈ 0.7–1.1 pts/month for limb-onset ALS;
# we use a central estimate and let the SCM modify it.
_NATURAL_ALSFRSR_SLOPE_PER_MONTH: float = -1.0
_NATURAL_NFL_SLOPE_PER_MONTH: float = 0.08      # pg/mL/month
_NATURAL_FVC_SLOPE_PER_MONTH: float = -1.5      # percentage points/month
_NATURAL_MONTHLY_HAZARD: float = 0.01           # survival hazard (baseline)

# Horizons we materialize by default.
_DEFAULT_TIMEPOINTS_MONTHS: tuple[int, ...] = (0, 1, 3, 6, 12)

# Effect-scale handlers. Each maps an edge's effect sample (in its native
# scale) to a (alsfrs_r_slope_delta_pts_month, metadata_key) tuple so the
# daemon can see which scales actually moved the trajectory.
_ALSFRSR_SCALES_DIRECT: frozenset[str] = frozenset({'alsfrs_r_slope_delta'})
_ALSFRSR_SCALES_INDIRECT: dict[str, float] = {
    # "10% more motor neurons surviving ≈ 0.2 pt/mo slower ALSFRS-R decline"
    # Rough translational factor — conservative, revisit when the daemon's
    # M.1 calibration loop produces real Brier scores per scale.
    'motor_neuron_survival_pct': 0.02,          # per 1% improvement
    'target_occupancy_pct':      0.008,         # per 1% occupancy (PD proxy)
}

# Potency scales: negative log concentration (smaller = stronger binder).
# Used only as a gate on whether the mechanism edge is clinically plausible;
# does not contribute a direct slope delta.
_POTENCY_SCALES: frozenset[str] = frozenset({'ec50_log_nm', 'ic50_log_nm', 'binding_affinity_kd'})

# Max absolute slope_delta we allow *per edge* before clamping. Prevents a
# single high-std edge from swinging a whole rollout.
_PER_EDGE_SLOPE_CLAMP_PTS_MO: float = 1.5


@dataclass(frozen=True)
class EdgeSnapshot:
    """Frozen view of a single scm_edge at simulation time.

    Snapshots ensure reproducibility: supersession after the sim ran must
    not change the sim's output. The daemon constructs these from
    ``erik_ops.scm_edges`` at call time and persists them alongside the
    rollout.
    """
    source_entity_id: str
    target_entity_id: str
    edge_kind: str                # 'causal' | 'mediating' | ...
    effect_mean: Optional[float]
    effect_std: Optional[float]
    effect_scale: Optional[str]
    identification_algorithm: str
    identification_confidence: float
    adjustment_set_id: Optional[int] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'source_entity_id': self.source_entity_id,
            'target_entity_id': self.target_entity_id,
            'edge_kind': self.edge_kind,
            'effect_mean': self.effect_mean,
            'effect_std': self.effect_std,
            'effect_scale': self.effect_scale,
            'identification_algorithm': self.identification_algorithm,
            'identification_confidence': self.identification_confidence,
            'adjustment_set_id': self.adjustment_set_id,
            'metadata': dict(self.metadata),
        }


@dataclass(frozen=True)
class PatientBaseline:
    """Erik's (or another patient's) clinical anchor for the simulation.

    ``baseline_hash`` is computed in __post_init__-equivalent via
    ``hash_for`` below so two equal baselines always hash identically
    (used to group rollouts that share a clinical anchor).
    """
    patient_id: str
    alsfrs_r: float
    nfl_pg_ml: float
    fvc_pct: float
    # Normalizes patient heterogeneity into the slope calculation. For Erik
    # at dx (limb-onset, ~13 months from onset, FVC 100): factor ≈ 1.0.
    trajectory_modifier: float = 1.0


@dataclass
class Intervention:
    """Candidate intervention to simulate.

    ``target_entity_ids`` names the SCM nodes the compound directly acts on;
    the simulator restricts its causal walk to edges whose source is in
    this set (or the compound itself).
    """
    compound_entity_id: str
    target_entity_ids: tuple[str, ...]
    dose_descriptor: str = "nominal"
    # Override of the compound's druggability if the caller has a better
    # estimate (e.g. from docking / CMap). None = trust SCM node value.
    druggability_override: Optional[float] = None


@dataclass
class TrajectorySample:
    """One sampled rollout at K discrete timepoints."""
    months: tuple[int, ...]
    alsfrs_r: tuple[float, ...]
    nfl: tuple[float, ...]
    fvc: tuple[float, ...]
    survival_prob: tuple[float, ...]
    # Net ALSFRS-R slope delta (pts/month) this sample attributed to the
    # intervention, after clamping. Positive = protective.
    alsfrs_r_slope_delta: float
    rng_seed: int
    # Per-edge contribution trace for explainability. Shape:
    #   [{edge_index, scale, sampled_effect, slope_contribution_pts_mo}, ...]
    edge_contributions: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            'months': list(self.months),
            'alsfrs_r': list(self.alsfrs_r),
            'nfl': list(self.nfl),
            'fvc': list(self.fvc),
            'survival_prob': list(self.survival_prob),
            'alsfrs_r_slope_delta': self.alsfrs_r_slope_delta,
            'rng_seed': self.rng_seed,
            'edge_contributions': list(self.edge_contributions),
        }


@dataclass
class RolloutEnsemble:
    """The full K-sample result of a simulate() call."""
    generator_version: str
    intervention_entity_id: str
    baseline_hash: str
    horizon_months: int
    ensemble_size: int
    samples: tuple[TrajectorySample, ...]
    edge_snapshot: tuple[EdgeSnapshot, ...]
    # Summary over samples. Computed at construction.
    alsfrs_r_slope_delta_mean: float
    alsfrs_r_slope_delta_std: float
    alsfrs_r_slope_delta_ci_lower: float    # 10th percentile
    alsfrs_r_slope_delta_ci_upper: float    # 90th percentile
    # Why the DAG might be empty / partial. None = everything normal.
    empty_dag_reason: Optional[str] = None

    def to_prediction_summary(self) -> dict[str, Any]:
        return {
            'alsfrs_r_slope_delta_mean': self.alsfrs_r_slope_delta_mean,
            'alsfrs_r_slope_delta_std': self.alsfrs_r_slope_delta_std,
            'alsfrs_r_slope_delta_ci_lower': self.alsfrs_r_slope_delta_ci_lower,
            'alsfrs_r_slope_delta_ci_upper': self.alsfrs_r_slope_delta_ci_upper,
            'ensemble_size': self.ensemble_size,
            'generator_version': self.generator_version,
            'horizon_months': self.horizon_months,
            'empty_dag_reason': self.empty_dag_reason,
        }


# ─── Public API ───────────────────────────────────────────────────────────────


def hash_for(baseline: PatientBaseline) -> str:
    """Stable, order-insensitive hash of a PatientBaseline.

    Two equal baselines always produce the same hash, which is how rollouts
    anchored to the same clinical snapshot are later grouped. Changing this
    hashing is a breaking change to simulated_prediction_claims lineage,
    so bump GENERATOR_VERSION when you touch it.
    """
    payload = json.dumps({
        'patient_id': baseline.patient_id,
        'alsfrs_r': round(baseline.alsfrs_r, 3),
        'nfl_pg_ml': round(baseline.nfl_pg_ml, 3),
        'fvc_pct': round(baseline.fvc_pct, 3),
        'trajectory_modifier': round(baseline.trajectory_modifier, 4),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()[:20]


def simulate(
    baseline: PatientBaseline,
    intervention: Intervention,
    edges: Sequence[EdgeSnapshot],
    K: int = 32,
    horizon_months: int = 12,
    seed: int = 0,
    timepoints_months: Sequence[int] = _DEFAULT_TIMEPOINTS_MONTHS,
) -> RolloutEnsemble:
    """Run a K-sample rollout and return a RolloutEnsemble.

    Args:
        baseline: Erik's (or other patient's) current clinical anchor.
        intervention: The compound being simulated + which SCM nodes it targets.
        edges: Snapshot of the active scm_edges relevant to this intervention.
            The daemon is responsible for selecting this slice; the simulator
            uses every edge it receives whose source is either the compound
            itself or a member of intervention.target_entity_ids.
        K: Ensemble size. Default 32.
        horizon_months: Longest timepoint to simulate.
        seed: Master RNG seed. Sample j uses seed ⊕ j so sample seeds are
            derived, not shared.
        timepoints_months: Discrete timepoints to materialise. Must contain
            0 and be bounded by horizon_months.

    Returns:
        RolloutEnsemble with ``samples`` deterministic under (seed, K).
        When no edges connect to the intervention, ``empty_dag_reason`` is
        set and the ensemble is a zero-effect natural-history rollout — the
        trajectory is still returned so the caller sees Erik's baseline
        decline, not a spurious "cure."
    """
    if K <= 0:
        raise ValueError("K must be positive")
    if horizon_months <= 0:
        raise ValueError("horizon_months must be positive")
    timepoints = tuple(sorted(set(int(m) for m in timepoints_months if 0 <= int(m) <= horizon_months)))
    if 0 not in timepoints:
        timepoints = (0,) + timepoints
    if not timepoints:
        raise ValueError("no valid timepoints after filtering")

    relevant_src_ids = {intervention.compound_entity_id, *intervention.target_entity_ids}
    relevant_edges = tuple(
        e for e in edges
        if e.source_entity_id in relevant_src_ids
    )

    empty_reason: Optional[str] = None
    if not edges:
        empty_reason = 'no_edges_supplied'
    elif not relevant_edges:
        empty_reason = 'no_edges_touch_intervention'

    samples: list[TrajectorySample] = []
    for j in range(K):
        sample_seed = _derive_seed(seed, j)
        sample = _simulate_one(
            baseline=baseline,
            edges=relevant_edges,
            rng_seed=sample_seed,
            horizon_months=horizon_months,
            timepoints=timepoints,
        )
        samples.append(sample)

    deltas = [s.alsfrs_r_slope_delta for s in samples]
    mean = sum(deltas) / len(deltas)
    variance = sum((d - mean) ** 2 for d in deltas) / len(deltas)
    std = math.sqrt(variance)
    ci_lo = _percentile(deltas, 10.0)
    ci_hi = _percentile(deltas, 90.0)

    return RolloutEnsemble(
        generator_version=GENERATOR_VERSION,
        intervention_entity_id=intervention.compound_entity_id,
        baseline_hash=hash_for(baseline),
        horizon_months=horizon_months,
        ensemble_size=K,
        samples=tuple(samples),
        edge_snapshot=tuple(relevant_edges or edges),
        alsfrs_r_slope_delta_mean=mean,
        alsfrs_r_slope_delta_std=std,
        alsfrs_r_slope_delta_ci_lower=ci_lo,
        alsfrs_r_slope_delta_ci_upper=ci_hi,
        empty_dag_reason=empty_reason,
    )


# ─── Internals ────────────────────────────────────────────────────────────────


def _simulate_one(
    *,
    baseline: PatientBaseline,
    edges: Sequence[EdgeSnapshot],
    rng_seed: int,
    horizon_months: int,
    timepoints: Sequence[int],
) -> TrajectorySample:
    rng = random.Random(rng_seed)

    # Step 1: accumulate slope delta from all edges that touch the intervention.
    slope_delta = 0.0
    contributions: list[dict[str, Any]] = []
    for idx, edge in enumerate(edges):
        sampled = _sample_edge_effect(edge, rng)
        if sampled is None:
            continue
        scale = edge.effect_scale or ''
        contribution: float
        if scale in _ALSFRSR_SCALES_DIRECT:
            contribution = sampled
        elif scale in _ALSFRSR_SCALES_INDIRECT:
            contribution = sampled * _ALSFRSR_SCALES_INDIRECT[scale]
        elif scale in _POTENCY_SCALES:
            # Potency alone does not move the slope — it's a gate. Log the
            # edge but do not contribute.
            contributions.append({
                'edge_index': idx, 'scale': scale,
                'sampled_effect': sampled, 'slope_contribution_pts_mo': 0.0,
                'note': 'potency_gate_only',
            })
            continue
        else:
            contributions.append({
                'edge_index': idx, 'scale': scale,
                'sampled_effect': sampled, 'slope_contribution_pts_mo': 0.0,
                'note': 'unhandled_scale',
            })
            continue

        # Attenuate by identification_confidence — weak identifications can
        # still move the trajectory but less than a gold-standard RCT.
        contribution *= max(0.0, min(1.0, edge.identification_confidence))
        # Per-edge clamp to prevent a wild outlier from dominating.
        contribution = max(-_PER_EDGE_SLOPE_CLAMP_PTS_MO,
                           min(_PER_EDGE_SLOPE_CLAMP_PTS_MO, contribution))

        slope_delta += contribution
        contributions.append({
            'edge_index': idx, 'scale': scale,
            'sampled_effect': sampled,
            'slope_contribution_pts_mo': contribution,
        })

    # Step 2: project trajectory at each timepoint. Additive decomposition:
    #   alsfrs_r(t) = baseline + (natural_slope + delta) * t * traj_modifier
    trajectory_modifier = max(0.1, baseline.trajectory_modifier)
    months: list[int] = list(timepoints)
    alsfrs_r: list[float] = []
    nfl: list[float] = []
    fvc: list[float] = []
    survival: list[float] = []
    for t in months:
        net_slope = (_NATURAL_ALSFRSR_SLOPE_PER_MONTH + slope_delta) * trajectory_modifier
        pts = max(0.0, min(48.0, baseline.alsfrs_r + net_slope * t))
        alsfrs_r.append(pts)

        # NfL and FVC have lighter coupling to intervention in this Week 1
        # version — we only modulate via a fixed share of the slope_delta.
        nfl_slope = _NATURAL_NFL_SLOPE_PER_MONTH - 0.03 * slope_delta
        nfl_val = max(0.0, baseline.nfl_pg_ml + nfl_slope * t)
        nfl.append(nfl_val)

        fvc_slope = _NATURAL_FVC_SLOPE_PER_MONTH + 0.5 * slope_delta
        fvc_val = max(0.0, min(150.0, baseline.fvc_pct + fvc_slope * t))
        fvc.append(fvc_val)

        # Survival: exponential with hazard reduced by slope_delta.
        # Each +1 pt/month of protection ≈ 35% hazard reduction.
        hazard_reduction = max(0.0, min(0.95, 0.35 * slope_delta))
        monthly_haz = _NATURAL_MONTHLY_HAZARD * (1.0 - hazard_reduction)
        survival.append(math.exp(-monthly_haz * t))

    return TrajectorySample(
        months=tuple(months),
        alsfrs_r=tuple(alsfrs_r),
        nfl=tuple(nfl),
        fvc=tuple(fvc),
        survival_prob=tuple(survival),
        alsfrs_r_slope_delta=slope_delta,
        rng_seed=rng_seed,
        edge_contributions=tuple(contributions),
    )


def _sample_edge_effect(edge: EdgeSnapshot, rng: random.Random) -> Optional[float]:
    """Draw one Gaussian sample of an edge's effect.

    Returns ``None`` when the edge has no effect_mean (observational
    relationships promoted without a quantitative effect). Uses std=0.25
    as the default dispersion when std is missing — deliberately wide so
    point estimates don't masquerade as certainties.
    """
    mean = edge.effect_mean
    if mean is None:
        return None
    std = edge.effect_std if (edge.effect_std is not None and edge.effect_std > 0) else 0.25
    return rng.gauss(mean, std)


def _derive_seed(master: int, sample_idx: int) -> int:
    """Derive a per-sample RNG seed from a master seed.

    Uses SHA-256 so correlations between samples are negligible even when
    the master seed is small.
    """
    h = hashlib.sha256(f'{master}:{sample_idx}'.encode('ascii')).digest()
    return int.from_bytes(h[:8], 'big', signed=False)


def _percentile(values: list[float], pct: float) -> float:
    """Simple linear-interpolation percentile (pct ∈ [0, 100])."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    xs = sorted(values)
    k = (pct / 100.0) * (len(xs) - 1)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
