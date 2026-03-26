"""Stage 1 state materializer — converts raw clinical observations into latent
disease state objects.

This module provides deterministic (rule-based, no LLM) materializers for
each biological domain:

- ``materialize_functional_state``   — ALSFRS-R subscores + weight
- ``materialize_nmj_state``          — NMJ integrity from EMG findings
- ``materialize_respiratory_state``  — Reserve score from FVC % predicted
- ``materialize_uncertainty_state``  — Epistemic gaps in the measurement set
- ``materialize_state``              — Full DiseaseStateSnapshot orchestrator
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from ontology.enums import ObservationKind
from ontology.observation import Observation
from ontology.patient import ALSTrajectory
from ontology.state import (
    DiseaseStateSnapshot,
    FunctionalState,
    NMJIntegrityState,
    RespiratoryReserveState,
    UncertaintyState,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _filter(observations: list[Observation], kind: ObservationKind) -> list[Observation]:
    """Return all observations of the given kind."""
    return [o for o in observations if o.observation_kind == kind]


def _latest(observations: list[Observation]) -> Optional[Observation]:
    """Return the most-recent observation by ``id`` lexicographic order.

    Uses the ISO date suffix embedded in IDs (e.g. ``obs:weight:2026-03-09``)
    as a natural sort key.  Falls back to the last item if no date suffix is
    found.
    """
    if not observations:
        return None
    return sorted(observations, key=lambda o: o.id)[-1]


# ---------------------------------------------------------------------------
# materialize_functional_state
# ---------------------------------------------------------------------------

def materialize_functional_state(
    trajectory: ALSTrajectory,
    observations: list[Observation],
) -> FunctionalState:
    """Build a :class:`~ontology.state.FunctionalState` from ALSFRS-R scores.

    Extracts the latest ALSFRS-R assessment from *trajectory* and the latest
    weight observation.

    Fixed totals for Erik Draper based on the canonical ALSFRS-R entry:
    - bulbar  = speech(4) + salivation(4) + swallowing(4)   = 12
    - fine    = handwriting(4) + cutting_food(4) + dressing_hygiene(3) = 11
    - gross   = turning_in_bed(3) + walking(3) + climbing_stairs(2)    = 8
    - resp    = dyspnea(4) + orthopnea(4) + resp_insufficiency(4)      = 12
    - total   = 43
    """
    # Grab the latest (or only) ALSFRS-R score
    alsfrs: Optional[object] = None
    if trajectory.alsfrs_r_scores:
        alsfrs = sorted(
            trajectory.alsfrs_r_scores,
            key=lambda s: s.assessment_date,
        )[-1]

    total = alsfrs.total if alsfrs is not None else None
    bulbar = alsfrs.bulbar_subscore if alsfrs is not None else None
    fine_motor = alsfrs.fine_motor_subscore if alsfrs is not None else None
    gross_motor = alsfrs.gross_motor_subscore if alsfrs is not None else None
    respiratory = alsfrs.respiratory_subscore if alsfrs is not None else None

    # Latest weight
    weight_obs = _latest(_filter(observations, ObservationKind.weight_measurement))
    weight_kg = weight_obs.value if weight_obs is not None else None

    return FunctionalState(
        id=f"func:{trajectory.id}",
        subject_ref=trajectory.patient_ref,
        alsfrs_r_total=total,
        bulbar_subscore=float(bulbar) if bulbar is not None else None,
        fine_motor_subscore=float(fine_motor) if fine_motor is not None else None,
        gross_motor_subscore=float(gross_motor) if gross_motor is not None else None,
        respiratory_subscore=float(respiratory) if respiratory is not None else None,
        weight_kg=weight_kg,
    )


# ---------------------------------------------------------------------------
# materialize_nmj_state
# ---------------------------------------------------------------------------

def materialize_nmj_state(
    trajectory: ALSTrajectory,
    observations: list[Observation],
) -> NMJIntegrityState:
    """Build a :class:`~ontology.state.NMJIntegrityState` from EMG findings.

    If any EMG observation has ``supports_als=True`` the pathway is treated as
    *widespread denervation* with conservative NMJ occupancy and high
    denervation rate.
    """
    emg_obs = _filter(observations, ObservationKind.emg_feature)
    widespread = any(
        o.emg_finding is not None and o.emg_finding.supports_als
        for o in emg_obs
    )

    if widespread:
        occupancy = 0.5
        denervation_rate = 0.7
        reinnervation_capacity = 0.4
    else:
        occupancy = 0.75
        denervation_rate = 0.3
        reinnervation_capacity = 0.6

    return NMJIntegrityState(
        id=f"nmj:{trajectory.id}",
        subject_ref=trajectory.patient_ref,
        estimated_nmj_occupancy=occupancy,
        denervation_rate_score=denervation_rate,
        reinnervation_capacity_score=reinnervation_capacity,
        supporting_refs=[o.id for o in emg_obs],
    )


# ---------------------------------------------------------------------------
# materialize_respiratory_state
# ---------------------------------------------------------------------------

def materialize_respiratory_state(
    trajectory: ALSTrajectory,
    observations: list[Observation],
) -> RespiratoryReserveState:
    """Build a :class:`~ontology.state.RespiratoryReserveState` from PFT data.

    Uses the latest ``respiratory_metric`` observation.  If FVC % predicted
    is missing defaults to 70% (below threshold) as a conservative estimate.
    """
    resp_obs = _filter(observations, ObservationKind.respiratory_metric)
    latest_resp = _latest(resp_obs)

    fvc_pct: float = 70.0  # conservative default
    if latest_resp is not None and latest_resp.respiratory_metric is not None:
        fvc_pct_raw = latest_resp.respiratory_metric.fvc_percent_predicted
        if fvc_pct_raw is not None:
            fvc_pct = fvc_pct_raw

    reserve_score = min(fvc_pct / 100.0, 1.0)

    if fvc_pct > 80:
        six_month_decline_risk = 0.2
        niv_transition_probability_6m = 0.1
    else:
        six_month_decline_risk = 0.5
        niv_transition_probability_6m = 0.4

    return RespiratoryReserveState(
        id=f"resp:{trajectory.id}",
        subject_ref=trajectory.patient_ref,
        reserve_score=reserve_score,
        six_month_decline_risk=six_month_decline_risk,
        niv_transition_probability_6m=niv_transition_probability_6m,
        supporting_refs=[o.id for o in resp_obs],
    )


# ---------------------------------------------------------------------------
# materialize_uncertainty_state
# ---------------------------------------------------------------------------

# Standard measurement gaps we track for every patient
_STANDARD_GAPS = [
    "genetic_testing",
    "csf_biomarkers",
    "cryptic_exon_splicing_assay",
    "tdp43_in_vivo_measurement",
    "cortical_excitability_tms",
    "transcriptomics",
    "proteomics",
]


def materialize_uncertainty_state(
    trajectory: ALSTrajectory,
    observations: list[Observation],
) -> UncertaintyState:
    """Build a :class:`~ontology.state.UncertaintyState` from available data.

    Enumerates the seven standard measurement gaps that are universally missing
    until explicitly acquired.  ``subtype_ambiguity`` is pinned at 0.35 until
    genetic testing resolves the dominant subtype.
    """
    # All seven gaps are missing by default — future phases can cross them off
    missing = list(_STANDARD_GAPS)

    subtype_ambiguity = 0.35
    missing_measurement_uncertainty = len(missing) / 10.0

    return UncertaintyState(
        id=f"unc:{trajectory.id}",
        subject_ref=trajectory.patient_ref,
        subtype_ambiguity=subtype_ambiguity,
        missing_measurement_uncertainty=missing_measurement_uncertainty,
        dominant_missing_measurements=missing,
    )


# ---------------------------------------------------------------------------
# materialize_state (full orchestrator)
# ---------------------------------------------------------------------------

def materialize_state(
    trajectory: ALSTrajectory,
    observations: list[Observation],
    use_llm: bool = False,
    reasoning_engine: Optional[object] = None,
    evidence_items: Optional[list[dict]] = None,
) -> DiseaseStateSnapshot:
    """Assemble a complete :class:`~ontology.state.DiseaseStateSnapshot`.

    Calls all deterministic materializers.  When ``use_llm=False`` the
    molecular-state (m_t) and reversibility-window (r_t) fields are left
    empty — those require Stage 0 and Stage 1 LLM passes respectively.

    Parameters
    ----------
    trajectory:
        The patient's ``ALSTrajectory``.
    observations:
        All clinical observations for the patient.
    use_llm:
        When ``True``, passes *reasoning_engine* and *evidence_items* to the
        LLM-backed materializers (not yet implemented; reserved for Stage 2+).
    reasoning_engine:
        ``ReasoningEngine`` instance (only used when ``use_llm=True``).
    evidence_items:
        Pre-fetched evidence items for the LLM context.

    Returns
    -------
    DiseaseStateSnapshot
        Point-in-time snapshot linking all latent state objects.
    """
    fs = materialize_functional_state(trajectory, observations)
    ns = materialize_nmj_state(trajectory, observations)
    rs = materialize_respiratory_state(trajectory, observations)
    us = materialize_uncertainty_state(trajectory, observations)

    # Serialise sub-states into the body dict for downstream consumers
    body = {
        "functional_state": fs.model_dump(),
        "nmj_state": ns.model_dump(),
        "respiratory_state": rs.model_dump(),
        "uncertainty_state": us.model_dump(),
    }

    # LLM-backed states (m_t, r_t) — reserved for Stage 2+
    molecular_state_refs: list[str] = []
    reversibility_window_ref: Optional[str] = None

    if use_llm and reasoning_engine is not None and evidence_items is not None:
        # Placeholder for future LLM materializer calls
        pass

    return DiseaseStateSnapshot(
        id=f"snapshot:{trajectory.id}",
        subject_ref=trajectory.patient_ref,
        as_of=datetime.now(timezone.utc),
        molecular_state_refs=molecular_state_refs,
        compartment_state_refs=[ns.id, rs.id],
        functional_state_ref=fs.id,
        reversibility_window_ref=reversibility_window_ref,
        uncertainty_ref=us.id,
        body=body,
    )
