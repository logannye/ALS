"""Latent state factor models for the Erik ALS causal research engine.

Each model extends BaseEnvelope and captures a different biological or
functional dimension of a patient's disease state.

Includes:
- TDP43FunctionalState       — TDP-43 nuclear/cytoplasmic pathology scores
- SplicingState              — Cryptic splicing burden and target-specific scores
- GlialState                 — Microglial / astrocytic neuroinflammation scores
- NMJIntegrityState          — Neuromuscular junction occupancy and reinnervation
- RespiratoryReserveState    — Pulmonary reserve and NIV transition probability
- FunctionalState            — ALSFRS-R subscores and clinical function scores
- ReversibilityWindowEstimate— Overall and domain-specific reversibility scores
- UncertaintyState           — Multi-source epistemic uncertainty decomposition
- DiseaseStateSnapshot       — Cross-reference hub for all latent state objects
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import Field

from ontology.base import BaseEnvelope


# ---------------------------------------------------------------------------
# TDP43FunctionalState
# ---------------------------------------------------------------------------

class TDP43FunctionalState(BaseEnvelope):
    """Captures TDP-43 nuclear function and cytoplasmic pathology estimates.

    ``type`` is locked to ``"TDP43FunctionalState"``.
    """

    type: str = Field(default="TDP43FunctionalState", min_length=1)

    subject_ref: str
    nuclear_function_score: float = 0
    cytoplasmic_pathology_probability: float = 0
    loss_of_function_probability: float = 0
    supporting_marker_refs: list[str] = Field(default_factory=list)
    dominant_uncertainties: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SplicingState
# ---------------------------------------------------------------------------

class SplicingState(BaseEnvelope):
    """Cryptic splicing burden and target-specific disruption scores.

    ``type`` is locked to ``"SplicingState"``.
    """

    type: str = Field(default="SplicingState", min_length=1)

    subject_ref: str
    cryptic_splicing_burden_score: float
    stmn2_disruption_score: float
    unc13a_disruption_score: float
    other_target_scores: dict[str, float] = Field(default_factory=dict)
    source_assay_refs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# GlialState
# ---------------------------------------------------------------------------

class GlialState(BaseEnvelope):
    """Neuroinflammation compartment — microglial and astrocytic scores.

    ``type`` is locked to ``"GlialState"``.
    """

    type: str = Field(default="GlialState", min_length=1)

    subject_ref: str
    microglial_activation_score: float
    astrocytic_toxicity_score: float
    inflammatory_amplification_score: float
    evidence_refs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# NMJIntegrityState
# ---------------------------------------------------------------------------

class NMJIntegrityState(BaseEnvelope):
    """Neuromuscular junction occupancy and denervation/reinnervation scores.

    ``type`` is locked to ``"NMJIntegrityState"``.
    """

    type: str = Field(default="NMJIntegrityState", min_length=1)

    subject_ref: str
    estimated_nmj_occupancy: float
    denervation_rate_score: float
    reinnervation_capacity_score: float
    supporting_refs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# RespiratoryReserveState
# ---------------------------------------------------------------------------

class RespiratoryReserveState(BaseEnvelope):
    """Pulmonary reserve and near-term NIV transition probability.

    ``type`` is locked to ``"RespiratoryReserveState"``.
    """

    type: str = Field(default="RespiratoryReserveState", min_length=1)

    subject_ref: str
    reserve_score: float
    six_month_decline_risk: float
    niv_transition_probability_6m: float
    supporting_refs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FunctionalState
# ---------------------------------------------------------------------------

class FunctionalState(BaseEnvelope):
    """Aggregated clinical functional scores including ALSFRS-R subscores.

    ``type`` is locked to ``"FunctionalState"``.
    All fields are optional to allow partial snapshots.
    """

    type: str = Field(default="FunctionalState", min_length=1)

    subject_ref: str
    alsfrs_r_total: Optional[int] = None
    bulbar_subscore: Optional[float] = None
    fine_motor_subscore: Optional[float] = None
    gross_motor_subscore: Optional[float] = None
    respiratory_subscore: Optional[float] = None
    speech_function_score: Optional[float] = None
    swallow_function_score: Optional[float] = None
    mobility_score: Optional[float] = None
    weight_kg: Optional[float] = None


# ---------------------------------------------------------------------------
# ReversibilityWindowEstimate
# ---------------------------------------------------------------------------

class ReversibilityWindowEstimate(BaseEnvelope):
    """Estimated reversibility across molecular, NMJ, and functional domains.

    ``type`` is locked to ``"ReversibilityWindowEstimate"``.
    """

    type: str = Field(default="ReversibilityWindowEstimate", min_length=1)

    subject_ref: str
    overall_reversibility_score: float
    molecular_correction_plausibility: float
    nmj_recovery_plausibility: float
    functional_recovery_plausibility: float
    dominant_bottleneck: str
    estimated_time_sensitivity_days: Optional[int] = None


# ---------------------------------------------------------------------------
# UncertaintyState
# ---------------------------------------------------------------------------

class UncertaintyState(BaseEnvelope):
    """Decomposed epistemic uncertainty across six sources.

    ``type`` is locked to ``"UncertaintyState"``.
    All float fields default to 0.
    """

    type: str = Field(default="UncertaintyState", min_length=1)

    subject_ref: str
    subtype_ambiguity: float = 0
    missing_measurement_uncertainty: float = 0
    model_form_uncertainty: float = 0
    intervention_effect_uncertainty: float = 0
    transportability_uncertainty: float = 0
    evidence_conflict_uncertainty: float = 0
    dominant_missing_measurements: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DiseaseStateSnapshot
# ---------------------------------------------------------------------------

class DiseaseStateSnapshot(BaseEnvelope):
    """Point-in-time cross-reference hub linking all latent state objects.

    ``type`` is locked to ``"DiseaseStateSnapshot"``.
    All reference fields are optional; absence means the state has not yet
    been estimated for that domain.
    """

    type: str = Field(default="DiseaseStateSnapshot", min_length=1)

    subject_ref: str
    as_of: datetime
    etiologic_driver_profile_ref: Optional[str] = None
    molecular_state_refs: list[str] = Field(default_factory=list)
    compartment_state_refs: list[str] = Field(default_factory=list)
    functional_state_ref: Optional[str] = None
    reversibility_window_ref: Optional[str] = None
    uncertainty_ref: Optional[str] = None
