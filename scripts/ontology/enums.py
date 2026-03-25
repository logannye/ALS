"""Canonical enum types for the Erik ALS causal research engine.

All string enums use `str, Enum` so they serialise transparently as plain
strings in JSON / Pydantic models.  PCHLayer uses IntEnum so causal layer
comparisons (`L1 < L2`) work naturally.
"""
from enum import Enum, IntEnum


class ObjectStatus(str, Enum):
    active = "active"
    superseded = "superseded"
    deprecated = "deprecated"
    deleted_logically = "deleted_logically"


class ConfidenceBand(str, Enum):
    very_low = "very_low"
    low = "low"
    moderate = "moderate"
    high = "high"
    very_high = "very_high"


class PrivacyClass(str, Enum):
    public = "public"
    restricted = "restricted"
    deidentified = "deidentified"
    phi = "phi"


class ApprovalState(str, Enum):
    not_required = "not_required"
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    expired = "expired"


class EvidenceDirection(str, Enum):
    supports = "supports"
    refutes = "refutes"
    mixed = "mixed"
    insufficient = "insufficient"


class ActionClass(str, Enum):
    read_only = "read_only"
    simulation_only = "simulation_only"
    recommendation_generation = "recommendation_generation"
    workflow_generation = "workflow_generation"
    research_execution = "research_execution"
    promotion_control = "promotion_control"
    rollback_control = "rollback_control"


class ALSOnsetRegion(str, Enum):
    upper_limb = "upper_limb"
    lower_limb = "lower_limb"
    bulbar = "bulbar"
    respiratory = "respiratory"
    multifocal = "multifocal"
    unknown = "unknown"


class SubtypeClass(str, Enum):
    sod1 = "sod1"
    c9orf72 = "c9orf72"
    fus = "fus"
    tardbp = "tardbp"
    sporadic_tdp43 = "sporadic_tdp43"
    glia_amplified = "glia_amplified"
    mixed = "mixed"
    unresolved = "unresolved"


class ProtocolLayer(str, Enum):
    """Ordered treatment protocol layers — index reflects therapeutic priority."""
    root_cause_suppression = "root_cause_suppression"
    pathology_reversal = "pathology_reversal"
    circuit_stabilization = "circuit_stabilization"
    regeneration_reinnervation = "regeneration_reinnervation"
    adaptive_maintenance = "adaptive_maintenance"


class PCHLayer(IntEnum):
    """Pearl Causal Hierarchy layer — enables integer comparisons (L1 < L2 < L3)."""
    L1_ASSOCIATIONAL = 1
    L2_INTERVENTIONAL = 2
    L3_COUNTERFACTUAL = 3


class ObservationKind(str, Enum):
    lab_result = "lab_result"
    emg_feature = "emg_feature"
    respiratory_metric = "respiratory_metric"
    speech_metric = "speech_metric"
    genomic_result = "genomic_result"
    imaging_finding = "imaging_finding"
    omics_measurement = "omics_measurement"
    workflow_signal = "workflow_signal"
    functional_score = "functional_score"
    vital_sign = "vital_sign"
    weight_measurement = "weight_measurement"
    medication_event = "medication_event"
    physical_exam_finding = "physical_exam_finding"


class InterpretationKind(str, Enum):
    diagnosis = "diagnosis"
    subtype_inference = "subtype_inference"
    progression_estimate = "progression_estimate"
    reversibility_estimate = "reversibility_estimate"
    treatment_response = "treatment_response"
    target_engagement = "target_engagement"
    eligibility_assessment = "eligibility_assessment"
    respiratory_decline_risk = "respiratory_decline_risk"


class InterventionClass(str, Enum):
    drug = "drug"
    aso = "aso"
    gene_editing = "gene_editing"
    small_molecule = "small_molecule"
    gene_silencing = "gene_silencing"
    supportive_care = "supportive_care"
    respiratory_support = "respiratory_support"
    feeding_support = "feeding_support"
    rehabilitation = "rehabilitation"
    workflow_action = "workflow_action"
    wet_lab_perturbation = "wet_lab_perturbation"
    trial_assignment = "trial_assignment"


class RelationCategory(str, Enum):
    structural = "structural"
    causal = "causal"
    temporal = "temporal"
    evidential = "evidential"
    therapeutic = "therapeutic"
    governance = "governance"


class SourceSystem(str, Enum):
    ehr = "ehr"
    registry = "registry"
    lims = "lims"
    omics = "omics"
    trial = "trial"
    manual = "manual"
    model = "model"
    workflow = "workflow"
    literature = "literature"
    database = "database"
