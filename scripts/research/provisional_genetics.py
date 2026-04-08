"""Provisional genetic profile inference for ALS patients awaiting genetic testing.

When Erik's genetic testing results have not yet arrived, this module infers a
provisional profile from clinical features. The provisional profile allows the
research engine to begin Layer 3 (Erik-specific) work while flagging all
downstream conclusions as pending confirmation.

Key clinical logic:
  - Family history → C9orf72 repeat expansion (most common fALS cause, ~40%)
  - Age of onset < 45 → FUS (young-onset sALS association)
  - Default → TDP-43 proteinopathy (TARDBP; ~97% of sporadic ALS)
  - NfL elevation adds a small confidence boost (biomarker of neurodegeneration)

The `provisional=True` flag is always set so callers can gate drug-design
(Layer 4) which requires confirmed genetics.
"""
from __future__ import annotations

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


def infer_provisional_profile(
    age_onset: int = 67,
    site_onset: str = "limb",
    alsfrs_r: int | None = 43,
    nfl_elevated: bool = True,
    family_history: bool = False,
) -> dict:
    """Infer a provisional genetic profile from clinical features.

    Args:
        age_onset: Age at symptom onset (years).
        site_onset: Anatomical region of first symptoms (e.g. "limb", "bulbar").
        alsfrs_r: ALSFRS-R score at time of inference (None if not available).
        nfl_elevated: Whether serum/CSF neurofilament light chain is elevated.
        family_history: Whether a first-degree relative has ALS.

    Returns:
        A dict with keys:
          gene          – inferred causal gene
          variant       – inferred variant or proteinopathy descriptor
          subtype       – "fALS" or "sALS"
          confidence    – float in [0, 1]; reflects inference uncertainty
          provisional   – always True (confirmed genetics not yet available)
          source        – "clinical_inference"
          clinical_features – dict of the input parameters used for inference
    """
    clinical_features = {
        "age_onset": age_onset,
        "site_onset": site_onset,
        "alsfrs_r": alsfrs_r,
        "nfl_elevated": nfl_elevated,
        "family_history": family_history,
    }

    if family_history:
        # Autosomal dominant family history strongly suggests C9orf72 hexanucleotide
        # repeat expansion — the single most common inherited ALS mutation (~40% fALS).
        gene = "C9orf72"
        variant = "hexanucleotide_repeat_expansion"
        subtype = "fALS"
        confidence = 0.40
    elif age_onset < 45:
        # Young-onset sporadic ALS has elevated FUS mutation prevalence.
        gene = "FUS"
        variant = "fus_mutation"
        subtype = "sALS"
        confidence = 0.15
    else:
        # Default: TDP-43 proteinopathy is the unifying pathology in ~97% of sALS cases.
        gene = "TARDBP"
        variant = "tdp43_proteinopathy"
        subtype = "sALS"
        confidence = 0.70

    # NfL elevation is a non-specific but supporting biomarker; add a small boost.
    if nfl_elevated:
        confidence = min(confidence + 0.05, MAX_CONFIDENCE)

    return {
        "gene": gene,
        "variant": variant,
        "subtype": subtype,
        "confidence": confidence,
        "provisional": True,
        "source": "clinical_inference",
        "clinical_features": clinical_features,
    }
