"""Parse raw lab panel data into typed Observation[LabResult] objects.

Usage::

    from ingestion.lab_results import parse_lab_panel

    obs_list = parse_lab_panel(raw_labs, subject_ref="patient:erik_draper")
"""
from __future__ import annotations

import re
from datetime import date
from typing import Optional

from ontology.base import Provenance
from ontology.enums import ObservationKind, SourceSystem
from ontology.observation import LabResult, Observation


def _snake_case(name: str) -> str:
    """Convert a lab name to snake_case for ID construction.

    Examples:
        "NfL Plasma" -> "nfl_plasma"
        "Sed Rate"   -> "sed_rate"
    """
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = s.strip("_")
    return s


def parse_lab_panel(
    raw_labs: list[dict],
    subject_ref: str,
) -> list[Observation]:
    """Convert a list of raw lab dicts into Observation envelopes.

    Parameters
    ----------
    raw_labs:
        Each dict must have keys: name, value, unit, ref_low, ref_high, date.
        ``date`` is an ISO-8601 string (YYYY-MM-DD).
        ``ref_low`` / ``ref_high`` may be None.
    subject_ref:
        The patient ID to link observations to (e.g. "patient:erik_draper").

    Returns
    -------
    list[Observation]
        One Observation per raw lab, with a populated ``lab_result`` sub-object.
    """
    observations: list[Observation] = []

    for raw in raw_labs:
        lab_name: str = raw["name"]
        lab_value: float = float(raw["value"])
        lab_unit: str = raw["unit"]
        ref_low: Optional[float] = (
            float(raw["ref_low"]) if raw.get("ref_low") is not None else None
        )
        ref_high: Optional[float] = (
            float(raw["ref_high"]) if raw.get("ref_high") is not None else None
        )
        lab_date = (
            date.fromisoformat(raw["date"])
            if isinstance(raw["date"], str)
            else raw["date"]
        )

        slug = _snake_case(lab_name)
        obs_id = f"obs:lab:{slug}:{lab_date.isoformat()}"

        lab_result = LabResult(
            name=lab_name,
            value=lab_value,
            unit=lab_unit,
            reference_low=ref_low,
            reference_high=ref_high,
            collection_date=lab_date,
            method="serum",
        )

        obs = Observation(
            id=obs_id,
            subject_ref=subject_ref,
            observation_kind=ObservationKind.lab_result,
            name=lab_name,
            lab_result=lab_result,
            value=lab_value,
            unit=lab_unit,
            provenance=Provenance(source_system=SourceSystem.ehr),
        )
        observations.append(obs)

    return observations
