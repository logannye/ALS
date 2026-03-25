"""Intervention model for the Erik ALS causal research engine.

Includes:
- Intervention — A therapeutic, supportive, or research intervention
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from ontology.base import BaseEnvelope
from ontology.enums import InterventionClass, ProtocolLayer


# ---------------------------------------------------------------------------
# Intervention
# ---------------------------------------------------------------------------

class Intervention(BaseEnvelope):
    """A single therapeutic, supportive care, or research intervention.

    ``type`` is locked to ``"Intervention"``.
    """

    type: str = Field(default="Intervention", min_length=1)

    name: str
    intervention_class: InterventionClass
    targets: list[str] = Field(default_factory=list)
    protocol_layer: Optional[ProtocolLayer] = None
    route: str
    intended_effects: list[str] = Field(default_factory=list)
    known_risks: list[str] = Field(default_factory=list)
    contraindications: list[str] = Field(default_factory=list)
