"""evidence/seed_builder.py

Loads curated seed data from ``data/seed/`` into PostgreSQL via EvidenceStore.

Usage
-----
    from evidence.seed_builder import load_seed
    stats = load_seed()
    # {"interventions_loaded": 25, "evidence_items_loaded": 93}

Can also be run directly::

    conda run -n erik-core python scripts/evidence/seed_builder.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ontology.evidence import EvidenceItem
from ontology.enums import EvidenceDirection, EvidenceStrength, InterventionClass, ProtocolLayer
from ontology.intervention import Intervention
from evidence.evidence_store import EvidenceStore

# ---------------------------------------------------------------------------
# Path constant
# ---------------------------------------------------------------------------

# Resolve relative to this file: scripts/evidence/seed_builder.py → project_root/data/seed
SEED_DIR: Path = Path(__file__).parent.parent.parent / "data" / "seed"

_LAYER_FILES = [
    "layer_a_root_cause.json",
    "layer_b_pathology.json",
    "layer_c_circuit.json",
    "layer_d_regeneration.json",
    "layer_e_maintenance.json",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_seed() -> dict[str, int]:
    """Load all seed JSON files into PostgreSQL via EvidenceStore.

    Reads ``data/seed/interventions.json`` and all ``layer_*.json`` files,
    parses them into Pydantic models, and upserts into ``erik_core.objects``.

    The operation is fully idempotent — calling it multiple times is safe.

    Returns
    -------
    dict with keys:
        ``interventions_loaded``  — number of Intervention objects processed
        ``evidence_items_loaded`` — number of EvidenceItem objects processed
    """
    store = EvidenceStore()

    # --- Interventions ---
    interventions_loaded = _load_interventions(store)

    # --- Evidence items (all layers) ---
    evidence_items_loaded = _load_evidence_items(store)

    return {
        "interventions_loaded": interventions_loaded,
        "evidence_items_loaded": evidence_items_loaded,
    }


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _load_interventions(store: EvidenceStore) -> int:
    """Parse and upsert all interventions from interventions.json."""
    path = SEED_DIR / "interventions.json"
    raw_list: list[dict] = json.loads(path.read_text())
    count = 0
    for raw in raw_list:
        intervention = _parse_intervention(raw)
        store.upsert_intervention(intervention)
        count += 1
    return count


def _load_evidence_items(store: EvidenceStore) -> int:
    """Parse and upsert all evidence items from every layer_*.json file."""
    count = 0
    for filename in _LAYER_FILES:
        path = SEED_DIR / filename
        raw_list: list[dict] = json.loads(path.read_text())
        for raw in raw_list:
            item = _parse_evidence_item(raw)
            store.upsert_evidence_item(item)
            count += 1
    return count


def _parse_evidence_item(raw: dict) -> EvidenceItem:
    """Convert a raw JSON dict from a layer_*.json seed file into an EvidenceItem.

    Parameters
    ----------
    raw:
        A single element from a layer seed file.  Expected keys:
        ``id``, ``claim``, ``direction``, ``strength``, plus optional
        ``source_refs``, ``supersedes_ref``, ``notes``, ``body``.

    Returns
    -------
    EvidenceItem
        Fully constructed Pydantic model ready for upsert.
    """
    return EvidenceItem(
        id=raw["id"],
        claim=raw["claim"],
        direction=EvidenceDirection(raw["direction"]),
        strength=EvidenceStrength(raw["strength"]),
        source_refs=raw.get("source_refs", []),
        supersedes_ref=raw.get("supersedes_ref"),
        notes=raw.get("notes", ""),
        body=raw.get("body", {}),
    )


def _parse_intervention(raw: dict) -> Intervention:
    """Convert a raw JSON dict from interventions.json into an Intervention.

    Parameters
    ----------
    raw:
        A single element from ``interventions.json``.  Expected keys:
        ``id``, ``name``, ``intervention_class``.  Optional: ``targets``,
        ``protocol_layer``, ``route``, ``intended_effects``, ``known_risks``,
        ``contraindications``, ``body``.

    Returns
    -------
    Intervention
        Fully constructed Pydantic model ready for upsert.
    """
    # Resolve optional protocol_layer enum
    protocol_layer_raw: Optional[str] = raw.get("protocol_layer")
    protocol_layer: Optional[ProtocolLayer] = (
        ProtocolLayer(protocol_layer_raw) if protocol_layer_raw else None
    )

    return Intervention(
        id=raw["id"],
        name=raw["name"],
        intervention_class=InterventionClass(raw["intervention_class"]),
        targets=raw.get("targets", []),
        protocol_layer=protocol_layer,
        route=raw.get("route", ""),
        intended_effects=raw.get("intended_effects", []),
        known_risks=raw.get("known_risks", []),
        contraindications=raw.get("contraindications", []),
        body=raw.get("body", {}),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stats = load_seed()
    print(
        f"Seed loaded successfully:\n"
        f"  interventions_loaded  = {stats['interventions_loaded']}\n"
        f"  evidence_items_loaded = {stats['evidence_items_loaded']}"
    )
