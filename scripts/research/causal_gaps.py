"""Structured causal gap tracker for drug discovery.

Each CausalGap represents a missing link in the causal chain from
"molecular biology" → "Erik's cure."  Gaps are ranked by therapeutic
leverage (how much does closing this gap advance drug design?) and
tagged with a resolution path (computational vs. clinical measurement).

Persisted in ``erik_ops.causal_gaps`` and updated after each research step.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

from db.pool import get_connection

logger = logging.getLogger(__name__)


@dataclass
class CausalGap:
    """A structured knowledge gap in the causal chain toward Erik's cure."""

    id: str
    upstream: str                       # Known entity/process
    downstream: str                     # Known entity/process
    missing_link: str                   # What we don't understand
    therapeutic_leverage: float         # 0-1: how much does closing this advance drug design?
    resolution_path: str                # "computational" | "literature" | "clinical_measurement"
    status: str = "open"                # "open" | "partially_resolved" | "resolved"
    evidence_refs: list[str] = field(default_factory=list)
    target_refs: list[str] = field(default_factory=list)  # Drug design target IDs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> CausalGap:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# DB operations
# ---------------------------------------------------------------------------

def _ensure_gaps_table() -> None:
    """Create the causal_gaps table if it doesn't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS erik_ops.causal_gaps (
                id TEXT PRIMARY KEY,
                data JSONB NOT NULL,
                status TEXT NOT NULL DEFAULT 'open',
                therapeutic_leverage REAL NOT NULL DEFAULT 0.5,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gaps_status
            ON erik_ops.causal_gaps(status)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_gaps_leverage
            ON erik_ops.causal_gaps(therapeutic_leverage DESC)
        """)
        conn.commit()


def save_gap(gap: CausalGap) -> None:
    """Upsert a CausalGap to the database."""
    with get_connection() as conn:
        conn.execute(
            """INSERT INTO erik_ops.causal_gaps (id, data, status, therapeutic_leverage, updated_at)
               VALUES (%s, %s::jsonb, %s, %s, NOW())
               ON CONFLICT (id) DO UPDATE SET
                 data = EXCLUDED.data,
                 status = EXCLUDED.status,
                 therapeutic_leverage = EXCLUDED.therapeutic_leverage,
                 updated_at = NOW()""",
            (gap.id, json.dumps(gap.to_dict()), gap.status, gap.therapeutic_leverage),
        )
        conn.commit()


def load_open_gaps(limit: int = 50) -> list[CausalGap]:
    """Load open gaps sorted by therapeutic leverage (highest first)."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT data FROM erik_ops.causal_gaps
               WHERE status = 'open'
               ORDER BY therapeutic_leverage DESC
               LIMIT %s""",
            (limit,),
        ).fetchall()
    return [CausalGap.from_dict(json.loads(row[0]) if isinstance(row[0], str) else row[0]) for row in rows]


def load_all_gaps() -> list[CausalGap]:
    """Load all gaps regardless of status."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT data FROM erik_ops.causal_gaps ORDER BY therapeutic_leverage DESC"
        ).fetchall()
    return [CausalGap.from_dict(json.loads(row[0]) if isinstance(row[0], str) else row[0]) for row in rows]


def resolve_gap(gap_id: str, evidence_refs: list[str]) -> bool:
    """Mark a gap as resolved with supporting evidence. Returns True if gap existed."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT data FROM erik_ops.causal_gaps WHERE id = %s", (gap_id,)
        ).fetchone()
        if not row:
            return False
        data = json.loads(row[0]) if isinstance(row[0], str) else row[0]
        data["status"] = "resolved"
        data["evidence_refs"] = list(set(data.get("evidence_refs", []) + evidence_refs))
        conn.execute(
            """UPDATE erik_ops.causal_gaps
               SET data = %s::jsonb, status = 'resolved', updated_at = NOW()
               WHERE id = %s""",
            (json.dumps(data), gap_id),
        )
        conn.commit()
    return True


def count_gaps_by_status() -> dict[str, int]:
    """Return gap counts grouped by status."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM erik_ops.causal_gaps GROUP BY status"
        ).fetchall()
    return {row[0]: row[1] for row in rows}


# ---------------------------------------------------------------------------
# Seed gaps from drug design targets
# ---------------------------------------------------------------------------

def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def seed_gaps_from_targets(targets_path: str = None) -> int:
    """Populate initial causal gaps from drug_design_targets.json.

    Each drug design target's notes field contains structured knowledge about
    what's known vs. unknown about drugging that target. Parse these into
    CausalGap objects.

    Returns the number of new gaps created.
    """
    import pathlib

    if targets_path is None:
        targets_path = str(
            pathlib.Path(__file__).parent.parent.parent / "data" / "seed" / "drug_design_targets.json"
        )

    with open(targets_path, "r") as f:
        targets = json.load(f)

    # Map each target to structured gaps based on strategy and notes
    gap_templates = {
        "ddt:tdp43_rrm": CausalGap(
            id="gap:tdp43_rrm_selectivity",
            upstream="TDP-43 RRM1/RRM2 RNA-binding domains",
            downstream="Selective modulation without disrupting normal RNA binding",
            missing_link="RRM domains are highly conserved — no selective small molecule binder validated in ALS context",
            therapeutic_leverage=0.9,
            resolution_path="computational",
            target_refs=["ddt:tdp43_rrm"],
        ),
        "ddt:tdp43_ctd": CausalGap(
            id="gap:tdp43_ctd_idr_drugging",
            upstream="TDP-43 C-terminal prion-like domain misfolding",
            downstream="Prevention of liquid-to-solid phase transition",
            missing_link="No high-resolution structure of CTD IDR — drug design for intrinsically disordered regions is frontier territory",
            therapeutic_leverage=0.95,
            resolution_path="computational",
            target_refs=["ddt:tdp43_ctd"],
        ),
        "ddt:unc13a_cryptic_splice": CausalGap(
            id="gap:unc13a_small_molecule_splicing",
            upstream="TDP-43 nuclear depletion → UNC13A cryptic exon inclusion",
            downstream="Oral splicing modifier that prevents UNC13A cryptic exon",
            missing_link="ASO approach validated preclinically but no oral small-molecule splicing modifier exists for UNC13A (risdiplam/SMA analog needed)",
            therapeutic_leverage=0.9,
            resolution_path="computational",
            target_refs=["ddt:unc13a_cryptic_splice"],
        ),
        "ddt:stmn2_cryptic_splice": CausalGap(
            id="gap:stmn2_restoration_compound",
            upstream="TDP-43 depletion → STMN2 cryptic exon → truncated non-functional protein",
            downstream="Full-length STMN2 restoration in motor neurons",
            missing_link="No oral compound validated to block STMN2 cryptic exon — only ASO approach proven preclinically",
            therapeutic_leverage=0.85,
            resolution_path="computational",
            target_refs=["ddt:stmn2_cryptic_splice"],
        ),
        "ddt:sod1_aggregation": CausalGap(
            id="gap:sod1_small_molecule_stabilizer",
            upstream="Mutant SOD1 monomer dissociation and misfolding",
            downstream="Native SOD1 dimer stabilization in vivo",
            missing_link="Small molecules (isoproterenol) shown to stabilize SOD1 dimer in vitro but no in vivo or clinical validation for ALS",
            therapeutic_leverage=0.75,
            resolution_path="computational",
            target_refs=["ddt:sod1_aggregation"],
        ),
    }

    # Also generate a generic gap for each target not in the template
    created = 0
    for target in targets:
        tid = target["id"]
        if tid in gap_templates:
            gap = gap_templates[tid]
        else:
            gap = CausalGap(
                id=f"gap:{_slug(target['target_name'])}_{_slug(target['druggable_site'])}",
                upstream=target["target_name"],
                downstream=target["strategy"],
                missing_link=target.get("notes", "Unknown druggability challenge"),
                therapeutic_leverage=0.7,
                resolution_path="computational",
                target_refs=[tid],
            )

        # Only create if doesn't exist
        with get_connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM erik_ops.causal_gaps WHERE id = %s", (gap.id,)
            ).fetchone()
        if not exists:
            save_gap(gap)
            created += 1

    # Add Erik-specific clinical gaps
    clinical_gaps = [
        CausalGap(
            id="gap:erik_genetic_subtype",
            upstream="Erik Draper's genome",
            downstream="ALS subtype determination (SOD1/C9orf72/FUS/TDP-43/sporadic)",
            missing_link="Invitae genetic panel results pending — subtype posterior is uniform 0.125 across 8 subtypes",
            therapeutic_leverage=1.0,
            resolution_path="clinical_measurement",
        ),
        CausalGap(
            id="gap:erik_csf_biomarkers",
            upstream="Erik's motor neuron state",
            downstream="Quantitative disease activity measurement",
            missing_link="No CSF NfL, phospho-TDP-43, or phospho-tau measurements — cannot confirm target engagement or track molecular response",
            therapeutic_leverage=0.8,
            resolution_path="clinical_measurement",
        ),
        CausalGap(
            id="gap:erik_transcriptomics",
            upstream="Erik's motor neuron gene expression",
            downstream="Patient-specific pathway dysregulation map",
            missing_link="No transcriptomic profiling — cannot identify which specific pathways are activated/suppressed in Erik's motor neurons",
            therapeutic_leverage=0.7,
            resolution_path="clinical_measurement",
        ),
    ]
    for gap in clinical_gaps:
        with get_connection() as conn:
            exists = conn.execute(
                "SELECT 1 FROM erik_ops.causal_gaps WHERE id = %s", (gap.id,)
            ).fetchone()
        if not exists:
            save_gap(gap)
            created += 1

    return created


# ---------------------------------------------------------------------------
# Evidence-based gap update
# ---------------------------------------------------------------------------

def update_gaps_from_evidence(evidence_items: list[dict]) -> list[str]:
    """Check if new evidence partially or fully resolves any open gaps.

    Returns list of gap IDs that were updated.
    """
    if not evidence_items:
        return []

    open_gaps = load_open_gaps(limit=100)
    if not open_gaps:
        return []

    updated_ids: list[str] = []
    for gap in open_gaps:
        for evi in evidence_items:
            claim = evi.get("claim", "") or evi.get("body", {}).get("claim", "")
            if not claim:
                continue

            # Check if evidence claim mentions entities from this gap
            upstream_match = any(
                word.lower() in claim.lower()
                for word in gap.upstream.split()
                if len(word) > 4
            )
            downstream_match = any(
                word.lower() in claim.lower()
                for word in gap.downstream.split()
                if len(word) > 4
            )

            if upstream_match and downstream_match:
                evi_id = evi.get("id", "")
                if evi_id and evi_id not in gap.evidence_refs:
                    gap.evidence_refs.append(evi_id)

                    # If 3+ evidence items touch this gap, mark partially resolved
                    if len(gap.evidence_refs) >= 3 and gap.status == "open":
                        gap.status = "partially_resolved"

                    save_gap(gap)
                    if gap.id not in updated_ids:
                        updated_ids.append(gap.id)

    return updated_ids
