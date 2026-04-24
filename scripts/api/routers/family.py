"""Family-facing endpoints created during the 2026-04-24 UX pass.

Four small, single-purpose endpoints the frontend uses to render the
"Erik-first" dashboard + rewritten Today panel:

  * GET /api/trajectory       — Erik's recent ALSFRS-R / NfL / FVC
                                values + sparkline data
  * GET /api/top-candidate    — current highest-scored non-approved
                                compound across all protocol layers
  * GET /api/current-hypothesis — what Galen is actively investigating
                                right now (most recent open TCG hypothesis)
  * GET /api/galen-believes   — one-sentence "headline" the family
                                sees at the top of the dashboard

These are deliberately thin — they read cached state, never do LLM
calls on the request path, and cache at most for a minute so the
family-side experience is snappy.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter

from db.pool import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# ─── /api/trajectory ─────────────────────────────────────────────────────────


_ALSFRS_KEYS = (
    "bulbar_speech", "bulbar_salivation", "bulbar_swallowing",
    "fine_motor_handwriting", "fine_motor_cutting", "fine_motor_dressing",
    "gross_motor_turning", "gross_motor_walking", "gross_motor_climbing",
    "respiratory_dyspnea", "respiratory_orthopnea", "respiratory_insufficiency",
)


def _sum_alsfrs_items(body: dict) -> Optional[float]:
    """ALSFRS-R rows may have a precomputed total_score, or just the 12
    items. Sum items if total isn't there."""
    for key in ("total_score", "total"):
        v = body.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    parts = []
    for k in _ALSFRS_KEYS:
        v = body.get(k)
        if v is not None:
            try:
                parts.append(float(v))
            except (TypeError, ValueError):
                pass
    if not parts:
        return None
    return sum(parts)


def _points_for(type_: str, extract) -> list[dict[str, Any]]:
    """Pull every observation of the given type and extract a numeric
    value via the caller-supplied extract(body) callable."""
    points: list[dict[str, Any]] = []
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """SELECT body, created_at
                     FROM erik_core.objects
                    WHERE type = %s AND status = 'active'
                    ORDER BY created_at ASC""",
                (type_,),
            ).fetchall()
    except Exception:
        logger.exception("trajectory: failed to read %s observations", type_)
        return points

    for body, created_at in rows:
        if isinstance(body, str):
            try:
                body = json.loads(body)
            except Exception:
                continue
        if not isinstance(body, dict):
            continue
        value = extract(body)
        if value is None:
            continue
        points.append({
            "date": created_at.isoformat() if created_at else None,
            "value": value,
        })
    return points


@router.get("/trajectory")
def trajectory():
    """Erik's recent clinical measurements.

    Returns three series (alsfrs_r, nfl, fvc) each with current value
    and a history sparkline, plus human-language summaries.
    """
    alsfrs = _points_for("ALSFRSRScore", _sum_alsfrs_items)
    nfl = _points_for(
        "LabResult",
        lambda b: (float(b["value"]) if b.get("value") and
                   "neurofilament" in str(b.get("name", "")).lower()
                   else None),
    )
    fvc = _points_for(
        "RespiratoryMetric",
        lambda b: (float(b["fvc_pct_predicted"]) if b.get("fvc_pct_predicted") else None),
    )

    def _summary(points: list[dict], unit: str, trend_direction: str) -> dict:
        """trend_direction is 'down_is_progression' or 'up_is_progression'."""
        if not points:
            return {"current": None, "points": [], "summary": None}
        current = points[-1]["value"]
        out = {"current": current, "points": points, "summary": None}
        if len(points) >= 2:
            prev = points[-2]["value"]
            delta = current - prev
            direction = "declined" if delta < 0 else ("improved" if delta > 0 else "unchanged")
            out["summary"] = (
                f"Most recent: {current:.1f} {unit}. "
                f"{direction.capitalize()} by {abs(delta):.1f} {unit} "
                f"from the previous measurement."
            )
        else:
            out["summary"] = f"Most recent: {current:.1f} {unit}."
        return out

    return {
        "alsfrs_r": _summary(alsfrs, "pts", "down_is_progression"),
        "nfl_pg_ml": _summary(nfl, "pg/mL", "up_is_progression"),
        "fvc_pct": _summary(fvc, "%", "down_is_progression"),
    }


# ─── /api/top-candidate ──────────────────────────────────────────────────────
#
# Reads the latest Protocol row, parses score hints out of each layer's
# `notes` field (existing shape: "Drug Name (score=0.80); Other (score=0.60)"),
# and returns the single highest-scored drug along with its layer context.


_SCORE_RE = None


def _parse_layer_notes(notes: str) -> list[tuple[str, float]]:
    """Parse 'DrugA (score=0.80); DrugB (score=0.60)' → [('DrugA', 0.8), ...].
    Tolerant of missing scores and extra whitespace."""
    global _SCORE_RE
    if _SCORE_RE is None:
        import re
        _SCORE_RE = re.compile(r"^(.*?)\s*\(score=([0-9.]+)\)\s*$")
    out: list[tuple[str, float]] = []
    if not notes:
        return out
    for chunk in notes.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _SCORE_RE.match(chunk)
        if m:
            try:
                out.append((m.group(1).strip(), float(m.group(2))))
            except ValueError:
                continue
    return out


# Human-language labels for the protocol layers (PR2 also uses this map
# on the frontend; keeping one copy here so the API is self-describing).
LAYER_LABELS = {
    "root_cause_suppression": "Stop the disease at its source",
    "pathology_reversal": "Reverse damage already done",
    "circuit_stabilization": "Protect motor neurons still working",
    "regeneration_reinnervation": "Regrow lost connections",
    "adaptive_maintenance": "Monitor & adjust",
}


@router.get("/top-candidate")
def top_candidate():
    """Return the single highest-scored drug in the current protocol."""
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT body FROM erik_core.objects
                    WHERE type = 'Protocol' AND status = 'active'
                    ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()
    except Exception:
        logger.exception("top_candidate: failed to read protocol")
        return {"candidate": None, "reason": "no_protocol"}

    if not row:
        return {"candidate": None, "reason": "no_protocol"}

    body = row[0] if isinstance(row[0], dict) else (json.loads(row[0]) if row[0] else {})
    best_drug: Optional[str] = None
    best_score: float = -1.0
    best_layer: Optional[str] = None
    for layer_entry in (body.get("layers") or []):
        layer_name = layer_entry.get("layer")
        notes = layer_entry.get("notes") or ""
        for drug, score in _parse_layer_notes(notes):
            if score > best_score:
                best_drug, best_score, best_layer = drug, score, layer_name
    if best_drug is None:
        return {"candidate": None, "reason": "no_scored_candidates"}

    return {
        "candidate": {
            "name": best_drug,
            "score": best_score,
            "layer": best_layer,
            "layer_label": LAYER_LABELS.get(best_layer or "", best_layer),
            "rationale": (
                f"Currently the highest-scored candidate in Galen's "
                f"“{LAYER_LABELS.get(best_layer or '', best_layer)}” layer "
                f"with a confidence of {best_score:.0%}."
            ),
        },
        "reason": None,
    }


# ─── /api/current-hypothesis ─────────────────────────────────────────────────
#
# Surfaces the most-recently-created open TCG hypothesis so the family
# sees "Galen is currently investigating: ..." rather than only aggregate
# step counts. Falls back to the most recent proposed hypothesis of any
# age so the panel is rarely blank.


@router.get("/current-hypothesis")
def current_hypothesis():
    cutoff = datetime.now(timezone.utc) - timedelta(days=3)
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT hypothesis, confidence, status, created_at
                     FROM erik_core.tcg_hypotheses
                    WHERE status IN ('proposed', 'investigating')
                      AND created_at >= %s
                    ORDER BY created_at DESC
                    LIMIT 1""",
                (cutoff,),
            ).fetchone()
            if not row:
                row = conn.execute(
                    """SELECT hypothesis, confidence, status, created_at
                         FROM erik_core.tcg_hypotheses
                         ORDER BY created_at DESC
                         LIMIT 1"""
                ).fetchone()
    except Exception:
        logger.exception("current_hypothesis query failed")
        return {"hypothesis": None}

    if not row:
        return {"hypothesis": None}
    hypothesis, confidence, status, created_at = row
    return {
        "hypothesis": {
            "text": hypothesis,
            "confidence": float(confidence or 0.0),
            "status": status,
            "created_at": created_at.isoformat() if created_at else None,
        }
    }


# ─── /api/galen-believes ─────────────────────────────────────────────────────
#
# One-sentence headline for the top of the dashboard. Assembled from
# existing state (top candidate + current hypothesis + latest trajectory
# point) without any LLM call on the request path — we want this to be
# instant and deterministic.


@router.get("/galen-believes")
def galen_believes():
    # Reuse the helpers above so the sentence is always consistent with
    # the other endpoints.
    tc = top_candidate()
    tj = trajectory()
    ch = current_hypothesis()

    parts: list[str] = []
    cand = tc.get("candidate") if isinstance(tc, dict) else None
    if cand:
        parts.append(
            f"Galen's current best bet for Erik is "
            f"{cand['name']} ({cand['score']:.0%} confidence)."
        )

    alsfrs = tj.get("alsfrs_r", {}) if isinstance(tj, dict) else {}
    if alsfrs.get("current") is not None:
        parts.append(
            f"Erik's most recent ALSFRS-R reading was "
            f"{float(alsfrs['current']):.1f} of 48."
        )

    hyp = ch.get("hypothesis") if isinstance(ch, dict) else None
    if hyp and hyp.get("text"):
        text = str(hyp["text"])
        if len(text) > 180:
            text = text[:180].rstrip() + "…"
        parts.append(f"Actively investigating: {text}")

    if not parts:
        belief = (
            "Galen is still gathering evidence. A clearer picture of the "
            "best treatment path for Erik will appear here as the research "
            "loop closes in on it."
        )
    else:
        belief = " ".join(parts)

    return {
        "belief": belief,
        "components": {
            "has_candidate": bool(cand),
            "has_trajectory": alsfrs.get("current") is not None,
            "has_hypothesis": bool(hyp),
        },
    }
