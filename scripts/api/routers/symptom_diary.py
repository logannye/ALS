"""Family symptom-diary endpoints.

Lets family members leave short "how is Erik feeling today?" reports
straight from the dashboard. Designed for one-tap mood + optional text
note — friction-free for a stressed family.

Writes land in append-only erik_ops.symptom_reports; the research loop
can use these as qualitative evidence when building Erik's patient
model (future wiring — not in this PR).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from db.pool import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


# Coarse keyword lifter — turns free text into symptom tags the research
# loop can aggregate. Deliberately small / conservative: we'd rather miss
# a subtle symptom than misclassify a family's note.
_SYMPTOM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "weakness":        ("weak", "weakness", "tired legs", "tired arms", "fatigue"),
    "cramping":        ("cramp", "cramps", "cramping", "spasm"),
    "fasciculations":  ("twitch", "twitches", "twitching", "fasciculation"),
    "speech":          ("slurred", "speech", "hard to talk", "talking"),
    "swallowing":      ("swallow", "choking", "swallowing"),
    "breathing":       ("breath", "breathless", "short of breath", "breathing"),
    "mobility":        ("walk", "walking", "falls", "fell"),
    "mood":            ("sad", "depressed", "anxious", "anxiety", "scared"),
    "sleep":           ("sleep", "insomnia", "woke up", "couldn't sleep"),
    "pain":            ("pain", "hurting", "ache", "aching"),
}


def _extract_tags(note: str) -> list[str]:
    if not note:
        return []
    nlow = note.lower()
    hits: set[str] = set()
    for tag, kws in _SYMPTOM_KEYWORDS.items():
        if any(kw in nlow for kw in kws):
            hits.add(tag)
    return sorted(hits)


# ─── POST /api/symptom-report ────────────────────────────────────────────────


class SymptomReportIn(BaseModel):
    mood: Optional[int] = Field(None, ge=1, le=5)
    note: str = Field("", max_length=2048)
    reporter_name: str = Field("family", max_length=64)

    @field_validator("note", "reporter_name", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v


@router.post("/symptom-report")
def create_symptom_report(report: SymptomReportIn):
    """Append a family symptom report.

    Requires at least one of: mood score OR note text. Empty reports
    are rejected to avoid accidental submissions from double-taps.
    """
    if report.mood is None and not report.note:
        raise HTTPException(
            status_code=400,
            detail="Provide a mood score, a note, or both.",
        )

    tags = _extract_tags(report.note)
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO erik_ops.symptom_reports
                          (reporter_name, mood, note, symptoms_mentioned)
                       VALUES (%s, %s, %s, %s)
                       RETURNING id, created_at""",
                    (
                        report.reporter_name or "family",
                        report.mood,
                        report.note or "",
                        tags,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
    except Exception:
        logger.exception("symptom_report insert failed")
        raise HTTPException(status_code=500, detail="Could not save the note.")

    return {
        "ok": True,
        "id": int(row[0]) if row else None,
        "created_at": row[1].isoformat() if row and row[1] else None,
        "symptoms_mentioned": tags,
    }


# ─── GET /api/symptom-report/recent ──────────────────────────────────────────


@router.get("/symptom-report/recent")
def recent_symptom_reports(days: int = 14, limit: int = 20):
    """Return the last N days of reports so the dashboard can show a
    short timeline. Most-recent first."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, min(days, 90)))
    lim = max(1, min(limit, 100))
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, reporter_name, mood, note,
                              symptoms_mentioned, created_at
                         FROM erik_ops.symptom_reports
                        WHERE created_at >= %s
                        ORDER BY created_at DESC
                        LIMIT %s""",
                    (cutoff, lim),
                )
                rows = cur.fetchall()
    except Exception:
        logger.exception("symptom_report list failed")
        return {"reports": []}

    return {
        "reports": [
            {
                "id": int(r[0]),
                "reporter_name": r[1],
                "mood": r[2],
                "note": r[3] or "",
                "symptoms_mentioned": list(r[4] or []),
                "created_at": r[5].isoformat() if r[5] else None,
            }
            for r in rows
        ]
    }
