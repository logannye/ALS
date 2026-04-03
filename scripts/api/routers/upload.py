"""Data upload endpoints — lab results and ALSFRS-R scores from the family."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.pool import get_connection

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class LabResultUpload(BaseModel):
    name: str
    value: float
    unit: str
    reference_low: float | None = None
    reference_high: float | None = None
    date: str  # ISO date string YYYY-MM-DD
    notes: str | None = None


class ALSFRSRUpload(BaseModel):
    """12-item ALSFRS-R questionnaire (each 0-4, total max 48)."""
    date: str
    speech: int
    salivation: int
    swallowing: int
    handwriting: int
    cutting: int
    dressing: int
    turning: int
    walking: int
    climbing: int
    dyspnea: int
    orthopnea: int
    respiratory_insufficiency: int
    notes: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload/lab-results")
def upload_lab_result(lab: LabResultUpload):
    """Upload a single lab result. Creates an Observation in erik_core.objects."""
    # Validate date
    try:
        parsed_date = datetime.strptime(lab.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    # Build slug for ID
    slug = re.sub(r"[^a-z0-9]+", "_", lab.name.lower()).strip("_")
    obj_id = f"obs:lab:{slug}:{lab.date}"

    body = {
        "observation_kind": "lab_result",
        "name": lab.name,
        "value": lab.value,
        "unit": lab.unit,
        "reference_low": lab.reference_low,
        "reference_high": lab.reference_high,
        "collection_date": lab.date,
        "notes": lab.notes,
        "is_abnormal": _is_abnormal(lab.value, lab.reference_low, lab.reference_high),
        "uploaded_by": "family",
    }

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO erik_core.objects (id, type, status, body, provenance_source_system, time_observed_at)
               VALUES (%s, 'Observation', 'active', %s, 'family_upload', %s)
               ON CONFLICT (id) DO UPDATE SET
                 body = EXCLUDED.body,
                 updated_at = NOW()""",
            (obj_id, json.dumps(body), parsed_date),
        )
        conn.commit()

    return {"id": obj_id, "status": "created"}


@router.post("/upload/alsfrs-r")
def upload_alsfrs_r(score: ALSFRSRUpload):
    """Upload an ALSFRS-R assessment."""
    try:
        parsed_date = datetime.strptime(score.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")

    items = {
        "speech": score.speech,
        "salivation": score.salivation,
        "swallowing": score.swallowing,
        "handwriting": score.handwriting,
        "cutting": score.cutting,
        "dressing": score.dressing,
        "turning": score.turning,
        "walking": score.walking,
        "climbing": score.climbing,
        "dyspnea": score.dyspnea,
        "orthopnea": score.orthopnea,
        "respiratory_insufficiency": score.respiratory_insufficiency,
    }
    total = sum(items.values())

    # Validate ranges
    for key, val in items.items():
        if not 0 <= val <= 4:
            raise HTTPException(status_code=400, detail=f"{key} must be 0-4, got {val}")

    # Subscores
    bulbar = score.speech + score.salivation + score.swallowing
    fine_motor = score.handwriting + score.cutting + score.dressing
    gross_motor = score.turning + score.walking + score.climbing
    respiratory = score.dyspnea + score.orthopnea + score.respiratory_insufficiency

    obj_id = f"obs:alsfrs_r:{score.date}"
    body = {
        "observation_kind": "alsfrs_r",
        "date": score.date,
        "items": items,
        "total": total,
        "subscores": {
            "bulbar": bulbar,
            "fine_motor": fine_motor,
            "gross_motor": gross_motor,
            "respiratory": respiratory,
        },
        "notes": score.notes,
        "uploaded_by": "family",
    }

    with get_connection() as conn:
        conn.execute(
            """INSERT INTO erik_core.objects (id, type, status, body, provenance_source_system, time_observed_at)
               VALUES (%s, 'Observation', 'active', %s, 'family_upload', %s)
               ON CONFLICT (id) DO UPDATE SET
                 body = EXCLUDED.body,
                 updated_at = NOW()""",
            (obj_id, json.dumps(body), parsed_date),
        )
        conn.commit()

    return {"id": obj_id, "total": total, "status": "created"}


def _is_abnormal(value: float, low: float | None, high: float | None) -> bool:
    if low is not None and value < low:
        return True
    if high is not None and value > high:
        return True
    return False
