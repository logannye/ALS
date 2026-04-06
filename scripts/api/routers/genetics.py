"""Genetic testing upload endpoint — unlocks Layer 2→3 transition."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from db.pool import get_connection

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class GeneticProfileUpload(BaseModel):
    gene: str
    variant: str
    subtype: str
    test_date: str  # ISO date YYYY-MM-DD
    lab_name: str | None = None
    notes: str | None = None


@router.post("/upload/genetics")
def upload_genetics(profile: GeneticProfileUpload):
    """Upload genetic testing results. This triggers the Layer 2→3 transition.

    Creates a GeneticProfile observation in erik_core.objects and updates
    the research_state with the genetic_profile field so the layer
    orchestrator can advance to erik_specific.
    """
    # Validate date
    try:
        parsed_date = datetime.strptime(profile.test_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Invalid date format. Use YYYY-MM-DD."
        )

    obj_id = f"obs:genetic_profile:{profile.gene.lower()}:{profile.test_date}"
    genetic_profile = {
        "gene": profile.gene,
        "variant": profile.variant,
        "subtype": profile.subtype,
    }

    body = {
        "observation_kind": "genetic_profile",
        **genetic_profile,
        "test_date": profile.test_date,
        "lab_name": profile.lab_name,
        "notes": profile.notes,
        "uploaded_by": "family",
    }

    with get_connection() as conn:
        # 1. Store the genetic profile as an observation
        conn.execute(
            """INSERT INTO erik_core.objects
                   (id, type, status, body, provenance_source_system, time_observed_at)
               VALUES (%s, 'Observation', 'active', %s, 'family_upload', %s)
               ON CONFLICT (id) DO UPDATE SET
                 body = EXCLUDED.body,
                 updated_at = NOW()""",
            (obj_id, json.dumps(body), parsed_date),
        )

        # 2. Update research_state with genetic_profile so the layer
        #    orchestrator can transition to Layer 3 on the next step
        try:
            row = conn.execute(
                """SELECT state_json FROM erik_ops.research_state
                   ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()
            if row:
                state = row[0]
                if isinstance(state, str):
                    state = json.loads(state)
                state["genetic_profile"] = genetic_profile
                conn.execute(
                    """UPDATE erik_ops.research_state
                       SET state_json = %s, updated_at = NOW()
                       WHERE updated_at = (
                           SELECT MAX(updated_at) FROM erik_ops.research_state
                       )""",
                    (json.dumps(state),),
                )
        except Exception:
            logger.exception("Failed to update research_state with genetic_profile")

        conn.commit()

    logger.info(
        "Genetic profile uploaded: gene=%s variant=%s subtype=%s",
        profile.gene,
        profile.variant,
        profile.subtype,
    )

    return {"id": obj_id, "status": "created", "genetic_profile": genetic_profile}
