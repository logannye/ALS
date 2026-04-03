"""Research loop state endpoint — mirrors monitor.py data."""
from __future__ import annotations

import json

from fastapi import APIRouter

from db.pool import get_connection

router = APIRouter(prefix="/api")


@router.get("/state")
def get_state():
    """Return current research loop state + DB counts."""
    state = None
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT state_json, updated_at
                   FROM erik_ops.research_state
                   ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()
            if row:
                state_json, updated_at = row
                if isinstance(state_json, str):
                    state_json = json.loads(state_json)
                state_json["_updated_at"] = str(updated_at)
                state = state_json
    except Exception as e:
        return {"error": str(e)}

    # Evidence and episode counts
    evidence_count = 0
    episode_count = 0
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM erik_core.objects WHERE type = 'EvidenceItem' AND status = 'active'"
            ).fetchone()
            evidence_count = row[0] if row else 0

            row = conn.execute(
                "SELECT COUNT(*) FROM erik_core.objects WHERE type = 'LearningEpisode'"
            ).fetchone()
            episode_count = row[0] if row else 0
    except Exception:
        pass

    return {
        "state": state,
        "evidence_count": evidence_count,
        "episode_count": episode_count,
    }
