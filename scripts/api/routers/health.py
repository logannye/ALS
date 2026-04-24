"""Health check endpoints — no auth required.

Two surfaces:

  * ``/health`` — process-level liveness. Used by Railway's default
    healthcheck + by Vercel/frontend "are we alive" pings. Stays 200 as
    long as the process and DB are reachable.

  * ``/health/research`` — research-loop freshness. Returns 503 when
    ``erik_ops.research_state.updated_at`` is older than the configured
    staleness window. This is what actually tells Railway to restart the
    *research worker* service when the loop has hung — a crashed loop
    inside a healthy API process was the single most-dangerous silent
    failure mode in the pre-split architecture.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from db.pool import get_connection

router = APIRouter()
_start_time = time.time()

# Canonical Erik subject_ref; mirrors ingestion/patient_builder.SUBJECT_REF.
_SUBJECT_REF = "traj:draper_001"

# Freshness window: if research_state hasn't been updated in this many
# seconds, we consider the loop hung. Overridable via env so Railway can
# tune without a redeploy. Default 30 min — generous since a deep-research
# step with LLM calls can take 2-3 minutes.
_DEFAULT_STALENESS_S = 30 * 60


@router.get("/health")
def health():
    """Basic health check: process uptime + DB connectivity."""
    db_ok = False
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
            db_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if db_ok else "degraded",
        "uptime_s": round(time.time() - _start_time, 1),
        "database": "connected" if db_ok else "unreachable",
    }


@router.get("/health/research")
def health_research():
    """Report whether the research loop is making progress.

    Returns 200 + liveness details when research_state has advanced
    within the staleness window. Returns 503 when:
      * research_state row is missing (the loop never started), OR
      * research_state.updated_at is older than ERIK_RESEARCH_STALENESS_S
        seconds (the loop hung or crashed).

    Railway can wire this to a service healthcheck so the *worker* gets
    restarted when the loop hangs, without taking the API down.
    """
    staleness_s = int(os.environ.get("ERIK_RESEARCH_STALENESS_S", _DEFAULT_STALENESS_S))
    now = datetime.now(timezone.utc)

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT updated_at, (state_json->>'step_count')::bigint
                         FROM erik_ops.research_state
                        WHERE subject_ref = %s""",
                    (_SUBJECT_REF,),
                )
                row = cur.fetchone()
    except Exception as exc:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "db_unreachable",
                "detail": str(exc)[:200],
            },
        )

    if row is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "reason": "no_research_state",
                "staleness_window_s": staleness_s,
            },
        )

    updated_at, step_count = row
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age_s = (now - updated_at).total_seconds()
    is_fresh = age_s < staleness_s

    body = {
        "status": "ok" if is_fresh else "degraded",
        "step_count": int(step_count) if step_count is not None else None,
        "last_updated_s_ago": round(age_s, 1),
        "staleness_window_s": staleness_s,
        "is_fresh": is_fresh,
    }
    if not is_fresh:
        body["reason"] = "research_loop_stalled"
        return JSONResponse(status_code=503, content=body)
    return body
