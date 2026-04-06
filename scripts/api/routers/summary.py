"""Daily summary endpoint — LLM-generated plain-English overview for the family."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter

from db.pool import get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# In-memory cache: (timestamp, response_dict)
_cache: tuple[float, dict] | None = None
_CACHE_TTL = 900  # 15 minutes


def _get_last_24h_stats() -> dict:
    """Query activity events and evidence from the last 24 hours."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=24)

    with get_connection() as conn:
        # Count new evidence items in last 24h
        row = conn.execute(
            """SELECT COUNT(*) FROM erik_core.objects
               WHERE type = 'EvidenceItem' AND status = 'active'
                 AND created_at >= %s""",
            [cutoff],
        ).fetchone()
        new_evidence = row[0] if row else 0

        # Count total evidence
        row = conn.execute(
            "SELECT COUNT(*) FROM erik_core.objects WHERE type = 'EvidenceItem' AND status = 'active'"
        ).fetchone()
        total_evidence = row[0] if row else 0

        # Get recent activity event types + counts
        rows = conn.execute(
            """SELECT event_type, COUNT(*) as cnt
               FROM erik_ops.audit_events
               WHERE created_at >= %s
               GROUP BY event_type
               ORDER BY cnt DESC""",
            [cutoff],
        ).fetchall()
        event_counts = {r[0]: r[1] for r in rows} if rows else {}

        # Get top 5 recent evidence claims for LLM context
        rows = conn.execute(
            """SELECT body->>'claim', body->>'strength', body->>'direction'
               FROM erik_core.objects
               WHERE type = 'EvidenceItem' AND status = 'active'
                 AND created_at >= %s
               ORDER BY created_at DESC LIMIT 5""",
            [cutoff],
        ).fetchall()
        recent_claims = [
            {"claim": r[0], "strength": r[1], "direction": r[2]}
            for r in rows
        ] if rows else []

        # Get research state
        row = conn.execute(
            """SELECT state_json, updated_at
               FROM erik_ops.research_state
               ORDER BY updated_at DESC LIMIT 1"""
        ).fetchone()
        state = None
        if row:
            state_json = row[0]
            if isinstance(state_json, str):
                state_json = json.loads(state_json)
            state = state_json

        # Check protocol updates in last 24h
        row = conn.execute(
            """SELECT COUNT(*) FROM erik_core.objects
               WHERE type = 'CureProtocolCandidate'
                 AND updated_at >= %s""",
            [cutoff],
        ).fetchone()
        protocol_updated = (row[0] if row else 0) > 0

        # Check new trials in last 24h
        row = conn.execute(
            """SELECT COUNT(*) FROM erik_core.objects
               WHERE type IN ('ClinicalTrial')
                 AND created_at >= %s""",
            [cutoff],
        ).fetchone()
        new_trials = row[0] if row else 0

        # Check last ALSFRS-R upload date
        row = conn.execute(
            """SELECT created_at FROM erik_core.objects
               WHERE type = 'Observation'
                 AND body->>'id' LIKE 'obs:alsfrs_r:%'
               ORDER BY created_at DESC LIMIT 1"""
        ).fetchone()
        last_alsfrs_date = str(row[0].date()) if row else None

    return {
        "new_evidence": new_evidence,
        "total_evidence": total_evidence,
        "event_counts": event_counts,
        "recent_claims": recent_claims,
        "state": state,
        "protocol_updated": protocol_updated,
        "new_trials": new_trials,
        "last_alsfrs_date": last_alsfrs_date,
    }


_SUMMARY_PROMPT = """\
You are writing a brief daily update for the Draper family about their ALS \
research system called Galen. The family is NOT medical professionals — write \
at an 11th-grade reading level. Be warm, clear, and confident. No jargon \
without explanation.

RESEARCH DATA FROM THE LAST 24 HOURS:
{data_json}

Write a JSON response with this EXACT schema:
{{
  "summary": "<2-4 sentence paragraph about what Galen did today and what it found. \
Mention specific drugs or findings if the evidence claims contain them. \
Be warm but factual. Do not fabricate claims beyond the data provided.>",
  "chips": [
    {{"label": "<short label like '+3 evidence' or 'Plan stable'>", "type": "<evidence|plan|trial|alert>"}}
  ],
  "action_items": [
    {{
      "title": "<what the family should do, e.g. 'Ask your neurologist about X'>",
      "description": "<1-2 sentence explanation of why>",
      "category": "<drug|trial|upload|general>",
      "link_to": "<optional route like /trials or /protocol>"
    }}
  ]
}}

RULES:
- The "chips" array should have 2-4 items summarizing key changes.
- The "action_items" array should have 1-3 items the family can act on.
- If the last ALSFRS-R assessment was more than 25 days ago, include an action \
item reminding them to do another one (category "upload", link_to "/upload").
- If new trials were found, include an action item about them (category "trial", \
link_to "/trials").
- Keep all language simple and compassionate. The patient's name is Erik.
- Return ONLY the JSON object. No markdown, no prose outside the JSON.
"""


def _generate_summary_with_llm(stats: dict) -> dict | None:
    """Call the LLM to generate a family-friendly daily summary."""
    try:
        from llm.inference import create_llm

        llm = create_llm(max_tokens=1000, temperature=0.3)
        prompt = _SUMMARY_PROMPT.format(data_json=json.dumps(stats, default=str))
        result = llm.generate_json(prompt, max_tokens=1000)
        return result
    except Exception:
        logger.exception("LLM summary generation failed")
        return None


def _build_fallback_summary(stats: dict) -> dict:
    """Template-based fallback if LLM is unavailable."""
    new_evi = stats["new_evidence"]
    total = stats["total_evidence"]
    state = stats.get("state") or {}

    # Summary text
    if new_evi > 0:
        summary = (
            f"Galen found {new_evi} new piece{'s' if new_evi != 1 else ''} of "
            f"evidence in the last 24 hours, bringing the total to {total:,}. "
            "The research is ongoing and the treatment plan is being kept up to date."
        )
    else:
        summary = (
            f"Galen continued researching over the last 24 hours. "
            f"There are {total:,} pieces of evidence collected so far. "
            "No major new findings today, but the search continues."
        )

    # Chips
    chips = []
    if new_evi > 0:
        chips.append({"label": f"+{new_evi} evidence", "type": "evidence"})
    if stats["protocol_updated"]:
        chips.append({"label": "Plan updated", "type": "plan"})
    else:
        chips.append({"label": "Plan stable", "type": "plan"})
    if stats["new_trials"] > 0:
        chips.append({"label": f"{stats['new_trials']} new trial{'s' if stats['new_trials'] != 1 else ''}", "type": "trial"})

    # Action items
    action_items = []
    if stats["new_trials"] > 0:
        action_items.append({
            "title": "New clinical trials found",
            "description": f"Galen found {stats['new_trials']} trial{'s' if stats['new_trials'] != 1 else ''} that may be relevant for Erik.",
            "category": "trial",
            "link_to": "/trials",
        })

    # ALSFRS-R reminder
    if stats.get("last_alsfrs_date"):
        from datetime import date
        days_since = (date.today() - datetime.fromisoformat(stats["last_alsfrs_date"]).date()).days
        if days_since > 25:
            action_items.append({
                "title": "Functional assessment is due",
                "description": f"The last one was {days_since} days ago. Monthly updates help Galen refine the plan.",
                "category": "upload",
                "link_to": "/upload",
            })
    elif stats.get("last_alsfrs_date") is None:
        action_items.append({
            "title": "Submit a functional assessment",
            "description": "An ALSFRS-R score helps Galen understand how Erik is doing day to day.",
            "category": "upload",
            "link_to": "/upload",
        })

    return {
        "summary": summary,
        "chips": chips,
        "action_items": action_items,
    }


@router.get("/summary")
def get_summary():
    """Return an LLM-generated daily summary for the Hub page.

    Cached in memory for 15 minutes to avoid redundant LLM calls.
    Falls back to a template-based summary if the LLM is unavailable.
    """
    global _cache

    now = time.time()
    if _cache and (now - _cache[0]) < _CACHE_TTL:
        return _cache[1]

    # Gather stats from DB
    stats = _get_last_24h_stats()
    state = stats.get("state") or {}

    # Calculate running days — use earliest object creation as proxy for launch
    step_count = state.get("step_count", 0)
    days_running = 1
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT MIN(created_at) FROM erik_core.objects
                   WHERE type = 'EvidenceItem' AND status = 'active'"""
            ).fetchone()
            if row and row[0]:
                delta = datetime.now(timezone.utc) - row[0]
                days_running = max(1, delta.days + 1)
    except Exception:
        pass

    # Try LLM, fall back to template
    llm_result = _generate_summary_with_llm(stats)
    if llm_result:
        summary_data = llm_result
    else:
        summary_data = _build_fallback_summary(stats)

    # Steps taken today — difference between current step and 24h-ago step
    steps_today = 0
    try:
        with get_connection() as conn:
            row = conn.execute(
                """SELECT (state_json->>'step_count')::int
                   FROM erik_ops.research_state
                   WHERE updated_at <= NOW() - INTERVAL '24 hours'
                   ORDER BY updated_at DESC LIMIT 1"""
            ).fetchone()
            prev_steps = row[0] if row else 0
            steps_today = max(0, step_count - prev_steps)
    except Exception:
        pass
    # If no prior state row exists, steps_today stays 0 (honest)

    response = {
        "date": datetime.now(timezone.utc).strftime("%A, %B %-d"),
        "summary": summary_data.get("summary", ""),
        "chips": summary_data.get("chips", []),
        "action_items": summary_data.get("action_items", []),
        "stats": {
            "steps_total": step_count,
            "evidence_total": stats["total_evidence"],
            "studies_today": steps_today,
            "days_running": days_running,
        },
        "plan_version": state.get("protocol_version", 0),
        "plan_stable": not stats["protocol_updated"],
    }

    _cache = (now, response)
    return response
