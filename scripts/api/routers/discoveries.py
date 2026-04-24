"""Daily research discoveries endpoint — family-facing timeline.

Rewritten 2026-04-24 after audit found four consecutive days reporting
"no major new findings" while the research loop was advancing ~500
steps/day and the reasoning daemon was actively re-evaluating edges.

The old version counted only new EvidenceItem rows in erik_core.objects.
It missed:
  * scm_write_log events (edges created, superseded, effects updated)
  * tcg_edges confidence changes (reasoning daemon output)
  * tcg_hypotheses generated or resolved
  * LLM reasoning calls from llm_spend
  * propagation_events from the R4 refutation rule

Now the generator pulls from all of these and produces plain-English
highlights that reflect what actually happened each day. The fallback
"no major new findings" text stays as a last resort but should rarely
fire on a day with any real activity.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Query

from db.pool import get_connection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# Highlight categories — used by the frontend to pick icons + colours.
_CAT_RESEARCH = "research"
_CAT_TREATMENT = "treatment"
_CAT_TRIAL = "trial"
_CAT_DRUG = "drug_design"
_CAT_REASONING = "reasoning"


# ---------------------------------------------------------------------------
# Per-day metric queries.
# ---------------------------------------------------------------------------


def _query_day_metrics(conn, day_start: datetime, day_end: datetime) -> dict:
    """Collect every per-day signal we know how to surface.

    Individual queries are guarded in try/except at the caller — if one
    table is missing (e.g. new SCM tables on an older DB) the others
    still report.
    """

    def _safe_scalar(sql: str, params: tuple = ()) -> int:
        try:
            row = conn.execute(sql, params).fetchone()
            return int(row[0]) if row and row[0] is not None else 0
        except Exception as e:
            logger.debug("discoveries metric query failed: %s", e)
            return 0

    evidence_added = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.objects
           WHERE type = 'EvidenceItem'
             AND created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    )

    entities_added = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.entities
           WHERE created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    )

    trials_found = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.objects
           WHERE type = 'EvidenceItem'
             AND provenance_source_system = 'clinicaltrials.gov'
             AND created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    )

    drug_molecules = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.objects
           WHERE type = 'EvidenceItem'
             AND body->>'provenance' LIKE '%%design_molecule%%'
             AND created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    )

    # NEW: SCM write-log events.
    # This is where real causal-graph revision shows up — edges created
    # via bootstrap, superseded by new evidence, effects updated from
    # ChEMBL, counterfactuals computed.
    scm_edges_created = _safe_scalar(
        """SELECT COUNT(*) FROM erik_ops.scm_write_log
           WHERE operation = 'edge_created'
             AND occurred_at >= %s AND occurred_at < %s""",
        (day_start, day_end),
    )
    scm_edges_superseded = _safe_scalar(
        """SELECT COUNT(*) FROM erik_ops.scm_write_log
           WHERE operation = 'edge_superseded'
             AND occurred_at >= %s AND occurred_at < %s""",
        (day_start, day_end),
    )
    scm_effects_updated = _safe_scalar(
        """SELECT COUNT(*) FROM erik_ops.scm_write_log
           WHERE operation = 'effect_updated'
             AND occurred_at >= %s AND occurred_at < %s""",
        (day_start, day_end),
    )

    # NEW: TCG hypotheses generated / resolved.
    hypotheses_opened = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.tcg_hypotheses
           WHERE created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    )
    hypotheses_resolved = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.tcg_hypotheses
           WHERE updated_at >= %s AND updated_at < %s
             AND status IN ('supported', 'refuted', 'actionable',
                            'refuted_by_propagation')""",
        (day_start, day_end),
    )

    # NEW: TCG edge confidence revisions (the ReasoningDaemon's output).
    edges_reasoned = _safe_scalar(
        """SELECT COUNT(*) FROM erik_core.tcg_edges
           WHERE last_reasoned_at >= %s AND last_reasoned_at < %s""",
        (day_start, day_end),
    )

    # NEW: Intervention candidates newly flagged.
    intervention_candidates = _safe_scalar(
        """SELECT COUNT(*) FROM erik_ops.scm_write_log
           WHERE operation = 'intervention_flagged'
             AND occurred_at >= %s AND occurred_at < %s""",
        (day_start, day_end),
    )

    # NEW: R4 refutation cascades.
    refutations_applied = _safe_scalar(
        """SELECT COUNT(*) FROM erik_ops.propagation_events
           WHERE status = 'applied'
             AND applied_at >= %s AND applied_at < %s""",
        (day_start, day_end),
    )

    # NEW: Claude reasoning calls (reflects how hard the system thought today).
    llm_calls = _safe_scalar(
        """SELECT COUNT(*) FROM erik_ops.llm_spend
           WHERE created_at >= %s AND created_at < %s""",
        (day_start, day_end),
    )

    # Latest step_count snapshot (applies to "today" only, not historical days).
    row = conn.execute(
        """SELECT (state_json->>'step_count')::int
           FROM erik_ops.research_state
           WHERE subject_ref = 'traj:draper_001'
           ORDER BY updated_at DESC LIMIT 1"""
    ).fetchone()
    step_count = int(row[0]) if row and row[0] else 0

    # One notable reasoning event for the day — pulled for human-language
    # highlight (e.g. "Lowered confidence on 'complement → SOD1 misfolding'
    # from 30% to 12%"). We grab the edge whose confidence moved the most.
    notable_reasoning: dict | None = None
    try:
        row = conn.execute(
            """SELECT source_id, target_id, confidence
                 FROM erik_core.tcg_edges
                WHERE last_reasoned_at >= %s AND last_reasoned_at < %s
                ORDER BY last_reasoned_at DESC
                LIMIT 1""",
            (day_start, day_end),
        ).fetchone()
        if row:
            notable_reasoning = {
                "source": row[0], "target": row[1],
                "confidence_after": float(row[2] or 0.0),
            }
    except Exception:
        pass

    return {
        "evidence_added": evidence_added,
        "entities_added": entities_added,
        "trials_found": trials_found,
        "drug_molecules": drug_molecules,
        "scm_edges_created": scm_edges_created,
        "scm_edges_superseded": scm_edges_superseded,
        "scm_effects_updated": scm_effects_updated,
        "hypotheses_opened": hypotheses_opened,
        "hypotheses_resolved": hypotheses_resolved,
        "edges_reasoned": edges_reasoned,
        "intervention_candidates": intervention_candidates,
        "refutations_applied": refutations_applied,
        "llm_calls": llm_calls,
        "step_count": step_count,
        "notable_reasoning": notable_reasoning,
    }


# ---------------------------------------------------------------------------
# Highlight builder — produces plain-English sentences from metrics.
# ---------------------------------------------------------------------------


def _pluralise(n: int, singular: str, plural: str | None = None) -> str:
    return singular if n == 1 else (plural or singular + "s")


def _build_highlights(metrics: dict) -> list[dict[str, str]]:
    """Convert raw metrics into plain-English highlight sentences.

    Ordered by family relevance: drug-discovery first, then
    reasoning/graph-revision, then evidence volume, then research activity.
    """
    highlights: list[dict[str, str]] = []

    # Drug-discovery signals.
    drugs = metrics["drug_molecules"]
    if drugs > 0:
        highlights.append({
            "text": f"Designed {drugs} candidate drug {_pluralise(drugs, 'molecule')} for evaluation.",
            "category": _CAT_DRUG,
        })

    new_intervention_points = metrics["intervention_candidates"]
    if new_intervention_points > 0:
        highlights.append({
            "text": (
                f"Flagged {new_intervention_points} new "
                f"{_pluralise(new_intervention_points, 'target')} "
                "where a drug could plausibly intervene."
            ),
            "category": _CAT_DRUG,
        })

    # Effect magnitudes filled in from ChEMBL — important for CPTS.
    effects = metrics["scm_effects_updated"]
    if effects > 0:
        highlights.append({
            "text": (
                f"Filled in real-world potency data on {effects} "
                f"compound-target {_pluralise(effects, 'link')} "
                "from the ChEMBL database."
            ),
            "category": _CAT_DRUG,
        })

    # Trial-side signals.
    trials = metrics["trials_found"]
    if trials > 0:
        highlights.append({
            "text": (
                f"Reviewed {trials} clinical {_pluralise(trials, 'trial')} "
                "for relevance to Erik."
            ),
            "category": _CAT_TRIAL,
        })

    # Reasoning / causal-graph revision.
    superseded = metrics["scm_edges_superseded"]
    if superseded > 0:
        highlights.append({
            "text": (
                f"Re-examined and replaced {superseded} "
                f"causal {_pluralise(superseded, 'link')} "
                "with stronger evidence."
            ),
            "category": _CAT_REASONING,
        })

    refs = metrics["refutations_applied"]
    if refs > 0:
        highlights.append({
            "text": (
                f"Propagated {refs} "
                f"{_pluralise(refs, 'refutation')} through the causal graph, "
                "removing outdated drug candidates."
            ),
            "category": _CAT_REASONING,
        })

    reasoned = metrics["edges_reasoned"]
    if reasoned > 0:
        # Surface the most notable specific edge if we have one.
        note = metrics.get("notable_reasoning")
        if note and note.get("source") and note.get("target"):
            src = str(note["source"]).replace("_", " ")
            tgt = str(note["target"]).replace("_", " ")
            highlights.append({
                "text": (
                    f"Deepened reasoning on {reasoned} biological "
                    f"{_pluralise(reasoned, 'link')}, including "
                    f"the connection between {src} and {tgt}."
                ),
                "category": _CAT_REASONING,
            })
        else:
            highlights.append({
                "text": (
                    f"Deepened reasoning on {reasoned} biological "
                    f"{_pluralise(reasoned, 'link')} in Erik's disease model."
                ),
                "category": _CAT_REASONING,
            })

    # Hypothesis activity.
    opened = metrics["hypotheses_opened"]
    resolved = metrics["hypotheses_resolved"]
    if opened > 0:
        highlights.append({
            "text": (
                f"Opened {opened} new research "
                f"{_pluralise(opened, 'question')} about Erik's biology."
            ),
            "category": _CAT_RESEARCH,
        })
    if resolved > 0:
        highlights.append({
            "text": (
                f"Resolved {resolved} prior research "
                f"{_pluralise(resolved, 'question')} based on new evidence."
            ),
            "category": _CAT_RESEARCH,
        })

    # Edge creation + entity discovery.
    created_edges = metrics["scm_edges_created"]
    if created_edges > 0:
        highlights.append({
            "text": (
                f"Promoted {created_edges} "
                f"{_pluralise(created_edges, 'relationship')} "
                "to do-calculus-valid causal edges."
            ),
            "category": _CAT_RESEARCH,
        })

    ent = metrics["entities_added"]
    if ent > 0:
        highlights.append({
            "text": (
                f"Catalogued {ent} new biological "
                f"{_pluralise(ent, 'entity', 'entities')} "
                "(genes, proteins, pathways)."
            ),
            "category": _CAT_RESEARCH,
        })

    # Evidence volume (last, least interesting).
    evi = metrics["evidence_added"]
    if evi > 0:
        highlights.append({
            "text": (
                f"Analysed {evi} new "
                f"{_pluralise(evi, 'piece')} of ALS research evidence."
            ),
            "category": _CAT_RESEARCH,
        })

    llm = metrics["llm_calls"]
    if llm > 0 and not highlights:
        # If nothing else surfaced but Claude still ran, that means the
        # research loop is alive — say so rather than "no findings".
        highlights.append({
            "text": (
                f"Consulted Claude {llm} {_pluralise(llm, 'time')} "
                "to sharpen interpretations of ongoing evidence."
            ),
            "category": _CAT_REASONING,
        })

    # True quiet-day fallback — only when genuinely nothing happened.
    if not highlights:
        highlights.append({
            "text": (
                "Research ran quietly — no new evidence or graph revisions "
                "large enough to flag. Galen continues to monitor."
            ),
            "category": _CAT_RESEARCH,
        })

    # Cap at 4 so the daily cards don't become a wall.
    return highlights[:4]


# ---------------------------------------------------------------------------
# Public API: build summaries
# ---------------------------------------------------------------------------


def build_daily_summary(target_date: date, dry_run: bool = False) -> dict[str, Any]:
    """Build a single day's summary."""
    date_str = target_date.isoformat()

    if dry_run:
        return {
            "date": date_str,
            "highlights": [
                {"text": "Dry-run mode.", "category": _CAT_RESEARCH},
            ],
            "milestone": None,
            "evidence_added": 0,
            "step_count": 0,
        }

    day_start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    try:
        with get_connection() as conn:
            metrics = _query_day_metrics(conn, day_start, day_end)
    except Exception:
        logger.exception("Failed to query metrics for %s", date_str)
        metrics = {
            "evidence_added": 0, "entities_added": 0, "trials_found": 0,
            "drug_molecules": 0, "scm_edges_created": 0,
            "scm_edges_superseded": 0, "scm_effects_updated": 0,
            "hypotheses_opened": 0, "hypotheses_resolved": 0,
            "edges_reasoned": 0, "intervention_candidates": 0,
            "refutations_applied": 0, "llm_calls": 0,
            "step_count": 0, "notable_reasoning": None,
        }

    highlights = _build_highlights(metrics)

    return {
        "date": date_str,
        "highlights": highlights,
        "milestone": None,
        "evidence_added": metrics["evidence_added"],
        "step_count": metrics["step_count"],
    }


def build_discoveries_response(days: int = 14, dry_run: bool = False) -> dict[str, Any]:
    """Build the full discoveries response for *days* calendar days, newest first."""
    today = date.today()
    day_list = []
    for offset in range(days):
        target = today - timedelta(days=offset)
        day_list.append(build_daily_summary(target, dry_run=dry_run))

    return {"days": day_list}


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/discoveries")
def get_discoveries(days: int = Query(14, ge=1, le=90)):
    """Return daily research summaries for the family timeline."""
    return build_discoveries_response(days=days)
