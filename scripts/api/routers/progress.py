"""Progress dashboard API endpoints."""
from fastapi import APIRouter
from db.pool import get_connection
from tcg.graph import TCGraph

router = APIRouter(prefix="/api")
_graph = TCGraph()


@router.get("/progress")
def get_progress():
    summary = _graph.summary()
    edge_count = summary["edge_count"]
    mean_conf = summary["mean_confidence"]

    # Therapeutic coverage: edges with confidence > 0.7 that have intervention_potential.druggable
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) FROM erik_core.tcg_edges
                WHERE confidence > 0.7 AND intervention_potential->>'druggable' = 'true'
            """)
            covered = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM erik_core.tcg_edges WHERE confidence > 0.7")
            high_conf = cur.fetchone()[0]
            cur.execute("""
                SELECT count(*) FROM erik_core.objects
                WHERE tcg_integrated = TRUE AND status = 'active'
            """)
            integrated = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM erik_core.objects WHERE status = 'active'")
            total_evidence = cur.fetchone()[0]

    # Hypothesis counts by status
    hyp_counts = {}
    for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
        hyp_counts[status] = len(_graph.get_hypotheses_by_status(status))

    return {
        "graph_confidence": mean_conf,
        "node_count": summary["node_count"],
        "edge_count": edge_count,
        "therapeutic_coverage": covered / max(high_conf, 1),
        "pathway_completeness": summary["clusters"],
        "hypothesis_pipeline": hyp_counts,
        "evidence_utilization": integrated / max(total_evidence, 1),
    }


@router.get("/progress/phases")
def get_phase_status():
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Last activity per phase
            cur.execute("""
                SELECT phase, max(created_at) as last_run, count(*) as total_events
                FROM erik_ops.activity_feed
                GROUP BY phase
            """)
            phases = {r[0]: {"last_run": r[1].isoformat() if r[1] else None,
                             "total_events": r[2]} for r in cur.fetchall()}

            # Monthly LLM spend
            cur.execute("""
                SELECT model, sum(cost_usd), count(*)
                FROM erik_ops.llm_spend
                WHERE created_at > date_trunc('month', now())
                GROUP BY model
            """)
            spend = {r[0]: {"cost_usd": float(r[1]), "calls": r[2]} for r in cur.fetchall()}

    return {"phases": phases, "llm_spend": spend}
