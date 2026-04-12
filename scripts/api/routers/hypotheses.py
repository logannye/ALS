"""Hypothesis pipeline API endpoints."""
from fastapi import APIRouter
from tcg.graph import TCGraph

router = APIRouter(prefix="/api")
_graph = TCGraph()


@router.get("/hypotheses")
def list_hypotheses():
    all_hyps = []
    for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
        all_hyps.extend(_graph.get_hypotheses_by_status(status))
    return {"hypotheses": [h.to_dict() for h in all_hyps]}


@router.get("/hypotheses/{hypothesis_id:path}")
def get_hypothesis(hypothesis_id: str):
    for status in ["proposed", "under_investigation", "supported", "refuted", "actionable"]:
        for h in _graph.get_hypotheses_by_status(status):
            if h.id == hypothesis_id:
                return h.to_dict()
    return {"error": "Hypothesis not found"}
