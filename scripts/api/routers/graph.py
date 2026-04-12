"""TCG exploration API endpoints."""
from fastapi import APIRouter, Query
from tcg.graph import TCGraph

router = APIRouter(prefix="/api")
_graph = TCGraph()


@router.get("/graph/summary")
def graph_summary():
    return _graph.summary()


@router.get("/graph/cluster/{cluster_name}")
def graph_cluster(cluster_name: str):
    nodes = _graph.list_nodes(pathway_cluster=cluster_name)
    return {"cluster": cluster_name, "nodes": [n.to_dict() for n in nodes]}


@router.get("/graph/edge/{edge_id:path}")
def graph_edge(edge_id: str):
    edge = _graph.get_edge(edge_id)
    if edge is None:
        return {"error": "Edge not found"}
    return edge.to_dict()


@router.get("/graph/weakest")
def graph_weakest(limit: int = Query(default=10, le=100)):
    edges = _graph.get_weakest_edges(limit=limit)
    return {"edges": [e.to_dict() for e in edges]}
