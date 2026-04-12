"""Tests for TCG API endpoints."""
import pytest
from unittest.mock import patch, MagicMock


class TestGraphEndpoints:
    def test_graph_summary_import(self):
        from api.routers.graph import router
        routes = [r.path for r in router.routes]
        assert any("graph/summary" in p for p in routes)
        assert any("graph/weakest" in p for p in routes)

    def test_hypotheses_import(self):
        from api.routers.hypotheses import router
        routes = [r.path for r in router.routes]
        assert any("hypotheses" in p for p in routes)

    def test_progress_import(self):
        from api.routers.progress import router
        routes = [r.path for r in router.routes]
        assert any(p.endswith("/progress") for p in routes)
        assert any("progress/phases" in p for p in routes)
