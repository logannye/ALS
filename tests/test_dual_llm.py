"""Tests for DualLLMManager — two-tier LLM with memory management."""
from __future__ import annotations
import pytest
from research.dual_llm import DualLLMManager

class TestDualLLMManager:
    def test_instantiates(self):
        mgr = DualLLMManager(lazy=True)
        assert mgr._research_engine is None
        assert mgr._protocol_engine is None

    def test_get_research_engine_creates_once(self):
        mgr = DualLLMManager(lazy=True)
        engine = mgr.get_research_engine()
        assert engine is not None
        engine2 = mgr.get_research_engine()
        assert engine is engine2

    def test_get_protocol_engine_creates(self):
        mgr = DualLLMManager(lazy=True)
        engine = mgr.get_protocol_engine()
        assert engine is not None

    def test_unload_protocol_clears(self):
        mgr = DualLLMManager(lazy=True)
        _ = mgr.get_protocol_engine()
        mgr.unload_protocol_model()
        assert mgr._protocol_engine is None

    def test_research_model_id(self):
        mgr = DualLLMManager(lazy=True)
        mid = mgr._research_model_id
        assert "9B" in mid or "fallback" in mid.lower() or "nova-micro" in mid

    def test_protocol_model_id(self):
        mgr = DualLLMManager(lazy=True)
        mid = mgr._protocol_model_id
        assert "35B" in mid or "mxfp4" in mid or "nova-pro" in mid
