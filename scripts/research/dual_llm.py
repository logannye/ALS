"""DualLLMManager — two-tier LLM management for the research loop.

- Research tier (9B, ~4.7GB): stays loaded for hypothesis generation,
  causal chain extension, evidence scoring.
- Protocol tier (35B, ~17GB): loaded on demand for protocol regeneration,
  unloaded immediately after to free memory.
"""
from __future__ import annotations
from typing import Optional
from llm.inference import LLMInference
from world_model.reasoning_engine import ReasoningEngine

_FALLBACK_MODEL = "/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit"
_PRIMARY_MODEL = "/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4"

class DualLLMManager:
    """Manages lazy loading and unloading of two LLM tiers."""

    def __init__(
        self,
        lazy: bool = False,
        research_model_path: Optional[str] = None,
        protocol_model_path: Optional[str] = None,
    ) -> None:
        self._research_model_path = research_model_path or _FALLBACK_MODEL
        self._protocol_model_path = protocol_model_path or _PRIMARY_MODEL
        self._lazy = lazy
        self._research_engine: Optional[ReasoningEngine] = None
        self._protocol_engine: Optional[ReasoningEngine] = None

    def get_research_engine(self) -> ReasoningEngine:
        """Return the 9B-backed reasoning engine (stays loaded)."""
        if self._research_engine is None:
            self._research_engine = ReasoningEngine(
                lazy=self._lazy,
                model_path=self._research_model_path,
            )
        return self._research_engine

    def get_protocol_engine(self) -> ReasoningEngine:
        """Return the 35B-backed reasoning engine (loaded on demand)."""
        if self._protocol_engine is None:
            self._protocol_engine = ReasoningEngine(
                lazy=self._lazy,
                model_path=self._protocol_model_path,
            )
        return self._protocol_engine

    def unload_protocol_model(self) -> None:
        """Free the 35B model memory."""
        if self._protocol_engine is not None:
            self._protocol_engine._llm.unload()
            self._protocol_engine = None
