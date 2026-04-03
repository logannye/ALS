"""DualLLMManager — two-tier LLM management for the research loop.

- Research tier: high-volume (hypothesis generation, causal chain extension,
  evidence scoring).  Local: 9B MLX.  Cloud: Nova Micro on Bedrock.
- Protocol tier: low-volume, complex reasoning (protocol regeneration).
  Local: 35B MLX.  Cloud: Nova Pro on Bedrock.

Backend selection is automatic via :func:`llm.inference.create_llm` which
reads the ``ERIK_LLM_BACKEND`` env var (``"mlx"`` or ``"bedrock"``).
"""
from __future__ import annotations
from typing import Optional
from llm.inference import create_llm
from world_model.reasoning_engine import ReasoningEngine

# MLX local model paths (used when ERIK_LLM_BACKEND=mlx)
_MLX_RESEARCH = "/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit"
_MLX_PROTOCOL = "/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4"

# Bedrock model IDs (used when ERIK_LLM_BACKEND=bedrock)
_BEDROCK_RESEARCH = "amazon.nova-micro-v1:0"
_BEDROCK_PROTOCOL = "amazon.nova-pro-v1:0"


class DualLLMManager:
    """Manages lazy loading and unloading of two LLM tiers."""

    def __init__(
        self,
        lazy: bool = False,
        research_model_id: Optional[str] = None,
        protocol_model_id: Optional[str] = None,
    ) -> None:
        import os
        backend = os.environ.get("ERIK_LLM_BACKEND", "mlx").lower()

        if backend == "bedrock":
            self._research_model_id = research_model_id or _BEDROCK_RESEARCH
            self._protocol_model_id = protocol_model_id or _BEDROCK_PROTOCOL
        else:
            self._research_model_id = research_model_id or _MLX_RESEARCH
            self._protocol_model_id = protocol_model_id or _MLX_PROTOCOL

        self._lazy = lazy
        self._research_engine: Optional[ReasoningEngine] = None
        self._protocol_engine: Optional[ReasoningEngine] = None

    def get_research_engine(self) -> ReasoningEngine:
        """Return the research-tier reasoning engine (stays loaded)."""
        if self._research_engine is None:
            self._research_engine = ReasoningEngine(
                lazy=self._lazy,
                model_id=self._research_model_id,
            )
        return self._research_engine

    def get_protocol_engine(self) -> ReasoningEngine:
        """Return the protocol-tier reasoning engine (loaded on demand)."""
        if self._protocol_engine is None:
            self._protocol_engine = ReasoningEngine(
                lazy=self._lazy,
                model_id=self._protocol_model_id,
            )
        return self._protocol_engine

    def unload_protocol_model(self) -> None:
        """Free the protocol model memory (no-op for API backends)."""
        if self._protocol_engine is not None:
            self._protocol_engine._llm.unload()
            self._protocol_engine = None
