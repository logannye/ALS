"""Thin wrapper for local MLX LLM inference.

Supports lazy loading so tests can instantiate LLMInference without a model
present on disk.
"""
from __future__ import annotations

import json
import re
from typing import Optional


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

_PRIMARY_MODEL = "/Volumes/Databank/models/mlx/Qwen3.5-35B-A3B-mlx-lm-mxfp4"
_FALLBACK_MODEL = "/Volumes/Databank/models/mlx/Qwen3.5-9B-mlx-lm-4bit"


def _extract_json(text: Optional[str]) -> Optional[dict]:
    """Extract the first JSON object from *text*.

    Handles:
    - Markdown fences: ```json ... ``` or ``` ... ```
    - Raw JSON embedded in surrounding prose

    Uses brace-depth counting so nested objects are captured correctly.
    Returns a ``dict`` on success or ``None`` if no valid JSON object is found.
    """
    if not text:
        return None

    # Strip markdown fences first so we can try the inner content directly.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    candidates = []
    if fence_match:
        candidates.append(fence_match.group(1).strip())
    # Always also try the raw text in case the fence strip wasn't needed.
    candidates.append(text)

    for candidate in candidates:
        # Find the first '{' and walk brace depth to locate the closing '}'.
        start = candidate.find("{")
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(candidate[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    json_str = candidate[start : i + 1]
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError:
                        break  # try next candidate

    return None


# ---------------------------------------------------------------------------
# LLMInference class
# ---------------------------------------------------------------------------


class LLMInference:
    """Thin wrapper around ``mlx_lm`` for local LLM inference.

    Parameters
    ----------
    model_path:
        Path to the MLX model directory.  Defaults to the primary 35B model;
        falls back to the 9B model if the primary path is unavailable.
    max_tokens:
        Default maximum tokens to generate.
    temperature:
        Sampling temperature.
    lazy:
        If ``True``, defer model loading until the first call to
        :meth:`generate`.  Useful for unit tests that don't have a model.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.1,
        lazy: bool = False,
    ) -> None:
        self.model_path: str = model_path or _PRIMARY_MODEL
        self.max_tokens: int = max_tokens
        self.temperature: float = temperature

        # Internal state — None until loaded.
        self._model = None
        self._tokenizer = None

        if not lazy:
            self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Load the MLX model and tokenizer.

        Tries :attr:`model_path` first; falls back to :data:`_FALLBACK_MODEL`
        if the primary path does not exist on disk.
        """
        import os
        from mlx_lm import load  # type: ignore[import-untyped]

        path = self.model_path
        if not os.path.isdir(path):
            path = _FALLBACK_MODEL
            self.model_path = path

        self._model, self._tokenizer = load(path)

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._load_model()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a text response for *prompt*.

        Returns the generated string (may be empty if the model produces
        nothing within *max_tokens*).
        """
        from mlx_lm import generate  # type: ignore[import-untyped]

        self._ensure_loaded()
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        result = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=tokens,
            temp=self.temperature,
            verbose=False,
        )
        # mlx_lm.generate returns the generated text as a string.
        return result if isinstance(result, str) else str(result)

    def generate_json(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Optional[dict]:
        """Generate a response and extract the first JSON object from it.

        Returns a ``dict`` on success, ``None`` if no valid JSON was found.
        """
        text = self.generate(prompt, max_tokens=max_tokens)
        return _extract_json(text)
