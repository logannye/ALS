"""Tests for LLMInference wrapper and _extract_json helper.

Unit tests do NOT require a local LLM model.
Real-generation tests are marked @pytest.mark.llm.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

def test_extract_json_plain_dict():
    from llm.inference import _extract_json
    result = _extract_json('text before {"key": "value"} text after')
    assert result == {"key": "value"}


def test_extract_json_markdown_fence():
    from llm.inference import _extract_json
    text = '```json\n{"answer": 42}\n```'
    result = _extract_json(text)
    assert result == {"answer": 42}


def test_extract_json_markdown_fence_no_lang():
    from llm.inference import _extract_json
    text = "```\n{\"x\": true}\n```"
    result = _extract_json(text)
    assert result == {"x": True}


def test_extract_json_nested_object():
    from llm.inference import _extract_json
    text = 'result: {"a": {"b": 1}, "c": [1, 2, 3]}'
    result = _extract_json(text)
    assert result == {"a": {"b": 1}, "c": [1, 2, 3]}


def test_extract_json_invalid_returns_none():
    from llm.inference import _extract_json
    assert _extract_json("no json here") is None
    assert _extract_json("") is None
    assert _extract_json("{ broken json }") is None


def test_extract_json_none_input():
    from llm.inference import _extract_json
    assert _extract_json(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LLMInference instantiation (no model load)
# ---------------------------------------------------------------------------

def test_llm_inference_lazy_no_load():
    """lazy=True must not attempt to load the model at construction time."""
    from llm.inference import LLMInference
    llm = LLMInference(lazy=True)
    # Model and tokenizer should not be loaded yet
    assert llm._model is None
    assert llm._tokenizer is None


def test_llm_inference_default_config():
    from llm.inference import LLMInference
    llm = LLMInference(lazy=True)
    assert llm.temperature == 0.1
    assert llm.max_tokens == 1000


def test_llm_inference_custom_config():
    from llm.inference import LLMInference
    llm = LLMInference(max_tokens=500, temperature=0.7, lazy=True)
    assert llm.max_tokens == 500
    assert llm.temperature == 0.7


def test_llm_inference_default_model_path():
    from llm.inference import LLMInference
    llm = LLMInference(lazy=True)
    # Primary default path
    assert "Qwen3.5-35B" in llm.model_path or "Qwen3.5-9B" in llm.model_path


def test_llm_inference_custom_model_path():
    from llm.inference import LLMInference
    llm = LLMInference(model_path="/tmp/fake-model", lazy=True)
    assert llm.model_path == "/tmp/fake-model"


# ---------------------------------------------------------------------------
# Real LLM tests — require local model, skip in CI
# ---------------------------------------------------------------------------

@pytest.mark.llm
def test_llm_generate_returns_string():
    from llm.inference import LLMInference
    llm = LLMInference()
    result = llm.generate("Say hello in one word.")
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.llm
def test_llm_generate_json_returns_dict():
    from llm.inference import LLMInference
    llm = LLMInference()
    result = llm.generate_json('Return JSON: {"status": "ok"}')
    assert isinstance(result, dict)
    assert result.get("status") == "ok"
