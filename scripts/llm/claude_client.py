"""Claude API client for deep reasoning phases — rate limiting, spend tracking, fallback."""
from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

from config.loader import ConfigLoader
from db.pool import get_connection


class LLMSpendTracker:
    """Track and enforce LLM API spend in PostgreSQL."""

    def log(
        self, model: str, phase: str, input_tokens: int, output_tokens: int,
        cost_usd: float, prompt_cached: bool = False,
    ) -> None:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO erik_ops.llm_spend
                            (model, phase, input_tokens, output_tokens, cost_usd, prompt_cached)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (model, phase, input_tokens, output_tokens, cost_usd, prompt_cached))
                conn.commit()
        except Exception:
            pass  # Spend logging should never block research

    def monthly_spend_usd(self) -> float:
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT coalesce(sum(cost_usd), 0)
                        FROM erik_ops.llm_spend
                        WHERE created_at > date_trunc('month', now())
                    """)
                    return float(cur.fetchone()[0])
        except Exception:
            return 0.0


class ClaudeClient:
    """Claude API client with rate limiting, spend tracking, and structured prompts."""

    # Approximate costs per 1M tokens (for spend tracking)
    _COST_PER_M_INPUT = {"claude-opus-4-6": 15.0, "claude-sonnet-4-6": 3.0}
    _COST_PER_M_OUTPUT = {"claude-opus-4-6": 75.0, "claude-sonnet-4-6": 15.0}

    def __init__(
        self, api_key: str, reasoning_model: str = "claude-opus-4-6",
        evaluation_model: str = "claude-sonnet-4-6",
        max_opus_per_hour: int = 30, max_sonnet_per_hour: int = 60,
        monthly_budget_usd: float = 100.0,
    ) -> None:
        self._api_key = api_key
        self._reasoning_model = reasoning_model
        self._evaluation_model = evaluation_model
        self._max_opus_per_hour = max_opus_per_hour
        self._max_sonnet_per_hour = max_sonnet_per_hour
        self._monthly_budget_usd = monthly_budget_usd
        self._spend_tracker = LLMSpendTracker()
        self._opus_calls: deque[float] = deque()
        self._sonnet_calls: deque[float] = deque()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            if anthropic is None:
                raise ImportError("anthropic package not installed")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def _check_rate_limit(self, model: str) -> bool:
        now = time.time()
        hour_ago = now - 3600
        if "opus" in model:
            self._opus_calls = deque(t for t in self._opus_calls if t > hour_ago)
            return len(self._opus_calls) < self._max_opus_per_hour
        else:
            self._sonnet_calls = deque(t for t in self._sonnet_calls if t > hour_ago)
            return len(self._sonnet_calls) < self._max_sonnet_per_hour

    def _record_call(self, model: str) -> None:
        if "opus" in model:
            self._opus_calls.append(time.time())
        else:
            self._sonnet_calls.append(time.time())

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        in_cost = self._COST_PER_M_INPUT.get(model, 3.0) * input_tokens / 1_000_000
        out_cost = self._COST_PER_M_OUTPUT.get(model, 15.0) * output_tokens / 1_000_000
        return in_cost + out_cost

    def _call_api(
        self, model: str, system: str, user_prompt: str, phase: str,
        max_tokens: int = 4096,
    ) -> Optional[dict]:
        # Budget check
        if self._spend_tracker.monthly_spend_usd() >= self._monthly_budget_usd:
            print(f"[CLAUDE] Monthly budget ${self._monthly_budget_usd} exceeded — skipping call")
            return {"budget_exceeded": True}

        # Rate limit check
        if not self._check_rate_limit(model):
            print(f"[CLAUDE] Rate limit hit for {model} — skipping call")
            return None

        try:
            client = self._get_client()
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user_prompt}],
            )
            self._record_call(model)

            # Parse response
            text = response.content[0].text if response.content else ""
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self._estimate_cost(model, input_tokens, output_tokens)
            cached = getattr(response.usage, "cache_read_input_tokens", 0) > 0

            self._spend_tracker.log(
                model=model, phase=phase, input_tokens=input_tokens,
                output_tokens=output_tokens, cost_usd=cost, prompt_cached=cached,
            )

            # Extract JSON from response
            return _extract_json(text)
        except Exception as e:
            print(f"[CLAUDE] API error: {e}")
            return None

    def reason_about_edge(
        self, edge_context: str, supporting_evidence: list[str],
        contradicting_evidence: list[str],
    ) -> Optional[dict]:
        system = (
            "You are an ALS research scientist analyzing mechanistic evidence. "
            "Return JSON with keys: confidence_assessment (0-1), mechanism (string), "
            "open_questions (list[string]), confounders (list[string])."
        )
        user = (
            f"Analyze this mechanistic edge in ALS biology:\n\n"
            f"Edge: {edge_context}\n\n"
            f"Supporting evidence:\n" + "\n".join(f"- {e}" for e in supporting_evidence) + "\n\n"
            f"Contradicting evidence:\n" + "\n".join(f"- {e}" for e in contradicting_evidence) + "\n\n"
            f"Assess: (1) confidence in this causal link, (2) most likely mechanism, "
            f"(3) what would resolve remaining uncertainty, (4) confounders."
        )
        return self._call_api(self._evaluation_model, system, user, phase="reasoning")

    def counterfactual_analysis(
        self, hypothesis: str, causal_path: list[str], tcg_context: str,
    ) -> Optional[dict]:
        system = (
            "You are an ALS therapeutic strategist performing counterfactual analysis. "
            "Return JSON with keys: downstream_effects (list[dict]), off_target_risks (list[string]), "
            "confidence (float), new_edges (list[dict with source, target, edge_type, rationale])."
        )
        user = (
            f"Hypothesis: {hypothesis}\n\n"
            f"Causal path through the therapeutic graph:\n"
            + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(causal_path)) + "\n\n"
            f"Graph context:\n{tcg_context}\n\n"
            f"Trace every downstream consequence of this intervention. "
            f"Assess off-target effects and unintended pathway interactions."
        )
        return self._call_api(self._reasoning_model, system, user, phase="reasoning")

    def cross_pathway_synthesis(
        self, cluster_a: str, cluster_a_evidence: list[str],
        cluster_b: str, cluster_b_evidence: list[str],
    ) -> Optional[dict]:
        system = (
            "You are an ALS systems biologist looking for undiscovered pathway interactions. "
            "Return JSON with key: proposed_edges (list[dict with source, target, edge_type, "
            "confidence (float), rationale (string)])."
        )
        user = (
            f"Cluster A ({cluster_a}):\n" + "\n".join(f"- {e}" for e in cluster_a_evidence) + "\n\n"
            f"Cluster B ({cluster_b}):\n" + "\n".join(f"- {e}" for e in cluster_b_evidence) + "\n\n"
            f"Are there undiscovered mechanistic links between these clusters? "
            f"Could intervening in one affect the other?"
        )
        return self._call_api(self._reasoning_model, system, user, phase="reasoning")

    def evaluate_compound(
        self, compound: str, target_edges: list[str], current_protocol: str,
    ) -> Optional[dict]:
        system = (
            "You are a medicinal chemist evaluating drug candidates for ALS. "
            "Return JSON with keys: suitability_score (0-1), strengths (list[string]), "
            "risks (list[string]), drug_interactions (list[string]), recommendation (string)."
        )
        user = (
            f"Compound: {compound}\n\n"
            f"Target edges in therapeutic graph:\n"
            + "\n".join(f"- {e}" for e in target_edges) + "\n\n"
            f"Current protocol:\n{current_protocol}\n\n"
            f"Evaluate this compound for inclusion in the treatment protocol."
        )
        return self._call_api(self._evaluation_model, system, user, phase="compound")


def _extract_json(text: str) -> Optional[dict]:
    """Extract first JSON object from text (handles markdown fences)."""
    import re
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Try markdown fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    # Try first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None
