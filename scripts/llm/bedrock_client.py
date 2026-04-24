"""Amazon Bedrock LLM client — drop-in replacement for MLX inference.

Uses the Converse API (model-agnostic) with exponential backoff retry.
Exposes the same ``generate()``, ``generate_json()``, ``unload()`` interface
as :class:`llm.inference.LLMInference` so the rest of the codebase
(ReasoningEngine, DualLLMManager) works unchanged.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from llm.inference import _extract_json

logger = logging.getLogger(__name__)

# Default model IDs
NOVA_MICRO = "amazon.nova-micro-v1:0"
NOVA_PRO = "amazon.nova-pro-v1:0"


class BedrockLLM:
    """Thin wrapper around the Amazon Bedrock Converse API.

    Parameters
    ----------
    model_id:
        Bedrock model identifier (e.g. ``"amazon.nova-micro-v1:0"``).
    region:
        AWS region for the Bedrock endpoint.
    max_tokens:
        Default maximum tokens to generate.
    temperature:
        Sampling temperature.
    max_retries:
        Number of retry attempts on throttling or transient errors.
    lazy:
        Accepted for interface compatibility; ignored (client is cheap to create).
    """

    def __init__(
        self,
        model_id: str = NOVA_MICRO,
        region: str = "us-east-1",
        max_tokens: int = 1000,
        temperature: float = 0.1,
        max_retries: int = 3,
        lazy: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._max_retries = max_retries
        self._client = boto3.client("bedrock-runtime", region_name=region)

    # ------------------------------------------------------------------
    # Public API (matches LLMInference interface)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> str:
        """Generate a text response for *prompt* via the Converse API.

        Returns the generated string (may be empty on failure after retries,
        empty on budget exceeded, or empty on unrecoverable API errors).
        """
        # Shared spend gate — prevents a runaway research loop from
        # burning an unbounded amount on Bedrock Nova.
        from llm.spend_gate import check_budget, record_call
        status = check_budget()
        if status.over_budget:
            logger.warning(
                "bedrock budget gate — skipping call (mtd=$%.2f, budget=$%.2f)",
                status.month_to_date_usd, status.budget_usd,
            )
            return ""

        tokens = max_tokens if max_tokens is not None else self.max_tokens

        messages = [{"role": "user", "content": [{"text": prompt}]}]
        inference_config = {
            "maxTokens": tokens,
            "temperature": self.temperature,
        }

        for attempt in range(self._max_retries):
            try:
                response = self._client.converse(
                    modelId=self.model_id,
                    messages=messages,
                    inferenceConfig=inference_config,
                )
                # Extract text from response
                output_message = response.get("output", {}).get("message", {})
                content_blocks = output_message.get("content", [])
                parts = [
                    block["text"]
                    for block in content_blocks
                    if "text" in block
                ]
                # Record spend — Bedrock's usage shape is on the top-level
                # "usage" key for the Converse API.
                usage = response.get("usage", {}) or {}
                record_call(
                    model=self.model_id,
                    phase='bedrock_generate',
                    input_tokens=int(usage.get("inputTokens", 0) or 0),
                    output_tokens=int(usage.get("outputTokens", 0) or 0),
                    prompt_cached=False,
                )
                return "".join(parts)

            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in (
                    "ThrottlingException",
                    "ServiceUnavailableException",
                    "ModelTimeoutException",
                ):
                    wait = 2 ** attempt
                    logger.warning(
                        "Bedrock %s (attempt %d/%d), retrying in %ds: %s",
                        error_code,
                        attempt + 1,
                        self._max_retries,
                        wait,
                        e,
                    )
                    time.sleep(wait)
                else:
                    logger.error("Bedrock API error: %s", e)
                    raise
            except Exception:
                logger.exception("Unexpected error calling Bedrock")
                if attempt < self._max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise

        logger.error("Bedrock: all %d retries exhausted", self._max_retries)
        return ""

    def generate_json(
        self, prompt: str, max_tokens: Optional[int] = None
    ) -> Optional[dict]:
        """Generate a response and extract the first JSON object from it.

        Returns a ``dict`` on success, ``None`` if no valid JSON was found.
        """
        text = self.generate(prompt, max_tokens=max_tokens)
        return _extract_json(text)

    def unload(self) -> None:
        """No-op — API-based client has no local model to free."""
        pass
