"""Evidence-grounded reasoning engine for the Erik ALS causal research pipeline.

Every factual claim emitted by the LLM must be backed by a cited evidence item.
This module provides:

- ``validate_citations``   — strip hallucinated IDs from LLM output
- ``strip_uncited_claims`` — remove unsupported sentences from text fields
- ``ReasoningEngine``      — orchestrates LLM calls + post-hoc citation audit
"""
from __future__ import annotations

import json
import re
from typing import Optional

from llm.inference import LLMInference
from world_model.prompts.templates import SYSTEM_PROMPT, VERIFICATION_TEMPLATE


# ---------------------------------------------------------------------------
# Free helpers
# ---------------------------------------------------------------------------

def validate_citations(
    output: dict, valid_ids: set[str]
) -> tuple[dict, list[str]]:
    """Check ``output["cited_evidence"]`` against *valid_ids*.

    Removes any IDs that do not appear in *valid_ids* (hallucinated references)
    and returns the cleaned output alongside a list of warning strings.

    Parameters
    ----------
    output:
        LLM output dict — must contain a ``"cited_evidence"`` key (list of str).
        If the key is absent an empty list is used instead.
    valid_ids:
        Set of evidence/intervention IDs that were actually provided to the LLM.

    Returns
    -------
    tuple[dict, list[str]]
        ``(cleaned_output, warnings)`` where *warnings* is empty when all IDs
        were valid.
    """
    cited: list[str] = output.get("cited_evidence") or []
    cleaned: list[str] = []
    warnings: list[str] = []

    for eid in cited:
        if eid in valid_ids:
            cleaned.append(eid)
        else:
            warnings.append(
                f"Hallucinated citation removed: {eid!r} not in supplied evidence"
            )

    result = dict(output)
    result["cited_evidence"] = cleaned
    return result, warnings


def strip_uncited_claims(text: str, valid_ids: set[str]) -> str:
    """Remove sentences that lack a valid citation reference.

    Splits *text* on sentence-boundary punctuation (``.!?``) and keeps a
    sentence only if:

    - It contains at least one ``[evi:*]`` or ``[int:*]`` reference that is
      present in *valid_ids*, **or**
    - It is a short structural sentence (fewer than 8 words), which is kept
      unconditionally to preserve transition phrases and headings.

    Parameters
    ----------
    text:
        Free-text output from the LLM (e.g. ``reasoning`` field).
    valid_ids:
        Set of evidence/intervention IDs that were actually provided.

    Returns
    -------
    str
        Filtered text with unsupported long sentences removed.
    """
    if not text:
        return text

    # Pattern matching inline citations like [evi:abc123] or [int:xyz789]
    _citation_pattern = re.compile(r"\[(evi:[^\]]+|int:[^\]]+)\]")

    # Split on sentence-ending punctuation, keeping the delimiter
    raw_sentences = re.split(r"(?<=[.!?])\s+", text.strip())

    kept: list[str] = []
    for sentence in raw_sentences:
        if not sentence.strip():
            continue

        word_count = len(sentence.split())

        # Keep short structural sentences unconditionally
        if word_count < 8:
            kept.append(sentence)
            continue

        # Check whether the sentence cites at least one valid ID
        refs_in_sentence = _citation_pattern.findall(sentence)
        if any(ref in valid_ids for ref in refs_in_sentence):
            kept.append(sentence)
        # else: drop the unsupported long sentence

    return " ".join(kept)


# ---------------------------------------------------------------------------
# ReasoningEngine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Orchestrates evidence-grounded LLM reasoning calls.

    Parameters
    ----------
    lazy:
        When ``True``, defers model loading until the first call.  Useful for
        tests that do not need a local model.
    model_path:
        Override the default model path passed to :class:`~llm.inference.LLMInference`.
    """

    def __init__(
        self,
        lazy: bool = False,
        model_path: Optional[str] = None,
    ) -> None:
        self._llm = LLMInference(model_path=model_path, lazy=lazy)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reason(
        self,
        template: str,
        evidence_items: list[dict],
        extra_context: Optional[dict] = None,
        max_tokens: int = 1500,
        verify_critical: bool = False,
    ) -> Optional[dict]:
        """Run a single reasoning pass and return a validated output dict.

        Steps
        -----
        1. Render a prompt from *template* + *evidence_items*.
        2. Prepend ``SYSTEM_PROMPT``.
        3. Call the LLM and parse JSON from the response.
        4. Validate citations against the supplied evidence item IDs.
        5. Strip uncited claims from free-text fields.
        6. Optionally verify the ``mechanism_argument`` field with a second call.

        Returns
        -------
        Optional[dict]
            Validated dict or ``None`` if the LLM produced no parseable JSON.
        """
        # Build the full prompt
        prompt = self._build_prompt(template, evidence_items, extra_context)

        # Call LLM
        output = self._llm.generate_json(prompt, max_tokens=max_tokens)
        if output is None:
            return None

        # Collect valid IDs
        valid_ids: set[str] = {item["id"] for item in evidence_items if "id" in item}

        # Validate citations
        output, _warnings = validate_citations(output, valid_ids)

        # Strip uncited long claims from text fields
        for field in ("reasoning", "mechanism_argument"):
            if isinstance(output.get(field), str):
                output[field] = strip_uncited_claims(output[field], valid_ids)

        # Optionally run a verification pass on the mechanism_argument
        if verify_critical and output.get("mechanism_argument"):
            verification = self._verify_claim(
                output["mechanism_argument"],
                output.get("cited_evidence", []),
            )
            if verification is not None:
                output["_verification"] = verification

        return output

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        template: str,
        evidence_items: list[dict],
        extra_context: Optional[dict] = None,
    ) -> str:
        """Render *template* by substituting ``{evidence_items_json}`` and any
        keys from *extra_context*.

        The ``SYSTEM_PROMPT`` is prepended so the model always receives the
        strict-grounding rules first.
        """
        evidence_json = json.dumps(evidence_items, indent=2)

        # Build substitution mapping
        subs: dict[str, str] = {"evidence_items_json": evidence_json}
        if extra_context:
            subs.update(extra_context)

        # Replace all known placeholders; leave unknown ones untouched
        rendered = template
        for key, value in subs.items():
            rendered = rendered.replace(f"{{{key}}}", str(value))

        return SYSTEM_PROMPT + "\n\n" + rendered

    def _verify_claim(
        self,
        claim_text: str,
        cited_evidence: list[str],
    ) -> Optional[dict]:
        """Run a second LLM pass to verify a single mechanistic claim.

        Uses ``VERIFICATION_TEMPLATE``.  Returns the parsed JSON dict or
        ``None`` if the LLM produces no valid JSON.
        """
        # Build a minimal evidence context from the cited IDs
        evidence_context = json.dumps(
            [{"id": eid} for eid in cited_evidence], indent=2
        )

        prompt = SYSTEM_PROMPT + "\n\n" + VERIFICATION_TEMPLATE.replace(
            "{claim_text}", claim_text
        ).replace("{evidence_items_json}", evidence_context)

        return self._llm.generate_json(prompt, max_tokens=500)
