"""Adversarial protocol verification utilities.

Provides rule-based tools to challenge candidate interventions by generating
falsification queries, classifying contradictory evidence, and selecting
which intervention to challenge next.

Public API
----------
generate_adversarial_queries(drug_name, mechanism) -> list[str]
classify_adversarial_result(title, abstract, drug_name) -> str
select_challenge_target(intervention_scores, challenge_counts, max_challenges) -> Optional[str]
"""
from __future__ import annotations

import re
from typing import Optional


# ---------------------------------------------------------------------------
# generate_adversarial_queries
# ---------------------------------------------------------------------------

def generate_adversarial_queries(drug_name: str, mechanism: str) -> list[str]:
    """Generate three adversarial PubMed search queries for a given intervention.

    Parameters
    ----------
    drug_name:
        Name of the drug or intervention (e.g. ``"Riluzole"``).
    mechanism:
        Mechanism of action string (e.g. ``"glutamate excitotoxicity inhibition"``).

    Returns
    -------
    list[str]
        Three queries: [failure, harm, mechanism_dispute].
    """
    # Derive mechanism_key: first word of mechanism, fallback to drug_name
    stripped = mechanism.strip()
    if stripped:
        mechanism_key = stripped.split()[0]
    else:
        mechanism_key = drug_name

    failure_query = (
        f'"{drug_name}" ALS (failed OR negative OR ineffective OR discontinued)'
    )
    harm_query = (
        f'"{drug_name}" (neurotoxicity OR adverse OR "motor neuron" harm OR contraindicated)'
    )
    dispute_query = (
        f'"{mechanism_key}" ALS (disputed OR disproven OR "no effect" OR insufficient)'
    )

    return [failure_query, harm_query, dispute_query]


# ---------------------------------------------------------------------------
# classify_adversarial_result
# ---------------------------------------------------------------------------

# Compiled patterns for performance
_STRONG_CONTRADICTION_PATTERNS = [
    re.compile(r'failed\s+(to\s+meet|endpoint|phase|trial)', re.IGNORECASE),
    re.compile(r'no\s+(benefit|effect|improvement|efficacy)', re.IGNORECASE),
    re.compile(r'did\s+not\s+(meet|reach)', re.IGNORECASE),
    re.compile(r'negative\s+result', re.IGNORECASE),
    re.compile(r'\bdiscontinued\b', re.IGNORECASE),
    re.compile(r'\bineffective\b', re.IGNORECASE),
    re.compile(r'failed\s+to\s+reach', re.IGNORECASE),
]

_HARM_PATTERNS = [
    re.compile(r'\bneurotox', re.IGNORECASE),
    re.compile(r'severe\s+adverse', re.IGNORECASE),
    re.compile(r'\bworsened\b', re.IGNORECASE),
    re.compile(r'\bcontraindicated\b', re.IGNORECASE),
]

_ALS_PATTERNS = [
    re.compile(r'\bals\b', re.IGNORECASE),
    re.compile(r'amyotrophic\s+lateral\s+sclerosis', re.IGNORECASE),
    re.compile(r'motor\s+neuron\s+disease', re.IGNORECASE),
]


def classify_adversarial_result(title: str, abstract: str, drug_name: str) -> str:
    """Classify a PubMed result as to how it bears on an intervention claim.

    Rule-based; no LLM required.

    Parameters
    ----------
    title:
        Article title.
    abstract:
        Article abstract.
    drug_name:
        Name of the drug being challenged.

    Returns
    -------
    str
        One of ``"contradicts"``, ``"weakens"``, ``"irrelevant"``,
        ``"context_dependent"``.
    """
    text = f"{title} {abstract}"

    # Rule 1: drug name must appear in text
    if not re.search(re.escape(drug_name), text, re.IGNORECASE):
        return "irrelevant"

    # Determine ALS context
    in_als_context = any(p.search(text) for p in _ALS_PATTERNS)

    # Rule 2: check for strong contradiction signals
    has_contradiction = any(p.search(text) for p in _STRONG_CONTRADICTION_PATTERNS)

    if has_contradiction:
        return "contradicts" if in_als_context else "context_dependent"

    # Rule 3: check for harm signals
    has_harm = any(p.search(text) for p in _HARM_PATTERNS)
    if has_harm:
        return "weakens"

    # Rule 4: ALS context but no clear signal
    if in_als_context:
        return "context_dependent"

    return "irrelevant"


# ---------------------------------------------------------------------------
# select_challenge_target
# ---------------------------------------------------------------------------

def select_challenge_target(
    intervention_scores: dict[str, float],
    challenge_counts: dict[str, int],
    max_challenges: int = 3,
) -> Optional[str]:
    """Select the next intervention to adversarially challenge.

    Priority score = ``score * (1.0 / (1.0 + count))``.
    Interventions that have already reached ``max_challenges`` are excluded.

    Parameters
    ----------
    intervention_scores:
        Mapping of intervention name → evidence score (higher = better supported).
    challenge_counts:
        Mapping of intervention name → number of challenges already conducted.
    max_challenges:
        Maximum number of challenges allowed per intervention.

    Returns
    -------
    Optional[str]
        The selected intervention name, or ``None`` if all are fully challenged
        or the input is empty.
    """
    if not intervention_scores:
        return None

    best_name: Optional[str] = None
    best_priority: float = -1.0

    for name, score in intervention_scores.items():
        count = challenge_counts.get(name, 0)
        if count >= max_challenges:
            continue
        priority = score * (1.0 / (1.0 + count))
        if priority > best_priority:
            best_priority = priority
            best_name = name

    return best_name
