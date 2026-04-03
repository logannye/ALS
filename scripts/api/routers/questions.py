"""Question-answering endpoint — KG-grounded LLM responses for the family."""
from __future__ import annotations

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from db.pool import get_connection
from llm.inference import create_llm

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


class QuestionRequest(BaseModel):
    question: str
    max_evidence: int = 20


class QuestionResponse(BaseModel):
    answer: str
    cited_evidence: list[dict]
    raw_sources: int


_QA_TEMPLATE = """You are Erik, an autonomous ALS research engine built by Galen Health.
A family member of your patient (Erik Draper, 67, diagnosed with ALS March 2026) is asking you a question.
Answer clearly and compassionately. Ground every claim in the evidence provided.
If you don't know, say so — never fabricate.

EVIDENCE FROM THE KNOWLEDGE GRAPH:
{evidence_json}

QUESTION: {question}

Respond with a JSON object:
{{
  "answer": "your answer here, with inline [evi:ID] citations",
  "cited_evidence": ["evi:id1", "evi:id2"]
}}"""


@router.post("/questions")
def ask_question(req: QuestionRequest):
    """Answer a family member's question using KG context + Bedrock LLM."""
    # 1. Retrieve relevant evidence from the KG
    evidence_items = _search_evidence(req.question, limit=req.max_evidence)

    if not evidence_items:
        return QuestionResponse(
            answer="I don't have enough evidence in my knowledge base to answer this question yet. "
                   "The research loop is continuously gathering new evidence — please try again later.",
            cited_evidence=[],
            raw_sources=0,
        )

    # 2. Build prompt with evidence context
    evidence_json = json.dumps(evidence_items, indent=2)
    prompt = _QA_TEMPLATE.replace("{evidence_json}", evidence_json).replace(
        "{question}", req.question
    )

    # 3. Call LLM
    llm = create_llm()
    result = llm.generate_json(prompt, max_tokens=1000)

    if result is None:
        # Fallback: return raw text answer
        text = llm.generate(prompt, max_tokens=1000)
        return QuestionResponse(
            answer=text or "I was unable to generate a response. Please try again.",
            cited_evidence=[],
            raw_sources=len(evidence_items),
        )

    # 4. Validate citations
    valid_ids = {item["id"] for item in evidence_items}
    cited = [eid for eid in result.get("cited_evidence", []) if eid in valid_ids]

    # Build cited evidence details
    cited_details = [item for item in evidence_items if item["id"] in set(cited)]

    return QuestionResponse(
        answer=result.get("answer", ""),
        cited_evidence=cited_details,
        raw_sources=len(evidence_items),
    )


def _search_evidence(query: str, limit: int = 20) -> list[dict]:
    """Search evidence items by keyword match on claim text."""
    words = [w.strip() for w in query.lower().split() if len(w.strip()) > 3]
    if not words:
        words = [query.lower()]

    # Build an OR query across claim text
    conditions = []
    params: list = []
    for word in words[:5]:  # cap at 5 terms
        conditions.append("body->>'claim' ILIKE %s")
        params.append(f"%{word}%")

    where = " OR ".join(conditions)
    params.append(limit)

    try:
        with get_connection() as conn:
            rows = conn.execute(
                f"""SELECT id, body, confidence
                    FROM erik_core.objects
                    WHERE type = 'EvidenceItem' AND status = 'active'
                      AND ({where})
                    ORDER BY confidence DESC NULLS LAST
                    LIMIT %s""",
                params,
            ).fetchall()

        items = []
        for row in rows:
            obj_id, body, confidence = row
            if isinstance(body, str):
                body = json.loads(body)
            items.append({
                "id": obj_id,
                "claim": body.get("claim", ""),
                "direction": body.get("direction"),
                "strength": body.get("strength"),
                "confidence": confidence,
            })
        return items
    except Exception:
        logger.exception("Evidence search failed")
        return []
