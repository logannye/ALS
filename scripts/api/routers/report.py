"""Physician report endpoint — structured clinical output for Erik's care team."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api")


@router.get("/report")
def get_report():
    """Generate a physician-ready clinical report."""
    from world_model.physician_reporter import generate_physician_report

    report = generate_physician_report(use_llm=True)
    return report
