"""Health check endpoint — no auth required."""
from __future__ import annotations

import time

from fastapi import APIRouter

from db.pool import get_connection

router = APIRouter()
_start_time = time.time()


@router.get("/health")
def health():
    """Basic health check: process uptime + DB connectivity."""
    db_ok = False
    try:
        with get_connection() as conn:
            conn.execute("SELECT 1")
            db_ok = True
    except Exception:
        pass

    return {
        "status": "ok" if db_ok else "degraded",
        "uptime_s": round(time.time() - _start_time, 1),
        "database": "connected" if db_ok else "unreachable",
    }
