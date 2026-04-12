"""Erik API — FastAPI application serving the family dashboard + research loop.

On Railway, this is the single entry point. The research loop runs as an
asyncio background task alongside the REST API in the same process.

Usage (local):
    PYTHONPATH=scripts uvicorn api.main:app --reload --port 8000

Usage (Railway):
    Set in Dockerfile CMD: uvicorn api.main:app --host 0.0.0.0 --port $PORT
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.auth import (
    _ensure_sessions_table,
    get_session_token,
    redeem_invite_code,
    validate_session,
)
from db.pool import close_pool

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Research loop background task
# ---------------------------------------------------------------------------

_loop_task: asyncio.Task | None = None


async def _run_research_loop():
    """Run the research loop in a thread (it's synchronous/blocking)."""
    loop = asyncio.get_event_loop()
    try:
        # Import here to avoid circular imports and keep startup fast
        from run_loop import main as loop_main

        # Run synchronous loop in executor so it doesn't block the event loop
        await loop.run_in_executor(None, loop_main)
    except asyncio.CancelledError:
        logger.info("Research loop cancelled")
    except Exception:
        logger.exception("Research loop crashed")


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the research loop on startup, clean up on shutdown."""
    global _loop_task

    # Run database migrations (ensures TCG tables exist)
    try:
        from db.migrate import run_migrations
        run_migrations()
    except Exception:
        logger.exception("Migration failed — tables may be missing")

    # Ensure sessions table exists
    try:
        _ensure_sessions_table()
    except Exception:
        logger.exception("Failed to create sessions table")

    # Start research loop if enabled
    if os.environ.get("ERIK_RESEARCH_LOOP", "true").lower() == "true":
        logger.info("Starting research loop background task")
        _loop_task = asyncio.create_task(_run_research_loop())
    else:
        logger.info("Research loop disabled (ERIK_RESEARCH_LOOP != true)")

    yield

    # Shutdown
    if _loop_task and not _loop_task.done():
        _loop_task.cancel()
        try:
            await asyncio.wait_for(_loop_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

    close_pool()
    logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Erik ALS Research Engine",
    description="Family dashboard API for the Erik autonomous ALS research system",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
allowed_origins = os.environ.get(
    "CORS_ALLOWED_ORIGINS", "http://localhost:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Auth middleware
# ---------------------------------------------------------------------------

# Routes that don't require authentication
_PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/api/auth/redeem"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    """Check session token on all /api/* routes except public paths."""
    path = request.url.path

    if path in _PUBLIC_PATHS or not path.startswith("/api"):
        return await call_next(request)

    token = get_session_token(request)
    try:
        family_member = validate_session(token)
        request.state.family_member = family_member
    except Exception as exc:
        return JSONResponse(
            status_code=401,
            content={"detail": str(exc)},
        )

    return await call_next(request)


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

from pydantic import BaseModel, field_validator


class RedeemRequest(BaseModel):
    code: str
    name: str = "family"

    @field_validator("code", "name", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


@app.post("/api/auth/redeem")
def auth_redeem(req: RedeemRequest):
    """Redeem an invite code and get a session token."""
    token = redeem_invite_code(req.code, req.name)
    response = JSONResponse(content={"status": "ok", "family_member": req.name})
    response.set_cookie(
        key="erik_session",
        value=token,
        httponly=True,
        secure=True,
        samesite="none",
        max_age=90 * 86400,  # 90 days
    )
    return response


# ---------------------------------------------------------------------------
# Mount routers
# ---------------------------------------------------------------------------

from api.routers.health import router as health_router
from api.routers.state import router as state_router
from api.routers.evidence import router as evidence_router
from api.routers.protocol import router as protocol_router
from api.routers.questions import router as questions_router
from api.routers.upload import router as upload_router
from api.routers.trials import router as trials_router
from api.routers.activity import router as activity_router
from api.routers.report import router as report_router
from api.routers.summary import router as summary_router
from api.routers.genetics import router as genetics_router
from api.routers.discoveries import router as discoveries_router
from api.routers.document_upload import router as document_upload_router
try:
    from api.routers.graph import router as graph_router
    from api.routers.hypotheses import router as hypotheses_router
    from api.routers.progress import router as progress_router
    _tcg_routers_available = True
except Exception as _tcg_err:
    _tcg_routers_available = False
    logger.error("TCG routers failed to load: %s", _tcg_err)

app.include_router(health_router)
app.include_router(state_router)
app.include_router(evidence_router)
app.include_router(protocol_router)
app.include_router(questions_router)
app.include_router(upload_router)
app.include_router(trials_router)
app.include_router(activity_router)
app.include_router(report_router)
app.include_router(summary_router)
app.include_router(genetics_router)
app.include_router(discoveries_router)
app.include_router(document_upload_router)
if _tcg_routers_available:
    app.include_router(graph_router)
    app.include_router(hypotheses_router)
    app.include_router(progress_router)
