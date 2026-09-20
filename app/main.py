"""FastAPI wrapper around the tools and agent construction."""
from __future__ import annotations

import logging
import os
import time
import uuid as _uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select

from .agents import tools
from .agents.orchestrator import build_orchestrator
from .config import settings
from .deps import Principal, Recording, Session, require_role
from .guard import inspect_input
from .models import (
    TENANT_SCOPED_TABLES,
    AgentTask,
    AuditEvent,
    Conversation,
    Message,
    Role,
    ToolInvocation,
)
from .routers import auth as auth_router
from .schemas import AuditOut, ConversationOut, MessageOut, ToolInvocationOut
from .tasks import TASK_NAME, broker_mode, celery_app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    problems = settings.validate_for_environment()
    if problems:
        # Refuse rather than warn. A warning about an unauthenticated production
        # deployment gets read after the incident, not before it.
        raise RuntimeError(
            "refusing to start with unsafe configuration: " + "; ".join(problems)
        )
    if not settings.auth_enabled:
        logger.warning(
            "running in OPEN MODE: no DATABASE_URL, so every endpoint is "
            "unauthenticated and nothing is recorded"
        )
    yield
    if settings.auth_enabled:
        from .db import dispose_engine

        await dispose_engine()


app = FastAPI(
    title="NetOps Agent Platform",
    description="Multi-agent network operations assistant: topology graph, "
                "anomaly scoring, and incident-relevance ranking. Multi-tenant, "
                "with isolation enforced by PostgreSQL row-level security.",
    version="0.2.0",
    lifespan=lifespan,
)

app.include_router(auth_router.router)

# GraphQL reuses the REST auth and session dependencies.
from .graphql_api import router as graphql_router  # noqa: E402

app.include_router(graphql_router, prefix="")

_orchestrator = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = build_orchestrator()
    return _orchestrator


# ── schemas ──────────────────────────────────────────────────────

class ImpactRequest(BaseModel):
    tower_id: str
    max_hops: int = Field(default=3, ge=1, le=10)


class HealthRequest(BaseModel):
    latency_ms: float = Field(ge=0)
    packet_loss_pct: float = Field(ge=0, le=100)
    retransmit_rate: float = Field(ge=0)
    connected_devices: float = Field(ge=0)


class RelevanceRequest(BaseModel):
    event_descriptions: list[str]
    focus_index: int = Field(ge=0)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    # Opt-in so existing callers keep getting a synchronous reply.
    background: bool = False


class TaskOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: _uuid.UUID
    status: str
    prompt: str
    reply: str | None = None
    error: str | None = None
    conversation_id: _uuid.UUID | None = None
    duration_ms: float
    created_at: datetime
    finished_at: datetime | None = None


# ── health ───────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Report what actually resolved, not what was configured."""
    body: dict[str, Any] = {
        "status": "ok",
        "environment": settings.environment,
        "model_configured": bool(os.environ.get("GOOGLE_API_KEY")),
        "graph_backend": tools._get_graph().name if tools._graph else "not yet loaded",
        "auth": "enabled" if settings.auth_enabled else "OPEN MODE (unauthenticated)",
        "background_execution": (
            broker_mode() if settings.background_enabled
            else "inline (needs both a broker and a database)"
        ),
    }

    if not settings.auth_enabled:
        body["row_level_security"] = "not applicable (no database)"
        return body

    from sqlalchemy import text

    from .db import get_sessionmaker

    try:
        async with get_sessionmaker()() as session:
            await session.execute(text("SELECT 1"))
            body["database"] = "ok"
            enforced = await session.scalar(
                text(
                    "SELECT count(*) FROM pg_class "
                    "WHERE relnamespace = 'public'::regnamespace AND relkind = 'r' "
                    "AND relrowsecurity AND relforcerowsecurity "
                    "AND relname = ANY(:names)"
                ),
                {"names": list(TENANT_SCOPED_TABLES)},
            )
            body["row_level_security"] = (
                "enforced" if enforced == len(TENANT_SCOPED_TABLES)
                else f"INCOMPLETE ({enforced}/{len(TENANT_SCOPED_TABLES)} tables)"
            )
            if enforced != len(TENANT_SCOPED_TABLES):
                body["status"] = "degraded"
    except Exception:  # noqa: BLE001 - health reports degradation, never raises it
        body["database"] = "unavailable"
        body["row_level_security"] = "unknown"
        body["status"] = "degraded"

    return body


# ── direct tool endpoints (no LLM required) ──────────────────────

def _run_tool(recorder, name: str, arguments: dict, fn, *args, **kwargs) -> dict:
    """Call a tool, time it, and record the call."""
    started = time.perf_counter()
    result = fn(*args, **kwargs)
    recorder.tool_call(name, arguments, result, (time.perf_counter() - started) * 1000)
    return result


@app.post("/tools/impact")
def impact(
    req: ImpactRequest,
    recorder: Recording,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
) -> dict:
    try:
        return _run_tool(
            recorder, "get_downstream_impact", req.model_dump(),
            tools.get_downstream_impact, req.tower_id, max_hops=req.max_hops,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/tools/health-score")
def health_score(
    req: HealthRequest,
    recorder: Recording,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
) -> dict:
    return _run_tool(
        recorder, "score_tower_health", req.model_dump(), tools.score_tower_health,
        req.latency_ms, req.packet_loss_pct, req.retransmit_rate, req.connected_devices,
    )


@app.post("/tools/relevance")
def relevance(
    req: RelevanceRequest,
    recorder: Recording,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
) -> dict:
    try:
        return _run_tool(
            recorder, "rank_relevant_events", req.model_dump(),
            tools.rank_relevant_events, req.event_descriptions, req.focus_index,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ── live agent endpoint (requires a model key) ───────────────────

@app.post("/chat")
async def chat(
    req: ChatRequest,
    session: Session,
    recorder: Recording,
    principal: Annotated[Principal, Depends(require_role(Role.OPERATOR))],
) -> dict:
    """A live agent conversation, gated at operator level."""
    verdict = inspect_input(req.message)
    if not verdict.allowed:
        # Refused before the model sees it, and recorded either way. A guard
        # nobody can audit is a guard nobody can tell has stopped working.
        recorder.audit("chat", "blocked", verdict.summary)
        # Committed before raising. The session dependency rolls back on any
        # exception, so without this the refusal is discarded with the request.
        await recorder.commit_now()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The message was refused by the prompt-injection guard.",
        )

    if not os.environ.get("GOOGLE_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail=(
                "No model credentials configured. Set GOOGLE_API_KEY to run a live "
                "agent conversation, or use the /tools/* endpoints, which need no "
                "credentials."
            ),
        )

    if req.background:
        if not settings.background_enabled:
            # Don't fall back to inline: the response shape would differ.
            raise HTTPException(
                status_code=503,
                detail=(
                    "Background execution is not configured. It needs both "
                    "BROKER_URL and DATABASE_URL; /health reports which are present."
                ),
            )

        # Guard and role check have already run, so nothing invalid is queued.
        task = AgentTask(
            tenant_id=principal.tenant_id, actor_id=principal.user_id,
            prompt=req.message, status="pending",
        )
        session.add(task)
        await session.flush()
        task_id = task.id

        recorder.audit("chat.enqueue", "allowed", {"task_id": str(task_id)})
        # Commit before dispatch so the worker can see the row.
        await recorder.commit_now()

        celery_app.send_task(
            TASK_NAME,
            # Identifiers only. The worker reads the prompt from the row under its
            # own tenant scope, so a misrouted message has nothing it can read.
            args=[str(task_id), str(principal.tenant_id),
                  str(principal.user_id) if principal.user_id else None],
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "task_id": str(task_id),
                "status": "pending",
                "poll": f"/tasks/{task_id}",
            },
        )

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    if session is not None:
        # End the transaction so its pooled connection goes back while the
        # model runs, which can take many seconds. The writes below start a
        # new transaction, and the session re-applies the tenant scope to it.
        await session.commit()

    actor = str(principal.user_id or "open-mode")
    runner = InMemoryRunner(agent=_get_orchestrator(), app_name="netops")
    adk_session = await runner.session_service.create_session(
        app_name="netops", user_id=actor
    )

    reply_text = ""
    async for event in runner.run_async(
        user_id=actor,
        session_id=adk_session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=req.message)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.text:
                    reply_text += part.text

    conversation_id = None
    if session is not None and principal.tenant_id is not None:
        conversation = Conversation(
            tenant_id=principal.tenant_id, created_by=principal.user_id,
            title=req.message[:120],
        )
        session.add(conversation)
        await session.flush()
        session.add_all([
            Message(tenant_id=principal.tenant_id, conversation_id=conversation.id,
                    role="user", content=req.message),
            Message(tenant_id=principal.tenant_id, conversation_id=conversation.id,
                    role="assistant", content=reply_text),
        ])
        conversation_id = conversation.id

    recorder.audit("chat", "allowed", {"signals": verdict.signals})
    return {"reply": reply_text, "conversation_id": conversation_id}


# ── tenant-scoped history ────────────────────────────────────────
# No tenant_id filters here: the session is scoped and RLS does the filtering.

@app.get("/tasks/{task_id}", response_model=TaskOut)
async def get_task(
    task_id: str,
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
) -> AgentTask:
    """Read one background run."""
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")

    try:
        parsed = _uuid.UUID(task_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found") from exc

    task = await session.scalar(select(AgentTask).where(AgentTask.id == parsed))
    # 404 rather than 403, because 403 would confirm the identifier is real.
    if task is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Task not found")
    return task


@app.get("/tasks", response_model=list[TaskOut])
async def list_tasks(
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list:
    if session is None:
        return []
    result = await session.scalars(
        select(AgentTask).order_by(AgentTask.created_at.desc()).limit(limit)
    )
    return list(result)


@app.get("/invocations", response_model=list[ToolInvocationOut])
async def list_invocations(
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list:
    if session is None:
        return []
    result = await session.scalars(
        select(ToolInvocation).order_by(ToolInvocation.created_at.desc()).limit(limit)
    )
    return list(result)


@app.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> list:
    if session is None:
        return []
    result = await session.scalars(
        select(Conversation).order_by(Conversation.created_at.desc()).limit(limit)
    )
    return list(result)


@app.get("/conversations/{conversation_id}/messages", response_model=list[MessageOut])
async def list_messages(
    conversation_id: str,
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
) -> list:
    if session is None:
        return []
    try:
        parsed = _uuid.UUID(conversation_id)
    except ValueError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found") from exc

    conversation = await session.scalar(
        select(Conversation).where(Conversation.id == parsed)
    )
    # 404 rather than 403. Another tenant's conversation is not visible here at
    # all, and a 403 would confirm the identifier is real.
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")

    result = await session.scalars(
        select(Message).where(Message.conversation_id == parsed)
        .order_by(Message.created_at)
    )
    return list(result)


@app.get("/audit", response_model=list[AuditOut])
async def list_audit(
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.ADMIN))],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list:
    """Administrators read their own tenant's audit trail, and only their own."""
    if session is None:
        return []
    result = await session.scalars(
        select(AuditEvent).order_by(AuditEvent.created_at.desc()).limit(limit)
    )
    return list(result)
