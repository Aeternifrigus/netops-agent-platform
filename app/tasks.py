"""Background execution for agent runs."""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from celery import Celery
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session as SyncSession
from sqlalchemy.orm import sessionmaker

from .config import settings

logger = logging.getLogger(__name__)

TASK_NAME = "netops.run_agent"


def _sync_database_url() -> str | None:
    """The application URL with the async driver swapped for the sync one."""
    if not settings.database_url:
        return None
    return settings.database_url.replace("+asyncpg", "+psycopg2")


def build_celery() -> Celery:
    """Configure Celery, falling back to eager execution with no broker."""
    broker = settings.broker_url
    app = Celery("netops", broker=broker or None, backend=broker or None)

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="UTC",
        enable_utc=True,
        # No automatic retries: a retried run is a second model charge.
        task_acks_late=True,
        task_reject_on_worker_lost=False,
        worker_prefetch_multiplier=1,
        result_expires=3600,
        task_always_eager=not broker,
        task_eager_propagates=False,
    )
    return app


celery_app = build_celery()

_engine = None
_sessionmaker: sessionmaker | None = None


def _worker_sessionmaker() -> sessionmaker:
    global _engine, _sessionmaker
    if _sessionmaker is None:
        url = _sync_database_url()
        if not url:
            raise RuntimeError("no database configured; the worker has nothing to write to")
        _engine = create_engine(url, pool_pre_ping=True, pool_size=2, max_overflow=4)
        _sessionmaker = sessionmaker(bind=_engine, expire_on_commit=False)
    return _sessionmaker


def _scope(session: SyncSession, tenant_id: str) -> None:
    """Re-apply the tenant scope inside the worker."""
    session.execute(
        text("SELECT set_config('app.tenant_id', :value, true)"),
        {"value": str(tenant_id)},
    )


def _run_agent_inline(message: str, actor: str) -> str:
    """Execute one agent conversation and return its text, synchronously."""
    import asyncio

    from google.adk.runners import InMemoryRunner
    from google.genai import types

    from .agents.orchestrator import build_orchestrator

    async def _run() -> str:
        runner = InMemoryRunner(agent=build_orchestrator(), app_name="netops")
        adk_session = await runner.session_service.create_session(
            app_name="netops", user_id=actor
        )
        out = ""
        async for event in runner.run_async(
            user_id=actor,
            session_id=adk_session.id,
            new_message=types.Content(role="user", parts=[types.Part(text=message)]),
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if part.text:
                        out += part.text
        return out

    return asyncio.run(_run())


@celery_app.task(bind=True, name=TASK_NAME)
def run_agent(
    self, task_id: str, tenant_id: str, actor_id: str | None
) -> dict[str, Any]:
    """Run one agent conversation and record the outcome against its tenant."""
    started = time.perf_counter()
    maker = _worker_sessionmaker()
    actor = actor_id or "open-mode"

    with maker() as session:
        _scope(session, tenant_id)
        row = session.execute(
            text(
                "UPDATE agent_tasks SET status = 'running', celery_task_id = :cid "
                "WHERE id = :id AND status = 'pending' RETURNING prompt"
            ),
            {"cid": getattr(self.request, "id", None), "id": task_id},
        ).first()
        session.commit()

    if row is None:
        # Wrong tenant (hidden by RLS) or already claimed (redelivery). Skip.
        logger.warning("task %s not claimable under tenant %s; skipping", task_id, tenant_id)
        return {"task_id": task_id, "status": "skipped", "duration_ms": 0.0}

    message = row.prompt

    status, reply, error = "succeeded", None, None
    conversation_id: uuid.UUID | None = None

    try:
        if not os.environ.get("GOOGLE_API_KEY"):
            # Record as a failed task rather than raising in the worker.
            status, error = "failed", "no model credentials configured"
        else:
            reply = _run_agent_inline(message, actor)
    except Exception as exc:  # noqa: BLE001 - worker boundary, recorded not raised
        logger.exception("agent task %s failed", task_id)
        status, error = "failed", f"{type(exc).__name__}: {exc}"

    duration_ms = (time.perf_counter() - started) * 1000

    with maker() as session:
        _scope(session, tenant_id)

        if status == "succeeded" and reply is not None:
            conversation_id = uuid.uuid4()
            session.execute(
                text(
                    "INSERT INTO conversations (id, tenant_id, created_by, title, created_at) "
                    "VALUES (:id, :tenant, :actor, :title, now())"
                ),
                {"id": conversation_id, "tenant": tenant_id,
                 "actor": actor_id, "title": message[:120]},
            )
            for role, content in (("user", message), ("assistant", reply)):
                session.execute(
                    text(
                        "INSERT INTO messages (id, tenant_id, conversation_id, role, "
                        "content, created_at) VALUES (:id, :tenant, :conv, :role, "
                        ":content, now())"
                    ),
                    {"id": uuid.uuid4(), "tenant": tenant_id, "conv": conversation_id,
                     "role": role, "content": content},
                )

        session.execute(
            text(
                "UPDATE agent_tasks SET status = :status, reply = :reply, error = :error, "
                "conversation_id = :conv, duration_ms = :ms, finished_at = :done "
                "WHERE id = :id"
            ),
            {"status": status, "reply": reply, "error": error, "conv": conversation_id,
             "ms": round(duration_ms, 3), "done": datetime.now(UTC), "id": task_id},
        )
        session.execute(
            text(
                "INSERT INTO audit_events (id, tenant_id, actor_id, action, outcome, "
                "detail, created_at) VALUES (:id, :tenant, :actor, 'chat.async', "
                ":outcome, :detail, now())"
            ),
            {"id": uuid.uuid4(), "tenant": tenant_id, "actor": actor_id,
             "outcome": "allowed" if status == "succeeded" else "error",
             "detail": '{"mode": "background"}'},
        )
        session.commit()

    return {
        "task_id": task_id,
        "status": status,
        "duration_ms": round(duration_ms, 3),
    }


def broker_mode() -> str:
    """What /health reports: a real broker, or inline execution."""
    return "celery" if settings.broker_url else "inline (no broker configured)"


def dispose_worker_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        _engine.dispose()
    _engine, _sessionmaker = None, None
