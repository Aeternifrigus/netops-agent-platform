"""A GraphQL read surface over the same tenant-scoped data."""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Annotated, Any

import strawberry
from fastapi import Depends
from sqlalchemy import select
from strawberry.fastapi import GraphQLRouter

from .deps import Principal, Session, require_role
from .models import AgentTask, AuditEvent, Conversation, Message, Role, ToolInvocation

logger = logging.getLogger(__name__)


@strawberry.type
class MessageType:
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime


@strawberry.type
class ConversationType:
    id: uuid.UUID
    title: str
    created_at: datetime

    @strawberry.field
    async def messages(self, info: strawberry.Info) -> list[MessageType]:
        """Resolved lazily, so a caller asking only for titles pays for no messages."""
        session = info.context["session"]
        if session is None:
            return []
        rows = await session.scalars(
            select(Message).where(Message.conversation_id == self.id)
            .order_by(Message.created_at)
        )
        return [
            MessageType(id=m.id, role=m.role, content=m.content, created_at=m.created_at)
            for m in rows
        ]


@strawberry.type
class ToolInvocationType:
    id: uuid.UUID
    tool: str
    duration_ms: float
    created_at: datetime

    @strawberry.field
    def arguments_json(self) -> str:
        """Serialised rather than typed."""
        import json

        return json.dumps(self._arguments)

    _arguments: strawberry.Private[dict[str, Any]]


@strawberry.type
class AgentTaskType:
    id: uuid.UUID
    status: str
    prompt: str
    reply: str | None
    error: str | None
    duration_ms: float
    created_at: datetime
    finished_at: datetime | None


@strawberry.type
class AuditEventType:
    id: uuid.UUID
    action: str
    outcome: str
    created_at: datetime


def _principal(info: strawberry.Info) -> Principal:
    return info.context["principal"]


def _require(info: strawberry.Info, minimum: Role) -> None:
    """Role check inside the resolver."""
    principal = _principal(info)
    if principal.is_open_mode:
        return
    if not principal.role.satisfies(minimum):
        raise PermissionError(f"This field requires the {minimum.value} role")


@strawberry.type
class Query:
    @strawberry.field
    async def conversations(
        self, info: strawberry.Info, limit: int = 20
    ) -> list[ConversationType]:
        _require(info, Role.VIEWER)
        session = info.context["session"]
        if session is None:
            return []
        rows = await session.scalars(
            select(Conversation).order_by(Conversation.created_at.desc())
            .limit(max(1, min(limit, 100)))
        )
        return [
            ConversationType(id=c.id, title=c.title, created_at=c.created_at)
            for c in rows
        ]

    @strawberry.field
    async def conversation(
        self, info: strawberry.Info, id: uuid.UUID
    ) -> ConversationType | None:
        """Fetch one conversation by id."""
        _require(info, Role.VIEWER)
        session = info.context["session"]
        if session is None:
            return None
        row = await session.scalar(select(Conversation).where(Conversation.id == id))
        if row is None:
            return None
        return ConversationType(id=row.id, title=row.title, created_at=row.created_at)

    @strawberry.field
    async def invocations(
        self, info: strawberry.Info, limit: int = 20
    ) -> list[ToolInvocationType]:
        _require(info, Role.VIEWER)
        session = info.context["session"]
        if session is None:
            return []
        rows = await session.scalars(
            select(ToolInvocation).order_by(ToolInvocation.created_at.desc())
            .limit(max(1, min(limit, 100)))
        )
        return [
            ToolInvocationType(
                id=r.id, tool=r.tool, duration_ms=r.duration_ms,
                created_at=r.created_at, _arguments=r.arguments or {},
            )
            for r in rows
        ]

    @strawberry.field
    async def tasks(self, info: strawberry.Info, limit: int = 20) -> list[AgentTaskType]:
        _require(info, Role.VIEWER)
        session = info.context["session"]
        if session is None:
            return []
        rows = await session.scalars(
            select(AgentTask).order_by(AgentTask.created_at.desc())
            .limit(max(1, min(limit, 100)))
        )
        return [
            AgentTaskType(
                id=t.id, status=t.status, prompt=t.prompt, reply=t.reply,
                error=t.error, duration_ms=t.duration_ms,
                created_at=t.created_at, finished_at=t.finished_at,
            )
            for t in rows
        ]

    @strawberry.field
    async def audit(self, info: strawberry.Info, limit: int = 50) -> list[AuditEventType]:
        """Administrators only, and only their own tenant's trail."""
        _require(info, Role.ADMIN)
        session = info.context["session"]
        if session is None:
            return []
        rows = await session.scalars(
            select(AuditEvent).order_by(AuditEvent.created_at.desc())
            .limit(max(1, min(limit, 200)))
        )
        return [
            AuditEventType(
                id=e.id, action=e.action, outcome=e.outcome, created_at=e.created_at
            )
            for e in rows
        ]


schema = strawberry.Schema(query=Query)


async def get_context(
    session: Session,
    principal: Annotated[Principal, Depends(require_role(Role.VIEWER))],
) -> dict[str, Any]:
    """Build the resolver context from the same dependencies REST uses."""
    return {"session": session, "principal": principal}


router = GraphQLRouter(schema, context_getter=get_context, path="/graphql")
