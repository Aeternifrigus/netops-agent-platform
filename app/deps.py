"""Dependencies: authentication, tenant scoping and role checks."""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Annotated, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import settings
from .models import AuditEvent, Role, ToolInvocation, User
from .security import TokenError, decode_token

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token", auto_error=False)

CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    # Same message for every auth failure, to avoid user enumeration.
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


@dataclass(frozen=True)
class Principal:
    """Who is acting, in a form both modes can produce."""

    user_id: uuid.UUID | None
    tenant_id: uuid.UUID | None
    role: Role
    authenticated: bool

    @property
    def is_open_mode(self) -> bool:
        return not self.authenticated


OPEN_MODE_PRINCIPAL = Principal(
    user_id=None, tenant_id=None, role=Role.ADMIN, authenticated=False
)


async def get_claims(token: Annotated[str | None, Depends(oauth2_scheme)]) -> dict | None:
    if not settings.auth_enabled:
        return None
    if not token:
        raise CREDENTIALS_ERROR
    try:
        return decode_token(token, expect="access")
    except TokenError as exc:
        logger.info("rejected token: %s", exc)
        raise CREDENTIALS_ERROR from exc


async def get_session(
    claims: Annotated[dict | None, Depends(get_claims)],
) -> AsyncIterator[AsyncSession | None]:
    """A session bound to the caller's tenant, or None in open mode."""
    if claims is None:
        yield None
        return

    from .db import apply_tenant_scope, get_sessionmaker

    async with get_sessionmaker()() as session:
        await apply_tenant_scope(session, uuid.UUID(claims["tid"]))
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_principal(
    claims: Annotated[dict | None, Depends(get_claims)],
    session: Annotated[AsyncSession | None, Depends(get_session)],
) -> Principal:
    if claims is None or session is None:
        return OPEN_MODE_PRINCIPAL

    user = await session.scalar(select(User).where(User.id == uuid.UUID(claims["sub"])))
    if user is None or not user.is_active:
        raise CREDENTIALS_ERROR

    try:
        role = Role(user.role)
    except ValueError as exc:
        logger.error("user %s has unrecognised role %r", user.id, user.role)
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Role not recognised") from exc

    # The role comes from the row, not the token. A privilege revoked five
    # minutes ago must not survive until an old token expires.
    return Principal(
        user_id=user.id, tenant_id=user.tenant_id, role=role, authenticated=True
    )


CurrentPrincipal = Annotated[Principal, Depends(get_principal)]
Session = Annotated["AsyncSession | None", Depends(get_session)]


def require_role(minimum: Role):
    """Authorise by rank, declared in the route signature."""

    async def dependency(principal: CurrentPrincipal) -> Principal:
        if principal.is_open_mode:
            return principal
        if not principal.role.satisfies(minimum):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"This action requires the {minimum.value} role",
            )
        return principal

    return dependency


class Recorder:
    """Writes the audit trail and the invocation log, or does nothing in open mode."""

    def __init__(self, session: AsyncSession | None, principal: Principal) -> None:
        self._session = session
        self._principal = principal

    @property
    def active(self) -> bool:
        return self._session is not None and self._principal.tenant_id is not None

    def tool_call(
        self, tool: str, arguments: dict[str, Any], result: dict[str, Any],
        duration_ms: float,
    ) -> None:
        if not self.active:
            return
        self._session.add(ToolInvocation(
            tenant_id=self._principal.tenant_id, actor_id=self._principal.user_id,
            tool=tool, arguments=arguments, result=result,
            duration_ms=round(duration_ms, 3),
        ))

    def audit(self, action: str, outcome: str, detail: dict | None = None) -> None:
        if not self.active:
            return
        self._session.add(AuditEvent(
            tenant_id=self._principal.tenant_id, actor_id=self._principal.user_id,
            action=action, outcome=outcome, detail=detail or {},
        ))

    async def commit_now(self) -> None:
        """Persist immediately, for records that must survive the request failing."""
        if self.active:
            await self._session.commit()


async def get_recorder(
    session: Annotated[AsyncSession | None, Depends(get_session)],
    principal: CurrentPrincipal,
) -> Recorder:
    return Recorder(session, principal)


Recording = Annotated[Recorder, Depends(get_recorder)]
