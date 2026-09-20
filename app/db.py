"""Database access, and the mechanism that makes tenant isolation real."""
from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session

from .config import settings

logger = logging.getLogger(__name__)

TENANT_SETTING = "app.tenant_id"

_engine: AsyncEngine | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    global _engine
    if not settings.database_url:
        raise RuntimeError(
            "no database is configured; the service is running in open mode and "
            "nothing should be reaching the database layer"
        )
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            echo=settings.db_echo,
            pool_pre_ping=True,
            pool_size=5,
            max_overflow=10,
        )
    return _engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _sessionmaker
    if _sessionmaker is None:
        _sessionmaker = async_sessionmaker(
            get_engine(), expire_on_commit=False, autoflush=False
        )
    return _sessionmaker


async def dispose_engine() -> None:
    global _engine, _sessionmaker
    if _engine is not None:
        await _engine.dispose()
    _engine, _sessionmaker = None, None


_TENANT_INFO_KEY = "tenant_id"
_SET_SCOPE = text("SELECT set_config(:key, :value, true)")


def _scope_params(tenant_id: uuid.UUID | None) -> dict[str, str]:
    return {"key": TENANT_SETTING, "value": str(tenant_id) if tenant_id else ""}


@event.listens_for(Session, "after_begin")
def _rescope_new_transaction(session: Session, transaction, connection) -> None:
    """Re-apply the tenant scope at the start of every transaction.

    set_config(..., true) only lasts until the transaction ends, so without
    this a session that commits and keeps going (Recorder.commit_now, or
    /chat releasing its connection during the model call) would continue
    with no tenant: reads return nothing and writes fail the RLS check.
    """
    if _TENANT_INFO_KEY in session.info:
        connection.execute(_SET_SCOPE, _scope_params(session.info[_TENANT_INFO_KEY]))


async def apply_tenant_scope(session: AsyncSession, tenant_id: uuid.UUID | None) -> None:
    """Bind this session to one tenant, for this and every later transaction."""
    session.info[_TENANT_INFO_KEY] = tenant_id
    # Also set it now, in case a transaction is already open (login resolves
    # the tenant first and scopes the same transaction afterwards).
    await session.execute(_SET_SCOPE, _scope_params(tenant_id))


@asynccontextmanager
async def tenant_session(tenant_id: uuid.UUID | None) -> AsyncIterator[AsyncSession]:
    """A session already scoped to one tenant, for background work and scripts."""
    async with get_sessionmaker()() as session:
        await apply_tenant_scope(session, tenant_id)
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@asynccontextmanager
async def unscoped_session() -> AsyncIterator[AsyncSession]:
    """A session with no tenant bound yet, used only by tenant creation and login."""
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
