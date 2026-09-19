"""Fixtures for the tenancy and authentication tests."""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

from app.config import settings

requires_database = pytest.mark.skipif(
    not settings.auth_enabled,
    reason="set DATABASE_URL to run the tenancy and authentication tests",
)

PASSWORD = "correct-horse-battery"


if settings.auth_enabled:
    from app.db import apply_tenant_scope, dispose_engine, get_sessionmaker
    from app.main import app
    from app.models import Role, Tenant, User
    from app.security import create_token, hash_password

    @pytest_asyncio.fixture(autouse=True)
    async def clean_database() -> AsyncIterator[None]:
        """Truncate before each test, dispose the engine after."""
        async with get_sessionmaker()() as session:
            await session.execute(text(
                "TRUNCATE agent_tasks, audit_events, messages, conversations, tool_invocations, "
                "users, tenants CASCADE"
            ))
            await session.commit()
        yield
        await dispose_engine()

    @pytest_asyncio.fixture
    async def client() -> AsyncIterator[AsyncClient]:
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as c:
            yield c

    async def make_tenant(slug: str) -> Tenant:
        async with get_sessionmaker()() as session:
            tenant = Tenant(id=uuid.uuid4(), slug=slug, name=slug.title())
            session.add(tenant)
            await session.commit()
            return tenant

    async def make_user(tenant: Tenant, email: str, role: Role) -> User:
        """Note the absence of a refresh after commit."""
        async with get_sessionmaker()() as session:
            await apply_tenant_scope(session, tenant.id)
            user = User(
                id=uuid.uuid4(), tenant_id=tenant.id, email=email,
                password_hash=hash_password(PASSWORD), role=role.value,
            )
            session.add(user)
            await session.commit()
            return user

    class Actor:
        """A user plus a ready-made Authorization header."""

        def __init__(self, tenant: Tenant, user: User) -> None:
            self.tenant = tenant
            self.user = user
            self.token = create_token(
                subject=user.id, tenant_id=tenant.id, role=user.role,
                token_type="access",
            )

        @property
        def headers(self) -> dict[str, str]:
            return {"Authorization": f"Bearer {self.token}"}

    @pytest_asyncio.fixture
    async def acme_admin() -> Actor:
        tenant = await make_tenant("acme")
        return Actor(tenant, await make_user(tenant, "admin@acme-corp.example", Role.ADMIN))

    @pytest_asyncio.fixture
    async def acme_viewer(acme_admin: Actor) -> Actor:
        user = await make_user(acme_admin.tenant, "viewer@acme-corp.example", Role.VIEWER)
        return Actor(acme_admin.tenant, user)

    @pytest_asyncio.fixture
    async def acme_operator(acme_admin: Actor) -> Actor:
        user = await make_user(acme_admin.tenant, "op@acme-corp.example", Role.OPERATOR)
        return Actor(acme_admin.tenant, user)

    @pytest_asyncio.fixture
    async def globex_admin() -> Actor:
        """A second, entirely separate tenant. The isolation argument needs one."""
        tenant = await make_tenant("globex")
        return Actor(tenant, await make_user(tenant, "admin@globex-inc.example", Role.ADMIN))
