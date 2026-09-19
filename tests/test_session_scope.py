"""Tenant scope across commits."""
from __future__ import annotations

from sqlalchemy import select, text

from tests.conftest import requires_database

pytestmark = requires_database


async def test_tenant_scope_survives_a_commit(acme_admin, globex_admin):
    """set_config(..., true) lasts one transaction. A session that commits and
    keeps going must still be scoped, or it silently reads and writes nothing."""
    from app.db import tenant_session
    from app.models import User

    async with tenant_session(acme_admin.tenant.id) as session:
        await session.commit()
        scope = await session.scalar(text("SELECT current_setting('app.tenant_id', true)"))
        emails = {u.email for u in await session.scalars(select(User))}

    assert scope == str(acme_admin.tenant.id)
    assert emails == {"admin@acme-corp.example"}


async def test_a_write_after_commit_now_lands_in_the_right_tenant(acme_operator):
    from app.db import tenant_session
    from app.deps import Principal, Recorder
    from app.models import AuditEvent, Role

    principal = Principal(
        user_id=acme_operator.user.id, tenant_id=acme_operator.tenant.id,
        role=Role.OPERATOR, authenticated=True,
    )
    async with tenant_session(acme_operator.tenant.id) as session:
        recorder = Recorder(session, principal)
        recorder.audit("first", "allowed")
        await recorder.commit_now()
        recorder.audit("second", "allowed")

    async with tenant_session(acme_operator.tenant.id) as session:
        actions = {e.action for e in await session.scalars(select(AuditEvent))}
    assert {"first", "second"} <= actions
