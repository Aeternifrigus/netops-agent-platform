"""Tenant scope across commits, and connection use during a live agent call."""
from __future__ import annotations

from types import SimpleNamespace

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


class _FakeRunner:
    """Stands in for the ADK runner and records pool use while it 'thinks'."""

    checked_out_during_run: list[int] = []

    def __init__(self, agent, app_name):
        async def create_session(app_name, user_id):
            return SimpleNamespace(id="adk-session")

        self.session_service = SimpleNamespace(create_session=create_session)

    async def run_async(self, user_id, session_id, new_message):
        from app.db import get_engine

        _FakeRunner.checked_out_during_run.append(get_engine().pool.checkedout())
        part = SimpleNamespace(text="tower-003 feeds two downstream sites")
        yield SimpleNamespace(content=SimpleNamespace(parts=[part]))


async def test_chat_holds_no_database_connection_while_the_model_runs(
    client, acme_operator, monkeypatch
):
    """A live agent call can take many seconds. Holding a pooled connection
    for all of it caps concurrent chats at the pool size and starves every
    other endpoint."""
    import google.adk.runners

    import app.main as app_main

    monkeypatch.setenv("GOOGLE_API_KEY", "test-key-never-sent")
    monkeypatch.setattr(google.adk.runners, "InMemoryRunner", _FakeRunner)
    monkeypatch.setattr(app_main, "_get_orchestrator", lambda: None)
    _FakeRunner.checked_out_during_run.clear()

    response = await client.post(
        "/chat", json={"message": "what fails if tower-003 goes down?"},
        headers=acme_operator.headers,
    )

    assert response.status_code == 200, response.text
    assert _FakeRunner.checked_out_during_run == [0]

    # The conversation written after the model call still lands in the
    # caller's tenant, which needs the scope to come back after the release.
    conversation_id = response.json()["conversation_id"]
    listed = await client.get("/conversations", headers=acme_operator.headers)
    assert [c["id"] for c in listed.json()] == [conversation_id]
