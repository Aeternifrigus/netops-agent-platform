"""The service must refuse to boot in any configuration that would leave it open."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings

STRONG_SECRET = "x" * 48
DB = "postgresql+asyncpg://app:app@localhost:5432/netops"


def make(**overrides) -> Settings:
    """Settings from explicit values only, ignoring the shell and any .env file."""
    values = {
        "environment": "production",
        "database_url": None,
        "allow_open_mode": False,
        "jwt_secret": STRONG_SECRET,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def test_environment_defaults_to_production(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    assert Settings(_env_file=None).environment == "production"


def test_open_mode_defaults_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ALLOW_OPEN_MODE", raising=False)
    assert Settings(_env_file=None).allow_open_mode is False


def test_nothing_configured_is_refused() -> None:
    """What a deployment that forgot its environment variables looks like."""
    problems = make().validate_for_environment()
    assert any("DATABASE_URL" in p for p in problems)


def test_missing_database_is_refused_even_locally() -> None:
    problems = make(environment="local").validate_for_environment()
    assert any("DATABASE_URL" in p for p in problems)


def test_open_mode_is_refused_outside_local() -> None:
    problems = make(allow_open_mode=True).validate_for_environment()
    assert any("ALLOW_OPEN_MODE" in p for p in problems)


def test_open_mode_is_allowed_locally() -> None:
    settings = make(environment="local", allow_open_mode=True,
                    jwt_secret="insecure-local-development-key")
    assert settings.validate_for_environment() == []
    assert not settings.auth_enabled


def test_production_with_database_and_strong_secret_is_accepted() -> None:
    settings = make(database_url=DB)
    assert settings.validate_for_environment() == []
    assert settings.auth_enabled


@pytest.mark.parametrize("secret", ["insecure-local-development-key", "too-short"])
def test_weak_secret_is_refused_in_production(secret: str) -> None:
    problems = make(database_url=DB, jwt_secret=secret).validate_for_environment()
    assert any("jwt_secret" in p for p in problems)


def test_app_does_not_start_with_unsafe_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.main as main

    monkeypatch.setattr(main, "settings", make())
    with pytest.raises(RuntimeError, match="refusing to start"), TestClient(main.app):
        pass
