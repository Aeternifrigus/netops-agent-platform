"""Settings for the authentication and multi-tenancy layer."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Defaults to production so a deployment that forgets to set it gets the
    # strict checks below, not the relaxed local ones.
    environment: str = Field(default="production")

    # Absent means open mode: no accounts, no tenants, no audit trail.
    database_url: str | None = Field(default=None)
    # Open mode has to be asked for. A missing DATABASE_URL on its own is
    # treated as a mistake, never as a request to switch authentication off.
    allow_open_mode: bool = Field(default=False)
    db_echo: bool = False

    # Unset: Celery runs eagerly and background /chat is disabled.
    broker_url: str | None = Field(default=None)

    jwt_secret: str = Field(default="insecure-local-development-key")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 14

    @property
    def auth_enabled(self) -> bool:
        return bool(self.database_url)

    @property
    def background_enabled(self) -> bool:
        """Background execution needs both a broker and somewhere to record runs."""
        return bool(self.broker_url and self.database_url)

    def validate_for_environment(self) -> list[str]:
        """Configuration that should stop the service from booting."""
        problems: list[str] = []
        is_local = self.environment == "local"

        if not self.database_url:
            if not self.allow_open_mode:
                problems.append(
                    "no DATABASE_URL, which would leave every endpoint unauthenticated; "
                    "set DATABASE_URL, or ALLOW_OPEN_MODE=true with ENVIRONMENT=local "
                    "for local development"
                )
            elif not is_local:
                problems.append(
                    f"ALLOW_OPEN_MODE is only permitted with ENVIRONMENT=local "
                    f"(ENVIRONMENT is {self.environment!r})"
                )

        if is_local:
            return problems

        if self.jwt_secret == "insecure-local-development-key":
            problems.append("jwt_secret is still the development default")
        elif len(self.jwt_secret) < 32:
            problems.append("jwt_secret is shorter than 32 characters")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
