"""Settings for the authentication and multi-tenancy layer."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    environment: str = Field(default="local")

    # Absent means open mode: no accounts, no tenants, no audit trail.
    database_url: str | None = Field(default=None)
    db_echo: bool = False

    jwt_secret: str = Field(default="insecure-local-development-key")
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 14

    @property
    def auth_enabled(self) -> bool:
        return bool(self.database_url)

    def validate_for_environment(self) -> list[str]:
        """Configuration that should stop a non-local boot rather than warn."""
        problems: list[str] = []
        if self.environment == "local":
            return problems

        if not self.database_url:
            problems.append(
                "no DATABASE_URL, which leaves every endpoint unauthenticated"
            )
        if self.jwt_secret == "insecure-local-development-key":
            problems.append("jwt_secret is still the development default")
        elif len(self.jwt_secret) < 32:
            problems.append("jwt_secret is shorter than 32 characters")
        return problems


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
