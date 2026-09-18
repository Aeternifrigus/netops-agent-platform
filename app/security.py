"""Credentials and tokens."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import bcrypt
from jose import JWTError, jwt

from .config import settings

TokenType = Literal["access", "refresh"]

# bcrypt truncates silently at 72 bytes, so long passphrases are rejected up
# front rather than quietly compared on their first 72 bytes.
MAX_PASSWORD_BYTES = 72
BCRYPT_ROUNDS = 12

# bcrypt directly, not passlib: passlib 1.7.4 breaks on bcrypt >= 4.1.


class TokenError(Exception):
    """Raised for any token that cannot be trusted, with no detail for the caller."""


def hash_password(password: str) -> str:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        raise ValueError(f"password exceeds {MAX_PASSWORD_BYTES} bytes")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    encoded = password.encode("utf-8")
    if len(encoded) > MAX_PASSWORD_BYTES:
        return False
    try:
        return bcrypt.checkpw(encoded, password_hash.encode("ascii"))
    except (ValueError, TypeError):
        # A malformed stored hash must read as a failed login, not a 500 that
        # tells the caller this account is different from the others.
        return False


def create_token(
    *,
    subject: uuid.UUID,
    tenant_id: uuid.UUID,
    role: str,
    token_type: TokenType = "access",
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(UTC)
    if expires_delta is None:
        expires_delta = (
            timedelta(minutes=settings.access_token_minutes)
            if token_type == "access"
            else timedelta(days=settings.refresh_token_days)
        )

    claims: dict[str, Any] = {
        "sub": str(subject),
        "tid": str(tenant_id),
        "role": role,
        "typ": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
        "jti": uuid.uuid4().hex,
    }
    return jwt.encode(claims, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, expect: TokenType = "access") -> dict[str, Any]:
    """Verify signature, expiry and token type together."""
    try:
        claims = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
    except JWTError as exc:
        raise TokenError("token is not valid") from exc

    if claims.get("typ") != expect:
        raise TokenError(f"expected a {expect} token")
    for field in ("sub", "tid", "role"):
        if not claims.get(field):
            raise TokenError("token is missing required claims")
    try:
        uuid.UUID(claims["sub"])
        uuid.UUID(claims["tid"])
    except (ValueError, TypeError) as exc:
        raise TokenError("token identifiers are malformed") from exc

    return claims
