"""Authentication: tenant bootstrap, login, refresh, and user administration."""
from __future__ import annotations

import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ..config import settings
from ..db import apply_tenant_scope, unscoped_session
from ..deps import CREDENTIALS_ERROR, CurrentPrincipal, Recording, Session, require_role
from ..models import AuditEvent, Role, Tenant, User
from ..schemas import (
    RefreshRequest,
    TenantCreate,
    TenantOut,
    TokenPair,
    UserCreate,
    UserOut,
)
from ..security import TokenError, create_token, decode_token, hash_password, verify_password

logger = logging.getLogger(__name__)
router = APIRouter(tags=["auth"])


def require_database() -> None:
    """Accounts need somewhere to live."""
    if not settings.auth_enabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Authentication is not configured: set DATABASE_URL and run the migrations.",
        )


@router.post("/tenants", response_model=TenantOut, status_code=status.HTTP_201_CREATED)
async def create_tenant(payload: TenantCreate) -> Tenant:
    require_database()
    """
    Register a tenant and its first administrator in one transaction.

    Both or neither. A tenant created without its admin is unreachable and has to
    be cleaned up by hand.
    """
    async with unscoped_session() as session:
        tenant = Tenant(slug=payload.slug, name=payload.name)
        session.add(tenant)
        try:
            await session.flush()
            # Scope only after the tenant exists, so the admin insert is written
            # under the same policy that will govern every later read of it.
            await apply_tenant_scope(session, tenant.id)
            session.add(User(
                tenant_id=tenant.id,
                email=str(payload.admin_email).lower(),
                password_hash=hash_password(payload.admin_password),
                role=Role.ADMIN.value,
            ))
            await session.flush()
        except IntegrityError as exc:
            await session.rollback()
            raise HTTPException(
                status.HTTP_409_CONFLICT, "That tenant slug is already taken"
            ) from exc
        await session.refresh(tenant)
        return tenant


@router.post("/auth/token", response_model=TokenPair)
async def login(form: Annotated[OAuth2PasswordRequestForm, Depends()]) -> TokenPair:
    require_database()
    """
    Standard OAuth2 password flow.

    The tenant is taken from the scope field rather than invented, because an
    email address is unique only within a tenant. Without it, one person in two
    tenants is ambiguous at login.
    """
    tenant_slug = (form.scopes[0] if form.scopes else "").strip().lower()
    if not tenant_slug:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Provide the tenant slug in the OAuth2 scope field",
        )

    placeholder = "$2b$12$" + "." * 53

    async with unscoped_session() as session:
        # tenants has no RLS policy; resolve it, then scope the session.
        tenant = await session.scalar(
            select(Tenant).where(Tenant.slug == tenant_slug, Tenant.is_active.is_(True))
        )
        if tenant is None:
            # Verify a hash anyway, so an unknown tenant and a wrong password
            # take comparable time and the response cannot be used to enumerate.
            verify_password(form.password, placeholder)
            raise CREDENTIALS_ERROR

        await apply_tenant_scope(session, tenant.id)
        user = await session.scalar(
            select(User).where(User.email == form.username.lower())
        )

        if user is None:
            verify_password(form.password, placeholder)
            raise CREDENTIALS_ERROR
        if not user.is_active or not verify_password(form.password, user.password_hash):
            session.add(AuditEvent(
                tenant_id=user.tenant_id, actor_id=user.id, action="login",
                outcome="blocked", detail={"reason": "bad credentials"},
            ))
            await session.commit()
            raise CREDENTIALS_ERROR

        session.add(AuditEvent(
            tenant_id=user.tenant_id, actor_id=user.id, action="login",
            outcome="allowed", detail={},
        ))
        return _issue(user)


@router.post("/auth/refresh", response_model=TokenPair)
async def refresh(payload: RefreshRequest) -> TokenPair:
    require_database()
    try:
        claims = decode_token(payload.refresh_token, expect="refresh")
    except TokenError as exc:
        raise CREDENTIALS_ERROR from exc

    async with unscoped_session() as session:
        await apply_tenant_scope(session, uuid.UUID(claims["tid"]))
        user = await session.scalar(select(User).where(User.id == uuid.UUID(claims["sub"])))
        if user is None or not user.is_active:
            raise CREDENTIALS_ERROR
        # Re-issued from the stored role, not the role inside the refresh token,
        # so a demotion takes effect at the next refresh instead of at expiry.
        return _issue(user)


@router.get("/users/me", response_model=UserOut)
async def me(principal: CurrentPrincipal, session: Session) -> User:
    if principal.is_open_mode:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            "No accounts exist: the platform is running in open mode with no database",
        )
    return await session.scalar(select(User).where(User.id == principal.user_id))


@router.post("/users", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    session: Session,
    recorder: Recording,
    admin: Annotated[object, Depends(require_role(Role.ADMIN))],
) -> User:
    """Add a user to the caller's own tenant."""
    if admin.is_open_mode:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Cannot create accounts in open mode; configure DATABASE_URL first",
        )

    user = User(
        tenant_id=admin.tenant_id,
        email=str(payload.email).lower(),
        password_hash=hash_password(payload.password),
        role=payload.role.value,
    )
    session.add(user)
    try:
        await session.flush()
    except IntegrityError as exc:
        await session.rollback()
        raise HTTPException(
            status.HTTP_409_CONFLICT, "That email already exists in this tenant"
        ) from exc

    recorder.audit("user.create", "allowed", {"role": payload.role.value})
    return user


def _issue(user: User) -> TokenPair:
    return TokenPair(
        access_token=create_token(
            subject=user.id, tenant_id=user.tenant_id, role=user.role, token_type="access"
        ),
        refresh_token=create_token(
            subject=user.id, tenant_id=user.tenant_id, role=user.role, token_type="refresh"
        ),
        expires_in=settings.access_token_minutes * 60,
    )
