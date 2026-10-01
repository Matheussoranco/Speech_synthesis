"
Authentication for remote administration
"

from fastapi import APIRouter, HTTPException, Request, Response, Depends
from pydantic import BaseModel
from typing import Literal, Optional
import secrets
import time
from datetime import UTC, datetime

from core.config import settings
from core.auth import (
    CredentialTransport,
    PrincipalKind,
    authorization_credential_present,
    legacy_master_cookie_valid,
    master_header_valid,
    principal_for,
    remote_api_key,
)

router = APIRouter(prefix="/auth", tags=["auth"])

class SessionRequest(BaseModel):
    transport: Literal["cookie", "bearer"]

class SessionResponse(BaseModel):
    token: str
    expires_at: float
    expires_in: int

# Simple in-memory session store
sessions = {}
SESSION_TTL = settings.session_ttl_seconds

@router.post("/session", response_model=SessionResponse)
async def create_session(payload: SessionRequest, request: Request, response: Response):
    configured_key = remote_api_key()
    if not configured_key:
        raise HTTPException(401, "API key not configured")
    
    auth_header = authorization_credential_present(request)
    header_ok = master_header_valid(request)
    legacy_ok = legacy_master_cookie_valid(request)
    
    if auth_header:
        if not header_ok:
            raise HTTPException(401, "Invalid API key")
    elif legacy_ok:
        if payload.transport != "cookie":
            raise HTTPException(403, "Cookie transport required for legacy migration")
    else:
        raise HTTPException(401, "API key required")
    
    token = secrets.token_urlsafe(32)
    expires_at = time.time() + SESSION_TTL
    sessions[token] = {"created": time.time(), "expires": expires_at}
    
    if payload.transport == "bearer":
        return SessionResponse(token=token, expires_at=expires_at, expires_in=SESSION_TTL)
    
    # Set cookie
    response.set_cookie(
        "ov_session",
        token,
        max_age=SESSION_TTL,
        expires=datetime.fromtimestamp(expires_at, tz=UTC),
        path="/",
        secure=request.url.scheme == "https",
        httponly=True,
        samesite="strict",
    )
    
    # Clear legacy cookie
    response.delete_cookie("ov_key", path="/", secure=request.url.scheme == "https", httponly=True, samesite="strict")
    
    return Response(status_code=204)

@router.delete("/session", status_code=204)
async def delete_session(request: Request, response: Response):
    principal = principal_for(request)
    if principal.kind == PrincipalKind.ADMIN_SESSION:
        sessions.pop(principal.credential_id, None)
    
    response.delete_cookie("ov_session", path="/", secure=request.url.scheme == "https", httponly=True, samesite="strict")
    return Response(status_code=204)

@router.get("/session")
async def get_session(request: Request):
    principal = principal_for(request)
    if principal.kind == PrincipalKind.ADMIN_SESSION:
        session = sessions.get(principal.credential_id)
        if session:
            return {"valid": True, "expires_at": session["expires"]}
    return {"valid": False}
