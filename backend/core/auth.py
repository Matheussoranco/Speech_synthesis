"""
Authentication utilities
"""

from enum import Enum
from typing import Optional
from fastapi import Request
import secrets
import time

sessions: dict[str, dict[str, float]] = {}

class CredentialTransport(str, Enum):
    COOKIE = "cookie"
    BEARER = "bearer"

class PrincipalKind(str, Enum):
    ANONYMOUS = "anonymous"
    ADMIN_SESSION = "admin_session"
    REMOTE_API_KEY = "remote_api_key"
    LEGACY_MASTER = "legacy_master"

class Principal:
    def __init__(self, kind: PrincipalKind, credential_id: str = "", transport: CredentialTransport = CredentialTransport.BEARER):
        self.kind = kind
        self.credential_id = credential_id
        self.transport = transport

def remote_api_key() -> Optional[str]:
    """Get the configured remote API key from settings"""
    from core.config import settings
    return settings.remote_api_key

def authorization_credential_present(request: Request) -> bool:
    """Check if Authorization header is present"""
    return "authorization" in request.headers

def master_header_valid(request: Request) -> bool:
    """Validate Authorization header against configured key"""
    key = remote_api_key()
    if not key:
        return False
    auth = request.headers.get("authorization", "")
    return secrets.compare_digest(auth.encode('utf-8'), f"Bearer {key}".encode('utf-8'))

def legacy_master_cookie_valid(request: Request) -> bool:
    """Check legacy ov_key cookie"""
    key = remote_api_key()
    if not key:
        return False
    cookie = request.cookies.get("ov_key")
    return bool(cookie) and secrets.compare_digest(cookie.encode('utf-8'), key.encode('utf-8'))

def principal_for(request: Request) -> Principal:
    """Determine the principal for a request"""
    key = remote_api_key()
    if not key:
        return Principal(PrincipalKind.ANONYMOUS)
    
    if authorization_credential_present(request):
        if master_header_valid(request):
            return Principal(PrincipalKind.REMOTE_API_KEY, key, CredentialTransport.BEARER)
        authorization = request.headers.get("authorization", "")
        if authorization.startswith("Bearer "):
            token = authorization[7:]
            if _session_valid(token):
                return Principal(PrincipalKind.ADMIN_SESSION, token, CredentialTransport.BEARER)
        return Principal(PrincipalKind.ANONYMOUS)
    
    if legacy_master_cookie_valid(request):
        return Principal(PrincipalKind.LEGACY_MASTER, key, CredentialTransport.COOKIE)
    
    # Check session cookie
    session_token = request.cookies.get("ov_session")
    if session_token and _session_valid(session_token):
        return Principal(PrincipalKind.ADMIN_SESSION, session_token, CredentialTransport.COOKIE)
    
    return Principal(PrincipalKind.ANONYMOUS)


def _session_valid(token: str) -> bool:
    session = sessions.get(token)
    if not session:
        return False
    if session["expires"] <= time.time():
        sessions.pop(token, None)
        return False
    return True
