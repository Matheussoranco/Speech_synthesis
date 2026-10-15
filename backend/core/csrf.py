"""
CSRF protection utilities
"""

from fastapi import Request
from fastapi.responses import JSONResponse
from urllib.parse import urlsplit
from core.auth import PrincipalKind, principal_for
from core.config import settings

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


async def request_guard(request: Request, call_next):
    """Reject DNS-rebinding hosts, anonymous remote access and cross-site writes."""
    host = (request.url.hostname or "").lower()
    if host not in _LOOPBACK:
        if host not in settings.allowed_hosts:
            return JSONResponse({"detail": "Host not allowed"}, status_code=400)
        if principal_for(request).kind is PrincipalKind.ANONYMOUS:
            return JSONResponse({"detail": "Authentication required"}, status_code=401)
    browser_sent = "origin" in request.headers or "referer" in request.headers
    if request.method not in _SAFE_METHODS and browser_sent and not cookie_csrf_allowed(request):
        return JSONResponse({"detail": "Cross-origin request rejected"}, status_code=403)
    return await call_next(request)

def effective_scheme(request: Request) -> str:
    """
    Determine the effective scheme (http/https) for the request.
    Handles trusted proxy headers.
    """
    # Check X-Forwarded-Proto from trusted proxies
    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto:
        return forwarded_proto.split(",")[0].strip().lower()
    
    # Check X-Forwarded-Ssl
    if request.headers.get("x-forwarded-ssl") == "on":
        return "https"
    
    # Fall back to request URL scheme
    return request.url.scheme

def cookie_csrf_allowed(request: Request) -> bool:
    """Allow cookie mutations only from an exact configured origin."""
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    value = origin if origin is not None else referer
    candidate = _origin(value, referer=origin is None)
    return candidate is not None and any(candidate == _origin(allowed) for allowed in settings.cors_origins)


def _origin(value: str | None, *, referer: bool = False):
    if not value or any(ord(char) <= 32 or ord(char) == 127 for char in value):
        return None
    try:
        url = urlsplit(value)
        if url.scheme not in {'http', 'https', 'voicestudio'} or not url.hostname or url.username or url.password:
            return None
        if not referer and (url.path not in {'', '/'} or url.query or url.fragment):
            return None
        port = url.port or {'http': 80, 'https': 443}.get(url.scheme)
        return url.scheme, url.hostname, port
    except ValueError:
        return None
