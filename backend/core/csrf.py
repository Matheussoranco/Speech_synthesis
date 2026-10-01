"
CSRF protection utilities
"

from fastapi import Request

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
    """
    Check if the request origin is allowed for cookie-based CSRF.
    In development, allow localhost. In production, require exact match.
    """
    origin = request.headers.get("origin")
    referer = request.headers.get("referer")
    
    if not origin and not referer:
        return False
    
    # In development, allow localhost
    if origin and origin.startswith("http://localhost"):
        return True
    if referer and referer.startswith("http://localhost"):
        return True
    
    # Check against app protocol
    if origin and origin.startswith("app://voicestudio"):
        return True
    if referer and referer.startswith("app://voicestudio"):
        return True
    
    return False
