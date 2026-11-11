"""
Tenant branding helpers — SINGLE source of truth for the tenant logo URL.
═══════════════════════════════════════════════════════════════════════════════
The MSystems master is the only authority for tenant branding. The logo BINARY
is served exclusively by the master endpoint:

    GET {gateway}/api/master/tenant-config/TenantLogo?domain=<host>   (Anonymous)

`tenant_mgmt.tenant_profile.logo_url` stores a PRIVATE relative path (e.g.
``logo/<file>.png``) that only the master can resolve into a stream — it is NOT a
renderable URL. Returning it raw to any client (VSCode webview, chatbot widget,
Teams package, web <img>) yields a broken image.

Every channel that needs to *display* a logo must therefore use an ABSOLUTE URL to
the master TenantLogo endpoint, exactly like agent.web and the Teams package
already do. This module is that one place, so web/api/Teams/chatbot/VSCode stay
in lock-step.

See MULTITENANT_MSYSTEM_ROUTING_ARCHITECTURE.md and MULTITENANT_MASTER_PLAN.md §4.
"""

from __future__ import annotations

import os

GATEWAY_API = os.environ.get("AG_PUBLIC_API_BASE", "https://api.storeglance.com")

MASTER_TENANT_LOGO_PATH = "/api/master/tenant-config/TenantLogo"


def normalize_domain(domain: str) -> str:
    """Lowercase host, strip any port — the form the master resolves tenants by."""
    return (domain or "").split(":")[0].strip().lower()


def gateway_base_from_request(request) -> str:
    """Reconstruct ``{scheme}://{host}`` of the gateway from the incoming request.

    Agent.API is reached through the same gateway as the master (e.g.
    ``uat-api.storeglance.com/api/agent/…``), so the request host IS the right base to
    build the master logo URL against — environment-correct (uat vs prod) WITHOUT
    relying on AG_PUBLIC_API_BASE being set, exactly how agent.web derives it from
    the served host. Falls back to the configured GATEWAY_API. NB: this is the API
    host, NOT the tenant domain — the tenant is carried separately as ?domain=.
    """
    try:
        host = (request.headers.get("x-forwarded-host")
                or request.headers.get("host") or "").split(",")[0].strip()
        if host:
            hostonly = host.split(":")[0].lower()
            is_local = (hostonly in ("127.0.0.1", "localhost", "::1")
                        or hostonly.startswith("127."))
            xfp = (request.headers.get("x-forwarded-proto") or "").split(",")[0].strip().lower()
            if is_local:
                proto = xfp or getattr(getattr(request, "url", None), "scheme", "") or "http"
            else:
                proto = "https"
            return f"{proto}://{host}"
    except Exception:
        pass
    return GATEWAY_API


def master_logo_url(domain: str, *, has_logo: bool = True, gateway_base: str = None) -> str:
    """Absolute URL to the tenant's logo on the MSystems master, or "" when the
    tenant has no custom logo (callers fall back to a neutral/local asset).

    `has_logo` lets callers pass the truthiness of `tenant_profile.logo_url` so the
    URL is only emitted when an image actually exists — mirroring agent.web's
    `hasLogo()` gate. The raw relative path itself is never returned.
    `gateway_base` overrides the host (pass gateway_base_from_request(request) so
    the URL points at the SAME environment that served the request).
    """
    d = normalize_domain(domain)
    if not d or not has_logo:
        return ""
    base = (gateway_base or GATEWAY_API).rstrip("/")
    return f"{base}{MASTER_TENANT_LOGO_PATH}?domain={d}"
