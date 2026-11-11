"""
Dynamic integration runtime — the generic HTTP executor + per-session provider.
═══════════════════════════════════════════════════════════════════════════════
One executor serves every API: load the op (tenant-routed), resolve the tenant's
credential (user→app→org, exactly like Jira/M365), build the request from the
operation's `request_map`, run it through the egress guard, call, normalize.

Dynamic tools are ORDINARY ToolRegistry entries carrying their real
method-derived ToolPolicy, so the existing ToolExecutor gate enforces
approval/dedup/audit (a dynamic DELETE is gated like outlook_send). They register
PER SESSION (never the shared startup registry) so tenant A's tool names never
leak into tenant B — the brain-tool pattern. See plan §5.2–5.5.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import re
import socket
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urljoin, urlsplit

from core.integrations import auth
from core.integrations.model import policy_for_method
from core.integrations.store import IntegrationStore

logger = logging.getLogger(__name__)

MAX_RESP_BYTES = 256 * 1024
DEFAULT_TIMEOUT = 30
MAX_REDIRECTS = 3
_CATALOG_TTL = 60.0


_pin_tls = threading.local()
_pin_installed = False
_pin_lock = threading.Lock()


def _install_dns_pin():
    global _pin_installed
    with _pin_lock:
        if _pin_installed:
            return
        _orig = socket.getaddrinfo

        def _pinned(host, *args, **kwargs):
            pin = getattr(_pin_tls, "map", None)
            if pin and host in pin:
                host = pin[host]
            return _orig(host, *args, **kwargs)

        socket.getaddrinfo = _pinned
        _pin_installed = True


@contextmanager
def _pin_dns(host: str, ip: Optional[str]):
    """Pin `host` → `ip` for the current thread for the duration of the block."""
    if not ip:
        yield
        return
    _install_dns_pin()
    prev = getattr(_pin_tls, "map", None)
    _pin_tls.map = {**(prev or {}), host: ip}
    try:
        yield
    finally:
        _pin_tls.map = prev


def principal_from_ctx(ctx) -> Tuple[Optional[int], Optional[str]]:
    """(user_id:int|None, app_id:str|None) from a UserContext, the SAME way the
    hand-coded tools derive them ('user_5' → 5; app_id flows through)."""
    if ctx is None:
        return None, None
    uid = None
    raw = getattr(ctx, "user_id", None)
    if raw:
        tail = str(raw).split("_")[-1]
        uid = int(tail) if tail.isdigit() else None
    return uid, getattr(ctx, "app_id", None)


class EgressBlocked(Exception):
    """The egress guard refused the request (SSRF defense)."""


def _host_allowed(host: str, base_url: str, egress_allow: List[str]) -> bool:
    allow = {urlsplit(base_url).hostname or ""}
    allow |= {h.lower() for h in (egress_allow or [])}
    return host.lower() in allow


def egress_check(url: str, *, base_url: str, egress_allow: List[str],
                 allow_private: bool) -> List[str]:
    """Validate an outbound URL. Returns the resolved IPs (for optional pinning)
    or raises EgressBlocked. Default-deny internal hosts; cloud-metadata never
    allowed; only the integration's own declared host(s)."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme not in ("http", "https"):
        raise EgressBlocked(f"scheme not allowed: {parts.scheme}")
    if parts.scheme == "http" and not allow_private:
        raise EgressBlocked("plain http is only allowed for opted-in private APIs")
    if not host:
        raise EgressBlocked("no host")
    if not _host_allowed(host, base_url, egress_allow):
        raise EgressBlocked(f"host '{host}' is not in this integration's allowlist")
    try:
        addrs = sorted({ai[4][0] for ai in socket.getaddrinfo(host, None)})
    except Exception as e:
        raise EgressBlocked(f"DNS resolution failed for {host}: {e}")
    for a in addrs:
        try:
            ip = ipaddress.ip_address(a)
        except ValueError:
            continue
        if getattr(ip, "ipv4_mapped", None) is not None:
            ip = ip.ipv4_mapped
        if _is_metadata(ip):
            raise EgressBlocked("target resolves to the link-local/metadata range")
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified
                ) and not allow_private:
            raise EgressBlocked(f"target resolves to a private/internal IP ({a})")
    return addrs


def _is_metadata(ip) -> bool:
    """Cloud metadata endpoints — blocked even for opted-in private APIs.
    Covers IPv4 link-local (169.254.0.0/16, AWS/GCP/Azure IMDS) and the IPv6
    AWS IMDS address fd00:ec2::254."""
    try:
        if ip.version == 4:
            return ip in ipaddress.ip_network("169.254.0.0/16")
        return ip.is_link_local or ip == ipaddress.ip_address("fd00:ec2::254")
    except Exception:
        return False


@dataclass
class _Req:
    method: str
    url: str
    headers: Dict[str, str] = field(default_factory=dict)
    params: Dict[str, Any] = field(default_factory=dict)
    json_body: Any = None
    basic_auth: Optional[Tuple[str, str]] = None


def _apply_auth(req: _Req, auth_config: dict, cred_config: dict, secret: str):
    """Attach the integration's credential to the request per its auth recipe.

    The auth header/param is set HERE (after host validation) so it is only ever
    sent to the integration's own host — never carried across a redirect.
    """
    where = (auth_config or {}).get("in")
    name = (auth_config or {}).get("name")
    if where == "header" and name:
        fmt = auth_config.get("fmt", "{token}")
        req.headers[name] = fmt.format(token=secret, key=secret)
    elif where == "query" and name:
        req.params[name] = secret
    elif where == "basic":
        req.basic_auth = ((cred_config or {}).get("username", ""), secret)


def _apply_hmac(req: _Req, auth_config: dict, secret: str):
    """Sign the request with HMAC and emit the signature header.

    Default recipe: HMAC(secret, "{METHOD}\\n{path}\\n{canonical-json-body}").
    `auth_config` may override alg ('sha256'|'sha1'|'sha512'), the header name,
    and an output `fmt` (e.g. 'HMAC {sig}'). Deterministic body serialization
    (sorted keys) so the signature is reproducible by the server.
    """
    import hashlib
    import hmac as _hmac
    alg = (auth_config.get("alg") or "sha256").lower()
    header = auth_config.get("header") or "X-Signature"
    body = (json.dumps(req.json_body, separators=(",", ":"), sort_keys=True)
            if req.json_body is not None else "")
    path = urlsplit(req.url).path
    msg = f"{req.method}\n{path}\n{body}"
    digestmod = getattr(hashlib, alg, hashlib.sha256)
    sig = _hmac.new(secret.encode(), msg.encode(), digestmod).hexdigest()
    req.headers[header] = auth_config.get("fmt", "{sig}").format(sig=sig)


def _parse_bundle(secret) -> Dict[str, str]:
    """The login_grant / oauth2_password secret is a JSON {access_token,
    refresh_token} produced by the connect-time exchange (never the password).
    Tolerate a bare string for resilience (treat as an access token)."""
    if not secret:
        return {}
    if isinstance(secret, dict):
        return secret
    try:
        d = json.loads(secret)
        return d if isinstance(d, dict) else {"access_token": str(secret)}
    except Exception:
        return {"access_token": str(secret)}


def _default_headers(op_or_integ: dict) -> Dict[str, str]:
    """Static headers configured on the integration (e.g. X-Tenant-Domain) — sent
    on every op call, the login exchange, and token refresh."""
    dh = (op_or_integ or {}).get("default_headers") or {}
    if isinstance(dh, str):
        try:
            dh = json.loads(dh)
        except Exception:
            dh = {}
    return {str(k): str(v) for k, v in dh.items() if v} if isinstance(dh, dict) else {}


def _fill_template(mapping: dict, values: dict) -> dict:
    """Replace '{placeholder}' values in a body_map with concrete values."""
    out = {}
    for k, v in (mapping or {}).items():
        out[k] = values.get(v[1:-1], "") if (isinstance(v, str) and v.startswith("{")
                                             and v.endswith("}")) else v
    return out


def _json_path_get(obj, path: str):
    """Minimal '$.a.b' lookup into a parsed JSON body."""
    if not path:
        return None
    node = obj
    for part in path.replace("$", "", 1).strip(".").split("."):
        if part == "":
            continue
        if isinstance(node, dict):
            node = node.get(part)
        else:
            return None
    return node


def _extract_token(obj, path):
    """Read a token from a response. `path` is a single '$.a.b' OR a LIST of
    candidates — many APIs wrap the token in an envelope (`{data:{accessToken}}`)
    and declare no response schema, so discovery hands several candidates and we
    take the first that yields a non-empty string."""
    for p in (path if isinstance(path, (list, tuple)) else [path]):
        v = _json_path_get(obj, p)
        if isinstance(v, str) and v:
            return v
    return None


def _jwt_exp(token: str):
    """The `exp` (unix seconds) from a JWT, or None. NO signature check — used only
    to size the local token cache TTL, never for trust."""
    try:
        import base64
        parts = (token or "").split(".")
        if len(parts) < 2:
            return None
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        data = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8", "replace"))
        exp = data.get("exp")
        return float(exp) if exp else None
    except Exception:
        return None


def login_exchange(json_post, auth_config: dict, username: str, password: str,
                   *, allow_private: bool = False, headers: Optional[dict] = None) -> Dict[str, str]:
    """Connect-time: exchange username+password for a token bundle via the API's
    own login endpoint. Returns {access_token, refresh_token} so only the refresh
    token is stored — the password is discarded. `headers` are the integration's
    static headers (e.g. X-Tenant-Domain) the login also needs. `json_post(url,
    body, headers) -> dict` is injectable (defaults to a guarded requests.post)."""
    login_url = (auth_config or {}).get("login_url")
    if not login_url:
        raise RuntimeError("login_grant integration has no login_url configured.")
    egress_check(login_url, base_url=login_url, egress_allow=[], allow_private=allow_private)
    body = _fill_template(auth_config.get("body_map") or {},
                          {"username": username, "password": password})
    if json_post is not None:
        resp = json_post(login_url, body, headers or {})
    else:
        import requests
        h = {"Accept": "application/json", **(headers or {})}
        r = requests.post(login_url, json=body, timeout=DEFAULT_TIMEOUT, headers=h)
        resp = r.json() if r.content else {}
    access = _extract_token(resp, auth_config.get("token_path") or "$.accessToken")
    refresh = _extract_token(resp, auth_config.get("refresh_token_path") or "$.refreshToken")
    if not access and not refresh:
        raise RuntimeError("login did not return a token (check the credentials).")
    return {"access_token": access or "", "refresh_token": refresh or ""}


def prepare_connect_fields(db, provider: str, fields: dict, *, json_post=None) -> dict:
    """Connect-time transform for `login_grant` providers: exchange the entered
    username+password for a token bundle and store ONLY the bundle (access+refresh)
    in the secret slot — the password is never persisted. All other providers pass
    through unchanged. Raises on a failed login so the UI can surface it."""
    if not provider or not str(provider).startswith("api_"):
        return fields
    integ = IntegrationStore(db).get_integration_by_provider(provider)
    if not integ or integ.get("auth_type") != "login_grant":
        return fields
    ac = integ.get("auth_config") or {}
    if isinstance(ac, str):
        ac = json.loads(ac)
    f = dict(fields or {})
    username, password = f.get("username", ""), f.get("password", "")
    if not username or not password:
        return f
    bundle = login_exchange(json_post, ac, username, password,
                            allow_private=bool(integ.get("allow_private")),
                            headers=_default_headers(integ))
    f["password"] = json.dumps(bundle)
    return f


class DynamicHttpExecutor:
    def __init__(self, db, connections, embeddings=None, http=None, store=None,
                 token_fetch=None, brain_ingest=None, json_post=None):
        self.db = db
        self.connections = connections
        self.embeddings = embeddings
        self._http = http
        self._token_fetch = token_fetch
        self._json_post = json_post
        self._brain_ingest = brain_ingest
        self._store = store if store is not None else IntegrationStore(db)
        self._op_cache: Dict[Tuple[str, str], Tuple[float, Optional[dict]]] = {}
        self._token_cache: Dict[Tuple[str, str, str], Tuple[str, float]] = {}

    def call(self, tool_name: str, args: dict, *, ctx=None,
             pre_gated: bool = True) -> str:
        """Execute one dynamic tool call. `pre_gated` True means the outer
        ToolExecutor already ran the gate for this op's policy (direct mode);
        meta-mode would pass False and gate here (Phase 3b)."""
        op = self._load(tool_name)
        if op is None:
            return (f"Error: integration tool '{tool_name}' is not available "
                    f"(unknown, disabled, or its integration is inactive).")
        cred_config, secret, uid, conn_id = {}, "", None, None
        if op.get("auth_provider"):
            uid, app_id = principal_from_ctx(ctx)
            cred = self.connections.resolve_or_none(
                op["auth_provider"], user_id=uid, app_id=app_id)
            if cred is None:
                return (f"Error: '{op['slug']}' is not connected for you. Ask an "
                        f"admin to add a connection for '{op['auth_provider']}'.")
            cred_config, secret = cred.config, cred.secret
            conn_id = getattr(cred, "connection_id", None)
        try:
            req = self._build_request(op, args or {})
        except KeyError as e:
            return f"Error: missing required parameter {e} for {tool_name}."
        try:
            self._apply_auth_for(op, req, cred_config, secret, principal=uid,
                                 connection_id=conn_id)
        except Exception as e:
            return f"Error: could not prepare authentication for {tool_name}: {e}"
        rl = self._rate_limited(op)
        if rl is not None:
            return rl
        slug = op.get("slug", "")
        t0 = time.time()
        try:
            status, body = self._send(op, req)
            if status == 401 and auth.resolve(
                    op.get("auth_type"), op.get("security_scheme")).is_token_based():
                self._invalidate_token(op)
                self._apply_auth_for(op, req, cred_config, secret, force_refresh=True,
                                     principal=uid, connection_id=conn_id)
                status, body = self._send(op, req)
        except EgressBlocked as e:
            _metric("blocked", slug, op.get("operation_id", ""), time.time() - t0, 0)
            return f"Error: request blocked by egress policy — {e}."
        except Exception as e:
            _metric("error", slug, op.get("operation_id", ""), time.time() - t0, 0)
            return f"Error calling {tool_name}: {e}"
        _metric(str(status), slug, op.get("operation_id", ""), time.time() - t0, len(body or b""))
        return self._normalize(status, body, op)

    def _rate_limited(self, op: dict) -> Optional[str]:
        """Per-op + per-integration outbound call caps. Returns an error string
        when over the cap, else None. Fail-open (a limiter outage never blocks)."""
        try:
            from core.rate_limit import check_rate
            rl = op.get("rate_limit") or {}
            if isinstance(rl, str):
                rl = json.loads(rl)
            slug = op.get("slug", "")
            op_rpm = int(rl.get("per_minute") or 0)
            if op_rpm <= 0:
                from core.config import app_config
                op_rpm = app_config.get_int("agent.integration_default_rpm", 120)
            ok, retry = check_rate(f"apiop:{slug}:{op['tool_name']}", op_rpm, 60)
            if not ok:
                return (f"Error: rate limit reached for {op['tool_name']} "
                        f"(retry in ~{retry}s).")
            integ_rpm = int(rl.get("integration_per_minute") or 0)
            if integ_rpm > 0:
                ok, retry = check_rate(f"apiint:{slug}", integ_rpm, 60)
                if not ok:
                    return (f"Error: rate limit reached for the '{slug}' API "
                            f"(retry in ~{retry}s).")
        except Exception as e:
            logger.debug("rate-limit check skipped: %s", e)
        return None

    def _apply_auth_for(self, op: dict, req: _Req, cred_config: dict, secret: str,
                        *, force_refresh: bool = False, principal=None, connection_id=None):
        if not op.get("auth_provider"):
            return
        strategy = auth.resolve(op.get("auth_type"), op.get("security_scheme"))
        strategy.apply(self, op, req, cred_config, secret, principal=principal,
                       force_refresh=force_refresh, connection_id=connection_id)

    @staticmethod
    def _cred_fingerprint(cred_config: dict, secret: str) -> str:
        """A short, stable digest of the RESOLVED credential (client_id + secret +
        token_url). The OAuth token cache is keyed on this so two principals
        (different apps/users) holding distinct client credentials for the SAME
        `api_{slug}` provider never share a token — otherwise the first caller's
        token would be served as everyone else's identity within the tenant."""
        import hashlib
        cc = cred_config or {}
        material = "\x00".join((str(cc.get("client_id", "")), secret or "",
                                str(cc.get("token_url", ""))))
        return hashlib.sha256(material.encode()).hexdigest()[:16]

    def _oauth_token(self, op: dict, cred_config: dict, secret: str,
                     auth_config: dict, *, force_refresh: bool = False) -> str:
        key = (_tid(), op["auth_provider"], self._cred_fingerprint(cred_config, secret))
        now = time.time()
        if not force_refresh:
            hit = self._token_cache.get(key)
            if hit and hit[1] - now > 30:
                return hit[0]
        token_url = (cred_config.get("token_url") or auth_config.get("token_url") or "").strip()
        if not token_url:
            raise RuntimeError("OAuth2 token_url is not configured")
        egress_check(token_url, base_url=token_url, egress_allow=[],
                     allow_private=bool(op.get("allow_private")))
        data = {"grant_type": "client_credentials",
                "client_id": cred_config.get("client_id", ""), "client_secret": secret}
        if auth_config.get("scopes"):
            data["scope"] = " ".join(auth_config["scopes"])
        token, ttl = self._fetch_token(token_url, data)
        self._token_cache[key] = (token, now + max(int(ttl or 0), 60))
        return token

    def _fetch_token(self, token_url: str, data: dict) -> Tuple[str, int]:
        if self._token_fetch is not None:
            return self._token_fetch(token_url, data)
        import requests
        r = requests.post(token_url, data=data, timeout=DEFAULT_TIMEOUT,
                          headers={"Accept": "application/json"})
        j = r.json() if r.content else {}
        tok = j.get("access_token")
        if not tok:
            raise RuntimeError(f"token endpoint returned no access_token (HTTP {r.status_code})")
        return tok, int(j.get("expires_in", 3600))

    def _invalidate_token(self, op: dict):
        tid, prov = _tid(), op.get("auth_provider")
        for k in [k for k in self._token_cache if k[0] == tid and k[1] == prov]:
            self._token_cache.pop(k, None)

    def _json_post_call(self, url: str, body: dict, headers: Optional[dict] = None) -> dict:
        if self._json_post is not None:
            return self._json_post(url, body, headers or {})
        import requests
        h = {"Accept": "application/json", **(headers or {})}
        r = requests.post(url, json=body, timeout=DEFAULT_TIMEOUT, headers=h)
        return r.json() if r.content else {}

    def _login_grant_token(self, op, cred_config, secret, auth_config, *,
                           principal=None, force_refresh=False, connection_id=None) -> str:
        """Access token for a login_grant integration. The stored secret is a JSON
        bundle {access_token, refresh_token} from the connect-time exchange (no
        password). Refreshes via the API's refresh endpoint and PERSISTS the rotated
        bundle, so a single-use refresh token survives restarts."""
        # Key on the STABLE credential identity (connection_id), not the secret:
        # the login_grant bundle rotates on every refresh, so keying on it would
        # change the cache key each call and defeat the cache (refresh-per-call).
        cred_key = connection_id if connection_id is not None else \
            self._cred_fingerprint(cred_config, secret)
        key = (_tid(), op["auth_provider"], cred_key, principal)
        now = time.time()
        if not force_refresh:
            hit = self._token_cache.get(key)
            if hit and hit[1] - now > 30:
                return hit[0]
        bundle = _parse_bundle(secret)
        refresh = auth_config.get("refresh") or {}
        if bundle.get("refresh_token") and refresh.get("url"):
            ru = refresh["url"]
            egress_check(ru, base_url=ru, egress_allow=[],
                         allow_private=bool(op.get("allow_private")))
            body = _fill_template(refresh.get("body_map") or {}, {
                "access_token": bundle.get("access_token", ""),
                "refresh_token": bundle.get("refresh_token", "")})
            resp = self._json_post_call(ru, body, _default_headers(op))
            new_access = _extract_token(
                resp, refresh.get("token_path") or auth_config.get("token_path") or "$.accessToken")
            if not new_access:
                raise RuntimeError("login refresh returned no access token — reconnect required.")
            new_refresh = _extract_token(
                resp, refresh.get("refresh_token_path") or auth_config.get("refresh_token_path")
                or "$.refreshToken") or bundle.get("refresh_token")
            if connection_id and new_refresh and new_refresh != bundle.get("refresh_token"):
                try:
                    self.connections.update(connection_id, fields={
                        "password": json.dumps({"access_token": new_access,
                                                "refresh_token": new_refresh})})
                except Exception as e:
                    logger.debug("login_grant: rotate-persist failed: %s", e)
            self._token_cache[key] = (new_access, _jwt_exp(new_access) or now + 50 * 60)
            return new_access
        acc = bundle.get("access_token")
        if acc and (_jwt_exp(acc) or now + 300) - now > 30:
            self._token_cache[key] = (acc, _jwt_exp(acc) or now + 50 * 60)
            return acc
        raise RuntimeError("login_grant credential needs reconnect (no usable token).")

    def _oauth_password_token(self, op, cred_config, secret, auth_config, *,
                              principal=None, force_refresh=False, connection_id=None) -> str:
        """OAuth2 password grant: connect-time minted a {access,refresh} bundle;
        runtime refreshes with grant_type=refresh_token (per-user)."""
        # Stable key on connection_id (the bundle rotates — see _login_grant_token).
        cred_key = connection_id if connection_id is not None else \
            self._cred_fingerprint(cred_config, secret)
        key = (_tid(), op["auth_provider"], cred_key, principal)
        now = time.time()
        if not force_refresh:
            hit = self._token_cache.get(key)
            if hit and hit[1] - now > 30:
                return hit[0]
        bundle = _parse_bundle(secret)
        token_url = (cred_config.get("token_url") or auth_config.get("token_url") or "").strip()
        if token_url and bundle.get("refresh_token"):
            egress_check(token_url, base_url=token_url, egress_allow=[],
                         allow_private=bool(op.get("allow_private")))
            tok, ttl = self._fetch_token(token_url, {
                "grant_type": "refresh_token", "refresh_token": bundle["refresh_token"],
                "client_id": cred_config.get("client_id", "")})
            self._token_cache[key] = (tok, now + max(int(ttl or 0), 60))
            return tok
        acc = bundle.get("access_token")
        if acc and (_jwt_exp(acc) or now + 300) - now > 30:
            return acc
        raise RuntimeError("oauth2_password credential needs reconnect.")

    def _build_request(self, op: dict, args: dict) -> _Req:
        if op.get("spec_kind") == "mcp":
            from core.integrations.mcp import call_envelope
            req = _Req(method="POST", url=op["base_url"].rstrip("/"))
            req.json_body = call_envelope(op["operation_id"], args or {})
            return req
        rmap = op.get("request_map") or {}
        if isinstance(rmap, str):
            rmap = json.loads(rmap)
        path = op["path_template"]
        for name in rmap.get("path", []):
            if name not in args:
                raise KeyError(name)
            path = path.replace("{%s}" % name, quote(str(args[name]), safe=""))
        leftover = re.search(r"{([^}/]+)}", path)
        if leftover:
            raise KeyError(leftover.group(1))
        url = op["base_url"].rstrip("/") + path

        req = _Req(method=op["http_method"], url=url)
        req.headers.update(_default_headers(op))
        for name in rmap.get("query", []):
            if name in args and args[name] is not None:
                req.params[name] = args[name]
        for name in rmap.get("header", []):
            if name in args and args[name] is not None:
                req.headers[name] = str(args[name])
        if rmap.get("has_body") and "body" in args:
            req.json_body = args["body"]
        return req

    def _send(self, op: dict, req: _Req) -> Tuple[int, bytes]:
        egress_allow = op.get("egress_allow") or []
        if isinstance(egress_allow, str):
            egress_allow = json.loads(egress_allow)
        base_url, allow_private = op["base_url"], bool(op.get("allow_private"))
        url = req.url
        for _ in range(MAX_REDIRECTS + 1):
            addrs = egress_check(url, base_url=base_url, egress_allow=egress_allow,
                                 allow_private=allow_private)
            pin_ip = addrs[0] if addrs else None
            status, headers, body = self._do_http(req.method, url, req, pin_ip)
            if status in (301, 302, 303, 307, 308) and headers.get("location"):
                url = urljoin(url, headers["location"])
                method = "GET" if status == 303 else req.method
                req = _Req(method=method, headers=req.headers,
                           params=req.params, json_body=req.json_body,
                           basic_auth=req.basic_auth, url=url)
                continue
            return status, body
        raise EgressBlocked("too many redirects")

    def _do_http(self, method: str, url: str, req: _Req,
                 pin_ip: Optional[str] = None) -> Tuple[int, dict, bytes]:
        if self._http is not None:
            return self._http(method, url, req)
        import requests
        host = urlsplit(url).hostname or ""
        with _pin_dns(host, pin_ip):
            r = requests.request(
                method, url, headers=req.headers or None, params=req.params or None,
                json=req.json_body if req.json_body is not None else None,
                auth=req.basic_auth, timeout=DEFAULT_TIMEOUT, allow_redirects=False,
                stream=True)
            body = r.raw.read(MAX_RESP_BYTES + 1, decode_content=True) or b""
        return r.status_code, {k.lower(): v for k, v in r.headers.items()}, body

    def _normalize(self, status: int, body: bytes, op: dict) -> str:
        spec_kind = op.get("spec_kind")
        rt = op.get("response_transform") or {}
        if isinstance(rt, str):
            try:
                rt = json.loads(rt)
            except Exception:
                rt = {}

        if spec_kind == "mcp":
            from core.integrations.mcp import parse_mcp_result
            try:
                text = parse_mcp_result(json.loads(body.decode("utf-8", "replace")))
            except Exception:
                text = body[:MAX_RESP_BYTES].decode("utf-8", "replace")
            return text[:MAX_RESP_BYTES] + ("\n…[truncated]" if len(text) > MAX_RESP_BYTES else "")

        full_text = body.decode("utf-8", "replace")

        shaped = self._apply_transform(full_text, rt) if rt else None
        display = shaped if shaped is not None else full_text

        if (rt.get("sink") == "brain" and self._brain_ingest is not None
                and op.get("side_effect") == "read"):
            try:
                ref = self._brain_ingest(
                    f"{op.get('slug')}:{op.get('operation_id')}", full_text)
                if ref:
                    return (f"HTTP {status}. Large response ingested into the knowledge "
                            f"base (ref: {ref}). Query it with brain_search / brain_ask.")
            except Exception as e:
                logger.debug("brain sink failed, returning inline: %s", e)

        truncated = len(display.encode("utf-8")) > MAX_RESP_BYTES
        text = display[:MAX_RESP_BYTES]
        if shaped is None:
            try:
                text = json.dumps(json.loads(text), indent=2, ensure_ascii=False)[:MAX_RESP_BYTES]
            except Exception:
                pass
        head = f"HTTP {status}" + (" (error)" if status >= 400 else "")
        return f"{head}\n{text}" + ("\n…[response truncated]" if truncated else "")

    @staticmethod
    def _apply_transform(text: str, rt: dict) -> Optional[str]:
        """Minimal JSONPath-ish shaping: `path` (dot/`[i]`) selects a subtree;
        `max_items` trims a list; `max_chars` caps. Returns None to fall back to
        the raw response on any mismatch (never raises)."""
        if not any(k in rt for k in ("path", "max_items", "max_chars")):
            return None
        try:
            data = json.loads(text)
        except Exception:
            return text[: int(rt["max_chars"])] if rt.get("max_chars") else None
        node = data
        for part in [p for p in (rt.get("path") or "").replace("]", "").split(".") if p != ""]:
            if "[" in part:
                key, idx = part.split("[", 1)
                if key:
                    node = node.get(key) if isinstance(node, dict) else None
                if isinstance(node, list) and idx.isdigit():
                    node = node[int(idx)] if int(idx) < len(node) else None
            else:
                node = node.get(part) if isinstance(node, dict) else None
            if node is None:
                break
        if isinstance(node, list) and rt.get("max_items"):
            node = node[: int(rt["max_items"])]
        out = json.dumps(node, indent=2, ensure_ascii=False)
        if rt.get("max_chars"):
            out = out[: int(rt["max_chars"])]
        return out

    def _load(self, tool_name: str) -> Optional[dict]:
        tid = _tid()
        key = (tid, tool_name)
        hit = self._op_cache.get(key)
        if hit and time.time() - hit[0] < _CATALOG_TTL:
            return hit[1]
        op = self._store.get_op_for_execution(tool_name)
        self._op_cache[key] = (time.time(), op)
        return op

    def invalidate(self, tid: Optional[str] = None):
        tid = tid or _tid()
        for k in [k for k in self._op_cache if k[0] == tid]:
            self._op_cache.pop(k, None)


def _tid() -> str:
    try:
        from core.tenant_context import current_tenant_id
        return current_tenant_id()
    except Exception:
        return "default"


def _keyword_pick(text: str, available, k: int) -> set:
    """Slug/operation token overlap with the turn text — the no-embeddings
    fallback. Only considers dynamic op names (those carrying the '__'
    namespace), so builtin tools are left to the keyword GROUPS path."""
    low = (text or "").lower()
    if not low:
        return set()
    scored = []
    for name in available:
        if "__" not in name:
            continue
        slug, _, op = name.partition("__")
        toks = {t for t in (slug.split("_") + op.split("_")) if len(t) > 2}
        hits = sum(1 for t in toks if t in low)
        if hits:
            scored.append((hits, name))
    scored.sort(reverse=True)
    return {name for _h, name in scored[:k]}


def _metric(status: str, slug: str, op: str, dt: float, nbytes: int):
    """Emit per-call observability (counter + duration + response size)."""
    try:
        from core.telemetry import telemetry
        telemetry.incr("integration_calls_total", slug=slug, op=op, status=status)
        telemetry.observe("integration_duration_seconds", dt, slug=slug, op=op)
        if nbytes:
            telemetry.observe("integration_response_bytes", float(nbytes), slug=slug)
    except Exception:
        pass


class IntegrationToolProvider:
    """Registers granted+enabled dynamic ops into a per-session registry, and
    owns the per-tenant catalog cache. Holds NO tenant state — resolves the
    active tenant on each call (so one instance serves all tenants safely)."""

    def __init__(self, connections, embeddings, executor: DynamicHttpExecutor,
                 store: Optional[IntegrationStore] = None, ttl: float = _CATALOG_TTL):
        self.connections = connections
        self.embeddings = embeddings
        self.executor = executor
        self._store = store if store is not None else executor._store
        self._ttl = ttl
        self._cache: Dict[Tuple[str, str, str, bool], Tuple[float, list]] = {}

    def _enabled(self) -> bool:
        try:
            from core.config import app_config
            return app_config.get_bool("agent.dynamic_integrations", False)
        except Exception:
            return False

    def _catalog(self, app_id: str, persona_id: str, embed_safe_only: bool) -> list:
        key = (_tid(), app_id or "", persona_id or "", embed_safe_only)
        hit = self._cache.get(key)
        if hit and time.time() - hit[0] < self._ttl:
            return hit[1]
        rows = self._store.catalog_for_principal(
            app_id=app_id or "", persona_id=persona_id or "",
            embed_safe_only=embed_safe_only)
        self._cache[key] = (time.time(), rows)
        return rows

    def invalidate(self, tid: Optional[str] = None):
        tid = tid or _tid()
        for k in [k for k in self._cache if k[0] == tid]:
            self._cache.pop(k, None)
        self.executor.invalidate(tid)

    def register_session_direct(self, user_registry, user_context,
                                *, persona_id: str = "", embed_safe_only: bool = False) -> int:
        """Register the granted+enabled ops as direct tools on the session
        registry. No-op when the flag is off. Returns the count registered."""
        if not self._enabled():
            return 0
        _uid, app_id = principal_from_ctx(user_context)
        rows = self._catalog(str(app_id or ""), str(persona_id or ""), embed_safe_only)
        n = 0
        for op in rows:
            tool_name = op["tool_name"]
            user_registry.register(
                tool_name, op.get("description") or tool_name, parameters={},
                callback=self._make_callback(tool_name, user_context),
                json_schema=op.get("parameters_schema") or {"type": "object", "properties": {}},
                policy=policy_for_method(op["http_method"], op.get("requires_approval")),
            )
            n += 1
        return n

    def _make_callback(self, tool_name: str, ctx):
        def _cb(**kwargs):
            return self.executor.call(tool_name, kwargs, ctx=ctx, pre_gated=True)
        return _cb

    def granted_tool_names(self, app_id: str, persona_id: str = "",
                           embed_safe_only: bool = False) -> set:
        """The op tool_names this principal may use — for whitelist expansion."""
        return {op["tool_name"] for op in
                self._catalog(str(app_id or ""), str(persona_id or ""), embed_safe_only)}

    def make_picker(self, user_context, *, persona_id: str = "",
                    embed_safe_only: bool = False, k: int = 8):
        """A dynamic_picker(text, available) for tool_selection (Phase 3 wiring).
        Embedding-ranks granted ops; returns the turn-relevant tool_names."""
        _uid, app_id = principal_from_ctx(user_context)
        app_id, persona_id = str(app_id or ""), str(persona_id or "")

        def _pick(text: str, available) -> set:
            emb = self.embeddings
            if emb is not None and getattr(emb, "available", False):
                try:
                    vec = emb.embed_one(text)
                    names = self._store.search_catalog(
                        vec, app_id=app_id, persona_id=persona_id, k=k,
                        embed_safe_only=embed_safe_only)
                    hits = set(names) & set(available)
                    if hits:
                        return hits
                except Exception:
                    pass
            return _keyword_pick(text, available, k)
        return _pick


def verify_integration_credential(provider: str, config: dict, secret: str, *,
                                  connections=None, timeout: int = 10) -> Tuple[bool, str]:
    """Probe an integration's declared GET op with the supplied (unsaved) cred."""
    if connections is None or getattr(connections, "_db", None) is None:
        return True, ("Saved. Connection couldn't be live-tested here (no DB "
                      "context); it will be exercised on first use.")
    store = IntegrationStore(connections._db)
    probe = store.first_probe_op(provider)
    if not probe:
        return True, "Saved. No GET operation is available to test this API."
    rmap = probe.get("request_map") or {}
    if isinstance(rmap, str):
        rmap = json.loads(rmap)
    if rmap.get("path"):
        return True, "Saved. The API has no parameter-free probe; verified on first use."
    ex = DynamicHttpExecutor(connections._db, connections)
    probe["auth_provider"] = provider
    req = _Req(method=probe["http_method"],
               url=probe["base_url"].rstrip("/") + probe["path_template"])
    pschema = probe.get("parameters_schema") or {}
    if isinstance(pschema, str):
        pschema = json.loads(pschema)
    props = pschema.get("properties") or {}
    for hdr_name in (rmap.get("header") or []):
        hdr_schema = props.get(hdr_name) or {}
        default = hdr_schema.get("default")
        if default is not None:
            req.headers[hdr_name] = str(default)
    try:
        ex._apply_auth_for(probe, req, config or {}, secret or "")
    except Exception as e:
        return False, f"Could not prepare authentication: {e}"
    try:
        status, _ = ex._send(probe, req)
    except EgressBlocked as e:
        return False, f"Blocked by egress policy: {e}"
    except Exception as e:
        return False, f"Connection failed: {e}"
    if 200 <= status < 300:
        return True, f"Connected ({status})."
    if status in (401, 403):
        return False, f"Authentication failed (HTTP {status}) — check the credential."
    return False, f"Probe returned HTTP {status}."

