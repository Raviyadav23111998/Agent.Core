"""
OpenAPI / Swagger → normalized Integration.
═══════════════════════════════════════════
Pure transformation (no DB, no network): a parsed spec dict becomes an
`Integration` with `IntegrationOperation`s whose `parameters_schema` is a single,
self-contained JSON Schema (path+query+header+body merged, $refs inlined) ready to
hand to the model verbatim, plus a `request_map` telling the executor where each
argument goes.

Supports OpenAPI 3.x directly and Swagger 2.0 via a light shape-coercion. MCP is a
separate source (Phase 6) and not handled here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlsplit

_PATH_PARAM_RE = re.compile(r"{([^}/]+)}")

from core.integrations.auth import (
    API_KEY, BEARER_STATIC, BASIC, CUSTOM,
    OAUTH2_CLIENT_CREDENTIALS, OAUTH2_PASSWORD, LOGIN_GRANT,
)
from core.integrations.model import (
    Integration, IntegrationOperation,
    side_effect_for_method, requires_approval_for_method, tool_name_for,
)

_HTTP_METHODS = ("get", "post", "put", "patch", "delete", "head")

_DROP_SCHEMA_KEYS = frozenset({
    "example", "examples", "xml", "externalDocs", "discriminator",
    "deprecated", "readOnly", "writeOnly", "nullable",
})

_MAX_REF_DEPTH = 40


class SpecError(ValueError):
    """The spec is malformed or unsupported."""


def detect_kind(spec: Dict[str, Any]) -> str:
    if str(spec.get("openapi", "")).startswith("3"):
        return "openapi"
    if str(spec.get("swagger", "")).startswith("2"):
        return "swagger"
    raise SpecError("Unrecognized spec: expected openapi 3.x or swagger 2.0.")


def coerce_to_openapi3(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Swagger 2.0 → an OpenAPI-3-shaped dict (enough for our normalizer).

    Converts: host+basePath+schemes → servers; securityDefinitions →
    components.securitySchemes; in:body param → requestBody; non-body params get
    their inline type wrapped under `schema`; definitions → components.schemas.
    $ref roots (#/definitions/… and #/components/…) are both resolvable later.
    """
    if detect_kind(spec) == "openapi":
        return spec

    out: Dict[str, Any] = {"openapi": "3.0.0", "info": spec.get("info", {}),
                           "paths": {}, "components": {}}
    scheme = (spec.get("schemes") or ["https"])[0]
    host = spec.get("host", "")
    base_path = spec.get("basePath", "") or ""
    if host:
        out["servers"] = [{"url": f"{scheme}://{host}{base_path}"}]
    elif base_path:
        out["servers"] = [{"url": base_path}]
    if spec.get("definitions"):
        out["components"]["schemas"] = spec["definitions"]
        out["definitions"] = spec["definitions"]
    if spec.get("securityDefinitions"):
        out["components"]["securitySchemes"] = spec["securityDefinitions"]

    for path, item in (spec.get("paths") or {}).items():
        new_item: Dict[str, Any] = {}
        shared = item.get("parameters", [])
        for method, op in item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            params, body = [], None
            for p in list(shared) + list(op.get("parameters", [])):
                if p.get("in") == "body":
                    body = p.get("schema")
                    body_required = p.get("required", False)
                elif p.get("in") == "formData":
                    body = body or {"type": "object", "properties": {}}
                    body["properties"][p["name"]] = _swagger_param_schema(p)
                    body_required = True
                else:
                    np = {"name": p.get("name"), "in": p.get("in"),
                          "required": p.get("required", False),
                          "description": p.get("description", ""),
                          "schema": _swagger_param_schema(p)}
                    params.append(np)
            new_op = {k: v for k, v in op.items() if k != "parameters"}
            new_op["parameters"] = params
            if body is not None:
                new_op["requestBody"] = {
                    "required": locals().get("body_required", False),
                    "content": {"application/json": {"schema": body}},
                }
            new_item[method] = new_op
        out["paths"][path] = new_item
    return out


def _swagger_param_schema(p: Dict[str, Any]) -> Dict[str, Any]:
    """A Swagger non-body parameter carries its type inline (not under `schema`)."""
    s = {k: p[k] for k in ("type", "format", "enum", "items", "default") if k in p}
    return s or {"type": "string"}


def _resolve_pointer(root: Dict[str, Any], ref: str) -> Any:
    if not ref.startswith("#/"):
        raise SpecError(f"Only local $refs are supported, got: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SpecError(f"Unresolved $ref: {ref}")
        node = node[part]
    return node


def _inline(node: Any, root: Dict[str, Any], stack: Tuple[str, ...] = (),
            depth: int = 0, unresolved: Optional[set] = None) -> Any:
    """Recursively inline local $refs; break cycles + cap depth.

    Fail-SOFT on a bad $ref: generated specs (esp. .NET/Swashbuckle) routinely
    emit $refs to schemas they never include — generic types (`T`, `byte`),
    fully-qualified names (`global::…`), or DTOs that weren't registered. A single
    dangling ref must NOT abort importing the whole API, so an unresolvable (or
    non-local) ref degrades to an open `{"type":"object"}` (accept-anything),
    exactly like the cycle case. The ref is recorded in `unresolved` so the import
    can warn the admin which schemas to fix."""
    if depth > _MAX_REF_DEPTH:
        return {}
    if isinstance(node, dict):
        if "$ref" in node:
            ref = node["$ref"]
            if ref in stack:
                return {"type": "object"}
            try:
                target = _resolve_pointer(root, ref)
            except SpecError:
                if unresolved is not None:
                    unresolved.add(ref)
                return {"type": "object"}
            return _inline(target, root, stack + (ref,), depth + 1, unresolved)
        return {k: _inline(v, root, stack, depth + 1, unresolved) for k, v in node.items()}
    if isinstance(node, list):
        return [_inline(v, root, stack, depth + 1, unresolved) for v in node]
    return node


def _clean_schema(node: Any) -> Any:
    """Drop OpenAPI-only / noisy keywords and x-* extensions from a JSON Schema."""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items()
                if k not in _DROP_SCHEMA_KEYS and not k.startswith("x-")}
    if isinstance(node, list):
        return [_clean_schema(v) for v in node]
    return node


def _prep(schema: Any, root: Dict[str, Any], unresolved: Optional[set] = None) -> Any:
    return _clean_schema(_inline(schema, root, unresolved=unresolved))


def extract_base_url(spec: Dict[str, Any], source_url: Optional[str] = None) -> str:
    """Absolute base URL for the API.

    OpenAPI permits RELATIVE server URLs (e.g. Swagger Petstore declares
    `servers: [{url: "/api/v3"}]`); they are relative to where the spec document
    itself is served. So when the declared server URL has no scheme — or no server
    is declared at all (OpenAPI default "/") — resolve it against `source_url`
    (the URL we fetched the spec from). Without this, a perfectly valid spec is
    rejected with "Unsupported scheme:".
    """
    servers = spec.get("servers") or []
    url = (servers[0].get("url") or "").strip() if servers else ""
    if url and urlsplit(url).scheme:
        return url.rstrip("/")
    if source_url and urlsplit(source_url).scheme:
        if url:
            return urljoin(source_url, url).rstrip("/")
        p = urlsplit(source_url)
        return f"{p.scheme}://{p.netloc}"
    if url:
        raise SpecError(
            f"Server URL {url!r} is relative and no source URL is available to "
            "resolve it. Provide the spec via a URL, or use an absolute server URL.")
    raise SpecError("Spec has no server/base URL.")


def discover_auth(spec: Dict[str, Any], slug: str,
                  base_url: Optional[str] = None) -> Dict[str, Any]:
    """Read the spec's declared security (+ heuristically any login endpoint) into a
    normalized auth descriptor that drives BOTH the Connections form (`fields`) and
    the runtime AuthStrategy (`auth_type` + `auth_config`):

        {auth_type, security_scheme, auth_config, fields, scope_default, warnings}

    See docs/INTEGRATION_AUTH_ARCHITECTURE.md §10b.
    """
    schemes = (spec.get("components") or {}).get("securitySchemes") or {}
    warnings: List[str] = []
    if schemes:
        key, s = _pick_scheme(schemes)
        if len(schemes) > 1:
            warnings.append(f"multiple security schemes declared; using '{key}'.")
        desc = _descriptor_for_scheme(s, slug)
    else:
        desc = {"auth_type": None, "security_scheme": None, "auth_config": {},
                "fields": [], "scope_default": "org"}

    if desc["auth_type"] in (None, BEARER_STATIC, CUSTOM):
        login = find_login_op(spec, base_url)
        if login:
            desc = _login_grant_descriptor(login)
            warnings.append("the access token is issued by the API's own login "
                            f"endpoint ({login.get('login_url')}); using the login flow.")

    desc["warnings"] = warnings
    return desc


_SCHEME_RANK = {"oauth2": 4, "http": 3, "apikey": 2, "openidconnect": 1}


def _pick_scheme(schemes: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """Pick the strongest declared scheme (oauth2 > http-bearer > http-basic > apiKey)."""
    def rank(item):
        _, s = item
        typ = (s.get("type") or "").lower()
        r = float(_SCHEME_RANK.get(typ, 0))
        if typ == "http" and (s.get("scheme") or "").lower() == "bearer":
            r += 0.5
        return r
    return max(schemes.items(), key=rank)


def _descriptor_for_scheme(s: Dict[str, Any], slug: str) -> Dict[str, Any]:
    typ = (s.get("type") or "").lower()
    scheme = (s.get("scheme") or "").lower()
    if typ == "apikey":
        where, name = s.get("in", "header"), s.get("name", "X-API-Key")
        return {"auth_type": API_KEY, "security_scheme": "apiKey",
                "auth_config": {"in": where, "name": name}, "scope_default": "user",
                "fields": [{"key": "api_key", "label": name, "type": "password",
                            "required": True, "secret": True, "scope": "identity",
                            "help": f"Sent as the '{name}' {where} on every request."}]}
    if typ == "http" and scheme == "bearer":
        return {"auth_type": BEARER_STATIC, "security_scheme": "bearer",
                "auth_config": {"in": "header", "name": "Authorization", "fmt": "Bearer {token}"},
                "scope_default": "user",
                "fields": [{"key": "token", "label": "Bearer Token", "type": "password",
                            "required": True, "secret": True, "scope": "identity"}]}
    if typ == "http" and scheme == "basic":
        return {"auth_type": BASIC, "security_scheme": "basic", "auth_config": {"in": "basic"},
                "scope_default": "user",
                "fields": [{"key": "username", "label": "Username", "type": "text",
                            "required": True, "secret": False, "scope": "identity"},
                           {"key": "password", "label": "Password", "type": "password",
                            "required": True, "secret": True, "scope": "identity"}]}
    if typ == "oauth2":
        flows = s.get("flows") or {}
        if "password" in flows and "clientCredentials" not in flows:
            pw = flows["password"] or {}
            token_url = pw.get("tokenUrl", "")
            return {"auth_type": OAUTH2_PASSWORD, "security_scheme": "oauth2",
                    "auth_config": {"flow": "password", "token_url": token_url,
                                    "scopes": list((pw.get("scopes") or {}).keys())},
                    "scope_default": "user",
                    "fields": [{"key": "username", "label": "Username", "type": "text",
                                "required": True, "secret": False, "scope": "identity"},
                               {"key": "password", "label": "Password", "type": "password",
                                "required": True, "secret": True, "scope": "identity"},
                               {"key": "token_url", "label": "Token URL", "type": "text",
                                "required": not token_url, "secret": False, "scope": "org"}]}
        cc = flows.get("clientCredentials") or flows.get("password") or {}
        token_url = cc.get("tokenUrl", "")
        return {"auth_type": OAUTH2_CLIENT_CREDENTIALS, "security_scheme": "oauth2",
                "auth_config": {"flow": "client_credentials", "token_url": token_url,
                                "scopes": list((cc.get("scopes") or {}).keys())},
                "scope_default": "org",
                "fields": [{"key": "client_id", "label": "Client ID", "type": "text",
                            "required": True, "secret": False, "scope": "org"},
                           {"key": "client_secret", "label": "Client Secret", "type": "password",
                            "required": True, "secret": True, "scope": "org"},
                           {"key": "token_url", "label": "Token URL", "type": "text",
                            "required": not token_url, "secret": False, "scope": "org",
                            "help": "OAuth2 client-credentials token endpoint."}]}
    return {"auth_type": CUSTOM, "security_scheme": "custom",
            "auth_config": {"in": "header", "name": f"X-{slug}-Key"}, "scope_default": "user",
            "fields": [{"key": "api_key", "label": "API Key", "type": "password",
                        "required": True, "secret": True, "scope": "identity"}]}


_LOGIN_PATH_RE = re.compile(r"(login|sign[_-]?in)", re.I)
_REFRESH_PATH_RE = re.compile(r"refresh", re.I)
_USER_KEYS = ("username", "user", "email", "login", "userid", "user_name", "mail")
_PASS_KEYS = ("password", "passwd", "pwd", "pass", "secret")
_TOKEN_KEYS = ("accesstoken", "access_token", "token", "jwt", "idtoken", "id_token", "bearer")
_REFRESH_KEYS = ("refreshtoken", "refresh_token")


def _match_key(keys, candidates) -> Optional[str]:
    low = {k.lower(): k for k in keys}
    for cand in candidates:
        if cand in low:
            return low[cand]
    for cand in candidates:
        for lk, orig in low.items():
            if cand in lk:
                return orig
    return None


def _json_body_props(op: Dict[str, Any], root: Dict[str, Any]) -> Dict[str, Any]:
    sch = (((op.get("requestBody") or {}).get("content") or {}).get("application/json") or {}).get("schema")
    if not sch:
        return {}
    inlined = _inline(sch, root)
    return (inlined.get("properties") or {}) if isinstance(inlined, dict) else {}


def _resp_token_keys(op: Dict[str, Any], root: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for code, r in (op.get("responses") or {}).items():
        if not str(code).startswith("2"):
            continue
        sch = (((r.get("content") or {}).get("application/json") or {}).get("schema"))
        if not sch:
            continue
        node = _inline(sch, root)
        props = (node.get("properties") or {}) if isinstance(node, dict) else {}
        if props:
            return _match_key(props, _TOKEN_KEYS), _match_key(props, _REFRESH_KEYS)
    return None, None


def find_login_op(spec: Dict[str, Any], base_url: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Detect an API that ISSUES its own bearer token via a login endpoint — e.g.
    fleet's `POST /auth/Login` ({email,password} → {accessToken,refreshToken}). A
    plain `bearer` scheme says nothing about how to OBTAIN the token; this finds it
    so the runtime can mint+refresh instead of asking for a pasted static token.
    Returns {login_url, body_map, token_path, refresh_token_path, refresh?} or None.
    """
    paths = spec.get("paths") or {}
    best = None
    for path, item in paths.items():
        if not isinstance(item, dict) or not _LOGIN_PATH_RE.search(path):
            continue
        post = item.get("post")
        if not isinstance(post, dict):
            continue
        props = _json_body_props(post, spec)
        pw = _match_key(props, _PASS_KEYS)
        if not pw:
            continue
        user = _match_key(props, _USER_KEYS) or "username"
        tok_key, refr_key = _resp_token_keys(post, spec)
        recipe = {
            "login_url": _abs_url(base_url, path),
            "body_map": {user: "{username}", pw: "{password}"},
            "token_path": f"$.{tok_key}" if tok_key else _token_path_candidates(_ACCESS_TOKEN_KEYS),
            "refresh_token_path": (f"$.{refr_key}" if refr_key
                                   else _token_path_candidates(_REFRESH_TOKEN_KEYS)),
        }
        ref = _find_refresh_op(spec, base_url, login_path=path)
        if ref:
            recipe["refresh"] = ref
        score = 2 if re.search(r"login", path, re.I) else 1
        if best is None or score > best[0]:
            best = (score, recipe)
    return best[1] if best else None


_ACCESS_TOKEN_KEYS = ("accessToken", "access_token", "token", "jwt", "idToken", "id_token")
_REFRESH_TOKEN_KEYS = ("refreshToken", "refresh_token")
_TOKEN_ENVELOPES = ("", "data", "result", "response", "payload", "body")


def _token_path_candidates(keys) -> List[str]:
    """'$.k', '$.data.k', '$.result.k', … — tried in order so the common API
    envelopes resolve without a per-integration override."""
    return [f"$.{env}.{k}" if env else f"$.{k}" for env in _TOKEN_ENVELOPES for k in keys]


def _find_refresh_op(spec: Dict[str, Any], base_url: Optional[str],
                     login_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """The token-refresh endpoint. SCORED, not first-match: a real spec may have
    several '*refresh*' paths (e.g. a /ledger/Refresh that has nothing to do with
    auth). Prefer the one in the SAME controller as login (/auth/RefreshToken next
    to /auth/Login), then token-named paths, then those whose body carries a
    refresh token."""
    login_seg = (login_path or "").strip("/").split("/")[0].lower()
    best = None
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict) or not _REFRESH_PATH_RE.search(path):
            continue
        post = item.get("post")
        if not isinstance(post, dict):
            continue
        props = _json_body_props(post, spec)
        body_map = {}
        for k in props:
            lk = k.lower()
            if "refresh" in lk:
                body_map[k] = "{refresh_token}"
            elif "access" in lk or lk == "token":
                body_map[k] = "{access_token}"
        if not body_map:
            body_map = {"refreshToken": "{refresh_token}"}
        seg = path.strip("/").split("/")[0].lower()
        score = (3 if login_seg and seg == login_seg else 0)
        score += 2 if re.search(r"token", path, re.I) else 0
        score += 1 if any("refresh" in k.lower() for k in props) else 0
        recipe = {"url": _abs_url(base_url, path), "body_map": body_map,
                  "token_path": _token_path_candidates(_ACCESS_TOKEN_KEYS),
                  "refresh_token_path": _token_path_candidates(_REFRESH_TOKEN_KEYS)}
        if best is None or score > best[0]:
            best = (score, recipe)
    return best[1] if best else None


def _abs_url(base_url: Optional[str], path: str) -> str:
    return (base_url.rstrip("/") + "/" + path.lstrip("/")) if base_url else path


def _login_grant_descriptor(login: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "auth_type": LOGIN_GRANT, "security_scheme": "bearer", "auth_config": login,
        "scope_default": "user",
        "fields": [
            {"key": "username", "label": "Username / Email", "type": "text",
             "required": True, "secret": False, "scope": "identity"},
            {"key": "password", "label": "Password", "type": "password",
             "required": True, "secret": True, "scope": "identity",
             "help": "Used once to obtain a token; only a refresh token is stored "
                     "— never your password."},
        ],
    }


def extract_security(spec: Dict[str, Any], slug: str) -> Tuple[
        Optional[str], Dict[str, Any], List[Dict[str, Any]]]:
    """Back-compat shim → (security_scheme, auth_config, fields). New callers use
    `discover_auth()` for the full descriptor (auth_type, scope_default, warnings)."""
    d = discover_auth(spec, slug)
    return d["security_scheme"], d["auth_config"], d["fields"]


def _operation(path: str, method: str, op: Dict[str, Any], slug: str,
               root: Dict[str, Any], shared_params: List[Dict[str, Any]],
               global_headers=()) -> IntegrationOperation:
    op_id = op.get("operationId") or f"{method}_{path}"
    properties: Dict[str, Any] = {}
    required: List[str] = []
    req_map: Dict[str, Any] = {"path": [], "query": [], "header": [], "has_body": False}

    for p in list(shared_params) + list(op.get("parameters", [])):
        p = _inline(p, root)
        loc, name = p.get("in"), p.get("name")
        if loc not in ("path", "query", "header") or not name:
            continue
        if loc == "header" and name in global_headers:
            continue
        sch = _prep(p.get("schema") or {"type": "string"}, root)
        if p.get("description"):
            sch.setdefault("description", p["description"])
        properties[name] = sch
        req_map[loc].append(name)
        if p.get("required") or loc == "path":
            required.append(name)

    body = (op.get("requestBody") or {})
    body_schema = (((body.get("content") or {}).get("application/json")) or {}).get("schema")
    if body_schema is not None:
        bsch = _prep(body_schema, root)
        bsch.setdefault("description", "Request body.")
        properties["body"] = bsch
        req_map["has_body"] = True
        if body.get("required"):
            required.append("body")

    declared = set(req_map["path"])
    for ph in _PATH_PARAM_RE.findall(path):
        if ph not in declared:
            properties.setdefault(ph, {
                "type": "string",
                "description": f"Path parameter '{ph}' (inferred from the URL template).",
            })
            req_map["path"].append(ph)
            if ph not in required:
                required.append(ph)
            declared.add(ph)

    params_schema = {"type": "object", "properties": properties}
    if required:
        params_schema["required"] = required

    method_u = method.upper()
    return IntegrationOperation(
        operation_id=op_id,
        tool_name=tool_name_for(slug, op_id),
        http_method=method_u,
        path_template=path,
        summary=op.get("summary", "") or "",
        description=_describe(slug, op, method_u, path),
        parameters_schema=params_schema,
        side_effect=side_effect_for_method(method_u),
        requires_approval=requires_approval_for_method(method_u),
        enabled=True,
    ), req_map


def _describe(slug: str, op: Dict[str, Any], method: str, path: str) -> str:
    """The text the model reasons over — description quality IS the product."""
    parts = [op.get("description") or op.get("summary") or ""]
    parts.append(f"({method} {path} on the '{slug}' API.)")
    return " ".join(p for p in parts if p).strip()


def _global_header_names(spec: Dict[str, Any]) -> set:
    """Header parameters shared across (nearly) all ops — e.g. `x-tenant-domain`,
    `api-version`. These are INFRASTRUCTURE config, not per-call LLM arguments, so
    discovery promotes them to integration-level `default_headers` and excludes them
    from each op's parameters (otherwise the model is asked to fill a tenant header
    it can't possibly know, on every call). Detected by frequency (≥80% of ops)."""
    paths = spec.get("paths") or {}
    counts: Dict[str, int] = {}
    total = 0
    for _path, item in paths.items():
        if not isinstance(item, dict):
            continue
        shared = item.get("parameters", [])
        for method, op in item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            total += 1
            names = set()
            for p in list(shared) + list(op.get("parameters", [])):
                p = _inline(p, spec)
                if isinstance(p, dict) and p.get("in") == "header" and p.get("name"):
                    names.add(p["name"])
            for n in names:
                counts[n] = counts.get(n, 0) + 1
    return {n for n, c in counts.items() if total and c >= 0.8 * total}


def normalize(raw: Dict[str, Any], slug: str,
              display_name: Optional[str] = None,
              source_url: Optional[str] = None) -> Tuple[Integration, Dict[str, Dict]]:
    """Lower a parsed spec into an Integration + {operation_id: request_map}."""
    spec = coerce_to_openapi3(raw)
    base_url = extract_base_url(spec, source_url=source_url)
    desc = discover_auth(spec, slug, base_url=base_url)
    global_headers = _global_header_names(spec)

    integ = Integration(
        slug=slug,
        display_name=display_name or (spec.get("info") or {}).get("title") or slug,
        base_url=base_url,
        spec_kind=detect_kind(raw),
        auth_provider=(f"api_{slug}" if desc["auth_type"] else None),
        security_scheme=desc["security_scheme"],
        auth_type=desc["auth_type"],
        auth_config=desc["auth_config"],
        default_headers={h: "" for h in sorted(global_headers)},
    )
    req_maps: Dict[str, Dict] = {}
    seen_ops = set()
    for path, item in (spec.get("paths") or {}).items():
        if not isinstance(item, dict):
            continue
        shared = item.get("parameters", [])
        for method, op in item.items():
            if method.lower() not in _HTTP_METHODS or not isinstance(op, dict):
                continue
            operation, rmap = _operation(path, method.lower(), op, slug, spec, shared,
                                         global_headers=global_headers)
            if operation.operation_id in seen_ops:
                operation = _dedup(operation, slug, method, seen_ops)
            seen_ops.add(operation.operation_id)
            integ.operations.append(operation)
            req_maps[operation.operation_id] = rmap
    if not integ.operations:
        raise SpecError("Spec defines no operations.")
    return integ, req_maps


def _dedup(op: IntegrationOperation, slug, method, seen_ops: set) -> IntegrationOperation:
    """Force a UNIQUE operation_id for a non-compliant spec (OpenAPI requires
    operationId to be document-unique, but real specs violate it). Without a
    guaranteed-unique id the (integration_id, operation_id) upsert would silently
    overwrite the earlier op — one endpoint lost. Append the method, then a
    counter, until unique."""
    from dataclasses import replace
    candidate = f"{op.operation_id}_{method.lower()}"
    n = 2
    while candidate in seen_ops:
        candidate = f"{op.operation_id}_{method.lower()}_{n}"
        n += 1
    return replace(op, operation_id=candidate, tool_name=tool_name_for(slug, candidate))


def provider_fields(spec: Dict[str, Any], slug: str,
                    base_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """The ag_cred.providers `fields` for this spec's auth (for ingest). `base_url`
    lets login-flow detection emit an absolute login_url (fields are identical
    either way)."""
    return discover_auth(coerce_to_openapi3(spec), slug, base_url=base_url)["fields"]
