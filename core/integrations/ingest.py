"""
Spec ingestion — the build-time pipeline.
═════════════════════════════════════════
admin spec (dict | URL | JSON/YAML text)
    → fetch → parse → validate(host/https/allow_private)
    → normalize (core.integrations.openapi)
    → materialize auth provider (ag_cred.providers via Connections.upsert_provider)
    → persist (ag_api.* via IntegrationStore, spec-hash diff)
    → embed operation descriptions (EmbeddingClient → operations.embedding)
    → IngestReport

Tenant-routed via the injected db/connections (TenantDBProxy). Gated by the caller;
the runtime that exposes the tools is Phase 2.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import logging
import re
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit

from core.integrations import openapi
from core.integrations.model import slugify
from core.integrations.store import IntegrationStore

logger = logging.getLogger(__name__)


@dataclass
class IngestReport:
    slug: str
    integration_id: Optional[int] = None
    operation_count: int = 0
    added: int = 0
    updated: int = 0
    deprecated: int = 0
    embedded: int = 0
    unchanged: bool = False
    security_scheme: Optional[str] = None
    auth_provider: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class SpecIngestor:
    def __init__(self, db, connections, embeddings=None, http=None, store=None,
                 mcp_post=None):
        self._db = db
        self._connections = connections
        self._embeddings = embeddings
        self._http = http
        self._mcp_post = mcp_post
        self._store = store if store is not None else IntegrationStore(db)

    def ingest(self, source: Union[str, dict], *, slug: str,
               spec_kind: Optional[str] = None,
               created_by: Optional[int] = None, allow_private: bool = False,
               display_name: Optional[str] = None,
               source_url: Optional[str] = None, force: bool = False) -> IngestReport:
        slug = slugify(slug)
        if not slug:
            raise openapi.SpecError("A non-empty slug is required.")
        report = IngestReport(slug=slug)

        is_mcp = (spec_kind == "mcp") or (
            isinstance(source, str) and source.lower().startswith(("mcp://", "mcps://")))
        if is_mcp:
            return self._ingest_mcp(source, slug, created_by, allow_private,
                                    display_name, report)

        raw = self._fetch(source, allow_private)
        src_url = source_url or (
            source if (isinstance(source, str) and
                       source.lstrip().lower().startswith(("http://", "https://"))) else None)
        base_url = openapi.extract_base_url(openapi.coerce_to_openapi3(raw), source_url=src_url)
        self._validate_host(base_url, allow_private, report)

        spec_hash = _spec_hash(raw)
        existing = self._store.get_integration_by_slug(slug)
        if existing and not force and existing.get("spec_hash") == spec_hash:
            iid = existing["integration_id"]
            report.integration_id = iid
            report.unchanged = True
            report.operation_count = len(self._store.list_operations(iid))
            healed = self.backfill_embeddings(iid)
            report.embedded = healed.get("embedded", 0)
            report.warnings.extend(healed.get("warnings", []))
            logger.info("ingest %s: unchanged (spec_hash match); embeddings backfilled=%d",
                        slug, report.embedded)
            return report

        integ, req_maps = openapi.normalize(raw, slug, display_name, source_url=src_url)
        integ.allow_private = allow_private
        if existing:
            prev = existing.get("default_headers") or {}
            if isinstance(prev, str):
                try:
                    prev = json.loads(prev)
                except Exception:
                    prev = {}
            if isinstance(prev, dict):
                integ.default_headers = {**integ.default_headers, **prev}
        report.security_scheme = integ.security_scheme
        report.auth_provider = integ.auth_provider
        report.operation_count = len(integ.operations)

        self._materialize_auth(raw, integ, report)

        integration_id = self._store.upsert_integration(
            integ, spec_raw=raw, spec_hash=spec_hash, created_by=created_by)
        report.integration_id = integration_id

        diff = self._store.replace_operations(integration_id, integ.operations, req_maps)
        report.added, report.updated, report.deprecated = (
            diff["added"], diff["updated"], diff["deprecated"])

        report.embedded = self._embed(integration_id, integ, report)
        logger.info("ingest %s: id=%s ops=%d (+%d ~%d -%d) embedded=%d",
                    slug, integration_id, report.operation_count,
                    report.added, report.updated, report.deprecated, report.embedded)
        return report

    def _ingest_mcp(self, source: str, slug: str, created_by, allow_private,
                    display_name, report: IngestReport) -> IngestReport:
        from core.integrations import mcp
        url = source
        for pfx in ("mcp://", "mcps://"):
            if url.lower().startswith(pfx):
                url = "https://" + url[len(pfx):]
                break
        self._validate_host(url, allow_private, report)
        tools = mcp.fetch_tools(url, http_post=self._mcp_post)

        spec_hash = _spec_hash({"tools": tools})
        existing = self._store.get_integration_by_slug(slug)
        if existing and existing.get("spec_hash") == spec_hash:
            report.integration_id = existing["integration_id"]
            report.unchanged = True
            report.operation_count = len(self._store.list_operations(existing["integration_id"]))
            return report

        integ, req_maps = mcp.normalize_mcp(tools, slug, url, display_name)
        integ.allow_private = allow_private
        report.operation_count = len(integ.operations)
        integration_id = self._store.upsert_integration(
            integ, spec_raw={"tools": tools}, spec_hash=spec_hash, created_by=created_by)
        report.integration_id = integration_id
        diff = self._store.replace_operations(integration_id, integ.operations, req_maps)
        report.added, report.updated, report.deprecated = (
            diff["added"], diff["updated"], diff["deprecated"])
        report.embedded = self._embed(integration_id, integ, report)
        logger.info("ingest %s (mcp): id=%s tools=%d", slug, integration_id, report.operation_count)
        return report

    def _fetch(self, source: Union[str, dict], allow_private: bool = False) -> dict:
        if isinstance(source, dict):
            return source
        if not isinstance(source, str):
            raise openapi.SpecError("source must be a dict, URL, or spec text.")
        text = source
        if source.lstrip().lower().startswith(("http://", "https://")):
            text = self._http_get(source, allow_private)
        return _parse_text(text)

    def _http_get(self, url: str, allow_private: bool = False) -> str:
        if self._http is not None:
            return self._http(url)
        import requests
        from core.integrations.runtime import egress_check, EgressBlocked
        for _ in range(4):
            try:
                egress_check(url, base_url=url, egress_allow=[], allow_private=allow_private)
            except EgressBlocked as e:
                raise openapi.SpecError(f"spec URL blocked by egress policy: {e}")
            r = requests.get(url, timeout=15, allow_redirects=False,
                             headers={"Accept": "application/json, application/yaml"})
            if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                from urllib.parse import urljoin
                url = urljoin(url, r.headers["location"])
                continue
            r.raise_for_status()
            return r.text
        raise openapi.SpecError("spec URL exceeded redirect limit.")

    def _validate_host(self, base_url: str, allow_private: bool, report: IngestReport):
        parts = urlsplit(base_url)
        if parts.scheme not in ("http", "https"):
            raise openapi.SpecError(f"Unsupported scheme: {parts.scheme}")
        host = parts.hostname or ""
        if not host:
            raise openapi.SpecError("base URL has no host.")
        if parts.scheme == "http" and not allow_private:
            report.warnings.append("base URL is http (not https).")
        try:
            infos = socket.getaddrinfo(host, None)
            addrs = {ai[4][0] for ai in infos}
        except Exception:
            addrs = set()
        for a in addrs:
            try:
                ip = ipaddress.ip_address(a)
            except ValueError:
                continue
            if ip.is_link_local and str(ip).startswith("169.254"):
                raise openapi.SpecError("base URL resolves to the cloud-metadata range — blocked.")
            if (ip.is_private or ip.is_loopback or ip.is_link_local) and not allow_private:
                raise openapi.SpecError(
                    f"base URL resolves to a private/internal address ({a}); "
                    f"set allow_private=True to onboard an internal business API.")

    def _materialize_auth(self, raw: dict, integ, report: IngestReport):
        if not integ.auth_type:
            return
        fields = openapi.provider_fields(raw, integ.slug, base_url=integ.base_url)
        try:
            self._connections.upsert_provider(
                integ.auth_provider, integ.display_name,
                kind="direct", category="api", fields=fields,
                supports_user=True, supports_verify=True)
        except Exception as e:
            report.warnings.append(f"auth provider not materialized: {e}")

    EMBED_BATCH = 64
    EMBED_BUDGET_S = 25.0

    def _embed(self, integration_id: int, integ, report: IngestReport) -> int:
        emb = self._embeddings
        if emb is None or not getattr(emb, "available", False):
            report.warnings.append("embeddings unavailable — selection falls back to keywords.")
            return 0
        items = [(op.operation_id, _embed_text(op.http_method, op.path_template,
                                               op.operation_id, op.summary, op.description))
                 for op in integ.operations]
        return self._embed_items(integration_id, items, len(integ.operations),
                                 report, budget_s=self.EMBED_BUDGET_S)

    def _embed_items(self, integration_id: int, items, total: int,
                     report: IngestReport, *, budget_s: float) -> int:
        """Embed (operation_id, text) pairs in resilient, persisted chunks."""
        emb = self._embeddings
        import time
        embedded = 0
        deadline = time.monotonic() + budget_s
        budget_hit = False
        for i in range(0, len(items), self.EMBED_BATCH):
            if time.monotonic() > deadline:
                budget_hit = True
                break
            chunk = items[i:i + self.EMBED_BATCH]
            try:
                vectors = emb.embed([t for _, t in chunk])
            except Exception as e:
                logger.warning("embed chunk failed (%d-%d): %s", i, i + len(chunk), e)
                vectors = []
            mapping = {oid: vectors[j] for j, (oid, _) in enumerate(chunk)
                       if j < len(vectors) and vectors[j]}
            if mapping:
                embedded += self._store.set_embeddings(integration_id, mapping)
        if embedded < total:
            done = "time budget reached" if budget_hit else "some batches failed"
            report.warnings.append(
                f"embedded {embedded}/{total} operations ({done}); the rest use keyword "
                f"selection — run the embeddings backfill to finish.")
        return embedded

    def backfill_embeddings(self, integration_id: int,
                            budget_s: float = 120.0) -> Dict[str, Any]:
        """Embed only the operations that still lack a vector (idempotent repair).

        Safe to call repeatedly: a re-import self-heals via this, and the admin API
        exposes it directly so a timed-out import can be finished without a full
        re-import. Returns {embedded, remaining, warnings}.
        """
        emb = self._embeddings
        if emb is None or not getattr(emb, "available", False):
            return {"embedded": 0, "remaining": None,
                    "warnings": ["embeddings unavailable — cannot backfill."]}
        rows = self._store.operations_missing_embeddings(integration_id)
        if not rows:
            return {"embedded": 0, "remaining": 0, "warnings": []}
        items = [(r["operation_id"], _embed_text(r.get("http_method"), r.get("path_template"),
                                                 r["operation_id"], r.get("summary"),
                                                 r.get("description"))) for r in rows]
        rep = IngestReport(slug="")
        embedded = self._embed_items(integration_id, items, len(items), rep, budget_s=budget_s)
        remaining = len(self._store.operations_missing_embeddings(integration_id))
        return {"embedded": embedded, "remaining": remaining, "warnings": rep.warnings}


def _embed_text(method: Optional[str], path: Optional[str], operation_id: str,
                summary: Optional[str], description: Optional[str]) -> str:
    """Build the text embedded (and keyword-matched) for an operation.

    Combine every signal we have — summary, description, the HTTP method + path,
    and a humanized operationId — so selection still has something to match even
    when a spec ships no summaries (common for generated specs), where embedding
    just the description would cluster all ops into one uninformative blob.
    """
    humanized = re.sub(r"[_/{}]+", " ", operation_id or "").strip()
    parts = [summary or "", description or "",
             f"{(method or '').upper()} {path or ''}".strip(), humanized]
    seen, out = set(), []
    for p in parts:
        p = (p or "").strip()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return "\n".join(out)


NORMALIZER_VERSION = "2"


def _spec_hash(raw: dict) -> str:
    payload = json.dumps(raw, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(
        f"{payload}\x00nv={NORMALIZER_VERSION}".encode()).hexdigest()


def _parse_text(text: str) -> dict:
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        import yaml
        data = yaml.safe_load(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    raise openapi.SpecError("Could not parse spec as JSON or YAML.")
