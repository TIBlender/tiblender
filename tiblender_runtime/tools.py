"""Bounded external tool adapters used by multi-perspective investigation."""

from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from .iocs import extract_iocs, filter_reportable_iocs
from .models import IocConfidence, IocSet, NormalizedPost, ToolQuery, ToolResult
from .contract import REQUIRED_TOOL_TYPES
from .services import ServiceStatus, resolve_services
from .settings import Settings
from .tool_fixtures import RecordingCorruptedError, ToolRecordingReplayLog, ToolRecordingStore
from .tools_searxng import (
    _build_searxng_query,
    _normalize_searxng_results,
    _search_snippet_iocs,
)
from .tools_virustotal import _VirusTotalHttpError, _vt_get_json, _vt_unavailable_reason
from .tools_social import (  # re-exported: tools stays the single public entry point
    SocialSearchMixin,
    ToolUnavailableError,
    _dedupe,
    _param,
    _read_secret_lines,
    _reddit_accounts,
    _result,
    _search_result_text,
    _unavailable,
    _x_bearer_tokens,
)

__all__ = [
    "ToolRecordingStore",
    "ToolRunner",
    "ToolUnavailableError",
    "_dedupe",
    "_read_secret_lines",
    "_reddit_accounts",
    "_search_result_text",
    "_x_bearer_tokens",
]

# Tools backed by an external service resolve their run mode (live/fixture/
# unavailable) through this map; DISCORD_SEARCH/TELEGRAM_SEARCH always use the
# internal collected-corpus path and are intentionally absent here.
_SERVICE_TOOL_MAP = {
    "SEARCH_ENGINE": "search_engine",
    "WHOIS_LOOKUP": "whois",
    "PASSIVE_DNS_LOOKUP": "passive_dns",
    "TLS_CT_LOOKUP": "tls_ct",
    "WEB_CRAWL": "web_crawl",
    "NVD_CVE_LOOKUP": "nvd",
    "GITHUB_SEARCH": "github",
    "X_SEARCH": "x",
    "REDDIT_SEARCH": "reddit",
}

# Social tools whose fixture-mode recording miss degrades to the collected
# corpus instead of an empty result.
_CORPUS_FALLBACK_PLATFORMS = {
    "X_SEARCH": ("x", "X"),
    "REDDIT_SEARCH": ("reddit", "Reddit"),
}


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json,text/html;q=0.9,*/*;q=0.8",
        "User-Agent": "tiblender/0.1",
    }
    if extra:
        headers.update(extra)
    return headers


# Fallback for the direct-call paths that carry no Settings; every call from
# ToolRunner passes settings.tool_max_bytes (TIBLENDER_TOOL_MAX_BYTES).
_DEFAULT_TOOL_MAX_BYTES = 2 * 1024 * 1024


def _http_text(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    max_bytes: int = _DEFAULT_TOOL_MAX_BYTES,
) -> tuple[str, str]:
    # All HTTP helpers cap time and bytes so a failed provider cannot stall the
    # full reporting cycle or persist large raw pages in tool output.
    request = urllib.request.Request(url, headers=headers or _headers(), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes)
            final_url = response.geturl()
            charset = response.headers.get_content_charset() or "utf-8"
    except urllib.error.HTTPError as error:
        detail = error.read(800).decode("utf-8", errors="replace")
        if error.code in {401, 403, 404, 408, 410, 429, 451}:
            raise ToolUnavailableError(f"HTTP {error.code}: {detail}") from error
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        reason = getattr(error, "reason", error)
        raise ToolUnavailableError(str(reason)) from error
    except OSError as error:
        raise ToolUnavailableError(str(error)) from error
    return data.decode(charset, errors="replace"), final_url


def _http_json(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    max_bytes: int = _DEFAULT_TOOL_MAX_BYTES,
) -> tuple[Any, str]:
    text, final_url = _http_text(
        url,
        headers=headers or _headers({"Accept": "application/json"}),
        timeout=timeout,
        max_bytes=max_bytes,
    )
    try:
        return json.loads(text), final_url
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid JSON response: {error}") from error


_PASSIVE_DNS_RESOLUTION_FIELDS = ("urls", "domains", "ips")


def _truncate_passive_dns(discovered: IocSet, queried: str, limit: int) -> IocSet:
    """Apply the deployed 20-resolution cap to a recorded passive-DNS answer.

    The deployed tool truncates the provider payload to its first ``limit``
    resolution rows (`layer2/tools.py`), and the live path here now asks
    VirusTotal for the same number. Bundled recordings were captured under an
    older ``limit=40`` request, so the same cap is applied when they are
    replayed -- deterministically, over the recording's own stored (sorted)
    order -- rather than re-recording every provider answer.

    The queried indicator is the *query* side of every resolution pair, not a
    resolution, so it is kept regardless; only the pivot values are counted.
    """
    normalised_query = str(queried).strip().lower().strip(".")
    kept: dict[str, list[str]] = {field_name: [] for field_name in _PASSIVE_DNS_RESOLUTION_FIELDS}
    budget = max(0, limit)
    for field_name in _PASSIVE_DNS_RESOLUTION_FIELDS:
        for value in getattr(discovered, field_name):
            if str(value).strip().lower().strip(".") == normalised_query:
                kept[field_name].append(value)
                continue
            if budget <= 0:
                continue
            budget -= 1
            kept[field_name].append(value)
    return IocSet(
        urls=kept["urls"],
        domains=kept["domains"],
        ips=kept["ips"],
        hashes=list(discovered.hashes),
        cves=list(discovered.cves),
    )


def _recording_result(query: ToolQuery, payload: dict[str, Any], *, passive_dns_limit: int) -> ToolResult:
    # Recording playback replays the recorded status, summary and IoCs
    # verbatim, and labels the result as replayed.
    iocs = payload.get("discovered_iocs") or {}
    discovered = IocSet(
        urls=list(iocs.get("urls", [])),
        domains=list(iocs.get("domains", [])),
        ips=list(iocs.get("ips", [])),
        hashes=list(iocs.get("hashes", [])),
        cves=list(iocs.get("cves", [])),
    )
    if query.tool_type == "PASSIVE_DNS_LOOKUP":
        discovered = _truncate_passive_dns(discovered, _param(query, "domain", "ip"), passive_dns_limit)
    status = payload.get("status", "success")
    return _result(
        query,
        summary=payload.get("summary", ""),
        status=status,
        result_url=payload.get("source_url", ""),
        discovered_iocs=discovered,
        confidence=IocConfidence.MEDIUM if status == "success" else IocConfidence.LOW,
        error=payload.get("error", ""),
        raw={
            "source": "recording",
            "recorded_at": payload.get("recorded_at", ""),
            "recording_source_url": payload.get("source_url", ""),
        },
    )


def _failed(query: ToolQuery, error: Exception) -> ToolResult:
    message = str(error)[:1000]
    return _result(query, summary=f"{query.tool_type} failed: {message}", status="error", confidence=IocConfidence.LOW, error=message)


# An absolute filesystem path, at a token boundary so a URL's path is left
# alone. Two or more segments, to avoid swallowing a bare "/x".
_ABSOLUTE_PATH_RE = re.compile(r"(?<![\w:/])(?:/[A-Za-z0-9._+-]+){2,}/?")


def _machine_independent_reason(error: Exception) -> str:
    """One bounded line describing ``error``, with local paths removed.

    A Playwright launch failure names the *recording host's* browser cache
    (`/home/<user>/.cache/ms-playwright/...`) and draws a multi-line box
    around it. Recorded verbatim into a WEB_CRAWL fixture, that ships the
    recording machine's username inside the bundle and means nothing on any
    other host. What the fallback needs to say is the cause, not the path it
    was looked for at.
    """
    text = str(error).strip()
    if not text:
        return type(error).__name__
    return _ABSOLUTE_PATH_RE.sub("<path>", text.splitlines()[0])[:300]


def _plain_text_from_html(text: str) -> str:
    cleaned = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
    cleaned = re.sub(r"(?s)<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _render_javascript_page(
    url: str,
    *,
    timeout: int,
    max_bytes: int,
    wait_ms: int,
) -> tuple[str, str, dict[str, Any]]:
    # JavaScript rendering is intentionally lightweight: it blocks heavy media
    # and returns body text rather than screenshots or complete browser traces.
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise ToolUnavailableError("Playwright is not installed for JavaScript rendering.") from error

    blocked_resource_types = {"image", "media", "font"}
    timeout_ms = max(1000, timeout * 1000)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-dev-shm-usage",
                    "--disable-extensions",
                    "--disable-gpu",
                    "--disable-background-networking",
                    "--no-sandbox",
                ],
            )
            try:
                context = browser.new_context(
                    java_script_enabled=True,
                    user_agent="tiblender/0.1",
                    viewport={"width": 1366, "height": 900},
                )
                page = context.new_page()

                def route_handler(route):
                    if route.request.resource_type in blocked_resource_types:
                        route.abort()
                    else:
                        route.continue_()

                page.route("**/*", route_handler)
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
                if wait_ms > 0:
                    page.wait_for_timeout(wait_ms)
                try:
                    page.wait_for_load_state("networkidle", timeout=min(3000, timeout_ms))
                except PlaywrightTimeoutError:
                    pass
                title = page.title()
                try:
                    body_text = page.locator("body").inner_text(timeout=min(3000, timeout_ms))
                except PlaywrightError:
                    body_text = _plain_text_from_html(page.content())
                body_text = body_text[:max_bytes]
                escaped_title = html.escape(title or "")
                escaped_body = html.escape(body_text)
                rendered_html = f"<html><title>{escaped_title}</title><body>{escaped_body}</body></html>"
                return (
                    rendered_html,
                    page.url,
                    {
                        "javascript_rendered": True,
                        "blocked_resource_types": sorted(blocked_resource_types),
                        "wait_ms": wait_ms,
                    },
                )
            finally:
                browser.close()
    except PlaywrightError as error:
        raise ToolUnavailableError(f"JavaScript rendering failed: {_machine_independent_reason(error)}") from error


class ToolRunner(SocialSearchMixin):
    def __init__(
        self,
        settings: Settings,
        corpus: list[NormalizedPost] | None = None,
        *,
        services: dict[str, ServiceStatus] | None = None,
        recordings: ToolRecordingStore | None = None,
        replay_log: ToolRecordingReplayLog | None = None,
    ) -> None:
        self.settings = settings
        self.corpus = corpus or []
        self.allowed_tools = set(REQUIRED_TOOL_TYPES)
        self.services = services if services is not None else resolve_services(settings)
        self.recordings = recordings if recordings is not None else ToolRecordingStore(settings.fixtures_dir / "tools")
        # Optional: when the caller supplies a log, every recording-backed
        # lookup is tallied as a hit or a miss so the cycle can report its
        # real replay coverage and name the keys that still need recording.
        self.replay_log = replay_log

    def _lookup_recording(self, query: ToolQuery) -> tuple[dict[str, Any] | None, str]:
        # Shared, corruption-tolerant core for both recording call sites: a
        # missing recording is a normal "no recording" case (empty message),
        # while a corrupted file on disk never raises out of this method.
        try:
            payload = self.recordings.lookup(query.tool_type, query.parameters)
        except RecordingCorruptedError as error:
            return None, str(error)
        return payload, ""

    def _recording_lookup_or_unavailable(self, query: ToolQuery) -> ToolResult:
        payload, error_message = self._lookup_recording(query)
        # Social tools are excluded from the replay tally: their fixture-mode
        # miss is not a coverage gap but a documented handover to the Step 1
        # collected corpus, and their requests are not recordable from a
        # provider in the first place.
        if self.replay_log is not None and query.tool_type not in _CORPUS_FALLBACK_PLATFORMS:
            self.replay_log.observe(query.tool_type, query.parameters, hit=payload is not None)
        if payload is not None:
            return _recording_result(query, payload, passive_dns_limit=self.settings.passive_dns_limit)
        if error_message:
            return _unavailable(query, error_message)
        # A missing recording for a social platform is not a dead end: the same
        # investigation path exists over the Step 1 collected corpus, which is
        # already local evidence and never contacts the live platform.
        fallback = _CORPUS_FALLBACK_PLATFORMS.get(query.tool_type)
        if fallback:
            platform, label = fallback
            corpus_result = self._corpus_search_result(
                query,
                platform=platform,
                label=label,
                include_iocs=True,
            )
            if corpus_result:
                return corpus_result
        return _unavailable(
            query,
            f"no recorded tool response for {query.tool_type} with parameters {query.parameters}",
        )

    def execute_many(self, queries: list[ToolQuery]) -> list[ToolResult]:
        return [self.execute(query) for query in queries]

    def execute(self, query: ToolQuery) -> ToolResult:
        # The runtime enforces the prompt/tool contract at dispatch time. Any
        # unknown tool name is a code error, not a best-effort external call.
        if query.tool_type not in self.allowed_tools:
            raise ValueError(f"{query.tool_type} is not in tool contract")
        if not self.settings.enable_real_tools:
            return _unavailable(query, "Real tool execution disabled by TIBLENDER_ENABLE_REAL_TOOLS=false.")
        service_name = _SERVICE_TOOL_MAP.get(query.tool_type)
        service = self.services.get(service_name) if service_name else None
        # The fixture-mode short-circuit lives inside the same try/except as
        # the live dispatch so a corrupted or unreadable recording degrades to
        # an unavailable ToolResult exactly like a failed live call, instead
        # of an uncaught exception crashing the whole investigation cycle.
        try:
            if service is not None and service.mode == "fixture":
                return self._recording_lookup_or_unavailable(query)
            return {
                "SEARCH_ENGINE": self._search_engine,
                "WHOIS_LOOKUP": self._whois_lookup,
                "PASSIVE_DNS_LOOKUP": self._passive_dns_lookup,
                "TLS_CT_LOOKUP": self._tls_ct_lookup,
                "WEB_CRAWL": self._web_crawl,
                "NVD_CVE_LOOKUP": self._nvd_cve_lookup,
                "GITHUB_SEARCH": self._github_search,
                "X_SEARCH": self._x_search,
                "REDDIT_SEARCH": self._reddit_search,
                "DISCORD_SEARCH": self._discord_search,
                "TELEGRAM_SEARCH": self._telegram_search,
            }[query.tool_type](query)
        except ToolUnavailableError as error:
            # auto mode keeps a recording as a second line of defense when a
            # live provider is temporarily unavailable; live mode never
            # substitutes recorded evidence for a live provider response.
            if service is not None and self.settings.mode == "auto":
                payload, _ = self._lookup_recording(query)
                if payload is not None:
                    return _recording_result(query, payload, passive_dns_limit=self.settings.passive_dns_limit)
            return _unavailable(query, f"{query.tool_type} unavailable: {error}")
        except Exception as error:  # noqa: BLE001
            return _failed(query, error)

    def _web_crawl(self, query: ToolQuery) -> ToolResult:
        url = _param(query, "url")
        if not url:
            return _unavailable(query, "WEB_CRAWL requires a URL parameter.")
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return _unavailable(query, f"WEB_CRAWL received an invalid URL: {url}")
        render_meta: dict[str, Any] = {"javascript_rendered": False, "blocked_resource_types": []}
        timeout = query.timeout_seconds or self.settings.tool_timeout_seconds
        try:
            # Render JavaScript when enabled, then fall back to static HTTP if
            # the browser runtime is unavailable in a compact environment.
            if self.settings.web_crawl_render_js:
                text, final_url, render_meta = _render_javascript_page(
                    url,
                    timeout=timeout,
                    max_bytes=self.settings.web_crawl_max_bytes,
                    wait_ms=self.settings.web_crawl_wait_ms,
                )
            else:
                text, final_url = _http_text(
                    url,
                    timeout=timeout,
                    max_bytes=self.settings.web_crawl_max_bytes,
                )
        except ToolUnavailableError as error:
            text, final_url = _http_text(
                url,
                timeout=timeout,
                max_bytes=self.settings.web_crawl_max_bytes,
            )
            render_meta = {
                "javascript_rendered": False,
                "blocked_resource_types": [],
                "fallback_reason": _machine_independent_reason(error),
            }
        plain = _plain_text_from_html(text)
        title_match = re.search(r"(?is)<title[^>]*>(.*?)</title>", text)
        title = _plain_text_from_html(title_match.group(1)) if title_match else ""
        excerpt = plain[:700]
        mode = "JavaScript-rendered" if render_meta.get("javascript_rendered") else "Static"
        summary = f"{mode} crawl of {final_url}. Title: {title or 'n/a'}. Excerpt: {excerpt}"
        return _result(
            query,
            summary=summary,
            result_url=final_url,
            discovered_iocs=extract_iocs(plain),
            raw={"title": title, "excerpt": excerpt, **render_meta},
        )

    def _whois_lookup(self, query: ToolQuery) -> ToolResult:
        target = _param(query, "domain", "ip")
        if not target:
            return _unavailable(query, "WHOIS_LOOKUP requires domain or ip.")
        kind = "ip" if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", target) else "domain"
        if kind == "domain" and not filter_reportable_iocs(IocSet(domains=[target.lower()])).domains:
            return _result(
                query,
                summary=f"WHOIS lookup skipped contextual or non-reportable domain {target}.",
                discovered_iocs=IocSet(),
                confidence=IocConfidence.LOW,
                raw={"skipped": True, "reason": "non_reportable_domain"},
            )
        url = f"https://rdap.org/{kind}/{urllib.parse.quote(target, safe='')}"
        data, final_url = _http_json(
            url,
            timeout=query.timeout_seconds or self.settings.tool_timeout_seconds,
            max_bytes=self.settings.tool_max_bytes,
        )
        nameservers = [item.get("ldhName", "") for item in data.get("nameservers", []) if isinstance(item, dict)]
        events = [f"{item.get('eventAction')}={item.get('eventDate')}" for item in data.get("events", [])[:4] if isinstance(item, dict)]
        entities = [entity.get("handle", "") for entity in data.get("entities", [])[:4] if isinstance(entity, dict)]
        summary = (
            f"RDAP {kind} lookup for {target}: handle={data.get('handle', 'n/a')}; "
            f"status={','.join(data.get('status', [])[:4]) if isinstance(data.get('status'), list) else 'n/a'}; "
            f"nameservers={','.join(nameservers[:4]) or 'n/a'}; events={'; '.join(events) or 'n/a'}; "
            f"entities={','.join(entities) or 'n/a'}."
        )
        discovered = IocSet(ips=[target]) if kind == "ip" else IocSet(domains=[target.lower()])
        return _result(query, summary=summary, result_url=final_url, discovered_iocs=discovered, raw=data)

    def _passive_dns_lookup(self, query: ToolQuery) -> ToolResult:
        # VirusTotal is the only passive DNS provider.
        target = _param(query, "domain", "ip")
        if not target:
            return _unavailable(query, "PASSIVE_DNS_LOOKUP requires a domain or ip parameter.")
        return self._passive_dns_lookup_virustotal(query, target)

    def _passive_dns_lookup_virustotal(self, query: ToolQuery, target: str) -> ToolResult:
        if not self.settings.virustotal_api_key:
            return _unavailable(
                query,
                "PASSIVE_DNS_LOOKUP provider virustotal requires VIRUSTOTAL_API_KEY; none configured.",
            )
        is_ip = bool(re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", target))
        collection = "ip_addresses" if is_ip else "domains"
        # The deployed tool truncates the provider payload to its first 20
        # resolutions; asking VirusTotal for exactly that many keeps the live
        # path identical instead of fetching rows it would then discard.
        limit = self.settings.passive_dns_limit
        url = (
            f"https://www.virustotal.com/api/v3/{collection}/"
            f"{urllib.parse.quote(target, safe='')}/resolutions?limit={limit}"
        )
        headers = _headers({"Accept": "application/json", "x-apikey": self.settings.virustotal_api_key})
        timeout = max(query.timeout_seconds, 45)
        try:
            data, final_url = _vt_get_json(
                url, headers=headers, timeout=timeout, max_bytes=self.settings.tool_max_bytes
            )
        except _VirusTotalHttpError as error:
            return _unavailable(query, _vt_unavailable_reason(error))
        records = data.get("data", []) if isinstance(data, dict) else []
        rows: list[dict[str, Any]] = []
        for item in records[:limit]:
            if not isinstance(item, dict):
                continue
            attributes = item.get("attributes")
            if not isinstance(attributes, dict):
                continue
            host_stats = attributes.get("host_name_last_analysis_stats") or {}
            ip_stats = attributes.get("ip_address_last_analysis_stats") or {}
            rows.append(
                {
                    "ip_address": str(attributes.get("ip_address", "")),
                    "host_name": str(attributes.get("host_name", "")),
                    "date": attributes.get("date", ""),
                    "resolver": str(attributes.get("resolver", "")),
                    "host_name_malicious": int(host_stats.get("malicious", 0) or 0),
                    "host_name_suspicious": int(host_stats.get("suspicious", 0) or 0),
                    "ip_address_malicious": int(ip_stats.get("malicious", 0) or 0),
                    "ip_address_suspicious": int(ip_stats.get("suspicious", 0) or 0),
                }
            )
        # The queried side is already known; the *other* side of each
        # resolution is what pivots the investigation forward.
        pivots = [row["host_name"] if is_ip else row["ip_address"] for row in rows]
        pivots = [value for value in pivots if value]
        pairs = [f"{row['host_name']} {row['ip_address']}".strip() for row in rows]
        summary = (
            f"VirusTotal passive DNS for {target}: {len(rows)} resolution(s); "
            f"{'hosts' if is_ip else 'addresses'}: {', '.join(_dedupe(pivots)[:10]) or 'n/a'}"
        )
        return _result(
            query,
            summary=summary,
            result_url=final_url,
            discovered_iocs=extract_iocs("\n".join(pairs)),
            raw={
                "provider": "virustotal",
                # total_count and the detection counts describe every parsed
                # resolution; "records" is a bounded sample of the same rows,
                # so the names say which population each number covers.
                "total_count": len(rows),
                "malicious_count": sum(row["host_name_malicious"] + row["ip_address_malicious"] for row in rows),
                "suspicious_count": sum(row["host_name_suspicious"] + row["ip_address_suspicious"] for row in rows),
                "records": rows[: self.settings.max_tool_results],
                "records_truncated": len(rows) > self.settings.max_tool_results,
            },
        )

    def _tls_ct_lookup(self, query: ToolQuery) -> ToolResult:
        domain = _param(query, "domain")
        if not domain:
            return _unavailable(query, "TLS_CT_LOOKUP requires a domain parameter.")
        timeout = query.timeout_seconds or self.settings.tool_timeout_seconds
        url = f"https://crt.sh/?q=%25.{urllib.parse.quote(domain)}&output=json"
        provider = "crt.sh"
        try:
            data, final_url = _http_json(url, timeout=timeout, max_bytes=self.settings.tool_max_bytes)
        except (ToolUnavailableError, RuntimeError):
            # crt.sh is useful but intermittently returns 502/timeouts. The
            # production pipeline treats CT as a provider class, so this
            # compact runtime falls back to Cert Spotter's public issuance API.
            provider = "certspotter"
            params = urllib.parse.urlencode(
                {
                    "domain": domain,
                    "include_subdomains": "true",
                    "expand": "dns_names",
                }
            )
            url = f"https://api.certspotter.com/v1/issuances?{params}"
            try:
                data, final_url = _http_json(url, timeout=timeout, max_bytes=self.settings.tool_max_bytes)
            except (ToolUnavailableError, RuntimeError):
                return _unavailable(
                    query,
                    f"TLS CT providers unavailable for {domain}; attempted crt.sh and certspotter.",
                )
        if not isinstance(data, list):
            return _unavailable(query, f"TLS CT lookup returned unexpected response for {domain}.")
        names = []
        issuers = []
        for item in data[: self.settings.max_tool_results]:
            if isinstance(item, dict):
                if isinstance(item.get("dns_names"), list):
                    names.extend(str(name) for name in item["dns_names"])
                names.extend(str(item.get("name_value", "")).splitlines())
                if item.get("issuer_name"):
                    issuers.append(str(item["issuer_name"]))
        text = "\n".join(names)
        summary = (
            f"Certificate transparency lookup for {domain}: {len(data)} certificate row(s); "
            f"sample names={', '.join(names[:5]) or 'n/a'}."
        )
        return _result(
            query,
            summary=summary,
            result_url=final_url,
            discovered_iocs=extract_iocs(text),
            raw={"provider": provider, "count": len(data), "names": names[:20], "issuers": issuers[:5]},
        )

    def _nvd_cve_lookup(self, query: ToolQuery) -> ToolResult:
        cve = _param(query, "cve").upper()
        if not cve:
            return _unavailable(query, "NVD_CVE_LOOKUP requires a cve parameter.")
        url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cveId={urllib.parse.quote(cve)}"
        headers = _headers({"Accept": "application/json"})
        if self.settings.nvd_api_key:
            headers["apiKey"] = self.settings.nvd_api_key
        try:
            data, final_url = _http_json(
                url,
                headers=headers,
                timeout=query.timeout_seconds or self.settings.tool_timeout_seconds,
                max_bytes=self.settings.tool_max_bytes,
            )
        except ToolUnavailableError:
            if not self.settings.nvd_api_key:
                raise
            retry_headers = dict(headers)
            retry_headers.pop("apiKey", None)
            data, final_url = _http_json(
                url,
                headers=retry_headers,
                timeout=query.timeout_seconds or self.settings.tool_timeout_seconds,
                max_bytes=self.settings.tool_max_bytes,
            )
        vulnerabilities = data.get("vulnerabilities", []) if isinstance(data, dict) else []
        if not vulnerabilities:
            return _unavailable(query, f"NVD did not return details for {cve}.")
        cve_data = vulnerabilities[0].get("cve", {}) if isinstance(vulnerabilities[0], dict) else {}
        descriptions = cve_data.get("descriptions", []) if isinstance(cve_data, dict) else []
        description = next((item.get("value", "") for item in descriptions if item.get("lang") == "en"), "")
        metrics = cve_data.get("metrics", {}) if isinstance(cve_data, dict) else {}
        summary = f"NVD lookup for {cve}: {description[:700] or 'description unavailable'}"
        return _result(query, summary=summary, result_url=final_url, discovered_iocs=IocSet(cves=[cve]), raw={"id": cve, "description": description, "metrics": metrics})

    def _github_search(self, query: ToolQuery) -> ToolResult:
        search = _param(query, "query", "cve", "hash")
        if not search:
            return _unavailable(query, "GITHUB_SEARCH requires query, cve, or hash.")
        url = f"https://api.github.com/search/repositories?q={urllib.parse.quote(search)}&sort=updated&order=desc&per_page={self.settings.max_tool_results}"
        headers = _headers({"Accept": "application/vnd.github+json"})
        if self.settings.github_token:
            headers["Authorization"] = f"Bearer {self.settings.github_token}"
        data, final_url = _http_json(
            url,
            headers=headers,
            timeout=query.timeout_seconds or self.settings.tool_timeout_seconds,
            max_bytes=self.settings.tool_max_bytes,
        )
        items = data.get("items", []) if isinstance(data, dict) else []
        summaries = [
            f"{item.get('full_name')}: {item.get('description') or ''}"
            for item in items[: self.settings.max_tool_results]
            if isinstance(item, dict)
        ]
        summary = f"GitHub repository search for {search}: {len(items)} result(s). " + " | ".join(summaries[:3])
        return _result(query, summary=summary, result_url=final_url, discovered_iocs=extract_iocs(_search_result_text(items, "full_name", "description", "html_url")), raw={"result_count": len(items), "items": items[: self.settings.max_tool_results]})

    def _search_engine(self, query: ToolQuery) -> ToolResult:
        # The public runtime keeps the production-compatible SearXNG path as
        # the single search implementation to avoid provider-specific behavior.
        search = _param(query, "query")
        if not search:
            return _unavailable(query, "SEARCH_ENGINE requires a query parameter.")
        if self.settings.search_engine_provider != "searxng":
            return _unavailable(query, "SEARCH_ENGINE supports the production SearXNG provider in this runtime.")
        max_results = int(query.parameters.get("max_results") or self.settings.search_engine_max_results)
        searxng_query = _build_searxng_query(
            search,
            include_domains=query.parameters.get("include_domains"),
            exclude_domains=query.parameters.get("exclude_domains"),
        )
        params = urllib.parse.urlencode({"q": searxng_query, "format": "json"})
        url = f"{self.settings.searxng_base_url.rstrip('/')}/search?{params}"
        headers = _headers(
            {
                "Accept": "application/json",
                "X-Forwarded-For": "127.0.0.1",
                "X-Real-IP": "127.0.0.1",
            }
        )
        try:
            payload, final_url = _http_json(
                url,
                headers=headers,
                timeout=query.timeout_seconds or self.settings.tool_timeout_seconds,
                max_bytes=self.settings.tool_max_bytes,
            )
        except ToolUnavailableError:
            corpus_result = self._corpus_search_result(
                query,
                platform=None,
                label="SearXNG-compatible",
                include_iocs=False,
            )
            if corpus_result:
                return corpus_result
            raise ToolUnavailableError("SearXNG search did not return usable results.")
        items = _normalize_searxng_results(payload, max_results=max_results)
        snippet_iocs = _search_snippet_iocs(items, search)
        summary = f"SearXNG search for {search}: {len(items)} result(s). Result titles are retained only in internal tool payloads."
        return _result(query, summary=summary, result_url=final_url, discovered_iocs=snippet_iocs, raw={"result_count": len(items), "items": items})
