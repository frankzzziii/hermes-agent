"""Crawl4ai extract provider — self-hosted, Bearer token auth (optional)."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List

import httpx
from agent.web_search_provider import WebSearchProvider, get_provider_env
from tools.url_safety import is_safe_url

logger = logging.getLogger(__name__)


# Configuration keys we read.  Hermes convention supports both the flat
# ``web.crawl4ai_url`` key (used by upstream PR #59300) and the nested
# ``web.crawl4ai.url`` key (what our local config already uses).  The nested
# form is checked first so an explicit local profile config wins.
_CRAWL4AI_URL_CFG_KEYS = ("crawl4ai.url", "crawl4ai_url")
_CRAWL4AI_API_TOKEN_CFG_KEYS = ("crawl4ai.api_key", "crawl4ai_api_key")

# Behaviour tuning (all overridable via web.crawl4ai.* in config.yaml):
#   filter            — /md content filter: fit (default) | raw | bm25 | llm
#   min_content_chars — below this, a result is treated as an SPA empty shell
#   spa_fallback      — retry with cache-bust, then /crawl with render delay
_DEFAULTS = {
    "filter": "fit",
    "min_content_chars": 100,
    "spa_fallback": True,
    "crawl_delay_seconds": 3.0,
    "crawl_page_timeout_ms": 45000,
}


def _read_nested_config(cfg: Dict[str, Any], dotted_key: str) -> Any:
    """Read a dotted config key from a dict, returning None on miss."""
    cur = cfg
    for segment in dotted_key.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(segment)
    return cur


def _crawl4ai_setting(key: str) -> Any:
    """Read a web.crawl4ai.<key> tuning value, falling back to _DEFAULTS."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("web", {})
        val = _read_nested_config(cfg, f"crawl4ai.{key}")
        if val is not None:
            return val
    except Exception:
        pass
    return _DEFAULTS[key]


def _crawl4ai_url() -> str:
    """Return Crawl4AI URL from config.yaml, then Hermes env, then process env."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("web", {})
        for key in _CRAWL4AI_URL_CFG_KEYS:
            val = _read_nested_config(cfg, key)
            if isinstance(val, str) and val.strip():
                return val.strip().rstrip("/")
    except Exception:
        pass

    # Hermes config-aware env (get_env_value checks .env files)
    val = get_provider_env("CRAWL4AI_URL")
    if val:
        return val.rstrip("/")

    # Process env fallback
    return (os.getenv("CRAWL4AI_URL", "") or "").strip().rstrip("/")


def _crawl4ai_token() -> str:
    """Return Crawl4AI API token (JWT) from Hermes env, then process env.

    The token is optional: default Docker deployments of Crawl4AI 0.8.x do
    not require authentication on the /md and /crawl endpoints.  When a token
    is configured we send it as a Bearer token; otherwise the request is
    sent unauthenticated.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config().get("web", {})
        for key in _CRAWL4AI_API_TOKEN_CFG_KEYS:
            val = _read_nested_config(cfg, key)
            if isinstance(val, str):
                return val.strip()
    except Exception:
        pass

    return get_provider_env("CRAWL4AI_API_TOKEN")


class Crawl4aiWebExtractProvider(WebSearchProvider):
    """Crawl4ai extraction provider using the /md endpoint.

    Supports self-hosted Crawl4AI instances.  Auth is optional: a Bearer
    token is only sent when ``CRAWL4AI_API_TOKEN`` / ``web.crawl4ai.api_key``
    is configured.
    """

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4ai (self-hosted)"

    def is_available(self) -> bool:
        """Available whenever a Crawl4AI URL is configured.

        We intentionally do NOT require a token here so default unauthenticated
        local Docker deployments work out of the box.
        """
        return bool(_crawl4ai_url())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        format = kwargs.get("format", "markdown")
        results: List[Dict[str, Any]] = []

        base_url = _crawl4ai_url()
        token = _crawl4ai_token()
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        content_filter = str(_crawl4ai_setting("filter"))
        min_chars = int(_crawl4ai_setting("min_content_chars"))
        spa_fallback = bool(_crawl4ai_setting("spa_fallback"))

        async with httpx.AsyncClient(timeout=120.0) as client:
            for url in urls:
                if not is_safe_url(url):
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": "Blocked: URL targets a private or internal network address",
                    })
                    continue

                try:
                    logger.info("Crawl4ai extracting: %s", url)
                    markdown = await self._fetch_md(
                        client, base_url, url, headers, content_filter,
                    )

                    # JS-heavy SPAs sometimes return an empty/pruned shell when
                    # /md snapshots before hydration completes.  Inspired by the
                    # official crawl4ai skill: retry with a cache-bust revision,
                    # then fall back to /crawl with an explicit render delay.
                    if spa_fallback and len(markdown) < min_chars:
                        logger.info(
                            "Crawl4ai /md thin (%d chars) for %s, cache-bust retry",
                            len(markdown), url,
                        )
                        markdown = await self._fetch_md(
                            client, base_url, url, headers, content_filter,
                            cache_bust=str(int(time.time())),
                        )

                    if spa_fallback and len(markdown) < min_chars:
                        logger.info(
                            "Crawl4ai /md still thin for %s, falling back to /crawl",
                            url,
                        )
                        crawled = await self._fetch_crawl(
                            client, base_url, url, headers,
                        )
                        if len(crawled) > len(markdown):
                            markdown = crawled

                    title = self._extract_title(markdown, url)

                    results.append({
                        "url": url,
                        "title": title,
                        "content": markdown,
                        "raw_content": markdown,
                        "metadata": {},
                    })

                except httpx.HTTPStatusError as e:
                    logger.warning("Crawl4ai HTTP error for %s: %s", url, e)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Crawl4ai HTTP {e.response.status_code}: {e.response.text[:200]}",
                    })
                except Exception as e:  # noqa: BLE001
                    logger.warning("Crawl4ai extraction error for %s: %s", url, e)
                    results.append({
                        "url": url,
                        "title": "",
                        "content": "",
                        "error": f"Crawl4ai extraction failed: {e}",
                    })

        return results

    async def _fetch_md(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        url: str,
        headers: Dict[str, str],
        content_filter: str,
        cache_bust: str | None = None,
    ) -> str:
        """Single /md call. ``f`` selects the server-side content filter
        (fit=pruned, raw, bm25, llm); ``c`` is a cache-bust revision counter."""
        payload: Dict[str, Any] = {"url": url, "f": content_filter}
        if cache_bust:
            payload["c"] = cache_bust
        response = await client.post(f"{base_url}/md", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        markdown = data.get("markdown", "")
        if isinstance(markdown, dict):
            markdown = markdown.get("raw_markdown", "") or markdown.get("markdown", "")
        return markdown or ""

    async def _fetch_crawl(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        url: str,
        headers: Dict[str, str],
    ) -> str:
        """Fallback via /crawl with render delay so JS SPAs fully hydrate."""
        delay = float(_crawl4ai_setting("crawl_delay_seconds"))
        page_timeout = int(_crawl4ai_setting("crawl_page_timeout_ms"))
        response = await client.post(
            f"{base_url}/crawl",
            json={
                "urls": [url],
                "crawler_config": {
                    "delay_before_return_html": delay,
                    "page_timeout": page_timeout,
                },
            },
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        for item in data.get("results", []) or []:
            md = item.get("markdown", "")
            if isinstance(md, dict):
                md = md.get("raw_markdown", "") or md.get("fit_markdown", "")
            if md:
                return md
        return ""

    def _extract_title(self, markdown: str, fallback_url: str) -> str:
        if not markdown:
            return fallback_url
        for line in markdown.split("\n"):
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return fallback_url

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": self.display_name,
            "badge": "self-hosted",
            "tag": "Uses your self-hosted crawl4ai instance at web.crawl4ai.url or CRAWL4AI_URL",
            "env_vars": [
                {
                    "key": "CRAWL4AI_URL",
                    "prompt": "Crawl4ai instance URL (e.g., http://localhost:11235)",
                    "url": "https://github.com/unclecode/crawl4ai",
                },
                {
                    "key": "CRAWL4AI_API_TOKEN",
                    "prompt": "Crawl4ai JWT API token (optional for default local deployments)",
                    "url": "",
                },
            ],
        }
