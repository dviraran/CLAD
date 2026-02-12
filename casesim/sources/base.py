"""Base class for legal database source connectors."""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import urlparse

import httpx
from diskcache import Cache
from robotexclusionrulesparser import RobotExclusionRulesParser
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import get_settings
from ..schemas import DiscoveryRecord, Jurisdiction, Source
from ..utils import get_logger, hash_content, normalize_url


@dataclass
class FetchResult:
    """Result of fetching a document."""

    url: str
    content: str
    content_type: str
    status_code: int
    fetched_at: datetime
    content_hash: str
    cached: bool = False
    headers: dict[str, str] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Result from a search query."""

    url: str
    title: str
    snippet: str | None = None
    date: str | None = None
    court: str | None = None
    citation: str | None = None
    estimated_length: int | None = None


class RateLimiter:
    """Simple rate limiter for HTTP requests."""

    def __init__(self, requests_per_second: float = 1.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def acquire(self, host: str) -> None:
        """Wait until we can make a request to the given host."""
        async with self._lock:
            now = time.monotonic()
            last = self.last_request.get(host, 0)
            wait_time = max(0, self.min_interval - (now - last))
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request[host] = time.monotonic()


class BaseSource(ABC):
    """Base class for legal database sources."""

    source: Source
    jurisdiction: Jurisdiction
    base_url: str

    def __init__(self):
        self.settings = get_settings()
        self.logger = get_logger(f"sources.{self.__class__.__name__.lower()}")
        self._client: httpx.AsyncClient | None = None
        self._cache: Cache | None = None
        self._rate_limiter = RateLimiter(
            self.settings.rate_limit.requests_per_second
        )
        self._robots_parser: RobotExclusionRulesParser | None = None
        self._robots_checked = False

    async def __aenter__(self) -> "BaseSource":
        """Enter async context."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            # Check if SSL verification should be disabled (for development/testing)
            import os
            verify_ssl = os.environ.get("CASESIM_VERIFY_SSL", "true").lower() != "false"

            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0, connect=10.0),
                follow_redirects=True,
                verify=verify_ssl,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

        if self._cache is None and self.settings.cache.enabled:
            cache_dir = self.settings.cache.directory / self.source.value.lower()
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache = Cache(str(cache_dir))

    async def close(self) -> None:
        """Close resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        if self._cache:
            self._cache.close()
            self._cache = None

    async def _check_robots(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt."""
        if not self.settings.discovery.respect_robots_txt:
            return True

        if not self._robots_checked:
            await self._load_robots()
            self._robots_checked = True

        if self._robots_parser is None:
            return True

        return self._robots_parser.is_allowed("*", url)

    async def _load_robots(self) -> None:
        """Load robots.txt for the source."""
        await self._ensure_client()
        assert self._client is not None

        robots_url = f"{self.base_url}/robots.txt"
        try:
            response = await self._client.get(robots_url)
            if response.status_code == 200:
                self._robots_parser = RobotExclusionRulesParser()
                self._robots_parser.parse(response.text)
                self.logger.debug(f"Loaded robots.txt from {robots_url}")
        except Exception as e:
            self.logger.warning(f"Could not load robots.txt: {e}")

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=30),
    )
    async def fetch(self, url: str, use_cache: bool = True) -> FetchResult:
        """Fetch a document from the source."""
        await self._ensure_client()
        assert self._client is not None

        normalized_url = normalize_url(url)

        # Check cache first
        if use_cache and self._cache is not None:
            cached = self._cache.get(normalized_url)
            if cached is not None:
                self.logger.debug(f"Cache hit: {url}")
                return FetchResult(
                    url=url,
                    content=cached["content"],
                    content_type=cached["content_type"],
                    status_code=200,
                    fetched_at=datetime.fromisoformat(cached["fetched_at"]),
                    content_hash=cached["content_hash"],
                    cached=True,
                )

        # Check robots.txt
        if not await self._check_robots(url):
            raise PermissionError(f"URL blocked by robots.txt: {url}")

        # Rate limit
        host = urlparse(url).netloc
        await self._rate_limiter.acquire(host)

        # Fetch
        self.logger.debug(f"Fetching: {url}")
        response = await self._client.get(url)
        response.raise_for_status()

        content = response.text
        content_hash = hash_content(content)
        content_type = response.headers.get("content-type", "text/html")
        fetched_at = datetime.utcnow()

        result = FetchResult(
            url=url,
            content=content,
            content_type=content_type,
            status_code=response.status_code,
            fetched_at=fetched_at,
            content_hash=content_hash,
            cached=False,
            headers=dict(response.headers),
        )

        # Cache the result
        if self._cache is not None:
            self._cache.set(
                normalized_url,
                {
                    "content": content,
                    "content_type": content_type,
                    "fetched_at": fetched_at.isoformat(),
                    "content_hash": content_hash,
                },
                expire=self.settings.cache.ttl_days * 86400,
            )

        return result

    async def save_raw(self, url: str, content: str, case_id: str) -> Path:
        """Save raw content to disk."""
        raw_dir = self.settings.paths.raw_dir / self.source.value.lower()
        raw_dir.mkdir(parents=True, exist_ok=True)

        # Determine file extension
        ext = ".html"
        if "pdf" in url.lower():
            ext = ".pdf"

        file_path = raw_dir / f"{case_id}{ext}"
        file_path.write_text(content, encoding="utf-8")
        self.logger.debug(f"Saved raw content: {file_path}")
        return file_path

    @abstractmethod
    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search for cases matching the given criteria."""
        ...

    @abstractmethod
    def build_discovery_record(
        self,
        search_result: SearchResult,
        discovery_methods: list[str],
        query_terms: list[str] | None = None,
    ) -> DiscoveryRecord:
        """Build a DiscoveryRecord from a search result."""
        ...

    @abstractmethod
    def extract_citations(self, content: str) -> list[str]:
        """Extract citations to other cases from document content."""
        ...

    @abstractmethod
    def parse_citation_url(self, citation: str) -> str | None:
        """Convert a citation to a URL, if possible."""
        ...
