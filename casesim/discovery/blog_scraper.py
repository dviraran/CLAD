"""Law firm blog scraper for extracting case citations.

This module scrapes medical malpractice law firm blogs to extract
case citations, which can then be resolved to primary legal databases.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import AsyncIterator

import httpx
from bs4 import BeautifulSoup

from ..utils import get_logger


@dataclass
class BlogSource:
    """A law firm blog to scrape for case citations."""

    name: str
    url: str
    article_selector: str  # CSS selector for article links
    content_selector: str  # CSS selector for article content
    jurisdiction: str  # Primary jurisdiction (UK, US, CA, AU)
    max_articles: int = 50


# Curated list of medical malpractice law firm blogs
# These are public blogs with case discussions
MALPRACTICE_BLOGS = [
    # UK blogs
    BlogSource(
        name="Leigh Day Medical Negligence",
        url="https://www.leighday.co.uk/latest-updates/?category=medical-negligence",
        article_selector="a.card__link",
        content_selector="article .content",
        jurisdiction="UK",
    ),
    BlogSource(
        name="Irwin Mitchell Clinical Negligence",
        url="https://www.irwinmitchell.com/news-and-insights",
        article_selector="a.news-card__link",
        content_selector="article .article-content",
        jurisdiction="UK",
    ),
    # US blogs
    BlogSource(
        name="National Law Review Medical Malpractice",
        url="https://www.natlawreview.com/basic-industries/medical-malpractice",
        article_selector="h2.title a",
        content_selector="div.article-content",
        jurisdiction="US",
    ),
    BlogSource(
        name="Lexology Medical Malpractice",
        url="https://www.lexology.com/search?q=medical%20malpractice",
        article_selector="a.article-link",
        content_selector="div.article-body",
        jurisdiction="US",
    ),
    # Canada blogs
    BlogSource(
        name="CMPA Case Studies",
        url="https://www.cmpa-acpm.ca/en/advice-publications/browse-articles",
        article_selector="a.article-title",
        content_selector="div.article-content",
        jurisdiction="CA",
    ),
]


class BlogCitationExtractor:
    """Extract case citations from law firm blog posts."""

    # Citation patterns for different jurisdictions
    CITATION_PATTERNS = {
        "UK": [
            # Neutral citations: [2019] EWHC 936 (QB)
            r"\[(\d{4})\]\s+(EWHC|EWCA|UKSC|UKHL|EWCOP|CSOH|CSIH|NIQB)\s+(\d+)\s*(?:\(([A-Za-z]+)\))?",
            # WLR citations: [2019] 1 WLR 123
            r"\[(\d{4})\]\s+(\d+)\s+(WLR|All\s*ER|Med\s*LR|BMLR)\s+(\d+)",
        ],
        "US": [
            # US Reports: 123 U.S. 456 (2019)
            r"(\d+)\s+U\.S\.\s+(\d+)\s*\((\d{4})\)",
            # Federal Reporter: 123 F.3d 456
            r"(\d+)\s+F\.\s*(2d|3d|4th)?\s+(\d+)",
            # Federal Supplement: 123 F. Supp. 2d 456
            r"(\d+)\s+F\.\s*Supp\.\s*(2d|3d)?\s+(\d+)",
        ],
        "CA": [
            # Neutral citations: 2019 ONSC 1234
            r"(\d{4})\s+(ONSC|ONCA|BCSC|BCCA|ABQB|ABCA|SCC|QCCS|QCCA)\s+(\d+)",
            # CanLII citations: 2019 CanLII 12345 (ON SC)
            r"(\d{4})\s+CanLII\s+(\d+)\s*\(([A-Z]{2}\s*[A-Z]{2,4})\)",
        ],
        "AU": [
            # Australian neutral citations: [2019] NSWSC 1234
            r"\[(\d{4})\]\s+(NSWSC|NSWCA|VSC|VCA|HCA|QSC|QCA|WASC|WASCA)\s+(\d+)",
        ],
    }

    def __init__(self, rate_limit_delay: float = 2.0):
        """Initialize the extractor.

        Args:
            rate_limit_delay: Seconds to wait between requests (be polite)
        """
        self.logger = get_logger("blog_scraper")
        self.rate_limit_delay = rate_limit_delay
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> None:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "CaseSim Academic Research Bot (medical malpractice case discovery)",
                    "Accept": "text/html,application/xhtml+xml",
                },
            )

    async def extract_from_blog(
        self,
        blog: BlogSource,
        max_articles: int | None = None,
    ) -> AsyncIterator[tuple[str, str, str]]:
        """Extract citations from a blog.

        Args:
            blog: Blog source configuration
            max_articles: Override max articles to fetch

        Yields:
            Tuples of (citation_string, source_article_url, jurisdiction)
        """
        await self._ensure_client()
        assert self._client is not None

        articles_to_fetch = max_articles or blog.max_articles

        try:
            self.logger.info(f"Fetching article list from {blog.name}")

            # Get article list
            response = await self._client.get(blog.url)
            if response.status_code != 200:
                self.logger.warning(f"Failed to fetch {blog.url}: {response.status_code}")
                return

            soup = BeautifulSoup(response.text, "lxml")
            article_links = soup.select(blog.article_selector)[:articles_to_fetch]

            self.logger.info(f"Found {len(article_links)} articles to process")

            for link in article_links:
                href = link.get("href", "")
                if not href:
                    continue

                # Make URL absolute
                if not href.startswith("http"):
                    from urllib.parse import urljoin
                    href = urljoin(blog.url, href)

                # Rate limit
                import asyncio
                await asyncio.sleep(self.rate_limit_delay)

                # Fetch article
                try:
                    self.logger.debug(f"Fetching article: {href}")
                    article_response = await self._client.get(href)

                    if article_response.status_code != 200:
                        continue

                    article_soup = BeautifulSoup(article_response.text, "lxml")
                    content_elem = article_soup.select_one(blog.content_selector)

                    if not content_elem:
                        # Fall back to body
                        content_elem = article_soup.find("body")

                    if content_elem:
                        text = content_elem.get_text()

                        # Extract citations for this jurisdiction
                        patterns = self.CITATION_PATTERNS.get(blog.jurisdiction, [])
                        for pattern in patterns:
                            for match in re.finditer(pattern, text):
                                citation = match.group(0)
                                yield (citation, href, blog.jurisdiction)

                except Exception as e:
                    self.logger.debug(f"Error fetching article {href}: {e}")
                    continue

        except Exception as e:
            self.logger.error(f"Error scraping blog {blog.name}: {e}")

    async def extract_from_all_blogs(
        self,
        jurisdictions: list[str] | None = None,
    ) -> AsyncIterator[tuple[str, str, str]]:
        """Extract citations from all configured blogs.

        Args:
            jurisdictions: Filter to specific jurisdictions (e.g., ["UK", "US"])

        Yields:
            Tuples of (citation_string, source_article_url, jurisdiction)
        """
        seen_citations: set[str] = set()

        for blog in MALPRACTICE_BLOGS:
            if jurisdictions and blog.jurisdiction not in jurisdictions:
                continue

            async for citation, source_url, jurisdiction in self.extract_from_blog(blog):
                # Deduplicate
                if citation not in seen_citations:
                    seen_citations.add(citation)
                    yield (citation, source_url, jurisdiction)

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
