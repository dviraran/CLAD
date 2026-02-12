"""Singapore eLitigation source connector."""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, DiscoveryStatus, Jurisdiction, Source
from ..utils import generate_case_id
from .base import BaseSource, SearchResult


class SingaporeSource(BaseSource):
    """Connector for Singapore eLitigation legal database.

    Singapore provides free public access to court judgments through
    the eLitigation portal. Judgments are in English with very clear,
    structured reasoning. Singapore courts explicitly discuss professional
    guidelines, making them valuable for medical negligence simulations.
    """

    source = Source.SINGAPORE
    jurisdiction = Jurisdiction.SG
    base_url = "https://www.elitigation.sg"

    # Singapore courts relevant for medical negligence
    RELEVANT_COURTS = {
        "SGCA": "Court of Appeal",
        "SGHC": "High Court",
        "SGHCF": "High Court (Family Division)",
        "SGDC": "District Court",
        "SGMC": "Magistrates' Court",
        "SGHCR": "High Court Registrar",
    }

    # Citation patterns for Singapore cases
    CITATION_PATTERNS = [
        # Neutral citations: [2019] SGCA 1, [2019] SGHC 123
        r"\[(\d{4})\]\s+(SGCA|SGHC|SGHCF|SGDC|SGMC|SGHCR)\s+(\d+)",
        # SLR citations: [2019] 1 SLR 123
        r"\[(\d{4})\]\s+(\d+)\s+SLR(?:\(R\))?\s+(\d+)",
        # MLJ citations: [2019] 1 MLJ 123
        r"\[(\d{4})\]\s+(\d+)\s+MLJ\s+(\d+)",
    ]

    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search Singapore eLitigation for cases matching keywords.

        Singapore eLitigation uses a different search approach - we browse
        by year and filter results containing our keywords from the listing.
        """
        await self._ensure_client()
        assert self._client is not None

        # Build search query for text matching
        query_lower = " ".join(keywords).lower()

        # eLitigation search URL - use the index page with filters
        search_url = f"{self.base_url}/gd/Home/Index"

        # Determine year range
        start_year = int(date_from[:4]) if date_from else 2015
        end_year = int(date_to[:4]) if date_to else 2026

        results_found = 0

        # Search through recent years
        for year in range(end_year, start_year - 1, -1):
            if results_found >= max_results:
                break

            params = {
                "Filter": "SUPCT",
                "YearOfDecision": str(year),
                "SortBy": "DateOfDecision",
                "CurrentPage": "1",
                "SortAscending": "False",
            }

            try:
                await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
                headers = {
                    "Referer": f"{self.base_url}/gd/",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }
                response = await self._client.get(search_url, params=params, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")

                # Parse case listings - look for case links and titles
                for item in soup.select("a[href*='/gd/s/'], .case-title, .judgment-link, tr td a"):
                    if results_found >= max_results:
                        break

                    result = self._parse_search_result(item)
                    if result:
                        # Filter by keywords in title or snippet
                        title_lower = result.title.lower() if result.title else ""
                        snippet_lower = result.snippet.lower() if result.snippet else ""

                        # Check if any keyword matches
                        if any(kw.lower() in title_lower or kw.lower() in snippet_lower for kw in keywords):
                            # Apply court filter
                            if court and result.court != court:
                                continue

                            results_found += 1
                            yield result

            except Exception as e:
                self.logger.warning(f"Error searching year {year}: {e}")
                continue

    def _parse_search_result(self, item: BeautifulSoup) -> SearchResult | None:
        """Parse a single search result item."""
        link = item.find("a", href=True)
        if not link:
            return None

        href = link.get("href", "")
        if not href or not self._is_case_url(href):
            return None

        url = urljoin(self.base_url, href)
        title = link.get_text(strip=True)

        if not title:
            return None

        # Extract snippet
        snippet = None
        snippet_elem = item.find(class_=["snippet", "summary", "case-summary"])
        if snippet_elem:
            snippet = snippet_elem.get_text(strip=True)[:500]
        else:
            text = item.get_text(strip=True)
            if text and len(text) > len(title):
                snippet = text[len(title):].strip()[:500]

        # Extract date and court
        date = self._extract_date(url, title)
        court = self._extract_court(url, title)
        citation = self._extract_citation(title)

        return SearchResult(
            url=url,
            title=title,
            snippet=snippet,
            date=date,
            court=court,
            citation=citation,
        )

    def _is_case_url(self, href: str) -> bool:
        """Check if URL is a case page."""
        if not href:
            return False

        # eLitigation case URLs typically contain /gd/ or /gdp/
        if "/gd/" not in href and "/gdp/" not in href:
            return False

        exclude_patterns = ["search", "login", "register", "about", "help", "faq"]
        return not any(p in href.lower() for p in exclude_patterns)

    def _extract_date(self, url: str, title: str) -> str | None:
        """Extract year from URL or title."""
        # URL pattern: /2019/
        match = re.search(r"/(\d{4})/", url)
        if match:
            return match.group(1)

        # Title pattern: [2019]
        match = re.search(r"\[(\d{4})\]", title)
        if match:
            return match.group(1)

        return None

    def _extract_court(self, url: str, title: str) -> str | None:
        """Extract court from URL or title."""
        combined = f"{url} {title}".upper()
        for court_code in self.RELEVANT_COURTS:
            if court_code in combined:
                return court_code
        return None

    def _extract_citation(self, text: str) -> str | None:
        """Extract citation from text."""
        for pattern in self.CITATION_PATTERNS:
            match = re.search(pattern, text)
            if match:
                return match.group(0)
        return None

    def build_discovery_record(
        self,
        search_result: SearchResult,
        discovery_methods: list[str],
        query_terms: list[str] | None = None,
    ) -> DiscoveryRecord:
        """Build a DiscoveryRecord from a search result."""
        # Generate case ID
        url_parts = urlparse(search_result.url).path.strip("/").split("/")
        identifier = "-".join(url_parts[-3:]) if len(url_parts) >= 3 else "-".join(url_parts)
        case_id = generate_case_id("singapore", identifier)

        year = None
        if search_result.date:
            try:
                year = int(search_result.date[:4])
            except (ValueError, TypeError):
                pass

        priority = self._calculate_priority(search_result, query_terms)

        return DiscoveryRecord(
            case_id=case_id,
            source=self.source,
            jurisdiction=self.jurisdiction,
            court=search_result.court,
            title=search_result.title,
            year=year,
            url=search_result.url,
            discovery_methods=discovery_methods,
            query_terms=query_terms,
            estimated_length=search_result.estimated_length,
            priority_score=priority,
            status=DiscoveryStatus.QUEUED,
        )

    def _calculate_priority(
        self, result: SearchResult, query_terms: list[str] | None
    ) -> float:
        """Calculate priority score."""
        score = 0.5

        # Boost for appellate courts
        if result.court == "SGCA":
            score += 0.2
        elif result.court == "SGHC":
            score += 0.15
        elif result.court == "SGHCF":
            score += 0.1

        # Boost for medical terms
        if result.snippet:
            medical_terms = [
                "negligence", "malpractice", "surgery", "patient", "doctor",
                "hospital", "diagnosis", "treatment", "consent", "breach",
                "medical", "clinical", "duty of care", "standard of care",
                "singapore medical council", "professional misconduct",
            ]
            snippet_lower = result.snippet.lower()
            matches = sum(1 for t in medical_terms if t in snippet_lower)
            score += min(0.2, matches * 0.04)

        return min(1.0, score)

    def extract_citations(self, content: str) -> list[str]:
        """Extract citations from document content."""
        citations = set()
        for pattern in self.CITATION_PATTERNS:
            for match in re.finditer(pattern, content):
                citations.add(match.group(0))
        return list(citations)

    def parse_citation_url(self, citation: str) -> str | None:
        """Convert citation to eLitigation URL.

        Singapore eLitigation URLs are not directly constructible from
        citations as they use internal document IDs. Returns None.
        Citation resolution would require a search API call.
        """
        # Singapore eLitigation doesn't support direct citation-to-URL mapping
        # Cases are accessed via internal document IDs
        return None
