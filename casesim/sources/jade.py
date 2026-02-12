"""BarNet JADE (Judgments And Decisions Enhanced) source connector."""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, DiscoveryStatus, Jurisdiction, Source
from ..utils import generate_case_id
from .base import BaseSource, SearchResult


class JADESource(BaseSource):
    """Connector for BarNet JADE legal database.

    JADE provides enhanced access to Australian case law with superior
    citation tracking via CaseTrace. It indexes 244,000+ Australian
    court and tribunal decisions with citators and advanced search.

    JADE complements AustLII by providing:
    - CaseTrace citator (paragraph-level citations)
    - Modern search interface
    - API access for case retrieval by citation
    """

    source = Source.JADE
    jurisdiction = Jurisdiction.AU
    base_url = "https://jade.io"
    api_base_url = "http://jade.barnet.com.au"

    # Australian courts (same as AustLII, JADE indexes Australian cases)
    RELEVANT_COURTS = {
        "HCA": "High Court of Australia",
        "NSWSC": "New South Wales Supreme Court",
        "NSWCA": "New South Wales Court of Appeal",
        "VSC": "Victorian Supreme Court",
        "VSCA": "Victorian Court of Appeal",
        "QSC": "Queensland Supreme Court",
        "QCA": "Queensland Court of Appeal",
        "WASC": "Western Australia Supreme Court",
        "SASC": "South Australia Supreme Court",
        "FCAFC": "Federal Court Full Court",
        "FCA": "Federal Court of Australia",
        "TASSC": "Tasmania Supreme Court",
        "ACTSC": "ACT Supreme Court",
        "NTSC": "Northern Territory Supreme Court",
    }

    # State abbreviations for URL construction
    STATE_MAP = {
        "HCA": "cth",
        "FCA": "cth",
        "FCAFC": "cth",
        "NSWSC": "nsw",
        "NSWCA": "nsw",
        "VSC": "vic",
        "VSCA": "vic",
        "QSC": "qld",
        "QCA": "qld",
        "WASC": "wa",
        "SASC": "sa",
        "TASSC": "tas",
        "ACTSC": "act",
        "NTSC": "nt",
    }

    # Citation patterns for Australian cases
    CITATION_PATTERNS = [
        # Neutral citations: [2019] NSWSC 1234
        r"\[(\d{4})\]\s+(HCA|NSWSC|NSWCA|VSC|VSCA|QSC|QCA|WASC|SASC|FCA|FCAFC|TASSC|ACTSC|NTSC)\s+(\d+)",
        # CLR citations: (2019) 123 CLR 456
        r"\((\d{4})\)\s+(\d+)\s+(CLR|ALR|ALJR)\s+(\d+)",
        # State reports: [2019] 1 Qd R 123
        r"\[(\d{4})\]\s+(\d+)\s+(Qd R|NSWLR|VR|SASR|WAR)\s+(\d+)",
    ]

    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search JADE for cases matching keywords.

        JADE uses a web-based search interface. This method searches
        the JADE website and parses the results.
        """
        await self._ensure_client()
        assert self._client is not None

        # Build search query
        query = " ".join(keywords)

        # JADE search URL
        search_url = f"{self.base_url}/article/search"

        params = {
            "q": query,
            "type": "decision",  # Focus on court decisions
        }

        # Add date filters if specified
        if date_from:
            params["from"] = date_from
        if date_to:
            params["to"] = date_to

        try:
            await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
            headers = {"Referer": f"{self.base_url}/"}
            response = await self._client.get(search_url, params=params, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            results_found = 0

            # Parse search results - JADE uses various result containers
            for item in soup.select(".search-result, .result-item, article, .decision"):
                if results_found >= max_results:
                    break

                result = self._parse_search_result(item)
                if result:
                    # Apply court filter
                    if court and result.court != court:
                        continue

                    results_found += 1
                    yield result

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            raise

    def _parse_search_result(self, item: BeautifulSoup) -> SearchResult | None:
        """Parse a single search result item."""
        link = item.find("a", href=True)
        if not link:
            return None

        href = link.get("href", "")
        if not href:
            return None

        url = urljoin(self.base_url, href)
        title = link.get_text(strip=True)

        if not title:
            return None

        # Extract snippet
        snippet = None
        snippet_elem = item.find(class_=["snippet", "summary", "excerpt"])
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
        case_id = generate_case_id("jade", identifier)

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
        if result.court == "HCA":
            score += 0.2
        elif result.court in ["NSWCA", "VSCA", "QCA", "FCAFC"]:
            score += 0.15
        elif result.court in ["NSWSC", "VSC", "QSC", "FCA"]:
            score += 0.1

        # Boost for medical terms
        if result.snippet:
            medical_terms = [
                "negligence", "malpractice", "surgery", "patient", "doctor",
                "hospital", "diagnosis", "treatment", "consent", "breach",
                "medical", "clinical", "duty of care",
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
        """Convert citation to JADE API URL.

        JADE provides API endpoints for direct case access by citation.
        Returns the content URL that provides HTML without the full interface.
        """
        # Parse neutral citation: [2019] NSWSC 1234
        match = re.match(r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)", citation)
        if match:
            year, court, number = match.groups()
            court_lower = court.lower()

            # Medium Neutral Citation format URL
            return f"{self.api_base_url}/mnc/{year}/{court_lower}/{number}"

        return None

    def get_austlii_style_url(self, citation: str) -> str | None:
        """Get JADE URL in AustLII-compatible format.

        JADE also supports AustLII-style URLs for compatibility.
        """
        match = re.match(r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)", citation)
        if match:
            year, court, number = match.groups()
            state = self.STATE_MAP.get(court, "cth")
            court_lower = court.lower()

            return f"{self.api_base_url}/au/cases/{state}/{court_lower}/{year}/{number}.html"

        return None

    def get_content_url(self, citation: str) -> str | None:
        """Get direct content URL (HTML without interface).

        This endpoint returns the case text directly wrapped in HTML,
        useful for programmatic access.
        """
        match = re.match(r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)", citation)
        if match:
            year, court, number = match.groups()
            court_lower = court.lower()

            return f"{self.api_base_url}/content/ext/mnc/{year}/{court_lower}/{number}"

        return None
