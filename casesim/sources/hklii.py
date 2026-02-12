"""HKLII (Hong Kong Legal Information Institute) source connector."""

from __future__ import annotations

import re
from typing import AsyncIterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, DiscoveryStatus, Jurisdiction, Source
from ..utils import generate_case_id
from .base import BaseSource, SearchResult


class HKLIISource(BaseSource):
    """Connector for HKLII legal database.

    HKLII is part of the LII network (powered by AustLII infrastructure)
    and uses the SINO search engine. Hong Kong follows UK-style common law
    with English judgments and strong medical negligence analysis.
    """

    source = Source.HKLII
    jurisdiction = Jurisdiction.HK
    base_url = "https://www.hklii.hk"

    # Hong Kong courts relevant for medical negligence
    RELEVANT_COURTS = {
        "HKCFA": "Court of Final Appeal",
        "HKCA": "Court of Appeal",
        "HKCFI": "Court of First Instance",
        "HKDC": "District Court",
        "HKFC": "Family Court",
        "HKLT": "Lands Tribunal",
        "HKCT": "Competition Tribunal",
    }

    # Citation patterns for Hong Kong cases
    CITATION_PATTERNS = [
        # Neutral citations: [2019] HKCFA 1, [2019] HKCFI 1234
        r"\[(\d{4})\]\s+(HKCFA|HKCA|HKCFI|HKDC|HKFC|HKLT|HKCT)\s+(\d+)",
        # HKLRD citations: [2019] 1 HKLRD 123
        r"\[(\d{4})\]\s+(\d+)\s+HKLRD\s+(\d+)",
        # HKC citations: [2019] 1 HKC 123
        r"\[(\d{4})\]\s+(\d+)\s+HKC\s+(\d+)",
        # Older format: CACV 123/2019
        r"(CACV|HCPI|HCAL|DCPI|HCMP)\s+(\d+)/(\d{4})",
    ]

    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search HKLII for cases matching keywords."""
        await self._ensure_client()
        assert self._client is not None

        # Build search query
        query = " AND ".join(f'"{kw}"' if " " in kw else kw for kw in keywords)

        # HKLII search URL (SINO search engine)
        search_url = f"{self.base_url}/cgi-bin/sinosrch.cgi"

        params = {
            "query": query,
            "method": "auto",
            "meta": "/hk",  # Hong Kong databases
            "rank": "on",
            "maxhits": str(min(max_results, 500)),
        }

        # Filter by court if specified
        if court:
            params["meta"] = f"/hk/cases/{court.upper()}"

        try:
            await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
            # HKLII requires Referer header
            headers = {"Referer": f"{self.base_url}/"}
            response = await self._client.get(search_url, params=params, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            results_found = 0

            # Parse search results
            for item in soup.select("ol li, .result, table tr"):
                if results_found >= max_results:
                    break

                result = self._parse_search_result(item)
                if result:
                    # Apply date filters
                    if date_from and result.date and result.date < date_from:
                        continue
                    if date_to and result.date and result.date > date_to:
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
        if not href or not self._is_case_url(href):
            return None

        url = urljoin(self.base_url, href)
        # Convert cgi-bin/disp.pl URLs to direct URLs
        url = self._normalize_case_url(url)
        title = link.get_text(strip=True)

        # Extract snippet
        snippet = None
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

        if "/cases/" not in href:
            return False

        exclude_patterns = ["index.htm", "search", "about", "help"]
        return not any(p in href.lower() for p in exclude_patterns)

    def _normalize_case_url(self, url: str) -> str:
        """Convert cgi-bin/disp.pl URLs to direct case URLs.

        HKLII search results may return URLs like:
        http://www.hklii.hk/cgi-bin/disp.pl/hk/cases/HKCFI/2016/123.html?...

        We convert these to direct URLs:
        https://www.hklii.hk/hk/cases/HKCFI/2016/123.html
        """
        # Extract the path after /cgi-bin/disp.pl/
        match = re.search(r"/cgi-bin/disp\.pl(/hk/cases/[^?]+)", url)
        if match:
            case_path = match.group(1)
            return f"{self.base_url}{case_path}"
        return url

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

        # Case number pattern: 123/2019
        match = re.search(r"/(\d{4})$", title)
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
        case_id = generate_case_id("hklii", identifier)

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
        if result.court == "HKCFA":
            score += 0.2
        elif result.court == "HKCA":
            score += 0.15
        elif result.court == "HKCFI":
            score += 0.1

        # Boost for medical terms
        if result.snippet:
            medical_terms = [
                "negligence", "malpractice", "surgery", "patient", "doctor",
                "hospital", "diagnosis", "treatment", "consent", "breach",
                "medical", "clinical", "duty of care", "standard of care",
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
        """Convert citation to HKLII URL."""
        # Parse neutral citation: [2019] HKCFI 1234
        match = re.match(r"\[(\d{4})\]\s+([A-Z]+)\s+(\d+)", citation)
        if match:
            year, court, number = match.groups()
            court_upper = court.upper()
            return f"{self.base_url}/hk/cases/{court_upper}/{year}/{number}.html"

        # Parse case number format: HCPI 123/2019
        match = re.match(r"(CACV|HCPI|HCAL|DCPI|HCMP)\s+(\d+)/(\d{4})", citation)
        if match:
            case_type, number, year = match.groups()
            # Map case types to courts
            court_map = {
                "CACV": "HKCA",
                "HCPI": "HKCFI",
                "HCAL": "HKCFI",
                "DCPI": "HKDC",
                "HCMP": "HKCFI",
            }
            court = court_map.get(case_type, "HKCFI")
            return f"{self.base_url}/hk/cases/{court}/{year}/{number}.html"

        return None
