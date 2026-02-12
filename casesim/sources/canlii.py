"""CanLII (Canadian Legal Information Institute) source connector."""

from __future__ import annotations

import re
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import quote_plus, urljoin, urlparse

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, DiscoveryStatus, Jurisdiction, Source
from ..utils import generate_case_id
from .base import BaseSource, RateLimiter, SearchResult


class CanLIISource(BaseSource):
    """Connector for CanLII legal database."""

    source = Source.CANLII
    jurisdiction = Jurisdiction.CA
    base_url = "https://www.canlii.org"

    # Canadian courts relevant for medical malpractice
    RELEVANT_COURTS = {
        # Ontario
        "ONSC": "Ontario Superior Court of Justice",
        "ONCA": "Ontario Court of Appeal",
        # British Columbia
        "BCSC": "British Columbia Supreme Court",
        "BCCA": "British Columbia Court of Appeal",
        # Alberta
        "ABQB": "Alberta Court of Queen's Bench",
        "ABCA": "Alberta Court of Appeal",
        # Federal
        "SCC": "Supreme Court of Canada",
        # Quebec
        "QCCS": "Quebec Superior Court",
        "QCCA": "Quebec Court of Appeal",
        # Saskatchewan
        "SKQB": "Saskatchewan Court of Queen's Bench",
        "SKCA": "Saskatchewan Court of Appeal",
        # Manitoba
        "MBQB": "Manitoba Court of Queen's Bench",
        "MBCA": "Manitoba Court of Appeal",
        # Nova Scotia
        "NSSC": "Nova Scotia Supreme Court",
        "NSCA": "Nova Scotia Court of Appeal",
        # New Brunswick
        "NBQB": "New Brunswick Court of Queen's Bench",
        "NBCA": "New Brunswick Court of Appeal",
        # Newfoundland
        "NLSC": "Newfoundland Supreme Court",
        "NLCA": "Newfoundland Court of Appeal",
        # Prince Edward Island
        "PESC": "Prince Edward Island Supreme Court",
    }

    def __init__(self):
        """Initialize with slower rate limit for CanLII."""
        super().__init__()
        # CanLII aggressively rate limits - use 0.2 req/sec (1 request every 5 seconds)
        self._rate_limiter = RateLimiter(requests_per_second=0.2)

    # Citation patterns for Canadian cases
    CITATION_PATTERNS = [
        # Neutral citations: 2019 ONSC 1234 (all provinces)
        r"(\d{4})\s+(ONSC|ONCA|BCSC|BCCA|ABQB|ABCA|SCC|QCCS|QCCA|SKQB|SKCA|MBQB|MBCA|NSSC|NSCA|NBQB|NBCA|NLSC|NLCA|PESC)\s+(\d+)",
        # CanLII citations: 2019 CanLII 12345 (ON SC)
        r"(\d{4})\s+CanLII\s+(\d+)\s*\(([A-Z]{2}\s*[A-Z]{2,4})\)",
        # SCR citations: [2019] 1 SCR 123
        r"\[(\d{4})\]\s+(\d+)\s+SCR\s+(\d+)",
    ]

    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search CanLII for cases matching keywords."""
        await self._ensure_client()
        assert self._client is not None

        # Build search query
        query = " AND ".join(f'"{kw}"' if " " in kw else kw for kw in keywords)

        # CanLII search URL
        search_url = f"{self.base_url}/en/search/search.do"

        params = {
            "searchUrlHash": "",
            "text": query,
            "sortBy": "date:desc",
            "page": "1",
        }

        # Add date filters
        if date_from:
            params["startDate"] = date_from
        if date_to:
            params["endDate"] = date_to

        results_found = 0
        page = 1

        while results_found < max_results:
            params["page"] = str(page)

            try:
                await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
                # Add Referer header to avoid 403 errors
                headers = {"Referer": f"{self.base_url}/en/"}
                response = await self._client.get(search_url, params=params, headers=headers)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")
                page_results = 0

                # Parse search results
                for item in soup.select(".result, .searchResult, li.case"):
                    if results_found >= max_results:
                        break

                    result = self._parse_search_result(item)
                    if result:
                        # Apply court filter if specified
                        if court and result.court and court not in result.court:
                            continue

                        results_found += 1
                        page_results += 1
                        yield result

                # Check if more pages available
                if page_results == 0:
                    break

                page += 1

            except Exception as e:
                self.logger.error(f"Search error on page {page}: {e}")
                break

    def _parse_search_result(self, item: BeautifulSoup) -> SearchResult | None:
        """Parse a single search result item."""
        # Try to find the case link
        link = item.find("a", href=True, class_="title") or item.find("a", href=True)
        if not link:
            return None

        href = link.get("href", "")
        if not href or not self._is_case_url(href):
            return None

        url = urljoin(self.base_url, href)
        title = link.get_text(strip=True)

        # Extract snippet
        snippet = None
        snippet_elem = item.find("div", class_="snippet") or item.find("p", class_="summary")
        if snippet_elem:
            snippet = snippet_elem.get_text(strip=True)[:500]

        # Extract date
        date = None
        date_elem = item.find("span", class_="date") or item.find("time")
        if date_elem:
            date = date_elem.get_text(strip=True)
        else:
            date = self._extract_date_from_url(url) or self._extract_date_from_title(title)

        # Extract court
        court = self._extract_court(url, title)

        # Extract citation
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

        # CanLII case URLs typically contain language code and database ID
        case_patterns = ["/en/", "/fr/", "/decisions/"]
        return any(p in href for p in case_patterns) and (
            ".html" in href or re.search(r"/\d{4}/\d+$", href)
        )

    def _extract_date_from_url(self, url: str) -> str | None:
        """Extract year from URL."""
        match = re.search(r"/(\d{4})/", url)
        if match:
            return match.group(1)
        return None

    def _extract_date_from_title(self, title: str) -> str | None:
        """Extract year from title."""
        match = re.search(r"(\d{4})\s+(ONSC|ONCA|BCSC|SCC|CanLII)", title)
        if match:
            return match.group(1)
        return None

    def _extract_court(self, url: str, title: str) -> str | None:
        """Extract court from URL or title."""
        # Check URL first
        for court_code in self.RELEVANT_COURTS:
            if court_code.lower() in url.lower():
                return court_code

        # Check title
        for court_code in self.RELEVANT_COURTS:
            if court_code in title:
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
        case_id = generate_case_id("canlii", identifier)

        # Parse year
        year = None
        if search_result.date:
            try:
                year = int(search_result.date[:4])
            except (ValueError, TypeError):
                pass

        # Calculate priority
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
        """Calculate priority score for a search result."""
        score = 0.5

        # Boost for appellate courts (more detailed analysis)
        appellate_courts = ["ONCA", "BCCA", "ABCA", "SCC", "QCCA", "SKCA", "MBCA", "NSCA", "NBCA", "NLCA"]
        trial_courts = ["ONSC", "BCSC", "ABQB", "QCCS", "SKQB", "MBQB", "NSSC", "NBQB", "NLSC", "PESC"]
        if result.court in appellate_courts:
            score += 0.15
        elif result.court in trial_courts:
            score += 0.1

        # Boost for medical terms in snippet
        if result.snippet:
            medical_terms = [
                "negligence", "malpractice", "surgery", "patient", "doctor",
                "hospital", "diagnosis", "treatment", "consent", "standard of care",
            ]
            snippet_lower = result.snippet.lower()
            matches = sum(1 for t in medical_terms if t in snippet_lower)
            score += min(0.2, matches * 0.04)

        # Boost for case vs company (v. indicates litigation)
        if " v. " in result.title or " v " in result.title:
            score += 0.05

        return min(1.0, score)

    def extract_citations(self, content: str) -> list[str]:
        """Extract citations from document content."""
        citations = set()

        for pattern in self.CITATION_PATTERNS:
            for match in re.finditer(pattern, content):
                citations.add(match.group(0))

        return list(citations)

    def parse_citation_url(self, citation: str) -> str | None:
        """Convert a citation to a CanLII URL."""
        # Parse neutral citation: 2019 ONSC 1234
        match = re.match(r"(\d{4})\s+([A-Z]{2,4})\s+(\d+)", citation)
        if match:
            year, court, number = match.groups()
            # CanLII URL structure
            url = f"{self.base_url}/en/{court.lower()}/doc/{year}/{year}{court.lower()}{number}/{year}{court.lower()}{number}.html"
            return url

        # Parse CanLII citation: 2019 CanLII 12345 (ON SC)
        match = re.match(r"(\d{4})\s+CanLII\s+(\d+)\s*\(([A-Z]{2})\s*([A-Z]{2,4})\)", citation)
        if match:
            year, number, province, court = match.groups()
            jurisdiction = province.lower()
            return f"{self.base_url}/en/{jurisdiction}/doc/{year}/{year}canlii{number}/{year}canlii{number}.html"

        return None
