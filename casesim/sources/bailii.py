"""BAILII (British and Irish Legal Information Institute) source connector."""

from __future__ import annotations

import re
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import quote_plus, urljoin, urlparse

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, DiscoveryStatus, Jurisdiction, Source
from ..utils import generate_case_id
from .base import BaseSource, SearchResult


class BAILIISource(BaseSource):
    """Connector for BAILII legal database."""

    source = Source.BAILII
    jurisdiction = Jurisdiction.UK
    base_url = "https://www.bailii.org"

    # Court mappings for medical negligence cases
    RELEVANT_COURTS = {
        # England and Wales
        "EWHC": "England and Wales High Court",
        "EWCA": "England and Wales Court of Appeal",
        "UKSC": "UK Supreme Court",
        "EWCOP": "Court of Protection",
        "UKHL": "House of Lords",
        # Scotland
        "CSOH": "Court of Session Outer House",
        "CSIH": "Court of Session Inner House",
        "ScotCS": "Scottish Court of Session",
        # Northern Ireland
        "NIQB": "Northern Ireland Queen's Bench",
        "NICA": "Northern Ireland Court of Appeal",
        # Ireland (via BAILII)
        "IEHC": "Irish High Court",
        "IESC": "Irish Supreme Court",
        "IECA": "Irish Court of Appeal",
    }

    # Common citation patterns
    CITATION_PATTERNS = [
        # Neutral citations: [2019] EWHC 936 (QB) - England, Wales, UK, Scotland, NI, Ireland
        r"\[(\d{4})\]\s+(EWHC|EWCA|UKSC|UKHL|EWCOP|CSOH|CSIH|ScotCS|NIQB|NICA|IEHC|IESC|IECA)\s+(\d+)\s*(?:\(([A-Za-z]+)\))?",
        # WLR citations: [2019] 1 WLR 123
        r"\[(\d{4})\]\s+(\d+)\s+(WLR|All\s*ER|Med\s*LR|BMLR)\s+(\d+)",
        # Traditional citations: Smith v Jones [2019] EWHC 936
        r"([A-Z][a-z]+)\s+v\.?\s+([A-Z][a-z]+).*?\[(\d{4})\]\s+(EWHC|EWCA|CSOH|CSIH|NIQB|IEHC)",
        # Scottish Session Cases: 2019 SC 123
        r"(\d{4})\s+(SC|SLT|SCLR)\s+(\d+)",
    ]

    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search BAILII for cases matching keywords."""
        await self._ensure_client()
        assert self._client is not None

        # Build search query - wrap multi-word terms in parens
        query_parts = []
        for kw in keywords:
            if " " in kw:
                query_parts.append(f'("{kw}")')
            else:
                query_parts.append(f"({kw})")
        query = " AND ".join(query_parts)

        # BAILII search URL - use lucy_search_1.cgi (new endpoint as of 2024)
        search_url = f"{self.base_url}/cgi-bin/lucy_search_1.cgi"

        params = {
            "query": query,
            "method": "boolean",
            "sort": "date",
        }

        # Filter by court if specified
        if court:
            params["mask_path"] = f"ew/cases/{court}"
        else:
            # Default to English courts for medical cases
            params["mask_path"] = "ew/cases/EWHC"

        try:
            await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
            # BAILII requires proper headers to avoid 403
            headers = {
                "Referer": f"{self.base_url}/form/search_cases.html",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }
            response = await self._client.get(search_url, params=params, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            results_found = 0

            # Parse search results - BAILII uses <li> elements with <a> links
            for item in soup.select("li"):
                if results_found >= max_results:
                    break

                result = self._parse_search_result(item)
                if result:
                    # Apply date filters if specified
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
        # Try to find link
        link = item.find("a", href=True)
        if not link:
            return None

        href = link.get("href", "")
        if not href or not self._is_case_url(href):
            return None

        url = urljoin(self.base_url, href)
        title = link.get_text(strip=True)

        # Try to extract snippet
        snippet = None
        snippet_elem = item.find("span", class_="snippet") or item.find("td", class_="snippet")
        if snippet_elem:
            snippet = snippet_elem.get_text(strip=True)
        else:
            # Get text after the link
            text = item.get_text(strip=True)
            if text and len(text) > len(title):
                snippet = text[len(title):].strip()[:500]

        # Try to extract date from URL or title
        date = self._extract_date(url, title)

        # Extract court from URL
        court = self._extract_court(url)

        # Try to extract citation
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
        """Check if URL is a case (not search/index page)."""
        if not href:
            return False

        # Must be a .html file or have case-like pattern
        if "/cases/" not in href:
            return False

        # Exclude search and index pages
        exclude_patterns = [
            "index.htm",
            "search",
            "find_by",
            "databases",
            "about",
        ]
        return not any(p in href.lower() for p in exclude_patterns)

    def _extract_date(self, url: str, title: str) -> str | None:
        """Extract date from URL or title."""
        # Try URL pattern: /2019/EWHC/...
        url_match = re.search(r"/(\d{4})/", url)
        if url_match:
            return url_match.group(1)

        # Try title pattern: [2019] EWHC
        title_match = re.search(r"\[(\d{4})\]", title)
        if title_match:
            return title_match.group(1)

        return None

    def _extract_court(self, url: str) -> str | None:
        """Extract court from URL."""
        for court_code in self.RELEVANT_COURTS:
            if court_code.lower() in url.lower():
                return court_code
        return None

    def _extract_citation(self, text: str) -> str | None:
        """Extract neutral citation from text."""
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
        # Generate case ID from URL
        url_parts = urlparse(search_result.url).path.split("/")
        id_parts = [p for p in url_parts if p and p not in ["ew", "cases", "html"]]
        identifier = "-".join(id_parts[-3:]) if len(id_parts) >= 3 else "-".join(id_parts)
        case_id = generate_case_id("bailii", identifier)

        # Parse year
        year = None
        if search_result.date:
            try:
                year = int(search_result.date[:4])
            except (ValueError, TypeError):
                pass

        # Calculate priority score based on signals
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
        score = 0.5  # Base score

        # Boost for specific courts
        if result.court in ["EWHC", "EWCA"]:
            score += 0.1

        # Boost for having snippet with medical terms
        if result.snippet:
            medical_terms = [
                "negligence", "clinical", "surgery", "patient", "doctor",
                "hospital", "diagnosis", "treatment", "consent", "duty",
            ]
            snippet_lower = result.snippet.lower()
            matches = sum(1 for t in medical_terms if t in snippet_lower)
            score += min(0.2, matches * 0.05)

        # Boost for title containing v (versus) - indicates a case
        if " v " in result.title.lower():
            score += 0.1

        # Boost for recent cases
        if result.date:
            try:
                year = int(result.date[:4])
                if year >= 2015:
                    score += 0.1
            except (ValueError, TypeError):
                pass

        return min(1.0, score)

    def extract_citations(self, content: str) -> list[str]:
        """Extract citations to other cases from document content."""
        citations = set()

        for pattern in self.CITATION_PATTERNS:
            for match in re.finditer(pattern, content):
                citations.add(match.group(0))

        return list(citations)

    def parse_citation_url(self, citation: str) -> str | None:
        """Convert a citation to a BAILII URL, if possible."""
        # Parse neutral citation: [2019] EWHC 936 (QB)
        match = re.match(
            r"\[(\d{4})\]\s+(EWHC|EWCA|UKSC|UKHL|EWCOP|CSOH|CSIH|ScotCS|NIQB|NICA|IEHC|IESC|IECA)\s+(\d+)\s*(?:\(([A-Za-z]+)\))?",
            citation,
        )
        if match:
            year, court, number, division = match.groups()

            # Build BAILII URL based on jurisdiction
            # England/Wales/UK courts
            ew_courts = {"EWHC", "EWCA", "UKSC", "UKHL", "EWCOP"}
            # Scottish courts
            scot_courts = {"CSOH", "CSIH", "ScotCS"}
            # Northern Ireland courts
            ni_courts = {"NIQB", "NICA"}
            # Irish courts
            ie_courts = {"IEHC", "IESC", "IECA"}

            if court in ew_courts:
                base_path = "/ew/cases"
            elif court in scot_courts:
                base_path = "/scot/cases"
            elif court in ni_courts:
                base_path = "/nie/cases"
            elif court in ie_courts:
                base_path = "/ie/cases"
            else:
                base_path = "/ew/cases"

            if division:
                url = f"{self.base_url}{base_path}/{court}/{division}/{year}/{number}.html"
            else:
                url = f"{self.base_url}{base_path}/{court}/{year}/{number}.html"

            return url

        return None

    async def browse_court_index(
        self,
        court: str = "EWHC",
        division: str = "QB",
        year: int | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Browse court index pages for cases."""
        await self._ensure_client()
        assert self._client is not None

        # Build index URL
        if year:
            index_url = f"{self.base_url}/ew/cases/{court}/{division}/{year}/"
        else:
            index_url = f"{self.base_url}/ew/cases/{court}/{division}/"

        try:
            await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
            response = await self._client.get(index_url)

            if response.status_code != 200:
                self.logger.warning(f"Index not found: {index_url}")
                return

            soup = BeautifulSoup(response.text, "lxml")
            results_found = 0

            for link in soup.find_all("a", href=True):
                if results_found >= max_results:
                    break

                href = link.get("href", "")
                if not self._is_case_url(href):
                    continue

                url = urljoin(index_url, href)
                title = link.get_text(strip=True)

                if not title or len(title) < 5:
                    continue

                results_found += 1
                yield SearchResult(
                    url=url,
                    title=title,
                    date=str(year) if year else None,
                    court=court,
                )

        except Exception as e:
            self.logger.error(f"Browse error for {index_url}: {e}")
