"""CourtListener source connector for US court opinions."""

from __future__ import annotations

import re
from datetime import datetime
from typing import AsyncIterator
from urllib.parse import quote_plus, urljoin, urlparse

from bs4 import BeautifulSoup

from ..schemas import DiscoveryRecord, DiscoveryStatus, Jurisdiction, Source
from ..utils import generate_case_id
from .base import BaseSource, SearchResult


class CourtListenerSource(BaseSource):
    """Connector for CourtListener (US court opinions)."""

    source = Source.COURTLISTENER
    jurisdiction = Jurisdiction.US
    base_url = "https://www.courtlistener.com"

    # US courts relevant for medical malpractice (state and federal)
    RELEVANT_COURTS = {
        # Federal - Supreme Court
        "scotus": "Supreme Court of the United States",
        # Federal - Circuit Courts of Appeals
        "ca1": "First Circuit Court of Appeals",
        "ca2": "Second Circuit Court of Appeals",
        "ca3": "Third Circuit Court of Appeals",
        "ca4": "Fourth Circuit Court of Appeals",
        "ca5": "Fifth Circuit Court of Appeals",
        "ca6": "Sixth Circuit Court of Appeals",
        "ca7": "Seventh Circuit Court of Appeals",
        "ca8": "Eighth Circuit Court of Appeals",
        "ca9": "Ninth Circuit Court of Appeals",
        "ca10": "Tenth Circuit Court of Appeals",
        "ca11": "Eleventh Circuit Court of Appeals",
        "cadc": "DC Circuit Court of Appeals",
        # Federal - District Courts (high volume)
        "nysd": "Southern District of New York",
        "nyed": "Eastern District of New York",
        "cand": "Northern District of California",
        "casd": "Southern District of California",
        "txsd": "Southern District of Texas",
        "flsd": "Southern District of Florida",
        "ilnd": "Northern District of Illinois",
        "paed": "Eastern District of Pennsylvania",
        # State Supreme Courts (high malpractice volume states)
        "ny": "New York Court of Appeals",
        "nyappdiv": "New York Appellate Division",
        "cal": "California Supreme Court",
        "calctapp": "California Court of Appeal",
        "tex": "Texas Supreme Court",
        "texapp": "Texas Court of Appeals",
        "fla": "Florida Supreme Court",
        "fladistctapp": "Florida District Court of Appeal",
        "pa": "Pennsylvania Supreme Court",
        "pasuperct": "Pennsylvania Superior Court",
        "ill": "Illinois Supreme Court",
        "illappct": "Illinois Appellate Court",
        "ohio": "Ohio Supreme Court",
        "ohioctapp": "Ohio Court of Appeals",
        "nj": "New Jersey Supreme Court",
        "njsuperctappdiv": "New Jersey Superior Court Appellate Division",
        "mass": "Massachusetts Supreme Judicial Court",
        "massappct": "Massachusetts Appeals Court",
    }

    # Citation patterns
    CITATION_PATTERNS = [
        # US Reports: 123 U.S. 456 (2019)
        r"(\d+)\s+U\.S\.\s+(\d+)\s*\((\d{4})\)",
        # Federal Reporter: 123 F.3d 456 (9th Cir. 2019)
        r"(\d+)\s+F\.\s*(2d|3d)?\s+(\d+)\s*\([^)]+(\d{4})\)",
        # Federal Supplement: 123 F. Supp. 2d 456
        r"(\d+)\s+F\.\s*Supp\.\s*(2d|3d)?\s+(\d+)",
        # State reporters with year
        r"(\d+)\s+([A-Z][a-z]+\.?\s*(?:2d|3d)?)\s+(\d+)\s*\((\d{4})\)",
    ]

    async def search(
        self,
        keywords: list[str],
        date_from: str | None = None,
        date_to: str | None = None,
        court: str | None = None,
        max_results: int = 100,
    ) -> AsyncIterator[SearchResult]:
        """Search CourtListener for cases matching keywords."""
        await self._ensure_client()
        assert self._client is not None

        # Build search query
        query = " AND ".join(f'"{kw}"' if " " in kw else kw for kw in keywords)

        # CourtListener search API (V4)
        search_url = f"{self.base_url}/api/rest/v4/search/"

        params = {
            "q": query,
            "type": "o",  # opinions
            "order_by": "dateFiled desc",
            "page_size": str(min(max_results, 100)),
        }

        # Add date filters
        if date_from:
            params["filed_after"] = date_from
        if date_to:
            params["filed_before"] = date_to

        # Add court filter
        if court:
            params["court"] = court

        results_found = 0
        page = 1

        while results_found < max_results:
            params["page"] = str(page)

            try:
                await self._rate_limiter.acquire(urlparse(self.base_url).netloc)

                headers = {"Accept": "application/json"}
                api_token = self.settings.sources.courtlistener_api_token
                if api_token:
                    headers["Authorization"] = f"Token {api_token}"

                response = await self._client.get(search_url, params=params, headers=headers)

                # Handle API response
                if response.status_code == 200:
                    data = response.json()
                    results = data.get("results", [])

                    if not results:
                        break

                    for item in results:
                        if results_found >= max_results:
                            break

                        result = self._parse_api_result(item)
                        if result:
                            results_found += 1
                            yield result

                    # Check for next page
                    if not data.get("next"):
                        break

                    page += 1
                else:
                    # Fall back to web search
                    async for result in self._web_search(keywords, max_results - results_found):
                        if results_found >= max_results:
                            break
                        results_found += 1
                        yield result
                    break

            except Exception as e:
                self.logger.error(f"Search error on page {page}: {e}")
                break

    def _parse_api_result(self, item: dict) -> SearchResult | None:
        """Parse an API search result (V4 format)."""
        try:
            # Get URL - V4 uses cluster_id in the absolute_url
            absolute_url = item.get("absolute_url", "")
            if not absolute_url:
                return None

            url = urljoin(self.base_url, absolute_url)

            # Get title (case name) - V4 uses caseName
            title = item.get("caseName", "") or item.get("case_name", "")
            if not title:
                return None

            # Get snippet from opinions array (V4 format)
            snippet = ""
            opinions = item.get("opinions", [])
            if opinions and isinstance(opinions, list):
                for opinion in opinions:
                    if opinion.get("snippet"):
                        snippet = opinion.get("snippet", "")
                        break

            # Get date - V4 uses dateFiled
            date = item.get("dateFiled") or item.get("date_filed")

            # Get court - V4 returns full court name
            court = item.get("court", "")

            # Get citation - V4 returns list of citations
            citation = item.get("citation", [])
            if isinstance(citation, list) and citation:
                citation = citation[0]
            else:
                citation = None

            return SearchResult(
                url=url,
                title=title,
                snippet=snippet[:500] if snippet else None,
                date=date[:10] if date else None,  # YYYY-MM-DD
                court=court,
                citation=citation,
            )

        except Exception as e:
            self.logger.debug(f"Error parsing result: {e}")
            return None

    async def _web_search(
        self, keywords: list[str], max_results: int
    ) -> AsyncIterator[SearchResult]:
        """Fallback web search if API unavailable."""
        await self._ensure_client()
        assert self._client is not None

        query = "+".join(quote_plus(kw) for kw in keywords)
        search_url = f"{self.base_url}/?q={query}&type=o"

        try:
            await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
            response = await self._client.get(search_url)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "lxml")
            results_found = 0

            for item in soup.select(".search-result, .result-item, article"):
                if results_found >= max_results:
                    break

                result = self._parse_web_result(item)
                if result:
                    results_found += 1
                    yield result

        except Exception as e:
            self.logger.error(f"Web search error: {e}")

    def _parse_web_result(self, item: BeautifulSoup) -> SearchResult | None:
        """Parse a web search result."""
        link = item.find("a", href=True)
        if not link:
            return None

        href = link.get("href", "")
        if not href or "/opinion/" not in href:
            return None

        url = urljoin(self.base_url, href)
        title = link.get_text(strip=True)

        snippet = None
        snippet_elem = item.find("p", class_="description") or item.find("div", class_="snippet")
        if snippet_elem:
            snippet = snippet_elem.get_text(strip=True)[:500]

        date = None
        date_elem = item.find("time") or item.find("span", class_="date")
        if date_elem:
            date = date_elem.get_text(strip=True)

        return SearchResult(
            url=url,
            title=title,
            snippet=snippet,
            date=date,
        )

    async def fetch(self, url: str, use_cache: bool = True) -> "FetchResult":
        """Fetch opinion content via API instead of web scraping.

        CourtListener blocks /opinion/ pages via robots.txt, so we need to
        use their API to fetch opinion text.
        """
        from .base import FetchResult
        from ..utils import hash_content

        await self._ensure_client()
        assert self._client is not None

        # Check cache first
        normalized_url = url.split("?")[0]  # Remove query params
        if use_cache and self._cache is not None:
            cached = self._cache.get(normalized_url)
            if cached is not None:
                self.logger.debug(f"Cache hit: {url}")
                from datetime import datetime
                return FetchResult(
                    url=url,
                    content=cached["content"],
                    content_type=cached["content_type"],
                    status_code=200,
                    fetched_at=datetime.fromisoformat(cached["fetched_at"]),
                    content_hash=cached["content_hash"],
                    cached=True,
                )

        # Extract cluster ID from URL
        # URL format: /opinion/{cluster_id}/slug/ or https://www.courtlistener.com/opinion/{cluster_id}/slug/
        cluster_id = None
        path = urlparse(url).path
        parts = path.strip("/").split("/")
        for i, part in enumerate(parts):
            if part == "opinion" and i + 1 < len(parts):
                cluster_id = parts[i + 1]
                break

        if not cluster_id:
            raise ValueError(f"Could not extract cluster ID from URL: {url}")

        headers = {"Accept": "application/json"}
        api_token = self.settings.sources.courtlistener_api_token
        if api_token:
            headers["Authorization"] = f"Token {api_token}"
        else:
            raise PermissionError("CourtListener API token required for fetching opinions")

        # Fetch cluster info to get case name and opinion IDs
        cluster_url = f"{self.base_url}/api/rest/v4/clusters/{cluster_id}/"
        await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
        self.logger.debug(f"Fetching cluster via API: {cluster_url}")
        cluster_response = await self._client.get(cluster_url, headers=headers)
        cluster_response.raise_for_status()
        cluster_data = cluster_response.json()

        case_name = cluster_data.get("case_name", "Unknown Case")
        opinion_ids = cluster_data.get("sub_opinions", [])

        if not opinion_ids:
            raise ValueError(f"No opinions found in cluster {cluster_id}")

        # Fetch all opinions in the cluster and combine them
        all_content = []
        for opinion_url in opinion_ids:
            # opinion_url is a full API URL like https://www.courtlistener.com/api/rest/v4/opinions/123/
            await self._rate_limiter.acquire(urlparse(self.base_url).netloc)
            self.logger.debug(f"Fetching opinion via API: {opinion_url}")
            opinion_response = await self._client.get(opinion_url, headers=headers)
            opinion_response.raise_for_status()
            opinion_data = opinion_response.json()

            # Get opinion text - prefer plain_text, fall back to various HTML formats
            opinion_text = (
                opinion_data.get("plain_text") or
                opinion_data.get("html_with_citations") or
                opinion_data.get("html") or
                opinion_data.get("html_lawbox") or
                opinion_data.get("html_columbia") or
                opinion_data.get("xml_harvard") or
                ""
            )
            if opinion_text:
                all_content.append(opinion_text)

        if not all_content:
            raise ValueError(f"No opinion text found for cluster {cluster_id}")

        combined_text = "\n\n---\n\n".join(all_content)

        # Wrap in HTML structure for consistency with other sources
        content = f"""<!DOCTYPE html>
<html>
<head><title>{case_name}</title></head>
<body>
<h1>{case_name}</h1>
<pre>{combined_text}</pre>
</body>
</html>"""

        from datetime import datetime
        content_hash = hash_content(content)
        fetched_at = datetime.utcnow()

        result = FetchResult(
            url=url,
            content=content,
            content_type="text/html",
            status_code=200,
            fetched_at=fetched_at,
            content_hash=content_hash,
            cached=False,
            headers={},
        )

        # Cache the result
        if self._cache is not None:
            self._cache.set(
                normalized_url,
                {
                    "content": content,
                    "content_type": "text/html",
                    "fetched_at": fetched_at.isoformat(),
                    "content_hash": content_hash,
                },
                expire=self.settings.cache.ttl_days * 86400,
            )

        return result

    def build_discovery_record(
        self,
        search_result: SearchResult,
        discovery_methods: list[str],
        query_terms: list[str] | None = None,
    ) -> DiscoveryRecord:
        """Build a DiscoveryRecord from a search result."""
        # Generate case ID
        url_parts = urlparse(search_result.url).path.strip("/").split("/")
        identifier = "-".join(url_parts[-2:]) if len(url_parts) >= 2 else url_parts[-1]
        case_id = generate_case_id("courtlistener", identifier)

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

        # Supreme courts (federal and state)
        supreme_courts = {"scotus", "ny", "cal", "tex", "fla", "pa", "ill", "ohio", "nj", "mass"}
        # Appellate courts (federal circuits and state appeals)
        appellate_courts = {
            "ca1", "ca2", "ca3", "ca4", "ca5", "ca6", "ca7", "ca8", "ca9", "ca10", "ca11", "cadc",
            "nyappdiv", "calctapp", "texapp", "fladistctapp", "pasuperct", "illappct", "ohioctapp",
            "njsuperctappdiv", "massappct"
        }

        # Boost for appellate courts
        if result.court:
            court_lower = result.court.lower()
            if court_lower in supreme_courts or "supreme" in court_lower:
                score += 0.2
            elif court_lower in appellate_courts or "circuit" in court_lower or "appeal" in court_lower:
                score += 0.15
            elif "district" in court_lower:
                score += 0.05

        # Boost for medical terms
        if result.snippet:
            medical_terms = [
                "malpractice", "negligence", "surgery", "patient", "physician",
                "hospital", "diagnosis", "treatment", "informed consent",
                "standard of care", "medical", "doctor",
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
        """Convert citation to CourtListener URL (limited capability)."""
        # CourtListener requires case lookup, can't directly convert citations
        # Return None as this would require an API call
        return None
