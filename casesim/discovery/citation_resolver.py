"""Citation resolver for converting case citations to URLs.

This module resolves case citations to URLs in primary legal databases,
supporting UK, US, Canadian, and Australian citation formats.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ..utils import get_logger

if TYPE_CHECKING:
    from ..sources import BaseSource


class CitationResolver:
    """Resolve case citations to URLs in primary legal databases."""

    def __init__(self):
        self.logger = get_logger("citation_resolver")
        self._sources: dict[str, BaseSource] = {}

    def _get_source(self, jurisdiction: str) -> BaseSource | None:
        """Get or create a source for the given jurisdiction."""
        if jurisdiction not in self._sources:
            from ..sources import (
                BAILIISource,
                CanLIISource,
                AustLIISource,
                CourtListenerSource,
            )

            source_map = {
                "UK": BAILIISource,
                "IE": BAILIISource,  # Ireland via BAILII
                "CA": CanLIISource,
                "AU": AustLIISource,
                "US": CourtListenerSource,
            }

            source_class = source_map.get(jurisdiction)
            if source_class:
                self._sources[jurisdiction] = source_class()

        return self._sources.get(jurisdiction)

    def detect_jurisdiction(self, citation: str) -> str | None:
        """Detect jurisdiction from citation format.

        Args:
            citation: A case citation string

        Returns:
            Jurisdiction code (UK, CA, AU, US, IE) or None if unknown
        """
        # UK patterns
        uk_courts = r"(EWHC|EWCA|UKSC|UKHL|EWCOP|CSOH|CSIH|ScotCS|NIQB|NICA)"
        if re.search(rf"\[\d{{4}}\]\s+{uk_courts}", citation):
            return "UK"

        # Irish patterns (via BAILII)
        ie_courts = r"(IEHC|IESC|IECA)"
        if re.search(rf"\[\d{{4}}\]\s+{ie_courts}", citation):
            return "IE"

        # Canadian patterns
        ca_courts = r"(ONSC|ONCA|BCSC|BCCA|ABQB|ABCA|SCC|QCCS|QCCA|SKQB|SKCA|MBQB|MBCA|NSSC|NSCA)"
        if re.search(rf"\d{{4}}\s+{ca_courts}", citation):
            return "CA"
        if re.search(r"\d{4}\s+CanLII\s+\d+", citation):
            return "CA"

        # Australian patterns
        au_courts = r"(NSWSC|NSWCA|VSC|VCA|HCA|QSC|QCA|WASC|WASCA|SASC|SASCA)"
        if re.search(rf"\[\d{{4}}\]\s+{au_courts}", citation):
            return "AU"

        # US patterns
        # US Reports
        if re.search(r"\d+\s+U\.S\.\s+\d+", citation):
            return "US"
        # Federal Reporter
        if re.search(r"\d+\s+F\.\s*(2d|3d|4th)?\s+\d+", citation):
            return "US"
        # Federal Supplement
        if re.search(r"\d+\s+F\.\s*Supp\.", citation):
            return "US"

        # UK report series (fall back to UK)
        if re.search(r"\[\d{4}\]\s+\d+\s+(WLR|All\s*ER|Med\s*LR|BMLR)", citation):
            return "UK"

        # Canadian SCR
        if re.search(r"\[\d{4}\]\s+\d+\s+SCR", citation):
            return "CA"

        return None

    def resolve(self, citation: str, jurisdiction: str | None = None) -> str | None:
        """Convert a citation to a URL.

        Args:
            citation: A case citation string
            jurisdiction: Optional jurisdiction override. If not provided,
                         will attempt to detect from citation format.

        Returns:
            URL string or None if cannot be resolved
        """
        # Detect jurisdiction if not provided
        if jurisdiction is None:
            jurisdiction = self.detect_jurisdiction(citation)

        if not jurisdiction:
            self.logger.debug(f"Could not detect jurisdiction for: {citation}")
            return None

        # Get source for jurisdiction
        source = self._get_source(jurisdiction)
        if not source:
            self.logger.debug(f"No source available for jurisdiction: {jurisdiction}")
            return None

        # Use source's citation parser
        url = source.parse_citation_url(citation)
        if url:
            self.logger.debug(f"Resolved {citation} -> {url}")
        else:
            self.logger.debug(f"Could not resolve citation: {citation}")

        return url

    def resolve_batch(
        self,
        citations: list[str],
        jurisdiction: str | None = None,
    ) -> dict[str, str | None]:
        """Resolve multiple citations.

        Args:
            citations: List of citation strings
            jurisdiction: Optional jurisdiction override for all

        Returns:
            Dict mapping citation -> URL (or None if unresolved)
        """
        results = {}
        for citation in citations:
            results[citation] = self.resolve(citation, jurisdiction)
        return results

    def resolve_with_metadata(
        self,
        citation: str,
        jurisdiction: str | None = None,
    ) -> dict:
        """Resolve a citation and return metadata.

        Args:
            citation: A case citation string
            jurisdiction: Optional jurisdiction override

        Returns:
            Dict with keys: citation, url, jurisdiction, source, resolved
        """
        detected_jurisdiction = jurisdiction or self.detect_jurisdiction(citation)
        url = self.resolve(citation, detected_jurisdiction)

        source = None
        if detected_jurisdiction:
            source_obj = self._get_source(detected_jurisdiction)
            if source_obj:
                source = source_obj.source.value

        return {
            "citation": citation,
            "url": url,
            "jurisdiction": detected_jurisdiction,
            "source": source,
            "resolved": url is not None,
        }
