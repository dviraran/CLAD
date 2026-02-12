"""Source connectors for legal databases."""

from .austlii import AustLIISource
from .bailii import BAILIISource
from .base import BaseSource, FetchResult, RateLimiter, SearchResult
from .canlii import CanLIISource
from .courtlistener import CourtListenerSource
from .hklii import HKLIISource
from .jade import JADESource
from .nzlii import NZLIISource
from .singapore import SingaporeSource

__all__ = [
    "AustLIISource",
    "BAILIISource",
    "BaseSource",
    "CanLIISource",
    "CourtListenerSource",
    "FetchResult",
    "HKLIISource",
    "JADESource",
    "NZLIISource",
    "RateLimiter",
    "SearchResult",
    "SingaporeSource",
]


def get_source(source_name: str) -> type[BaseSource]:
    """Get source class by name."""
    sources = {
        "bailii": BAILIISource,
        "canlii": CanLIISource,
        "austlii": AustLIISource,
        "courtlistener": CourtListenerSource,
        "nzlii": NZLIISource,
        "hklii": HKLIISource,
        "singapore": SingaporeSource,
        "jade": JADESource,
    }
    source_lower = source_name.lower()
    if source_lower not in sources:
        raise ValueError(f"Unknown source: {source_name}. Available: {list(sources.keys())}")
    return sources[source_lower]
