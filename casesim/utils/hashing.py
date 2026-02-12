"""Hashing utilities for deduplication and caching."""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

import xxhash


def normalize_url(url: str) -> str:
    """Normalize a URL for comparison and deduplication."""
    parsed = urlparse(url.lower().strip())

    # Remove trailing slashes from path
    path = parsed.path.rstrip("/")

    # Remove common tracking parameters
    query = parsed.query
    if query:
        params = query.split("&")
        filtered = [
            p
            for p in params
            if not any(
                p.startswith(prefix)
                for prefix in ["utm_", "ref=", "source=", "click="]
            )
        ]
        query = "&".join(sorted(filtered))

    # Reconstruct URL
    normalized = urlunparse(
        (
            parsed.scheme or "https",
            parsed.netloc,
            path,
            "",  # params
            query,
            "",  # fragment
        )
    )
    return normalized


def normalize_title(title: str) -> str:
    """Normalize a case title for comparison."""
    # Remove common prefixes/suffixes
    title = title.lower().strip()

    # Remove punctuation except essential characters
    title = re.sub(r"[^\w\s&v]", "", title)

    # Normalize whitespace
    title = " ".join(title.split())

    # Remove common words
    remove_words = ["the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for"]
    words = title.split()
    words = [w for w in words if w not in remove_words]

    return " ".join(words)


def hash_content(content: str) -> str:
    """Create a hash of content for deduplication."""
    return xxhash.xxh64(content.encode()).hexdigest()


def hash_case_identity(title: str, parties: str | None = None) -> str:
    """Create a hash for case identity based on title and parties."""
    normalized = normalize_title(title)
    if parties:
        normalized += " " + normalize_title(parties)
    return xxhash.xxh64(normalized.encode()).hexdigest()


def generate_case_id(source: str, identifier: str) -> str:
    """Generate a standardized case ID."""
    # Clean the identifier
    clean_id = re.sub(r"[^\w-]", "-", identifier.lower())
    clean_id = re.sub(r"-+", "-", clean_id).strip("-")

    # Combine with source
    source_prefix = source.lower()[:10]
    return f"{source_prefix}-{clean_id}"[:64]
