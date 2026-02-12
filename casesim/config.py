"""Configuration management for casesim."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OpenAISettings(BaseSettings):
    """OpenAI API configuration."""

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="Model to use for extraction")
    base_url: str | None = Field(default=None, description="Custom API base URL")
    max_tokens: int = Field(default=4096, description="Max tokens per completion")
    temperature: float = Field(default=0.0, description="Temperature for generation (0.0 for deterministic)")
    max_retries: int = Field(default=3, description="Max retries on failure")
    retry_delay: float = Field(default=1.0, description="Initial retry delay in seconds")
    request_timeout: float = Field(default=120.0, description="Request timeout in seconds")

    @field_validator("api_key", mode="before")
    @classmethod
    def get_api_key(cls, v: str) -> str:
        """Get API key from environment if not set."""
        return v or os.environ.get("OPENAI_API_KEY", "")


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    model_config = SettingsConfigDict(env_prefix="RATELIMIT_")

    requests_per_second: float = Field(
        default=1.0, description="Max requests per second per host"
    )
    concurrent_requests: int = Field(default=5, description="Max concurrent requests")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff factor")
    max_delay: float = Field(default=60.0, description="Max delay between retries")


class CacheSettings(BaseSettings):
    """Cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CACHE_")

    enabled: bool = Field(default=True, description="Enable caching")
    directory: Path = Field(
        default=Path("data/cache"), description="Cache directory path"
    )
    ttl_days: int = Field(default=30, description="Cache TTL in days")
    max_size_mb: int = Field(default=1000, description="Max cache size in MB")


class DiscoverySettings(BaseSettings):
    """Discovery configuration."""

    model_config = SettingsConfigDict(env_prefix="DISCOVERY_")

    min_document_length: int = Field(
        default=10000, description="Minimum document length in characters"
    )
    max_document_length: int = Field(
        default=500000, description="Maximum document length in characters"
    )
    priority_threshold: float = Field(
        default=0.5, description="Minimum priority score to fetch"
    )
    max_cases_per_query: int = Field(
        default=100, description="Max cases to discover per query"
    )
    respect_robots_txt: bool = Field(default=True, description="Respect robots.txt")


class ExtractionSettings(BaseSettings):
    """Extraction configuration."""

    model_config = SettingsConfigDict(env_prefix="EXTRACTION_")

    chunk_size: int = Field(default=12000, description="Chunk size for long documents")
    chunk_overlap: int = Field(default=500, description="Overlap between chunks")
    max_evidence_items: int = Field(
        default=200, description="Max evidence items to extract"
    )
    min_decision_points: int = Field(
        default=2, description="Minimum decision points required"
    )


class QASettings(BaseSettings):
    """Quality assurance configuration."""

    model_config = SettingsConfigDict(env_prefix="QA_")

    min_evidence_coverage: float = Field(
        default=0.8, description="Minimum evidence coverage score"
    )
    min_simulation_completeness: float = Field(
        default=0.7, description="Minimum simulation completeness score"
    )
    strict_validation: bool = Field(
        default=True, description="Enable strict schema validation"
    )


class PathSettings(BaseSettings):
    """Path configuration."""

    model_config = SettingsConfigDict(env_prefix="PATH_")

    data_dir: Path = Field(default=Path("data"), description="Data directory")
    raw_dir: Path = Field(default=Path("data/raw"), description="Raw documents directory")
    processed_dir: Path = Field(
        default=Path("data/processed"), description="Processed data directory"
    )
    exports_dir: Path = Field(default=Path("data/exports"), description="Exports directory")
    seeds_dir: Path = Field(default=Path("data/seeds"), description="Seeds directory")
    prompts_dir: Path = Field(
        default=Path("casesim/extraction/prompts"), description="Prompts directory"
    )
    database_path: Path = Field(
        default=Path("data/cases_index.sqlite"), description="SQLite database path"
    )

    def ensure_dirs(self) -> None:
        """Ensure all directories exist."""
        for field_name in self.model_fields:
            path = getattr(self, field_name)
            if isinstance(path, Path) and "dir" in field_name:
                path.mkdir(parents=True, exist_ok=True)


class SourceSettings(BaseSettings):
    """Source-specific configuration."""

    model_config = SettingsConfigDict(env_prefix="SOURCE_", env_file=".env", extra="ignore")

    # BAILII settings
    bailii_base_url: str = Field(
        default="https://www.bailii.org", description="BAILII base URL"
    )
    bailii_search_url: str = Field(
        default="https://www.bailii.org/cgi-bin/find_by_content.cgi",
        description="BAILII search URL",
    )

    # CanLII settings
    canlii_base_url: str = Field(
        default="https://www.canlii.org", description="CanLII base URL"
    )
    canlii_api_key: str = Field(default="", description="CanLII API key (optional)")

    # AustLII settings
    austlii_base_url: str = Field(
        default="https://www.austlii.edu.au", description="AustLII base URL"
    )

    # CourtListener settings
    courtlistener_base_url: str = Field(
        default="https://www.courtlistener.com", description="CourtListener base URL"
    )
    courtlistener_api_token: str = Field(
        default="", description="CourtListener API token (optional)"
    )


class Settings(BaseSettings):
    """Main settings aggregator."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    discovery: DiscoverySettings = Field(default_factory=DiscoverySettings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    qa: QASettings = Field(default_factory=QASettings)
    paths: PathSettings = Field(default_factory=PathSettings)
    sources: SourceSettings = Field(default_factory=SourceSettings)

    # Global settings
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    version: str = Field(default="0.1.0", description="Application version")

    def ensure_dirs(self) -> None:
        """Ensure all required directories exist."""
        self.paths.ensure_dirs()
        self.cache.directory.mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_dirs()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    _settings.ensure_dirs()
    return _settings
