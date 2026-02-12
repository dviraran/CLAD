"""Main discovery engine for orchestrating case discovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator

from ..config import get_settings
from ..schemas import (
    DiscoveryRecord,
    DiscoveryStatus,
    KeywordConfig,
    RejectionReason,
    StructuralHeuristicsConfig,
)
from ..sources import BaseSource, get_source
from ..utils import CaseDatabase, get_logger, hash_content
from .filters import CaseFilter, ContentFilter, StructuralFilter
from .strategies import (
    CitationStrategy,
    DiscoveryBatch,
    DiscoveryStrategy,
    KeywordStrategy,
    SeedStrategy,
)


@dataclass
class DiscoveryStats:
    """Statistics from a discovery run."""

    started_at: datetime
    completed_at: datetime | None = None
    cases_discovered: int = 0
    cases_added: int = 0
    cases_rejected: int = 0
    cases_duplicate: int = 0
    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    strategies_used: list[str] = field(default_factory=list)


class DiscoveryEngine:
    """Main engine for discovering medical malpractice cases."""

    def __init__(
        self,
        source: str | BaseSource,
        strategies: list[DiscoveryStrategy] | None = None,
        filters: list[CaseFilter] | None = None,
    ):
        """Initialize the discovery engine."""
        self.settings = get_settings()
        self.logger = get_logger("discovery")

        # Set up source
        if isinstance(source, str):
            self._source_class = get_source(source)
            self._source_instance: BaseSource | None = None
        else:
            self._source_class = type(source)
            self._source_instance = source

        # Set up strategies (default to keyword search)
        self.strategies = strategies or [KeywordStrategy()]

        # Set up filters
        self.filters = filters or [
            StructuralFilter(),
            ContentFilter(),
        ]

        # Database
        self.db = CaseDatabase(self.settings.paths.database_path)

    async def discover(
        self,
        max_results: int = 100,
        fetch_content: bool = True,
        apply_filters: bool = True,
    ) -> DiscoveryStats:
        """Run discovery pipeline."""
        stats = DiscoveryStats(
            started_at=datetime.utcnow(),
            strategies_used=[s.name for s in self.strategies],
        )

        # Get or create source instance
        if self._source_instance:
            source = self._source_instance
        else:
            source = self._source_class()

        async with source:
            for strategy in self.strategies:
                self.logger.info(f"Running strategy: {strategy.name}")

                async for batch in strategy.discover(source, max_results=max_results):
                    for record in batch.records:
                        stats.cases_discovered += 1

                        # Try to add to database
                        if not self.db.add_case(record):
                            stats.cases_duplicate += 1
                            continue

                        # Optionally fetch content for filtering
                        content: str | None = None
                        if fetch_content or apply_filters:
                            try:
                                fetch_result = await source.fetch(str(record.url))
                                content = fetch_result.content

                                # Update record with fetch info
                                raw_path = await source.save_raw(
                                    str(record.url),
                                    content,
                                    record.case_id,
                                )
                                self.db.update_fetched(
                                    record.case_id,
                                    str(raw_path),
                                    fetch_result.content_hash,
                                    len(content),
                                )
                            except Exception as e:
                                self.logger.warning(
                                    f"Fetch error for {record.case_id}: {e}"
                                )
                                self.db.update_status(
                                    record.case_id,
                                    DiscoveryStatus.REJECTED,
                                    RejectionReason.FETCH_ERROR,
                                )
                                stats.cases_rejected += 1
                                stats.rejections_by_reason["FETCH_ERROR"] = (
                                    stats.rejections_by_reason.get("FETCH_ERROR", 0) + 1
                                )
                                continue

                        # Apply filters
                        if apply_filters and content:
                            passed, rejection_reason = self._apply_filters(
                                record, content
                            )

                            if not passed:
                                self.db.update_status(
                                    record.case_id,
                                    DiscoveryStatus.REJECTED,
                                    rejection_reason,
                                )
                                stats.cases_rejected += 1
                                reason_key = (
                                    rejection_reason.value if rejection_reason else "UNKNOWN"
                                )
                                stats.rejections_by_reason[reason_key] = (
                                    stats.rejections_by_reason.get(reason_key, 0) + 1
                                )
                                continue

                        stats.cases_added += 1

                        if stats.cases_added >= max_results:
                            break

                    if stats.cases_added >= max_results:
                        break

                if stats.cases_added >= max_results:
                    break

        stats.completed_at = datetime.utcnow()
        self.logger.info(
            f"Discovery complete: {stats.cases_added} added, "
            f"{stats.cases_rejected} rejected, {stats.cases_duplicate} duplicates"
        )

        return stats

    def _apply_filters(
        self,
        record: DiscoveryRecord,
        content: str,
    ) -> tuple[bool, RejectionReason | None]:
        """Apply all filters to a case."""
        total_score_adjustment = 0.0

        for filter_obj in self.filters:
            result = filter_obj.apply(record, content)

            if not result.passed:
                return False, result.reason

            total_score_adjustment += result.score_adjustment

        # Update priority score
        if record.priority_score is not None:
            new_score = min(1.0, max(0.0, record.priority_score + total_score_adjustment))
            # Note: Would need to update in DB if we want to persist this
            record.priority_score = new_score

        return True, None

    async def expand_citations(
        self,
        source_case_ids: list[str] | None = None,
        max_results: int = 100,
    ) -> DiscoveryStats:
        """Expand discovery through citation network."""
        citation_strategy = CitationStrategy(seed_case_ids=source_case_ids)

        # Temporarily replace strategies
        original_strategies = self.strategies
        self.strategies = [citation_strategy]

        try:
            return await self.discover(max_results=max_results)
        finally:
            self.strategies = original_strategies

    async def load_seeds(
        self,
        seeds_path: Path,
        max_results: int = 100,
    ) -> DiscoveryStats:
        """Load cases from seed files."""
        if seeds_path.is_dir():
            seed_strategy = SeedStrategy.from_directory(seeds_path)
        else:
            seed_strategy = SeedStrategy.from_yaml(seeds_path)

        original_strategies = self.strategies
        self.strategies = [seed_strategy]

        try:
            return await self.discover(max_results=max_results, apply_filters=False)
        finally:
            self.strategies = original_strategies

    def get_queue_stats(self) -> dict[str, int]:
        """Get current queue statistics."""
        return self.db.get_queue_stats()

    def get_pending_cases(self, limit: int = 100) -> list[DiscoveryRecord]:
        """Get cases pending fetch."""
        return self.db.get_cases_by_status(DiscoveryStatus.QUEUED, limit=limit)

    def get_fetched_cases(self, limit: int = 100) -> list[DiscoveryRecord]:
        """Get fetched cases ready for extraction."""
        return self.db.get_cases_by_status(DiscoveryStatus.FETCHED, limit=limit)


async def run_discovery(
    source: str,
    keywords_path: Path | None = None,
    seeds_path: Path | None = None,
    expand_citations: bool = False,
    max_results: int = 100,
    min_length: int = 10000,
) -> DiscoveryStats:
    """Run discovery with standard configuration."""
    logger = get_logger("discovery")

    # Build strategies
    strategies: list[DiscoveryStrategy] = []

    if keywords_path and keywords_path.exists():
        strategies.append(KeywordStrategy.from_yaml(keywords_path))
    else:
        strategies.append(KeywordStrategy())

    if seeds_path and seeds_path.exists():
        if seeds_path.is_dir():
            strategies.append(SeedStrategy.from_directory(seeds_path))
        else:
            strategies.append(SeedStrategy.from_yaml(seeds_path))

    # Build filters
    filters = [
        StructuralFilter(StructuralHeuristicsConfig(min_length=min_length)),
        ContentFilter(),
    ]

    # Create engine
    engine = DiscoveryEngine(
        source=source,
        strategies=strategies,
        filters=filters,
    )

    # Run discovery
    stats = await engine.discover(max_results=max_results)

    # Optionally expand citations
    if expand_citations and stats.cases_added > 0:
        logger.info("Expanding through citation network...")
        citation_stats = await engine.expand_citations(max_results=max_results)
        stats.cases_added += citation_stats.cases_added
        stats.cases_rejected += citation_stats.cases_rejected

    return stats
