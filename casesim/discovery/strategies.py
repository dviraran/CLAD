"""Discovery strategies for finding medical malpractice cases."""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import yaml

from ..config import get_settings
from ..schemas import (
    DiscoveryRecord,
    KeywordConfig,
    SeedCase,
    Source,
)
from ..sources import BaseSource, SearchResult
from ..utils import get_logger


@dataclass
class DiscoveryBatch:
    """A batch of discovered cases."""

    records: list[DiscoveryRecord]
    strategy: str
    query_info: dict = field(default_factory=dict)


class DiscoveryStrategy(ABC):
    """Base class for discovery strategies."""

    name: str

    def __init__(self):
        self.logger = get_logger(f"discovery.{self.name}")
        self.settings = get_settings()

    @abstractmethod
    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases using this strategy."""
        ...


class KeywordStrategy(DiscoveryStrategy):
    """Keyword-based discovery strategy."""

    name = "keyword_search"

    def __init__(self, config: KeywordConfig | None = None):
        super().__init__()
        self.config = config or KeywordConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> "KeywordStrategy":
        """Load keyword configuration from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        config = KeywordConfig(**data)
        return cls(config)

    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases using keyword search."""
        # Build query combinations
        queries = self._build_queries()

        for query_terms in queries:
            self.logger.info(f"Searching with terms: {query_terms}")

            records: list[DiscoveryRecord] = []

            try:
                async for result in source.search(
                    keywords=query_terms,
                    max_results=max_results,
                ):
                    record = source.build_discovery_record(
                        search_result=result,
                        discovery_methods=[self.name],
                        query_terms=query_terms,
                    )
                    records.append(record)

                    if len(records) >= max_results:
                        break

                if records:
                    yield DiscoveryBatch(
                        records=records,
                        strategy=self.name,
                        query_info={"terms": query_terms},
                    )

            except Exception as e:
                self.logger.error(f"Search error for {query_terms}: {e}")

    def _build_queries(self) -> list[list[str]]:
        """Build query combinations from config."""
        queries = []

        # Each legal term paired with clinical terms
        for legal_term in self.config.legal_terms:
            # Legal term alone
            queries.append([legal_term])

            # Legal term + each clinical term
            for clinical_term in self.config.clinical_signal_terms[:5]:
                queries.append([legal_term, clinical_term])

        # Specific medical malpractice queries
        specific_queries = [
            ["clinical negligence", "surgery"],
            ["medical negligence", "diagnosis"],
            ["informed consent", "surgery"],
            ["breach of duty", "hospital"],
            ["failure to warn", "operation"],
            ["medical malpractice", "patient"],
        ]
        queries.extend(specific_queries)

        # Add required terms if specified
        if self.config.required_terms:
            queries = [q + self.config.required_terms for q in queries]

        return queries


class CitationStrategy(DiscoveryStrategy):
    """Citation-network expansion strategy."""

    name = "citation_expansion"

    def __init__(self, seed_case_ids: list[str] | None = None):
        super().__init__()
        self.seed_case_ids = seed_case_ids or []
        self.discovered_citations: set[str] = set()

    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases by following citations."""
        from ..utils import CaseDatabase

        db = CaseDatabase(self.settings.paths.database_path)

        # Get cases to expand from
        cases_to_expand = []

        if self.seed_case_ids:
            for case_id in self.seed_case_ids:
                record = db.get_case(case_id)
                if record and record.raw_file_path:
                    cases_to_expand.append(record)
        else:
            # Get recently fetched cases
            from ..schemas import DiscoveryStatus
            cases_to_expand = db.get_cases_by_status(DiscoveryStatus.FETCHED, limit=50)

        self.logger.info(f"Expanding citations from {len(cases_to_expand)} cases")

        for record in cases_to_expand:
            if not record.raw_file_path:
                continue

            try:
                # Read raw content
                raw_path = Path(record.raw_file_path)
                if not raw_path.exists():
                    continue

                content = raw_path.read_text(encoding="utf-8", errors="ignore")

                # Extract citations
                citations = source.extract_citations(content)
                self.logger.debug(f"Found {len(citations)} citations in {record.case_id}")

                new_records: list[DiscoveryRecord] = []

                for citation in citations:
                    if citation in self.discovered_citations:
                        continue

                    self.discovered_citations.add(citation)

                    # Try to convert citation to URL
                    url = source.parse_citation_url(citation)
                    if not url:
                        continue

                    # Check if already in database
                    existing = db.get_case_by_url(url)
                    if existing:
                        # Add citation link
                        db.add_citation(record.case_id, existing.case_id, citation)
                        continue

                    # Create new discovery record
                    from ..utils import generate_case_id
                    from ..schemas import DiscoveryStatus

                    # Generate ID from citation
                    clean_citation = re.sub(r"[^\w\s-]", "", citation)
                    case_id = generate_case_id(source.source.value, clean_citation[:30])

                    new_record = DiscoveryRecord(
                        case_id=case_id,
                        source=source.source,
                        jurisdiction=source.jurisdiction,
                        title=citation,
                        url=url,
                        discovery_methods=[self.name],
                        query_terms=None,
                        priority_score=0.7,  # Cited cases get higher priority
                        status=DiscoveryStatus.QUEUED,
                        cited_by=[record.case_id],
                    )
                    new_records.append(new_record)

                    # Add citation link
                    db.add_citation(record.case_id, case_id, citation)

                    if len(new_records) >= max_results:
                        break

                if new_records:
                    yield DiscoveryBatch(
                        records=new_records,
                        strategy=self.name,
                        query_info={"source_case": record.case_id},
                    )

            except Exception as e:
                self.logger.error(f"Citation extraction error for {record.case_id}: {e}")


class SeedStrategy(DiscoveryStrategy):
    """Discovery from curated seed lists."""

    name = "seed_list"

    def __init__(self, seeds: list[SeedCase] | None = None):
        super().__init__()
        self.seeds = seeds or []

    @classmethod
    def from_yaml(cls, path: Path) -> "SeedStrategy":
        """Load seeds from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        seeds = []
        for item in data.get("cases", []):
            seeds.append(SeedCase(**item))

        return cls(seeds)

    @classmethod
    def from_directory(cls, directory: Path) -> "SeedStrategy":
        """Load seeds from all YAML files in a directory."""
        seeds = []

        for yaml_file in directory.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)

            for item in data.get("cases", []):
                seeds.append(SeedCase(**item))

        return cls(seeds)

    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases from seed list."""
        from ..utils import CaseDatabase, generate_case_id
        from ..schemas import DiscoveryStatus
        from urllib.parse import urlparse

        db = CaseDatabase(self.settings.paths.database_path)
        records: list[DiscoveryRecord] = []

        for seed in self.seeds:
            if len(records) >= max_results:
                break

            # Check if URL matches source
            url_host = urlparse(str(seed.url)).netloc
            source_host = urlparse(source.base_url).netloc

            if source_host not in url_host:
                continue

            # Check if already in database
            existing = db.get_case_by_url(str(seed.url))
            if existing:
                self.logger.debug(f"Seed already exists: {seed.url}")
                continue

            # Create discovery record
            url_path = urlparse(str(seed.url)).path
            case_id = generate_case_id(
                source.source.value,
                url_path.replace("/", "-")[:30]
            )

            record = DiscoveryRecord(
                case_id=case_id,
                source=source.source,
                jurisdiction=source.jurisdiction,
                title=seed.title or f"Seed case from {seed.url}",
                url=seed.url,
                discovery_methods=[self.name],
                priority_score=seed.priority,
                status=DiscoveryStatus.QUEUED,
            )
            records.append(record)

        if records:
            yield DiscoveryBatch(
                records=records,
                strategy=self.name,
                query_info={"seed_count": len(self.seeds)},
            )


class HistoricalSweepStrategy(DiscoveryStrategy):
    """Systematic year-by-year discovery for historical cases."""

    name = "historical_sweep"

    def __init__(
        self,
        start_year: int = 2000,
        end_year: int = 2024,
        keywords: list[str] | None = None,
    ):
        super().__init__()
        self.start_year = start_year
        self.end_year = end_year
        self.keywords = keywords or ["clinical negligence", "medical negligence"]

    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases by sweeping through years."""
        results_per_year = max(1, max_results // (self.end_year - self.start_year + 1))

        for year in range(self.end_year, self.start_year - 1, -1):
            self.logger.info(f"Sweeping year {year}")

            records: list[DiscoveryRecord] = []

            try:
                async for result in source.search(
                    keywords=self.keywords,
                    date_from=f"{year}-01-01",
                    date_to=f"{year}-12-31",
                    max_results=results_per_year,
                ):
                    record = source.build_discovery_record(
                        search_result=result,
                        discovery_methods=[self.name],
                        query_terms=self.keywords,
                    )
                    records.append(record)

                if records:
                    yield DiscoveryBatch(
                        records=records,
                        strategy=self.name,
                        query_info={"year": year, "keywords": self.keywords},
                    )

            except Exception as e:
                self.logger.error(f"Historical sweep error for year {year}: {e}")


class EnhancedCitationStrategy(DiscoveryStrategy):
    """Multi-hop citation expansion with medical relevance filtering."""

    name = "enhanced_citation_expansion"

    def __init__(
        self,
        seed_case_ids: list[str] | None = None,
        max_depth: int = 2,
        medical_filter: bool = True,
    ):
        super().__init__()
        self.seed_case_ids = seed_case_ids or []
        self.max_depth = max_depth
        self.medical_filter = medical_filter
        self.discovered_citations: set[str] = set()

    # Medical terms to filter citations by relevance
    MEDICAL_INDICATORS = [
        "negligence", "malpractice", "surgery", "patient", "doctor",
        "hospital", "diagnosis", "treatment", "consent", "medical",
        "clinical", "physician", "nurse", "care", "injury", "death",
        "causation", "breach", "duty", "standard",
    ]

    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases through multi-hop citation expansion."""
        from ..utils import CaseDatabase
        from ..schemas import DiscoveryStatus
        from pathlib import Path

        db = CaseDatabase(self.settings.paths.database_path)

        # Track cases at each depth level
        current_level_cases: list[DiscoveryRecord] = []

        # Initialize with seed cases or recently fetched cases
        if self.seed_case_ids:
            for case_id in self.seed_case_ids:
                record = db.get_case(case_id)
                if record and record.raw_file_path:
                    current_level_cases.append(record)
        else:
            current_level_cases = db.get_cases_by_status(DiscoveryStatus.FETCHED, limit=50)

        total_found = 0

        for depth in range(self.max_depth):
            if total_found >= max_results:
                break

            self.logger.info(f"Citation expansion depth {depth + 1}/{self.max_depth}, "
                           f"expanding {len(current_level_cases)} cases")

            next_level_cases: list[DiscoveryRecord] = []

            for record in current_level_cases:
                if total_found >= max_results:
                    break

                if not record.raw_file_path:
                    continue

                try:
                    raw_path = Path(record.raw_file_path)
                    if not raw_path.exists():
                        continue

                    content = raw_path.read_text(encoding="utf-8", errors="ignore")

                    # Extract citations
                    citations = source.extract_citations(content)

                    new_records: list[DiscoveryRecord] = []

                    for citation in citations:
                        if citation in self.discovered_citations:
                            continue

                        self.discovered_citations.add(citation)

                        # Apply medical relevance filter
                        if self.medical_filter:
                            # Check if citation context has medical terms
                            citation_context = self._get_citation_context(content, citation)
                            if not self._is_medical_relevant(citation_context):
                                continue

                        # Convert to URL
                        url = source.parse_citation_url(citation)
                        if not url:
                            continue

                        # Check database
                        existing = db.get_case_by_url(url)
                        if existing:
                            db.add_citation(record.case_id, existing.case_id, citation)
                            continue

                        # Create new record
                        from ..utils import generate_case_id

                        clean_citation = re.sub(r"[^\w\s-]", "", citation)
                        case_id = generate_case_id(source.source.value, clean_citation[:30])

                        # Priority decreases with depth
                        priority = max(0.5, 0.8 - (depth * 0.1))

                        new_record = DiscoveryRecord(
                            case_id=case_id,
                            source=source.source,
                            jurisdiction=source.jurisdiction,
                            title=citation,
                            url=url,
                            discovery_methods=[self.name, f"depth_{depth + 1}"],
                            priority_score=priority,
                            status=DiscoveryStatus.QUEUED,
                            cited_by=[record.case_id],
                        )
                        new_records.append(new_record)
                        next_level_cases.append(new_record)

                        db.add_citation(record.case_id, case_id, citation)
                        total_found += 1

                        if total_found >= max_results:
                            break

                    if new_records:
                        yield DiscoveryBatch(
                            records=new_records,
                            strategy=self.name,
                            query_info={
                                "source_case": record.case_id,
                                "depth": depth + 1,
                            },
                        )

                except Exception as e:
                    self.logger.error(f"Citation extraction error: {e}")

            # Move to next depth level
            current_level_cases = next_level_cases

            if not current_level_cases:
                self.logger.info(f"No more cases to expand at depth {depth + 1}")
                break

    def _get_citation_context(self, content: str, citation: str, window: int = 200) -> str:
        """Get text surrounding a citation for context analysis."""
        idx = content.find(citation)
        if idx == -1:
            return ""
        start = max(0, idx - window)
        end = min(len(content), idx + len(citation) + window)
        return content[start:end]

    def _is_medical_relevant(self, text: str) -> bool:
        """Check if text contains medical indicators."""
        text_lower = text.lower()
        matches = sum(1 for term in self.MEDICAL_INDICATORS if term in text_lower)
        return matches >= 2  # Require at least 2 medical terms in context


class BlogSeedStrategy(DiscoveryStrategy):
    """Discover cases from law firm blog citations."""

    name = "blog_seed"

    def __init__(
        self,
        jurisdictions: list[str] | None = None,
        max_articles_per_blog: int = 30,
    ):
        """Initialize the blog seed strategy.

        Args:
            jurisdictions: Filter to specific jurisdictions (e.g., ["UK", "US"])
            max_articles_per_blog: Max articles to fetch per blog
        """
        super().__init__()
        self.jurisdictions = jurisdictions
        self.max_articles_per_blog = max_articles_per_blog

    async def discover(
        self,
        source: BaseSource,
        max_results: int = 100,
    ) -> AsyncIterator[DiscoveryBatch]:
        """Discover cases by extracting citations from law firm blogs."""
        from ..utils import CaseDatabase, generate_case_id
        from ..schemas import DiscoveryStatus
        from .blog_scraper import BlogCitationExtractor, MALPRACTICE_BLOGS
        from .citation_resolver import CitationResolver

        db = CaseDatabase(self.settings.paths.database_path)
        extractor = BlogCitationExtractor()
        resolver = CitationResolver()

        records: list[DiscoveryRecord] = []
        seen_urls: set[str] = set()

        try:
            # Filter blogs to match source jurisdiction
            source_jurisdiction = source.jurisdiction.value

            for blog in MALPRACTICE_BLOGS:
                if len(records) >= max_results:
                    break

                # Filter by jurisdiction
                if self.jurisdictions and blog.jurisdiction not in self.jurisdictions:
                    continue

                # Only process blogs matching the source's jurisdiction
                if blog.jurisdiction != source_jurisdiction:
                    continue

                self.logger.info(f"Processing blog: {blog.name}")

                async for citation, article_url, jurisdiction in extractor.extract_from_blog(
                    blog, max_articles=self.max_articles_per_blog
                ):
                    if len(records) >= max_results:
                        break

                    # Resolve citation to URL
                    url = resolver.resolve(citation, jurisdiction)
                    if not url:
                        continue

                    # Deduplicate
                    if url in seen_urls:
                        continue
                    seen_urls.add(url)

                    # Check if already in database
                    existing = db.get_case_by_url(url)
                    if existing:
                        self.logger.debug(f"Already in DB: {citation}")
                        continue

                    # Create discovery record
                    clean_citation = re.sub(r"[^\w\s-]", "", citation)
                    case_id = generate_case_id(source.source.value, clean_citation[:30])

                    record = DiscoveryRecord(
                        case_id=case_id,
                        source=source.source,
                        jurisdiction=source.jurisdiction,
                        title=citation,
                        url=url,
                        discovery_methods=[self.name, f"blog:{blog.name}"],
                        priority_score=0.8,  # High priority - curated source
                        status=DiscoveryStatus.QUEUED,
                    )
                    records.append(record)
                    self.logger.info(f"Found case from blog: {citation}")

        except Exception as e:
            self.logger.error(f"Blog scraping error: {e}")
        finally:
            await extractor.close()

        if records:
            yield DiscoveryBatch(
                records=records,
                strategy=self.name,
                query_info={
                    "blogs_processed": len([b for b in MALPRACTICE_BLOGS
                                           if not self.jurisdictions or b.jurisdiction in self.jurisdictions]),
                    "citations_found": len(records),
                },
            )
