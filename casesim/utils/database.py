"""SQLite database management for case discovery and tracking."""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from ..schemas import DiscoveryRecord, DiscoveryStatus, RejectionReason, Source, Jurisdiction


class CaseDatabase:
    """SQLite database for tracking discovered cases."""

    def __init__(self, db_path: Path):
        """Initialize the database."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    jurisdiction TEXT NOT NULL,
                    court TEXT,
                    title TEXT NOT NULL,
                    year INTEGER,
                    url TEXT UNIQUE NOT NULL,
                    discovery_methods TEXT,
                    query_terms TEXT,
                    estimated_length INTEGER,
                    priority_score REAL,
                    status TEXT DEFAULT 'queued',
                    rejection_reason TEXT,
                    discovered_at TEXT NOT NULL,
                    fetched_at TEXT,
                    raw_file_path TEXT,
                    content_hash TEXT,
                    cited_by TEXT,
                    cites TEXT,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_cases_status ON cases(status);
                CREATE INDEX IF NOT EXISTS idx_cases_source ON cases(source);
                CREATE INDEX IF NOT EXISTS idx_cases_priority ON cases(priority_score DESC);
                CREATE INDEX IF NOT EXISTS idx_cases_url ON cases(url);

                CREATE TABLE IF NOT EXISTS extractions (
                    case_id TEXT PRIMARY KEY,
                    extracted_at TEXT NOT NULL,
                    extractor_version TEXT,
                    model_used TEXT,
                    extraction_passes INTEGER,
                    output_path TEXT,
                    quality_score REAL,
                    validation_passed INTEGER,
                    errors TEXT,
                    FOREIGN KEY (case_id) REFERENCES cases(case_id)
                );

                CREATE TABLE IF NOT EXISTS citations (
                    from_case_id TEXT NOT NULL,
                    to_case_id TEXT NOT NULL,
                    citation_text TEXT,
                    discovered_at TEXT NOT NULL,
                    PRIMARY KEY (from_case_id, to_case_id),
                    FOREIGN KEY (from_case_id) REFERENCES cases(case_id)
                );

                CREATE INDEX IF NOT EXISTS idx_citations_to ON citations(to_case_id);
            """
            )

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection with WAL mode for better concurrent access."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=60000")  # 60 second busy timeout
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _execute_with_retry(
        self, operation: str, params: tuple = (), max_retries: int = 5, retry_delay: float = 1.0
    ) -> None:
        """Execute a SQL operation with retry logic for locked database."""
        last_error = None
        for attempt in range(max_retries):
            try:
                with self._connection() as conn:
                    conn.execute(operation, params)
                return
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e) and attempt < max_retries - 1:
                    last_error = e
                    time.sleep(retry_delay * (attempt + 1))
                    continue
                raise
        if last_error:
            raise last_error

    def add_case(self, record: DiscoveryRecord) -> bool:
        """Add a case to the database. Returns True if added, False if duplicate."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO cases (
                        case_id, source, jurisdiction, court, title, year, url,
                        discovery_methods, query_terms, estimated_length, priority_score,
                        status, rejection_reason, discovered_at, fetched_at,
                        raw_file_path, content_hash, cited_by, cites, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        record.case_id,
                        record.source.value,
                        record.jurisdiction.value,
                        record.court,
                        record.title,
                        record.year,
                        str(record.url),
                        json.dumps(record.discovery_methods),
                        json.dumps(record.query_terms) if record.query_terms else None,
                        record.estimated_length,
                        record.priority_score,
                        record.status.value,
                        record.rejection_reason.value if record.rejection_reason else None,
                        record.discovered_at.isoformat(),
                        record.fetched_at.isoformat() if record.fetched_at else None,
                        record.raw_file_path,
                        record.content_hash,
                        json.dumps(record.cited_by) if record.cited_by else None,
                        json.dumps(record.cites) if record.cites else None,
                        datetime.utcnow().isoformat(),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def update_status(
        self,
        case_id: str,
        status: DiscoveryStatus,
        rejection_reason: RejectionReason | None = None,
    ) -> None:
        """Update the status of a case with retry logic."""
        self._execute_with_retry(
            """
            UPDATE cases SET status = ?, rejection_reason = ?, updated_at = ?
            WHERE case_id = ?
            """,
            (
                status.value,
                rejection_reason.value if rejection_reason else None,
                datetime.utcnow().isoformat(),
                case_id,
            ),
        )

    def update_fetched(
        self,
        case_id: str,
        raw_file_path: str,
        content_hash: str,
        actual_length: int | None = None,
    ) -> None:
        """Update case after fetching."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE cases SET
                    status = ?,
                    fetched_at = ?,
                    raw_file_path = ?,
                    content_hash = ?,
                    estimated_length = COALESCE(?, estimated_length),
                    updated_at = ?
                WHERE case_id = ?
            """,
                (
                    DiscoveryStatus.FETCHED.value,
                    datetime.utcnow().isoformat(),
                    raw_file_path,
                    content_hash,
                    actual_length,
                    datetime.utcnow().isoformat(),
                    case_id,
                ),
            )

    def get_case(self, case_id: str) -> DiscoveryRecord | None:
        """Get a case by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_case_by_url(self, url: str) -> DiscoveryRecord | None:
        """Get a case by URL."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE url = ?", (url,)
            ).fetchone()
            if row:
                return self._row_to_record(row)
        return None

    def get_cases_by_status(
        self, status: DiscoveryStatus, limit: int = 100
    ) -> list[DiscoveryRecord]:
        """Get cases by status, ordered by priority."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM cases
                WHERE status = ?
                ORDER BY priority_score DESC NULLS LAST
                LIMIT ?
            """,
                (status.value, limit),
            ).fetchall()
            return [self._row_to_record(row) for row in rows]

    def get_queue_stats(self) -> dict[str, int]:
        """Get counts by status."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) as count FROM cases GROUP BY status"
            ).fetchall()
            return {row["status"]: row["count"] for row in rows}

    def add_citation(
        self, from_case_id: str, to_case_id: str, citation_text: str | None = None
    ) -> bool:
        """Add a citation link between cases."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO citations (from_case_id, to_case_id, citation_text, discovered_at)
                    VALUES (?, ?, ?, ?)
                """,
                    (from_case_id, to_case_id, citation_text, datetime.utcnow().isoformat()),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_cited_cases(self, case_id: str) -> list[str]:
        """Get cases cited by a case."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT to_case_id FROM citations WHERE from_case_id = ?", (case_id,)
            ).fetchall()
            return [row["to_case_id"] for row in rows]

    def get_citing_cases(self, case_id: str) -> list[str]:
        """Get cases that cite a case."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT from_case_id FROM citations WHERE to_case_id = ?", (case_id,)
            ).fetchall()
            return [row["from_case_id"] for row in rows]

    def _row_to_record(self, row: sqlite3.Row) -> DiscoveryRecord:
        """Convert a database row to a DiscoveryRecord."""
        return DiscoveryRecord(
            case_id=row["case_id"],
            source=Source(row["source"]),
            jurisdiction=Jurisdiction(row["jurisdiction"]),
            court=row["court"],
            title=row["title"],
            year=row["year"],
            url=row["url"],
            discovery_methods=json.loads(row["discovery_methods"]),
            query_terms=json.loads(row["query_terms"]) if row["query_terms"] else None,
            estimated_length=row["estimated_length"],
            priority_score=row["priority_score"],
            status=DiscoveryStatus(row["status"]),
            rejection_reason=(
                RejectionReason(row["rejection_reason"]) if row["rejection_reason"] else None
            ),
            discovered_at=datetime.fromisoformat(row["discovered_at"]),
            fetched_at=(
                datetime.fromisoformat(row["fetched_at"]) if row["fetched_at"] else None
            ),
            raw_file_path=row["raw_file_path"],
            content_hash=row["content_hash"],
            cited_by=json.loads(row["cited_by"]) if row["cited_by"] else None,
            cites=json.loads(row["cites"]) if row["cites"] else None,
        )

    def record_extraction(
        self,
        case_id: str,
        extractor_version: str,
        model_used: str,
        extraction_passes: int,
        output_path: str,
        quality_score: float | None = None,
        validation_passed: bool = False,
        errors: list[str] | None = None,
    ) -> None:
        """Record extraction results with retry logic."""
        self._execute_with_retry(
            """
            INSERT OR REPLACE INTO extractions (
                case_id, extracted_at, extractor_version, model_used,
                extraction_passes, output_path, quality_score,
                validation_passed, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                case_id,
                datetime.utcnow().isoformat(),
                extractor_version,
                model_used,
                extraction_passes,
                output_path,
                quality_score,
                1 if validation_passed else 0,
                json.dumps(errors) if errors else None,
            ),
        )
        # Also update case status
        self.update_status(case_id, DiscoveryStatus.EXTRACTED)
