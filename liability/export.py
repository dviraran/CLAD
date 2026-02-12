"""Export functions for CSV, Parquet, and SQLite."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from .models import CriterionDetail, RunRecord


def export_csv(
    records: list[RunRecord],
    output_path: Path,
    criteria: list[CriterionDetail] | None = None,
) -> tuple[Path, Path | None]:
    """
    Export records to CSV files.

    Args:
        records: List of RunRecord to export
        output_path: Path for main CSV file (runs.csv)
        criteria: Optional list of CriterionDetail for detail table

    Returns:
        Tuple of (main_csv_path, criteria_csv_path or None)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export main table
    if records:
        fieldnames = list(records[0].to_dict().keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_dict())

    # Export criteria table if provided
    criteria_path = None
    if criteria:
        criteria_path = output_path.parent / "criteria.csv"
        fieldnames = list(criteria[0].to_dict().keys())

        with open(criteria_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for item in criteria:
                writer.writerow(item.to_dict())

    return output_path, criteria_path


def export_parquet(
    records: list[RunRecord],
    output_path: Path,
    criteria: list[CriterionDetail] | None = None,
) -> tuple[Path, Path | None]:
    """
    Export records to Parquet files.

    Args:
        records: List of RunRecord to export
        output_path: Path for main Parquet file (runs.parquet)
        criteria: Optional list of CriterionDetail for detail table

    Returns:
        Tuple of (main_parquet_path, criteria_parquet_path or None)
    """
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        raise ImportError(
            "pyarrow is required for Parquet export. Install with: pip install pyarrow"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export main table
    if records:
        data = [record.to_dict() for record in records]
        table = pa.Table.from_pylist(data)
        pq.write_table(table, output_path)

    # Export criteria table if provided
    criteria_path = None
    if criteria:
        criteria_path = output_path.parent / "criteria.parquet"
        data = [item.to_dict() for item in criteria]
        table = pa.Table.from_pylist(data)
        pq.write_table(table, criteria_path)

    return output_path, criteria_path


def export_sqlite(
    records: list[RunRecord],
    output_path: Path,
    criteria: list[CriterionDetail] | None = None,
) -> Path:
    """
    Export records to SQLite database.

    Args:
        records: List of RunRecord to export
        output_path: Path for SQLite database file
        criteria: Optional list of CriterionDetail for detail table

    Returns:
        Path to SQLite database
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to start fresh
    if output_path.exists():
        output_path.unlink()

    conn = sqlite3.connect(output_path)
    cursor = conn.cursor()

    # Create runs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            case_id TEXT,
            jurisdiction TEXT,
            specialty TEXT,
            malpractice_type TEXT,
            liability_code INTEGER,
            llm_name TEXT,
            score_0_2 INTEGER,
            risk_flag INTEGER,
            defendant_action TEXT,
            expected_action TEXT,
            missing_criteria_count INTEGER,
            met_criteria_count INTEGER,
            reasoning_quality_score REAL,
            started_at TEXT,
            ended_at TEXT,
            feedback TEXT,
            recommendation_length INTEGER,
            questions_asked INTEGER,
            score_valid INTEGER,
            deferral_reason TEXT,
            flesch_kincaid_grade REAL,
            smog_index REAL,
            transformer_readability_score REAL,
            transformer_model_name TEXT,
            lexical_overlap_adjacent REAL,
            lexical_overlap_global REAL,
            pronoun_density REAL,
            semantic_coherence_local REAL,
            semantic_coherence_global REAL
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_case_id ON runs(case_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_llm_name ON runs(llm_name)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_jurisdiction ON runs(jurisdiction)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_malpractice_type ON runs(malpractice_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_liability_code ON runs(liability_code)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_specialty ON runs(specialty)")

    # Insert records
    for record in records:
        d = record.to_dict()
        cursor.execute("""
            INSERT INTO runs (
                run_id, case_id, jurisdiction, specialty, malpractice_type,
                liability_code, llm_name, score_0_2, risk_flag, defendant_action,
                expected_action, missing_criteria_count, met_criteria_count,
                reasoning_quality_score, started_at, ended_at, feedback,
                recommendation_length, questions_asked, score_valid, deferral_reason,
                flesch_kincaid_grade, smog_index, transformer_readability_score,
                transformer_model_name, lexical_overlap_adjacent, lexical_overlap_global,
                pronoun_density, semantic_coherence_local, semantic_coherence_global
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            d["run_id"],
            d["case_id"],
            d["jurisdiction"],
            d["specialty"],
            d["malpractice_type"],
            d["liability_code"],
            d["llm_name"],
            d["score_0_2"],
            1 if d["risk_flag"] else 0,
            d["defendant_action"],
            d["expected_action"],
            d["missing_criteria_count"],
            d["met_criteria_count"],
            d["reasoning_quality_score"],
            d["started_at"],
            d["ended_at"],
            d["feedback"],
            d["recommendation_length"],
            d["questions_asked"],
            1 if d["score_valid"] else 0,
            d["deferral_reason"],
            d["flesch_kincaid_grade"],
            d["smog_index"],
            d["transformer_readability_score"],
            d["transformer_model_name"],
            d["lexical_overlap_adjacent"],
            d["lexical_overlap_global"],
            d["pronoun_density"],
            d["semantic_coherence_local"],
            d["semantic_coherence_global"],
        ))

    # Create criteria table if data provided
    if criteria:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS criteria (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                criterion TEXT,
                met INTEGER,
                reason TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_criteria_run_id ON criteria(run_id)")

        for item in criteria:
            d = item.to_dict()
            cursor.execute("""
                INSERT INTO criteria (run_id, criterion, met, reason)
                VALUES (?, ?, ?, ?)
            """, (
                d["run_id"],
                d["criterion"],
                1 if d["met"] else 0,
                d["reason"],
            ))

    conn.commit()
    conn.close()

    return output_path


def export_all(
    records: list[RunRecord],
    output_dir: Path,
    criteria: list[CriterionDetail] | None = None,
) -> dict[str, Path]:
    """
    Export records to all formats (CSV, Parquet, SQLite).

    Args:
        records: List of RunRecord to export
        output_dir: Directory for output files
        criteria: Optional list of CriterionDetail for detail table

    Returns:
        Dictionary mapping format name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, Path] = {}

    # CSV
    csv_path, _ = export_csv(
        records,
        output_dir / "runs.csv",
        criteria,
    )
    outputs["csv"] = csv_path

    # Parquet
    try:
        parquet_path, _ = export_parquet(
            records,
            output_dir / "runs.parquet",
            criteria,
        )
        outputs["parquet"] = parquet_path
    except ImportError:
        # Parquet optional
        pass

    # SQLite
    sqlite_path = export_sqlite(
        records,
        output_dir / "runs.sqlite",
        criteria,
    )
    outputs["sqlite"] = sqlite_path

    return outputs
