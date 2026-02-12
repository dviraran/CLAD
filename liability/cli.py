"""Typer CLI for liability analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .export import export_all
from .ingest import LogIngester
from .models import MalpracticeType

app = typer.Typer(
    name="liability",
    help="Liability analysis tools for medical consultation simulation logs.",
)


@app.command()
def ingest(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing JSON log files",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_dir: Path = typer.Option(
        Path("exports"),
        "--out",
        "-o",
        help="Output directory for exported files",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed progress",
    ),
    compute_readability: bool = typer.Option(
        True,
        "--readability/--no-readability",
        help="Compute readability metrics (slower but recommended)",
    ),
    use_gpu: bool = typer.Option(
        False,
        "--gpu/--cpu",
        help="Use GPU acceleration for readability (requires CUDA)",
    ),
) -> None:
    """
    Ingest simulation logs and export to CSV, Parquet, and SQLite.

    Reads all JSON log files from INPUT_DIR, processes them, and writes
    summary tables to the output directory.
    """
    typer.echo(f"Ingesting logs from: {input_dir}")

    # Check for existing CSV for incremental readability
    existing_csv = output_dir / "runs.csv"
    has_existing = existing_csv.exists()

    if verbose:
        typer.echo(f"  Readability computation: {'enabled' if compute_readability else 'disabled'}")
        if compute_readability:
            typer.echo(f"  Device: {'GPU' if use_gpu else 'CPU'}")
            if has_existing:
                typer.echo(f"  Incremental mode: reusing existing readability from {existing_csv}")

    ingester = LogIngester(
        compute_readability=compute_readability,
        use_gpu=use_gpu,
        existing_csv_path=existing_csv if compute_readability else None,
    )
    records, criteria = ingester.ingest_directory(input_dir)

    if not records:
        typer.echo("No valid log files found.", err=True)
        if ingester.errors:
            typer.echo(f"Encountered {len(ingester.errors)} errors:", err=True)
            for path, error in ingester.errors[:5]:
                typer.echo(f"  {path}: {error}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Processed {len(records)} log files")

    if ingester.skipped_untestable:
        unique_untestable = set(ingester.skipped_untestable)
        typer.echo(f"Skipped {len(ingester.skipped_untestable)} runs from {len(unique_untestable)} untestable case(s)")
        if verbose:
            for case_id in unique_untestable:
                typer.echo(f"  - {case_id}")

    if verbose and ingester.errors:
        typer.echo(f"Skipped {len(ingester.errors)} files with errors")

    # Export
    outputs = export_all(records, output_dir, criteria)

    typer.echo(f"\nExported to:")
    for fmt, path in outputs.items():
        typer.echo(f"  {fmt.upper()}: {path}")

    # Show QA summary
    metrics = ingester.compute_qa_metrics(records)
    typer.echo(f"\nQA Summary:")
    typer.echo(f"  Total runs: {metrics.total_runs}")
    typer.echo(f"  Missing jurisdiction: {metrics.missing_jurisdiction_count} ({metrics.missing_jurisdiction_pct:.1f}%)")
    typer.echo(f"  Unknown specialty: {metrics.unknown_specialty_count} ({metrics.unknown_specialty_pct:.1f}%)")


@app.command()
def summarize(
    input_dir: Path = typer.Option(
        Path("exports"),
        "--input",
        "-i",
        help="Directory containing exported files (or raw logs)",
    ),
    group_by: str = typer.Option(
        "llm_name",
        "--group-by",
        "-g",
        help="Field to group by: llm_name, jurisdiction, specialty, malpractice_type",
    ),
    raw_logs: bool = typer.Option(
        False,
        "--raw",
        "-r",
        help="Input is raw log directory (not exported files)",
    ),
) -> None:
    """
    Summarize liability analysis results.

    Shows mean score, liability rate, and counts grouped by the specified field.
    """
    # Load data
    if raw_logs:
        ingester = LogIngester()
        records, _ = ingester.ingest_directory(input_dir)
    else:
        # Load from CSV
        csv_path = input_dir / "runs.csv"
        if not csv_path.exists():
            typer.echo(f"No runs.csv found in {input_dir}", err=True)
            typer.echo("Use --raw flag to process raw log files", err=True)
            raise typer.Exit(1)

        import csv
        from .models import LiabilityCode, RunRecord

        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Reconstruct minimal record for summary
                records.append(RunRecord(
                    run_id=row["run_id"],
                    case_id=row["case_id"],
                    jurisdiction=row.get("jurisdiction"),
                    specialty=row.get("specialty", "unknown"),
                    malpractice_type=MalpracticeType(row.get("malpractice_type", "other")),
                    liability_code=LiabilityCode(int(row.get("liability_code", 1))),
                    llm_name=row.get("llm_name", "unknown"),
                    score_0_2=int(row.get("score_0_2", 1)),
                    risk_flag=row.get("risk_flag", "").lower() == "true",
                ))

    if not records:
        typer.echo("No records found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\n{'=' * 60}")
    typer.echo(f"LIABILITY SUMMARY (grouped by {group_by})")
    typer.echo(f"{'=' * 60}\n")

    # Group records
    groups: dict[str, list] = {}
    for record in records:
        key = getattr(record, group_by, "unknown")
        if key is None:
            key = "unknown"
        if isinstance(key, MalpracticeType):
            key = key.value
        groups.setdefault(key, []).append(record)

    # Print summary for each group
    typer.echo(f"{'Group':<30} {'Count':>8} {'Mean Score':>12} {'Liability %':>12}")
    typer.echo("-" * 65)

    total_count = 0
    total_score = 0
    total_liability = 0

    for key in sorted(groups.keys()):
        group_records = groups[key]
        count = len(group_records)
        mean_score = sum(r.score_0_2 for r in group_records) / count
        liability_count = sum(1 for r in group_records if r.liability_code.value == 2)
        liability_pct = (liability_count / count) * 100

        typer.echo(f"{key:<30} {count:>8} {mean_score:>12.2f} {liability_pct:>11.1f}%")

        total_count += count
        total_score += sum(r.score_0_2 for r in group_records)
        total_liability += liability_count

    typer.echo("-" * 65)
    overall_mean = total_score / total_count if total_count > 0 else 0
    overall_liability = (total_liability / total_count * 100) if total_count > 0 else 0
    typer.echo(f"{'TOTAL':<30} {total_count:>8} {overall_mean:>12.2f} {overall_liability:>11.1f}%")

    # Malpractice type breakdown
    typer.echo(f"\n{'=' * 60}")
    typer.echo("MALPRACTICE TYPE DISTRIBUTION")
    typer.echo(f"{'=' * 60}\n")

    type_counts: dict[str, int] = {}
    for record in records:
        t = record.malpractice_type.value
        type_counts[t] = type_counts.get(t, 0) + 1

    for mtype, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        pct = (count / len(records)) * 100
        typer.echo(f"  {mtype:<35} {count:>6} ({pct:>5.1f}%)")


@app.command()
def validate(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing JSON log files",
        exists=True,
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "-s",
        help="Exit with error if any validation issues found",
    ),
) -> None:
    """
    Validate log files and report data quality issues.

    Reports missing fields, unknown specialties, unknown malpractice types,
    and other data quality concerns.
    """
    typer.echo(f"Validating logs in: {input_dir}")

    ingester = LogIngester()
    records, _ = ingester.ingest_directory(input_dir)

    # Report file errors
    if ingester.errors:
        typer.echo(f"\n{'=' * 60}")
        typer.echo("FILE ERRORS")
        typer.echo(f"{'=' * 60}")
        for path, error in ingester.errors:
            typer.echo(f"  {Path(path).name}: {error[:80]}")

    if not records:
        typer.echo("\nNo valid records to analyze.", err=True)
        raise typer.Exit(1)

    # Compute and display QA metrics
    metrics = ingester.compute_qa_metrics(records)
    typer.echo(metrics.to_report())

    # Detailed validation issues
    issues: list[str] = []

    # Check for unknown specialties
    unknown_specialty = [r for r in records if r.specialty == "unknown"]
    if unknown_specialty:
        issues.append(f"Unknown specialty: {len(unknown_specialty)} runs")
        typer.echo(f"\nRuns with unknown specialty (showing first 5):")
        for r in unknown_specialty[:5]:
            typer.echo(f"  {r.run_id}: {r.case_id}")

    # Check for unknown malpractice type
    unknown_malpractice = [r for r in records if r.malpractice_type == MalpracticeType.OTHER]
    if unknown_malpractice:
        issues.append(f"Unknown malpractice type: {len(unknown_malpractice)} runs")
        typer.echo(f"\nRuns with unknown malpractice type (showing first 5):")
        for r in unknown_malpractice[:5]:
            typer.echo(f"  {r.run_id}: {r.case_id}")

    # Check for missing jurisdiction
    missing_jurisdiction = [r for r in records if not r.jurisdiction]
    if missing_jurisdiction:
        issues.append(f"Missing jurisdiction: {len(missing_jurisdiction)} runs")

    # Summary
    typer.echo(f"\n{'=' * 60}")
    if issues:
        typer.echo(f"VALIDATION ISSUES FOUND: {len(issues)}")
        for issue in issues:
            typer.echo(f"  - {issue}")
        if strict:
            raise typer.Exit(1)
    else:
        typer.echo("VALIDATION PASSED: No issues found")
    typer.echo(f"{'=' * 60}")


@app.command()
def info() -> None:
    """Show configuration and rule information."""
    from .classifier import get_classifier

    classifier = get_classifier()

    typer.echo(f"\n{'=' * 60}")
    typer.echo("LIABILITY ANALYSIS CONFIGURATION")
    typer.echo(f"{'=' * 60}\n")

    typer.echo(f"Specialty Rules: {len(classifier.specialty_rules)} patterns")
    for rule in classifier.specialty_rules[:5]:
        typer.echo(f"  - {rule['specialty']}")
    if len(classifier.specialty_rules) > 5:
        typer.echo(f"  ... and {len(classifier.specialty_rules) - 5} more")

    typer.echo(f"\nMalpractice Type Rules: {len(classifier.malpractice_rules)} patterns")
    for rule in classifier.malpractice_rules:
        typer.echo(f"  - {rule['type']}")

    typer.echo(f"\nMalpractice Priority Order:")
    for i, t in enumerate(classifier.malpractice_priority, 1):
        typer.echo(f"  {i}. {t}")


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
