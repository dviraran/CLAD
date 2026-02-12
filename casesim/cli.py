"""Command-line interface for casesim."""

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .config import get_settings, reload_settings
from .utils import setup_logging

app = typer.Typer(
    name="cases",
    help="Medical malpractice case simulation system",
    add_completion=False,
)

console = Console()


def version_callback(value: bool) -> None:
    if value:
        console.print("casesim version 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit",
    ),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
) -> None:
    """Medical malpractice case simulation system."""
    setup_logging(debug=debug)


# ============================================================================
# Discovery Commands
# ============================================================================


@app.command()
def discover(
    source: str = typer.Argument(..., help="Source database (bailii, canlii, austlii, courtlistener)"),
    keywords: Optional[Path] = typer.Option(None, "--keywords", "-k", help="Keywords config YAML"),
    seeds: Optional[Path] = typer.Option(None, "--seeds", "-s", help="Seed cases YAML or directory"),
    expand_citations: bool = typer.Option(False, "--expand-citations", "-e", help="Expand through citations"),
    max_results: int = typer.Option(100, "--max", "-m", help="Maximum cases to discover"),
    min_length: int = typer.Option(10000, "--min-length", help="Minimum document length"),
) -> None:
    """Discover medical malpractice cases from legal databases."""
    from .discovery import run_discovery

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Discovering cases...", total=None)

        stats = asyncio.run(run_discovery(
            source=source,
            keywords_path=keywords,
            seeds_path=seeds,
            expand_citations=expand_citations,
            max_results=max_results,
            min_length=min_length,
        ))

        progress.update(task, completed=True)

    # Display results
    table = Table(title="Discovery Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Cases Discovered", str(stats.cases_discovered))
    table.add_row("Cases Added", str(stats.cases_added))
    table.add_row("Cases Rejected", str(stats.cases_rejected))
    table.add_row("Duplicates", str(stats.cases_duplicate))

    if stats.rejections_by_reason:
        for reason, count in stats.rejections_by_reason.items():
            table.add_row(f"  - {reason}", str(count))

    console.print(table)


@app.command()
def queue(
    limit: int = typer.Option(20, "--limit", "-l", help="Number of cases to show"),
) -> None:
    """Show discovery queue status."""
    from .utils import CaseDatabase
    from .schemas import DiscoveryStatus

    settings = get_settings()
    db = CaseDatabase(settings.paths.database_path)

    stats = db.get_queue_stats()

    table = Table(title="Queue Status")
    table.add_column("Status", style="cyan")
    table.add_column("Count", style="green")

    total = 0
    for status, count in sorted(stats.items()):
        table.add_row(status, str(count))
        total += count

    table.add_row("TOTAL", str(total), style="bold")
    console.print(table)

    # Show next cases to process
    queued = db.get_cases_by_status(DiscoveryStatus.QUEUED, limit=limit)
    if queued:
        console.print("\n[bold]Next cases to fetch:[/bold]")
        for case in queued[:5]:
            console.print(f"  - {case.case_id}: {case.title[:50]}...")


# ============================================================================
# Fetch Commands
# ============================================================================


@app.command()
def fetch(
    case_id: Optional[str] = typer.Argument(None, help="Specific case ID to fetch"),
    all_queued: bool = typer.Option(False, "--all", "-a", help="Fetch all queued cases"),
    limit: int = typer.Option(10, "--limit", "-l", help="Max cases to fetch"),
) -> None:
    """Fetch case documents from sources."""
    from .utils import CaseDatabase
    from .schemas import DiscoveryStatus, RejectionReason
    from .sources import get_source

    settings = get_settings()
    db = CaseDatabase(settings.paths.database_path)

    if case_id:
        cases = [db.get_case(case_id)]
        cases = [c for c in cases if c]
    elif all_queued:
        cases = db.get_cases_by_status(DiscoveryStatus.QUEUED, limit=limit)
    else:
        console.print("[red]Specify --case-id or --all[/red]")
        raise typer.Exit(1)

    if not cases:
        console.print("[yellow]No cases to fetch[/yellow]")
        return

    async def fetch_cases():
        fetched = 0
        failed = 0

        for case in cases:
            source_class = get_source(case.source.value)
            async with source_class() as source:
                try:
                    console.print(f"Fetching {case.case_id}...")
                    result = await source.fetch(str(case.url))

                    raw_path = await source.save_raw(
                        str(case.url),
                        result.content,
                        case.case_id,
                    )

                    db.update_fetched(
                        case.case_id,
                        str(raw_path),
                        result.content_hash,
                        len(result.content),
                    )

                    fetched += 1
                    console.print(f"  [green]✓ Saved to {raw_path}[/green]")

                except Exception as e:
                    failed += 1
                    db.update_status(
                        case.case_id,
                        DiscoveryStatus.REJECTED,
                        RejectionReason.FETCH_ERROR,
                    )
                    console.print(f"  [red]✗ Error: {e}[/red]")

        return fetched, failed

    fetched, failed = asyncio.run(fetch_cases())
    console.print(f"\n[bold]Fetched: {fetched}, Failed: {failed}[/bold]")


# ============================================================================
# Extract Commands
# ============================================================================


@app.command()
def extract(
    case_id: Optional[str] = typer.Argument(None, help="Case ID to extract"),
    all_fetched: bool = typer.Option(False, "--all", "-a", help="Extract all fetched cases"),
    limit: int = typer.Option(5, "--limit", "-l", help="Max cases to extract"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
) -> None:
    """Extract case simulations using LLM."""
    from .utils import CaseDatabase
    from .schemas import DiscoveryStatus
    from .parsing import JudgmentParser
    from .extraction import LLMExtractor

    settings = get_settings()
    db = CaseDatabase(settings.paths.database_path)
    output_dir = output_dir or settings.paths.processed_dir

    if case_id:
        cases = [db.get_case(case_id)]
        cases = [c for c in cases if c]
    elif all_fetched:
        cases = db.get_cases_by_status(DiscoveryStatus.FETCHED, limit=limit)
    else:
        console.print("[red]Specify case_id or --all[/red]")
        raise typer.Exit(1)

    if not cases:
        console.print("[yellow]No cases to extract[/yellow]")
        return

    parser = JudgmentParser()
    extractor = LLMExtractor()

    extracted = 0
    failed = 0

    for case in cases:
        if not case.raw_file_path:
            console.print(f"[yellow]Skipping {case.case_id}: no raw file[/yellow]")
            continue

        raw_path = Path(case.raw_file_path)
        if not raw_path.exists():
            console.print(f"[yellow]Skipping {case.case_id}: raw file not found[/yellow]")
            continue

        try:
            console.print(f"Extracting {case.case_id}...")

            # Parse
            content = raw_path.read_text(encoding="utf-8", errors="ignore")
            parsed = parser.parse_html(content)

            # Extract
            simulation = extractor.extract(parsed, case.case_id, str(case.url))

            # Save
            output_path = output_dir / f"{case.case_id}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(simulation.model_dump(mode="json"), f, indent=2, default=str)

            # Update database
            quality_score = simulation.quality.evidence_coverage_score if simulation.quality else None
            db.record_extraction(
                case.case_id,
                extractor.VERSION,
                extractor.model,
                3,
                str(output_path),
                quality_score,
                True,
            )

            extracted += 1
            console.print(f"  [green]✓ Saved to {output_path}[/green]")

        except Exception as e:
            failed += 1
            console.print(f"  [red]✗ Error: {e}[/red]")

    console.print(f"\n[bold]Extracted: {extracted}, Failed: {failed}[/bold]")


# ============================================================================
# Validate Commands
# ============================================================================


@app.command()
def validate(
    input_path: Path = typer.Argument(..., help="Case JSON file or directory"),
    strict: bool = typer.Option(True, "--strict/--no-strict", help="Strict validation"),
    report: Optional[Path] = typer.Option(None, "--report", "-r", help="Output report file"),
) -> None:
    """Validate case simulation files."""
    from .qa import CaseValidator

    validator = CaseValidator()

    if input_path.is_file():
        files = [input_path]
    else:
        files = list(input_path.glob("*.json"))

    if not files:
        console.print("[yellow]No JSON files found[/yellow]")
        return

    results = {}

    for file_path in files:
        try:
            with open(file_path) as f:
                case = json.load(f)

            result = validator.validate(case, strict=strict)
            case_id = case.get("case_id", file_path.stem)
            results[case_id] = result

            status = "[green]✓[/green]" if result.valid else "[red]✗[/red]"
            console.print(f"{status} {case_id}: {len(result.errors)} errors, {len(result.warnings)} warnings")

            for issue in result.errors[:5]:
                console.print(f"    [red]ERROR: {issue.message}[/red]")

        except Exception as e:
            console.print(f"[red]Error loading {file_path}: {e}[/red]")

    # Generate report
    if report:
        report_content = validator.generate_report(results)
        report.write_text(report_content)
        console.print(f"\nReport saved to {report}")


# ============================================================================
# Export Commands
# ============================================================================


@app.command()
def export(
    input_dir: Path = typer.Argument(..., help="Directory with processed cases"),
    output_dir: Path = typer.Option(None, "--output", "-o", help="Output directory"),
    format: str = typer.Option("json", "--format", "-f", help="Export format (json, jsonl)"),
) -> None:
    """Export processed cases."""
    settings = get_settings()
    output_dir = output_dir or settings.paths.exports_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.json"))

    if not files:
        console.print("[yellow]No JSON files found[/yellow]")
        return

    if format == "jsonl":
        output_path = output_dir / f"cases_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(output_path, "w") as out:
            for file_path in files:
                with open(file_path) as f:
                    case = json.load(f)
                out.write(json.dumps(case) + "\n")
        console.print(f"Exported {len(files)} cases to {output_path}")

    else:
        output_path = output_dir / f"cases_{datetime.now().strftime('%Y%m%d')}.json"
        cases = []
        for file_path in files:
            with open(file_path) as f:
                cases.append(json.load(f))

        with open(output_path, "w") as f:
            json.dump({"cases": cases, "count": len(cases)}, f, indent=2)

        console.print(f"Exported {len(files)} cases to {output_path}")


# ============================================================================
# Simulation Commands
# ============================================================================


@app.command()
def simulate(
    case_file: Path = typer.Argument(..., help="Case simulation JSON file"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Interactive mode"),
) -> None:
    """Run an interactive case simulation."""
    from .sim import SimulationRunner

    with open(case_file) as f:
        case = json.load(f)

    runner = SimulationRunner(case)

    if interactive:
        runner.run_interactive()
    else:
        runner.run_automated()


# ============================================================================
# Pipeline Commands
# ============================================================================


@app.command()
def pipeline(
    source: str = typer.Argument(..., help="Source database"),
    max_cases: int = typer.Option(5, "--max", "-m", help="Maximum cases"),
    skip_discovery: bool = typer.Option(False, "--skip-discovery", help="Skip discovery"),
    skip_fetch: bool = typer.Option(False, "--skip-fetch", help="Skip fetching"),
    skip_extract: bool = typer.Option(False, "--skip-extract", help="Skip extraction"),
) -> None:
    """Run the full pipeline: discover → fetch → extract → validate."""
    from .discovery import run_discovery
    from .utils import CaseDatabase
    from .schemas import DiscoveryStatus
    from .sources import get_source
    from .parsing import JudgmentParser
    from .extraction import LLMExtractor
    from .qa import CaseValidator

    settings = get_settings()
    db = CaseDatabase(settings.paths.database_path)

    console.print("[bold]Starting pipeline...[/bold]\n")

    # Step 1: Discovery
    if not skip_discovery:
        console.print("[cyan]Step 1: Discovery[/cyan]")
        stats = asyncio.run(run_discovery(source=source, max_results=max_cases))
        console.print(f"  Discovered: {stats.cases_added} cases\n")

    # Step 2: Fetch
    if not skip_fetch:
        console.print("[cyan]Step 2: Fetch[/cyan]")
        cases = db.get_cases_by_status(DiscoveryStatus.QUEUED, limit=max_cases)

        async def do_fetch():
            source_class = get_source(source)
            async with source_class() as src:
                for case in cases:
                    try:
                        result = await src.fetch(str(case.url))
                        raw_path = await src.save_raw(str(case.url), result.content, case.case_id)
                        db.update_fetched(case.case_id, str(raw_path), result.content_hash, len(result.content))
                        console.print(f"  ✓ Fetched {case.case_id}")
                    except Exception as e:
                        console.print(f"  ✗ Failed {case.case_id}: {e}")

        asyncio.run(do_fetch())
        console.print()

    # Step 3: Extract
    if not skip_extract:
        console.print("[cyan]Step 3: Extract[/cyan]")
        cases = db.get_cases_by_status(DiscoveryStatus.FETCHED, limit=max_cases)
        parser = JudgmentParser()
        extractor = LLMExtractor()

        for case in cases:
            if not case.raw_file_path:
                continue

            raw_path = Path(case.raw_file_path)
            if not raw_path.exists():
                continue

            try:
                content = raw_path.read_text(encoding="utf-8", errors="ignore")
                parsed = parser.parse_html(content)
                simulation = extractor.extract(parsed, case.case_id, str(case.url))

                output_path = settings.paths.processed_dir / f"{case.case_id}.json"
                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, "w") as f:
                    json.dump(simulation.model_dump(mode="json"), f, indent=2, default=str)

                console.print(f"  ✓ Extracted {case.case_id}")

            except Exception as e:
                console.print(f"  ✗ Failed {case.case_id}: {e}")

        console.print()

    # Step 4: Validate
    console.print("[cyan]Step 4: Validate[/cyan]")
    validator = CaseValidator()
    files = list(settings.paths.processed_dir.glob("*.json"))

    valid_count = 0
    for file_path in files:
        try:
            with open(file_path) as f:
                case = json.load(f)
            result = validator.validate(case)
            if result.valid:
                valid_count += 1
                console.print(f"  ✓ {file_path.stem}")
            else:
                console.print(f"  ✗ {file_path.stem}: {len(result.errors)} errors")
        except Exception as e:
            console.print(f"  ✗ {file_path.stem}: {e}")

    console.print(f"\n[bold green]Pipeline complete: {valid_count}/{len(files)} valid cases[/bold green]")


# ============================================================================
# Info Commands
# ============================================================================


@app.command()
def info() -> None:
    """Show system information and configuration."""
    settings = get_settings()

    table = Table(title="CaseSim Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Version", "0.1.0")
    table.add_row("OpenAI Model", settings.openai.model)
    table.add_row("Data Directory", str(settings.paths.data_dir))
    table.add_row("Database", str(settings.paths.database_path))
    table.add_row("Cache Enabled", str(settings.cache.enabled))
    table.add_row("Rate Limit", f"{settings.rate_limit.requests_per_second}/sec")

    console.print(table)


if __name__ == "__main__":
    app()
