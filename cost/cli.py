"""
CLI commands for cost analysis module.

Usage:
    python -m cost.cli index          # Build CPT embedding index
    python -m cost.cli analyze LOGS   # Analyze logs directory
    python -m cost.cli search QUERY   # Search CPT codes
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="CPT-based cost analysis for CLAD")


@app.command()
def index(
    descriptions: Path = typer.Option(
        Path("data/cpt/clarified_descriptions.csv"),
        "--descriptions", "-d",
        help="Path to CPT descriptions CSV",
    ),
    output: Path = typer.Option(
        Path("data/cpt/embeddings/cpt_embeddings.npz"),
        "--output", "-o",
        help="Path for cached embeddings",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force rebuild even if cache exists",
    ),
    gpu: bool = typer.Option(
        False,
        "--gpu",
        help="Use GPU for embedding computation",
    ),
):
    """Build or rebuild the CPT embedding index."""
    from cost.rag import CPTVectorStore

    print(f"Building CPT index from {descriptions}")

    store = CPTVectorStore(
        descriptions_path=descriptions,
        embeddings_path=output,
        use_gpu=gpu,
    )

    store.build_index(force_rebuild=force)
    print(f"Index complete: {store.num_codes} CPT codes")


@app.command()
def search(
    query: str = typer.Argument(..., help="Procedure description to search"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    threshold: float = typer.Option(0.3, "--threshold", "-t", help="Minimum similarity"),
    show_prices: bool = typer.Option(True, "--prices/--no-prices", help="Show pricing"),
):
    """Search for CPT codes matching a procedure description."""
    from cost.rag import CPTVectorStore
    from cost.pricing import CPTPricingDatabase

    store = CPTVectorStore()
    store.build_index()

    matches = store.search(query, top_k=top_k, threshold=threshold)

    if not matches:
        print(f"No matches found for: {query}")
        return

    pricing = CPTPricingDatabase() if show_prices else None

    print(f"\nResults for: {query}\n")
    print("-" * 80)

    for i, match in enumerate(matches, 1):
        print(f"{i}. [{match.cpt_code}] {match.clear_description}")
        print(f"   Short: {match.description}")
        print(f"   Similarity: {match.similarity_score:.3f}")

        if pricing:
            price = pricing.get_price(match.cpt_code)
            if price and price.negotiated_dollar:
                print(f"   Price: ${price.negotiated_dollar:.2f} ({price.plan_name})")
            else:
                print("   Price: Not available")
        print()


@app.command()
def analyze(
    logs_dir: Path = typer.Argument(..., help="Directory containing log files"),
    output: Path = typer.Option(
        Path("cost/exports"),
        "--output", "-o",
        help="Output directory for results",
    ),
    format: str = typer.Option(
        "csv",
        "--format", "-f",
        help="Output format: csv, json, parquet",
    ),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock extractor (no API calls)",
    ),
    threshold: float = typer.Option(
        0.4,
        "--threshold", "-t",
        help="Minimum similarity for CPT matching",
    ),
    detailed: bool = typer.Option(
        True,
        "--detailed/--no-detailed",
        help="Also export detailed per-procedure costs",
    ),
):
    """Analyze log files for procedure costs."""
    from cost.analyzer import CostAnalyzer

    print(f"Analyzing logs in {logs_dir}")

    analyzer = CostAnalyzer(
        similarity_threshold=threshold,
        use_mock_extractor=mock,
    )

    results = analyzer.analyze_logs_directory(logs_dir)

    if not results:
        print("No results to export")
        return

    # Summary statistics
    total_recs = len(results)
    with_procedures = sum(1 for r in results if r.procedure_count > 0)
    total_procedures = sum(r.procedure_count for r in results)
    total_matched = sum(r.matched_count for r in results)
    costs = [r.total_cost_negotiated for r in results if r.total_cost_negotiated]

    print(f"\n=== Summary ===")
    print(f"Recommendations analyzed: {total_recs}")
    print(f"With procedures: {with_procedures} ({100*with_procedures/total_recs:.1f}%)")
    print(f"Total procedures extracted: {total_procedures}")
    print(f"Total matched to CPT: {total_matched} ({100*total_matched/total_procedures:.1f}%)" if total_procedures else "")

    if costs:
        import numpy as np
        print(f"\n=== Cost Statistics ===")
        print(f"Recommendations with costs: {len(costs)}")
        print(f"Mean cost: ${np.mean(costs):.2f}")
        print(f"Median cost: ${np.median(costs):.2f}")
        print(f"Min: ${min(costs):.2f}, Max: ${max(costs):.2f}")

    # Export
    output.mkdir(parents=True, exist_ok=True)

    summary_path = output / f"cost_summary.{format}"
    analyzer.export_results(results, summary_path, format=format)

    if detailed:
        detailed_path = output / "cost_details.csv"
        analyzer.export_detailed_costs(results, detailed_path)

    # Also export full JSON for debugging
    json_path = output / "cost_results.json"
    analyzer.export_results(results, json_path, format="json")


@app.command()
def stats():
    """Show statistics about the CPT and pricing data."""
    from cost.rag import CPTVectorStore
    from cost.pricing import CPTPricingDatabase

    print("=== CPT Descriptions ===")
    store = CPTVectorStore()
    store._load_cpt_data()
    print(f"Total CPT codes: {store.num_codes}")

    print("\n=== Pricing Data ===")
    pricing = CPTPricingDatabase()
    stats = pricing.get_stats()
    print(f"CPT codes with pricing: {stats['num_codes']}")
    print(f"Total price entries: {stats['num_prices']}")
    print(f"Plans: {', '.join(stats['plans'])}")

    if 'price_median' in stats:
        print(f"\nPrice range: ${stats['price_min']:.2f} - ${stats['price_max']:.2f}")
        print(f"Median: ${stats['price_median']:.2f}")
        print(f"Mean: ${stats['price_mean']:.2f}")


@app.command()
def test_extraction(
    text: str = typer.Argument(..., help="Recommendation text to test"),
    mock: bool = typer.Option(
        False,
        "--mock",
        help="Use mock extractor",
    ),
):
    """Test procedure extraction on a text sample."""
    from cost.extractor import ProcedureExtractor, MockProcedureExtractor

    if mock:
        extractor = MockProcedureExtractor()
        print("Using mock extractor")
    else:
        extractor = ProcedureExtractor()
        print("Using LLM extractor")

    print(f"\nInput: {text[:200]}...")
    print("\nExtracted procedures:")

    procedures = extractor.extract(text)

    if not procedures:
        print("  (none found)")
    else:
        for p in procedures:
            print(f"  - {p.procedure_name} [{p.procedure_type.value if p.procedure_type else 'unknown'}]")


def main():
    """Entry point."""
    app()


if __name__ == "__main__":
    main()
