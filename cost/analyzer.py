"""
Main orchestrator for cost analysis.

Coordinates procedure extraction, CPT matching, and pricing lookup.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from cost.models import (
    CostAnalysisResult,
    ExtractedProcedure,
    ProcedureCost,
    MatchConfidence,
)
from cost.rag import CPTVectorStore
from cost.pricing import CPTPricingDatabase
from cost.extractor import ProcedureExtractor, MockProcedureExtractor

if TYPE_CHECKING:
    from cost.models import CPTMatch


class CostAnalyzer:
    """
    Orchestrates procedure extraction, CPT matching, and pricing.

    Usage:
        analyzer = CostAnalyzer()
        result = analyzer.analyze_recommendation(
            recommendation="Order chest X-ray and refer to cardiologist",
            run_id="abc123",
            case_id="case-001",
            llm_name="gpt-4o",
        )
        print(f"Total cost: ${result.total_cost_negotiated:.2f}")
    """

    def __init__(
        self,
        rag: CPTVectorStore | None = None,
        pricing: CPTPricingDatabase | None = None,
        extractor: ProcedureExtractor | MockProcedureExtractor | None = None,
        similarity_threshold: float = 0.4,
        use_mock_extractor: bool = False,
    ):
        """
        Initialize cost analyzer.

        Args:
            rag: CPT vector store (created with defaults if None)
            pricing: Pricing database (created with defaults if None)
            extractor: Procedure extractor (created with defaults if None)
            similarity_threshold: Minimum similarity for CPT matching
            use_mock_extractor: Use keyword-based extractor instead of LLM
        """
        self.rag = rag or CPTVectorStore()
        self.pricing = pricing or CPTPricingDatabase()

        if extractor is not None:
            self.extractor = extractor
        elif use_mock_extractor:
            self.extractor = MockProcedureExtractor()
        else:
            self.extractor = ProcedureExtractor()

        self.similarity_threshold = similarity_threshold

    def analyze_recommendation(
        self,
        recommendation: str,
        run_id: str,
        case_id: str,
        llm_name: str,
    ) -> CostAnalysisResult:
        """
        Analyze a single recommendation for procedure costs.

        Args:
            recommendation: The LLM's final recommendation text
            run_id: Session/run identifier
            case_id: Case identifier
            llm_name: Name of the LLM that generated the recommendation

        Returns:
            CostAnalysisResult with extracted procedures and costs
        """
        result = CostAnalysisResult(
            run_id=run_id,
            case_id=case_id,
            llm_name=llm_name,
        )

        if not recommendation or not recommendation.strip():
            result.compute_totals()
            return result

        # Step 1: Extract procedures from recommendation
        procedures = self.extractor.extract(recommendation)
        result.procedures = procedures

        if not procedures:
            result.compute_totals()
            return result

        # Step 2: Match procedures to CPT codes
        procedure_names = [p.procedure_name for p in procedures]
        matches_lists = self.rag.batch_search(
            procedure_names,
            top_k=1,  # Take best match only
            threshold=self.similarity_threshold,
        )

        # Step 3: Look up prices for matches
        matched_costs = []
        unmatched = []

        for procedure, matches in zip(procedures, matches_lists):
            if not matches:
                unmatched.append(procedure.procedure_name)
                continue

            # Take best match
            best_match = matches[0]

            # Get pricing
            price_info = self.pricing.get_price(best_match.cpt_code)

            if price_info:
                cost = ProcedureCost.from_match_and_price(
                    procedure_name=procedure.procedure_name,
                    match=best_match,
                    negotiated_dollar=price_info.negotiated_dollar,
                    min_charge=price_info.min_charge,
                    max_charge=price_info.max_charge,
                )
            else:
                # Matched CPT code but no pricing
                cost = ProcedureCost(
                    procedure_name=procedure.procedure_name,
                    cpt_code=best_match.cpt_code,
                    matched_description=best_match.clear_description,
                    similarity_score=best_match.similarity_score,
                    match_confidence=MatchConfidence.from_score(best_match.similarity_score),
                    negotiated_dollar=None,
                    min_charge=None,
                    max_charge=None,
                )

            matched_costs.append(cost)

        result.matched_costs = matched_costs
        result.unmatched_procedures = unmatched
        result.compute_totals()

        return result

    def analyze_log(self, log_path: Path | str) -> CostAnalysisResult | None:
        """
        Analyze a single log file.

        Args:
            log_path: Path to the JSON log file

        Returns:
            CostAnalysisResult or None if log is invalid
        """
        log_path = Path(log_path)

        try:
            with open(log_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Failed to load {log_path}: {e}")
            return None

        recommendation = data.get("final_recommendation", "")
        if not recommendation:
            return None

        return self.analyze_recommendation(
            recommendation=recommendation,
            run_id=data.get("session_id", log_path.stem),
            case_id=data.get("case_id", "unknown"),
            llm_name=data.get("llm_name", "unknown"),
        )

    def analyze_logs_directory(
        self,
        logs_dir: Path | str,
        show_progress: bool = True,
    ) -> list[CostAnalysisResult]:
        """
        Analyze all log files in a directory.

        Args:
            logs_dir: Directory containing JSON log files
            show_progress: Show progress bar

        Returns:
            List of CostAnalysisResult objects
        """
        logs_dir = Path(logs_dir)

        if not logs_dir.exists():
            raise FileNotFoundError(f"Logs directory not found: {logs_dir}")

        log_files = sorted(logs_dir.glob("*.json"))

        if not log_files:
            print(f"No JSON files found in {logs_dir}")
            return []

        print(f"Found {len(log_files)} log files")

        # Ensure index is built before batch processing
        self.rag.build_index()

        results = []

        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(log_files, desc="Analyzing logs")
            except ImportError:
                iterator = log_files
        else:
            iterator = log_files

        for log_path in iterator:
            result = self.analyze_log(log_path)
            if result is not None:
                results.append(result)

        return results

    def export_results(
        self,
        results: list[CostAnalysisResult],
        output_path: Path | str,
        format: str = "csv",
    ) -> None:
        """
        Export results to file.

        Args:
            results: List of CostAnalysisResult objects
            output_path: Output file path
            format: "csv", "json", or "parquet"
        """
        import pandas as pd

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            # Full export with all details
            data = [r.model_dump() for r in results]
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        else:
            # Tabular export (summary only)
            rows = [r.to_summary_dict() for r in results]
            df = pd.DataFrame(rows)

            if format == "csv":
                df.to_csv(output_path, index=False)
            elif format == "parquet":
                df.to_parquet(output_path, index=False)
            else:
                raise ValueError(f"Unknown format: {format}")

        print(f"Exported {len(results)} results to {output_path}")

    def export_detailed_costs(
        self,
        results: list[CostAnalysisResult],
        output_path: Path | str,
    ) -> None:
        """
        Export detailed per-procedure costs (long format).

        Args:
            results: List of CostAnalysisResult objects
            output_path: Output CSV path
        """
        import pandas as pd

        rows = []
        for result in results:
            for cost in result.matched_costs:
                rows.append({
                    "run_id": result.run_id,
                    "case_id": result.case_id,
                    "llm_name": result.llm_name,
                    "procedure_name": cost.procedure_name,
                    "cpt_code": cost.cpt_code,
                    "matched_description": cost.matched_description,
                    "similarity_score": cost.similarity_score,
                    "match_confidence": cost.match_confidence.value,
                    "negotiated_dollar": cost.negotiated_dollar,
                    "min_charge": cost.min_charge,
                    "max_charge": cost.max_charge,
                })

        df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Exported {len(rows)} procedure costs to {output_path}")
