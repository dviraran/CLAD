"""
Cost analysis module for CLAD.

Extracts medical procedures from LLM recommendations and estimates costs
using CPT code matching via RAG-based semantic search.

Usage:
    from cost import CostAnalyzer

    analyzer = CostAnalyzer()
    result = analyzer.analyze_recommendation(
        recommendation="Order chest X-ray and refer to cardiologist",
        run_id="session-123",
        case_id="case-001",
        llm_name="gpt-4o",
    )
    print(f"Total cost: ${result.total_cost_negotiated:.2f}")

CLI:
    python -m cost.cli index      # Build CPT embedding index
    python -m cost.cli analyze gui/logs --out cost/exports
    python -m cost.cli search "chest x-ray"
"""

# Lazy imports to avoid loading heavy dependencies on import
def __getattr__(name: str):
    if name == "ExtractedProcedure":
        from cost.models import ExtractedProcedure
        return ExtractedProcedure
    elif name == "CPTMatch":
        from cost.models import CPTMatch
        return CPTMatch
    elif name == "ProcedureCost":
        from cost.models import ProcedureCost
        return ProcedureCost
    elif name == "CostAnalysisResult":
        from cost.models import CostAnalysisResult
        return CostAnalysisResult
    elif name == "CPTVectorStore":
        from cost.rag import CPTVectorStore
        return CPTVectorStore
    elif name == "CPTPricingDatabase":
        from cost.pricing import CPTPricingDatabase
        return CPTPricingDatabase
    elif name == "ProcedureExtractor":
        from cost.extractor import ProcedureExtractor
        return ProcedureExtractor
    elif name == "CostAnalyzer":
        from cost.analyzer import CostAnalyzer
        return CostAnalyzer
    raise AttributeError(f"module 'cost' has no attribute '{name}'")


__all__ = [
    "ExtractedProcedure",
    "CPTMatch",
    "ProcedureCost",
    "CostAnalysisResult",
    "CPTVectorStore",
    "CPTPricingDatabase",
    "ProcedureExtractor",
    "CostAnalyzer",
]
