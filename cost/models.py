"""
Pydantic data models for cost analysis.
"""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ProcedureType(str, Enum):
    """Type of medical procedure."""
    IMAGING = "imaging"
    LAB = "lab"
    PROCEDURE = "procedure"
    CONSULTATION = "consultation"
    OTHER = "other"


class MatchConfidence(str, Enum):
    """Confidence level for CPT code match."""
    HIGH = "high"        # similarity > 0.7
    MEDIUM = "medium"    # 0.5 < similarity <= 0.7
    LOW = "low"          # 0.4 < similarity <= 0.5

    @classmethod
    def from_score(cls, score: float) -> MatchConfidence:
        """Determine confidence level from similarity score."""
        if score > 0.7:
            return cls.HIGH
        elif score > 0.5:
            return cls.MEDIUM
        else:
            return cls.LOW


class ExtractedProcedure(BaseModel):
    """A procedure extracted from LLM recommendation."""

    procedure_name: str = Field(
        description="Natural language procedure name (e.g., 'chest X-ray')"
    )
    procedure_type: ProcedureType | None = Field(
        default=None,
        description="Type of procedure: imaging, lab, procedure, consultation"
    )

    model_config = {"frozen": True}


class CPTMatch(BaseModel):
    """A CPT code matched to an extracted procedure."""

    cpt_code: str = Field(description="5-digit CPT code (e.g., '71046')")
    description: str = Field(description="Short/abbreviated description")
    clear_description: str = Field(description="Full clarified description")
    similarity_score: float = Field(
        ge=0.0, le=1.0,
        description="Cosine similarity score (0-1)"
    )

    model_config = {"frozen": True}


class ProcedureCost(BaseModel):
    """Cost information for a matched procedure."""

    # Procedure identification
    procedure_name: str = Field(description="Original extracted procedure name")
    cpt_code: str = Field(description="Matched CPT code")
    matched_description: str = Field(description="CPT description that was matched")

    # Match quality
    similarity_score: float = Field(
        ge=0.0, le=1.0,
        description="Similarity score from RAG matching"
    )
    match_confidence: MatchConfidence = Field(
        description="Confidence level: high, medium, low"
    )

    # Pricing (Managed Medicaid)
    negotiated_dollar: float | None = Field(
        default=None,
        description="Negotiated price in dollars"
    )
    min_charge: float | None = Field(
        default=None,
        description="Minimum charge"
    )
    max_charge: float | None = Field(
        default=None,
        description="Maximum charge"
    )

    model_config = {"frozen": True}

    @classmethod
    def from_match_and_price(
        cls,
        procedure_name: str,
        match: CPTMatch,
        negotiated_dollar: float | None,
        min_charge: float | None,
        max_charge: float | None,
    ) -> ProcedureCost:
        """Create ProcedureCost from match and price data."""
        return cls(
            procedure_name=procedure_name,
            cpt_code=match.cpt_code,
            matched_description=match.clear_description,
            similarity_score=match.similarity_score,
            match_confidence=MatchConfidence.from_score(match.similarity_score),
            negotiated_dollar=negotiated_dollar,
            min_charge=min_charge,
            max_charge=max_charge,
        )


class CostAnalysisResult(BaseModel):
    """Complete cost analysis for one recommendation."""

    # Identifiers
    run_id: str = Field(description="Session/run ID from log")
    case_id: str = Field(description="Case ID")
    llm_name: str = Field(description="LLM model name")

    # Extracted procedures
    procedures: list[ExtractedProcedure] = Field(
        default_factory=list,
        description="All procedures extracted from recommendation"
    )

    # Matched costs
    matched_costs: list[ProcedureCost] = Field(
        default_factory=list,
        description="Procedures successfully matched to CPT codes with pricing"
    )
    unmatched_procedures: list[str] = Field(
        default_factory=list,
        description="Procedure names that could not be matched"
    )

    # Aggregated metrics
    total_cost_negotiated: float | None = Field(
        default=None,
        description="Sum of negotiated prices for all matched procedures"
    )
    total_cost_min: float | None = Field(
        default=None,
        description="Sum of minimum charges"
    )
    total_cost_max: float | None = Field(
        default=None,
        description="Sum of maximum charges"
    )

    # Summary stats
    procedure_count: int = Field(
        default=0,
        description="Total number of procedures extracted"
    )
    matched_count: int = Field(
        default=0,
        description="Number of procedures successfully matched"
    )
    match_rate: float = Field(
        default=0.0,
        ge=0.0, le=1.0,
        description="Proportion of procedures matched (0-1)"
    )

    def compute_totals(self) -> None:
        """Compute aggregate totals from matched_costs."""
        self.procedure_count = len(self.procedures)
        self.matched_count = len(self.matched_costs)
        self.match_rate = (
            self.matched_count / self.procedure_count
            if self.procedure_count > 0 else 0.0
        )

        # Sum costs (only where prices are available)
        negotiated = [c.negotiated_dollar for c in self.matched_costs if c.negotiated_dollar]
        min_costs = [c.min_charge for c in self.matched_costs if c.min_charge]
        max_costs = [c.max_charge for c in self.matched_costs if c.max_charge]

        self.total_cost_negotiated = sum(negotiated) if negotiated else None
        self.total_cost_min = sum(min_costs) if min_costs else None
        self.total_cost_max = sum(max_costs) if max_costs else None

    def to_summary_dict(self) -> dict:
        """Return summary dict for export (flattened, no nested objects)."""
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "llm_name": self.llm_name,
            "procedure_count": self.procedure_count,
            "matched_count": self.matched_count,
            "match_rate": self.match_rate,
            "total_cost_negotiated": self.total_cost_negotiated,
            "total_cost_min": self.total_cost_min,
            "total_cost_max": self.total_cost_max,
            "unmatched_procedures": "|".join(self.unmatched_procedures),
        }
