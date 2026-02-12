"""Pydantic models for liability analysis."""

from __future__ import annotations

from datetime import datetime
from enum import IntEnum, Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class LiabilityCode(IntEnum):
    """Liability outcome codes."""
    NO_LIABILITY = 0  # Defensible (score >= 2, no risk flag)
    PARTIAL = 1       # Partially defensible (score = 1)
    LIABILITY_LIKELY = 2  # Not defensible (score = 0 or risk flag)


class MalpracticeType(str, Enum):
    """Major malpractice category types."""
    INFORMED_CONSENT = "informed_consent"
    DIAGNOSIS_DELAY_OR_ERROR = "diagnosis_delay_or_error"
    TEST_SELECTION_ERROR = "test_selection_error"
    MEDICATION_SELECTION_ERROR = "medication_selection_error"
    TREATMENT_TIMING_ERROR = "treatment_timing_error"
    SURGICAL_TECHNIQUE_ERROR = "surgical_technique_error"
    DISCHARGE_DISPOSITION_ERROR = "discharge_disposition_error"
    REFERRAL_FAILURE = "referral_failure"
    CARE_MANAGEMENT_ERROR = "care_management_error"
    MONITORING_OR_ESCALATION_FAILURE = "monitoring_or_escalation_failure"
    DOCUMENTATION_FAILURE = "documentation_failure"
    COMMUNICATION_FAILURE = "communication_failure"
    # New types (added to reduce 'other' category)
    PROFESSIONAL_BOUNDARIES_VIOLATION = "professional_boundaries_violation"
    EQUIPMENT_OR_FACILITY_SAFETY = "equipment_or_facility_safety"
    CARE_PLANNING_ERROR = "care_planning_error"
    # Legacy types (kept for backwards compatibility)
    TREATMENT_OR_PROCEDURE_ERROR = "treatment_or_procedure_error"
    OTHER = "other"


class ChecklistItem(BaseModel):
    """Individual criterion from evaluation checklist."""
    criterion: str
    met: bool
    reason: str | None = None


class ReasoningQuality(BaseModel):
    """Reasoning quality metrics from evaluation."""
    considers_differential: bool = False
    integrates_evidence: bool = False
    acknowledges_uncertainty: bool = False
    considers_urgency: bool = False
    quality_score: float = 0.0


class EvaluationData(BaseModel):
    """Evaluation section from run log."""
    score: int = Field(ge=0, le=2)
    risk_flag: bool = False
    feedback: str | None = None
    defendant_action: str | None = None
    expected_action: str | None = None
    checklist: list[ChecklistItem] = Field(default_factory=list)
    reasoning_quality: ReasoningQuality | None = None
    cognitive_error_avoided: str | None = None
    score_valid: bool = True  # False if decision point not reached (score should be NA)
    deferral_reason: str | None = None  # Why score is not valid

    @field_validator("checklist", mode="before")
    @classmethod
    def parse_checklist(cls, v: Any) -> list[ChecklistItem]:
        if v is None:
            return []
        if isinstance(v, list):
            return [ChecklistItem(**item) if isinstance(item, dict) else item for item in v]
        return []


class ConversationMessage(BaseModel):
    """Single message in conversation."""
    role: str
    content: str
    timestamp: str | None = None


class RunLogInput(BaseModel):
    """Input model for parsing run log JSON files."""
    session_id: str
    case_id: str
    jurisdiction: str | None = None
    llm_name: str = "unknown"
    started_at: str | None = None
    ended_at: str | None = None
    conversation: list[ConversationMessage] = Field(default_factory=list)
    final_recommendation: str | None = None
    evaluation: EvaluationData | None = None
    # Multi-evaluator scores (from validation study)
    evaluation_claude: EvaluationData | None = None
    evaluation_grok: EvaluationData | None = None
    evaluation_gpt5: EvaluationData | None = None
    evaluation_majority: dict | None = None  # Pre-computed majority vote
    evaluation_primary: EvaluationData | None = None  # Primary-action scoring
    state_history: list[dict] = Field(default_factory=list)
    revealed_info: list[str] = Field(default_factory=list)

    def get_system_prompt(self) -> str:
        """Extract system prompt from conversation."""
        for msg in self.conversation:
            if msg.role == "system":
                return msg.content
        return ""


class RunRecord(BaseModel):
    """Canonical record for the summary table."""
    run_id: str  # session_id
    case_id: str
    jurisdiction: str | None = None
    specialty: str = "unknown"
    malpractice_type: MalpracticeType = MalpracticeType.OTHER
    liability_code: LiabilityCode = LiabilityCode.PARTIAL
    llm_name: str = "unknown"
    score_0_2: int = Field(ge=0, le=2, default=1)
    risk_flag: bool = False
    defendant_action: str | None = None
    expected_action: str | None = None
    missing_criteria_count: int = 0
    met_criteria_count: int = 0
    reasoning_quality_score: float | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None
    feedback: str | None = None
    recommendation_length: int | None = None
    questions_asked: int | None = None  # Number of questions before final recommendation
    score_valid: bool = True  # False if decision point not reached (NA)
    deferral_reason: str | None = None  # Why score is NA

    # Multi-evaluator scores (from validation study)
    gpt4o_score: int | None = None
    claude_score: int | None = None
    grok_score: int | None = None
    gpt5_score: int | None = None
    majority_score: int | None = None  # Majority vote of Claude, Grok, GPT-5.2
    mean_score: float | None = None  # Mean of Claude, Grok, GPT-5.2

    # Primary-action scoring (alternative scoring method)
    primary_score: int | None = None  # Score from primary-action scoring method

    # Readability metrics
    flesch_kincaid_grade: float | None = None
    smog_index: float | None = None
    transformer_readability_score: float | None = None
    transformer_model_name: str | None = None
    lexical_overlap_adjacent: float | None = None
    lexical_overlap_global: float | None = None
    pronoun_density: float | None = None
    semantic_coherence_local: float | None = None
    semantic_coherence_global: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "run_id": self.run_id,
            "case_id": self.case_id,
            "jurisdiction": self.jurisdiction,
            "specialty": self.specialty,
            "malpractice_type": self.malpractice_type.value,
            "liability_code": self.liability_code.value,
            "llm_name": self.llm_name,
            "score_0_2": self.score_0_2,
            "risk_flag": self.risk_flag,
            "defendant_action": self.defendant_action,
            "expected_action": self.expected_action,
            "missing_criteria_count": self.missing_criteria_count,
            "met_criteria_count": self.met_criteria_count,
            "reasoning_quality_score": self.reasoning_quality_score,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "feedback": self.feedback,
            "recommendation_length": self.recommendation_length,
            "questions_asked": self.questions_asked,
            "score_valid": self.score_valid,
            "deferral_reason": self.deferral_reason,
            # Multi-evaluator scores
            "gpt4o_score": self.gpt4o_score,
            "claude_score": self.claude_score,
            "grok_score": self.grok_score,
            "gpt5_score": self.gpt5_score,
            "majority_score": self.majority_score,
            "mean_score": self.mean_score,
            # Primary-action scoring
            "primary_score": self.primary_score,
            # Readability metrics
            "flesch_kincaid_grade": self.flesch_kincaid_grade,
            "smog_index": self.smog_index,
            "transformer_readability_score": self.transformer_readability_score,
            "transformer_model_name": self.transformer_model_name,
            "lexical_overlap_adjacent": self.lexical_overlap_adjacent,
            "lexical_overlap_global": self.lexical_overlap_global,
            "pronoun_density": self.pronoun_density,
            "semantic_coherence_local": self.semantic_coherence_local,
            "semantic_coherence_global": self.semantic_coherence_global,
        }


class CriterionDetail(BaseModel):
    """Detail record for criteria (long table)."""
    run_id: str
    criterion: str
    met: bool
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "run_id": self.run_id,
            "criterion": self.criterion,
            "met": self.met,
            "reason": self.reason,
        }


class QAMetrics(BaseModel):
    """Quality assurance metrics for the dataset."""
    total_runs: int = 0
    missing_jurisdiction_count: int = 0
    missing_jurisdiction_pct: float = 0.0
    unknown_specialty_count: int = 0
    unknown_specialty_pct: float = 0.0
    unknown_malpractice_type_count: int = 0
    unknown_malpractice_type_pct: float = 0.0
    missing_evaluation_count: int = 0
    missing_evaluation_pct: float = 0.0

    # Distribution by model
    runs_by_llm: dict[str, int] = Field(default_factory=dict)

    # Score distribution
    score_distribution: dict[int, int] = Field(default_factory=dict)

    # Liability distribution
    liability_distribution: dict[int, int] = Field(default_factory=dict)

    def to_report(self) -> str:
        """Generate human-readable QA report."""
        lines = [
            "=" * 60,
            "LIABILITY ANALYSIS QA REPORT",
            "=" * 60,
            f"Total runs analyzed: {self.total_runs}",
            "",
            "DATA COMPLETENESS:",
            f"  Missing jurisdiction: {self.missing_jurisdiction_count} ({self.missing_jurisdiction_pct:.1f}%)",
            f"  Unknown specialty: {self.unknown_specialty_count} ({self.unknown_specialty_pct:.1f}%)",
            f"  Unknown malpractice type: {self.unknown_malpractice_type_count} ({self.unknown_malpractice_type_pct:.1f}%)",
            f"  Missing evaluation: {self.missing_evaluation_count} ({self.missing_evaluation_pct:.1f}%)",
            "",
            "RUNS BY LLM:",
        ]
        for llm, count in sorted(self.runs_by_llm.items()):
            lines.append(f"  {llm}: {count}")

        lines.extend([
            "",
            "SCORE DISTRIBUTION (0-2):",
        ])
        for score in [0, 1, 2]:
            count = self.score_distribution.get(score, 0)
            pct = (count / self.total_runs * 100) if self.total_runs > 0 else 0
            lines.append(f"  Score {score}: {count} ({pct:.1f}%)")

        lines.extend([
            "",
            "LIABILITY DISTRIBUTION:",
        ])
        liability_labels = {0: "No Liability", 1: "Partial", 2: "Liability Likely"}
        for code in [0, 1, 2]:
            count = self.liability_distribution.get(code, 0)
            pct = (count / self.total_runs * 100) if self.total_runs > 0 else 0
            lines.append(f"  {liability_labels[code]}: {count} ({pct:.1f}%)")

        lines.append("=" * 60)
        return "\n".join(lines)
