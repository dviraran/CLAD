"""Pydantic models for the case simulation schema."""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl, field_validator

from .enums import (
    ActionType,
    ClinicalDomain,
    ConsentSubtype,
    DiagnosisSubtype,
    DiscoveryStatus,
    EvidenceType,
    Jurisdiction,
    MalpracticeCategory,
    OutcomeSeverity,
    PhaseId,
    ProcedureSubtype,
    RejectionReason,
    RequestableType,
    Sex,
    Source,
    Verdict,
)


# Type aliases for evidence IDs
EvidenceId = Annotated[str, Field(pattern=r"^E\d+$")]
RequestId = Annotated[str, Field(pattern=r"^R\d+$")]
DecisionId = Annotated[str, Field(pattern=r"^D\d+$")]


class EvidenceItem(BaseModel):
    """A single piece of evidence from the judgment."""

    evidence_id: EvidenceId = Field(description="Unique evidence identifier (E001, E002, etc.)")
    type: EvidenceType
    text: str = Field(description="Verbatim or closely paraphrased text from judgment")
    paragraph_ref: str = Field(description="Paragraph number or section reference")
    speaker: str | None = Field(
        default=None, description="Who said/wrote this (judge, expert witness name, etc.)"
    )


class PatientDemographics(BaseModel):
    """Patient demographic information."""

    age_at_presentation: str | None = Field(default=None, description="Age or age range if stated")
    sex: Sex | None = None
    relevant_social_history: str | None = None


class VitalSigns(BaseModel):
    """Vital signs with flexible fields."""

    blood_pressure: str | None = None
    heart_rate: str | None = None
    respiratory_rate: str | None = None
    temperature: str | None = None
    oxygen_saturation: str | None = None
    other: dict[str, str] | None = None


class PhysicalExamination(BaseModel):
    """Physical examination findings."""

    vital_signs: VitalSigns | None = None
    general_appearance: str | None = None
    focused_exam: dict[str, str] | None = Field(
        default=None, description="System-specific exam findings"
    )

    @field_validator("vital_signs", mode="before")
    @classmethod
    def coerce_vital_signs(cls, v: VitalSigns | dict | str | None) -> VitalSigns | dict | None:
        """Convert string values to None when LLM returns 'Not documented' etc."""
        if v is None:
            return None
        if isinstance(v, str):
            if "not documented" in v.lower() or "unknown" in v.lower() or "n/a" in v.lower():
                return None
            return None
        return v

    @field_validator("focused_exam", mode="before")
    @classmethod
    def coerce_focused_exam(cls, v: dict | str | None) -> dict | None:
        """Convert string values to None when LLM returns 'Not documented' etc."""
        if v is None:
            return None
        if isinstance(v, str):
            if "not documented" in v.lower() or "unknown" in v.lower() or "n/a" in v.lower():
                return None
            return None
        return v


class InitialState(BaseModel):
    """Initial clinical presentation state."""

    patient_demographics: PatientDemographics | None = None
    chief_complaint: str = Field(description="Primary presenting complaint")
    history_of_present_illness: str | None = Field(
        default=None, description="HPI as available from judgment"
    )
    past_medical_history: list[str] | None = None
    medications: list[str] | None = None
    allergies: list[str] | None = None
    physical_examination: PhysicalExamination | None = None
    initial_working_problem: str | None = Field(
        default=None, description="Initial differential or working diagnosis"
    )
    known_constraints: list[str] | None = Field(
        default=None,
        description="Contextual constraints (resource limitations, patient preferences, etc.)",
    )
    # New fields for enhanced simulation
    initial_differential: list[str] | None = Field(
        default=None, description="Differential diagnosis at presentation"
    )
    critical_findings: list[str] | None = Field(
        default=None, description="Findings that should trigger urgent action"
    )
    urgency_level: str | None = Field(
        default=None, description="ROUTINE, URGENT, or EMERGENT"
    )
    evidence_ids: list[EvidenceId] = Field(description="Evidence items supporting initial state")

    @field_validator(
        "initial_differential",
        "critical_findings",
        "past_medical_history",
        "medications",
        "allergies",
        "known_constraints",
        mode="before",
    )
    @classmethod
    def coerce_to_list(cls, v: list | str | None) -> list | None:
        """Convert string values to empty list when LLM returns 'Not documented' etc."""
        if v is None:
            return None
        if isinstance(v, str):
            # If LLM returned a string like "Not documented in judgment", convert to empty list
            if "not documented" in v.lower() or "unknown" in v.lower() or "none" in v.lower():
                return []
            # Otherwise wrap single string in a list
            return [v]
        return v


class RevealContent(BaseModel):
    """Content revealed when a requestable is accessed."""

    result_summary: str | None = None
    detailed_findings: str | None = None
    clinical_significance: str | None = None
    # New fields for enhanced clinical decision support
    decision_impact: str | None = Field(
        default=None, description="How this result should impact clinical decision-making"
    )
    if_normal_would_mean: str | None = Field(
        default=None, description="Clinical interpretation if result is normal"
    )
    if_abnormal_means: str | None = Field(
        default=None, description="Clinical interpretation if result is abnormal"
    )


class Requestable(BaseModel):
    """An item that can be requested during simulation."""

    request_id: RequestId
    type: RequestableType
    name: str = Field(description="Human-readable name of the requestable")
    description: str | None = Field(default=None, description="What this item represents")
    available_phase: PhaseId = Field(description="Phase when this becomes available")
    reveal: RevealContent | None = Field(
        default=None, description="Information revealed when requested"
    )
    was_ordered_in_case: bool | None = Field(
        default=None, description="Whether this was actually ordered in the real case"
    )
    evidence_ids: list[EvidenceId] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    """A single event in a timeline phase."""

    event: str
    timestamp: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class TimelinePhase(BaseModel):
    """A phase in the clinical timeline."""

    phase_id: PhaseId
    name: str
    description: str
    duration: str | None = Field(default=None, description="Approximate duration if known")
    key_events: list[TimelineEvent] | None = None


class DecisionOption(BaseModel):
    """An option at a decision point."""

    option_id: str
    description: str
    is_defendant_choice: bool | None = None
    is_court_endorsed: bool | None = None
    clinical_reasoning: str | None = None

    @field_validator("is_defendant_choice", "is_court_endorsed", mode="before")
    @classmethod
    def coerce_to_bool(cls, v: bool | str | None) -> bool | None:
        """Convert string values to None when LLM returns 'Not documented' etc."""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            if "not documented" in v_lower or "unknown" in v_lower or "n/a" in v_lower:
                return None
            if v_lower in ("true", "yes", "1"):
                return True
            if v_lower in ("false", "no", "0"):
                return False
            return None
        return v


class ActualAction(BaseModel):
    """What the defendant actually did."""

    option_id: str | None = None
    description: str
    reasoning_stated: str | None = None
    evidence_ids: list[EvidenceId]


class ExpectedAction(BaseModel):
    """What the court determined should have been done."""

    option_id: str | None = None
    description: str
    standard_of_care_basis: str | None = None
    evidence_ids: list[EvidenceId]


class ScoringCriterion(BaseModel):
    """A single scoring criterion."""

    criterion: str
    points: int
    explanation: str | None = None


class ScoringRubric(BaseModel):
    """Scoring rubric for a decision point."""

    max_score: int
    criteria: list[ScoringCriterion] | None = None


class DecisionExplanation(BaseModel):
    """Explanation of why the decision was wrong and what should have happened."""

    why_defendant_wrong: str | None = None
    what_should_have_happened: str | None = None
    legal_standard_applied: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class DifferentialItem(BaseModel):
    """A single diagnosis in the differential at a decision point."""

    diagnosis: str
    probability: str | None = Field(default=None, description="high, medium, low")
    supporting_evidence: list[EvidenceId] | None = None
    against_evidence: list[EvidenceId] | None = None
    would_be_ruled_out_by: str | None = Field(
        default=None, description="What test/finding would exclude this diagnosis"
    )


class DecisionPoint(BaseModel):
    """A point where the learner must make a clinical decision."""

    decision_id: DecisionId
    phase_id: PhaseId
    clinical_context: str | None = Field(
        default=None, description="Clinical situation at this decision point"
    )
    prompt: str = Field(description="Question posed to the learner")
    action_type: ActionType
    options: list[DecisionOption] = Field(min_length=2)
    actual_action_defendant: ActualAction
    expected_action_court: ExpectedAction
    scoring_rubric: ScoringRubric
    explanation: DecisionExplanation | None = None
    is_malpractice_point: bool | None = Field(
        default=None, description="Whether this is the key point of malpractice failure"
    )
    # Cognitive error analysis (for malpractice points)
    reasoning_error_type: str | None = Field(
        default=None,
        description="Type of cognitive error: ANCHORING, PREMATURE_CLOSURE, CONFIRMATION_BIAS, AVAILABILITY_BIAS, OVERCONFIDENCE, DIAGNOSTIC_MOMENTUM, TRIAGE_ERROR"
    )
    reasoning_error_explanation: str | None = Field(
        default=None, description="Why the defendant made this cognitive error"
    )
    defendant_ignored_evidence: list[EvidenceId] | None = Field(
        default=None, description="Evidence IDs the defendant should have considered but didn't"
    )
    what_triggered_error: str | None = Field(
        default=None, description="What caused the cognitive error"
    )
    # Differential diagnosis at this decision point
    differential_at_decision: list[DifferentialItem] | None = Field(
        default=None, description="Differential diagnosis at this decision point"
    )


class PatientOutcome(BaseModel):
    """Patient outcome information."""

    description: str | None = None
    severity: str | None = None
    long_term_effects: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class LegalOutcome(BaseModel):
    """Legal outcome information."""

    verdict: Verdict | None = None
    damages_awarded: str | None = None
    key_findings: list[str] | None = None
    evidence_ids: list[EvidenceId] | None = None

    @field_validator("key_findings", mode="before")
    @classmethod
    def coerce_to_list(cls, v: list | str | None) -> list | None:
        """Convert string values to empty list when LLM returns 'Not documented' etc."""
        if v is None:
            return None
        if isinstance(v, str):
            if "not documented" in v.lower() or "unknown" in v.lower() or "none" in v.lower():
                return []
            return [v]
        return v


class MalpracticeDetermination(BaseModel):
    """Court's malpractice determination."""

    breach_found: bool | None = None
    causation_established: bool | None = None
    point_of_failure: str | None = Field(
        default=None, description="The specific action/inaction that constituted malpractice"
    )
    counterfactual: str | None = Field(
        default=None, description="What the court determined should have been done"
    )
    evidence_ids: list[EvidenceId] | None = None

    @field_validator("breach_found", "causation_established", mode="before")
    @classmethod
    def coerce_to_bool(cls, v: bool | str | None) -> bool | None:
        """Convert string values to None when LLM returns 'Not documented' etc."""
        if v is None:
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            v_lower = v.lower()
            if "not documented" in v_lower or "unknown" in v_lower or "n/a" in v_lower:
                return None
            if v_lower in ("true", "yes", "1"):
                return True
            if v_lower in ("false", "no", "0"):
                return False
            return None
        return v


class EndState(BaseModel):
    """End state of the simulation."""

    patient_outcome: PatientOutcome
    legal_outcome: LegalOutcome
    malpractice_determination: MalpracticeDetermination


class Simulation(BaseModel):
    """The simulation structure."""

    # Testability assessment - must be evaluated first
    testable: bool = Field(
        default=True,
        description="Whether this case is suitable for LLM evaluation. False if: (1) procedural judgment only, (2) claim dismissed/no liability found, (3) no substantive breach determination"
    )
    testable_reason: str | None = Field(
        default=None,
        description="Explanation of why testable=False, if applicable"
    )

    initial_state: InitialState
    requestables: list[Requestable] = Field(default_factory=list)
    timeline_phases: list[TimelinePhase] = Field(default_factory=list)
    decision_points: list[DecisionPoint] = Field(default_factory=list)  # Allow empty for untestable cases
    end_state: EndState | None = Field(default=None)  # Allow None for untestable cases


class FactualTimelineEvent(BaseModel):
    """An event in the factual timeline."""

    timestamp: str = Field(description="Date/time or relative timing")
    event: str
    actor: str | None = None
    location: str | None = None
    evidence_ids: list[EvidenceId]


class TestPerformed(BaseModel):
    """A test that was performed."""

    test_name: str
    date: str | None = None
    result: str | None = None
    interpretation: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class Diagnosis(BaseModel):
    """A diagnosis made during the case."""

    diagnosis: str
    when_made: str | None = None
    made_by: str | None = None
    correct: bool | None = None
    evidence_ids: list[EvidenceId] | None = None


class Treatment(BaseModel):
    """A treatment provided."""

    treatment: str
    date: str | None = None
    provider: str | None = None
    outcome: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class Complication(BaseModel):
    """A complication that occurred."""

    complication: str
    when_occurred: str | None = None
    severity: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class Allegation(BaseModel):
    """An allegation made in the case."""

    allegation: str
    against: str | None = None
    found_proven: bool | None = None
    evidence_ids: list[EvidenceId] | None = None


class CourtFinding(BaseModel):
    """A finding by the court."""

    finding: str
    legal_significance: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class MalpracticeAnalysis(BaseModel):
    """Analysis of the malpractice."""

    point_of_failure: str | None = None
    court_approved_counterfactual: str | None = None
    standard_of_care_violated: str | None = None
    evidence_ids: list[EvidenceId] | None = None


class GroundTruth(BaseModel):
    """Ground truth information from the judgment."""

    factual_timeline: list[FactualTimelineEvent]
    tests_performed: list[TestPerformed] | None = None
    diagnoses: list[Diagnosis] | None = None
    treatments: list[Treatment] | None = None
    complications: list[Complication] | None = None
    allegations: list[Allegation] | None = None
    court_findings: list[CourtFinding] | None = None
    malpractice_analysis: MalpracticeAnalysis | None = None


class TaxonomyLabels(BaseModel):
    """Taxonomy labels for the case."""

    malpractice_categories: list[MalpracticeCategory] = Field(min_length=1)
    consent_subtypes: list[ConsentSubtype] | None = None
    diagnosis_subtypes: list[DiagnosisSubtype] | None = None
    procedure_subtypes: list[ProcedureSubtype] | None = None


class QualityMetrics(BaseModel):
    """Quality metrics for the extracted case."""

    evidence_coverage_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Proportion of claims backed by evidence citations",
    )
    simulation_completeness: float | None = Field(
        default=None, ge=0, le=1, description="Completeness of simulation structure"
    )
    decision_point_quality: float | None = Field(
        default=None, ge=0, le=1, description="Quality score for decision points"
    )
    missing_elements: list[str] | None = Field(
        default=None, description="List of expected but missing elements"
    )
    warnings: list[str] | None = Field(default=None, description="Quality warnings")
    validation_passed: bool | None = None

    @field_validator(
        "evidence_coverage_score", "simulation_completeness", "decision_point_quality",
        mode="before"
    )
    @classmethod
    def clamp_score(cls, v: float | None) -> float | None:
        """Clamp scores to valid range [0, 1] to handle LLM errors gracefully."""
        if v is None:
            return None
        # If LLM returns percentage (e.g., 85), convert to decimal
        if v > 1:
            v = v / 100 if v <= 100 else 1.0
        return max(0.0, min(1.0, v))


class CaseSummary(BaseModel):
    """Case summary information."""

    brief: str = Field(max_length=500, description="One-paragraph case summary")
    clinical_synopsis: str = Field(description="Clinical narrative summary")
    legal_synopsis: str = Field(description="Legal proceedings summary")


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""

    extracted_at: datetime | None = None
    extractor_version: str | None = None
    model_used: str | None = None
    extraction_passes: int | None = None
    human_reviewed: bool | None = None


class CaseSimulation(BaseModel):
    """Complete case simulation model."""

    schema_version: str = Field(default="1.0.0", description="Schema version")
    case_id: Annotated[str, Field(pattern=r"^[a-zA-Z0-9-]+$")] = Field(
        description="Unique identifier for the case"
    )
    source: Source
    jurisdiction: Jurisdiction
    court: str = Field(description="Court that issued the judgment")
    decision_date: date = Field(description="Date of judgment")
    url: HttpUrl = Field(description="Source URL of the judgment")
    case_name: str | None = Field(default=None, description="Full case name (parties)")
    neutral_citation: str | None = Field(default=None, description="Neutral citation if available")
    clinical_domain: ClinicalDomain
    outcome_severity: OutcomeSeverity
    summary: CaseSummary
    evidence_index: list[EvidenceItem]
    simulation: Simulation
    ground_truth: GroundTruth
    taxonomy_labels: TaxonomyLabels
    quality: QualityMetrics
    extraction_metadata: ExtractionMetadata | None = None

    @field_validator("case_id")
    @classmethod
    def validate_case_id(cls, v: str) -> str:
        """Validate case_id format."""
        if not v or len(v) < 3:
            raise ValueError("case_id must be at least 3 characters")
        return v


# Discovery models


class DiscoveryRecord(BaseModel):
    """Record of a discovered case before fetch."""

    case_id: str
    source: Source
    jurisdiction: Jurisdiction
    court: str | None = None
    title: str
    year: int | None = None
    url: HttpUrl
    discovery_methods: list[str] = Field(
        description="Methods used to discover this case (keyword_search, structural_heuristic, etc.)"
    )
    query_terms: list[str] | None = Field(
        default=None, description="Query terms that matched this case"
    )
    estimated_length: int | None = Field(
        default=None, description="Estimated document length in characters"
    )
    priority_score: float | None = Field(
        default=None, ge=0, le=1, description="Priority score for fetch ordering"
    )
    status: DiscoveryStatus = Field(default=DiscoveryStatus.QUEUED)
    rejection_reason: RejectionReason | None = None
    discovered_at: datetime = Field(default_factory=datetime.utcnow)
    fetched_at: datetime | None = None
    raw_file_path: str | None = None
    content_hash: str | None = None
    cited_by: list[str] | None = Field(
        default=None, description="Case IDs that cite this case"
    )
    cites: list[str] | None = Field(
        default=None, description="Case IDs cited by this case"
    )


class KeywordConfig(BaseModel):
    """Configuration for keyword-based discovery."""

    legal_terms: list[str] = Field(
        default_factory=lambda: [
            "clinical negligence",
            "medical negligence",
            "malpractice",
            "breach of duty",
            "failure to warn",
            "informed consent",
            "causation",
        ]
    )
    clinical_signal_terms: list[str] = Field(
        default_factory=lambda: [
            "surgery",
            "operation",
            "biopsy",
            "CT",
            "MRI",
            "scan",
            "post-operative",
            "haemorrhage",
            "bleeding",
            "stroke",
            "sepsis",
            "tumour",
            "cancer",
            "pregnancy",
            "labour",
            "emergency department",
        ]
    )
    required_terms: list[str] | None = Field(
        default=None, description="Terms that must be present"
    )
    excluded_terms: list[str] | None = Field(
        default=None, description="Terms that exclude a case"
    )


class StructuralHeuristicsConfig(BaseModel):
    """Configuration for structural heuristics filtering."""

    min_length: int = Field(default=10000, description="Minimum document length in characters")
    required_headings: list[str] = Field(
        default_factory=lambda: [
            "The Facts",
            "Clinical Background",
            "Medical Evidence",
            "Expert Evidence",
            "Consent",
            "Causation",
            "Background",
            "Factual Background",
        ]
    )
    min_heading_matches: int = Field(
        default=1, description="Minimum number of required headings that must be present"
    )
    preferred_courts: list[str] = Field(
        default_factory=lambda: ["EWHC", "QB", "ONSC", "ONCA", "NSWSC"]
    )


class SeedCase(BaseModel):
    """A seed case for discovery."""

    url: HttpUrl
    title: str | None = None
    notes: str | None = None
    priority: float = Field(default=1.0, ge=0, le=1)
