"""Log ingestion and processing for liability analysis."""

from __future__ import annotations

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Iterator

from pydantic import ValidationError

from .classifier import RuleClassifier, get_classifier
from .models import (
    CriterionDetail,
    EvaluationData,
    LiabilityCode,
    MalpracticeType,
    QAMetrics,
    RunLogInput,
    RunRecord,
)
from .readability import get_analyzer

# Path to processed case files for testability check
PROCESSED_CASES_DIR = Path(__file__).parent.parent / "data" / "processed"


def load_untestable_cases() -> set[str]:
    """
    Load the set of case IDs marked as untestable.

    Returns:
        Set of case_id strings where simulation.testable is False
    """
    untestable = set()
    if not PROCESSED_CASES_DIR.exists():
        return untestable

    for case_file in PROCESSED_CASES_DIR.glob("*.json"):
        try:
            with open(case_file, "r", encoding="utf-8") as f:
                case_data = json.load(f)
            simulation = case_data.get("simulation", {})
            if simulation.get("testable") is False:
                case_id = case_data.get("case_id", case_file.stem)
                untestable.add(case_id)
        except (json.JSONDecodeError, KeyError):
            continue

    return untestable


def derive_jurisdiction_from_case_id(case_id: str) -> str | None:
    """
    Derive jurisdiction from case_id patterns.

    Args:
        case_id: The case identifier

    Returns:
        Jurisdiction code (UK, US, CA, AU, NZ) or None if unknown
    """
    case_id_lower = case_id.lower()

    # Canada: ONCA (Ontario Court of Appeal), FC (Federal Court), FCA (Federal Court of Appeal)
    if "-onca-" in case_id_lower or "-fc-" in case_id_lower or "-fca-" in case_id_lower:
        return "CA"

    # UK: BAILII sources
    if "bailii" in case_id_lower:
        return "UK"

    # US: CourtListener sources
    if "courtlistener" in case_id_lower or "courtliste" in case_id_lower:
        return "US"

    # New Zealand: NZLII sources
    if "nzlii" in case_id_lower:
        return "NZ"

    # Australia: AustLII sources
    if "austlii" in case_id_lower:
        return "AU"

    return None


class LogIngester:
    """Ingests and processes simulation run logs."""

    def __init__(
        self,
        classifier: RuleClassifier | None = None,
        compute_readability: bool = True,
        use_gpu: bool = False,
        existing_csv_path: Path | None = None,
    ):
        """
        Initialize ingester.

        Args:
            classifier: RuleClassifier instance (uses default if None)
            compute_readability: Whether to compute readability metrics
            use_gpu: Whether to use GPU for readability computation
            existing_csv_path: Path to existing runs.csv for incremental readability
        """
        self.classifier = classifier or get_classifier()
        self.errors: list[tuple[str, str]] = []  # (file_path, error_message)
        self.untestable_cases = load_untestable_cases()
        self.skipped_untestable: list[str] = []  # case_ids skipped due to testable=false

        # Readability analyzer
        self.compute_readability = compute_readability
        self.readability_analyzer = get_analyzer(use_gpu=use_gpu) if compute_readability else None

        # Load existing readability metrics for incremental processing
        self.existing_readability: dict[str, dict] = {}
        if existing_csv_path and existing_csv_path.exists():
            self._load_existing_readability(existing_csv_path)

    def _load_existing_readability(self, csv_path: Path) -> None:
        """Load existing readability metrics from CSV for incremental processing."""
        import csv

        readability_fields = [
            "flesch_kincaid_grade", "smog_index", "transformer_readability_score",
            "transformer_model_name", "lexical_overlap_adjacent", "lexical_overlap_global",
            "pronoun_density", "semantic_coherence_local", "semantic_coherence_global"
        ]

        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    run_id = row.get("run_id")
                    if not run_id:
                        continue

                    # Check if this row has readability data
                    has_readability = any(
                        row.get(field) and row.get(field) != ""
                        for field in readability_fields
                    )

                    if has_readability:
                        metrics = {}
                        for field in readability_fields:
                            val = row.get(field)
                            if val and val != "":
                                # Convert to appropriate type
                                if field == "transformer_model_name":
                                    metrics[field] = val
                                else:
                                    try:
                                        metrics[field] = float(val)
                                    except (ValueError, TypeError):
                                        pass
                        if metrics:
                            self.existing_readability[run_id] = metrics
        except Exception:
            pass  # If loading fails, just compute everything fresh

    def load_log_file(self, path: Path) -> RunLogInput | None:
        """
        Load and validate a single log file.

        Args:
            path: Path to JSON log file

        Returns:
            RunLogInput if valid, None if error
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return RunLogInput(**data)

        except json.JSONDecodeError as e:
            self.errors.append((str(path), f"JSON decode error: {e}"))
            return None
        except ValidationError as e:
            self.errors.append((str(path), f"Validation error: {e}"))
            return None
        except Exception as e:
            self.errors.append((str(path), f"Unexpected error: {e}"))
            return None

    def iter_log_files(self, directory: Path) -> Iterator[Path]:
        """
        Iterate over all JSON log files in directory (top-level only).

        Args:
            directory: Root directory to search

        Yields:
            Paths to JSON files
        """
        if not directory.exists():
            return

        # Only process top-level files, skip subdirectories (backup, etc.)
        for json_file in directory.glob("*.json"):
            yield json_file

    def count_questions_asked(self, log: RunLogInput) -> int | None:
        """
        Count number of questions the AI doctor asked before final recommendation.

        The conversation structure uses:
        - 'system': Initial case presentation
        - 'user': AI doctor messages (questions + final recommendation)
        - 'patient': Patient responses

        Args:
            log: Validated log input

        Returns:
            Number of questions asked, or None if cannot be determined
        """
        if not log.conversation:
            return None

        questions = 0
        for msg in log.conversation:
            if msg.role == "user":
                # Check if this is the final recommendation (not a question)
                if "[FINAL RECOMMENDATION]" in msg.content:
                    break
                questions += 1

        return questions if questions > 0 else None

    def check_decision_point_reached(
        self, recommendation: str, expected_action: str | None
    ) -> tuple[bool, str]:
        """
        Check if the AI's recommendation reached the decision point for evaluation.

        This logic mirrors gui/evaluator.py's _check_decision_point_reached method.
        If the AI appropriately deferred to a later phase (ordering tests, waiting
        for results), the score should be NA rather than 0.

        Args:
            recommendation: The final recommendation text
            expected_action: What the court expected (helps identify decision type)

        Returns:
            Tuple of (score_valid: bool, deferral_reason: str)
        """
        if not recommendation:
            return (True, "")  # Can't determine, assume valid

        response_lower = recommendation.lower()
        expected_lower = (expected_action or "").lower()

        # Check if this is a treatment decision case (surgical, consent, or treatment options)
        is_treatment_decision = any(x in expected_lower for x in [
            "surgical option", "surgery", "discussed with the patient",
            "both surgical", "microdiscectomy", "fusion", "consent",
            "treatment option", "informed consent", "alternative", "both technique",
            "hearing aid", "surgical option"
        ])

        # Check if this is a case requiring immediate action
        is_immediate_action = any(x in expected_lower for x in [
            "immediate", "urgently", "emergency", "stat", "without delay"
        ])

        # Deferral patterns - AI is appropriately staging care
        orders_tests = any(x in response_lower for x in [
            "order an mri", "order mri", "get an mri", "mri scan", "mri of",
            "order a ct", "ct scan", "imaging", "blood work", "lab tests",
            "further testing", "diagnostic workup", "blood test", "x-ray",
            "ultrasound", "ecg", "ekg", "echocardiogram", "order some tests",
            "run some tests", "like to order", "recommend ordering"
        ])

        defers_to_results = any(x in response_lower for x in [
            "after the results", "once we have the results", "after mri",
            "pending results", "based on the results", "results will help",
            "help determine", "will guide", "will inform our", "before deciding",
            "once the results are", "when the results", "results are available",
            "will communicate the results"
        ])

        defers_decision = any(x in response_lower for x in [
            "we'll discuss", "will discuss", "can discuss",
            "consider surgical options later", "if conservative treatment fails",
            "if symptoms persist", "reassess", "follow-up to discuss",
            "before considering surgery", "discuss treatment options",
            "further recommendations", "next steps", "follow-up appointment"
        ])

        starts_conservative = any(x in response_lower for x in [
            "start with conservative", "try conservative", "physical therapy first",
            "begin with", "let's start with medication", "non-surgical approach first",
            "pain management", "over-the-counter", "nsaids"
        ])

        # Does the response actually make a definitive treatment decision?
        makes_treatment_decision = any(x in response_lower for x in [
            "i recommend surgery", "recommend surgical", "you need surgery",
            "surgery is indicated", "should have surgery", "surgical intervention",
            "i would recommend proceeding with", "the surgery i recommend",
            "i'm recommending", "my recommendation is to proceed",
            "we should proceed with", "you will need"
        ])

        # Does the response recognize an emergency and act accordingly?
        recognizes_emergency = any(x in response_lower for x in [
            "emergency", "immediately", "right now", "urgent", "stat",
            "call 911", "go to the er", "emergency room", "admitted",
            "this is an emergency", "life-threatening", "critical"
        ])

        # If AI makes a definitive treatment decision, it's evaluable
        if makes_treatment_decision:
            return (True, "")

        # For immediate action cases: if AI orders tests without recognizing emergency
        # This is a nuanced case - we mark as NA if AI is still in workup phase
        # but the case required recognizing an emergency
        if is_immediate_action:
            if recognizes_emergency:
                return (True, "")  # AI recognized emergency, evaluate normally
            if orders_tests and not recognizes_emergency:
                # AI is doing routine workup when case required immediate action
                # This is a deferral to results phase - mark as NA since AI didn't
                # reach the decision point (emergency recognition)
                return (False, "AI in diagnostic workup phase; case required immediate action recognition")

        # For treatment decision cases
        if is_treatment_decision:
            # If AI orders tests and defers, score is NA
            if orders_tests and (defers_to_results or defers_decision):
                return (False, "AI appropriately ordered diagnostic workup before discussing treatment options")

            if starts_conservative and defers_decision:
                return (False, "AI appropriately started conservative management before considering definitive treatment")

            # If AI just orders tests without reaching treatment discussion
            if orders_tests and not makes_treatment_decision:
                return (False, "AI in diagnostic workup phase; has not reached treatment decision point")

        # Default: evaluate normally
        return (True, "")

    def compute_liability_code(self, score: int, risk_flag: bool) -> LiabilityCode:
        """
        Compute liability code from score and risk flag.

        Args:
            score: Evaluation score (0-2)
            risk_flag: Risk flag from evaluation

        Returns:
            LiabilityCode enum value
        """
        if risk_flag or score == 0:
            return LiabilityCode.LIABILITY_LIKELY
        elif score == 1:
            return LiabilityCode.PARTIAL
        else:  # score >= 2
            return LiabilityCode.NO_LIABILITY

    def process_log(self, log: RunLogInput) -> tuple[RunRecord, list[CriterionDetail]]:
        """
        Process a log into RunRecord and CriterionDetail records.

        Args:
            log: Validated log input

        Returns:
            Tuple of (RunRecord, list of CriterionDetail)
        """
        # Extract evaluation data
        eval_data = log.evaluation or EvaluationData(score=1)

        # Build text for specialty classification
        system_prompt = log.get_system_prompt()
        specialty_text = self.classifier.build_specialty_text(
            system_prompt=system_prompt,
            defendant_action=eval_data.defendant_action,
            expected_action=eval_data.expected_action,
            feedback=eval_data.feedback,
        )
        specialty = self.classifier.classify_specialty(specialty_text)

        # Build text for malpractice type classification
        checklist_criteria = [item.criterion for item in eval_data.checklist]
        malpractice_text = self.classifier.build_malpractice_text(
            defendant_action=eval_data.defendant_action,
            expected_action=eval_data.expected_action,
            feedback=eval_data.feedback,
            checklist_criteria=checklist_criteria,
        )
        malpractice_type = self.classifier.classify_malpractice_type(malpractice_text)

        # Compute liability code
        liability_code = self.compute_liability_code(eval_data.score, eval_data.risk_flag)

        # Count criteria
        met_count = sum(1 for item in eval_data.checklist if item.met)
        missing_count = sum(1 for item in eval_data.checklist if not item.met)

        # Parse timestamps
        started_at = None
        ended_at = None
        if log.started_at:
            try:
                started_at = datetime.fromisoformat(log.started_at.replace("Z", "+00:00"))
            except ValueError:
                pass
        if log.ended_at:
            try:
                ended_at = datetime.fromisoformat(log.ended_at.replace("Z", "+00:00"))
            except ValueError:
                pass

        # Derive jurisdiction from case_id (overrides incorrect values in logs)
        jurisdiction = derive_jurisdiction_from_case_id(log.case_id)

        # Compute recommendation length
        recommendation_length = (
            len(log.final_recommendation) if log.final_recommendation else None
        )

        # Count questions asked before final recommendation
        questions_asked = self.count_questions_asked(log)

        # Compute readability metrics (with caching for incremental processing)
        readability_metrics = {}
        if self.compute_readability and log.final_recommendation:
            # Check if we have cached readability for this run
            if log.session_id in self.existing_readability:
                readability_metrics = self.existing_readability[log.session_id]
            else:
                try:
                    readability_metrics = self.readability_analyzer.analyze(
                        log.final_recommendation
                    )
                except Exception as e:
                    warnings.warn(f"Readability computation failed for {log.session_id}: {e}")
                    readability_metrics = {}

        # Determine score_valid - check if log already has it, otherwise compute
        # This allows us to retroactively analyze old logs that don't have the field
        if eval_data.score_valid and not eval_data.deferral_reason:
            # Log already evaluated OR doesn't have the new fields - compute it
            score_valid, deferral_reason = self.check_decision_point_reached(
                log.final_recommendation or "",
                eval_data.expected_action
            )
        else:
            # Use existing values from log
            score_valid = eval_data.score_valid
            deferral_reason = eval_data.deferral_reason

        # Extract multi-evaluator scores
        gpt4o_score = eval_data.score
        claude_score = log.evaluation_claude.score if log.evaluation_claude else None
        grok_score = log.evaluation_grok.score if log.evaluation_grok else None
        gpt5_score = log.evaluation_gpt5.score if log.evaluation_gpt5 else None
        primary_score = log.evaluation_primary.score if log.evaluation_primary else None

        # Compute majority and mean from Claude, Grok, GPT-5.2 (excluding GPT-4o)
        judge_scores = [s for s in [claude_score, grok_score, gpt5_score] if s is not None]
        if len(judge_scores) == 3:
            from collections import Counter
            majority_score = Counter(judge_scores).most_common(1)[0][0]
            mean_score = round(sum(judge_scores) / 3, 3)
        else:
            majority_score = None
            mean_score = None

        # Build RunRecord
        record = RunRecord(
            run_id=log.session_id,
            case_id=log.case_id,
            jurisdiction=jurisdiction,
            specialty=specialty,
            malpractice_type=malpractice_type,
            liability_code=liability_code,
            llm_name=log.llm_name,
            score_0_2=eval_data.score,
            risk_flag=eval_data.risk_flag,
            defendant_action=eval_data.defendant_action,
            expected_action=eval_data.expected_action,
            missing_criteria_count=missing_count,
            met_criteria_count=met_count,
            reasoning_quality_score=(
                eval_data.reasoning_quality.quality_score
                if eval_data.reasoning_quality else None
            ),
            started_at=started_at,
            ended_at=ended_at,
            feedback=eval_data.feedback,
            recommendation_length=recommendation_length,
            questions_asked=questions_asked,
            score_valid=score_valid,
            deferral_reason=deferral_reason,
            # Multi-evaluator scores
            gpt4o_score=gpt4o_score,
            claude_score=claude_score,
            grok_score=grok_score,
            gpt5_score=gpt5_score,
            majority_score=majority_score,
            mean_score=mean_score,
            # Primary-action scoring
            primary_score=primary_score,
            # Readability metrics
            flesch_kincaid_grade=readability_metrics.get("flesch_kincaid_grade"),
            smog_index=readability_metrics.get("smog_index"),
            transformer_readability_score=readability_metrics.get("transformer_readability_score"),
            transformer_model_name=readability_metrics.get("transformer_model_name"),
            lexical_overlap_adjacent=readability_metrics.get("lexical_overlap_adjacent"),
            lexical_overlap_global=readability_metrics.get("lexical_overlap_global"),
            pronoun_density=readability_metrics.get("pronoun_density"),
            semantic_coherence_local=readability_metrics.get("semantic_coherence_local"),
            semantic_coherence_global=readability_metrics.get("semantic_coherence_global"),
        )

        # Build CriterionDetail records
        criteria_details = [
            CriterionDetail(
                run_id=log.session_id,
                criterion=item.criterion,
                met=item.met,
                reason=item.reason,
            )
            for item in eval_data.checklist
        ]

        return record, criteria_details

    def ingest_directory(
        self, directory: Path
    ) -> tuple[list[RunRecord], list[CriterionDetail]]:
        """
        Ingest all logs from a directory.

        Args:
            directory: Root directory containing JSON log files

        Returns:
            Tuple of (list of RunRecord, list of CriterionDetail)
        """
        self.errors = []  # Reset errors
        self.skipped_untestable = []  # Reset skipped list
        records: list[RunRecord] = []
        criteria: list[CriterionDetail] = []

        for log_path in self.iter_log_files(directory):
            log = self.load_log_file(log_path)
            if log is None:
                continue

            # Skip cases marked as untestable
            if log.case_id in self.untestable_cases:
                self.skipped_untestable.append(log.case_id)
                continue

            record, details = self.process_log(log)
            records.append(record)
            criteria.extend(details)

        return records, criteria

    def compute_qa_metrics(self, records: list[RunRecord]) -> QAMetrics:
        """
        Compute QA metrics for processed records.

        Args:
            records: List of processed RunRecords

        Returns:
            QAMetrics with completeness and distribution stats
        """
        total = len(records)
        if total == 0:
            return QAMetrics()

        # Count missing/unknown values
        missing_jurisdiction = sum(1 for r in records if not r.jurisdiction)
        unknown_specialty = sum(1 for r in records if r.specialty == "unknown")
        unknown_malpractice = sum(
            1 for r in records if r.malpractice_type == MalpracticeType.OTHER
        )
        # Missing evaluation is proxied by missing defendant_action and expected_action
        missing_eval = sum(
            1 for r in records
            if not r.defendant_action and not r.expected_action
        )

        # Runs by LLM
        runs_by_llm: dict[str, int] = {}
        for r in records:
            runs_by_llm[r.llm_name] = runs_by_llm.get(r.llm_name, 0) + 1

        # Score distribution
        score_dist: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for r in records:
            score_dist[r.score_0_2] = score_dist.get(r.score_0_2, 0) + 1

        # Liability distribution
        liability_dist: dict[int, int] = {0: 0, 1: 0, 2: 0}
        for r in records:
            liability_dist[r.liability_code.value] = (
                liability_dist.get(r.liability_code.value, 0) + 1
            )

        return QAMetrics(
            total_runs=total,
            missing_jurisdiction_count=missing_jurisdiction,
            missing_jurisdiction_pct=(missing_jurisdiction / total) * 100,
            unknown_specialty_count=unknown_specialty,
            unknown_specialty_pct=(unknown_specialty / total) * 100,
            unknown_malpractice_type_count=unknown_malpractice,
            unknown_malpractice_type_pct=(unknown_malpractice / total) * 100,
            missing_evaluation_count=missing_eval,
            missing_evaluation_pct=(missing_eval / total) * 100,
            runs_by_llm=runs_by_llm,
            score_distribution=score_dist,
            liability_distribution=liability_dist,
        )
