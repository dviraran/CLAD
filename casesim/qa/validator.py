"""Quality assurance and validation for case simulations."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, ValidationError
from pydantic import ValidationError as PydanticValidationError

from ..config import get_settings
from ..schemas import CaseSimulation
from ..utils import get_logger


@dataclass
class ValidationIssue:
    """A single validation issue."""

    severity: str  # error, warning, info
    code: str
    field: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of validation."""

    valid: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]


class CaseValidator:
    """Validates case simulations for quality and correctness."""

    def __init__(self, schema_path: Path | None = None):
        """Initialize validator."""
        self.settings = get_settings()
        self.logger = get_logger("qa.validator")

        # Load JSON schema
        schema_path = schema_path or Path("schema/case_sim_v1.json")
        if schema_path.exists():
            with open(schema_path) as f:
                self.schema = json.load(f)
            self.json_validator = Draft202012Validator(self.schema)
        else:
            self.schema = None
            self.json_validator = None
            self.logger.warning(f"Schema not found at {schema_path}")

    def validate(
        self,
        case: CaseSimulation | dict[str, Any],
        strict: bool = True,
    ) -> ValidationResult:
        """Validate a case simulation."""
        result = ValidationResult(valid=True)

        # Convert to dict if needed
        if isinstance(case, CaseSimulation):
            case_dict = case.model_dump(mode="json")
        else:
            case_dict = case

        # Run all validators
        self._validate_schema(case_dict, result)
        self._validate_evidence_references(case_dict, result)
        self._validate_simulation_structure(case_dict, result)
        self._validate_decision_points(case_dict, result)
        self._validate_clinical_consistency(case_dict, result)
        self._validate_completeness(case_dict, result)

        # Calculate quality scores
        self._calculate_scores(case_dict, result)

        # Determine overall validity
        if strict:
            result.valid = len(result.errors) == 0
        else:
            result.valid = not any(
                i.code in ["SCHEMA_ERROR", "MISSING_REQUIRED", "INVALID_REFERENCE"]
                for i in result.errors
            )

        return result

    def _validate_schema(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate against JSON schema."""
        if not self.json_validator:
            return

        try:
            errors = list(self.json_validator.iter_errors(case))
            for error in errors[:10]:  # Limit to first 10
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="SCHEMA_ERROR",
                    field=".".join(str(p) for p in error.path),
                    message=error.message,
                ))
        except Exception as e:
            result.issues.append(ValidationIssue(
                severity="error",
                code="SCHEMA_VALIDATION_FAILED",
                field="",
                message=str(e),
            ))

    def _validate_evidence_references(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate all evidence ID references."""
        # Build set of valid evidence IDs
        evidence_index = case.get("evidence_index", [])
        valid_ids = {e.get("evidence_id") for e in evidence_index if e.get("evidence_id")}

        result.stats["evidence_count"] = len(valid_ids)

        # Check all evidence_ids fields
        referenced_ids: set[str] = set()
        orphan_refs: list[tuple[str, str]] = []

        def check_evidence_ids(obj: Any, path: str = "") -> None:
            if isinstance(obj, dict):
                if "evidence_ids" in obj:
                    ids = obj["evidence_ids"]
                    if isinstance(ids, list):
                        for eid in ids:
                            referenced_ids.add(eid)
                            if eid not in valid_ids:
                                orphan_refs.append((path + ".evidence_ids", eid))

                for key, value in obj.items():
                    check_evidence_ids(value, f"{path}.{key}")

            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_evidence_ids(item, f"{path}[{i}]")

        check_evidence_ids(case)

        # Report orphan references
        for path, eid in orphan_refs[:10]:  # Limit
            result.issues.append(ValidationIssue(
                severity="error",
                code="ORPHAN_EVIDENCE_REF",
                field=path,
                message=f"Evidence ID '{eid}' not found in evidence index",
            ))

        # Calculate coverage
        if valid_ids:
            coverage = len(referenced_ids & valid_ids) / len(valid_ids)
            result.stats["evidence_coverage"] = coverage
            if coverage < 0.5:
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="LOW_EVIDENCE_COVERAGE",
                    field="evidence_index",
                    message=f"Only {coverage:.0%} of evidence items are referenced",
                ))

    def _validate_simulation_structure(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate simulation structure."""
        sim = case.get("simulation", {})

        # Check initial_state
        initial = sim.get("initial_state", {})
        if not initial.get("chief_complaint"):
            result.issues.append(ValidationIssue(
                severity="error",
                code="MISSING_REQUIRED",
                field="simulation.initial_state.chief_complaint",
                message="Initial state must have chief complaint",
            ))

        if not initial.get("evidence_ids"):
            result.issues.append(ValidationIssue(
                severity="warning",
                code="NO_EVIDENCE",
                field="simulation.initial_state.evidence_ids",
                message="Initial state has no evidence citations",
            ))

        # Check decision_points
        decision_points = sim.get("decision_points", [])
        result.stats["decision_point_count"] = len(decision_points)

        if len(decision_points) < 2:
            result.issues.append(ValidationIssue(
                severity="error",
                code="INSUFFICIENT_DECISION_POINTS",
                field="simulation.decision_points",
                message=f"Need at least 2 decision points, found {len(decision_points)}",
            ))

        # Check end_state
        end_state = sim.get("end_state", {})
        if not end_state.get("patient_outcome"):
            result.issues.append(ValidationIssue(
                severity="error",
                code="MISSING_REQUIRED",
                field="simulation.end_state.patient_outcome",
                message="End state must have patient outcome",
            ))

        if not end_state.get("legal_outcome"):
            result.issues.append(ValidationIssue(
                severity="error",
                code="MISSING_REQUIRED",
                field="simulation.end_state.legal_outcome",
                message="End state must have legal outcome",
            ))

    def _validate_decision_points(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate decision points in detail."""
        sim = case.get("simulation", {})
        decision_points = sim.get("decision_points", [])

        seen_ids: set[str] = set()
        malpractice_points = 0

        for i, dp in enumerate(decision_points):
            path = f"simulation.decision_points[{i}]"

            # Check for unique ID
            dp_id = dp.get("decision_id")
            if dp_id in seen_ids:
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="DUPLICATE_ID",
                    field=f"{path}.decision_id",
                    message=f"Duplicate decision ID: {dp_id}",
                ))
            seen_ids.add(dp_id)

            # Check for options
            options = dp.get("options", [])
            if len(options) < 2:
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="INSUFFICIENT_OPTIONS",
                    field=f"{path}.options",
                    message="Decision point must have at least 2 options",
                ))

            # Check for actual vs expected actions
            actual = dp.get("actual_action_defendant", {})
            expected = dp.get("expected_action_court", {})

            if not actual.get("description"):
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="MISSING_ACTUAL_ACTION",
                    field=f"{path}.actual_action_defendant",
                    message="Missing defendant's actual action",
                ))

            if not expected.get("description"):
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="MISSING_EXPECTED_ACTION",
                    field=f"{path}.expected_action_court",
                    message="Missing court's expected action",
                ))

            # Check evidence for actions
            if not actual.get("evidence_ids"):
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="NO_EVIDENCE",
                    field=f"{path}.actual_action_defendant.evidence_ids",
                    message="Actual action has no evidence citations",
                ))

            if not expected.get("evidence_ids"):
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="NO_EVIDENCE",
                    field=f"{path}.expected_action_court.evidence_ids",
                    message="Expected action has no evidence citations",
                ))

            # Track malpractice points
            if dp.get("is_malpractice_point"):
                malpractice_points += 1

        result.stats["malpractice_points"] = malpractice_points

        if malpractice_points == 0 and decision_points:
            result.issues.append(ValidationIssue(
                severity="warning",
                code="NO_MALPRACTICE_POINT",
                field="simulation.decision_points",
                message="No decision point marked as malpractice point",
            ))

    def _validate_clinical_consistency(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate clinical consistency."""
        # Check ground_truth timeline
        gt = case.get("ground_truth", {})
        timeline = gt.get("factual_timeline", [])

        result.stats["timeline_events"] = len(timeline)

        if not timeline:
            result.issues.append(ValidationIssue(
                severity="warning",
                code="EMPTY_TIMELINE",
                field="ground_truth.factual_timeline",
                message="No factual timeline events",
            ))

        # Check for timeline evidence
        for i, event in enumerate(timeline):
            if not event.get("evidence_ids"):
                result.issues.append(ValidationIssue(
                    severity="warning",
                    code="NO_EVIDENCE",
                    field=f"ground_truth.factual_timeline[{i}].evidence_ids",
                    message=f"Timeline event has no evidence: {event.get('event', '')[:50]}",
                ))

        # Check requestables
        sim = case.get("simulation", {})
        requestables = sim.get("requestables", [])
        result.stats["requestable_count"] = len(requestables)

        # Check for duplicate request IDs
        req_ids: set[str] = set()
        for i, req in enumerate(requestables):
            req_id = req.get("request_id")
            if req_id in req_ids:
                result.issues.append(ValidationIssue(
                    severity="error",
                    code="DUPLICATE_ID",
                    field=f"simulation.requestables[{i}].request_id",
                    message=f"Duplicate request ID: {req_id}",
                ))
            req_ids.add(req_id)

    def _validate_completeness(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Validate overall completeness."""
        missing: list[str] = []

        # Check top-level required fields
        required_fields = [
            "case_id", "source", "jurisdiction", "court",
            "decision_date", "url", "clinical_domain", "outcome_severity",
            "summary", "evidence_index", "simulation", "ground_truth",
            "taxonomy_labels", "quality",
        ]

        for field_name in required_fields:
            if not case.get(field_name):
                missing.append(field_name)

        result.stats["missing_top_level"] = missing

        for field_name in missing:
            result.issues.append(ValidationIssue(
                severity="error",
                code="MISSING_REQUIRED",
                field=field_name,
                message=f"Required field '{field_name}' is missing or empty",
            ))

        # Check summary completeness
        summary = case.get("summary", {})
        if not summary.get("brief"):
            result.issues.append(ValidationIssue(
                severity="warning",
                code="INCOMPLETE_SUMMARY",
                field="summary.brief",
                message="Summary brief is empty",
            ))

        # Check taxonomy
        taxonomy = case.get("taxonomy_labels", {})
        if not taxonomy.get("malpractice_categories"):
            result.issues.append(ValidationIssue(
                severity="error",
                code="MISSING_TAXONOMY",
                field="taxonomy_labels.malpractice_categories",
                message="No malpractice categories specified",
            ))

    def _calculate_scores(
        self,
        case: dict[str, Any],
        result: ValidationResult,
    ) -> None:
        """Calculate quality scores."""
        # Evidence coverage score
        evidence_coverage = result.stats.get("evidence_coverage", 0)
        result.scores["evidence_coverage"] = evidence_coverage

        # Simulation completeness score
        completeness_factors = []

        # Has initial state with content
        sim = case.get("simulation", {})
        initial = sim.get("initial_state", {})
        if initial.get("chief_complaint"):
            completeness_factors.append(1.0)
        else:
            completeness_factors.append(0.0)

        # Has adequate decision points
        dp_count = result.stats.get("decision_point_count", 0)
        if dp_count >= 2:
            completeness_factors.append(1.0)
        elif dp_count == 1:
            completeness_factors.append(0.5)
        else:
            completeness_factors.append(0.0)

        # Has requestables
        req_count = result.stats.get("requestable_count", 0)
        if req_count >= 3:
            completeness_factors.append(1.0)
        elif req_count >= 1:
            completeness_factors.append(0.5)
        else:
            completeness_factors.append(0.0)

        # Has timeline
        timeline_count = result.stats.get("timeline_events", 0)
        if timeline_count >= 5:
            completeness_factors.append(1.0)
        elif timeline_count >= 2:
            completeness_factors.append(0.5)
        else:
            completeness_factors.append(0.0)

        # Has end state
        end_state = sim.get("end_state", {})
        if end_state.get("patient_outcome") and end_state.get("legal_outcome"):
            completeness_factors.append(1.0)
        elif end_state.get("patient_outcome") or end_state.get("legal_outcome"):
            completeness_factors.append(0.5)
        else:
            completeness_factors.append(0.0)

        if completeness_factors:
            result.scores["simulation_completeness"] = sum(completeness_factors) / len(completeness_factors)
        else:
            result.scores["simulation_completeness"] = 0.0

        # Decision point quality score
        error_count = len([i for i in result.issues if i.severity == "error" and "decision" in i.field.lower()])
        warning_count = len([i for i in result.issues if i.severity == "warning" and "decision" in i.field.lower()])

        if dp_count > 0:
            quality = 1.0 - (error_count * 0.2 + warning_count * 0.1)
            result.scores["decision_point_quality"] = max(0.0, quality)
        else:
            result.scores["decision_point_quality"] = 0.0

        # Overall score
        result.scores["overall"] = (
            result.scores.get("evidence_coverage", 0) * 0.3 +
            result.scores.get("simulation_completeness", 0) * 0.4 +
            result.scores.get("decision_point_quality", 0) * 0.3
        )

    def validate_batch(
        self,
        cases: list[CaseSimulation | dict[str, Any]],
    ) -> dict[str, ValidationResult]:
        """Validate multiple cases."""
        results: dict[str, ValidationResult] = {}

        for case in cases:
            if isinstance(case, CaseSimulation):
                case_id = case.case_id
            else:
                case_id = case.get("case_id", "unknown")

            results[case_id] = self.validate(case)

        return results

    def generate_report(
        self,
        results: dict[str, ValidationResult],
    ) -> str:
        """Generate a validation report."""
        lines: list[str] = []
        lines.append("# Case Simulation Validation Report")
        lines.append("")

        total = len(results)
        valid = sum(1 for r in results.values() if r.valid)
        lines.append(f"**Total Cases:** {total}")
        lines.append(f"**Valid:** {valid} ({valid/total*100:.1f}%)")
        lines.append("")

        # Summary by case
        lines.append("## Case Summary")
        lines.append("")
        lines.append("| Case ID | Valid | Errors | Warnings | Overall Score |")
        lines.append("|---------|-------|--------|----------|---------------|")

        for case_id, result in results.items():
            status = "✓" if result.valid else "✗"
            overall = result.scores.get("overall", 0)
            lines.append(
                f"| {case_id} | {status} | {len(result.errors)} | "
                f"{len(result.warnings)} | {overall:.2f} |"
            )

        lines.append("")

        # Common issues
        lines.append("## Common Issues")
        lines.append("")

        issue_counts: dict[str, int] = {}
        for result in results.values():
            for issue in result.issues:
                issue_counts[issue.code] = issue_counts.get(issue.code, 0) + 1

        for code, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- **{code}**: {count} occurrences")

        return "\n".join(lines)
