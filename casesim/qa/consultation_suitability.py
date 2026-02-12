"""
Consultation Suitability Validator

Evaluates whether extracted malpractice cases are suitable for use in
a medical consultation simulation where doctors interact with patients.

A case is suitable if:
1. It represents an actual patient-doctor encounter (not institutional/legal)
2. It has sufficient clinical data for realistic patient simulation
3. The decision points involve clinical decisions (not administrative/legal)
4. There's enough information for the patient to respond to questions
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from enum import Enum


class SuitabilityGrade(str, Enum):
    """Overall suitability grade for consultation simulation."""
    EXCELLENT = "excellent"  # Ready to use, rich clinical data
    GOOD = "good"           # Usable with minor gaps
    MARGINAL = "marginal"   # Usable but limited
    POOR = "poor"           # Major issues, not recommended
    UNSUITABLE = "unsuitable"  # Not a medical consultation case


@dataclass
class SuitabilityIssue:
    """A specific issue affecting suitability."""
    category: str  # clinical_data, decision_point, case_type, patient_info
    severity: str  # critical, major, minor
    description: str
    field_path: str = ""


@dataclass
class SuitabilityResult:
    """Result of consultation suitability assessment."""
    case_id: str
    grade: SuitabilityGrade
    overall_score: float  # 0-100

    # Component scores (0-100)
    clinical_data_score: float = 0.0
    patient_info_score: float = 0.0
    decision_point_score: float = 0.0
    case_type_score: float = 0.0

    # Detailed findings
    issues: list[SuitabilityIssue] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)

    # Flags
    is_medical_case: bool = True
    has_patient_encounter: bool = True
    has_clinical_decision: bool = True

    # Recommendation
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "grade": self.grade.value,
            "overall_score": round(self.overall_score, 1),
            "scores": {
                "clinical_data": round(self.clinical_data_score, 1),
                "patient_info": round(self.patient_info_score, 1),
                "decision_point": round(self.decision_point_score, 1),
                "case_type": round(self.case_type_score, 1),
            },
            "is_medical_case": self.is_medical_case,
            "has_patient_encounter": self.has_patient_encounter,
            "has_clinical_decision": self.has_clinical_decision,
            "issues_count": len(self.issues),
            "critical_issues": len([i for i in self.issues if i.severity == "critical"]),
            "recommendation": self.recommendation,
        }


# Placeholder phrases that indicate missing data
PLACEHOLDER_PHRASES = [
    "not documented in judgment",
    "not documented",
    "not specified",
    "unknown",
    "n/a",
    "none",
    "examination mentioned but findings not specified",
]

# Non-medical case indicators (in case name, summary, or evidence)
# These indicate the case is NOT about medical consultation
NON_MEDICAL_INDICATORS = [
    # Institutional/administrative
    "social services",
    "local authority",
    "county council",
    "child protection",
    "care order",
    "care proceedings",

    # Criminal matters
    "false imprisonment",
    "sexual assault",
    "sexual abuse",
    "buggery",
    "rape",
    "criminal conviction",

    # Other legal
    "defamation",
    "contempt of court",
    "fraud",
    "perjury",
    "strike out",

    # Premises/travel
    "tour operator",
    "package holiday",
    "hotel accident",
    "resort accident",
    "premises liability",

    # Employment
    "employment tribunal",
    "workplace accident",
    "industrial injury",
]

# Phrases that look like non-medical but are OK in medical context
FALSE_POSITIVE_OVERRIDES = [
    "surgical",
    "diagnosis",
    "treatment",
    "patient",
    "clinical",
    "hospital",
    "doctor",
    "nurse",
    "medical negligence",
    "clinical negligence",
]

# Clinical action types that indicate real medical decisions
CLINICAL_ACTION_TYPES = [
    "ORDER_TEST",
    "CHOOSE_MANAGEMENT",
    "PRESCRIBE",
    "REFER",
    "PROCEDURE",
    "ADMIT",
    "DISCHARGE",
    "DISCLOSE_ALTERNATIVES",  # informed consent - can be clinical
]

# Administrative/legal action types that don't fit consultation sim
NON_CLINICAL_ACTION_TYPES = [
    "ESCALATE_CARE",  # Often administrative in non-medical contexts
]


def is_placeholder(value: Any) -> bool:
    """Check if a value is a placeholder indicating missing data."""
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    value_lower = value.lower().strip()
    return any(phrase in value_lower for phrase in PLACEHOLDER_PHRASES)


def has_non_medical_indicators(text: str) -> bool:
    """Check if text contains indicators of non-medical case.

    Returns True only if non-medical indicators are present AND
    there are no strong medical indicators that override them.
    """
    if not text:
        return False
    text_lower = text.lower()

    # First check for non-medical indicators
    has_negative = any(indicator in text_lower for indicator in NON_MEDICAL_INDICATORS)
    if not has_negative:
        return False

    # If we found negative indicators, check if medical context overrides them
    medical_indicator_count = sum(1 for phrase in FALSE_POSITIVE_OVERRIDES if phrase in text_lower)
    if medical_indicator_count >= 2:
        # Strong medical context - likely a medical case with some legal language
        return False

    return True


class ConsultationSuitabilityValidator:
    """Validates cases for suitability in medical consultation simulation."""

    def __init__(self):
        pass

    def validate(self, case: dict) -> SuitabilityResult:
        """Assess a case's suitability for consultation simulation."""
        case_id = case.get("case_id", "unknown")
        result = SuitabilityResult(case_id=case_id, grade=SuitabilityGrade.GOOD, overall_score=0)

        # Check if case is explicitly marked as not testable (from extraction)
        sim = case.get("simulation", {})
        if sim.get("testable") is False:
            testable_reason = sim.get("testable_reason", "Case marked as not testable during extraction")
            result.issues.append(SuitabilityIssue(
                category="case_type",
                severity="critical",
                description=f"Case marked untestable: {testable_reason[:200]}",
                field_path="simulation.testable"
            ))
            result.is_medical_case = False
            result.has_clinical_decision = False
            result.grade = SuitabilityGrade.UNSUITABLE
            result.overall_score = 0
            result.case_type_score = 0
            result.recommendation = f"EXCLUDE: {testable_reason[:100]}"
            return result

        # Run all assessments
        self._assess_case_type(case, result)
        self._assess_clinical_data(case, result)
        self._assess_patient_info(case, result)
        self._assess_decision_points(case, result)

        # Calculate overall score and grade
        self._calculate_overall(result)

        # Generate recommendation
        self._generate_recommendation(result)

        return result

    def _assess_case_type(self, case: dict, result: SuitabilityResult) -> None:
        """Assess whether this is actually a medical consultation case."""
        score = 100.0

        # Check case name
        case_name = case.get("case_name", "")
        if has_non_medical_indicators(case_name):
            result.issues.append(SuitabilityIssue(
                category="case_type",
                severity="critical",
                description=f"Case name suggests non-medical case: '{case_name[:100]}'",
                field_path="case_name"
            ))
            score -= 50
            result.is_medical_case = False

        # Check summary
        summary = case.get("summary", {})
        brief = summary.get("brief", "")
        if has_non_medical_indicators(brief):
            result.issues.append(SuitabilityIssue(
                category="case_type",
                severity="critical",
                description="Summary contains non-medical indicators",
                field_path="summary.brief"
            ))
            score -= 30
            result.is_medical_case = False

        # Check evidence for non-medical content
        evidence_index = case.get("evidence_index", [])
        non_medical_evidence_count = 0
        for ev in evidence_index[:20]:  # Check first 20
            text = ev.get("text", "")
            if has_non_medical_indicators(text):
                non_medical_evidence_count += 1

        if non_medical_evidence_count > 5:
            result.issues.append(SuitabilityIssue(
                category="case_type",
                severity="major",
                description=f"{non_medical_evidence_count} evidence items contain non-medical content",
                field_path="evidence_index"
            ))
            score -= 20

        # Check clinical domain mismatch
        clinical_domain = case.get("clinical_domain", "")
        sim = case.get("simulation", {})
        initial = sim.get("initial_state", {})
        chief_complaint = initial.get("chief_complaint", "")

        # If domain says surgery but no surgical context
        if clinical_domain and is_placeholder(chief_complaint):
            result.issues.append(SuitabilityIssue(
                category="case_type",
                severity="major",
                description=f"Domain is '{clinical_domain}' but no chief complaint to verify",
                field_path="clinical_domain"
            ))
            score -= 15

        # Check if there's actually a defendant who is a healthcare provider
        ground_truth = case.get("ground_truth", {})
        allegations = ground_truth.get("allegations", [])
        has_medical_defendant = False
        medical_defendant_terms = [
            "doctor", "hospital", "trust", "nhs", "gp", "surgeon", "nurse",
            "consultant", "physician", "clinician", "practitioner", "midwife",
            "obstetrician", "anaesthetist", "registrar"
        ]
        for allegation in allegations:
            against = allegation.get("against", "").lower()
            if any(term in against for term in medical_defendant_terms):
                has_medical_defendant = True
                break

        # Also check if clinical_domain suggests medical context
        clinical_domain = case.get("clinical_domain", "")
        has_clinical_domain = clinical_domain and not is_placeholder(clinical_domain)

        if not has_medical_defendant and allegations:
            # Only flag as issue if we also lack clinical domain
            if not has_clinical_domain:
                result.issues.append(SuitabilityIssue(
                    category="case_type",
                    severity="major",
                    description="No medical professional identified as defendant",
                    field_path="ground_truth.allegations"
                ))
                score -= 15
            # Don't set has_patient_encounter=False here - that's too harsh

        result.case_type_score = max(0, score)

        if score >= 80:
            result.strengths.append("Clearly a medical malpractice case")

    def _assess_clinical_data(self, case: dict, result: SuitabilityResult) -> None:
        """Assess richness of clinical data for simulation."""
        score = 0.0
        max_score = 100.0

        sim = case.get("simulation", {})
        initial = sim.get("initial_state", {})

        # Chief complaint (critical - 25 points)
        chief_complaint = initial.get("chief_complaint", "")
        if chief_complaint and not is_placeholder(chief_complaint):
            score += 25
            result.strengths.append(f"Has chief complaint: '{chief_complaint[:50]}'")
        else:
            result.issues.append(SuitabilityIssue(
                category="clinical_data",
                severity="critical",
                description="Missing or placeholder chief complaint",
                field_path="simulation.initial_state.chief_complaint"
            ))

        # History of present illness (important - 20 points)
        hpi = initial.get("history_of_present_illness", "")
        if hpi and not is_placeholder(hpi):
            score += 20
            if len(hpi) > 100:
                score += 5  # Bonus for detailed HPI
        else:
            result.issues.append(SuitabilityIssue(
                category="clinical_data",
                severity="major",
                description="Missing or placeholder HPI",
                field_path="simulation.initial_state.history_of_present_illness"
            ))

        # Vital signs (important - 15 points)
        exam = initial.get("physical_examination", {})
        vitals = exam.get("vital_signs", {})
        vital_count = 0
        for key, value in vitals.items():
            if value and not is_placeholder(str(value)):
                vital_count += 1

        if vital_count >= 3:
            score += 15
            result.strengths.append(f"Has {vital_count} vital signs documented")
        elif vital_count >= 1:
            score += 8
        else:
            result.issues.append(SuitabilityIssue(
                category="clinical_data",
                severity="major",
                description="No vital signs documented",
                field_path="simulation.initial_state.physical_examination.vital_signs"
            ))

        # Physical examination findings (important - 15 points)
        focused_exam = exam.get("focused_exam", {})
        exam_findings = 0
        for key, value in focused_exam.items():
            if value and not is_placeholder(str(value)):
                exam_findings += 1

        general = exam.get("general_appearance")
        if general and not is_placeholder(str(general)):
            exam_findings += 1

        if exam_findings >= 2:
            score += 15
        elif exam_findings >= 1:
            score += 8
        else:
            result.issues.append(SuitabilityIssue(
                category="clinical_data",
                severity="major",
                description="No physical examination findings",
                field_path="simulation.initial_state.physical_examination"
            ))

        # Requestables/tests (important - 15 points)
        requestables = sim.get("requestables", [])
        useful_requestables = 0
        for req in requestables:
            reveal = req.get("reveal", {})
            result_summary = reveal.get("result_summary", "")
            if result_summary and not is_placeholder(result_summary):
                useful_requestables += 1

        if useful_requestables >= 2:
            score += 15
            result.strengths.append(f"Has {useful_requestables} tests with results")
        elif useful_requestables >= 1:
            score += 8
        else:
            result.issues.append(SuitabilityIssue(
                category="clinical_data",
                severity="minor",
                description="No test results available for simulation",
                field_path="simulation.requestables"
            ))

        # Timeline with timestamps (nice to have - 10 points)
        ground_truth = case.get("ground_truth", {})
        timeline = ground_truth.get("factual_timeline", [])
        timestamped_events = 0
        for event in timeline:
            ts = event.get("timestamp", "")
            if ts and not is_placeholder(ts):
                timestamped_events += 1

        if timestamped_events >= 3:
            score += 10
        elif timestamped_events >= 1:
            score += 5

        result.clinical_data_score = min(score, max_score)

    def _assess_patient_info(self, case: dict, result: SuitabilityResult) -> None:
        """Assess whether there's enough patient info for simulation."""
        score = 0.0

        sim = case.get("simulation", {})
        initial = sim.get("initial_state", {})

        # Demographics (20 points)
        demo = initial.get("patient_demographics", {})
        age = demo.get("age_at_presentation", "")
        sex = demo.get("sex", "")

        if age and not is_placeholder(age):
            score += 10
        else:
            result.issues.append(SuitabilityIssue(
                category="patient_info",
                severity="minor",
                description="Patient age not documented",
                field_path="simulation.initial_state.patient_demographics.age_at_presentation"
            ))

        if sex and sex != "unknown" and not is_placeholder(sex):
            score += 10

        # Medical history (20 points)
        pmh = initial.get("past_medical_history", [])
        if pmh and isinstance(pmh, list):
            real_pmh = [p for p in pmh if p and not is_placeholder(p)]
            if real_pmh:
                score += 20
                result.strengths.append(f"Has {len(real_pmh)} PMH items")

        # Medications (15 points)
        meds = initial.get("medications", [])
        if meds and isinstance(meds, list):
            real_meds = [m for m in meds if m and not is_placeholder(m)]
            if real_meds:
                score += 15

        # Allergies (10 points)
        allergies = initial.get("allergies", [])
        if allergies and isinstance(allergies, list):
            real_allergies = [a for a in allergies if a and not is_placeholder(a)]
            if real_allergies:
                score += 10

        # Social history (15 points)
        social = demo.get("relevant_social_history", "")
        if social and not is_placeholder(social):
            score += 15

        # Initial working problem (20 points) - helps patient respond appropriately
        working_problem = initial.get("initial_working_problem", "")
        if working_problem and not is_placeholder(working_problem):
            score += 20
        else:
            result.issues.append(SuitabilityIssue(
                category="patient_info",
                severity="minor",
                description="No initial working problem/diagnosis",
                field_path="simulation.initial_state.initial_working_problem"
            ))

        result.patient_info_score = min(score, 100)

    def _assess_decision_points(self, case: dict, result: SuitabilityResult) -> None:
        """Assess whether decision points are suitable for consultation."""
        score = 100.0

        sim = case.get("simulation", {})
        decision_points = sim.get("decision_points", [])

        if not decision_points:
            result.issues.append(SuitabilityIssue(
                category="decision_point",
                severity="critical",
                description="No decision points defined",
                field_path="simulation.decision_points"
            ))
            result.decision_point_score = 0
            result.has_clinical_decision = False
            return

        clinical_decisions = 0
        non_clinical_decisions = 0

        for i, dp in enumerate(decision_points):
            action_type = dp.get("action_type", "")
            prompt = dp.get("prompt", "")
            context = dp.get("clinical_context", "")

            # Check if action type is clinical
            if action_type in CLINICAL_ACTION_TYPES:
                clinical_decisions += 1
            elif has_non_medical_indicators(prompt) or has_non_medical_indicators(context):
                non_clinical_decisions += 1
                result.issues.append(SuitabilityIssue(
                    category="decision_point",
                    severity="major",
                    description=f"Decision point {i+1} appears non-clinical: '{prompt[:60]}'",
                    field_path=f"simulation.decision_points[{i}]"
                ))
                score -= 25
            else:
                # Neutral - might be clinical
                clinical_decisions += 0.5

            # Check if there's a clear correct answer
            expected = dp.get("expected_action_court", {})
            if not expected.get("description"):
                result.issues.append(SuitabilityIssue(
                    category="decision_point",
                    severity="minor",
                    description=f"Decision point {i+1} missing expected action",
                    field_path=f"simulation.decision_points[{i}].expected_action_court"
                ))
                score -= 10

            # Check if options are meaningful
            options = dp.get("options", [])
            if len(options) < 2:
                result.issues.append(SuitabilityIssue(
                    category="decision_point",
                    severity="major",
                    description=f"Decision point {i+1} has fewer than 2 options",
                    field_path=f"simulation.decision_points[{i}].options"
                ))
                score -= 15

        if clinical_decisions == 0:
            result.issues.append(SuitabilityIssue(
                category="decision_point",
                severity="critical",
                description="No clinical decision points found",
                field_path="simulation.decision_points"
            ))
            result.has_clinical_decision = False
            score -= 30
        elif clinical_decisions >= len(decision_points):
            result.strengths.append(f"All {len(decision_points)} decision points are clinical")

        result.decision_point_score = max(0, score)

    def _calculate_overall(self, result: SuitabilityResult) -> None:
        """Calculate overall score and grade."""
        # Weighted average
        weights = {
            "case_type": 0.30,      # Most important - is it even a medical case?
            "clinical_data": 0.30,   # Rich clinical data for simulation
            "patient_info": 0.20,    # Patient info for questions
            "decision_point": 0.20,  # Meaningful decisions
        }

        result.overall_score = (
            result.case_type_score * weights["case_type"] +
            result.clinical_data_score * weights["clinical_data"] +
            result.patient_info_score * weights["patient_info"] +
            result.decision_point_score * weights["decision_point"]
        )

        # Determine grade based on score and critical issues
        critical_issues = len([i for i in result.issues if i.severity == "critical"])

        # Strong clinical data can override uncertainty about case type
        has_strong_clinical_data = (
            result.clinical_data_score >= 60 and
            result.patient_info_score >= 30
        )

        # Automatic unsuitable only if clearly NOT a medical case
        if not result.is_medical_case and not has_strong_clinical_data:
            result.grade = SuitabilityGrade.UNSUITABLE
        elif not result.has_clinical_decision:
            result.grade = SuitabilityGrade.POOR
        elif critical_issues >= 3:  # More lenient - was 2
            result.grade = SuitabilityGrade.POOR
        elif result.overall_score >= 70 and critical_issues == 0:  # Was 75
            result.grade = SuitabilityGrade.EXCELLENT
        elif result.overall_score >= 50 and critical_issues <= 1:  # Was 55
            result.grade = SuitabilityGrade.GOOD
        elif result.overall_score >= 35:
            result.grade = SuitabilityGrade.MARGINAL
        else:
            result.grade = SuitabilityGrade.POOR

    def _generate_recommendation(self, result: SuitabilityResult) -> None:
        """Generate actionable recommendation."""
        if result.grade == SuitabilityGrade.EXCELLENT:
            result.recommendation = "INCLUDE: Case is well-suited for consultation simulation."
        elif result.grade == SuitabilityGrade.GOOD:
            result.recommendation = "INCLUDE: Case is suitable with minor gaps in data."
        elif result.grade == SuitabilityGrade.MARGINAL:
            issues = ", ".join([i.description[:40] for i in result.issues if i.severity in ["critical", "major"]][:2])
            result.recommendation = f"REVIEW: Marginal suitability. Issues: {issues}"
        elif result.grade == SuitabilityGrade.POOR:
            result.recommendation = "EXCLUDE: Major issues prevent realistic simulation."
        else:  # UNSUITABLE
            result.recommendation = "EXCLUDE: Not a medical consultation case."

    def validate_batch(self, cases: list[dict]) -> list[SuitabilityResult]:
        """Validate multiple cases."""
        return [self.validate(case) for case in cases]

    def generate_report(self, results: list[SuitabilityResult]) -> str:
        """Generate a comprehensive report."""
        lines = []
        lines.append("=" * 80)
        lines.append("CONSULTATION SUITABILITY REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Summary statistics
        total = len(results)
        by_grade = {}
        for r in results:
            by_grade[r.grade] = by_grade.get(r.grade, 0) + 1

        lines.append("SUMMARY")
        lines.append("-" * 40)
        lines.append(f"Total cases analyzed: {total}")
        lines.append("")
        lines.append("Grade distribution:")
        for grade in SuitabilityGrade:
            count = by_grade.get(grade, 0)
            pct = count / total * 100 if total > 0 else 0
            bar = "█" * int(pct / 5)
            lines.append(f"  {grade.value:12} : {count:3} ({pct:5.1f}%) {bar}")

        lines.append("")

        # Usability summary
        usable = sum(1 for r in results if r.grade in [SuitabilityGrade.EXCELLENT, SuitabilityGrade.GOOD])
        marginal = by_grade.get(SuitabilityGrade.MARGINAL, 0)
        unusable = sum(1 for r in results if r.grade in [SuitabilityGrade.POOR, SuitabilityGrade.UNSUITABLE])

        lines.append(f"Ready to use:     {usable:3} cases ({usable/total*100:.1f}%)")
        lines.append(f"Need review:      {marginal:3} cases ({marginal/total*100:.1f}%)")
        lines.append(f"Not recommended:  {unusable:3} cases ({unusable/total*100:.1f}%)")
        lines.append("")

        # Common issues
        all_issues = []
        for r in results:
            all_issues.extend(r.issues)

        issue_counts = {}
        for issue in all_issues:
            key = f"{issue.category}:{issue.severity}"
            issue_counts[key] = issue_counts.get(key, 0) + 1

        lines.append("COMMON ISSUES")
        lines.append("-" * 40)
        for key, count in sorted(issue_counts.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"  {key:30} : {count} cases")
        lines.append("")

        # Detailed results by grade
        for grade in [SuitabilityGrade.UNSUITABLE, SuitabilityGrade.POOR,
                      SuitabilityGrade.MARGINAL, SuitabilityGrade.GOOD, SuitabilityGrade.EXCELLENT]:
            grade_results = [r for r in results if r.grade == grade]
            if not grade_results:
                continue

            lines.append("")
            lines.append(f"{grade.value.upper()} CASES ({len(grade_results)})")
            lines.append("-" * 40)

            for r in sorted(grade_results, key=lambda x: x.overall_score):
                lines.append(f"\n  {r.case_id}")
                lines.append(f"    Score: {r.overall_score:.1f}/100")
                lines.append(f"    Components: clinical={r.clinical_data_score:.0f}, patient={r.patient_info_score:.0f}, "
                           f"decision={r.decision_point_score:.0f}, type={r.case_type_score:.0f}")

                if r.strengths:
                    lines.append(f"    Strengths: {'; '.join(r.strengths[:2])}")

                critical = [i for i in r.issues if i.severity == "critical"]
                if critical:
                    lines.append(f"    Critical: {critical[0].description[:60]}")

                lines.append(f"    → {r.recommendation}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        return "\n".join(lines)


def load_cases_from_directory(directory: Path) -> list[dict]:
    """Load all JSON case files from a directory."""
    cases = []
    for json_file in sorted(directory.glob("*.json")):
        try:
            with open(json_file) as f:
                case = json.load(f)
                cases.append(case)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return cases


def main():
    """Run suitability validation on all cases."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate cases for consultation suitability")
    parser.add_argument("--input", "-i", type=str, default="data/processed",
                       help="Input directory containing case JSON files")
    parser.add_argument("--output", "-o", type=str, default=None,
                       help="Output file for report (default: stdout)")
    parser.add_argument("--json", "-j", type=str, default=None,
                       help="Output file for JSON results")
    parser.add_argument("--min-grade", type=str, default=None,
                       choices=["excellent", "good", "marginal", "poor", "unsuitable"],
                       help="Only show cases at or above this grade")

    args = parser.parse_args()

    # Load cases
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return 1

    print(f"Loading cases from {input_dir}...")
    cases = load_cases_from_directory(input_dir)
    print(f"Loaded {len(cases)} cases")

    # Validate
    validator = ConsultationSuitabilityValidator()
    results = validator.validate_batch(cases)

    # Filter by grade if requested
    if args.min_grade:
        grade_order = list(SuitabilityGrade)
        min_idx = grade_order.index(SuitabilityGrade(args.min_grade))
        results = [r for r in results if grade_order.index(r.grade) <= min_idx]

    # Generate report
    report = validator.generate_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)

    # Write JSON results if requested
    if args.json:
        json_results = {
            "summary": {
                "total": len(results),
                "by_grade": {grade.value: len([r for r in results if r.grade == grade])
                            for grade in SuitabilityGrade},
            },
            "cases": [r.to_dict() for r in results]
        }
        with open(args.json, "w") as f:
            json.dump(json_results, f, indent=2)
        print(f"JSON results written to {args.json}")

    # Return exit code based on usable cases
    usable = len([r for r in results if r.grade in [SuitabilityGrade.EXCELLENT, SuitabilityGrade.GOOD]])
    return 0 if usable > 0 else 1


if __name__ == "__main__":
    exit(main())
