"""Tests for QA validation module."""

import pytest

from casesim.qa import CaseValidator, ValidationResult


class TestCaseValidator:
    """Tests for CaseValidator."""

    @pytest.fixture
    def validator(self):
        return CaseValidator(schema_path=None)  # Skip JSON schema for unit tests

    @pytest.fixture
    def valid_case(self):
        """A valid case for testing."""
        return {
            "schema_version": "1.0.0",
            "case_id": "test-case-001",
            "source": "BAILII",
            "jurisdiction": "UK",
            "court": "EWHC QB",
            "decision_date": "2020-01-15",
            "url": "https://example.com/case",
            "clinical_domain": "SURGERY_GENERAL",
            "outcome_severity": "PERMANENT_MODERATE_DISABILITY",
            "summary": {
                "brief": "Test case summary",
                "clinical_synopsis": "Clinical details",
                "legal_synopsis": "Legal details",
            },
            "evidence_index": [
                {
                    "evidence_id": "E001",
                    "type": "FACTUAL_FINDING",
                    "text": "Test evidence",
                    "paragraph_ref": "1",
                },
                {
                    "evidence_id": "E002",
                    "type": "EXPERT_TESTIMONY",
                    "text": "Expert said this",
                    "paragraph_ref": "5",
                },
            ],
            "simulation": {
                "initial_state": {
                    "chief_complaint": "Test complaint",
                    "evidence_ids": ["E001"],
                },
                "requestables": [
                    {
                        "request_id": "R001",
                        "type": "LAB",
                        "name": "Full Blood Count",
                        "available_phase": "workup",
                        "evidence_ids": ["E001"],
                    }
                ],
                "timeline_phases": [
                    {"phase_id": "presentation", "name": "Presentation", "description": "Initial presentation"},
                ],
                "decision_points": [
                    {
                        "decision_id": "D001",
                        "phase_id": "decision",
                        "prompt": "What to do?",
                        "action_type": "CHOOSE_MANAGEMENT",
                        "options": [
                            {"option_id": "A", "description": "Option A", "is_court_endorsed": True},
                            {"option_id": "B", "description": "Option B", "is_defendant_choice": True},
                        ],
                        "actual_action_defendant": {
                            "description": "Did B",
                            "evidence_ids": ["E001"],
                        },
                        "expected_action_court": {
                            "description": "Should do A",
                            "evidence_ids": ["E002"],
                        },
                        "scoring_rubric": {"max_score": 10},
                        "is_malpractice_point": True,
                    },
                    {
                        "decision_id": "D002",
                        "phase_id": "procedure",
                        "prompt": "Second decision?",
                        "action_type": "SELECT_TECHNIQUE",
                        "options": [
                            {"option_id": "A", "description": "Technique A"},
                            {"option_id": "B", "description": "Technique B"},
                        ],
                        "actual_action_defendant": {
                            "description": "Used A",
                            "evidence_ids": ["E001"],
                        },
                        "expected_action_court": {
                            "description": "Should use B",
                            "evidence_ids": ["E002"],
                        },
                        "scoring_rubric": {"max_score": 10},
                    },
                ],
                "end_state": {
                    "patient_outcome": {
                        "description": "Adverse outcome",
                        "evidence_ids": ["E001"],
                    },
                    "legal_outcome": {
                        "verdict": "LIABILITY_FOUND",
                        "evidence_ids": ["E002"],
                    },
                    "malpractice_determination": {
                        "breach_found": True,
                        "evidence_ids": ["E002"],
                    },
                },
            },
            "ground_truth": {
                "factual_timeline": [
                    {
                        "timestamp": "2020-01-01",
                        "event": "Patient presented",
                        "evidence_ids": ["E001"],
                    },
                    {
                        "timestamp": "2020-01-02",
                        "event": "Surgery performed",
                        "evidence_ids": ["E001"],
                    },
                ],
                "tests_performed": [
                    {"test_name": "FBC", "result": "Normal", "evidence_ids": ["E001"]},
                ],
            },
            "taxonomy_labels": {
                "malpractice_categories": ["DIAGNOSIS_ERROR"],
            },
            "quality": {
                "evidence_coverage_score": 0.8,
                "validation_passed": True,
            },
        }

    def test_valid_case_passes(self, validator, valid_case):
        result = validator.validate(valid_case, strict=False)
        assert len(result.errors) == 0

    def test_missing_chief_complaint_fails(self, validator, valid_case):
        valid_case["simulation"]["initial_state"]["chief_complaint"] = ""
        result = validator.validate(valid_case)
        assert any("chief_complaint" in str(i.field) for i in result.issues)

    def test_insufficient_decision_points_fails(self, validator, valid_case):
        valid_case["simulation"]["decision_points"] = [
            valid_case["simulation"]["decision_points"][0]
        ]
        result = validator.validate(valid_case)
        assert any("INSUFFICIENT_DECISION_POINTS" in i.code for i in result.issues)

    def test_orphan_evidence_reference_detected(self, validator, valid_case):
        valid_case["simulation"]["initial_state"]["evidence_ids"] = ["E999"]
        result = validator.validate(valid_case)
        assert any("ORPHAN_EVIDENCE_REF" in i.code for i in result.issues)

    def test_duplicate_decision_id_detected(self, validator, valid_case):
        valid_case["simulation"]["decision_points"][1]["decision_id"] = "D001"
        result = validator.validate(valid_case)
        assert any("DUPLICATE_ID" in i.code for i in result.issues)

    def test_missing_taxonomy_fails(self, validator, valid_case):
        valid_case["taxonomy_labels"]["malpractice_categories"] = []
        result = validator.validate(valid_case)
        assert any("MISSING_TAXONOMY" in i.code for i in result.issues)

    def test_quality_scores_calculated(self, validator, valid_case):
        result = validator.validate(valid_case)
        assert "evidence_coverage" in result.scores
        assert "simulation_completeness" in result.scores
        assert "decision_point_quality" in result.scores

    def test_no_malpractice_point_warning(self, validator, valid_case):
        for dp in valid_case["simulation"]["decision_points"]:
            dp["is_malpractice_point"] = False
        result = validator.validate(valid_case)
        assert any("NO_MALPRACTICE_POINT" in i.code for i in result.issues)


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_errors_property(self):
        from casesim.qa.validator import ValidationIssue
        result = ValidationResult(
            valid=False,
            issues=[
                ValidationIssue(severity="error", code="TEST", field="x", message="Error"),
                ValidationIssue(severity="warning", code="TEST", field="y", message="Warning"),
            ],
        )
        assert len(result.errors) == 1
        assert len(result.warnings) == 1
