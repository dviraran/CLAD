"""Tests for schema models."""

import pytest
from datetime import date
from pydantic import ValidationError

from casesim.schemas import (
    CaseSimulation,
    CaseSummary,
    ClinicalDomain,
    DecisionPoint,
    DiscoveryRecord,
    EndState,
    EvidenceItem,
    EvidenceType,
    GroundTruth,
    InitialState,
    Jurisdiction,
    LegalOutcome,
    MalpracticeCategory,
    MalpracticeDetermination,
    OutcomeSeverity,
    PatientOutcome,
    QualityMetrics,
    Simulation,
    Source,
    TaxonomyLabels,
)


class TestEvidenceItem:
    """Tests for EvidenceItem model."""

    def test_valid_evidence_item(self):
        item = EvidenceItem(
            evidence_id="E001",
            type=EvidenceType.FACTUAL_FINDING,
            text="The patient presented with severe abdominal pain.",
            paragraph_ref="15",
        )
        assert item.evidence_id == "E001"
        assert item.type == EvidenceType.FACTUAL_FINDING

    def test_invalid_evidence_id_format(self):
        with pytest.raises(ValidationError):
            EvidenceItem(
                evidence_id="invalid",
                type=EvidenceType.FACTUAL_FINDING,
                text="Test",
                paragraph_ref="1",
            )


class TestInitialState:
    """Tests for InitialState model."""

    def test_valid_initial_state(self):
        state = InitialState(
            chief_complaint="Severe abdominal pain",
            evidence_ids=["E001", "E002"],
        )
        assert state.chief_complaint == "Severe abdominal pain"
        assert len(state.evidence_ids) == 2

    def test_with_full_data(self):
        state = InitialState(
            chief_complaint="Chest pain",
            history_of_present_illness="Patient reports 2 hours of chest pain",
            past_medical_history=["Hypertension", "Diabetes"],
            medications=["Metformin"],
            allergies=["Penicillin"],
            evidence_ids=["E001"],
        )
        assert len(state.past_medical_history) == 2


class TestDecisionPoint:
    """Tests for DecisionPoint model."""

    def test_valid_decision_point(self):
        dp = DecisionPoint(
            decision_id="D001",
            phase_id="workup",
            prompt="What investigation should be ordered?",
            action_type="ORDER_TEST",
            options=[
                {"option_id": "A", "description": "Order CT scan"},
                {"option_id": "B", "description": "Order ultrasound"},
            ],
            actual_action_defendant={
                "description": "Ordered ultrasound only",
                "evidence_ids": ["E010"],
            },
            expected_action_court={
                "description": "Should have ordered CT scan",
                "evidence_ids": ["E020"],
            },
            scoring_rubric={"max_score": 10},
        )
        assert dp.decision_id == "D001"
        assert len(dp.options) == 2

    def test_minimum_options_required(self):
        with pytest.raises(ValidationError):
            DecisionPoint(
                decision_id="D001",
                phase_id="workup",
                prompt="Test",
                action_type="ORDER_TEST",
                options=[{"option_id": "A", "description": "Only one option"}],
                actual_action_defendant={
                    "description": "Test",
                    "evidence_ids": [],
                },
                expected_action_court={
                    "description": "Test",
                    "evidence_ids": [],
                },
                scoring_rubric={"max_score": 10},
            )


class TestDiscoveryRecord:
    """Tests for DiscoveryRecord model."""

    def test_valid_discovery_record(self):
        record = DiscoveryRecord(
            case_id="bailii-ewhc-2019-123",
            source=Source.BAILII,
            jurisdiction=Jurisdiction.UK,
            title="Smith v NHS Trust",
            url="https://www.bailii.org/ew/cases/EWHC/QB/2019/123.html",
            discovery_methods=["keyword_search"],
        )
        assert record.source == Source.BAILII
        assert record.jurisdiction == Jurisdiction.UK


class TestCaseSimulation:
    """Tests for full CaseSimulation model."""

    @pytest.fixture
    def minimal_case(self):
        """Create a minimal valid case simulation."""
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
                }
            ],
            "simulation": {
                "initial_state": {
                    "chief_complaint": "Test complaint",
                    "evidence_ids": ["E001"],
                },
                "requestables": [],
                "timeline_phases": [],
                "decision_points": [
                    {
                        "decision_id": "D001",
                        "phase_id": "decision",
                        "prompt": "What to do?",
                        "action_type": "CHOOSE_MANAGEMENT",
                        "options": [
                            {"option_id": "A", "description": "Option A"},
                            {"option_id": "B", "description": "Option B"},
                        ],
                        "actual_action_defendant": {
                            "description": "Did A",
                            "evidence_ids": ["E001"],
                        },
                        "expected_action_court": {
                            "description": "Should do B",
                            "evidence_ids": ["E001"],
                        },
                        "scoring_rubric": {"max_score": 10},
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
                            "evidence_ids": ["E001"],
                        },
                        "scoring_rubric": {"max_score": 10},
                    },
                ],
                "end_state": {
                    "patient_outcome": {"description": "Adverse outcome"},
                    "legal_outcome": {"verdict": "LIABILITY_FOUND"},
                    "malpractice_determination": {"breach_found": True},
                },
            },
            "ground_truth": {
                "factual_timeline": [
                    {
                        "timestamp": "2020-01-01",
                        "event": "Patient presented",
                        "evidence_ids": ["E001"],
                    }
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

    def test_valid_case_simulation(self, minimal_case):
        case = CaseSimulation(**minimal_case)
        assert case.case_id == "test-case-001"
        assert len(case.simulation.decision_points) == 2

    def test_case_serialization(self, minimal_case):
        case = CaseSimulation(**minimal_case)
        json_data = case.model_dump(mode="json")
        assert json_data["case_id"] == "test-case-001"

    def test_case_id_validation(self, minimal_case):
        minimal_case["case_id"] = "Invalid ID With Spaces"
        with pytest.raises(ValidationError):
            CaseSimulation(**minimal_case)
