"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    (tmp_path / "raw").mkdir()
    (tmp_path / "processed").mkdir()
    (tmp_path / "exports").mkdir()
    (tmp_path / "cache").mkdir()
    return tmp_path


@pytest.fixture
def sample_bailii_html():
    """Sample BAILII judgment HTML for testing."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mills v Oxford University Hospitals NHS Foundation Trust [2019] EWHC 936 (QB)</title>
    </head>
    <body>
        <div class="bailii-title">
            <h1>Mills v Oxford University Hospitals NHS Foundation Trust [2019] EWHC 936 (QB)</h1>
        </div>

        <div class="bailii-metadata">
            <p>Case No: QB-2017-003456</p>
            <p>Neutral Citation Number: [2019] EWHC 936 (QB)</p>
            <p>IN THE HIGH COURT OF JUSTICE</p>
            <p>QUEEN'S BENCH DIVISION</p>
            <p>Before: MRS JUSTICE LAMBERT DBE</p>
            <p>Date: 15/04/2019</p>
        </div>

        <h2>Introduction</h2>
        <p>[1] This is a clinical negligence claim brought by the claimant,
        Mrs Susan Mills, against the defendant NHS Trust.</p>

        <h2>The Facts</h2>
        <p>[2] On 15 March 2016, Mrs Mills presented to the emergency department
        at the John Radcliffe Hospital with severe abdominal pain.</p>
        <p>[3] She was 52 years old at the time and had a history of hypertension
        and type 2 diabetes.</p>
        <p>[4] The claimant complained of pain in her right lower abdomen that had
        been present for approximately 6 hours.</p>

        <h2>Clinical Background</h2>
        <p>[5] On examination, the claimant was noted to be tachycardic with a
        heart rate of 102 bpm. Her blood pressure was 140/90.</p>
        <p>[6] Abdominal examination revealed tenderness in the right iliac fossa
        with guarding.</p>
        <p>[7] A full blood count showed an elevated white cell count of 15.2.</p>

        <h2>Expert Evidence</h2>
        <p>[8] Professor James Patterson, consultant general surgeon, gave evidence
        on behalf of the claimant.</p>
        <p>[9] He stated that the standard of care required an urgent CT scan to be
        performed within 2 hours of presentation given the clinical picture.</p>
        <p>[10] Mr David Hughes, consultant surgeon for the defendant, agreed that
        CT scanning was appropriate but contended that ultrasound was a reasonable
        first-line investigation.</p>

        <h2>Informed Consent</h2>
        <p>[11] The claimant alleges that she was not adequately informed of the
        risks of appendicitis if imaging was delayed.</p>
        <p>[12] Dr Smith stated that she discussed the options with the patient.</p>

        <h2>Breach of Duty</h2>
        <p>[13] I find that the defendant breached its duty of care by failing to
        order a CT scan in a timely manner.</p>
        <p>[14] The ultrasound, while not unreasonable as an initial investigation,
        was reported as normal when in fact appendicitis was present.</p>

        <h2>Causation</h2>
        <p>[15] Had a CT scan been performed promptly, it would have revealed the
        inflamed appendix.</p>
        <p>[16] Surgery would have been performed before perforation occurred.</p>

        <h2>Conclusion</h2>
        <p>[17] For these reasons, I find the defendant liable for clinical negligence.</p>
        <p>[18] The claimant is awarded damages of £185,000.</p>
    </body>
    </html>
    """


@pytest.fixture
def minimal_case_simulation():
    """Minimal valid case simulation dictionary."""
    return {
        "schema_version": "1.0.0",
        "case_id": "test-case-001",
        "source": "BAILII",
        "jurisdiction": "UK",
        "court": "EWHC QB",
        "decision_date": "2020-01-15",
        "url": "https://www.bailii.org/ew/cases/EWHC/QB/2020/1.html",
        "clinical_domain": "SURGERY_GENERAL",
        "outcome_severity": "PERMANENT_MODERATE_DISABILITY",
        "summary": {
            "brief": "Clinical negligence case involving delayed diagnosis of appendicitis.",
            "clinical_synopsis": "Patient presented with abdominal pain. Diagnosis delayed due to inadequate imaging.",
            "legal_synopsis": "Defendant found liable for breach of duty in failing to order CT scan.",
        },
        "evidence_index": [
            {
                "evidence_id": "E001",
                "type": "FACTUAL_FINDING",
                "text": "The claimant presented to ED with abdominal pain.",
                "paragraph_ref": "2",
            },
            {
                "evidence_id": "E002",
                "type": "EXPERT_TESTIMONY",
                "text": "Standard of care required CT within 2 hours.",
                "paragraph_ref": "9",
            },
        ],
        "simulation": {
            "initial_state": {
                "chief_complaint": "Severe abdominal pain for 6 hours",
                "history_of_present_illness": "Right lower quadrant pain, sudden onset",
                "past_medical_history": ["Hypertension", "Type 2 diabetes"],
                "evidence_ids": ["E001"],
            },
            "requestables": [
                {
                    "request_id": "R001",
                    "type": "LAB",
                    "name": "Full Blood Count",
                    "available_phase": "workup",
                    "reveal": {
                        "result_summary": "WBC 15.2 (elevated)",
                        "clinical_significance": "Suggests infection/inflammation",
                    },
                    "was_ordered_in_case": True,
                    "evidence_ids": ["E001"],
                },
            ],
            "timeline_phases": [
                {
                    "phase_id": "presentation",
                    "name": "ED Presentation",
                    "description": "Initial presentation to emergency department",
                },
            ],
            "decision_points": [
                {
                    "decision_id": "D001",
                    "phase_id": "workup",
                    "prompt": "What imaging should be ordered?",
                    "action_type": "ORDER_TEST",
                    "options": [
                        {
                            "option_id": "A",
                            "description": "Order CT abdomen",
                            "is_court_endorsed": True,
                        },
                        {
                            "option_id": "B",
                            "description": "Order ultrasound only",
                            "is_defendant_choice": True,
                        },
                    ],
                    "actual_action_defendant": {
                        "description": "Ordered ultrasound only",
                        "evidence_ids": ["E001"],
                    },
                    "expected_action_court": {
                        "description": "CT scan should have been ordered",
                        "evidence_ids": ["E002"],
                    },
                    "scoring_rubric": {"max_score": 10},
                    "is_malpractice_point": True,
                },
                {
                    "decision_id": "D002",
                    "phase_id": "decision",
                    "prompt": "Based on normal ultrasound, what next?",
                    "action_type": "CHOOSE_MANAGEMENT",
                    "options": [
                        {"option_id": "A", "description": "Proceed to CT"},
                        {"option_id": "B", "description": "Observe and discharge"},
                    ],
                    "actual_action_defendant": {
                        "description": "Discharged patient",
                        "evidence_ids": ["E001"],
                    },
                    "expected_action_court": {
                        "description": "Should have ordered CT",
                        "evidence_ids": ["E002"],
                    },
                    "scoring_rubric": {"max_score": 10},
                },
            ],
            "end_state": {
                "patient_outcome": {
                    "description": "Developed peritonitis, required emergency surgery",
                    "severity": "PERMANENT_MODERATE_DISABILITY",
                },
                "legal_outcome": {
                    "verdict": "LIABILITY_FOUND",
                    "damages_awarded": "£185,000",
                },
                "malpractice_determination": {
                    "breach_found": True,
                    "causation_established": True,
                    "point_of_failure": "Failure to order CT scan",
                },
            },
        },
        "ground_truth": {
            "factual_timeline": [
                {
                    "timestamp": "2016-03-15 14:00",
                    "event": "Patient presented to ED",
                    "evidence_ids": ["E001"],
                },
            ],
        },
        "taxonomy_labels": {
            "malpractice_categories": ["DIAGNOSIS_ERROR"],
            "diagnosis_subtypes": ["DELAYED_DIAGNOSIS"],
        },
        "quality": {
            "evidence_coverage_score": 0.8,
            "simulation_completeness": 0.9,
            "validation_passed": True,
        },
    }
