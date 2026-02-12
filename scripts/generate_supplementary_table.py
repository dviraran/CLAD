#!/usr/bin/env python3
"""Generate supplementary_cases_table.csv from processed case JSONs."""

import json
import csv
from pathlib import Path

# Paths
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
OUTPUT_FILE = Path(__file__).parent.parent / "paper" / "supplementary_cases_table.csv"

# Excluded malpractice types (non-clinical, administrative)
EXCLUDED_MALPRACTICE_TYPES = {
    "professional_boundaries_violation",
    "documentation_failure",
    "equipment_or_facility_safety",
    "care_planning_error"
}

# Source mapping
SOURCE_MAP = {
    "BAILII": "BAILII (UK)",
    "CourtListener": "CourtListener (US)",
    "NZLII": "NZLII (NZ)",
    "AustLII": "AustLII (AU)",
    "CanLII": "CanLII (CA)"
}

# Malpractice type mapping to display names
MALPRACTICE_MAP = {
    "DIAGNOSIS_DELAY_OR_ERROR": "Diagnosis Delay Or Error",
    "TEST_SELECTION_ERROR": "Test Selection Error",
    "REFERRAL_FAILURE": "Referral Failure",
    "SURGICAL_TECHNIQUE_ERROR": "Surgical Technique Error",
    "INFORMED_CONSENT": "Informed Consent",
    "TREATMENT_TIMING_ERROR": "Treatment Timing Error",
    "MEDICATION_SELECTION_ERROR": "Medication Selection Error",
    "DISCHARGE_DISPOSITION_ERROR": "Discharge Disposition Error",
    "MONITORING_OR_ESCALATION_FAILURE": "Monitoring Or Escalation Failure",
    "CARE_MANAGEMENT_ERROR": "Care Management Error",
    "OTHER": "Other"
}

def get_malpractice_type(case_data):
    """Extract malpractice type from case data."""
    # Try taxonomy_labels first
    if "taxonomy_labels" in case_data:
        cats = case_data["taxonomy_labels"].get("malpractice_categories", [])
        if cats:
            return MALPRACTICE_MAP.get(cats[0], cats[0])

    # Try ground_truth
    if "ground_truth" in case_data:
        gt = case_data["ground_truth"]
        if "malpractice_analysis" in gt:
            cats = gt["malpractice_analysis"].get("malpractice_type", [])
            if cats:
                return MALPRACTICE_MAP.get(cats[0], cats[0])

    return "Other"

def get_specialty(case_data):
    """Extract clinical domain/specialty."""
    domain = case_data.get("clinical_domain", "Unknown")
    # Convert from UPPERCASE_FORMAT to Title Case
    return domain.replace("_", "/").title()

def get_decision_point_info(case_data):
    """Extract defendant action and expected action from decision points."""
    simulation = case_data.get("simulation", {})
    decision_points = simulation.get("decision_points", [])

    defendant_action = "Not documented"
    expected_action = "Not documented"

    for dp in decision_points:
        if dp.get("is_malpractice_point", False):
            if "actual_action_defendant" in dp:
                defendant_action = dp["actual_action_defendant"].get("description", "Not documented")
            if "expected_action_court" in dp:
                expected_action = dp["expected_action_court"].get("description", "Not documented")
            break

    return defendant_action, expected_action

def get_excluded_case_ids():
    """Get case IDs to exclude based on malpractice type (same logic as R script).

    R filters at run level, so a case is only excluded if ALL its runs
    have excluded malpractice types. Cases with any non-excluded run are kept.
    """
    import csv
    from collections import defaultdict
    runs_file = Path(__file__).parent.parent / "liability" / "exports" / "runs.csv"

    # Track all malpractice types per case
    case_types = defaultdict(set)
    with open(runs_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = row.get("case_id", "")
            mal_type = row.get("malpractice_type", "").lower()
            if case_id and mal_type:
                case_types[case_id].add(mal_type)

    # Only exclude cases where ALL runs have excluded types
    excluded_cases = set()
    for case_id, types in case_types.items():
        if types and all(t in EXCLUDED_MALPRACTICE_TYPES for t in types):
            excluded_cases.add(case_id)

    return excluded_cases

def main():
    # Get excluded case IDs (same logic as R script)
    excluded_case_ids = get_excluded_case_ids()

    cases = []
    excluded_count = 0

    # Process all JSON files
    for json_file in sorted(DATA_DIR.glob("*.json")):
        try:
            with open(json_file, "r") as f:
                case_data = json.load(f)

            case_id = case_data.get("case_id", json_file.stem)

            # Skip cases with excluded malpractice types
            if case_id in excluded_case_ids:
                excluded_count += 1
                continue

            source = case_data.get("source", "Unknown")
            jurisdiction = case_data.get("jurisdiction", "Unknown")

            malpractice_type = get_malpractice_type(case_data)
            specialty = get_specialty(case_data)
            defendant_action, expected_action = get_decision_point_info(case_data)

            # Map source to display name
            source_display = SOURCE_MAP.get(source, source)

            cases.append({
                "Source": source_display,
                "Case ID": case_id,
                "Jurisdiction": jurisdiction,
                "Malpractice Type": malpractice_type,
                "Specialty": specialty,
                "Defendant Action": defendant_action,
                "Expected Action (Court Standard)": expected_action
            })

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"Excluded {excluded_count} cases with non-clinical malpractice types")

    # Write CSV
    with open(OUTPUT_FILE, "w", newline="") as f:
        fieldnames = ["Source", "Case ID", "Jurisdiction", "Malpractice Type",
                      "Specialty", "Defendant Action", "Expected Action (Court Standard)"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)

    print(f"Generated {OUTPUT_FILE} with {len(cases)} cases")

if __name__ == "__main__":
    main()
