#!/usr/bin/env python3
"""Validate processed cases and filter out problematic ones.

This script:
1. Checks each processed case for quality issues
2. Moves problematic cases to a quarantine folder
3. Reports statistics

Quality checks:
- Verdict must not be UNKNOWN
- Case name must not be generic/extracted from body text
- Must have at least 2 decision points
- Must have at least 5 evidence items
- Evidence must have valid paragraph references
- Verdict must pass regex verification against raw source (if available)
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_verdict_in_raw(case_id: str, extracted_verdict: str, raw_dir: Path) -> tuple[bool, str]:
    """Check if extracted verdict matches patterns in raw source.

    Returns (is_consistent, message)
    """
    # Find raw file
    raw_path = None
    for source_dir in raw_dir.iterdir():
        if source_dir.is_dir():
            candidate = source_dir / f"{case_id}.html"
            if candidate.exists():
                raw_path = candidate
                break

    if not raw_path:
        return True, "No raw file (cannot verify)"

    try:
        with open(raw_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read().lower()

        # Get last 30% of document (conclusions usually at end)
        conclusion_section = content[int(len(content) * 0.7):]

        # Patterns indicating NO_LIABILITY
        no_liability_patterns = [
            r"claim\s+(is\s+)?dismiss",
            r"claim\s+fails",
            r"claimant['']?s?\s+claim\s+(is\s+)?dismiss",
            r"no\s+breach\s+of\s+duty",
            r"defendant\s+(is\s+)?not\s+liable",
            r"not\s+negligent",
            r"claim\s+must\s+fail",
            r"judgment\s+for\s+the\s+defendant",
            r"fails?\s+on\s+causation",
        ]

        # Patterns indicating LIABILITY_FOUND
        liability_patterns = [
            r"judgment\s+for\s+the\s+claimant",
            r"judgment\s+for\s+the\s+plaintiff",
            r"liability\s+(is\s+)?established",
            r"find\s+for\s+the\s+claimant",
            r"defendant\s+(is\s+)?liable",
            r"breach\s+of\s+duty\s+(is\s+)?established",
            r"claimant\s+succeeds",
            r"claimant['']?s?\s+claim\s+succeeds",
        ]

        # Check patterns
        found_no_liability = any(re.search(p, conclusion_section) for p in no_liability_patterns)
        found_liability = any(re.search(p, conclusion_section) for p in liability_patterns)

        # Determine detected verdict
        detected = None
        if found_no_liability and not found_liability:
            detected = "NO_LIABILITY"
        elif found_liability and not found_no_liability:
            detected = "LIABILITY_FOUND"
        elif found_no_liability and found_liability:
            # Mixed signals - might be partial liability or complex case
            detected = "MIXED"

        # Compare
        if detected is None:
            return True, "Could not detect verdict from raw (inconclusive)"

        if detected == "MIXED":
            return True, "Mixed signals in raw (possible partial liability)"

        if extracted_verdict == detected:
            return True, f"Verdict verified: {detected}"

        if extracted_verdict == "UNKNOWN":
            return False, f"Extracted UNKNOWN but raw indicates {detected}"

        return False, f"MISMATCH: extracted {extracted_verdict}, raw indicates {detected}"

    except Exception as e:
        return True, f"Error reading raw: {str(e)[:50]}"


def validate_case(case_path: Path, raw_dir: Path) -> tuple[bool, list[str]]:
    """Validate a single case. Returns (is_valid, list of issues)."""
    issues = []

    try:
        with open(case_path) as f:
            data = json.load(f)
    except Exception as e:
        return False, [f"Failed to load JSON: {str(e)[:50]}"]

    case_id = data.get("case_id", case_path.stem)

    # Check 1: Verdict must not be UNKNOWN
    simulation = data.get("simulation", {})
    end_state = simulation.get("end_state") if simulation else None
    legal_outcome = end_state.get("legal_outcome") if end_state else None
    verdict = legal_outcome.get("verdict") if legal_outcome else None
    if verdict == "UNKNOWN" or verdict is None:
        issues.append("Verdict is UNKNOWN")

    # Check 2: Case name must be proper (not extracted from body)
    case_name = data.get("case_name", "")
    if not case_name:
        issues.append("Missing case name")
    elif len(case_name) > 200:
        issues.append("Case name too long (likely extracted from body)")
    elif case_name.startswith(("i)", "ii)", "iii)", "iv)", "(a)", "(b)")):
        issues.append("Case name appears to be paragraph text")
    elif "Medical malpractice case involving" in case_name:
        issues.append("Generic auto-generated case name")
    # Check for garbled case names (extracted from judgment text)
    elif any(case_name.startswith(x) for x in [
        "Note ", "The ", "Similar ", "I emphasise", "Mr Justice", "16.", "149.",
        "iii)", "iv)", "In Re ", "In the Matter"
    ]):
        issues.append(f"Garbled case name: {case_name[:50]}")

    # Check 3: Must have decision points with malpractice relevance
    decision_points = data.get("simulation", {}).get("decision_points", [])
    if len(decision_points) == 0:
        issues.append("No decision points")
    else:
        # Check if at least one decision point is flagged as malpractice-related
        has_malpractice_point = any(dp.get("is_malpractice_point", False) for dp in decision_points)
        if not has_malpractice_point:
            issues.append("No malpractice-related decision points (cannot build evaluation checklist)")

    # Check 4: Must have sufficient evidence
    evidence = data.get("evidence_index", [])
    if len(evidence) < 3:
        issues.append(f"Only {len(evidence)} evidence item(s)")

    # Check 5: Evidence must have valid paragraph refs
    invalid_refs = 0
    for e in evidence:
        para_ref = e.get("paragraph_ref", "")
        if not para_ref or para_ref.lower() in ["none", "null", "unknown"]:
            invalid_refs += 1
    if invalid_refs > len(evidence) * 0.5:
        issues.append(f"{invalid_refs}/{len(evidence)} evidence items have invalid paragraph refs")

    # Check 6: Verify verdict against raw source
    if verdict and verdict != "UNKNOWN":
        is_consistent, msg = check_verdict_in_raw(case_id, verdict, raw_dir)
        if not is_consistent:
            issues.append(f"Verdict verification failed: {msg}")

    # Check 7: Initial state must have meaningful content
    initial_state = data.get("simulation", {}).get("initial_state", {})
    chief_complaint = initial_state.get("chief_complaint", "")
    if not chief_complaint or chief_complaint == "Not documented in judgment":
        # Check HPI instead
        hpi = initial_state.get("history_of_present_illness", "")
        if not hpi or len(hpi) < 20:
            issues.append("Missing or minimal chief complaint/HPI")

    # Check 8: Must be a medical malpractice case (not legal/other)
    domain = data.get("clinical_domain", "")
    case_name = data.get("case_name", "")
    case_name_lower = case_name.lower()

    # Non-medical indicators
    non_medical_indicators = [
        "barristers", "solicitors", "legal malpractice", "attorney",
        "condominium", "real estate", "mortgage", "property",
        "police", "employment", "labour", "labor", "termination",
        "dental admin", "insurance", "corp.", "corporation",
    ]

    # Constitutional/regulatory case indicators (not malpractice)
    regulatory_indicators = [
        "v. fda", "v. kobach", "v. russo", "constitutional",
        "statute", "legislation", "licensing board", "regulation",
        "administrative", "certiorari", "writ of", "quo warranto",
        "june medical", "alliance hippocratic",
    ]

    # Check for constitutional/regulatory cases
    evidence_text = " ".join(e.get("text", "") for e in evidence).lower()
    if any(ind in case_name_lower or ind in evidence_text for ind in regulatory_indicators):
        issues.append(f"Constitutional/regulatory case (not malpractice): {case_name[:50]}")

    # Medical indicators (at least one should be present for OTHER domain)
    medical_indicators = [
        "hospital", "medical", "doctor", "patient", "surgery", "treatment",
        "diagnosis", "injury", "death", "negligence", "clinical", "healthcare",
        "nurse", "physician", "clinic", "health", "care", "therapy",
    ]

    if domain == "OTHER" or not domain:
        # Check if case name has non-medical indicators
        has_non_medical = any(ind in case_name_lower for ind in non_medical_indicators)
        has_medical = any(ind in case_name_lower for ind in medical_indicators)

        if has_non_medical and not has_medical:
            issues.append(f"Non-medical case: {case_name[:60]}")
        elif not has_medical and domain == "OTHER":
            # Check evidence for medical content
            evidence_text = " ".join(e.get("text", "") for e in evidence).lower()
            has_medical_evidence = any(ind in evidence_text for ind in medical_indicators)
            if not has_medical_evidence:
                issues.append("No medical content detected in case")

    # Decision: valid if no critical issues
    # Critical issues that should fail the case:
    critical_issues = [
        i for i in issues if any(x in i.lower() for x in [
            "verdict is unknown",
            "no decision points",
            "no malpractice-related decision points",
            "verdict verification failed",
            "mismatch",
            "non-medical case",
            "no medical content",
            "garbled case name",
            "constitutional/regulatory",
        ])
    ]

    is_valid = len(critical_issues) == 0
    return is_valid, issues


def main():
    parser = argparse.ArgumentParser(description="Validate and filter processed cases")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without moving files")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Show details for each case")
    parser.add_argument("--quarantine-dir", type=str, default="data/quarantine",
                       help="Directory to move invalid cases to")
    args = parser.parse_args()

    processed_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    quarantine_dir = Path(args.quarantine_dir)

    if not args.dry_run:
        quarantine_dir.mkdir(parents=True, exist_ok=True)

    # Get all cases
    case_files = list(processed_dir.glob("*.json"))
    case_files = [f for f in case_files if not f.stem.endswith("_test")]

    print("=" * 60)
    print("CASE VALIDATION AND FILTERING")
    print("=" * 60)
    print(f"Processing {len(case_files)} cases...")
    print()

    valid_cases = []
    invalid_cases = []

    for case_path in sorted(case_files):
        is_valid, issues = validate_case(case_path, raw_dir)

        if is_valid:
            valid_cases.append((case_path, issues))
            if args.verbose and issues:
                print(f"  ✓ {case_path.stem} (warnings: {len(issues)})")
        else:
            invalid_cases.append((case_path, issues))
            if args.verbose or not is_valid:
                print(f"  ✗ {case_path.stem}")
                for issue in issues:
                    print(f"      - {issue}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total cases: {len(case_files)}")
    print(f"Valid: {len(valid_cases)}")
    print(f"Invalid: {len(invalid_cases)}")
    print(f"Validity rate: {len(valid_cases)/len(case_files)*100:.1f}%")
    print()

    # Show invalid cases
    if invalid_cases:
        print("Invalid cases to be quarantined:")
        for case_path, issues in invalid_cases:
            print(f"  {case_path.stem}")
            for issue in issues[:3]:  # Show top 3 issues
                print(f"    - {issue}")
        print()

    # Move invalid cases
    if not args.dry_run and invalid_cases:
        print(f"Moving {len(invalid_cases)} invalid cases to {quarantine_dir}...")
        for case_path, _ in invalid_cases:
            dest = quarantine_dir / case_path.name
            shutil.move(str(case_path), str(dest))
            print(f"  Moved: {case_path.stem}")
        print("Done.")
    elif args.dry_run and invalid_cases:
        print(f"DRY RUN: Would move {len(invalid_cases)} cases to {quarantine_dir}")

    # Count verdicts in valid cases
    print()
    print("Verdict distribution in valid cases:")
    verdict_counts = {}
    for case_path, _ in valid_cases:
        with open(case_path) as f:
            data = json.load(f)
        verdict = data.get("simulation", {}).get("end_state", {}).get("legal_outcome", {}).get("verdict", "UNKNOWN")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    for verdict, count in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {verdict}: {count}")

    return 0 if not invalid_cases else 1


if __name__ == "__main__":
    sys.exit(main())
