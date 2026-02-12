#!/usr/bin/env python3
"""
Quarantine untestable cases and clean up their simulation logs.

This script:
1. Identifies cases that should be quarantined (NO_LIABILITY, procedural, already marked untestable)
2. Moves case JSON files to data/quarantine/
3. Removes any simulation logs for those cases from gui/logs/
"""

import json
import re
import shutil
from pathlib import Path
from dataclasses import dataclass, field


# Keywords indicating NO LIABILITY / CLAIM DISMISSED
NO_LIABILITY_PATTERNS = [
    r"claim\s+is\s+dismissed",
    r"i\s+dismiss\s+the\s+claim",
    r"claimant'?s?\s+claim\s+is\s+dismissed",
    r"judgment\s+for\s+the\s+defendant",
    r"there\s+must\s+be\s+judgment\s+for\s+the\s+defendant",
    r"no\s+breach\s+of\s+duty",
    r"i\s+find\s+no\s+breach",
    r"not\s+negligent",
    r"was\s+not\s+negligent",
    r"defendant\s+was\s+not\s+negligent",
    r"claimant\s+has\s+failed\s+to\s+prove",
    r"claimant\s+fails",
    r"claim\s+fails",
    r"i\s+therefore\s+dismiss",
    r"in\s+accordance\s+with\s+.*?reasonable.*?body",
    r"bolam\s+test\s+.*?satisfied",
    r"no\s+causation",
    r"causation\s+not\s+established",
]

NO_LIABILITY_RE = [re.compile(p, re.IGNORECASE) for p in NO_LIABILITY_PATTERNS]


def should_quarantine(case: dict) -> tuple[bool, str]:
    """Determine if a case should be quarantined. Returns (should_quarantine, reason)."""

    sim = case.get("simulation", {})

    # Already marked untestable
    if sim.get("testable") is False:
        reason = sim.get("testable_reason", "Marked untestable")
        return True, f"Already marked untestable: {reason[:100]}"

    # Check verdict
    end_state = sim.get("end_state", {})
    if end_state:
        legal_outcome = end_state.get("legal_outcome", {})
        verdict = legal_outcome.get("verdict", "")
        if isinstance(verdict, dict):
            verdict = verdict.get("value", str(verdict))

        malpractice = end_state.get("malpractice_determination", {})
        breach_found = malpractice.get("breach_found")

        # Clear NO_LIABILITY cases
        if verdict == "NO_LIABILITY":
            return True, f"Verdict is NO_LIABILITY (breach_found={breach_found})"

        # Breach explicitly false with unknown verdict
        if breach_found is False and verdict in ["UNKNOWN", ""]:
            return True, "breach_found=False with unknown verdict"

    # Check evidence for dismissal language
    evidence_index = case.get("evidence_index", [])
    all_evidence_text = " ".join(e.get("text", "") for e in evidence_index)

    for pattern in NO_LIABILITY_RE:
        match = pattern.search(all_evidence_text)
        if match:
            # Double-check this isn't a false positive by verifying verdict isn't LIABILITY_FOUND
            if end_state:
                legal_outcome = end_state.get("legal_outcome", {})
                verdict = legal_outcome.get("verdict", "")
                if verdict == "LIABILITY_FOUND":
                    continue  # Skip - likely false positive
            return True, f"Evidence contains dismissal language: '{match.group()[:50]}'"

    return False, ""


def quarantine_cases(
    processed_dir: Path,
    quarantine_dir: Path,
    logs_dir: Path,
    dry_run: bool = False
) -> dict:
    """Quarantine untestable cases and clean logs."""

    stats = {
        "scanned": 0,
        "quarantined": 0,
        "logs_removed": 0,
        "cases": []
    }

    # Scan all cases
    for json_file in sorted(processed_dir.glob("*.json")):
        stats["scanned"] += 1

        try:
            with open(json_file) as f:
                case = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file.name}: {e}")
            continue

        case_id = case.get("case_id", json_file.stem)
        should_q, reason = should_quarantine(case)

        if should_q:
            stats["quarantined"] += 1
            stats["cases"].append({"case_id": case_id, "reason": reason})

            if not dry_run:
                # Move case file to quarantine
                dest = quarantine_dir / json_file.name
                shutil.move(str(json_file), str(dest))
                print(f"Quarantined: {case_id}")
            else:
                print(f"[DRY RUN] Would quarantine: {case_id} - {reason[:60]}")

    # Clean up logs
    if logs_dir.exists():
        quarantined_ids = {c["case_id"] for c in stats["cases"]}

        for log_file in logs_dir.glob("*.json"):
            try:
                with open(log_file) as f:
                    log = json.load(f)
                log_case_id = log.get("case_id", "")

                if log_case_id in quarantined_ids:
                    stats["logs_removed"] += 1
                    if not dry_run:
                        log_file.unlink()
                        print(f"Removed log: {log_file.name}")
                    else:
                        print(f"[DRY RUN] Would remove log: {log_file.name}")
            except Exception as e:
                print(f"Error processing log {log_file.name}: {e}")

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Quarantine untestable cases")
    parser.add_argument("--processed", "-p", type=str, default="data/processed",
                       help="Directory containing processed case JSONs")
    parser.add_argument("--quarantine", "-q", type=str, default="data/quarantine",
                       help="Directory to move quarantined cases to")
    parser.add_argument("--logs", "-l", type=str, default="gui/logs",
                       help="Directory containing simulation logs")
    parser.add_argument("--dry-run", "-n", action="store_true",
                       help="Show what would be done without making changes")

    args = parser.parse_args()

    processed_dir = Path(args.processed)
    quarantine_dir = Path(args.quarantine)
    logs_dir = Path(args.logs)

    if not processed_dir.exists():
        print(f"Error: Processed directory {processed_dir} does not exist")
        return 1

    # Create quarantine directory
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {processed_dir}...")
    if args.dry_run:
        print("DRY RUN MODE - no changes will be made\n")

    stats = quarantine_cases(processed_dir, quarantine_dir, logs_dir, args.dry_run)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Cases scanned: {stats['scanned']}")
    print(f"Cases quarantined: {stats['quarantined']}")
    print(f"Logs removed: {stats['logs_removed']}")
    print(f"Remaining testable cases: {stats['scanned'] - stats['quarantined']}")

    if args.dry_run:
        print("\nRun without --dry-run to apply changes")

    return 0


if __name__ == "__main__":
    exit(main())
