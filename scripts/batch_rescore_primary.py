#!/usr/bin/env python3
"""
Rescore all logs using the Primary Action scoring method.

This creates a SECOND evaluation alongside existing evaluations,
stored in `evaluation_primary` field. Original evaluations are preserved.

Primary Action Scoring:
- Score 0: Primary action NOT met (regardless of secondary criteria)
- Score 1: Primary action MET, <50% of secondary criteria met
- Score 2: Primary action MET, ≥50% of secondary criteria met

The primary action is the `expected_action_court.description` from each
decision point - what the court said the defendant should have done.

Usage:
  # Run rescoring (uses existing checklist evaluations)
  python scripts/batch_rescore_primary.py run

  # Check status
  python scripts/batch_rescore_primary.py status

  # Analyze score changes from original to primary
  python scripts/batch_rescore_primary.py analyze

  # Compare model rankings
  python scripts/batch_rescore_primary.py compare
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Add gui to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gui"))

from evaluator import ResponseEvaluator


LOGS_DIR = Path(__file__).parent.parent / "gui" / "logs"
CASES_DIR = Path(__file__).parent.parent / "data" / "processed"


def compute_primary_score(evaluation: dict, case: dict) -> dict:
    """
    Recompute score using primary-action logic.

    Args:
        evaluation: Original evaluation dict (with checklist)
        case: Case dict (to identify primary criteria)

    Returns:
        New evaluation dict with primary-action score
    """
    checklist = evaluation.get("checklist", [])
    if not checklist:
        return None

    # Build a set of primary criteria from the case
    # Primary = expected_action_court.description from malpractice points
    primary_criteria = set()
    decision_points = case.get("simulation", {}).get("decision_points", [])
    for dp in decision_points:
        if dp.get("is_malpractice_point", False):
            expected = dp.get("expected_action_court", {})
            desc = expected.get("description", "")
            if desc:
                primary_criteria.add(desc.lower().strip())

    # Classify checklist items as primary or secondary
    primary_items = []
    secondary_items = []

    for item in checklist:
        criterion = item.get("criterion", "").lower().strip()
        source = item.get("source", "")

        # Items from court_expected are primary
        # Also check if criterion text matches known primary criteria
        is_primary = (
            source == "court_expected" or
            criterion in primary_criteria or
            any(criterion in p or p in criterion for p in primary_criteria if len(criterion) > 10)
        )

        if is_primary:
            primary_items.append(item)
        else:
            secondary_items.append(item)

    # Check if primary items are met
    def item_met(item):
        completeness = item.get("completeness_score", 0)
        met = item.get("met", False)
        return met or completeness >= 0.5

    primary_met = all(item_met(item) for item in primary_items) if primary_items else True

    # Check defendant match (from original evaluation)
    defendant_match = evaluation.get("risk_flag", False) and evaluation.get("score", 2) == 0

    # Calculate score
    if not primary_met or (defendant_match and not primary_items):
        score = 0
        risk_flag = True
    else:
        # Calculate secondary ratio
        if secondary_items:
            secondary_score = sum(
                item.get("completeness_score", 0) if item.get("completeness_score", 0) > 0
                else (1.0 if item.get("met", False) else 0.0)
                for item in secondary_items
            )
            secondary_ratio = secondary_score / len(secondary_items)
        else:
            secondary_ratio = 1.0

        # Factor in reasoning quality if available
        reasoning = evaluation.get("reasoning_quality", {})
        if isinstance(reasoning, dict):
            quality_score = reasoning.get("quality_score", 0)
        else:
            quality_score = 0

        adjusted_ratio = secondary_ratio * 0.9 + quality_score * 0.1

        if adjusted_ratio < 0.5:
            score = 1
            risk_flag = False
        else:
            score = 2
            risk_flag = False

    return {
        "score": score,
        "risk_flag": risk_flag,
        "scoring_method": "primary_action",
        "primary_criteria_count": len(primary_items),
        "secondary_criteria_count": len(secondary_items),
        "primary_met": primary_met,
        "original_score": evaluation.get("score"),
    }


def run_rescoring(force: bool = False):
    """Rescore all logs using primary-action logic."""
    log_files = list(LOGS_DIR.glob("*.json"))
    print(f"Found {len(log_files)} log files")

    processed = 0
    skipped_existing = 0
    skipped_no_eval = 0
    skipped_no_case = 0
    errors = 0

    score_changes = defaultdict(int)  # (old, new) -> count

    for i, log_path in enumerate(log_files):
        if (i + 1) % 200 == 0:
            print(f"Processing {i + 1}/{len(log_files)}...")

        try:
            with open(log_path) as f:
                log = json.load(f)

            # Skip if already has primary evaluation (unless force)
            if not force and "evaluation_primary" in log:
                skipped_existing += 1
                continue

            # Need original evaluation with checklist
            evaluation = log.get("evaluation", {})
            if not evaluation or not evaluation.get("checklist"):
                skipped_no_eval += 1
                continue

            # Need case file
            case_id = log.get("case_id")
            case_path = CASES_DIR / f"{case_id}.json"
            if not case_path.exists():
                # Try gui/data/processed
                case_path = Path(__file__).parent.parent / "gui" / "data" / "processed" / f"{case_id}.json"
                if not case_path.exists():
                    skipped_no_case += 1
                    continue

            with open(case_path) as f:
                case = json.load(f)

            # Compute primary score
            primary_eval = compute_primary_score(evaluation, case)
            if primary_eval is None:
                skipped_no_eval += 1
                continue

            # Track score changes
            old_score = evaluation.get("score", -1)
            new_score = primary_eval["score"]
            score_changes[(old_score, new_score)] += 1

            # Add to log
            log["evaluation_primary"] = primary_eval

            # Write back
            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)

            processed += 1

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {log_path.name}: {e}")

    print(f"\nResults:")
    print(f"  Processed: {processed}")
    print(f"  Skipped (already has primary eval): {skipped_existing}")
    print(f"  Skipped (no evaluation/checklist): {skipped_no_eval}")
    print(f"  Skipped (no case file): {skipped_no_case}")
    print(f"  Errors: {errors}")

    print(f"\nScore changes (original -> primary):")
    for (old, new), count in sorted(score_changes.items()):
        direction = "↓" if new < old else ("↑" if new > old else "=")
        print(f"  {old} -> {new} {direction}: {count}")


def check_status():
    """Check how many logs have primary evaluation."""
    log_files = list(LOGS_DIR.glob("*.json"))

    has_primary = 0
    has_original = 0
    total = len(log_files)

    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)
            if "evaluation_primary" in log:
                has_primary += 1
            if "evaluation" in log:
                has_original += 1
        except:
            pass

    print(f"Total logs: {total}")
    print(f"With original evaluation: {has_original}")
    print(f"With primary evaluation: {has_primary}")
    print(f"Missing primary: {has_original - has_primary}")


def analyze_changes():
    """Analyze how scores changed between original and primary."""
    log_files = list(LOGS_DIR.glob("*.json"))

    changes = []
    by_model = defaultdict(lambda: {"original": [], "primary": []})

    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)

            eval_orig = log.get("evaluation", {})
            eval_primary = log.get("evaluation_primary", {})

            if not eval_orig or not eval_primary:
                continue

            old_score = eval_orig.get("score", -1)
            new_score = eval_primary.get("score", -1)

            if old_score < 0 or new_score < 0:
                continue

            llm_name = log.get("llm_name", "unknown")

            changes.append({
                "case_id": log.get("case_id"),
                "llm_name": llm_name,
                "old_score": old_score,
                "new_score": new_score,
                "change": new_score - old_score,
                "primary_met": eval_primary.get("primary_met", None),
            })

            by_model[llm_name]["original"].append(old_score)
            by_model[llm_name]["primary"].append(new_score)

        except:
            pass

    if not changes:
        print("No logs with both original and primary evaluations found.")
        print("Run 'python scripts/batch_rescore_primary.py run' first.")
        return

    # Overall stats
    print(f"Analyzed {len(changes)} evaluations\n")

    # Count transitions
    transitions = defaultdict(int)
    for c in changes:
        transitions[(c["old_score"], c["new_score"])] += 1

    print("Score transitions (original -> primary):")
    print("-" * 40)
    for (old, new), count in sorted(transitions.items()):
        pct = count / len(changes) * 100
        direction = "↓ STRICTER" if new < old else ("↑ more lenient" if new > old else "= same")
        print(f"  {old} -> {new}: {count:4d} ({pct:5.1f}%) {direction}")

    # Cases where primary is stricter
    stricter = [c for c in changes if c["new_score"] < c["old_score"]]
    print(f"\nCases where primary scoring is STRICTER: {len(stricter)} ({len(stricter)/len(changes)*100:.1f}%)")

    if stricter:
        print("\nExamples where primary action was missed but original score was >0:")
        for c in stricter[:5]:
            print(f"  {c['case_id']}: {c['old_score']} -> {c['new_score']} ({c['llm_name']})")

    # Model-level comparison
    print("\n" + "="*60)
    print("Model-level comparison:")
    print("="*60)
    print(f"{'Model':<25} {'Orig Mean':>10} {'Prim Mean':>10} {'Change':>10}")
    print("-" * 60)

    for model in sorted(by_model.keys()):
        orig_scores = by_model[model]["original"]
        prim_scores = by_model[model]["primary"]
        if orig_scores:
            orig_mean = sum(orig_scores) / len(orig_scores)
            prim_mean = sum(prim_scores) / len(prim_scores)
            # Convert to 0-1 scale for easier comparison
            orig_pct = orig_mean / 2
            prim_pct = prim_mean / 2
            change = prim_pct - orig_pct
            print(f"{model:<25} {orig_pct:>10.2%} {prim_pct:>10.2%} {change:>+10.1%}")


def compare_rankings():
    """Compare model rankings between original and primary scoring."""
    log_files = list(LOGS_DIR.glob("*.json"))

    by_model = defaultdict(lambda: {"original": [], "primary": []})

    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)

            eval_orig = log.get("evaluation", {})
            eval_primary = log.get("evaluation_primary", {})

            if not eval_orig or not eval_primary:
                continue

            old_score = eval_orig.get("score", -1)
            new_score = eval_primary.get("score", -1)

            if old_score < 0 or new_score < 0:
                continue

            llm_name = log.get("llm_name", "unknown")
            by_model[llm_name]["original"].append(old_score)
            by_model[llm_name]["primary"].append(new_score)

        except:
            pass

    if not by_model:
        print("No data found. Run rescoring first.")
        return

    # Calculate means and rankings
    models = []
    for model, scores in by_model.items():
        if scores["original"]:
            orig_mean = sum(scores["original"]) / len(scores["original"]) / 2  # 0-1 scale
            prim_mean = sum(scores["primary"]) / len(scores["primary"]) / 2
            models.append({
                "model": model,
                "orig_mean": orig_mean,
                "prim_mean": prim_mean,
                "n": len(scores["original"]),
            })

    # Sort by original score
    models_by_orig = sorted(models, key=lambda x: x["orig_mean"], reverse=True)
    models_by_prim = sorted(models, key=lambda x: x["prim_mean"], reverse=True)

    print("Model Rankings Comparison")
    print("="*70)
    print(f"{'Rank':<6} {'Original Ranking':<25} {'Primary Ranking':<25}")
    print("-"*70)

    for i in range(len(models)):
        orig = models_by_orig[i]
        prim = models_by_prim[i]
        print(f"{i+1:<6} {orig['model']:<20} ({orig['orig_mean']:.2%})   {prim['model']:<20} ({prim['prim_mean']:.2%})")

    # Check if rankings changed
    orig_order = [m["model"] for m in models_by_orig]
    prim_order = [m["model"] for m in models_by_prim]

    if orig_order == prim_order:
        print("\n✓ Rankings are IDENTICAL between original and primary scoring")
    else:
        print("\n⚠ Rankings CHANGED:")
        for i, (o, p) in enumerate(zip(orig_order, prim_order)):
            if o != p:
                print(f"  Rank {i+1}: {o} -> {p}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    command = sys.argv[1]

    if command == "run":
        force = "--force" in sys.argv
        run_rescoring(force=force)
    elif command == "status":
        check_status()
    elif command == "analyze":
        analyze_changes()
    elif command == "compare":
        compare_rankings()
    else:
        print(f"Unknown command: {command}")
        print(__doc__)


if __name__ == "__main__":
    main()
