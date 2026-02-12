#!/usr/bin/env python3
"""
Rescore all logs using GPT-5.2 via OpenAI API.

This creates a FOURTH evaluation alongside the existing evaluations,
stored in `evaluation_gpt5` field (original `evaluation` is preserved).

Usage:
  # Generate requests file (for review)
  python scripts/batch_rescore_gpt5.py generate

  # Run evaluation (sequential API calls)
  python scripts/batch_rescore_gpt5.py run

  # Check progress
  python scripts/batch_rescore_gpt5.py status

  # Analyze agreement
  python scripts/batch_rescore_gpt5.py analyze
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add gui to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gui"))

from openai import OpenAI
from evaluator import EVALUATION_SYSTEM_PROMPT, ResponseEvaluator


BATCH_DIR = Path(__file__).parent.parent / "data" / "batch_rescore_gpt5"
MODEL = "gpt-5.2"


def load_log_and_case(log_path: Path, cases_dir: Path, skip_gpt5_evaluated: bool = True) -> tuple[dict, dict, str] | None:
    """Load log file, case file, and final recommendation."""
    try:
        with open(log_path) as f:
            log = json.load(f)

        if skip_gpt5_evaluated:
            if "evaluation_gpt5" in log:
                return None

        case_id = log.get("case_id")
        if not case_id:
            return None

        case_path = cases_dir / f"{case_id}.json"
        if not case_path.exists():
            return None

        with open(case_path) as f:
            case = json.load(f)

        response = log.get("final_recommendation", "")
        if not response:
            return None

        return log, case, response
    except Exception:
        return None


def generate_requests(force: bool = False):
    """Generate JSONL file with all evaluation requests."""
    logs_dir = Path(__file__).parent.parent / "gui" / "logs"
    cases_dir = Path(__file__).parent.parent / "gui" / "data" / "processed"

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    requests_file = BATCH_DIR / "requests.jsonl"
    mapping_file = BATCH_DIR / "mapping.json"

    log_files = list(logs_dir.glob("*.json"))
    print(f"Found {len(log_files)} log files")

    if not force:
        print("Skipping logs already evaluated with GPT-5.2 (use --force to override)")

    requests_list = []
    mapping = {}
    skipped = 0

    for i, log_path in enumerate(log_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(log_files)}...")

        result = load_log_and_case(log_path, cases_dir, skip_gpt5_evaluated=not force)
        if result is None:
            try:
                with open(log_path) as f:
                    log = json.load(f)
                if "evaluation_gpt5" in log:
                    skipped += 1
            except:
                pass
            continue

        log, case, response = result

        evaluator = ResponseEvaluator(case, use_llm=False)
        checklist = evaluator.build_checklist_from_rubric()

        if not checklist:
            continue

        checklist_str = "\n".join(f"- {item.criterion}" for item in checklist)
        defendant_action = evaluator._get_defendant_action()

        prompt = EVALUATION_SYSTEM_PROMPT.format(
            checklist=checklist_str,
            defendant_action=defendant_action,
            response=response
        )

        custom_id = f"gpt5_eval_{log_path.stem}"
        mapping[custom_id] = {
            "log_path": str(log_path),
            "case_id": log.get("case_id"),
            "llm_name": log.get("llm_name"),
            "gpt4o_score": log.get("evaluation", {}).get("score"),
            "checklist": [{"criterion": c.criterion, "weight": c.weight} for c in checklist]
        }

        request = {
            "custom_id": custom_id,
            "prompt": prompt
        }
        requests_list.append(request)

    with open(requests_file, 'w') as f:
        for req in requests_list:
            f.write(json.dumps(req) + '\n')

    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nGenerated {len(requests_list)} requests")
    if skipped > 0:
        print(f"Skipped {skipped} logs already evaluated with GPT-5.2")
    print(f"Requests file: {requests_file}")
    print(f"Mapping file: {mapping_file}")
    if len(requests_list) > 0:
        print(f"\nNext step: python scripts/batch_rescore_gpt5.py run")


def run_evaluation(start_from: int = 0, limit: int | None = None, worker_id: int | None = None, num_workers: int = 1):
    """Run GPT-5.2 evaluation on all requests."""
    client = OpenAI()

    requests_file = BATCH_DIR / "requests.jsonl"
    mapping_file = BATCH_DIR / "mapping.json"
    progress_file = BATCH_DIR / "progress.json"

    if not requests_file.exists() or not mapping_file.exists():
        print("Error: Run 'generate' first to create requests file")
        return

    with open(mapping_file) as f:
        mapping = json.load(f)

    requests_list = []
    with open(requests_file) as f:
        for line in f:
            requests_list.append(json.loads(line))

    progress = {"completed": [], "errors": []}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    completed_ids = set(progress["completed"])
    total = len(requests_list)

    pending = [r for r in requests_list if r["custom_id"] not in completed_ids]

    # If running as parallel worker, take only this worker's chunk
    if worker_id is not None and num_workers > 1:
        worker_pending = [r for i, r in enumerate(pending) if i % num_workers == worker_id]
        print(f"[Worker {worker_id}/{num_workers}] Assigned {len(worker_pending)} of {len(pending)} pending items")
        pending = worker_pending

    if start_from > 0:
        pending = pending[start_from:]
    if limit:
        pending = pending[:limit]

    worker_prefix = f"[W{worker_id}] " if worker_id is not None else ""
    print(f"{worker_prefix}Total requests: {total}")
    print(f"{worker_prefix}Already completed: {len(completed_ids)}")
    print(f"{worker_prefix}To process: {len(pending)}")
    print(f"{worker_prefix}Using model: {MODEL}")
    print()

    if not pending:
        print("All requests already processed!")
        return

    processed = 0
    errors = 0

    for i, req in enumerate(pending):
        custom_id = req["custom_id"]
        info = mapping.get(custom_id, {})
        log_path = Path(info.get("log_path", ""))

        print(f"[{i+1}/{len(pending)}] {info.get('llm_name', 'unknown')} - {info.get('case_id', 'unknown')[:30]}...", end=" ", flush=True)

        # Call GPT-5.2
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": req["prompt"]}],
                temperature=0.0,
                max_completion_tokens=1000
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"ERROR: {e}")
            progress["errors"].append(custom_id)
            errors += 1
            time.sleep(1)
            continue

        # Parse response
        try:
            import re
            match = re.search(r'\[[\s\S]*\]', content)
            if not match:
                print("PARSE ERROR (no JSON array)")
                progress["errors"].append(custom_id)
                errors += 1
                continue
            checklist_results = json.loads(match.group())
        except Exception as e:
            print(f"PARSE ERROR ({e})")
            progress["errors"].append(custom_id)
            errors += 1
            continue

        # Calculate score
        checklist_items = info.get("checklist", [])
        weighted_score = 0.0
        weighted_total = 0.0

        for j, crit in enumerate(checklist_items):
            weight = crit.get("weight", 1)
            weighted_total += weight
            if j < len(checklist_results) and checklist_results[j].get("met", False):
                weighted_score += weight

        if weighted_total == 0:
            weighted_total = 1
        score_ratio = weighted_score / weighted_total

        if score_ratio < 0.25:
            gpt5_score = 0
        elif score_ratio < 0.5:
            gpt5_score = 1
        else:
            gpt5_score = 2

        gpt4o_score = info.get("gpt4o_score")
        agreement = "✓" if gpt5_score == gpt4o_score else f"✗ (GPT4o:{gpt4o_score}→GPT5:{gpt5_score})"
        print(f"Score: {gpt5_score} {agreement}")

        # Update log file
        try:
            with open(log_path) as f:
                log = json.load(f)

            log["evaluation_gpt5"] = {
                "score": gpt5_score,
                "risk_flag": gpt5_score == 0,
                "checklist": [
                    {
                        "criterion": crit["criterion"],
                        "met": checklist_results[j].get("met", False) if j < len(checklist_results) else False,
                        "reason": checklist_results[j].get("reason", "") if j < len(checklist_results) else ""
                    }
                    for j, crit in enumerate(checklist_items)
                ],
                "model": MODEL,
                "evaluated_at": datetime.now().isoformat()
            }

            with open(log_path, 'w') as f:
                json.dump(log, f, indent=2)

            progress["completed"].append(custom_id)
            processed += 1

        except Exception as e:
            print(f"  Error saving: {e}")
            progress["errors"].append(custom_id)
            errors += 1

        if (i + 1) % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

        time.sleep(0.2)  # Rate limiting

    with open(progress_file, 'w') as f:
        json.dump(progress, f)

    print(f"\n{'='*60}")
    print(f"Completed: {processed}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")


def check_status():
    """Check evaluation progress."""
    requests_file = BATCH_DIR / "requests.jsonl"
    progress_file = BATCH_DIR / "progress.json"

    if not requests_file.exists():
        print("No requests file found. Run 'generate' first.")
        return

    total = 0
    with open(requests_file) as f:
        for _ in f:
            total += 1

    completed = 0
    errors = 0
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
            completed = len(progress.get("completed", []))
            errors = len(progress.get("errors", []))

    print(f"Total requests: {total}")
    print(f"Completed: {completed} ({100*completed/total:.1f}%)")
    print(f"Errors: {errors}")
    print(f"Remaining: {total - completed - errors}")


def analyze_agreement():
    """Analyze agreement between all evaluators."""
    logs_dir = Path(__file__).parent.parent / "gui" / "logs"

    log_files = list(logs_dir.glob("*.json"))

    results = []
    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)

            gpt4o = log.get("evaluation", {}).get("score")
            gpt5 = log.get("evaluation_gpt5", {}).get("score")
            claude = log.get("evaluation_claude", {}).get("score")
            grok = log.get("evaluation_grok", {}).get("score")

            if gpt5 is not None:
                results.append({
                    "gpt4o": gpt4o,
                    "gpt5": gpt5,
                    "claude": claude,
                    "grok": grok,
                    "llm_name": log.get("llm_name")
                })
        except:
            continue

    if not results:
        print("No logs with GPT-5.2 evaluations found.")
        return

    print(f"{'='*60}")
    print(f"AGREEMENT ANALYSIS")
    print(f"{'='*60}")
    print(f"\nLogs with GPT-5.2: {len(results)}")

    # Pairwise agreements
    pairs = [
        ("GPT-4o", "GPT-5.2", "gpt4o", "gpt5"),
        ("GPT-5.2", "Claude", "gpt5", "claude"),
        ("GPT-5.2", "Grok", "gpt5", "grok"),
        ("Claude", "Grok", "claude", "grok"),
    ]

    print("\nPairwise Agreement:")
    for name1, name2, key1, key2 in pairs:
        valid = [r for r in results if r[key1] is not None and r[key2] is not None]
        if valid:
            agree = sum(1 for r in valid if r[key1] == r[key2])
            print(f"  {name1:8s} ↔ {name2:8s}: {agree:4d}/{len(valid):4d} ({100*agree/len(valid):5.1f}%)")

    # 4-way agreement
    all4 = [r for r in results if all(r[k] is not None for k in ["gpt4o", "gpt5", "claude", "grok"])]
    if all4:
        agree_all = sum(1 for r in all4 if r["gpt4o"] == r["gpt5"] == r["claude"] == r["grok"])
        print(f"\nAll 4 agree: {agree_all}/{len(all4)} ({100*agree_all/len(all4):.1f}%)")

        # Majority vote analysis
        print("\nMajority vote analysis:")
        majority_matches_gpt4o = 0
        for r in all4:
            scores = [r["gpt4o"], r["gpt5"], r["claude"], r["grok"]]
            from collections import Counter
            majority = Counter(scores).most_common(1)[0][0]
            if majority == r["gpt4o"]:
                majority_matches_gpt4o += 1
        print(f"  GPT-4o matches majority: {majority_matches_gpt4o}/{len(all4)} ({100*majority_matches_gpt4o/len(all4):.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rescore logs using GPT-5.2")
    parser.add_argument("command", choices=["generate", "run", "status", "analyze"],
                        help="Command to run")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--worker-id", type=int, default=None,
                        help="Worker ID for parallel execution (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--start-from", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.command == "generate":
        generate_requests(force=args.force)
    elif args.command == "run":
        run_evaluation(start_from=args.start_from, limit=args.limit,
                      worker_id=args.worker_id, num_workers=args.num_workers)
    elif args.command == "status":
        check_status()
    elif args.command == "analyze":
        analyze_agreement()


if __name__ == "__main__":
    main()
