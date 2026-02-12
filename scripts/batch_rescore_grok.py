#!/usr/bin/env python3
"""
Rescore all logs using Grok 4.1 Fast via OpenRouter.

This creates a THIRD evaluation alongside the existing GPT-4o and Grok evaluations,
stored in `evaluation_grok` field (original `evaluation` is preserved).

Workflow:
1. Generate batch requests file (JSONL)
2. Process sequentially (OpenRouter doesn't have batch API)
3. Update logs with Grok evaluation in separate field

Usage:
  # Generate requests file (for review)
  python scripts/batch_rescore_grok.py generate

  # Run evaluation (sequential API calls)
  python scripts/batch_rescore_grok.py run

  # Check progress
  python scripts/batch_rescore_grok.py status

  # Analyze agreement between GPT-4o and Grok
  python scripts/batch_rescore_grok.py analyze
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add gui to path
sys.path.insert(0, str(Path(__file__).parent.parent / "gui"))

import requests
from evaluator import EVALUATION_SYSTEM_PROMPT, ResponseEvaluator


BATCH_DIR = Path(__file__).parent.parent / "data" / "batch_rescore_grok"
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
MODEL = "x-ai/grok-4.1-fast"


def load_log_and_case(log_path: Path, cases_dir: Path, skip_grok_evaluated: bool = True) -> tuple[dict, dict, str] | None:
    """Load log file, case file, and final recommendation.

    Args:
        log_path: Path to log file
        cases_dir: Path to cases directory
        skip_grok_evaluated: If True, skip logs already evaluated with Grok
    """
    try:
        with open(log_path) as f:
            log = json.load(f)

        # Skip logs already evaluated with Grok
        if skip_grok_evaluated:
            if "evaluation_grok" in log:
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
    """Generate JSONL file with all evaluation requests (for review)."""
    logs_dir = Path(__file__).parent.parent / "gui" / "logs"
    cases_dir = Path(__file__).parent.parent / "gui" / "data" / "processed"

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    requests_file = BATCH_DIR / "requests.jsonl"
    mapping_file = BATCH_DIR / "mapping.json"

    log_files = list(logs_dir.glob("*.json"))
    print(f"Found {len(log_files)} log files")

    if not force:
        print("Skipping logs already evaluated with Grok (use --force to override)")

    requests_list = []
    mapping = {}
    skipped = 0

    for i, log_path in enumerate(log_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(log_files)}...")

        result = load_log_and_case(log_path, cases_dir, skip_grok_evaluated=not force)
        if result is None:
            # Check if skipped due to existing Grok evaluation
            try:
                with open(log_path) as f:
                    log = json.load(f)
                if "evaluation_grok" in log:
                    skipped += 1
            except:
                pass
            continue

        log, case, response = result

        # Build evaluation request
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

        custom_id = f"grok_eval_{log_path.stem}"
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

    # Write requests file
    with open(requests_file, 'w') as f:
        for req in requests_list:
            f.write(json.dumps(req) + '\n')

    # Write mapping file
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nGenerated {len(requests_list)} requests")
    if skipped > 0:
        print(f"Skipped {skipped} logs already evaluated with Grok")
    print(f"Requests file: {requests_file}")
    print(f"Mapping file: {mapping_file}")
    if len(requests_list) > 0:
        print(f"\nNext step: python scripts/batch_rescore_claude.py run")
    else:
        print("\nNo new logs to evaluate.")


def call_grok(prompt: str) -> str | None:
    """Call Grok 4.1 Fast via OpenRouter."""
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/clad-benchmark",
            "X-Title": "CLAD Medical Benchmark"
        },
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 1000
        }
    )

    if response.status_code != 200:
        print(f"  API error: {response.status_code} - {response.text[:200]}")
        return None

    try:
        return response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"  Parse error: {e}")
        return None


def run_evaluation(start_from: int = 0, limit: int | None = None, worker_id: int | None = None, num_workers: int = 1):
    """Run Grok evaluation on all requests.

    Args:
        start_from: Start from this request index
        limit: Limit number of requests to process
        worker_id: Worker ID for parallel execution (0-indexed)
        num_workers: Total number of parallel workers
    """
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("Get an API key from https://openrouter.ai/")
        return

    requests_file = BATCH_DIR / "requests.jsonl"
    mapping_file = BATCH_DIR / "mapping.json"
    progress_file = BATCH_DIR / "progress.json"

    if not requests_file.exists() or not mapping_file.exists():
        print("Error: Run 'generate' first to create requests file")
        return

    with open(mapping_file) as f:
        mapping = json.load(f)

    # Load requests
    requests_list = []
    with open(requests_file) as f:
        for line in f:
            requests_list.append(json.loads(line))

    # Load progress
    progress = {"completed": [], "errors": []}
    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)

    completed_ids = set(progress["completed"])
    total = len(requests_list)

    # Filter to pending requests
    pending = [r for r in requests_list if r["custom_id"] not in completed_ids]

    # If running as parallel worker, take only this worker's chunk
    if worker_id is not None and num_workers > 1:
        # Distribute pending items across workers
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

        # Call Grok
        content = call_grok(req["prompt"])

        if content is None:
            print("ERROR")
            progress["errors"].append(custom_id)
            errors += 1
            time.sleep(1)  # Back off on errors
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
            grok_score = 0
        elif score_ratio < 0.5:
            grok_score = 1
        else:
            grok_score = 2

        gpt4o_score = info.get("gpt4o_score")
        agreement = "✓" if grok_score == gpt4o_score else f"✗ (GPT:{gpt4o_score}→Grok:{grok_score})"
        print(f"Score: {grok_score} {agreement}")

        # Update log file with Grok evaluation (preserve original)
        try:
            with open(log_path) as f:
                log = json.load(f)

            log["evaluation_grok"] = {
                "score": grok_score,
                "risk_flag": grok_score == 0,
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

        # Save progress periodically
        if (i + 1) % 10 == 0:
            with open(progress_file, 'w') as f:
                json.dump(progress, f)

        # Rate limiting (OpenRouter has generous limits but be polite)
        time.sleep(0.5)

    # Final progress save
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

    print(f"\n{'='*60}")
    print(f"Completed: {processed}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")
    print("\nRun 'analyze' to compare GPT-4o and Grok scores")


def check_status():
    """Check evaluation progress."""
    requests_file = BATCH_DIR / "requests.jsonl"
    progress_file = BATCH_DIR / "progress.json"

    if not requests_file.exists():
        print("No requests file found. Run 'generate' first.")
        return

    # Count total requests
    total = 0
    with open(requests_file) as f:
        for _ in f:
            total += 1

    # Load progress
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

    if completed < total:
        print(f"\nResume with: python scripts/batch_rescore_claude.py run")


def analyze_agreement():
    """Analyze agreement between GPT-4o and Grok evaluations."""
    logs_dir = Path(__file__).parent.parent / "gui" / "logs"

    log_files = list(logs_dir.glob("*.json"))

    results = []
    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)

            if "evaluation" not in log or "evaluation_grok" not in log:
                continue

            gpt_score = log["evaluation"].get("score")
            grok_score = log["evaluation_grok"].get("score")

            if gpt_score is None or grok_score is None:
                continue

            results.append({
                "log_path": str(log_path),
                "case_id": log.get("case_id"),
                "llm_name": log.get("llm_name"),
                "gpt4o_score": gpt_score,
                "grok_score": grok_score,
                "agreement": gpt_score == grok_score
            })
        except Exception:
            continue

    if not results:
        print("No logs with both GPT-4o and Grok evaluations found.")
        print("Run 'python scripts/batch_rescore_claude.py run' first.")
        return

    print(f"{'='*60}")
    print(f"AGREEMENT ANALYSIS: GPT-4o vs Grok 4.1 Fast")
    print(f"{'='*60}")
    print(f"\nTotal logs with both evaluations: {len(results)}")

    # Overall agreement
    agreements = sum(1 for r in results if r["agreement"])
    print(f"Exact agreement: {agreements}/{len(results)} ({100*agreements/len(results):.1f}%)")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("                    Grok Score")
    print("                    0       1       2")
    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for r in results:
        matrix[r["gpt4o_score"]][r["grok_score"]] += 1

    for gpt_score in range(3):
        label = "GPT-4o  " if gpt_score == 1 else "        "
        print(f"   {label}{gpt_score}      {matrix[gpt_score][0]:3d}     {matrix[gpt_score][1]:3d}     {matrix[gpt_score][2]:3d}")

    # Score distribution
    print("\nScore Distributions:")
    gpt_dist = [sum(1 for r in results if r["gpt4o_score"] == s) for s in range(3)]
    claude_dist = [sum(1 for r in results if r["grok_score"] == s) for s in range(3)]
    print(f"  GPT-4o:  0={gpt_dist[0]:3d} ({100*gpt_dist[0]/len(results):4.1f}%)  1={gpt_dist[1]:3d} ({100*gpt_dist[1]/len(results):4.1f}%)  2={gpt_dist[2]:3d} ({100*gpt_dist[2]/len(results):4.1f}%)")
    print(f"  Grok:  0={claude_dist[0]:3d} ({100*claude_dist[0]/len(results):4.1f}%)  1={claude_dist[1]:3d} ({100*claude_dist[1]/len(results):4.1f}%)  2={claude_dist[2]:3d} ({100*claude_dist[2]/len(results):4.1f}%)")

    # Agreement by model being evaluated
    print("\nAgreement by Model Being Evaluated:")
    by_model = {}
    for r in results:
        model = r["llm_name"]
        if model not in by_model:
            by_model[model] = {"total": 0, "agree": 0}
        by_model[model]["total"] += 1
        if r["agreement"]:
            by_model[model]["agree"] += 1

    for model, counts in sorted(by_model.items(), key=lambda x: -x[1]["total"]):
        pct = 100 * counts["agree"] / counts["total"]
        print(f"  {model:35s} {counts['agree']:3d}/{counts['total']:3d} ({pct:5.1f}%)")

    # Cohen's Kappa
    try:
        from sklearn.metrics import cohen_kappa_score
        gpt_scores = [r["gpt4o_score"] for r in results]
        grok_scores = [r["grok_score"] for r in results]
        kappa = cohen_kappa_score(gpt_scores, grok_scores, weights="quadratic")
        print(f"\nCohen's Kappa (quadratic weighted): {kappa:.3f}")
    except ImportError:
        print("\n(Install sklearn for Cohen's Kappa: pip install scikit-learn)")

    # Save results
    results_file = BATCH_DIR / "agreement_analysis.json"
    with open(results_file, 'w') as f:
        json.dump({
            "total": len(results),
            "agreement_count": agreements,
            "agreement_pct": 100 * agreements / len(results),
            "confusion_matrix": matrix,
            "by_model": by_model,
            "results": results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Rescore logs using Grok 4.1 Fast via OpenRouter")
    parser.add_argument("command", choices=["generate", "run", "status", "analyze"],
                        help="Command to run")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate even for logs already evaluated")
    parser.add_argument("--start-from", type=int, default=0,
                        help="Start from this request index (for resuming)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of requests to process")
    parser.add_argument("--worker-id", type=int, default=None,
                        help="Worker ID for parallel execution (0-indexed)")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Total number of parallel workers")
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
