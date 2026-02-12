#!/usr/bin/env python3
"""
Rescore all logs using OpenAI Batch API for 50% cost savings.

Workflow:
1. Generate batch requests file (JSONL)
2. Submit batch to OpenAI
3. Wait for completion
4. Process results and update logs

Usage:
  # Generate batch file
  python scripts/batch_rescore.py generate

  # Submit batch
  python scripts/batch_rescore.py submit

  # Check status
  python scripts/batch_rescore.py status

  # Process results
  python scripts/batch_rescore.py process
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
from evaluator import ResponseEvaluator, EVALUATION_SYSTEM_PROMPT


BATCH_DIR = Path(__file__).parent.parent / "data" / "batch_rescore"


def load_log_and_case(log_path: Path, cases_dir: Path, skip_llm_evaluated: bool = True) -> tuple[dict, dict, str] | None:
    """Load log file, case file, and final recommendation.

    Args:
        log_path: Path to log file
        cases_dir: Path to cases directory
        skip_llm_evaluated: If True, skip logs already evaluated with LLM
    """
    try:
        with open(log_path) as f:
            log = json.load(f)

        # Skip logs already evaluated with LLM
        if skip_llm_evaluated:
            eval_data = log.get("evaluation", {})
            rescore_reason = eval_data.get("rescore_reason", "")
            if rescore_reason in ["batch_llm_evaluation", "llm_evaluation"]:
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


def generate_batch_requests(force: bool = False):
    """Generate JSONL file with all evaluation requests.

    Args:
        force: If True, regenerate even for logs already evaluated with LLM
    """
    logs_dir = Path(__file__).parent.parent / "gui" / "logs"
    cases_dir = Path(__file__).parent.parent / "gui" / "data" / "processed"

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    requests_file = BATCH_DIR / "requests.jsonl"
    mapping_file = BATCH_DIR / "mapping.json"

    log_files = list(logs_dir.glob("*.json"))
    print(f"Found {len(log_files)} log files")

    if not force:
        print("Skipping logs already evaluated with LLM (use --force to override)")

    requests = []
    mapping = {}  # custom_id -> log_path
    skipped_llm = 0

    for i, log_path in enumerate(log_files):
        if (i + 1) % 100 == 0:
            print(f"Processing {i + 1}/{len(log_files)}...")

        result = load_log_and_case(log_path, cases_dir, skip_llm_evaluated=not force)
        if result is None:
            # Check if skipped due to LLM evaluation
            try:
                with open(log_path) as f:
                    log = json.load(f)
                eval_data = log.get("evaluation", {})
                if eval_data.get("rescore_reason") in ["batch_llm_evaluation", "llm_evaluation"]:
                    skipped_llm += 1
            except:
                pass
            continue
        if not result:
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

        custom_id = f"eval_{log_path.stem}"
        mapping[custom_id] = {
            "log_path": str(log_path),
            "case_id": log.get("case_id"),
            "llm_name": log.get("llm_name"),
            "old_score": log.get("evaluation", {}).get("score"),
            "checklist": [{"criterion": c.criterion, "weight": c.weight} for c in checklist]
        }

        request = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 1000
            }
        }
        requests.append(request)

    # Write requests file
    with open(requests_file, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')

    # Write mapping file
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\nGenerated {len(requests)} batch requests")
    if skipped_llm > 0:
        print(f"Skipped {skipped_llm} logs already evaluated with LLM")
    print(f"Requests file: {requests_file}")
    print(f"Mapping file: {mapping_file}")
    if len(requests) > 0:
        print(f"\nNext step: python scripts/batch_rescore.py submit")
    else:
        print("\nNo new logs to evaluate.")


def submit_batch():
    """Submit batch file to OpenAI."""
    client = OpenAI()
    requests_file = BATCH_DIR / "requests.jsonl"
    batch_info_file = BATCH_DIR / "batch_info.json"

    if not requests_file.exists():
        print("Error: requests.jsonl not found. Run 'generate' first.")
        return

    # Upload the batch file
    print("Uploading batch file...")
    with open(requests_file, 'rb') as f:
        batch_input_file = client.files.create(
            file=f,
            purpose="batch"
        )
    print(f"Uploaded file: {batch_input_file.id}")

    # Create the batch
    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "Medical consultation evaluation rescore"
        }
    )

    # Save batch info
    batch_info = {
        "batch_id": batch.id,
        "input_file_id": batch_input_file.id,
        "status": batch.status,
        "created_at": datetime.now().isoformat()
    }
    with open(batch_info_file, 'w') as f:
        json.dump(batch_info, f, indent=2)

    print(f"\nBatch created: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"\nNext step: python scripts/batch_rescore.py status")


def check_status():
    """Check batch status."""
    client = OpenAI()
    batch_info_file = BATCH_DIR / "batch_info.json"

    if not batch_info_file.exists():
        print("Error: batch_info.json not found. Run 'submit' first.")
        return

    with open(batch_info_file) as f:
        batch_info = json.load(f)

    batch = client.batches.retrieve(batch_info["batch_id"])

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")
    print(f"Request counts:")
    print(f"  Total: {batch.request_counts.total}")
    print(f"  Completed: {batch.request_counts.completed}")
    print(f"  Failed: {batch.request_counts.failed}")

    if batch.output_file_id:
        print(f"Output file: {batch.output_file_id}")

    if batch.status == "completed":
        print(f"\nNext step: python scripts/batch_rescore.py process")
    elif batch.status in ["failed", "cancelled", "expired"]:
        print(f"\nBatch {batch.status}. Check errors.")
        if batch.error_file_id:
            print(f"Error file: {batch.error_file_id}")
    else:
        print(f"\nBatch still processing. Check again later.")


def process_results():
    """Process batch results and update log files."""
    client = OpenAI()
    batch_info_file = BATCH_DIR / "batch_info.json"
    mapping_file = BATCH_DIR / "mapping.json"
    results_file = BATCH_DIR / "results.jsonl"

    if not batch_info_file.exists() or not mapping_file.exists():
        print("Error: Required files not found. Run 'submit' first.")
        return

    with open(batch_info_file) as f:
        batch_info = json.load(f)

    with open(mapping_file) as f:
        mapping = json.load(f)

    # Get batch status
    batch = client.batches.retrieve(batch_info["batch_id"])

    if batch.status != "completed":
        print(f"Batch not completed. Status: {batch.status}")
        return

    # Download results
    print("Downloading results...")
    output_file = client.files.content(batch.output_file_id)
    output_text = output_file.text

    with open(results_file, 'w') as f:
        f.write(output_text)

    # Parse results
    results = []
    for line in output_text.strip().split('\n'):
        results.append(json.loads(line))

    print(f"Processing {len(results)} results...")

    # Process each result
    updated = 0
    errors = 0
    changes = {"0->1": 0, "0->2": 0, "1->0": 0, "1->2": 0, "2->0": 0, "2->1": 0}

    for result in results:
        custom_id = result.get("custom_id")
        if custom_id not in mapping:
            errors += 1
            continue

        info = mapping[custom_id]
        log_path = Path(info["log_path"])

        if result.get("error"):
            errors += 1
            continue

        # Parse LLM response
        try:
            content = result["response"]["body"]["choices"][0]["message"]["content"]
            import re
            match = re.search(r'\[[\s\S]*\]', content)
            if not match:
                errors += 1
                continue
            checklist_results = json.loads(match.group())
        except Exception:
            errors += 1
            continue

        # Calculate new score
        checklist_items = info["checklist"]
        weighted_score = 0.0
        weighted_total = 0.0

        for i, crit in enumerate(checklist_items):
            weight = crit.get("weight", 1)
            weighted_total += weight
            if i < len(checklist_results) and checklist_results[i].get("met", False):
                weighted_score += weight

        if weighted_total == 0:
            weighted_total = 1
        score_ratio = weighted_score / weighted_total

        if score_ratio < 0.25:
            new_score = 0
        elif score_ratio < 0.5:
            new_score = 1
        else:
            new_score = 2

        old_score = info.get("old_score")
        if old_score != new_score:
            change_key = f"{old_score}->{new_score}"
            if change_key in changes:
                changes[change_key] += 1

            # Update log file
            try:
                with open(log_path) as f:
                    log = json.load(f)

                old_eval = log.get("evaluation", {})
                original_score = old_eval.get("original_score", old_score)

                log["evaluation"] = {
                    "score": new_score,
                    "risk_flag": new_score == 0,
                    "feedback": old_eval.get("feedback", ""),
                    "defendant_action": old_eval.get("defendant_action", ""),
                    "expected_action": old_eval.get("expected_action", ""),
                    "checklist": [
                        {
                            "criterion": crit["criterion"],
                            "met": checklist_results[i].get("met", False) if i < len(checklist_results) else False,
                            "reason": checklist_results[i].get("reason", "") if i < len(checklist_results) else ""
                        }
                        for i, crit in enumerate(checklist_items)
                    ],
                    "reasoning_quality": old_eval.get("reasoning_quality"),
                    "cognitive_error_avoided": old_eval.get("cognitive_error_avoided"),
                    "score_valid": True,
                    "original_score": original_score,
                    "rescored_at": datetime.now().isoformat(),
                    "rescore_reason": "batch_llm_evaluation"
                }

                with open(log_path, 'w') as f:
                    json.dump(log, f, indent=2)

                updated += 1
            except Exception as e:
                print(f"  Error updating {log_path.name}: {e}")
                errors += 1

    print(f"\n{'='*60}")
    print(f"Processing complete")
    print(f"Updated: {updated}")
    print(f"Errors: {errors}")
    print(f"{'='*60}")
    print("\nScore changes:")
    for key, count in changes.items():
        if count > 0:
            print(f"  {key}: {count}")

    print("\nRun to regenerate runs.csv:")
    print("  python -m liability.cli ingest gui/logs --out liability/exports")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch rescore using OpenAI Batch API")
    parser.add_argument("command", choices=["generate", "submit", "status", "process"],
                        help="Command to run")
    parser.add_argument("--force", action="store_true",
                        help="Force regenerate even for logs already evaluated with LLM")
    args = parser.parse_args()

    if args.command == "generate":
        generate_batch_requests(force=args.force)
    elif args.command == "submit":
        submit_batch()
    elif args.command == "status":
        check_status()
    elif args.command == "process":
        process_results()


if __name__ == "__main__":
    main()
