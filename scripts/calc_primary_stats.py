#!/usr/bin/env python3
"""Calculate all statistics needed for paper v2 using primary-action scoring."""

import json
from pathlib import Path
from collections import defaultdict
import statistics

LOGS_DIR = Path(__file__).parent.parent / "gui" / "logs"

def main():
    log_files = list(LOGS_DIR.glob("*.json"))

    # Collect data
    by_model = defaultdict(lambda: {"scores": [], "lengths": [], "costs": []})
    by_category = defaultdict(list)
    by_jurisdiction = defaultdict(list)
    all_scores = []

    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)

            eval_primary = log.get("evaluation_primary", {})
            eval_orig = log.get("evaluation", {})

            if not eval_primary:
                continue

            score = eval_primary.get("score", -1)
            if score < 0:
                continue

            # Normalize score to 0-1 scale
            score_normalized = score / 2.0

            llm_name = log.get("llm_name", "unknown")
            # Simplify model names
            if "gpt-5.2" in llm_name.lower():
                model = "GPT-5.2"
            elif "gpt-4o" in llm_name.lower():
                model = "GPT-4o"
            elif "claude" in llm_name.lower():
                model = "Claude Sonnet 4"
            elif "grok" in llm_name.lower():
                model = "Grok 4.1"
            elif "gemini" in llm_name.lower():
                model = "Gemini 3 Pro"
            elif "llama" in llm_name.lower():
                model = "Llama 4"
            elif "qwen" in llm_name.lower():
                model = "Qwen 2.5 72B"
            else:
                model = llm_name

            by_model[model]["scores"].append(score_normalized)
            all_scores.append(score_normalized)

            # Get response length if available
            response = log.get("final_recommendation", "")
            if response:
                by_model[model]["lengths"].append(len(response))

            # Get cost if available
            cost = log.get("procedure_cost", {})
            if isinstance(cost, dict):
                total_cost = cost.get("total_cost", 0)
            else:
                total_cost = 0
            by_model[model]["costs"].append(total_cost)

            # Category
            case_id = log.get("case_id", "")
            category = log.get("malpractice_type", "unknown")
            by_category[category].append(score_normalized)

            # Jurisdiction
            if "bailii" in case_id.lower():
                jurisdiction = "UK"
            elif "courtlistener" in case_id.lower():
                jurisdiction = "US"
            elif "nzlii" in case_id.lower():
                jurisdiction = "NZ"
            else:
                jurisdiction = "Unknown"
            by_jurisdiction[jurisdiction].append(score_normalized)

        except Exception as e:
            pass

    # Print results
    print("=" * 70)
    print("PRIMARY-ACTION SCORING STATISTICS FOR PAPER V2")
    print("=" * 70)

    print("\n## Overall Statistics")
    print(f"Total evaluations: {len(all_scores)}")
    print(f"Mean defensibility: {statistics.mean(all_scores):.2f}")

    # Score distribution
    score_0 = sum(1 for s in all_scores if s == 0)
    score_05 = sum(1 for s in all_scores if s == 0.5)
    score_1 = sum(1 for s in all_scores if s == 1.0)
    print(f"\nScore distribution:")
    print(f"  Score 0 (Liability Likely): {score_0} ({score_0/len(all_scores)*100:.0f}%)")
    print(f"  Score 0.5 (Partial): {score_05} ({score_05/len(all_scores)*100:.0f}%)")
    print(f"  Score 1.0 (Defensible): {score_1} ({score_1/len(all_scores)*100:.0f}%)")

    print("\n## Model Performance (sorted by mean score)")
    print("-" * 70)
    print(f"{'Model':<20} {'N':>6} {'Mean':>8} {'Def%':>8} {'Chars':>8} {'Cost':>10}")
    print("-" * 70)

    model_stats = []
    for model, data in by_model.items():
        if len(data["scores"]) < 10:
            continue
        mean_score = statistics.mean(data["scores"])
        def_pct = sum(1 for s in data["scores"] if s == 1.0) / len(data["scores"]) * 100
        mean_len = statistics.mean(data["lengths"]) if data["lengths"] else 0
        mean_cost = statistics.mean(data["costs"]) if data["costs"] else 0
        model_stats.append({
            "model": model,
            "n": len(data["scores"]),
            "mean": mean_score,
            "def_pct": def_pct,
            "chars": mean_len,
            "cost": mean_cost
        })

    model_stats.sort(key=lambda x: x["mean"], reverse=True)

    for m in model_stats:
        print(f"{m['model']:<20} {m['n']:>6} {m['mean']:>8.2f} {m['def_pct']:>7.0f}% {m['chars']:>8.0f} ${m['cost']:>9.0f}")

    print("\n## For Abstract (top models)")
    for m in model_stats[:3]:
        print(f"  {m['model']}: {m['mean']:.2f} mean defensibility")

    print("\n## For Results - Score Distribution")
    print(f"  {score_1/len(all_scores)*100:.0f}% achieved full defensibility (score 1.0)")
    print(f"  {score_05/len(all_scores)*100:.0f}% achieved partial defensibility (score 0.5)")
    print(f"  {score_0/len(all_scores)*100:.0f}% were scored as liability likely (score 0)")
    print(f"  Overall mean defensibility: {statistics.mean(all_scores):.2f}")

    print("\n## By Category")
    print("-" * 50)
    cat_stats = []
    for cat, scores in by_category.items():
        if len(scores) >= 5:
            cat_stats.append((cat, statistics.mean(scores), len(scores)))
    cat_stats.sort(key=lambda x: x[1], reverse=True)
    for cat, mean, n in cat_stats[:10]:
        print(f"  {cat}: {mean:.2f} (n={n})")

    print("\n## By Jurisdiction")
    print("-" * 50)
    for jur, scores in sorted(by_jurisdiction.items()):
        if len(scores) >= 10:
            print(f"  {jur}: {statistics.mean(scores):.2f} (n={len(scores)})")

    print("\n## Supplementary Table 3 Data")
    print("-" * 70)
    print("Model & N & Score & Def% & Chars & Words & Reasoning & Read. & Cost & CPDR \\\\")
    for m in model_stats:
        words = int(m['chars'] / 5)
        # CPDR = cost / (defensible count / total count) approximately
        cpdr = m['cost'] / (m['def_pct']/100) if m['def_pct'] > 0 else 0
        print(f"{m['model']} & {m['n']} & {m['mean']:.2f} & {m['def_pct']:.0f}\\% & {m['chars']:.0f} & {words} & -- & -- & \\${m['cost']:.0f} & \\${cpdr:.0f} \\\\")


if __name__ == "__main__":
    main()
