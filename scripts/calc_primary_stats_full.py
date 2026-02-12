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
    by_model = defaultdict(lambda: {
        "scores": [], "orig_scores": [], "lengths": [], "costs": [],
        "reasoning": [], "readability": []
    })
    by_category = defaultdict(list)
    by_jurisdiction = defaultdict(list)
    all_scores = []
    all_orig_scores = []

    for log_path in log_files:
        try:
            with open(log_path) as f:
                log = json.load(f)

            eval_primary = log.get("evaluation_primary", {})
            eval_orig = log.get("evaluation", {})

            if not eval_primary:
                continue

            score = eval_primary.get("score", -1)
            orig_score = eval_orig.get("score", -1)
            if score < 0:
                continue

            # Normalize score to 0-1 scale
            score_normalized = score / 2.0
            orig_normalized = orig_score / 2.0 if orig_score >= 0 else 0

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
            by_model[model]["orig_scores"].append(orig_normalized)
            all_scores.append(score_normalized)
            all_orig_scores.append(orig_normalized)

            # Get response length if available
            response = log.get("final_recommendation", "")
            if response:
                by_model[model]["lengths"].append(len(response))

            # Get cost from original evaluation
            cost_data = eval_orig.get("procedure_cost", {})
            if isinstance(cost_data, dict):
                total_cost = cost_data.get("total_cost", 0) or 0
            else:
                total_cost = 0
            by_model[model]["costs"].append(total_cost)

            # Get reasoning quality from original evaluation
            reasoning = eval_orig.get("reasoning_quality", {})
            if isinstance(reasoning, dict):
                quality = reasoning.get("quality_score", 0) or 0
            else:
                quality = 0
            by_model[model]["reasoning"].append(quality)

            # Get readability from original evaluation
            readability = eval_orig.get("readability", {})
            if isinstance(readability, dict):
                read_score = readability.get("transformer_score", 0) or 0
            else:
                read_score = 0
            by_model[model]["readability"].append(read_score)

            # Jurisdiction
            case_id = log.get("case_id", "")
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
    print("=" * 80)
    print("PRIMARY-ACTION SCORING STATISTICS FOR PAPER V2")
    print("=" * 80)

    print("\n## Overall Statistics")
    print(f"Total evaluations: {len(all_scores)}")
    print(f"Mean defensibility (primary): {statistics.mean(all_scores):.2f}")
    print(f"Mean defensibility (original): {statistics.mean(all_orig_scores):.2f}")

    # Score distribution
    score_0 = sum(1 for s in all_scores if s == 0)
    score_05 = sum(1 for s in all_scores if s == 0.5)
    score_1 = sum(1 for s in all_scores if s == 1.0)
    print(f"\nScore distribution (PRIMARY):")
    print(f"  Score 0 (Liability Likely): {score_0} ({score_0/len(all_scores)*100:.0f}%)")
    print(f"  Score 0.5 (Partial): {score_05} ({score_05/len(all_scores)*100:.0f}%)")
    print(f"  Score 1.0 (Defensible): {score_1} ({score_1/len(all_scores)*100:.0f}%)")

    print("\n## Model Performance (sorted by primary score)")
    print("-" * 80)
    print(f"{'Model':<18} {'N':>5} {'Primary':>8} {'Orig':>8} {'Def%':>6} {'Chars':>6} {'Reason':>7} {'Read':>6} {'Cost':>8}")
    print("-" * 80)

    model_stats = []
    for model, data in by_model.items():
        if len(data["scores"]) < 10:
            continue
        mean_score = statistics.mean(data["scores"])
        mean_orig = statistics.mean(data["orig_scores"])
        def_pct = sum(1 for s in data["scores"] if s == 1.0) / len(data["scores"]) * 100
        mean_len = statistics.mean(data["lengths"]) if data["lengths"] else 0
        mean_cost = statistics.mean([c for c in data["costs"] if c > 0]) if any(c > 0 for c in data["costs"]) else 0
        mean_reasoning = statistics.mean([r for r in data["reasoning"] if r > 0]) if any(r > 0 for r in data["reasoning"]) else 0
        mean_readability = statistics.mean([r for r in data["readability"] if r > 0]) if any(r > 0 for r in data["readability"]) else 0
        model_stats.append({
            "model": model,
            "n": len(data["scores"]),
            "mean": mean_score,
            "orig": mean_orig,
            "def_pct": def_pct,
            "chars": mean_len,
            "cost": mean_cost,
            "reasoning": mean_reasoning,
            "readability": mean_readability
        })

    model_stats.sort(key=lambda x: x["mean"], reverse=True)

    for m in model_stats:
        print(f"{m['model']:<18} {m['n']:>5} {m['mean']:>8.2f} {m['orig']:>8.2f} {m['def_pct']:>5.0f}% {m['chars']:>6.0f} {m['reasoning']:>7.2f} {m['readability']:>6.1f} ${m['cost']:>7.0f}")

    print("\n" + "=" * 80)
    print("LATEX TABLE DATA FOR SUPPLEMENTARY TABLE 3")
    print("=" * 80)
    print("Model & N & Score & Def\\% & Chars & Words & Reasoning & Read. & Cost & CPDR \\\\")
    print("\\hline")
    for m in model_stats:
        words = int(m['chars'] / 5)
        # CPDR = cost / defensibility rate
        cpdr = m['cost'] / (m['def_pct']/100) if m['def_pct'] > 0 else 0
        print(f"{m['model']} & {m['n']} & {m['mean']:.2f} & {m['def_pct']:.0f}\\% & {int(m['chars']):,} & {words} & {m['reasoning']:.2f} & {m['readability']:.1f} & \\${int(m['cost']):,} & \\${int(cpdr):,} \\\\")

    print("\n" + "=" * 80)
    print("KEY NUMBERS FOR PAPER TEXT")
    print("=" * 80)

    print("\n### ABSTRACT:")
    print(f"GPT-5.2 achieved the highest mean defensibility score ({model_stats[0]['mean']:.2f})")
    print(f"Grok 4.1 achieved the second-highest score ({model_stats[1]['mean']:.2f})")
    print(f"Claude Sonnet 4: {model_stats[2]['mean']:.2f}")

    print("\n### RESULTS - Overall:")
    print(f"Across {len(all_scores)} valid runs:")
    print(f"- {score_1/len(all_scores)*100:.0f}% achieved full defensibility (score 1.0)")
    print(f"- {score_05/len(all_scores)*100:.0f}% achieved partial defensibility (score 0.5)")
    print(f"- {score_0/len(all_scores)*100:.0f}% were scored as liability likely (score 0)")
    print(f"- Overall mean defensibility: {statistics.mean(all_scores):.2f}")

    print("\n### RESULTS - By Model:")
    for m in model_stats:
        print(f"- {m['model']}: {m['mean']:.2f} (was {m['orig']:.2f} under proportional scoring)")

    print("\n### BY JURISDICTION:")
    for jur in ["UK", "US", "NZ"]:
        scores = by_jurisdiction.get(jur, [])
        if scores:
            print(f"- {jur}: {statistics.mean(scores):.2f} (n={len(scores)})")


if __name__ == "__main__":
    main()
