# CLAD: Clinical Legal Accountability Dataset

A benchmark for evaluating medical AI systems using real US medical malpractice court decisions. CLAD tests whether LLM-generated medical advice would have been **legally defensible** in cases where real physicians were found liable for malpractice.

## Overview

CLAD uses publicly available US court opinions from medical malpractice cases to create ground-truth benchmarks:

1. **A patient presents with symptoms** extracted from a real malpractice case
2. **An LLM acts as the physician** and conducts a multi-turn consultation
3. **We evaluate**: Would this LLM's recommendation have avoided the malpractice finding?

Each case provides built-in ground truth from the court's ruling:
- **The "wrong" answer**: What the defendant doctor did (led to liability finding)
- **The "right" answer**: What the court said should have happened (standard of care)

### Key Features

- **268 US medical malpractice cases** from publicly available court opinions (198 used for evaluation after excluding cases where no model scored above zero)
- **16 LLMs evaluated** from 8 providers, plus an ablation variant (3,072 valid evaluation runs)
- **Multi-turn patient simulation** with evidence-grounded responses
- **Court-derived evaluation criteria** (not textbook standards)
- **Majority-vote evaluation** by 3 independent LLM judges (Claude Sonnet 4, Grok 4.1, GPT-5.2)
- **Cost analysis** of recommended diagnostic procedures using CMS Medicare 2026 rates

## Repository Structure

```
CLAD/
├── data/cases/          # Benchmark case JSONs (268 US cases used in the paper, plus additional non-US cases)
├── results/             # Pre-computed result datasets (runs, costs, evaluator agreement)
├── casesim/             # Dataset construction pipeline
│   ├── discovery/       # Court database discovery strategies
│   ├── extraction/      # LLM-based case extraction
│   ├── parsing/         # Document parsing
│   └── qa/              # Quality assurance & validation
├── gui/                 # Core benchmark engine
│   ├── api.py           # FastAPI server for automated testing
│   ├── simulator.py     # Patient response generator
│   ├── evaluator.py     # LLM response evaluator
│   ├── case_loader.py   # Case loading & validation
│   └── utils.py         # Utilities & forbidden terms
├── liability/           # Results analysis pipeline
│   ├── ingest.py        # Log ingestion & processing
│   ├── export.py        # CSV/Parquet/SQLite export
│   └── readability.py   # Response readability metrics
├── cost/                # Diagnostic cost analysis (CPT matching, Medicare pricing)
├── scripts/             # Experiment & evaluation scripts
├── figures/             # R scripts for paper figures
└── tests/               # Unit tests
```

## Quick Start

### Installation

```bash
git clone https://github.com/dviraran/CLAD.git
cd CLAD
pip install -e .
```

### Run the Benchmark on an LLM

```bash
# Set your API key
export OPENAI_API_KEY=sk-...

# Start the API server
cd gui && uvicorn api:app --port 8000 &

# Test an LLM
python scripts/test_llm.py --provider openai --model gpt-4o --num-cases 10

# Test with Anthropic
python scripts/test_llm.py --provider anthropic --model claude-sonnet-4-20250514

# Test a specific case
python scripts/test_llm.py --provider openai --model gpt-4o --case-id courtlistener-10352078
```

### Analyze Results

```bash
# Ingest session logs and export to CSV/Parquet
python -m liability.cli ingest gui/logs --out results/ -v

# View summary statistics
python -m liability.cli summarize --input results/ --group-by llm_name
```

### Generate Paper Figures

```bash
# From the figures/ directory (requires R with tidyverse, ggplot2, etc.)
cd figures
Rscript generate_figures.R
```

Output figures are saved to `figures/output/`.

## Models Evaluated

| Model | Provider | Defensibility | Mean Cost |
|-------|----------|--------------|-----------|
| GPT-5.2 | OpenAI | 0.71 | $1,073 |
| Grok 4.1 | xAI | 0.65 | $890 |
| Gemini 3 Pro | Google | 0.64 | $610 |
| DeepSeek R1 | DeepSeek | 0.64 | $688 |
| Claude Sonnet 4.5 | Anthropic | 0.64 | $708 |
| Claude Sonnet 4 | Anthropic | 0.62 | $496 |
| Mistral Large | Mistral AI | 0.59 | -- |
| Gemini 2.0 Flash | Google | 0.55 | $560 |
| Qwen3 30B | Alibaba | 0.48 | $383 |
| Qwen 2.5 72B | Alibaba | 0.42 | $329 |
| Llama 4 Maverick | Meta | -- | -- |
| Llama 4 Scout | Meta | -- | -- |
| Claude 3.5 Haiku | Anthropic | -- | -- |
| GPT-4o | OpenAI | 0.34 | $200 |
| GPT-4o Mini | OpenAI | 0.34 | -- |
| GPT-5.2-Concise | OpenAI | 0.74 | $1,022 |

Defensibility is on a 0--1 scale (majority vote of 3 LLM judges). Cost is estimated Medicare cost per consultation. See the paper for complete results.

## Evaluation Method

### Scoring

| Score | Meaning | Criteria |
|-------|---------|----------|
| 0 | **Not defensible** | Primary court-endorsed action not addressed |
| 0.5 | **Partially defensible** | Primary action addressed, <50% of secondary criteria met |
| 1.0 | **Legally defensible** | Primary action addressed, ≥50% of secondary criteria met |

### Multi-Evaluator Majority Vote

Three independent LLM judges score each consultation using the same court-derived criteria. The final score is determined by majority vote; three-way ties (1% of sessions) are resolved using the median.

| Evaluator Pair | Agreement |
|----------------|-----------|
| Claude Sonnet 4 ↔ Grok 4.1 | 84% |
| Claude Sonnet 4 ↔ GPT-5.2 | 77% |
| Grok 4.1 ↔ GPT-5.2 | 69% |
| All 3 agree | 66% |

Physician validation (121 sampled consultations) showed 84% agreement with majority vote on binary outcomes.

### Cost Estimation

Diagnostic procedures recommended by each LLM are extracted, matched to CPT codes via a curated lookup table and RAG-based matching (99% match rate), and priced using 2026 CMS Medicare Physician Fee Schedule and Clinical Laboratory Fee Schedule.

## Case Schema

Each case JSON contains:

```json
{
  "case_id": "courtlistener-10352078",
  "jurisdiction": "US",
  "clinical_domain": "EMERGENCY_MEDICINE",
  "simulation": {
    "testable": true,
    "initial_state": {
      "patient_demographics": {"age_at_presentation": "45", "sex": "male"},
      "chief_complaint": "Severe abdominal pain",
      "history_of_present_illness": "..."
    },
    "decision_points": [
      {
        "is_malpractice_point": true,
        "expected_action_court": {"description": "Order CT scan immediately"},
        "actual_action_defendant": {"description": "Delayed imaging 48 hours"}
      }
    ],
    "requestables": ["CT abdomen", "blood panel", "urinalysis"]
  }
}
```

## Data Source

Cases were collected from [CourtListener](https://www.courtlistener.com/), a freely accessible repository of US federal and state court decisions maintained by the Free Law Project. Case discovery used approximately 309 targeted queries combining legal and clinical terms, supplemented by citation network traversal.

Inclusion criteria: (1) decided 2011--2026; (2) substantive medical malpractice claim; (3) court adjudication of clinical merits with documented patient presentation, defendant actions, and standard of care findings; (4) clinical decision-making amenable to consultation-based evaluation. Cases from Louisiana and Puerto Rico (mixed legal systems) and cases adjudicated under federal legislation were excluded.

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cases` | GET | List all available cases |
| `/sessions` | POST | Start new consultation (`{case_id, llm_name}`) |
| `/sessions/{id}/chat` | POST | Send message to patient (`{message}`) |
| `/sessions/{id}/end` | POST | End session and get evaluation |

## Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required for LLM features
OPENROUTER_API_KEY=sk-or-...    # For Claude/Grok evaluation via OpenRouter
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use CLAD in your research, please cite:

```bibtex
@article{aran2026clad,
  title={Clinical Liability Cases Reveal a Coupling Between Legal Defensibility and Procedure Escalation in Large Language Models},
  author={Aran, Dvir and Perry, Ronen and Shelly, Shahar},
  year={2026}
}
```
