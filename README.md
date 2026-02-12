# CLAD: Clinical Legal Accountability Dataset

A benchmark for evaluating medical AI systems using real malpractice court decisions. CLAD tests whether LLM-generated medical advice would have been **legally defensible** in cases where real physicians were found liable for malpractice.

## Overview

CLAD uses publicly available court judgments from medical malpractice cases to create ground-truth benchmarks:

1. **A patient presents with symptoms** extracted from a real malpractice case
2. **An LLM acts as the physician** and provides a consultation
3. **We evaluate**: Would this LLM's recommendation have avoided the malpractice finding?

Each case provides built-in ground truth from the court's ruling:
- **The "wrong" answer**: What the defendant doctor did (led to liability finding)
- **The "right" answer**: What the court said should have happened (standard of care)

### Key Features

- **276 cases** from 5 jurisdictions (UK, US, Australia, New Zealand, Canada)
- **Multi-turn patient simulation** with evidence-grounded responses
- **Court-derived evaluation criteria** (not textbook standards)
- **Multi-evaluator validation** (GPT-4o, Claude Sonnet 4, Grok 4.1, GPT-5.2)
- **Cost analysis** of recommended diagnostic procedures

## Repository Structure

```
CLAD/
├── data/cases/          # 276 benchmark case JSONs
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
├── cost/                # Diagnostic cost analysis
├── scripts/             # Experiment & evaluation scripts
├── figures/             # R scripts for paper figures
├── results/             # Pre-computed result datasets
├── config/              # Configuration files
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
python scripts/test_llm.py --provider openai --model gpt-4o --case-id bailii-qb-2021-169-html
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

## Evaluation Method

### Scoring (0-2 Scale)

| Score | Meaning | Criteria |
|-------|---------|----------|
| 0 | **Not defensible** | Primary action not met |
| 1 | **Partially defensible** | Primary action met, <50% secondary criteria met |
| 2 | **Legally defensible** | Primary action met, ≥50% secondary criteria met |

### Multi-Evaluator Validation

Four independent LLM evaluators score each consultation using the same court-derived criteria:

| Evaluator | Pairwise Agreement |
|-----------|--------------------|
| Claude Sonnet 4 ↔ Grok 4.1 | ~85% |
| GPT-5.2 ↔ Claude Sonnet 4 | ~83% |
| GPT-4o matches majority | ~85% |
| All 4 agree | ~60% (5.6x chance) |

## Case Schema

Each case JSON contains:

```json
{
  "case_id": "bailii-qb-2021-169-html",
  "jurisdiction": "UK",
  "clinical_domain": "SURGERY_GENERAL",
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

## Data Sources

| Source | Jurisdiction | Cases |
|--------|-------------|-------|
| BAILII | UK, Ireland | Primary |
| CourtListener | US | Active |
| AustLII | Australia | Active |
| NZLII | New Zealand | Active |
| CanLII | Canada | Limited |

## Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required for LLM features
OPENROUTER_API_KEY=sk-or-...    # For Claude/Grok evaluation via OpenRouter
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/cases` | GET | List all available cases |
| `/sessions` | POST | Start new consultation |
| `/sessions/{id}/chat` | POST | Send message to patient |
| `/sessions/{id}/end` | POST | End session and get evaluation |

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

If you use CLAD in your research, please cite:

```bibtex
@article{clad2026,
  title={CLAD: Clinical Legal Accountability Dataset for Evaluating Medical AI},
  author={Aran, Dvir},
  year={2026}
}
```
