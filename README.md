# ADS_thesis_final_code

# Reducing Demographic Representation Bias in Humanitarian Summarization using a Masking–Balancing–Generating (MBG) Architecture

This repository contains the implementation accompanying the master's thesis:

> **Reducing Demographic Representation Bias in Humanitarian Summarization using a Masking–Balancing–Generating (MBG) Architecture**

The project investigates whether a multi-agent preprocessing architecture can reduce demographic representation bias in humanitarian text summarization while maintaining acceptable summary quality and computational costs.

---

## Repository Overview

The repository contains implementations of:

- **Single-Pass Baseline** – standard single-agent summarization.
- **Reflective Baseline** – summarization followed by a self-reflection and revision step.
- **MBG Architecture** – a multi-stage pipeline consisting of:
  - Masking
  - Rule-based normalization
  - Balancing
  - Summary generation
- **MBG Ablations**
  - No Masking
  - No Balancing
  - Generator Only
- Evaluation scripts for summary quality, fairness, hedging behaviour, and computational cost.

---

## Repository Structure

```
.
├── baseline1_plain.py          # Single-pass baseline
├── baseline2_reflect.py        # Reflective baseline
├── mbg_architecture.py         # Full MBG pipeline
├── grr_architecture.py         # GRR implementation
├── grr_ablations.py            # GRR ablations
├── prompts.py                  # Prompt templates
├── normalization.py            # Rule-based normalization
├── compute_metrics.py          # Evaluation pipeline
├── run_evaluation.py           # Executes experiments
├── config.py                   # Model and experiment configuration
├── token_utils.py              # Token accounting utilities
├── data/                       # Input datasets
├── results/                    # Generated outputs and metrics
└── README.md
```

---

## Requirements

The experiments were developed using **Python 3.11**.

Main dependencies include:

- pandas
- numpy
- openpyxl
- python-dotenv
- scikit-learn
- spaCy
- sentence-transformers
- DeepEval
- LangChain
- Ollama
- OpenAI Python SDK

Depending on the selected model provider, either:

- OpenAI API
- Ollama

must be available.

---

## Configuration

Experiment settings are configured in `config.py`.

This includes:

- language model provider
- model name
- API configuration
- generation parameters
- experiment settings

---

## Running the Experiments

Run the experimental pipeline:

```bash
python run_evaluation.py
```

After generation, compute the evaluation metrics:

```bash
python compute_metrics.py
```

Results are written to the `results/` directory.

---

## Evaluation

The implementation evaluates generated summaries using:

- DeepEval Summarization metrics
  - Alignment
  - Coverage
  - Summarization score
- Equal Coverage (EC)
- Hedging analysis
- Runtime
- Token usage
- Number of model requests

---

## Notes

This repository was developed as part of an academic master's thesis.

The implementation is intended to reproduce the experiments reported in the thesis and therefore prioritizes research reproducibility over production-level software engineering. Some scripts are configured for specific experimental settings described in the accompanying thesis.

---

## Thesis

For methodological details, experimental design, evaluation metrics, and discussion of the results, please refer to the accompanying master's thesis.

---

## License

This repository is provided for academic and research purposes.
