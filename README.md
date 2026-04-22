# The Self-Pruning Neural Network  
Tredence Analytics — AI Engineering Internship Case Study (2025 Cohort)

**Akshat Bakshi** | akshatbakshi93@gmail.com | [GitHub](https://www.github.com/akshat448) | [LinkedIn](https://www.linkedin.com/in/abakshi05)

---

## Overview

A feed-forward neural network trained on CIFAR-10 that learns to prune its own weights during training. Each weight has an associated learnable gate parameter. An L1 sparsity penalty drives most gates to zero, producing a compressed sparse network without a separate pruning step.

---

## Setup

Requirements: Python 3.10+, CUDA GPU

```bash
git clone https://github.com/<your-username>/tredence-self-pruning.git
cd tredence-self-pruning

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt`
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
jupyter>=1.0.0
```

---

## Running

```bash
jupyter notebook tredence_self_pruning_final.ipynb
```

Run cells top to bottom. CIFAR-10 downloads automatically on first run.

---

## Outputs

The notebook produces:

- `case_study_report.md` — markdown report with L1 sparsity explanation, results table, and analysis
- `self_pruning_results.png` — gate value distribution plot and accuracy vs sparsity curve
- `experiment_log.csv` — results across all lambda experiments
- `champion_model.pt` — best model checkpoint

---

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| :--- | :--- | :--- |
| Low   | | |
| Medium | | |
| High  | | |

> Fill in after running the notebook.