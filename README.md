# The Self-Pruning Neural Network  
Tredence Analytics — AI Engineering Internship Case Study (2025 Cohort)

**Akshat Bakshi**  
akshatbakshi93@gmail.com  
[GitHub](https://www.github.com/akshat448) | [LinkedIn](https://www.linkedin.com/in/abakshi05) | [Resume](https://drive.google.com/file/d/1ySFJiv_wmWfkbg4JD32QLndwMB-D38Gf/view?usp=sharing)

---

## Overview

This project implements a self-pruning neural network that learns to remove its own weights during training. Each weight is associated with a learnable gate parameter, enabling the model to dynamically determine which connections are necessary.

The objective is to jointly optimize classification performance and model sparsity, eliminating the need for post-training pruning.

---

## Problem

Neural network deployment is often constrained by memory and computational limits. Traditional pruning methods operate after training and require manual intervention.

This work addresses the problem by integrating pruning directly into the training process. The model learns a sparse structure by driving unnecessary connections toward zero through a regularized objective.

---

## Methodology

### Prunable Layer

A custom `PrunableLinear` layer is implemented with:

- Standard weight and bias parameters  
- A learnable `gate_scores` tensor of the same shape as weights  

During the forward pass:

```

gates = sigmoid(gate_scores)
pruned_weights = weight * gates
output = F.linear(x, pruned_weights, bias)

```

This ensures gradients flow through both weights and gates.

---

### Loss Function

The training objective is defined as:

```

Total Loss = CrossEntropyLoss + λ · Σ sigmoid(gate_scores)

````

- The L1 penalty on gate activations encourages sparsity  
- The hyperparameter λ controls the trade-off between accuracy and pruning strength  

---

### Hyperparameter Search

A two-stage search is used to identify optimal λ:

1. **Coarse Search**  
   Log-scale sweep over 15 values  

2. **Fine Search**  
   Linear sweep over 50 values within the best region  

Training includes:

- Early stopping  
- Automatic Mixed Precision (AMP)  
- Parallel execution with caching  

---

## Setup

**Requirements**

- Python 3.10+  
- CUDA-enabled GPU (recommended)  

```bash
git clone https://github.com/akshat448/tredence-self-pruning.git
cd tredence-self-pruning

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
````

---

## Running

```bash
jupyter notebook tredence_self_pruning_final.ipynb
```

Run all cells sequentially. CIFAR-10 is downloaded automatically.

---

## Outputs

The pipeline generates:

* `case_study_report.md` — analysis and results
* `self_pruning_results.png` — gate distribution and accuracy–sparsity plot
* `experiment_log.csv` — full λ search results
* `champion_model.pt` — best model checkpoint

---

## Results

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
| :----- | :---------------- | :----------------- |
| 0.0034 | 64.74             | 0.0                |
| 0.0101 | 64.69             | 57.1               |
| 0.0117 | 64.50             | 100.0              |

---

## Analysis

The model exhibits a sharp transition in sparsity behavior:

* Low λ values result in negligible sparsity
* Slight increases in λ lead to abrupt and often complete pruning

This indicates a bimodal pruning regime rather than gradual sparsification.

### Root Cause

* The sigmoid transformation constrains gates to the range (0, 1)
* L1 regularization is applied on post-sigmoid values
* This leads to either weak gradients (no pruning) or saturation (collapse)

As a result, the model fails to achieve controlled sparsity.

---

## Limitations

* Lack of smooth sparsity–accuracy trade-off
* Sensitivity to λ selection
* Instability due to sigmoid-based gating

---

## Future Work

* Apply L1 regularization on raw gate parameters instead of sigmoid outputs
* Explore L0-based methods such as Hard Concrete gates
* Introduce temperature scaling for smoother gate transitions
* Improve architectural capacity to maintain accuracy under pruning

---

## About

I am a third-year ECE undergraduate at Thapar Institute of Engineering and Technology, with a focus on machine learning, deep learning, and AI systems engineering.

My work spans model training pipelines, retrieval-augmented generation systems, and backend engineering. I have built systems using FastAPI, vector databases, and modern embedding models, and have worked on multimodal time-series prediction and transformer-based architectures.

I have also participated in large-scale ML competitions and advanced training programs, focusing on building efficient, scalable AI systems.
