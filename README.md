# TinyML Acoustic Monitoring — USM Undergraduate Symposium 2026

**Author:** M M Nishat  
**Event:** USM Undergraduate Symposium 2026

## Overview

End-to-end simulation framework for **event-triggered acoustic monitoring** using TinyML techniques applied to the ESC-50 dataset. Demonstrates how a circular buffer strategy dramatically reduces storage requirements on edge devices while preserving high event capture rates.

## Pipeline

```
ESC-50 Dataset → MFCC Feature Extraction → Random Forest Classifier
→ Multi-System Simulation → Evaluation Metrics → Figures
```

## Key Results

| System | DRR | ECR | FPR | PPV |
|---|---|---|---|---|
| Store-All (Baseline A) | 0.0% | 100.0% | 100.0% | 30.0% |
| Detect-Only / B=0 (Baseline B) | 69.5% | 95.0% | 2.9% | 93.4% |
| Random Baseline (same budget) | 69.5% | 32.5% | 29.6% | 32.0% |
| **Circular Buffer B=1 (Proposed)** | **46.2%** | **96.7%** | **35.4%** | **53.9%** |
| Circular Buffer B=2 | 31.0% | 97.5% | 56.8% | 42.4% |

**Key Finding:** Intelligence gap — Detector vs Random at same 122-clip budget: +62.5pp ECR (95.0% vs 32.5%).

## Setup

### 1. Download the ESC-50 dataset
```bash
# Clone ESC-50 into tinyml-acoustic/data/
git clone https://github.com/karolpiczak/ESC-50 tinyml-acoustic/data/ESC-50-master
```

### 2. Create and activate virtual environment
```bash
cd tinyml-acoustic
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas librosa scikit-learn matplotlib seaborn tqdm
```

### 3. Run the pipeline
```bash
"tinyml-acoustic/venv/bin/python3" "tinyml-acoustic/pipeline.py"
```

Output figures are saved to `tinyml-acoustic/figures/` and metrics to `tinyml-acoustic/results/metrics.json`.

## Metrics Glossary

- **DRR** — Data Reduction Ratio: fraction of audio *not* stored
- **ECR** — Event Capture Rate: fraction of true events saved (recall at system level)
- **FPR** — False Positive Rate: fraction of non-events saved unnecessarily
- **PPV** — Positive Predictive Value: precision at system level
