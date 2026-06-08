# EchoNet Segmentation & Temporal XAI

## Project Overview

This project investigates explainable AI (XAI) methods for echocardiography segmentation using the EchoNet-Dynamic dataset.

The long-term goal is to develop temporal explainability methods that capture how information evolves across ultrasound video sequences, rather than providing explanations on individual frames only.

---

## Dataset

- EchoNet-Dynamic
- Echocardiography videos
- Expert left ventricle tracings
- Clinical measurements including EF, EDV, and ESV

---

## Current Objectives

### Phase 1: Baseline Segmentation

- Convert VolumeTracings.csv annotations into binary masks
- Build preprocessing pipeline
- Train 2D U-Net baseline
- Evaluate segmentation performance

### Phase 2: Advanced Segmentation Models

- UNETR
- MediViSTA
- MemSAM

### Phase 3: Explainable AI

- Grad-CAM
- SHAP
- Attention visualization
- Temporal explainability methods

### Phase 4: Trustworthy AI Evaluation

- Stability
- Robustness
- Clinical relevance
- Transparency

---

## Repository Structure

```text
src/
    Dataset, preprocessing, training, and evaluation code

notebooks/
    Kaggle and experimentation notebooks

data/
    Raw and processed data (not tracked in Git)

models/
    Model implementations
```

---

## Environment

Python

PyTorch

MONAI

OpenCV

NumPy

Pandas

Scikit-learn

---

## Status

Current stage:
Mask generation pipeline and U-Net baseline development.