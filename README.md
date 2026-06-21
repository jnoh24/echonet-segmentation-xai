# EchoNet Segmentation & Temporal XAI

This repository contains segmentation and explainability experiments for the
EchoNet-Dynamic echocardiography dataset. It includes 2D frame-based baselines,
temporal ConvLSTM U-Net models, variable temporal stride experiments, UNETR, and
a Grad-CAM temporal explainability evaluation pipeline.

## Project Goals

- Convert EchoNet-Dynamic LV tracing coordinates into reusable binary masks.
- Train and compare 2D and temporal segmentation models.
- Study whether wider temporal frame spacing improves ConvLSTM U-Net behavior.
- Evaluate temporal Grad-CAM explanations across frames, strides, and model types.
- Keep experiments Kaggle-ready with smoke modes before full runs.

## Dataset Layout

The notebooks expect EchoNet-Dynamic raw data and processed masks in this shape:

```text
data/
  raw/
    EchoNet-Dynamic/
      FileList.csv
      VolumeTracings.csv
      Videos/
        *.avi
  processed/
    metadata.csv
    images/
      *.png
    masks/
      *.png
```

On Kaggle, the paths can be overridden with:

```python
os.environ["ECHONET_RAW_DIR"] = "/kaggle/input/<raw-dataset>/EchoNet-Dynamic"
os.environ["ECHONET_PROCESSED_DIR"] = "/kaggle/input/<processed-dataset>/processed"
```

## Notebooks

Run notebooks in this general order when starting from raw data:

```text
notebooks/01_explore_dataset.ipynb
notebooks/02_create_masks.ipynb
notebooks/03_train_unet_baseline.ipynb
notebooks/04_temporal_baseline.ipynb
notebooks/04_temporal_baseline_variable_strides.ipynb
notebooks/05_unetr_baseline.ipynb
notebooks/05_gradcam_temporal_evaluation.ipynb
```

### `01_explore_dataset.ipynb`

Validates the raw EchoNet-Dynamic layout, file metadata, split labels, frame
geometry, and tracing coverage.

### `02_create_masks.ipynb`

Converts `VolumeTracings.csv` LV tracing coordinates into processed image/mask
PNG pairs and writes `data/processed/metadata.csv`.

### `03_train_unet_baseline.ipynb`

Trains the 2D MONAI U-Net baseline on processed center-frame image/mask pairs
using the official EchoNet split when available.

### `04_temporal_baseline.ipynb`

Trains the original ConvLSTM U-Net baseline with adjacent five-frame windows:

```text
t-2, t-1, t, t+1, t+2
```

The model predicts only the center-frame mask.

### `04_temporal_baseline_variable_strides.ipynb`

Trains the same ConvLSTM U-Net architecture using configurable temporal strides:

```text
stride 4:  t-8,  t-4,  t, t+4,  t+8
stride 6:  t-12, t-6,  t, t+6,  t+12
stride 8:  t-16, t-8,  t, t+8,  t+16
stride 10: t-20, t-10, t, t+10, t+20
```

It keeps the official EchoNet split unchanged and saves each stride separately:

```text
outputs/runs/convlstm_unet_variable_strides/
  convlstm_unet_stride_4/
  convlstm_unet_stride_6/
  convlstm_unet_stride_8/
  convlstm_unet_stride_10/
```

It also writes validation/test comparison tables and selects the best stride
using validation Dice only.

### `05_unetr_baseline.ipynb`

Trains a MONAI UNETR baseline on the same processed image/mask pairs and official
EchoNet split.

### `05_gradcam_temporal_evaluation.ipynb`

Loads trained checkpoints and computes Grad-CAM temporal explainability metrics
on the official test set only. It supports:

- ConvLSTM U-Net stride 1, 4, 6, 8, and 10 checkpoints.
- ConvLSTM target layers:
  - `bottleneck_encoder`
  - `temporal_bottleneck`
  - `decoder3`
- 2D U-Net baseline Grad-CAM run independently on each frame of the same
  five-frame temporal windows.
- Smoke mode on one official test sequence.
- Full mode on the complete official test set.

Metrics kept in the current Grad-CAM pipeline:

- Saliency consistency across frames.
- Saliency centroid motion.
- Temporal saliency IoU.
- Center-frame CAM overlap with the center-frame LV ground-truth mask.

The processed dataset does not contain LV masks for neighboring temporal frames,
so saliency-cardiac motion alignment and per-frame LV mask motion metrics are not
computed.

## Source Modules

```text
src/utils.py
```

Common utilities for paths, seeding, EchoNet table loading, video frame reading,
mask creation, overlays, JSON writing, and preprocessing traced frames.

```text
src/dataset.py
```

Processed image/mask datasets, MONAI transforms, processed sample loading, and
official EchoNet split helpers.

```text
src/model.py
```

2D MONAI U-Net builder and simple mask prediction helper.

```text
src/train.py
```

2D U-Net loss, training loop, evaluation, checkpoint saving, training curves, and
prediction example visualizations.

```text
src/temporal_dataset.py
```

Original adjacent-frame temporal dataset for the stride-1 ConvLSTM baseline.

```text
src/temporal_dataset_variable_stride.py
```

Configurable temporal dataset for five-frame windows with arbitrary positive
frame stride. It also records frame indices, FPS, and approximate temporal window
span in seconds.

```text
src/temporal_model.py
```

ConvLSTM U-Net model definition. It encodes each frame, fuses bottleneck
features with a ConvLSTM cell, and decodes a center-frame segmentation mask.

```text
src/temporal_train.py
```

Original ConvLSTM U-Net training and evaluation utilities.

```text
src/temporal_train_version_2.py
```

Variable-stride ConvLSTM training/evaluation utilities with validation Dice
model selection, early stopping, final held-out test evaluation, segmentation
metrics, temporal LV area summaries, and per-video area curve artifacts.

```text
src/unetr_model.py
src/unetr_train.py
```

MONAI UNETR model builder plus training, evaluation, plotting, prediction, JSON,
and experiment logging utilities.

```text
src/gradcam.py
```

Grad-CAM helpers for ConvLSTM U-Net and framewise 2D U-Net evaluation. ConvLSTM
hooks append activations and gradients for repeated temporal layer calls so
per-frame heatmaps can be saved.

```text
src/temporal_evaluation.py
```

Temporal saliency metric functions and aggregation helpers.

```text
src/visualization.py
```

Heatmap saving, overlay grid saving, name sanitization, and metric plot helpers.

## Checkpoint Layout For Grad-CAM

Because each ConvLSTM checkpoint is named `best_model.pt`, folder structure is
important. Recommended Kaggle layout:

```text
/kaggle/input/convlstm-stride-1/
  checkpoints/
    best_model.pt

/kaggle/input/convlstm_variable_strides/
  convlstm_unet_stride_4/
    checkpoints/
      best_model.pt
  convlstm_unet_stride_6/
    checkpoints/
      best_model.pt
  convlstm_unet_stride_8/
    checkpoints/
      best_model.pt
  convlstm_unet_stride_10/
    checkpoints/
      best_model.pt

/kaggle/input/unet-baseline/
  best_unet.pt
```

Then set:

```python
os.environ["CONVLSTM_STRIDE1_RUN_DIR"] = "/kaggle/input/convlstm-stride-1"
os.environ["CONVLSTM_VARIABLE_STRIDE_RUN_DIR"] = "/kaggle/input/convlstm_variable_strides"
os.environ["UNET_CHECKPOINT_PATH"] = "/kaggle/input/unet-baseline/best_unet.pt"
```

## Outputs

Common output locations:

```text
outputs/
  checkpoints/
  figures/
  runs/
    convlstm_unet/
    convlstm_unet_variable_strides/
    gradcam_temporal_evaluation/
```

The Grad-CAM notebook writes:

```text
outputs/runs/gradcam_temporal_evaluation/
  heatmaps/
  overlays/
  metrics/
    per_sample_metrics.csv
    aggregated_metrics.csv
  figures/
  tables/
    checkpoint_discovery.csv
    comparison_across_strides.csv
    comparison_across_target_layers.csv
    comparison_convlstm_vs_unet.csv
  gradcam_temporal_evaluation_summary.json
```

Full Grad-CAM mode computes metrics for all selected test samples but saves
overlay grids only for a limited representative set:

```python
MAX_OVERLAY_SAMPLES = 1 if RUN_MODE == "smoke" else 20
```

## Environment

Install dependencies from:

```text
requirements.txt
```

Main packages:

- Python
- PyTorch
- MONAI
- OpenCV
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm

Most notebooks include a Kaggle setup cell similar to:

```python
%pip install -q monai opencv-python-headless pandas matplotlib tqdm
```

## Smoke Vs Full Runs

Several experiment notebooks include:

```python
RUN_MODE = "smoke"
```

Use smoke mode first to verify paths, checkpoint loading, data loading, output
writing, and plotting. After smoke mode succeeds, switch to:

```python
RUN_MODE = "full"
```

Then restart the Kaggle kernel and run the notebook end to end.

## Current Status

Implemented:

- EchoNet-Dynamic exploration and mask preprocessing.
- 2D U-Net baseline.
- ConvLSTM U-Net stride-1 temporal baseline.
- ConvLSTM U-Net variable temporal stride experiments.
- UNETR baseline.
- Grad-CAM temporal explainability evaluation for ConvLSTM and 2D U-Net models.

Planned/possible extensions:

- Additional temporal XAI methods.
- Broader robustness/stability tests.
- Clinical relevance analysis of saliency behavior.
