# EchoNet Temporal XAI

This repository develops a research framework for temporal explainability in
echocardiography. The current codebase supports EchoNet-Dynamic preprocessing,
LV segmentation models, EF-regression models, temporal Grad-CAM extraction,
optical-flow comparison, and statistical evaluation of temporal saliency.

The long-term goal is to use these tools to evaluate whether explanations follow
cardiac motion over time, and to use those findings to motivate new
motion-aware temporal XAI architectures for echocardiography.

## Project Goals

- Convert EchoNet-Dynamic LV tracings into reusable image/mask datasets.
- Remove a recurring upper-right burned-in white annotation artifact from echo
  frames before training and evaluation.
- Train and compare 2D, ConvLSTM, multitask, and R(2+1)D baselines.
- Explain LV segmentation and EF-regression predictions with Grad-CAM.
- Evaluate temporal faithfulness of explanations using framewise saliency
  consistency, centroid motion, LV-mask overlap, temporal IoU, and optical-flow
  alignment.
- Keep experiments Kaggle-compatible with smoke modes before full runs.

## Dataset Layout

The notebooks expect EchoNet-Dynamic raw data and processed masks in this form:

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

On Kaggle, paths are usually set in each notebook configuration cell, or by
environment variables such as:

```python
os.environ["ECHONET_RAW_DIR"] = "/kaggle/input/<raw-dataset>/EchoNet-Dynamic"
os.environ["ECHONET_PROCESSED_DIR"] = "/kaggle/input/<processed-dataset>/processed"
```

## Updated Preprocessing

The preprocessing pipeline now includes conservative artifact removal for the
thin diagonal white annotation line that appears in the upper-right region of
some EchoNet-Dynamic frames.

Relevant files:

```text
src/artifact_removal.py
notebooks/02_create_masks.ipynb
```

The artifact-removal module:

- detects candidate upper-right diagonal annotation lines;
- uses temporal background consistency to avoid erasing real cardiac anatomy;
- restricts removal to peripheral/background regions;
- uses OpenCV inpainting to blend removed pixels into surrounding texture;
- preserves diagnostic figures for sanity-checking detections.

The processed dataset still contains the standard image/mask pairs:

```text
processed/
  metadata.csv
  images/
  masks/
```

`metadata.csv` is required by downstream training notebooks.

## Model Families

### 2D Segmentation Baselines

- MONAI 2D U-Net for framewise LV segmentation.
- MONAI UNETR baseline for framewise LV segmentation.

Relevant notebooks and modules:

```text
notebooks/03_train_unet_baseline.ipynb
notebooks/05_unetr_baseline.ipynb
src/model.py
src/train.py
src/unetr_model.py
src/unetr_train.py
```

### ConvLSTM U-Net Segmentation Models

The temporal segmentation baseline uses a ConvLSTM U-Net that receives a short
frame sequence and predicts the LV segmentation mask for the center frame.

Implemented variants include:

- original adjacent-frame ConvLSTM U-Net;
- variable temporal stride ConvLSTM U-Nets;
- bidirectional target-aligned ConvLSTM U-Net with 23-frame windows.

Relevant files:

```text
notebooks/04_temporal_baseline.ipynb
notebooks/04_temporal_baseline_variable_strides.ipynb
notebooks/07_bidirectional_convlstm.ipynb
src/temporal_model.py
src/temporal_dataset.py
src/temporal_dataset_variable_stride.py
src/bidirectional_convlstm_unet.py
```

### Multitask Segmentation + EF Models

The repository includes multitask ConvLSTM models that combine LV segmentation
and EF regression:

- segmentation-primary ConvLSTM with secondary EF regression;
- EF-primary ConvLSTM with auxiliary LV segmentation;
- EF-primary ConvLSTM with auxiliary LV segmentation and auxiliary motion head.

The EF-primary models use 23 grayscale frames:

```text
11 frames before target + target ED/ES frame + 11 frames after target
```

with temporal stride 2.

Relevant notebooks:

```text
notebooks/11_multitask_segmentation_ef.ipynb
notebooks/12_ef_primary_motion_head.ipynb
notebooks/14_convlstm_ef_regression_gradcam.ipynb
notebooks/17_multitask_EF_model_temporal_gradcam_evaluation.ipynb
```

### R(2+1)D EF Baseline

The repository also includes an EF-primary R(2+1)D baseline trained on
deterministic cardiac-cycle clips. This model is intended as a video backbone
comparison for future motion-aware EF and XAI models.

Relevant files:

```text
notebooks/13_r2plus1d_ef_baseline.ipynb
notebooks/15_r2plus1d_gradcam.ipynb
src/cardiac_cycle_dataset.py
src/r2plus1d_ef.py
src/gradcam_r2plus1d.py
```

## Explainability Pipelines

### Segmentation Grad-CAM

Segmentation Grad-CAM evaluates temporal stability of LV segmentation
explanations across 2D U-Net and ConvLSTM U-Net models.

Relevant notebooks:

```text
notebooks/05_gradcam_temporal_evaluation.ipynb
notebooks/06_convlstm_gradcam_overlay_generation.ipynb
notebooks/06_convlstm_gradcam_overlay_generation.py
notebooks/08_seg_gradcam.ipynb
```

Supported target layers include:

- 2D U-Net final convolution;
- 2D U-Net encoder bottleneck;
- ConvLSTM encoder bottleneck;
- ConvLSTM temporal bottleneck;
- ConvLSTM decoder3.

### EF Grad-CAM For ConvLSTM Models

EF Grad-CAM explains the EF-regression scalar output, not the segmentation or
motion losses. The pipeline supports two EF attribution modes:

- `encoder_bottleneck`: faithful per-frame attribution from frame-specific
  encoder activations;
- `temporal_representation`: EF-head probe maps from fused bidirectional
  temporal representations. These are useful diagnostics, but non-target
  timesteps are counterfactual probes rather than official model predictions.

Relevant files:

```text
notebooks/14_convlstm_ef_regression_gradcam.ipynb
src/gradcam_ef_regression.py
```

Saved CAM variants include:

- signed raw CAMs;
- positive CAMs;
- clip-normalized positive CAMs;
- frame-normalized positive CAMs for visualization;
- signed clip-normalized CAMs;
- robust signed display variants for visualization.

### Multitask EF Temporal Grad-CAM Evaluation

The full temporal evaluation notebook computes quantitative temporal metrics for
both EF-primary multitask models and both EF Grad-CAM types.

Relevant notebook:

```text
notebooks/17_multitask_EF_model_temporal_gradcam_evaluation.ipynb
```

Core outputs include:

```text
temporal_gradcam_per_sample_metrics.csv
temporal_gradcam_dataset_summary.csv
temporal_gradcam_paired_statistics.csv
temporal_gradcam_frame_metrics.csv
temporal_gradcam_transition_metrics.csv
```

The quantitative convention is:

- raw positive CAMs (`positive_cams`) are used for saliency consistency,
  mass-weighted centroid motion, and LV saliency overlap;
- clip-normalized CAMs (`clip_normalized_cams`) are used only when a common
  fixed threshold is needed, especially temporal saliency IoU;
- frame-normalized and signed CAMs are supplementary visualization/diagnostic
  outputs.

### Optical Flow And Motion Alignment

Optical-flow notebooks compare saliency against estimated cardiac motion. The
current flow pipeline uses RAFT, dynamic LV masks, transition-level Grad-CAM
saliency, and motion-saliency metrics.

Relevant notebooks:

```text
notebooks/16_convlstm_optical_flow_and_gradcam_comparison.ipynb
notebooks/19_optical_flow_explanation_faithfulness.ipynb
```

The optical-flow faithfulness pipeline computes metrics such as:

- saliency-weighted flow magnitude;
- correlation between flow magnitude and Grad-CAM;
- overlap between high-motion and high-saliency regions;
- saliency contained inside moving LV regions;
- flow-warped Grad-CAM consistency where feasible.

### Statistical Testing

Notebook 20 performs analysis-only statistical testing on saved temporal metric
outputs. It does not rerun inference, Grad-CAM, or optical flow.

Relevant notebook:

```text
notebooks/20_gradcam_temporal_evaluation_statistical_testing.ipynb
```

Typical outputs:

```text
descriptive_statistics.csv
wilcoxon_final_explanations.csv
wilcoxon_internal_representations.csv
friedman_stride_effects.csv
wilcoxon_stride_posthoc.csv
publication_summary_table.csv
publication_summary_table.tex
data_validation_report.txt
figures/
```

## Notebook Guide

Run notebooks in stages depending on the experiment.

### Data And Preprocessing

```text
01_explore_dataset.ipynb
02_create_masks.ipynb
18_segmentation_pseudolabels.ipynb
```

- `01_explore_dataset.ipynb`: validates raw EchoNet-Dynamic paths, file lists,
  splits, video metadata, and tracing coverage.
- `02_create_masks.ipynb`: creates processed image/mask pairs and optionally
  applies artifact removal.
- `18_segmentation_pseudolabels.ipynb`: generates dense LV pseudo-labels for
  frames needed by later Grad-CAM and optical-flow analyses.

### Segmentation Models

```text
03_train_unet_baseline.ipynb
04_temporal_baseline.ipynb
04_temporal_baseline_variable_strides.ipynb
05_unetr_baseline.ipynb
07_bidirectional_convlstm.ipynb
```

### Multitask And EF Models

```text
11_multitask_segmentation_ef.ipynb
12_ef_primary_motion_head.ipynb
13_r2plus1d_ef_baseline.ipynb
```

### Grad-CAM And Temporal XAI

```text
05_gradcam_temporal_evaluation.ipynb
08_seg_gradcam.ipynb
14_convlstm_ef_regression_gradcam.ipynb
15_r2plus1d_gradcam.ipynb
17_multitask_EF_model_temporal_gradcam_evaluation.ipynb
20_gradcam_temporal_evaluation_statistical_testing.ipynb
```

### Motion And Optical Flow

```text
09_motion_metrics.ipynb
16_convlstm_optical_flow_and_gradcam_comparison.ipynb
19_optical_flow_explanation_faithfulness.ipynb
```

### Ablations

```text
10_temporal_bypass_ablation.ipynb
```

## Source Modules

```text
src/artifact_removal.py
```

Artifact detection and inpainting for upper-right white annotation lines in
EchoNet frames.

```text
src/utils.py
```

EchoNet table loading, video frame reading, mask creation, preprocessing,
plotting, and general utility functions.

```text
src/dataset.py
```

Processed image/mask datasets, MONAI transforms, processed sample loading, and
official EchoNet split helpers.

```text
src/model.py
src/train.py
```

2D U-Net model construction, training, evaluation, checkpointing, and example
visualization utilities.

```text
src/temporal_dataset.py
src/temporal_dataset_variable_stride.py
src/temporal_model.py
src/temporal_train.py
src/temporal_train_version_2.py
```

Temporal ConvLSTM U-Net datasets, models, and training/evaluation utilities.

```text
src/bidirectional_convlstm_unet.py
```

Target-aligned bidirectional ConvLSTM U-Net used by newer segmentation and
multitask experiments.

```text
src/cardiac_cycle_dataset.py
src/r2plus1d_ef.py
```

Cardiac-cycle sampling and R(2+1)D EF-regression model utilities.

```text
src/gradcam.py
src/gradcam_ef_regression.py
src/gradcam_r2plus1d.py
```

Grad-CAM implementations for segmentation ConvLSTM/2D U-Net, EF-regression
ConvLSTM, and R(2+1)D video models.

```text
src/temporal_evaluation.py
src/visualization.py
```

Temporal saliency metrics, aggregation helpers, heatmap saving, overlays, and
metric plots.

## Outputs

Common run locations:

```text
outputs/runs/
outputs/updated_preprocessing/
outputs/iMIMIC_additional_experiments/
```

Important output types:

- `config.json`: exact run configuration.
- `checkpoints/`: model checkpoints.
- `manifests/`: prediction tables, Grad-CAM manifests, metric tables, and
  evaluation summaries.
- `npz/`: saved CAM arrays, dynamic masks, or flow arrays.
- `overlays/`, `figures/`, `qualitative_examples/`: visualization outputs.

Representative result files:

```text
normal_test_metrics.json
model_perturbation_comparison.csv
ef_gradcam_manifest.csv
temporal_gradcam_dataset_summary.csv
temporal_gradcam_per_sample_metrics.csv
temporal_gradcam_paired_statistics.csv
motion_saliency_dataset_summary.csv
flow_transition_metrics.csv
```

## Environment

Install dependencies from:

```text
requirements.txt
```

Main packages:

- Python
- PyTorch
- TorchVision
- MONAI
- OpenCV
- NumPy
- Pandas
- Matplotlib
- SciPy / statsmodels
- scikit-learn
- tqdm

Most notebooks are designed to run on Kaggle and include setup/configuration
cells for Kaggle paths, GPU selection, smoke mode, and output directories.

## Smoke Vs Full Runs

Many notebooks include a run mode:

```python
RUN_MODE = "smoke"
```

Use smoke mode first to verify:

- paths;
- checkpoint loading;
- dataset shapes;
- one forward pass;
- Grad-CAM gradients;
- output writing;
- visualization layout.

Then switch to:

```python
RUN_MODE = "full"
```

Restart the kernel and run the notebook end to end.

## Current Status

Implemented:

- EchoNet-Dynamic exploration and processed mask generation.
- Artifact-aware preprocessing with upper-right annotation-line removal.
- 2D U-Net and UNETR segmentation baselines.
- ConvLSTM U-Net segmentation baselines with multiple temporal strides.
- Bidirectional ConvLSTM U-Net segmentation.
- Multitask segmentation + EF models.
- EF-primary multitask models with auxiliary segmentation and optional motion
  head.
- R(2+1)D EF-regression baseline with cardiac-cycle sampling.
- Segmentation Grad-CAM temporal evaluation.
- EF-regression Grad-CAM for ConvLSTM and R(2+1)D models.
- Optical-flow and Grad-CAM motion-alignment evaluation.
- Temporal Grad-CAM statistical testing.

Planned research direction:

- use the temporal explanation and motion-alignment framework to design and
  evaluate new motion-aware temporal XAI architectures for echocardiography.
