# %% [markdown]
# # ConvLSTM Temporal Grad-CAM Overlay Generation
#
# This notebook-style script regenerates center-frame ConvLSTM prediction masks and
# combines them with existing temporal Grad-CAM heatmaps. It is designed for Kaggle
# execution and keeps the helper functions modular so they can later move into
# `src/`.
#
# Output per sample:
# - Row 1: 5 temporal raw frames
# - Row 2: 5 temporal Grad-CAM overlays
# - Row 3: center-frame prediction mask and center-frame ground-truth mask

# %% [markdown]
# ## Imports and Configuration
#
# Update the configuration below for Kaggle paths, model stride, checkpoint, and
# output locations. The defaults mirror the local project layout.

# %%
from __future__ import annotations

import json
import os
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import split_by_echonet_filelist
from src.temporal_dataset_variable_stride import (
    EchoNetTemporalVariableStrideDataset,
    build_fps_lookup,
    load_temporal_metadata,
)
from src.temporal_model import build_convlstm_unet
from src.utils import load_echonet_tables
from src.visualization import sanitize_name


CONFIG = {
    # Switch this to 1, 4, 6, 8, or 10.
    "stride": 1,
    "target_layer": "temporal_bottleneck",
    # Required model checkpoint. On Kaggle, point this to the attached checkpoint.
    "checkpoint_path": PROJECT_ROOT / "outputs" / "runs" / "ConvLSTM_Unet_06_11" / "checkpoints" / "best_model.pt",
    # Existing Grad-CAM heatmap directory for the selected stride/layer.
    "heatmap_dir": PROJECT_ROOT / "outputs" / "runs" / "gradcam_temporal_evaluation_06_21" / "heatmaps" / "convlstm_stride_1" / "temporal_bottleneck",
    # EchoNet raw/processed data locations.
    "raw_dir": PROJECT_ROOT / "data" / "raw" / "EchoNet-Dynamic",
    "videos_dir": PROJECT_ROOT / "data" / "raw" / "EchoNet-Dynamic" / "Videos",
    "metadata_path": PROJECT_ROOT / "data" / "processed" / "metadata.csv",
    "image_dir": PROJECT_ROOT / "data" / "processed" / "images",
    "mask_dir": PROJECT_ROOT / "data" / "processed" / "masks",
    # Outputs.
    "output_prediction_dir": PROJECT_ROOT / "outputs" / "runs" / "convlstm_gradcam_overlay_generation" / "predictions",
    "output_visualization_dir": PROJECT_ROOT / "outputs" / "runs" / "convlstm_gradcam_overlay_generation" / "visualizations",
    # Model/data options.
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "batch_size": 1,
    "num_workers": 0,
    "sequence_length": 5,
    "image_size": (112, 112),
    "threshold": 0.5,
    "channels": [16, 32, 64, 128],
    "saliency_alpha": 0.45,
    # Set to None for every test sample. Use a small integer for a quick check.
    "max_samples": None,
}

CONFIG["model_id"] = f"convlstm_stride_{CONFIG['stride']}"
CONFIG["heatmap_prefix"] = f"{CONFIG['model_id']}_{CONFIG['target_layer']}"
CONFIG

# %% [markdown]
# ## Path Setup
#
# This cell normalizes paths, creates output directories, and prints missing-file
# warnings. Missing heatmaps for individual samples are skipped later rather than
# crashing the full batch.

# %%
def as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


for key in [
    "checkpoint_path",
    "heatmap_dir",
    "raw_dir",
    "videos_dir",
    "metadata_path",
    "image_dir",
    "mask_dir",
    "output_prediction_dir",
    "output_visualization_dir",
]:
    CONFIG[key] = as_path(CONFIG[key])

CONFIG["output_prediction_dir"].mkdir(parents=True, exist_ok=True)
CONFIG["output_visualization_dir"].mkdir(parents=True, exist_ok=True)

for key in ["checkpoint_path", "heatmap_dir", "raw_dir", "videos_dir", "metadata_path"]:
    path = CONFIG[key]
    if not path.exists():
        warnings.warn(f"{key} does not exist: {path}")

print(json.dumps({key: str(value) for key, value in CONFIG.items() if isinstance(value, Path)}, indent=2))

# %% [markdown]
# ## Dataset / Test-Set Loading
#
# The test split is reconstructed from `metadata.csv` and EchoNet's official
# `FileList.csv`. Temporal windows are loaded with the same variable-stride dataset
# used by the Grad-CAM evaluation notebook, so frame ordering matches the heatmaps.

# %%
def load_official_test_samples(config: dict[str, Any]) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, float]]:
    samples = load_temporal_metadata(config["metadata_path"])
    file_list, _ = load_echonet_tables(config["raw_dir"])
    fps_by_video = build_fps_lookup(file_list)
    _, _, test_samples = split_by_echonet_filelist(samples, file_list)

    if config["max_samples"] is not None:
        test_samples = test_samples[: int(config["max_samples"])]

    print(f"Official test samples selected: {len(test_samples):,}")
    return test_samples, file_list, fps_by_video


def make_temporal_dataset(config: dict[str, Any], samples: list[dict[str, Any]], fps_by_video: dict[str, float]):
    return EchoNetTemporalVariableStrideDataset(
        samples,
        videos_dir=config["videos_dir"],
        sequence_length=int(config["sequence_length"]),
        temporal_stride=int(config["stride"]),
        image_size=tuple(config["image_size"]),
        augment=False,
        fps_by_video=fps_by_video,
    )


def make_loader(dataset, config: dict[str, Any]) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=int(config["num_workers"]),
        pin_memory=(str(config["device"]).startswith("cuda")),
    )


test_samples, file_list, fps_by_video = load_official_test_samples(CONFIG)
test_dataset = make_temporal_dataset(CONFIG, test_samples, fps_by_video)
test_loader = make_loader(test_dataset, CONFIG)

# %% [markdown]
# ## Model Loading From Best Checkpoint
#
# This cell builds the ConvLSTM U-Net, loads the checkpoint, and switches the model
# to evaluation mode. If the checkpoint contains a saved config, its channel list is
# used automatically.

# %%
def checkpoint_config(checkpoint_path: Path) -> dict[str, Any]:
    sidecar = checkpoint_path.parent.parent / "config.json"
    if sidecar.exists():
        with sidecar.open("r", encoding="utf-8") as file:
            return json.load(file)
    return {}


def build_model(config: dict[str, Any]) -> torch.nn.Module:
    ckpt_cfg = checkpoint_config(config["checkpoint_path"])
    channels = tuple(ckpt_cfg.get("channels", config["channels"]))
    model = build_convlstm_unet(
        in_channels=1,
        out_channels=1,
        channels=channels,
    )
    return model.to(config["device"])


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: str) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


model = build_model(CONFIG)
model = load_checkpoint(model, CONFIG["checkpoint_path"], CONFIG["device"])
print(f"Loaded checkpoint: {CONFIG['checkpoint_path']}")

# %% [markdown]
# ## Inference and Prediction-Mask Saving
#
# This cell reruns the model on the selected official test set and saves one
# center-frame prediction mask per sample. Prediction filenames use the same sample
# id used by Grad-CAM heatmaps.

# %%
def batch_size_from_batch(batch: dict[str, Any]) -> int:
    sequence = batch["sequence"]
    return int(sequence.shape[0])


def batch_item(batch: dict[str, Any], key: str, index: int):
    value = batch[key]
    if isinstance(value, torch.Tensor):
        selected = value[index]
        return selected.detach().cpu().tolist() if selected.ndim > 0 else selected.detach().cpu().item()
    if isinstance(value, (list, tuple)):
        return value[index]
    return value


def save_prediction_mask(sample_id: str, probability: np.ndarray, config: dict[str, Any]) -> dict[str, str]:
    safe_id = sanitize_name(sample_id)
    probability = np.asarray(probability, dtype=np.float32)
    binary = (probability >= float(config["threshold"])).astype(np.uint8)

    npy_path = config["output_prediction_dir"] / f"{safe_id}_pred_probability.npy"
    png_path = config["output_prediction_dir"] / f"{safe_id}_pred_mask.png"
    np.save(npy_path, probability)
    cv2.imwrite(str(png_path), binary * 255)
    return {"prediction_probability_path": str(npy_path), "prediction_mask_path": str(png_path)}


@torch.no_grad()
def generate_predictions(model: torch.nn.Module, loader: DataLoader, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    device = config["device"]

    for batch in tqdm(loader, desc="Saving prediction masks"):
        sequence = batch["sequence"].to(device, non_blocking=True)
        logits = model(sequence)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]

        for index in range(batch_size_from_batch(batch)):
            sample_id = str(batch_item(batch, "id", index))
            paths = save_prediction_mask(sample_id, probs[index], config)
            rows.append(
                {
                    "sample_id": sample_id,
                    "video_id": batch_item(batch, "video_id", index),
                    "center_frame_idx": batch_item(batch, "frame_idx", index),
                    "frame_indices": " ".join(str(x) for x in batch_item(batch, "frame_indices", index)),
                    **paths,
                }
            )

        del sequence, logits, probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prediction_df = pd.DataFrame(rows)
    csv_path = config["output_prediction_dir"] / "prediction_manifest.csv"
    prediction_df.to_csv(csv_path, index=False)
    print(f"Saved prediction manifest: {csv_path}")
    return prediction_df


prediction_df = generate_predictions(model, test_loader, CONFIG)
prediction_df.head()

# %% [markdown]
# ## Heatmap / Frame / Mask Matching Utilities
#
# These helpers match Grad-CAM heatmaps, temporal frames, center predictions, and
# center-frame ground-truth masks by the canonical `sample_id`, for example
# `0X7012247CA8314DC5_frame0124`.

# %%
HEATMAP_FRAME_RE = re.compile(r"_frame_(\d+)_heatmap\.png$")


def normalize_01(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    array = array - float(np.nanmin(array))
    denom = float(np.nanmax(array))
    if denom > 0:
        array = array / denom
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


def find_heatmap_npy(sample_id: str, config: dict[str, Any]) -> Path | None:
    safe_id = sanitize_name(sample_id)
    exact = config["heatmap_dir"] / f"{config['heatmap_prefix']}_{safe_id}_heatmaps.npy"
    if exact.exists():
        return exact
    matches = sorted(config["heatmap_dir"].glob(f"*{safe_id}*_heatmaps.npy"))
    if matches:
        return matches[0]
    return None


def find_heatmap_pngs(sample_id: str, config: dict[str, Any]) -> list[Path]:
    safe_id = sanitize_name(sample_id)
    matches = sorted(config["heatmap_dir"].glob(f"*{safe_id}*_frame_*_heatmap.png"))

    def frame_number(path: Path) -> int:
        match = HEATMAP_FRAME_RE.search(path.name)
        return int(match.group(1)) if match else 999

    return sorted(matches, key=frame_number)


def load_heatmaps(sample_id: str, config: dict[str, Any]) -> np.ndarray | None:
    npy_path = find_heatmap_npy(sample_id, config)
    if npy_path is not None:
        heatmaps = np.load(npy_path).astype(np.float32)
        return np.stack([normalize_01(item) for item in heatmaps], axis=0)

    png_paths = find_heatmap_pngs(sample_id, config)
    if len(png_paths) != int(config["sequence_length"]):
        warnings.warn(f"Missing heatmaps for {sample_id}: found {len(png_paths)} PNGs in {config['heatmap_dir']}")
        return None

    heatmaps = []
    for path in png_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            warnings.warn(f"Could not read heatmap PNG: {path}")
            return None
        heatmaps.append(normalize_01(image))
    return np.stack(heatmaps, axis=0)


def prediction_path(sample_id: str, config: dict[str, Any]) -> Path:
    return config["output_prediction_dir"] / f"{sanitize_name(sample_id)}_pred_mask.png"


def load_prediction_mask(sample_id: str, config: dict[str, Any]) -> np.ndarray | None:
    path = prediction_path(sample_id, config)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        warnings.warn(f"Prediction mask missing for {sample_id}: {path}")
        return None
    return (mask > 0).astype(np.float32)


def load_ground_truth_mask_from_batch(batch: dict[str, Any], index: int) -> np.ndarray:
    mask = batch["mask"][index, 0].detach().cpu().numpy().astype(np.float32)
    return (mask > 0.5).astype(np.float32)


def load_sequence_from_batch(batch: dict[str, Any], index: int) -> np.ndarray:
    sequence = batch["sequence"][index, :, 0].detach().cpu().numpy().astype(np.float32)
    return np.stack([normalize_01(frame) for frame in sequence], axis=0)

# %% [markdown]
# ## Overlay Visualization Generation
#
# Each output figure has 5 raw frames, 5 heatmap overlays, and the center-frame
# prediction and ground-truth masks. Missing files produce warnings and skip only
# that sample.

# %%
def save_temporal_overlay_figure(
    sample_id: str,
    frames: np.ndarray,
    heatmaps: np.ndarray,
    prediction_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
    metadata: dict[str, Any],
    config: dict[str, Any],
) -> Path:
    n_frames = int(config["sequence_length"])
    center_idx = n_frames // 2
    output_path = config["output_visualization_dir"] / f"{sanitize_name(sample_id)}_temporal_gradcam_overlay.png"

    fig = plt.figure(figsize=(3.0 * n_frames, 8.5))
    gs = fig.add_gridspec(3, n_frames, height_ratios=[1.0, 1.0, 1.05])

    for idx in range(n_frames):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Raw t{idx}\nframe {metadata.get('frame_indices', ['?'] * n_frames)[idx]}")
        ax.axis("off")

    for idx in range(n_frames):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(frames[idx], cmap="gray", vmin=0, vmax=1)
        ax.imshow(heatmaps[idx], cmap="magma", vmin=0, vmax=1, alpha=float(config["saliency_alpha"]))
        ax.set_title(f"Grad-CAM t{idx}")
        ax.axis("off")

    pred_ax = fig.add_subplot(gs[2, 1:3])
    pred_ax.imshow(frames[center_idx], cmap="gray", vmin=0, vmax=1)
    pred_ax.imshow(prediction_mask, cmap="Blues", alpha=0.45, vmin=0, vmax=1)
    pred_ax.set_title("Center prediction mask")
    pred_ax.axis("off")

    gt_ax = fig.add_subplot(gs[2, 3:5])
    gt_ax.imshow(frames[center_idx], cmap="gray", vmin=0, vmax=1)
    gt_ax.imshow(ground_truth_mask, cmap="Greens", alpha=0.45, vmin=0, vmax=1)
    gt_ax.set_title("Center ground truth mask")
    gt_ax.axis("off")

    fig.suptitle(
        f"{config['model_id']} | {config['target_layer']} | {sample_id} | "
        f"center frame {metadata.get('center_frame_idx')}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def make_sample_visualization(batch: dict[str, Any], index: int, config: dict[str, Any]) -> Path | None:
    sample_id = str(batch_item(batch, "id", index))
    frames = load_sequence_from_batch(batch, index)
    heatmaps = load_heatmaps(sample_id, config)
    prediction_mask = load_prediction_mask(sample_id, config)
    ground_truth_mask = load_ground_truth_mask_from_batch(batch, index)

    if heatmaps is None or prediction_mask is None:
        return None
    if len(heatmaps) != int(config["sequence_length"]):
        warnings.warn(f"Skipping {sample_id}: expected {config['sequence_length']} heatmaps, got {len(heatmaps)}")
        return None

    metadata = {
        "video_id": batch_item(batch, "video_id", index),
        "center_frame_idx": batch_item(batch, "frame_idx", index),
        "frame_indices": batch_item(batch, "frame_indices", index),
    }
    return save_temporal_overlay_figure(
        sample_id=sample_id,
        frames=frames,
        heatmaps=heatmaps,
        prediction_mask=prediction_mask,
        ground_truth_mask=ground_truth_mask,
        metadata=metadata,
        config=config,
    )

# %% [markdown]
# ## Sanity-Check Cell To Visualize One Sample
#
# Run this cell first after prediction generation. It finds the first sample with
# available heatmaps and writes exactly one visualization.

# %%
def generate_one_sanity_check(loader: DataLoader, config: dict[str, Any]) -> Path | None:
    for batch in loader:
        for index in range(batch_size_from_batch(batch)):
            sample_id = str(batch_item(batch, "id", index))
            if load_heatmaps(sample_id, config) is None:
                continue
            path = make_sample_visualization(batch, index, config)
            if path is not None:
                print(f"Saved sanity-check visualization: {path}")
                return path
    warnings.warn("No sample with matching heatmaps was found for the sanity check.")
    return None


sanity_path = generate_one_sanity_check(test_loader, CONFIG)
sanity_path

# %% [markdown]
# ## Batch Generation For All Test Samples
#
# This cell creates one visualization per test sample with available heatmaps and
# saved predictions. It writes a manifest CSV with success/skipped status.

# %%
def generate_all_visualizations(loader: DataLoader, config: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch in tqdm(loader, desc="Generating overlay visualizations"):
        for index in range(batch_size_from_batch(batch)):
            sample_id = str(batch_item(batch, "id", index))
            try:
                path = make_sample_visualization(batch, index, config)
                rows.append(
                    {
                        "sample_id": sample_id,
                        "video_id": batch_item(batch, "video_id", index),
                        "center_frame_idx": batch_item(batch, "frame_idx", index),
                        "visualization_path": str(path) if path else "",
                        "status": "saved" if path else "skipped_missing_inputs",
                    }
                )
            except Exception as exc:
                warnings.warn(f"Failed to visualize {sample_id}: {exc}")
                rows.append(
                    {
                        "sample_id": sample_id,
                        "video_id": batch_item(batch, "video_id", index),
                        "center_frame_idx": batch_item(batch, "frame_idx", index),
                        "visualization_path": "",
                        "status": f"error: {exc}",
                    }
                )

    manifest = pd.DataFrame(rows)
    manifest_path = config["output_visualization_dir"] / "visualization_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"Saved visualization manifest: {manifest_path}")
    print(manifest["status"].value_counts(dropna=False))
    return manifest


visualization_manifest = generate_all_visualizations(test_loader, CONFIG)
visualization_manifest.head()

# %% [markdown]
# ## Kaggle Running Instructions
#
# 1. Upload or attach this repository as a Kaggle dataset, or copy this notebook-style
#    script into a Kaggle notebook.
# 2. Attach the EchoNet-Dynamic raw data dataset containing `FileList.csv`,
#    `VolumeTracings.csv`, and `Videos/`.
# 3. Attach the processed dataset containing `metadata.csv`, `images/`, and `masks/`.
# 4. Attach model checkpoint files and the existing Grad-CAM heatmap output folder.
# 5. In the configuration cell, update:
#    - `checkpoint_path`
#    - `heatmap_dir`
#    - `raw_dir`
#    - `videos_dir`
#    - `metadata_path`
#    - `image_dir`
#    - `mask_dir`
#    - `output_prediction_dir`
#    - `output_visualization_dir`
#    - `stride`
#    - `device`
#    - `batch_size`
# 6. Run cells in order. First use `max_samples = 5` to check paths and figure layout.
# 7. Set `max_samples = None` for the full official test split.
# 8. To switch from stride 1 to stride 8, update `stride`, `checkpoint_path`, and
#    `heatmap_dir`. The rest of the matching is sample-id based.
