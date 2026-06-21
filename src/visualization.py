from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_name(value: object) -> str:
    text = str(value)
    keep = [char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text]
    return "".join(keep)


def save_heatmaps(
    heatmaps: np.ndarray,
    output_dir: str | Path,
    prefix: str,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    npy_path = output_dir / f"{prefix}_heatmaps.npy"
    np.save(npy_path, heatmaps.astype(np.float32))

    paths = [npy_path]
    for idx, heatmap in enumerate(heatmaps):
        png_path = output_dir / f"{prefix}_frame_{idx}_heatmap.png"
        plt.imsave(png_path, heatmap, cmap="magma", vmin=0, vmax=1)
        paths.append(png_path)
    return paths


def save_overlay_grid(
    frames: np.ndarray,
    heatmaps: np.ndarray,
    output_path: str | Path,
    title: str,
    alpha: float = 0.45,
) -> Path:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    frames = np.asarray(frames)
    heatmaps = np.asarray(heatmaps)
    n_frames = len(heatmaps)
    fig, axes = plt.subplots(2, n_frames, figsize=(3 * n_frames, 6))
    if n_frames == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for idx in range(n_frames):
        frame = frames[idx]
        if frame.ndim == 3:
            frame = frame[0]
        axes[0, idx].imshow(frame, cmap="gray", vmin=0, vmax=1)
        axes[0, idx].set_title(f"Frame {idx}")
        axes[0, idx].axis("off")

        axes[1, idx].imshow(frame, cmap="gray", vmin=0, vmax=1)
        axes[1, idx].imshow(heatmaps[idx], cmap="magma", vmin=0, vmax=1, alpha=alpha)
        axes[1, idx].set_title(f"CAM {idx}")
        axes[1, idx].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_metric_plots(
    aggregate_df: pd.DataFrame,
    output_dir: str | Path,
    metrics: Iterable[str] | None = None,
) -> list[Path]:
    output_dir = ensure_dir(output_dir)
    if aggregate_df.empty:
        return []

    if metrics is None:
        metrics = [
            "saliency_consistency_mean_mean",
            "saliency_centroid_motion_mean_mean",
            "center_saliency_mask_overlap_mean",
            "temporal_saliency_iou_mean_mean",
        ]

    paths: list[Path] = []
    for metric in metrics:
        if metric not in aggregate_df.columns:
            continue
        fig, axis = plt.subplots(figsize=(9, 4.5))
        plot_df = aggregate_df.copy()
        plot_df["label"] = (
            plot_df["model_id"].astype(str)
            + " | "
            + plot_df["target_layer"].astype(str)
        )
        axis.bar(plot_df["label"], plot_df[metric])
        axis.set_title(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=70)
        fig.tight_layout()
        path = output_dir / f"{metric}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
