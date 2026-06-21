from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def normalize_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    normalized = np.zeros_like(heatmaps, dtype=np.float32)
    for idx, heatmap in enumerate(heatmaps):
        item = heatmap - float(np.nanmin(heatmap))
        max_value = float(np.nanmax(item))
        normalized[idx] = item / max_value if max_value > 0 else item
    return normalized


def binarize_saliency(heatmaps: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    heatmaps = normalize_heatmaps(heatmaps)
    masks: list[np.ndarray] = []
    for heatmap in heatmaps:
        threshold = np.percentile(heatmap, percentile)
        masks.append(heatmap >= threshold)
    return np.stack(masks, axis=0)


def centroid_from_map(values: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    total = float(values.sum())
    if total <= eps:
        return np.array([np.nan, np.nan], dtype=np.float64)
    yy, xx = np.indices(values.shape)
    return np.array([(xx * values).sum() / total, (yy * values).sum() / total], dtype=np.float64)


def adjacent_iou(binary_masks: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    values: list[float] = []
    for idx in range(len(binary_masks) - 1):
        first = binary_masks[idx].astype(bool)
        second = binary_masks[idx + 1].astype(bool)
        intersection = np.logical_and(first, second).sum()
        union = np.logical_or(first, second).sum()
        values.append(float((intersection + eps) / (union + eps)))
    return np.asarray(values, dtype=np.float64)


def adjacent_pearson(heatmaps: np.ndarray) -> np.ndarray:
    values: list[float] = []
    for idx in range(len(heatmaps) - 1):
        first = heatmaps[idx].reshape(-1)
        second = heatmaps[idx + 1].reshape(-1)
        if np.std(first) <= 0 or np.std(second) <= 0:
            values.append(np.nan)
        else:
            values.append(float(np.corrcoef(first, second)[0, 1]))
    return np.asarray(values, dtype=np.float64)


def center_saliency_mask_overlap(heatmaps: np.ndarray, center_mask: np.ndarray, eps: float = 1e-7) -> float:
    heatmaps = normalize_heatmaps(heatmaps)
    center_mask = np.asarray(center_mask)
    if center_mask.ndim == 3:
        center_mask = center_mask[0]
    center_mask = center_mask.astype(bool)
    center_idx = len(heatmaps) // 2
    center_heatmap = heatmaps[center_idx]
    total_saliency = float(center_heatmap.sum())
    return float(center_heatmap[center_mask].sum() / (total_saliency + eps))


def compute_temporal_saliency_metrics(
    heatmaps: np.ndarray,
    center_mask: np.ndarray,
    metadata: dict[str, Any],
    saliency_percentile: float = 80.0,
) -> dict[str, Any]:
    heatmaps = normalize_heatmaps(heatmaps)

    binary_saliency = binarize_saliency(heatmaps, percentile=saliency_percentile)
    saliency_centroids = np.stack([centroid_from_map(heatmap) for heatmap in heatmaps], axis=0)
    saliency_motion = np.diff(saliency_centroids, axis=0)

    consistency = adjacent_pearson(heatmaps)
    saliency_iou = adjacent_iou(binary_saliency)
    center_overlap = center_saliency_mask_overlap(heatmaps, center_mask)
    centroid_step = np.linalg.norm(saliency_motion, axis=1)

    row = dict(metadata)
    row.update(
        {
            "saliency_consistency_mean": float(np.nanmean(consistency)) if len(consistency) else np.nan,
            "saliency_centroid_motion_mean": float(np.nanmean(centroid_step)) if len(centroid_step) else np.nan,
            "center_saliency_mask_overlap": center_overlap,
            "temporal_saliency_iou_mean": float(np.nanmean(saliency_iou)) if len(saliency_iou) else np.nan,
            "saliency_percentile": saliency_percentile,
        }
    )
    for idx, centroid in enumerate(saliency_centroids):
        row[f"saliency_centroid_x_frame_{idx}"] = float(centroid[0])
        row[f"saliency_centroid_y_frame_{idx}"] = float(centroid[1])
    return row


def aggregate_metrics(per_sample: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_family", "model_id", "temporal_stride", "target_layer"]
    metric_cols = [
        "saliency_consistency_mean",
        "saliency_centroid_motion_mean",
        "center_saliency_mask_overlap",
        "temporal_saliency_iou_mean",
    ]
    available_metrics = [column for column in metric_cols if column in per_sample.columns]
    if per_sample.empty or not available_metrics:
        return pd.DataFrame(columns=group_cols)
    summary = (
        per_sample.groupby(group_cols, dropna=False)[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    return summary
