from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from .r2plus1d_ef import denormalize_ef, ef_regression_metrics


IMAGENET_VIDEO_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32).reshape(3, 1, 1, 1)
IMAGENET_VIDEO_STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32).reshape(3, 1, 1, 1)


@dataclass
class R2Plus1DGradCAMResult:
    layer_name: str
    signed_native_cams: np.ndarray
    positive_native_cams: np.ndarray
    native_clip_normalized_cams: np.ndarray
    native_frame_normalized_positive_cams: np.ndarray
    native_signed_clip_normalized_cams: np.ndarray
    signed_spatial_upsampled_cams: np.ndarray
    positive_spatial_upsampled_cams: np.ndarray
    spatial_clip_normalized_cams: np.ndarray
    spatial_frame_normalized_positive_cams: np.ndarray
    spatial_signed_clip_normalized_cams: np.ndarray
    signed_raw_cams: np.ndarray
    positive_cams: np.ndarray
    clip_normalized_cams: np.ndarray
    frame_normalized_positive_cams: np.ndarray
    signed_clip_normalized_cams: np.ndarray
    native_temporal_positions: np.ndarray
    temporal_diagnostics: pd.DataFrame
    pred_ef: float
    pred_ef_normalized: float
    activation_shape: tuple[int, ...]
    gradient_shape: tuple[int, ...]
    native_t: int
    aligned_t: int
    temporal_interpolation_applied: bool

    @property
    def layer_resolution_label(self) -> str:
        return f"{self.layer_name} | native T={self.native_t} | aligned T={self.aligned_t}"


def _layer_resolution_label(layer_name: str, native_t: int, aligned_t: int) -> str:
    return f"{layer_name} | native T={int(native_t)} | aligned T={int(aligned_t)}"


def temporal_interpolation_note(native_t: int, aligned_t: int) -> str:
    if int(native_t) == int(aligned_t):
        return "Native temporal resolution equals aligned visualization resolution."
    return (
        f"Native temporal resolution: {int(native_t)}; interpolated to {int(aligned_t)} for visualization. "
        f"The {int(aligned_t)} aligned maps are not {int(aligned_t)} independent native explanations."
    )


def unnormalize_video_for_display(video: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert one normalized video from [C,T,H,W] to grayscale [T,H,W] in [0,1]."""

    if isinstance(video, torch.Tensor):
        array = video.detach().cpu().numpy().astype(np.float32)
    else:
        array = np.asarray(video, dtype=np.float32)
    if array.ndim != 4:
        raise AssertionError(f"Expected video [C,T,H,W], got {array.shape}.")
    if array.shape[0] == 3:
        array = array * IMAGENET_VIDEO_STD + IMAGENET_VIDEO_MEAN
        array = np.clip(array, 0.0, 1.0)
        gray = array.mean(axis=0)
    elif array.shape[0] == 1:
        gray = array[0]
    else:
        raise AssertionError(f"Expected 1 or 3 channels, got {array.shape[0]}.")
    return np.clip(gray.astype(np.float32), 0.0, 1.0)


def regression_metrics_from_predictions(predictions: pd.DataFrame) -> dict[str, float]:
    metrics = ef_regression_metrics(predictions["ef_pred"].to_numpy(), predictions["ef_true"].to_numpy())
    return {
        "mae": float(metrics["ef_mae"]),
        "rmse": float(metrics["ef_rmse"]),
        "pearson": float(metrics["ef_pearson"]),
        "mse": float(metrics["ef_rmse"] ** 2),
    }


@torch.inference_mode()
def run_r2plus1d_normal_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ef_mean: float,
    ef_std: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        out = model(video)
        pred = denormalize_ef(out["ef"].detach().cpu(), ef_mean, ef_std).numpy()
        true = batch["ef"].detach().cpu().numpy()
        for i, video_id in enumerate(batch["video_id"]):
            frame_indices = batch["sampled_frame_indices"][i].detach().cpu().numpy().astype(np.int64)
            rows.append(
                {
                    "video_id": str(video_id),
                    "ef_true": float(true[i]),
                    "ef_pred": float(pred[i]),
                    "ef_error": float(pred[i] - true[i]),
                    "abs_ef_error": float(abs(pred[i] - true[i])),
                    "ed_frame_idx": int(batch["ed_frame_idx"][i]),
                    "es_frame_idx": int(batch["es_frame_idx"][i]),
                    "window_start_frame": int(batch["window_start_frame"][i]),
                    "window_end_frame": int(batch["window_end_frame"][i]),
                    "sampled_frame_indices": "[" + ",".join(str(int(v)) for v in frame_indices.tolist()) + "]",
                }
            )
    predictions = pd.DataFrame(rows)
    return predictions, regression_metrics_from_predictions(predictions)


def select_representative_videos(
    predictions: pd.DataFrame,
    count: int = 10,
    low_error: int = 2,
    high_error: int = 2,
    representative: int = 6,
) -> list[str]:
    if predictions.empty:
        return []
    df = predictions.drop_duplicates("video_id").copy()
    selected: list[str] = []

    def add(ids: Iterable[str]) -> None:
        for video_id in ids:
            if str(video_id) not in selected:
                selected.append(str(video_id))
            if len(selected) >= count:
                break

    add(df.sort_values("abs_ef_error", ascending=True)["video_id"].head(low_error))
    add(df.sort_values("abs_ef_error", ascending=False)["video_id"].head(high_error))
    remaining = df[~df["video_id"].isin(selected)].sort_values("ef_true")
    if representative > 0 and len(remaining):
        quantiles = np.linspace(0.0, 1.0, representative)
        indices = sorted(set(int(round(q * (len(remaining) - 1))) for q in quantiles))
        add(remaining.iloc[indices]["video_id"])
    if len(selected) < count:
        add(df[~df["video_id"].isin(selected)].sort_values("abs_ef_error")["video_id"])
    return selected[:count]


def _named_module(model: nn.Module, layer_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        raise KeyError(f"Layer {layer_name!r} not found. Available examples: {list(modules)[:20]}")
    return modules[layer_name]


def _standard_3d_gradcam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    if activations.shape != gradients.shape:
        raise AssertionError(f"Activation/gradient shape mismatch: {activations.shape} vs {gradients.shape}.")
    weights = gradients.mean(dim=(2, 3, 4), keepdim=True)
    return (weights * activations).sum(dim=1)


def _upsample_cam_volume(cams: torch.Tensor, output_tdhw: tuple[int, int, int]) -> torch.Tensor:
    return F.interpolate(cams.unsqueeze(1), size=output_tdhw, mode="trilinear", align_corners=False).squeeze(1)


def _clip_normalize(positive: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if not np.isfinite(positive).all():
        raise AssertionError("CAM contains non-finite values.")
    max_value = float(positive.max()) if positive.size else 0.0
    if max_value <= eps:
        return np.zeros_like(positive, dtype=np.float32)
    return (positive / max_value).astype(np.float32)


def _frame_normalize_positive(positive: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if positive.ndim != 3:
        raise AssertionError(f"Expected CAMs shaped [T,H,W], got {positive.shape}.")
    out = np.zeros_like(positive, dtype=np.float32)
    for t in range(positive.shape[0]):
        max_value = float(positive[t].max())
        if max_value > eps:
            out[t] = (positive[t] / max_value).astype(np.float32)
    return out


def _signed_clip_normalize(signed: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if not np.isfinite(signed).all():
        raise AssertionError("Signed CAM contains non-finite values.")
    max_abs = float(np.abs(signed).max()) if signed.size else 0.0
    if max_abs <= eps:
        return np.zeros_like(signed, dtype=np.float32)
    return (signed / max_abs).astype(np.float32)


def r2plus1d_cam_metric_table(
    signed_cams: np.ndarray,
    positive_cams: np.ndarray,
    *,
    layer_name: str,
    space: str,
) -> pd.DataFrame:
    """Return simple temporal CAM statistics for either native or visualization-aligned CAMs."""

    if signed_cams.shape != positive_cams.shape:
        raise AssertionError(f"CAM shape mismatch: {signed_cams.shape} vs {positive_cams.shape}.")
    if signed_cams.ndim != 3:
        raise AssertionError(f"Expected CAMs shaped [T,H,W], got {signed_cams.shape}.")
    rows: list[dict[str, float | int | str]] = []
    for t in range(signed_cams.shape[0]):
        signed = signed_cams[t].astype(np.float32)
        positive = positive_cams[t].astype(np.float32)
        rows.append(
            {
                "layer_name": layer_name,
                "metric_space": space,
                "timestep": int(t),
                "signed_cam_mean": float(signed.mean()),
                "signed_cam_abs_mean": float(np.abs(signed).mean()),
                "signed_cam_max": float(signed.max()),
                "signed_cam_min": float(signed.min()),
                "positive_cam_max": float(positive.max()),
                "positive_cam_mean": float(positive.mean()),
                "positive_cam_mass": float(positive.sum()),
            }
        )
    return pd.DataFrame(rows)


def _temporal_diagnostics(
    signed: np.ndarray,
    positive: np.ndarray,
    signed_native: np.ndarray,
    positive_native: np.ndarray,
    gradients: torch.Tensor,
    activations: torch.Tensor,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    grad_cpu = gradients.detach().cpu()
    act_cpu = activations.detach().cpu()
    for t in range(signed.shape[0]):
        native_t = min(int(round(t * (signed_native.shape[0] - 1) / max(signed.shape[0] - 1, 1))), signed_native.shape[0] - 1)
        rows.append(
            {
                "timestep": int(t),
                "native_timestep_nearest": int(native_t),
                "gradient_abs_mean_native_nearest": float(grad_cpu[0, :, native_t].abs().mean()),
                "gradient_abs_max_native_nearest": float(grad_cpu[0, :, native_t].abs().max()),
                "activation_abs_mean_native_nearest": float(act_cpu[0, :, native_t].abs().mean()),
                "activation_abs_max_native_nearest": float(act_cpu[0, :, native_t].abs().max()),
                "signed_cam_mean": float(signed[t].mean()),
                "signed_cam_abs_mean": float(np.abs(signed[t]).mean()),
                "signed_cam_max": float(signed[t].max()),
                "signed_cam_min": float(signed[t].min()),
                "positive_cam_max": float(positive[t].max()),
                "positive_cam_mean": float(positive[t].mean()),
                "positive_cam_mass": float(positive[t].sum()),
                "native_signed_cam_mean_nearest": float(signed_native[native_t].mean()),
                "native_positive_cam_mass_nearest": float(positive_native[native_t].sum()),
            }
        )
    return pd.DataFrame(rows)


def r2plus1d_ef_gradcam(
    model: nn.Module,
    video: torch.Tensor,
    layer_name: str,
    ef_mean: float,
    ef_std: float,
) -> R2Plus1DGradCAMResult:
    """Run standard 3D Grad-CAM for EF regression at one R(2+1)D layer."""

    model.eval()
    model.zero_grad(set_to_none=True)
    if video.ndim != 5:
        raise AssertionError(f"Expected video shape [B,C,T,H,W], got {tuple(video.shape)}.")
    if video.shape[0] != 1:
        raise AssertionError("Grad-CAM expects batch size 1 for reliable hook handling.")

    activation: torch.Tensor | None = None

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        nonlocal activation
        output.retain_grad()
        activation = output

    handle = _named_module(model, layer_name).register_forward_hook(hook)
    try:
        out = model(video)
        score = out["ef"].sum()
        score.backward()
    finally:
        handle.remove()

    if activation is None:
        raise AssertionError(f"No activation captured for layer {layer_name}.")
    if activation.grad is None:
        raise AssertionError(f"No gradient captured for layer {layer_name}.")
    gradients = activation.grad
    if not torch.isfinite(activation).all() or not torch.isfinite(gradients).all():
        raise AssertionError(f"Non-finite activation or gradient at {layer_name}.")
    if float(gradients.abs().max().detach().cpu()) <= 1e-12:
        raise AssertionError(f"All-zero gradients at {layer_name}.")

    signed_native_t = _standard_3d_gradcam(activation, gradients)
    positive_native_t = torch.clamp(signed_native_t, min=0.0)
    input_tdhw = (int(video.shape[2]), int(video.shape[3]), int(video.shape[4]))
    spatial_tdhw = (int(signed_native_t.shape[1]), int(video.shape[3]), int(video.shape[4]))
    signed_spatial_up = _upsample_cam_volume(signed_native_t, spatial_tdhw)
    positive_spatial_up = _upsample_cam_volume(positive_native_t, spatial_tdhw)
    signed_up = _upsample_cam_volume(signed_spatial_up, input_tdhw)
    positive_up = _upsample_cam_volume(positive_spatial_up, input_tdhw)

    signed_native = signed_native_t.detach().cpu().numpy()[0].astype(np.float32)
    positive_native = positive_native_t.detach().cpu().numpy()[0].astype(np.float32)
    signed_spatial = signed_spatial_up.detach().cpu().numpy()[0].astype(np.float32)
    positive_spatial = positive_spatial_up.detach().cpu().numpy()[0].astype(np.float32)
    signed = signed_up.detach().cpu().numpy()[0].astype(np.float32)
    positive = positive_up.detach().cpu().numpy()[0].astype(np.float32)
    native_normalized = _clip_normalize(positive_native)
    native_frame_normalized = _frame_normalize_positive(positive_native)
    native_signed_normalized = _signed_clip_normalize(signed_native)
    spatial_normalized = _clip_normalize(positive_spatial)
    spatial_frame_normalized = _frame_normalize_positive(positive_spatial)
    spatial_signed_normalized = _signed_clip_normalize(signed_spatial)
    normalized = _clip_normalize(positive)
    frame_normalized = _frame_normalize_positive(positive)
    signed_normalized = _signed_clip_normalize(signed)
    if signed.shape != input_tdhw or positive.shape != input_tdhw:
        raise AssertionError(f"Aligned CAM shape mismatch at {layer_name}: {signed.shape}, expected {input_tdhw}.")
    if float(np.abs(signed).max()) <= 1e-12:
        raise AssertionError(f"Degenerate CAM at {layer_name}.")

    native_t = signed_native.shape[0]
    native_positions = np.linspace(0.0, video.shape[2] - 1, native_t, dtype=np.float32)
    diagnostics = _temporal_diagnostics(signed, positive, signed_native, positive_native, gradients, activation)
    pred_norm = float(out["ef"].detach().cpu()[0])
    return R2Plus1DGradCAMResult(
        layer_name=layer_name,
        signed_native_cams=signed_native,
        positive_native_cams=positive_native,
        native_clip_normalized_cams=native_normalized,
        native_frame_normalized_positive_cams=native_frame_normalized,
        native_signed_clip_normalized_cams=native_signed_normalized,
        signed_spatial_upsampled_cams=signed_spatial,
        positive_spatial_upsampled_cams=positive_spatial,
        spatial_clip_normalized_cams=spatial_normalized,
        spatial_frame_normalized_positive_cams=spatial_frame_normalized,
        spatial_signed_clip_normalized_cams=spatial_signed_normalized,
        signed_raw_cams=signed,
        positive_cams=positive,
        clip_normalized_cams=normalized,
        frame_normalized_positive_cams=frame_normalized,
        signed_clip_normalized_cams=signed_normalized,
        native_temporal_positions=native_positions,
        temporal_diagnostics=diagnostics,
        pred_ef=float(denormalize_ef(torch.tensor(pred_norm), ef_mean, ef_std)),
        pred_ef_normalized=pred_norm,
        activation_shape=tuple(activation.shape),
        gradient_shape=tuple(gradients.shape),
        native_t=int(native_t),
        aligned_t=int(video.shape[2]),
        temporal_interpolation_applied=bool(native_t != int(video.shape[2])),
    )


def save_r2plus1d_gradcam_npz(
    output_path: str | Path,
    result: R2Plus1DGradCAMResult,
    batch: dict[str, Any],
    checkpoint_metadata: dict[str, Any],
    dataset_metrics: dict[str, float],
    diagnostics_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_frames = unnormalize_video_for_display(batch["video"][0])
    frame_indices = batch["sampled_frame_indices"][0].detach().cpu().numpy().astype(np.int32)
    normalized_positions = batch["sampled_normalized_positions"][0].detach().cpu().numpy().astype(np.float32)
    ed_frame = int(batch["ed_frame_idx"][0])
    es_frame = int(batch["es_frame_idx"][0])
    true_ef = float(batch["ef"][0])
    abs_error = float(abs(result.pred_ef - true_ef))
    video_id = str(batch["video_id"][0])
    metadata = {
        "video_id": video_id,
        "layer_name": result.layer_name,
        "layer_resolution_label": result.layer_resolution_label,
        "native_t": result.native_t,
        "aligned_t": result.aligned_t,
        "temporal_interpolation_applied": result.temporal_interpolation_applied,
        "temporal_interpolation_display_note": temporal_interpolation_note(result.native_t, result.aligned_t),
        "predicted_ef": result.pred_ef,
        "predicted_ef_normalized": result.pred_ef_normalized,
        "ground_truth_ef": true_ef,
        "absolute_ef_error": abs_error,
        "sampled_frame_indices": frame_indices.tolist(),
        "sampled_normalized_positions": normalized_positions.tolist(),
        "ed_frame_idx": ed_frame,
        "es_frame_idx": es_frame,
        "window_start_frame": int(batch["window_start_frame"][0]),
        "window_end_frame": int(batch["window_end_frame"][0]),
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_metrics": dataset_metrics,
        "activation_shape": result.activation_shape,
        "gradient_shape": result.gradient_shape,
        "temporal_interpolation_note": "signed_native_cams/positive_native_cams preserve original layer resolution. signed_spatial_upsampled_cams/positive_spatial_upsampled_cams preserve original layer temporal resolution while upsampling each CAM slice to input H,W. signed_raw_cams/positive_cams are temporally aligned to input frames for visualization.",
        "quantitative_metric_note": "Native temporal metrics should use native CAMs. Aligned CAM metrics are visualization-space metrics, especially for layers with native_t < aligned_t.",
    }
    np.savez_compressed(
        output_path,
        input_frames=input_frames,
        signed_native_cams=result.signed_native_cams,
        positive_native_cams=result.positive_native_cams,
        native_clip_normalized_cams=result.native_clip_normalized_cams,
        native_frame_normalized_positive_cams=result.native_frame_normalized_positive_cams,
        native_signed_clip_normalized_cams=result.native_signed_clip_normalized_cams,
        signed_spatial_upsampled_cams=result.signed_spatial_upsampled_cams,
        positive_spatial_upsampled_cams=result.positive_spatial_upsampled_cams,
        spatial_clip_normalized_cams=result.spatial_clip_normalized_cams,
        spatial_frame_normalized_positive_cams=result.spatial_frame_normalized_positive_cams,
        spatial_signed_clip_normalized_cams=result.spatial_signed_clip_normalized_cams,
        signed_raw_cams=result.signed_raw_cams,
        positive_cams=result.positive_cams,
        clip_normalized_cams=result.clip_normalized_cams,
        frame_normalized_positive_cams=result.frame_normalized_positive_cams,
        signed_clip_normalized_cams=result.signed_clip_normalized_cams,
        native_temporal_positions=result.native_temporal_positions,
        temporal_diagnostics=result.temporal_diagnostics.to_records(index=False),
        sampled_frame_indices=frame_indices,
        sampled_normalized_positions=normalized_positions,
        timestep_indices=np.arange(input_frames.shape[0], dtype=np.int32),
        ed_frame_idx=np.array(ed_frame, dtype=np.int32),
        es_frame_idx=np.array(es_frame, dtype=np.int32),
        video_id=np.array(video_id),
        predicted_ef=np.array(result.pred_ef, dtype=np.float32),
        ground_truth_ef=np.array(true_ef, dtype=np.float32),
        absolute_ef_error=np.array(abs_error, dtype=np.float32),
        layer_name=np.array(result.layer_name),
        layer_resolution_label=np.array(result.layer_resolution_label),
        native_t=np.array(result.native_t, dtype=np.int32),
        aligned_t=np.array(result.aligned_t, dtype=np.int32),
        temporal_interpolation_applied=np.array(result.temporal_interpolation_applied),
        metadata_json=np.array(json.dumps(metadata, default=str)),
    )
    if diagnostics_csv_path is not None:
        diagnostics_csv_path = Path(diagnostics_csv_path)
        diagnostics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        df = result.temporal_diagnostics.copy()
        df.insert(0, "video_id", video_id)
        df.insert(1, "layer_name", result.layer_name)
        df["ed_frame_idx"] = ed_frame
        df["es_frame_idx"] = es_frame
        df["predicted_ef"] = result.pred_ef
        df["ground_truth_ef"] = true_ef
        df["absolute_ef_error"] = abs_error
        df.to_csv(diagnostics_csv_path, index=False)
    return {
        "video_id": video_id,
        "layer_name": result.layer_name,
        "predicted_ef": result.pred_ef,
        "ground_truth_ef": true_ef,
        "absolute_ef_error": abs_error,
        "ed_frame_idx": ed_frame,
        "es_frame_idx": es_frame,
        "npz_path": str(output_path),
        "diagnostics_csv_path": str(diagnostics_csv_path) if diagnostics_csv_path is not None else "",
        "activation_shape": str(result.activation_shape),
        "gradient_shape": str(result.gradient_shape),
        "native_cam_shape": str(result.signed_native_cams.shape),
        "aligned_cam_shape": str(result.signed_raw_cams.shape),
        "native_t": result.native_t,
        "aligned_t": result.aligned_t,
        "layer_resolution_label": result.layer_resolution_label,
        "temporal_interpolation_applied": result.temporal_interpolation_applied,
        "temporal_interpolation_display_note": temporal_interpolation_note(result.native_t, result.aligned_t),
    }


def _ed_es_label(frame_idx: int, ed_frame: int, es_frame: int) -> str:
    labels = []
    if int(frame_idx) == int(ed_frame):
        labels.append("ED")
    if int(frame_idx) == int(es_frame):
        labels.append("ES")
    return "/".join(labels)


def make_r2plus1d_overlay_figure(
    npz_path: str | Path,
    figure_path: str | Path,
    dataset_metrics: dict[str, float],
    cam_key: str = "frame_normalized_positive_cams",
    overlay_name: str = "positive frame-normalized",
    signed: bool = False,
    signed_display_mode: str = "faithful",
    signed_display_percentile: float | None = None,
    cmap_name: str | None = None,
    alpha: float = 0.45,
) -> None:
    npz_path = Path(npz_path)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["input_frames"].astype(np.float32)
        cams = data[cam_key].astype(np.float32)
        signed_raw = data["signed_raw_cams"].astype(np.float32)
        frame_indices = data["sampled_frame_indices"].astype(np.int32)
        ed_frame = int(data["ed_frame_idx"])
        es_frame = int(data["es_frame_idx"])
        pred_ef = float(data["predicted_ef"])
        true_ef = float(data["ground_truth_ef"])
        abs_error = float(data["absolute_ef_error"])
        video_id = str(data["video_id"])
        layer_name = str(data["layer_name"])
        native_t = int(data["native_t"]) if "native_t" in data.files else int(cams.shape[0])
        aligned_t = int(data["aligned_t"]) if "aligned_t" in data.files else int(frames.shape[0])
        resolution_label = (
            str(data["layer_resolution_label"])
            if "layer_resolution_label" in data.files
            else _layer_resolution_label(layer_name, native_t, aligned_t)
        )

    if signed and signed_display_mode not in {"faithful", "enhanced"}:
        raise ValueError("signed_display_mode must be 'faithful' or 'enhanced'.")
    display_cams = cams
    if signed and signed_display_percentile is not None:
        scale = float(np.percentile(np.abs(signed_raw), signed_display_percentile))
        display_cams = np.clip(signed_raw / max(scale, 1e-8), -1.0, 1.0).astype(np.float32)
    cmap = plt.get_cmap(cmap_name or ("coolwarm" if signed else "turbo"))
    timesteps = frames.shape[0]
    cols = 8
    rows = int(math.ceil((timesteps + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes_flat = np.asarray(axes).reshape(-1)
    colorbar_axis = axes_flat[timesteps] if timesteps < len(axes_flat) else None
    for t in range(rows * cols):
        ax = axes_flat[t]
        ax.axis("off")
        if t >= timesteps:
            continue
        frame = np.clip(frames[t], 0.0, 1.0)
        heat = np.clip(display_cams[t], -1.0, 1.0) if signed else np.clip(display_cams[t], 0.0, 1.0)
        rgb = np.repeat(frame[..., None], 3, axis=2)
        if signed:
            heat_rgb = cmap((heat + 1.0) / 2.0)[..., :3]
            mag = np.abs(heat)
            if signed_display_mode == "enhanced":
                mag = np.sqrt(mag)
            local_alpha = alpha * mag[..., None]
        else:
            heat_rgb = cmap(heat)[..., :3]
            local_alpha = alpha * np.sqrt(heat[..., None])
        overlay = np.clip((1.0 - local_alpha) * rgb + local_alpha * heat_rgb, 0.0, 1.0)
        ax.imshow(overlay)
        label = _ed_es_label(int(frame_indices[t]), ed_frame, es_frame)
        title = f"t{t:02d} src {int(frame_indices[t])}"
        if label:
            title += f"\n{label}"
        ax.set_title(title, fontsize=7)
    header = (
        f"R(2+1)D EF | {resolution_label} | {overlay_name} | {video_id}\n"
        f"pred EF={pred_ef:.2f} true EF={true_ef:.2f} abs err={abs_error:.2f} | "
        f"test MAE={dataset_metrics.get('mae', float('nan')):.2f} "
        f"RMSE={dataset_metrics.get('rmse', float('nan')):.2f} "
        f"Pearson r={dataset_metrics.get('pearson', float('nan')):.3f}"
    )
    if native_t != aligned_t:
        header += f"\n{temporal_interpolation_note(native_t, aligned_t)}"
    fig.suptitle(header, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.035, top=0.86, wspace=0.14, hspace=0.42)
    norm = plt.Normalize(vmin=-1.0, vmax=1.0) if signed else plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if colorbar_axis is not None:
        colorbar_axis.set_visible(True)
        colorbar_axis.clear()
        cbar = fig.colorbar(sm, cax=colorbar_axis)
    else:
        cbar = fig.colorbar(sm, ax=axes_flat.tolist(), fraction=0.016, pad=0.01)
    cbar.set_label("Signed CAM display" if signed else "Positive CAM display", fontsize=8)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_r2plus1d_heatmap_only_figure(
    npz_path: str | Path,
    figure_path: str | Path,
    dataset_metrics: dict[str, float],
    cam_key: str = "frame_normalized_positive_cams",
    overlay_name: str = "positive frame-normalized heatmap only",
    cmap_name: str = "turbo",
) -> None:
    """Save a heatmap-only diagnostic figure for inspecting faint CAM structure."""

    npz_path = Path(npz_path)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        cams = data[cam_key].astype(np.float32)
        frame_indices = data["sampled_frame_indices"].astype(np.int32)
        ed_frame = int(data["ed_frame_idx"])
        es_frame = int(data["es_frame_idx"])
        pred_ef = float(data["predicted_ef"])
        true_ef = float(data["ground_truth_ef"])
        abs_error = float(data["absolute_ef_error"])
        video_id = str(data["video_id"])
        layer_name = str(data["layer_name"])
        native_t = int(data["native_t"]) if "native_t" in data.files else int(cams.shape[0])
        aligned_t = int(data["aligned_t"]) if "aligned_t" in data.files else int(cams.shape[0])
        resolution_label = (
            str(data["layer_resolution_label"])
            if "layer_resolution_label" in data.files
            else _layer_resolution_label(layer_name, native_t, aligned_t)
        )

    timesteps = cams.shape[0]
    cols = 8
    rows = int(math.ceil((timesteps + 1) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes_flat = np.asarray(axes).reshape(-1)
    cmap = plt.get_cmap(cmap_name)
    for t in range(rows * cols):
        ax = axes_flat[t]
        ax.axis("off")
        if t >= timesteps:
            continue
        heat = np.clip(cams[t], 0.0, 1.0)
        ax.imshow(heat, cmap=cmap, vmin=0.0, vmax=1.0)
        label = _ed_es_label(int(frame_indices[t]), ed_frame, es_frame)
        title = f"t{t:02d} src {int(frame_indices[t])}"
        if label:
            title += f"\n{label}"
        ax.set_title(title, fontsize=7)
    header = (
        f"R(2+1)D EF | {resolution_label} | {overlay_name} | visualization only | {video_id}\n"
        f"pred EF={pred_ef:.2f} true EF={true_ef:.2f} abs err={abs_error:.2f} | "
        f"test MAE={dataset_metrics.get('mae', float('nan')):.2f} "
        f"RMSE={dataset_metrics.get('rmse', float('nan')):.2f} "
        f"Pearson r={dataset_metrics.get('pearson', float('nan')):.3f}"
    )
    if native_t != aligned_t:
        header += f"\n{temporal_interpolation_note(native_t, aligned_t)}"
    fig.suptitle(header, fontsize=12)
    fig.subplots_adjust(left=0.02, right=0.985, bottom=0.035, top=0.86, wspace=0.14, hspace=0.42)
    colorbar_axis = axes_flat[timesteps] if timesteps < len(axes_flat) else None
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    if colorbar_axis is not None:
        colorbar_axis.set_visible(True)
        colorbar_axis.clear()
        cbar = fig.colorbar(sm, cax=colorbar_axis)
    else:
        cbar = fig.colorbar(sm, ax=axes_flat.tolist(), fraction=0.016, pad=0.01)
    cbar.set_label("Positive CAM display", fontsize=8)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cam_centroid_table(cams: np.ndarray, eps: float = 1e-8) -> pd.DataFrame:
    rows: list[dict[str, float | int | bool]] = []
    height, width = cams.shape[1], cams.shape[2]
    yy, xx = np.mgrid[0:height, 0:width]
    for t in range(cams.shape[0]):
        cam = np.clip(cams[t].astype(np.float64), 0.0, None)
        mass = float(cam.sum())
        valid = mass > eps
        rows.append(
            {
                "timestep": int(t),
                "cam_mass": mass,
                "valid_centroid": bool(valid),
                "centroid_x": float((cam * xx).sum() / mass) if valid else float("nan"),
                "centroid_y": float((cam * yy).sum() / mass) if valid else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def make_r2plus1d_motion_trace_overlay(
    npz_path: str | Path,
    figure_path: str | Path,
    dataset_metrics: dict[str, float],
    cam_key: str = "frame_normalized_positive_cams",
    overlay_name: str = "frame-normalized motion trace",
    alpha: float = 0.45,
    contour_level: float | None = None,
) -> pd.DataFrame:
    npz_path = Path(npz_path)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["input_frames"].astype(np.float32)
        cams = np.clip(data[cam_key].astype(np.float32), 0.0, 1.0)
        frame_indices = data["sampled_frame_indices"].astype(np.int32)
        ed_frame = int(data["ed_frame_idx"])
        es_frame = int(data["es_frame_idx"])
        pred_ef = float(data["predicted_ef"])
        true_ef = float(data["ground_truth_ef"])
        abs_error = float(data["absolute_ef_error"])
        video_id = str(data["video_id"])
        layer_name = str(data["layer_name"])
        native_t = int(data["native_t"]) if "native_t" in data.files else int(cams.shape[0])
        aligned_t = int(data["aligned_t"]) if "aligned_t" in data.files else int(frames.shape[0])
        resolution_label = (
            str(data["layer_resolution_label"])
            if "layer_resolution_label" in data.files
            else _layer_resolution_label(layer_name, native_t, aligned_t)
        )
    centroids = cam_centroid_table(cams)
    valid = centroids[centroids["valid_centroid"]]
    timesteps = frames.shape[0]
    cols = 8
    rows = int(math.ceil(timesteps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.2))
    axes_flat = np.asarray(axes).reshape(-1)
    cmap = plt.get_cmap("turbo")
    for t in range(rows * cols):
        ax = axes_flat[t]
        ax.axis("off")
        if t >= timesteps:
            continue
        frame = np.clip(frames[t], 0.0, 1.0)
        heat = cams[t]
        rgb = np.repeat(frame[..., None], 3, axis=2)
        heat_rgb = cmap(heat)[..., :3]
        overlay = np.clip((1.0 - alpha * np.sqrt(heat[..., None])) * rgb + alpha * np.sqrt(heat[..., None]) * heat_rgb, 0.0, 1.0)
        ax.imshow(overlay)
        if contour_level is not None and float(heat.max()) > contour_level:
            ax.contour(heat, levels=[contour_level], colors="white", linewidths=0.8)
        prior = valid[valid["timestep"] <= t]
        if len(prior) > 1:
            ax.plot(prior["centroid_x"], prior["centroid_y"], color="cyan", linewidth=1.2)
        current = centroids[centroids["timestep"] == t].iloc[0]
        if bool(current["valid_centroid"]):
            ax.scatter([current["centroid_x"]], [current["centroid_y"]], s=16, color="yellow", edgecolors="black", linewidths=0.4)
        label = _ed_es_label(int(frame_indices[t]), ed_frame, es_frame)
        title = f"t{t:02d} src {int(frame_indices[t])}"
        if label:
            title += f"\n{label}"
        ax.set_title(title, fontsize=7)
    fig.suptitle(
        f"R(2+1)D EF | {resolution_label} | {overlay_name} | {video_id}\n"
        f"pred EF={pred_ef:.2f} true EF={true_ef:.2f} abs err={abs_error:.2f} | "
        f"test MAE={dataset_metrics.get('mae', float('nan')):.2f} RMSE={dataset_metrics.get('rmse', float('nan')):.2f}"
        + (f"\n{temporal_interpolation_note(native_t, aligned_t)}" if native_t != aligned_t else ""),
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return centroids


def make_r2plus1d_overlay_video(
    npz_path: str | Path,
    video_path: str | Path,
    cam_key: str,
    signed: bool = False,
    signed_display_percentile: float | None = None,
    fps: int = 6,
    alpha: float = 0.45,
) -> None:
    import cv2

    npz_path = Path(npz_path)
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["input_frames"].astype(np.float32)
        cams = data[cam_key].astype(np.float32)
        signed_raw = data["signed_raw_cams"].astype(np.float32)
        frame_indices = data["sampled_frame_indices"].astype(np.int32)
        ed_frame = int(data["ed_frame_idx"])
        es_frame = int(data["es_frame_idx"])
        native_t = int(data["native_t"]) if "native_t" in data.files else int(cams.shape[0])
        aligned_t = int(data["aligned_t"]) if "aligned_t" in data.files else int(frames.shape[0])
    if signed and signed_display_percentile is not None:
        scale = float(np.percentile(np.abs(signed_raw), signed_display_percentile))
        cams = np.clip(signed_raw / max(scale, 1e-8), -1.0, 1.0).astype(np.float32)
    cmap = plt.get_cmap("coolwarm" if signed else "turbo")
    height, width = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")
    try:
        for t in range(frames.shape[0]):
            frame = np.clip(frames[t], 0.0, 1.0)
            rgb = np.repeat(frame[..., None], 3, axis=2)
            heat = np.clip(cams[t], -1.0, 1.0) if signed else np.clip(cams[t], 0.0, 1.0)
            if signed:
                heat_rgb = cmap((heat + 1.0) / 2.0)[..., :3]
                local_alpha = alpha * np.sqrt(np.abs(heat))[..., None]
            else:
                heat_rgb = cmap(heat)[..., :3]
                local_alpha = alpha * np.sqrt(heat)[..., None]
            overlay = np.clip((1.0 - local_alpha) * rgb + local_alpha * heat_rgb, 0.0, 1.0)
            image = (overlay * 255).astype(np.uint8)
            label = f"t{t:02d} src {int(frame_indices[t])} {_ed_es_label(int(frame_indices[t]), ed_frame, es_frame)}"
            cv2.putText(image, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            if native_t != aligned_t:
                note = f"Native T={native_t}; interpolated to {aligned_t} for visualization"
                cv2.putText(image, note, (5, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
