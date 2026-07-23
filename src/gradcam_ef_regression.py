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

from .bidirectional_convlstm_unet import BidirectionalConvLSTMUNet
from .dataset import EchoNetTemporalDataset


class EFPrimaryConvLSTM(BidirectionalConvLSTMUNet):
    """Notebook-12 EF-primary bidirectional ConvLSTM U-Net.

    The architecture intentionally mirrors ``notebooks/12_ef_primary_motion_head.ipynb``
    so checkpoints from that notebook can be loaded without key translation.
    """

    def __init__(self, *args: Any, ef_hidden_dim: int = 128, dropout: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        bottleneck_channels = self.bidirectional_fusion[0].out_channels
        self.ef_pool = nn.AdaptiveAvgPool2d(1)
        self.ef_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(bottleneck_channels, ef_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ef_hidden_dim, 1),
        )

    @staticmethod
    def _run_temporal_sequence(
        cell: nn.Module,
        features: list[torch.Tensor],
        order: list[int],
    ) -> dict[int, torch.Tensor]:
        state = None
        hidden_by_index: dict[int, torch.Tensor] = {}
        for time_idx in order:
            encoded = features[time_idx]
            if state is None:
                state = cell.init_state(encoded)
            state = cell(encoded, state)
            hidden_by_index[time_idx] = state[0]
        return hidden_by_index

    def temporal_features(
        self,
        sequence: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        self._validate_sequence(sequence)
        bottlenecks, target_skips = self._encode_sequence(sequence)
        fwd = self._run_temporal_sequence(
            self.forward_temporal_bottleneck,
            bottlenecks,
            list(range(self.expected_sequence_length)),
        )
        bwd = self._run_temporal_sequence(
            self.backward_temporal_bottleneck,
            bottlenecks,
            list(range(self.expected_sequence_length - 1, -1, -1)),
        )
        fused = [
            self.bidirectional_fusion(torch.cat([fwd[t], bwd[t]], dim=1))
            for t in range(self.expected_sequence_length)
        ]
        return torch.stack(fused, dim=1), target_skips

    def decode_segmentation(
        self,
        fused_target: torch.Tensor,
        target_skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        skip1, skip2, skip3 = target_skips
        x = self.decoder3(fused_target, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.output(x)

    def forward(self, sequence: torch.Tensor, return_temporal: bool = False) -> dict[str, torch.Tensor]:
        fused_all, target_skips = self.temporal_features(sequence)
        fused_target = fused_all[:, self.target_idx]
        seg_logits = self.decode_segmentation(fused_target, target_skips)
        ef_normalized = self.ef_head(self.ef_pool(fused_target)).squeeze(1)
        out = {"seg_logits": seg_logits, "ef_normalized": ef_normalized}
        if return_temporal:
            out["temporal_features"] = fused_all
        return out


class MotionHead(nn.Module):
    """Notebook-12 optical-flow auxiliary head."""

    def __init__(self, channels: int, hidden_channels: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3 * channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 2, kernel_size=1),
        )

    def forward(self, temporal_features: torch.Tensor) -> torch.Tensor:
        pairs = []
        for t in range(temporal_features.shape[1] - 1):
            h0 = temporal_features[:, t]
            h1 = temporal_features[:, t + 1]
            pairs.append(self.net(torch.cat([h0, h1, h1 - h0], dim=1)))
        return torch.stack(pairs, dim=1)


class EFPrimaryMotionConvLSTM(EFPrimaryConvLSTM):
    """Notebook-12 EF-primary model with optical-flow auxiliary head."""

    def __init__(self, *args: Any, motion_hidden_channels: int = 64, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        bottleneck_channels = self.bidirectional_fusion[0].out_channels
        self.motion_head = MotionHead(bottleneck_channels, hidden_channels=motion_hidden_channels)

    def forward(self, sequence: torch.Tensor, return_temporal: bool = False) -> dict[str, torch.Tensor]:
        out = super().forward(sequence, return_temporal=True)
        out["flow_pred"] = self.motion_head(out["temporal_features"])
        if not return_temporal:
            out.pop("temporal_features")
        return out


def build_ef_regression_convlstm(config: dict[str, Any], with_motion: bool) -> EFPrimaryConvLSTM:
    cls: type[EFPrimaryConvLSTM] = EFPrimaryMotionConvLSTM if with_motion else EFPrimaryConvLSTM
    kwargs: dict[str, Any] = {
        "in_channels": 1,
        "out_channels": 1,
        "channels": tuple(config.get("channels", (16, 32, 64, 128))),
        "num_frames_before": int(config.get("num_frames_before", 11)),
        "num_frames_after": int(config.get("num_frames_after", 11)),
        "ef_hidden_dim": int(config.get("ef_hidden_dim", 128)),
        "dropout": float(config.get("dropout", 0.1)),
    }
    if with_motion:
        kwargs["motion_hidden_channels"] = int(config.get("motion_hidden_channels", 64))
    return cls(**kwargs)


def checkpoint_state_dict(checkpoint: dict[str, Any] | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))  # type: ignore[union-attr]
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def load_exact_checkpoint(model: nn.Module, checkpoint_path: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint_state_dict(checkpoint)
    model.load_state_dict(state_dict, strict=True)
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_keys": sorted(list(checkpoint.keys())) if isinstance(checkpoint, dict) else [],
        "epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "stage": checkpoint.get("stage") if isinstance(checkpoint, dict) else None,
        "metrics": checkpoint.get("metrics") if isinstance(checkpoint, dict) else None,
        "config": checkpoint.get("config") if isinstance(checkpoint, dict) else None,
        "ef_mean": checkpoint.get("ef_mean") if isinstance(checkpoint, dict) else None,
        "ef_std": checkpoint.get("ef_std") if isinstance(checkpoint, dict) else None,
    }


class EchoNetTemporalEFDataset(torch.utils.data.Dataset):
    """Wrap ``EchoNetTemporalDataset`` with EF labels and normalization metadata."""

    def __init__(self, base_dataset: EchoNetTemporalDataset, ef_mean: float, ef_std: float) -> None:
        self.base_dataset = base_dataset
        self.ef_mean = float(ef_mean)
        self.ef_std = float(ef_std)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = dict(self.base_dataset[idx])
        sample = self.base_dataset.samples[idx]
        ef = float(sample["ef"])
        item["ef"] = torch.tensor(ef, dtype=torch.float32)
        item["ef_normalized"] = torch.tensor((ef - self.ef_mean) / self.ef_std, dtype=torch.float32)
        return item


def denormalize_ef(ef_normalized: torch.Tensor | np.ndarray | float, ef_mean: float, ef_std: float) -> torch.Tensor:
    if not torch.is_tensor(ef_normalized):
        ef_normalized = torch.as_tensor(ef_normalized)
    return ef_normalized * float(ef_std) + float(ef_mean)


def regression_metrics(pred: Sequence[float], true: Sequence[float]) -> dict[str, float]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    true_arr = np.asarray(true, dtype=np.float64)
    error = pred_arr - true_arr
    mae = float(np.mean(np.abs(error))) if error.size else float("nan")
    mse = float(np.mean(error**2)) if error.size else float("nan")
    rmse = float(math.sqrt(mse)) if np.isfinite(mse) else float("nan")
    pearson = (
        float(np.corrcoef(pred_arr, true_arr)[0, 1])
        if error.size > 1 and np.std(pred_arr) > 1e-8 and np.std(true_arr) > 1e-8
        else float("nan")
    )
    return {"mae": mae, "mse": mse, "rmse": rmse, "pearson": pearson}


@torch.no_grad()
def run_normal_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    ef_mean: float,
    ef_std: float,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, float]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    for batch in loader:
        sequence = batch["sequence"].to(device, non_blocking=True)
        out = model(sequence)
        ef_pred = denormalize_ef(out["ef_normalized"].detach().cpu(), ef_mean, ef_std).numpy()
        ef_true = batch["ef"].detach().cpu().numpy()
        for i, sample_id in enumerate(batch["id"]):
            rows.append(
                {
                    "model_name": model_name,
                    "sample_id": str(sample_id),
                    "video_id": str(batch["video_id"][i]),
                    "target_frame_idx": int(batch["frame_idx"][i]),
                    "target_idx": int(batch["target_idx"][i]),
                    "ef_true": float(ef_true[i]),
                    "ef_pred": float(ef_pred[i]),
                    "ef_error": float(ef_pred[i] - ef_true[i]),
                    "abs_ef_error": float(abs(ef_pred[i] - ef_true[i])),
                }
            )
    df = pd.DataFrame(rows)
    metrics = regression_metrics(df["ef_pred"], df["ef_true"])
    return df, metrics


def select_representative_samples(
    predictions: pd.DataFrame,
    count: int = 10,
    low_error: int = 2,
    high_error: int = 2,
    representative: int = 6,
) -> list[str]:
    if predictions.empty:
        return []
    df = predictions.drop_duplicates("sample_id").copy()
    selected: list[str] = []

    def add(ids: Iterable[str]) -> None:
        for sample_id in ids:
            if sample_id not in selected:
                selected.append(str(sample_id))
            if len(selected) >= count:
                break

    add(df.sort_values("abs_ef_error", ascending=True)["sample_id"].head(low_error))
    add(df.sort_values("abs_ef_error", ascending=False)["sample_id"].head(high_error))

    remaining = df[~df["sample_id"].isin(selected)].sort_values("ef_true")
    if representative > 0 and len(remaining):
        quantiles = np.linspace(0.0, 1.0, representative)
        indices = sorted(set(int(round(q * (len(remaining) - 1))) for q in quantiles))
        add(remaining.iloc[indices]["sample_id"])

    if len(selected) < count:
        add(df[~df["sample_id"].isin(selected)].sort_values("abs_ef_error")["sample_id"])
    return selected[:count]


@dataclass
class GradCAMResult:
    signed_raw_cams: np.ndarray
    positive_cams: np.ndarray
    clip_normalized_cams: np.ndarray
    frame_normalized_positive_cams: np.ndarray
    signed_clip_normalized_cams: np.ndarray
    temporal_diagnostics: pd.DataFrame
    pred_ef: float
    pred_ef_normalized: float
    temporal_features_shape: tuple[int, ...] | None = None
    activation_shape: tuple[int, ...] | None = None
    gradient_shape: tuple[int, ...] | None = None


def _standard_gradcam(activations: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor:
    if activations.shape != gradients.shape:
        raise AssertionError(f"Activation/gradient shape mismatch: {activations.shape} vs {gradients.shape}")
    weights = gradients.mean(dim=(-2, -1), keepdim=True)
    return (weights * activations).sum(dim=1)


def _upsample_cams(cams: torch.Tensor, output_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(cams.unsqueeze(1), size=output_hw, mode="bilinear", align_corners=False).squeeze(1)


def _clip_normalize(positive: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    finite = np.isfinite(positive)
    if not finite.all():
        raise AssertionError("CAM contains non-finite values.")
    max_value = float(positive.max()) if positive.size else 0.0
    if max_value <= eps:
        return np.zeros_like(positive, dtype=np.float32)
    return (positive / max_value).astype(np.float32)


def _frame_normalize_positive(positive: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if positive.ndim != 3:
        raise AssertionError(f"Expected positive CAMs shaped [T,H,W], got {positive.shape}.")
    out = np.zeros_like(positive, dtype=np.float32)
    for timestep in range(positive.shape[0]):
        max_value = float(positive[timestep].max())
        if max_value > eps:
            out[timestep] = (positive[timestep] / max_value).astype(np.float32)
    return out


def _signed_clip_normalize(signed: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    if not np.isfinite(signed).all():
        raise AssertionError("Signed CAM contains non-finite values.")
    max_abs = float(np.abs(signed).max()) if signed.size else 0.0
    if max_abs <= eps:
        return np.zeros_like(signed, dtype=np.float32)
    return (signed / max_abs).astype(np.float32)


def _temporal_diagnostics(
    activations: torch.Tensor,
    gradients: torch.Tensor,
    signed: np.ndarray,
    positive: np.ndarray,
) -> pd.DataFrame:
    if activations.shape != gradients.shape:
        raise AssertionError(f"Activation/gradient shape mismatch: {activations.shape} vs {gradients.shape}.")
    if activations.shape[0] != signed.shape[0] or signed.shape != positive.shape:
        raise AssertionError("Temporal diagnostic inputs have inconsistent timestep counts.")

    rows: list[dict[str, float | int]] = []
    activations_cpu = activations.detach().cpu()
    gradients_cpu = gradients.detach().cpu()
    for timestep in range(signed.shape[0]):
        grad_t = gradients_cpu[timestep]
        act_t = activations_cpu[timestep]
        signed_t = signed[timestep]
        positive_t = positive[timestep]
        rows.append(
            {
                "timestep": int(timestep),
                "gradient_abs_mean": float(grad_t.abs().mean()),
                "gradient_abs_max": float(grad_t.abs().max()),
                "activation_abs_mean": float(act_t.abs().mean()),
                "activation_abs_max": float(act_t.abs().max()),
                "signed_cam_mean": float(signed_t.mean()),
                "signed_cam_abs_mean": float(np.abs(signed_t).mean()),
                "signed_cam_max": float(signed_t.max()),
                "signed_cam_min": float(signed_t.min()),
                "positive_cam_max": float(positive_t.max()),
                "positive_cam_mean": float(positive_t.mean()),
                "positive_cam_mass": float(positive_t.sum()),
            }
        )
    return pd.DataFrame(rows)


def _assert_cam_valid(name: str, signed: np.ndarray, positive: np.ndarray, normalized: np.ndarray, expected_t: int) -> None:
    if signed.shape != positive.shape or positive.shape != normalized.shape:
        raise AssertionError(f"{name}: inconsistent CAM shapes.")
    if signed.ndim != 3 or signed.shape[0] != expected_t:
        raise AssertionError(f"{name}: expected [T,H,W] with T={expected_t}, got {signed.shape}.")
    for array_name, array in [("signed", signed), ("positive", positive), ("normalized", normalized)]:
        if not np.isfinite(array).all():
            raise AssertionError(f"{name}: {array_name} CAM has non-finite values.")
    if float(np.abs(signed).max()) <= 1e-12:
        raise AssertionError(f"{name}: signed CAM is all zero.")
    if float(positive.max()) <= 1e-12:
        raise AssertionError(f"{name}: positive CAM is all zero.")


def encoder_bottleneck_ef_gradcam(
    model: EFPrimaryConvLSTM,
    sequence: torch.Tensor,
    ef_mean: float,
    ef_std: float,
) -> GradCAMResult:
    """EF Grad-CAM at ``model.bottleneck_encoder`` for all input frames.

    A single complete 23-frame forward pass is used. The target is only
    ``out["ef_normalized"]`` from the actual model output.
    """

    model.eval()
    model.zero_grad(set_to_none=True)
    if sequence.ndim != 5:
        raise AssertionError(f"Expected input shape [B,T,C,H,W], got {tuple(sequence.shape)}.")
    if sequence.shape[0] != 1:
        raise AssertionError("Grad-CAM currently expects batch size 1 for ordering safety.")
    if sequence.shape[1] != model.expected_sequence_length:
        raise AssertionError(f"Expected {model.expected_sequence_length} timesteps, got {sequence.shape[1]}.")

    captured: list[torch.Tensor] = []

    def forward_hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        output.retain_grad()
        captured.append(output)

    handle = model.bottleneck_encoder.register_forward_hook(forward_hook)
    try:
        out = model(sequence)
        score = out["ef_normalized"].sum()
        score.backward()
    finally:
        handle.remove()

    expected_t = model.expected_sequence_length
    if len(captured) != expected_t:
        raise AssertionError(f"Expected {expected_t} bottleneck activations, captured {len(captured)}.")

    activations = torch.cat(captured, dim=0)
    gradients_list = []
    for idx, activation in enumerate(captured):
        if activation.grad is None:
            raise AssertionError(f"Missing bottleneck gradient for timestep {idx}.")
        gradients_list.append(activation.grad)
    gradients = torch.cat(gradients_list, dim=0)

    if activations.shape[0] != expected_t or gradients.shape[0] != expected_t:
        raise AssertionError("Activation-gradient ordering check failed.")
    if not torch.isfinite(activations).all() or not torch.isfinite(gradients).all():
        raise AssertionError("Non-finite encoder activation or gradient.")

    cams = _standard_gradcam(activations, gradients)
    positive_native = torch.clamp(cams, min=0.0)
    cams_up = _upsample_cams(cams, output_hw=(int(sequence.shape[-2]), int(sequence.shape[-1])))
    positive_up = _upsample_cams(positive_native, output_hw=(int(sequence.shape[-2]), int(sequence.shape[-1])))
    signed = cams_up.detach().cpu().numpy().astype(np.float32)
    positive = positive_up.detach().cpu().numpy().astype(np.float32)
    normalized = _clip_normalize(positive)
    frame_normalized = _frame_normalize_positive(positive)
    signed_normalized = _signed_clip_normalize(signed)
    diagnostics = _temporal_diagnostics(activations, gradients, signed, positive)
    _assert_cam_valid("encoder_bottleneck", signed, positive, normalized, expected_t)
    pred_norm = float(out["ef_normalized"].detach().cpu()[0])
    return GradCAMResult(
        signed_raw_cams=signed,
        positive_cams=positive,
        clip_normalized_cams=normalized,
        frame_normalized_positive_cams=frame_normalized,
        signed_clip_normalized_cams=signed_normalized,
        temporal_diagnostics=diagnostics,
        pred_ef=float(denormalize_ef(pred_norm, ef_mean, ef_std)),
        pred_ef_normalized=pred_norm,
        activation_shape=tuple(activations.shape),
        gradient_shape=tuple(gradients.shape),
    )


def temporal_representation_ef_probe_gradcam(
    model: EFPrimaryConvLSTM,
    sequence: torch.Tensor,
    ef_mean: float,
    ef_std: float,
) -> GradCAMResult:
    """Grad-CAM over fused temporal representations using timestep EF probes.

    Each timestep's fused representation is fed through ``ef_pool`` and
    ``ef_head``, but that head was trained on the target-index representation.
    Non-target timestep maps are therefore out-of-distribution counterfactual
    probes, not a faithful record of how each input frame contributed to the
    actual EF prediction. For faithful per-frame attribution, such as optical
    flow comparison, prefer ``encoder_bottleneck_ef_gradcam``.
    """

    model.eval()
    if sequence.ndim != 5:
        raise AssertionError(f"Expected input shape [B,T,C,H,W], got {tuple(sequence.shape)}.")
    if sequence.shape[0] != 1:
        raise AssertionError("Grad-CAM currently expects batch size 1 for ordering safety.")
    if sequence.shape[1] != model.expected_sequence_length:
        raise AssertionError(f"Expected {model.expected_sequence_length} timesteps, got {sequence.shape[1]}.")

    with torch.enable_grad():
        official = model(sequence)
    official_pred_norm = float(official["ef_normalized"].detach().cpu()[0])
    official_pred = float(denormalize_ef(official_pred_norm, ef_mean, ef_std))

    signed_maps: list[np.ndarray] = []
    positive_maps: list[np.ndarray] = []
    activation_maps: list[torch.Tensor] = []
    gradient_maps: list[torch.Tensor] = []
    feature_shape: tuple[int, ...] | None = None
    gradient_shape: tuple[int, ...] | None = None
    expected_t = model.expected_sequence_length

    for timestep in range(expected_t):
        model.zero_grad(set_to_none=True)
        fused_all, _ = model.temporal_features(sequence)
        if fused_all.shape[1] != expected_t:
            raise AssertionError(f"Temporal features should have {expected_t} timesteps, got {fused_all.shape[1]}.")
        timestep_feature = fused_all[:, timestep]
        timestep_feature.retain_grad()
        probe_score = model.ef_head(model.ef_pool(timestep_feature)).squeeze(1).sum()
        probe_score.backward()
        if timestep_feature.grad is None:
            raise AssertionError(f"Missing temporal probe gradient for timestep {timestep}.")
        if not torch.isfinite(timestep_feature).all() or not torch.isfinite(timestep_feature.grad).all():
            raise AssertionError(f"Non-finite temporal feature or gradient at timestep {timestep}.")
        activation_maps.append(timestep_feature.detach().cpu()[0])
        gradient_maps.append(timestep_feature.grad.detach().cpu()[0])
        cam = _standard_gradcam(timestep_feature, timestep_feature.grad)
        positive_native = torch.clamp(cam, min=0.0)
        cam_up = _upsample_cams(cam, output_hw=(int(sequence.shape[-2]), int(sequence.shape[-1])))
        positive_up = _upsample_cams(positive_native, output_hw=(int(sequence.shape[-2]), int(sequence.shape[-1])))
        signed_maps.append(cam_up.detach().cpu().numpy()[0].astype(np.float32))
        positive_maps.append(positive_up.detach().cpu().numpy()[0].astype(np.float32))
        feature_shape = tuple(timestep_feature.shape)
        gradient_shape = tuple(timestep_feature.grad.shape)

    signed = np.stack(signed_maps, axis=0).astype(np.float32)
    positive = np.stack(positive_maps, axis=0).astype(np.float32)
    normalized = _clip_normalize(positive)
    frame_normalized = _frame_normalize_positive(positive)
    signed_normalized = _signed_clip_normalize(signed)
    diagnostics = _temporal_diagnostics(
        torch.stack(activation_maps, dim=0),
        torch.stack(gradient_maps, dim=0),
        signed,
        positive,
    )
    _assert_cam_valid("temporal_representation", signed, positive, normalized, expected_t)
    return GradCAMResult(
        signed_raw_cams=signed,
        positive_cams=positive,
        clip_normalized_cams=normalized,
        frame_normalized_positive_cams=frame_normalized,
        signed_clip_normalized_cams=signed_normalized,
        temporal_diagnostics=diagnostics,
        pred_ef=official_pred,
        pred_ef_normalized=official_pred_norm,
        temporal_features_shape=(1, expected_t, *feature_shape[1:]) if feature_shape else None,
        activation_shape=feature_shape,
        gradient_shape=gradient_shape,
    )


def save_gradcam_npz(
    output_path: str | Path,
    result: GradCAMResult,
    batch: dict[str, Any],
    model_name: str,
    checkpoint_metadata: dict[str, Any],
    cam_type: str,
    target_layer: str,
    dataset_metrics: dict[str, float],
    diagnostics_csv_path: str | Path | None = None,
) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sequence = batch["sequence"][0].detach().cpu().numpy().astype(np.float32)
    input_frames = sequence[:, 0]
    frame_indices = batch["frame_indices"][0].detach().cpu().numpy().astype(np.int32)
    target_idx = int(batch["target_idx"][0])
    target_frame_idx = int(batch["frame_idx"][0])
    ef_true = float(batch["ef"][0])
    abs_error = float(abs(result.pred_ef - ef_true))
    metadata = {
        "sample_id": str(batch["id"][0]),
        "video_id": str(batch["video_id"][0]),
        "model_name": model_name,
        "cam_type": cam_type,
        "target_layer": target_layer,
        "target_idx": target_idx,
        "target_frame_idx": target_frame_idx,
        "predicted_ef": result.pred_ef,
        "predicted_ef_normalized": result.pred_ef_normalized,
        "ground_truth_ef": ef_true,
        "absolute_ef_error": abs_error,
        "checkpoint_metadata": checkpoint_metadata,
        "dataset_metrics": dataset_metrics,
        "activation_shape": result.activation_shape,
        "gradient_shape": result.gradient_shape,
        "temporal_features_shape": result.temporal_features_shape,
    }
    np.savez_compressed(
        output_path,
        input_frames=input_frames,
        signed_raw_cams=result.signed_raw_cams,
        positive_cams=result.positive_cams,
        clip_normalized_cams=result.clip_normalized_cams,
        frame_normalized_positive_cams=result.frame_normalized_positive_cams,
        signed_clip_normalized_cams=result.signed_clip_normalized_cams,
        temporal_diagnostics=result.temporal_diagnostics.to_records(index=False),
        sampled_frame_indices=frame_indices,
        timestep_indices=np.arange(input_frames.shape[0], dtype=np.int32),
        target_idx=np.array(target_idx, dtype=np.int32),
        target_frame_idx=np.array(target_frame_idx, dtype=np.int32),
        sample_id=np.array(str(batch["id"][0])),
        video_id=np.array(str(batch["video_id"][0])),
        predicted_ef=np.array(result.pred_ef, dtype=np.float32),
        ground_truth_ef=np.array(ef_true, dtype=np.float32),
        absolute_ef_error=np.array(abs_error, dtype=np.float32),
        model_name=np.array(model_name),
        cam_type=np.array(cam_type),
        target_layer=np.array(target_layer),
        metadata_json=np.array(json.dumps(metadata, default=str)),
    )
    if diagnostics_csv_path is not None:
        diagnostics_csv_path = Path(diagnostics_csv_path)
        diagnostics_csv_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_df = result.temporal_diagnostics.copy()
        diagnostics_df.insert(0, "sample_id", metadata["sample_id"])
        diagnostics_df.insert(1, "video_id", metadata["video_id"])
        diagnostics_df.insert(2, "model_name", model_name)
        diagnostics_df.insert(3, "cam_type", cam_type)
        diagnostics_df.insert(4, "target_layer", target_layer)
        diagnostics_df["target_idx"] = target_idx
        diagnostics_df["target_frame_idx"] = target_frame_idx
        diagnostics_df["predicted_ef"] = result.pred_ef
        diagnostics_df["ground_truth_ef"] = ef_true
        diagnostics_df["absolute_ef_error"] = abs_error
        diagnostics_df.to_csv(diagnostics_csv_path, index=False)
    return {
        "sample_id": metadata["sample_id"],
        "video_id": metadata["video_id"],
        "model_name": model_name,
        "cam_type": cam_type,
        "target_layer": target_layer,
        "target_idx": target_idx,
        "target_frame_idx": target_frame_idx,
        "predicted_ef": result.pred_ef,
        "ground_truth_ef": ef_true,
        "absolute_ef_error": abs_error,
        "npz_path": str(output_path),
        "diagnostics_csv_path": str(diagnostics_csv_path) if diagnostics_csv_path is not None else "",
    }


def make_gradcam_overlay_figure(
    npz_path: str | Path,
    figure_path: str | Path,
    dataset_metrics: dict[str, float],
    cam_key: str = "frame_normalized_positive_cams",
    overlay_name: str = "positive clip-normalized",
    signed: bool = False,
    positive_display_mode: str = "faithful",
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
        target_idx = int(data["target_idx"])
        pred_ef = float(data["predicted_ef"])
        true_ef = float(data["ground_truth_ef"])
        abs_error = float(data["absolute_ef_error"])
        model_name = str(data["model_name"])
        cam_type = str(data["cam_type"])
        sample_id = str(data["sample_id"])

    timesteps = frames.shape[0]
    cols = 6
    rows = int(math.ceil(timesteps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.55))
    axes_flat = np.asarray(axes).reshape(-1)
    if not signed and positive_display_mode not in {"faithful", "enhanced"}:
        raise ValueError("positive_display_mode must be 'faithful' or 'enhanced'.")
    if signed and signed_display_mode not in {"faithful", "enhanced"}:
        raise ValueError("signed_display_mode must be 'faithful' or 'enhanced'.")
    cmap = plt.get_cmap(cmap_name or ("coolwarm" if signed else "inferno"))
    display_cams = cams
    if signed and signed_display_percentile is not None:
        scale = float(np.percentile(np.abs(signed_raw), signed_display_percentile))
        scale = max(scale, 1e-8)
        display_cams = np.clip(signed_raw / scale, -1.0, 1.0).astype(np.float32)

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
            magnitude_2d = np.clip(np.abs(heat), 0.0, 1.0)
            if signed_display_mode == "enhanced":
                magnitude_2d = np.sqrt(magnitude_2d)
            magnitude = magnitude_2d[..., None]
        else:
            heat_rgb = cmap(heat)[..., :3]
            alpha_heat = np.sqrt(heat) if positive_display_mode == "enhanced" else heat
            local_alpha = alpha * alpha_heat[..., None]
            magnitude = None
        if signed:
            local_alpha = alpha * magnitude
        overlay = np.clip((1.0 - local_alpha) * rgb + local_alpha * heat_rgb, 0.0, 1.0)
        ax.imshow(overlay)
        suffix = "target" if t == target_idx else ("probe" if cam_type == "temporal_representation" else "")
        title = f"t{t:02d} frm {int(frame_indices[t])}"
        if suffix:
            title += f"\n{suffix}"
        ax.set_title(title, fontsize=8)
        if t == target_idx:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3.0)
                spine.set_edgecolor("cyan")

    header = (
        f"{model_name} | {cam_type} | {overlay_name} | {sample_id}\n"
        f"pred EF={pred_ef:.2f}  true EF={true_ef:.2f}  abs err={abs_error:.2f} | "
        f"test MAE={dataset_metrics.get('mae', float('nan')):.2f}  "
        f"MSE={dataset_metrics.get('mse', float('nan')):.2f}  "
        f"RMSE={dataset_metrics.get('rmse', float('nan')):.2f}  "
        f"Pearson r={dataset_metrics.get('pearson', float('nan')):.3f}"
    )
    fig.suptitle(header, fontsize=12)
    fig.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.86, wspace=0.18, hspace=0.38)
    norm = plt.Normalize(vmin=-1.0, vmax=1.0) if signed else plt.Normalize(vmin=0.0, vmax=1.0)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    if colorbar_axis is not None:
        colorbar_axis.set_visible(True)
        colorbar_axis.clear()
        cbar = fig.colorbar(sm, cax=colorbar_axis)
    else:
        cbar = fig.colorbar(sm, ax=axes_flat.tolist(), fraction=0.018, pad=0.01)
    cbar.set_label("Signed CAM display" if signed else "Positive CAM display", fontsize=9)
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_temporal_diagnostic_plot(
    diagnostics_csv_path: str | Path,
    figure_path: str | Path,
    target_idx: int,
    title: str,
) -> None:
    diagnostics = pd.read_csv(diagnostics_csv_path)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 4))
    x = diagnostics["timestep"].to_numpy()
    ax1.plot(x, diagnostics["positive_cam_mass"], label="positive CAM mass", color="tab:blue")
    ax1.plot(x, diagnostics["positive_cam_max"], label="positive CAM max", color="tab:orange")
    ax1.set_xlabel("Timestep")
    ax1.set_ylabel("CAM value")
    ax2 = ax1.twinx()
    ax2.plot(x, diagnostics["gradient_abs_mean"], label="gradient abs mean", color="tab:green")
    ax2.set_ylabel("Gradient abs mean")
    ax1.axvline(int(target_idx), color="black", linestyle="--", linewidth=1.5, label="target")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def cam_centroid_table(cams: np.ndarray, eps: float = 1e-8) -> pd.DataFrame:
    """Compute visualization-only CAM centroids from positive CAM maps."""
    if cams.ndim != 3:
        raise AssertionError(f"Expected CAMs shaped [T,H,W], got {cams.shape}.")
    rows: list[dict[str, float | int | bool]] = []
    height, width = cams.shape[1], cams.shape[2]
    yy, xx = np.mgrid[0:height, 0:width]
    for timestep in range(cams.shape[0]):
        cam = np.clip(cams[timestep].astype(np.float64), 0.0, None)
        mass = float(cam.sum())
        valid = mass > eps
        if valid:
            x = float((cam * xx).sum() / mass)
            y = float((cam * yy).sum() / mass)
        else:
            x = float("nan")
            y = float("nan")
        rows.append(
            {
                "timestep": int(timestep),
                "centroid_x": x,
                "centroid_y": y,
                "cam_mass": mass,
                "valid_centroid": bool(valid),
            }
        )
    return pd.DataFrame(rows)


def make_motion_trace_overlay_figure(
    npz_path: str | Path,
    figure_path: str | Path,
    dataset_metrics: dict[str, float],
    cam_key: str = "frame_normalized_positive_cams",
    overlay_name: str = "frame-normalized motion trace",
    alpha: float = 0.45,
    positive_display_mode: str = "faithful",
    contour_level: float = 0.5,
) -> pd.DataFrame:
    """Create a movement-oriented overlay using positive CAM maps.

    This is a diagnostic visualization, not a new Grad-CAM computation.
    """

    npz_path = Path(npz_path)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["input_frames"].astype(np.float32)
        cams = data[cam_key].astype(np.float32)
        frame_indices = data["sampled_frame_indices"].astype(np.int32)
        target_idx = int(data["target_idx"])
        pred_ef = float(data["predicted_ef"])
        true_ef = float(data["ground_truth_ef"])
        abs_error = float(data["absolute_ef_error"])
        model_name = str(data["model_name"])
        cam_type = str(data["cam_type"])
        sample_id = str(data["sample_id"])

    centroids = cam_centroid_table(cams)
    timesteps = frames.shape[0]
    cols = 6
    rows = int(math.ceil(timesteps / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.55))
    axes_flat = np.asarray(axes).reshape(-1)
    if positive_display_mode not in {"faithful", "enhanced"}:
        raise ValueError("positive_display_mode must be 'faithful' or 'enhanced'.")
    cmap = plt.get_cmap("turbo")
    valid = centroids[centroids["valid_centroid"]]

    for t in range(rows * cols):
        ax = axes_flat[t]
        ax.axis("off")
        if t >= timesteps:
            continue
        frame = np.clip(frames[t], 0.0, 1.0)
        heat = np.clip(cams[t], 0.0, 1.0)
        rgb = np.repeat(frame[..., None], 3, axis=2)
        heat_rgb = cmap(heat)[..., :3]
        alpha_heat = np.sqrt(heat) if positive_display_mode == "enhanced" else heat
        local_alpha = alpha * alpha_heat[..., None]
        overlay = np.clip((1.0 - local_alpha) * rgb + local_alpha * heat_rgb, 0.0, 1.0)
        ax.imshow(overlay)
        if float(heat.max()) > 0:
            ax.contour(heat, levels=[contour_level], colors="white", linewidths=0.8, alpha=0.9)

        prior = valid[valid["timestep"] <= t]
        if len(prior) > 1:
            ax.plot(prior["centroid_x"], prior["centroid_y"], color="cyan", linewidth=1.4, alpha=0.8)
        current = centroids[centroids["timestep"] == t].iloc[0]
        if bool(current["valid_centroid"]):
            ax.scatter([current["centroid_x"]], [current["centroid_y"]], s=18, color="yellow", edgecolors="black", linewidths=0.4)

        suffix = "target" if t == target_idx else ("probe" if cam_type == "temporal_representation" else "")
        title = f"t{t:02d} frm {int(frame_indices[t])}"
        if suffix:
            title += f"\n{suffix}"
        ax.set_title(title, fontsize=8)
        if t == target_idx:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(3.0)
                spine.set_edgecolor("cyan")

    header = (
        f"{model_name} | {cam_type} | {overlay_name} | {sample_id}\n"
        f"pred EF={pred_ef:.2f}  true EF={true_ef:.2f}  abs err={abs_error:.2f} | "
        f"test MAE={dataset_metrics.get('mae', float('nan')):.2f}  "
        f"RMSE={dataset_metrics.get('rmse', float('nan')):.2f}  "
        f"Pearson r={dataset_metrics.get('pearson', float('nan')):.3f}"
    )
    fig.suptitle(header, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return centroids


def make_centroid_trajectory_plot(
    centroid_csv_path: str | Path,
    figure_path: str | Path,
    target_idx: int,
    title: str,
) -> None:
    centroids = pd.read_csv(centroid_csv_path)
    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    valid = centroids[centroids["valid_centroid"].astype(bool)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    if len(valid):
        sc = axes[0].scatter(valid["centroid_x"], valid["centroid_y"], c=valid["timestep"], cmap="viridis", s=35)
        axes[0].plot(valid["centroid_x"], valid["centroid_y"], color="gray", alpha=0.6, linewidth=1)
        fig.colorbar(sc, ax=axes[0], label="timestep")
    axes[0].invert_yaxis()
    axes[0].set_title("CAM centroid path")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[1].plot(centroids["timestep"], centroids["centroid_x"], label="x")
    axes[1].plot(centroids["timestep"], centroids["centroid_y"], label="y")
    axes[1].axvline(int(target_idx), color="black", linestyle="--", linewidth=1.5, label="target")
    axes[1].set_xlabel("timestep")
    axes[1].set_ylabel("centroid coordinate")
    axes[1].set_title("Centroid coordinates over time")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
