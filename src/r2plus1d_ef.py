from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Callable

import numpy as np
import torch
from torch import nn


def build_r2plus1d_backbone(pretrained: bool = True):
    try:
        from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18
    except ImportError as exc:
        raise ImportError("torchvision is required for the R(2+1)D EF baseline.") from exc

    weights = R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None
    try:
        return r2plus1d_18(weights=weights), weights
    except TypeError:
        return r2plus1d_18(pretrained=pretrained), None


class R2Plus1DEFRegressor(nn.Module):
    """EF regressor with a torchvision R(2+1)D backbone.

    Forward returns a dictionary so future XAI and motion-supervision code can
    target stable feature names without editing the backbone.
    """

    feature_layer_names = ("stem", "layer1", "layer2", "layer3", "layer4")

    def __init__(
        self,
        pretrained: bool = True,
        dropout: float = 0.2,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        backbone, weights = build_r2plus1d_backbone(pretrained=pretrained)
        self.weights = weights
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        in_features = int(backbone.fc.in_features)
        self.ef_head = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        video: torch.Tensor,
        return_features: bool = False,
        feature_layers: tuple[str, ...] | list[str] | None = None,
    ) -> dict[str, torch.Tensor]:
        if video.ndim != 5:
            raise ValueError("Expected video shape [B, C, T, H, W].")

        requested = set(feature_layers or self.feature_layer_names)
        features: dict[str, torch.Tensor] = {}

        x = self.stem(video)
        if "stem" in requested:
            features["stem"] = x
        x = self.layer1(x)
        if "layer1" in requested:
            features["layer1"] = x
        x = self.layer2(x)
        if "layer2" in requested:
            features["layer2"] = x
        x = self.layer3(x)
        if "layer3" in requested:
            features["layer3"] = x
        x = self.layer4(x)
        if "layer4" in requested or "final_spatiotemporal" in requested:
            features["layer4"] = x
            features["final_spatiotemporal"] = x

        pooled = self.global_pool(x).flatten(1)
        ef = self.ef_head(pooled).squeeze(1)
        out = {"ef": ef}
        if return_features:
            out["pooled_features"] = pooled
            out["features"] = features
        return out


def ef_regression_metrics(pred_percent: np.ndarray, target_percent: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred_percent, dtype=np.float64)
    target = np.asarray(target_percent, dtype=np.float64)
    error = pred - target
    mae = float(np.mean(np.abs(error))) if len(error) else float("nan")
    rmse = float(np.sqrt(np.mean(error**2))) if len(error) else float("nan")
    if len(error) > 1 and np.std(pred) > 1e-8 and np.std(target) > 1e-8:
        corr = float(np.corrcoef(pred, target)[0, 1])
    else:
        corr = float("nan")
    return {"ef_mae": mae, "ef_rmse": rmse, "ef_pearson": corr}


def denormalize_ef(value: torch.Tensor | np.ndarray, ef_mean: float, ef_std: float) -> torch.Tensor | np.ndarray:
    return value * float(ef_std) + float(ef_mean)


def autocast_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            return torch.amp.autocast("cuda", enabled=True)
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch_ef(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    device: torch.device,
    ef_mean: float,
    ef_std: float,
    mixed_precision: bool = True,
    scaler: Any | None = None,
) -> dict[str, float]:
    model.train()
    rows: list[dict[str, Any]] = []
    autocast_enabled = bool(mixed_precision and torch.cuda.is_available())
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        target = batch["ef_normalized"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(autocast_enabled):
            pred = model(video)["ef"]
            loss = loss_fn(pred, target)
        if scaler is not None and autocast_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        pred_pct = denormalize_ef(pred.detach().cpu(), ef_mean, ef_std).numpy()
        true_pct = batch["ef"].detach().cpu().numpy()
        rows.append({"n": int(video.shape[0]), "loss": float(loss.detach().cpu()), "pred": pred_pct, "target": true_pct})
    return aggregate_ef_rows(rows, "train")


@torch.no_grad()
def evaluate_ef(
    model: nn.Module,
    loader,
    loss_fn: Callable,
    device: torch.device,
    ef_mean: float,
    ef_std: float,
    sequence_transform: Callable[..., torch.Tensor] | None = None,
    save_features: bool = False,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for batch in loader:
        video = batch["video"].to(device, non_blocking=True)
        if sequence_transform is not None:
            try:
                video = sequence_transform(video, batch)
            except TypeError:
                video = sequence_transform(video)
        target = batch["ef_normalized"].to(device, non_blocking=True)
        out = model(video, return_features=save_features)
        pred = out["ef"]
        loss = loss_fn(pred, target)
        pred_pct = denormalize_ef(pred.detach().cpu(), ef_mean, ef_std).numpy()
        true_pct = batch["ef"].detach().cpu().numpy()
        rows.append({"n": int(video.shape[0]), "loss": float(loss.detach().cpu()), "pred": pred_pct, "target": true_pct})
        for i, video_id in enumerate(batch["video_id"]):
            predictions.append(
                {
                    "video_id": str(video_id),
                    "ef_true": float(true_pct[i]),
                    "ef_pred": float(pred_pct[i]),
                    "ef_error": float(pred_pct[i] - true_pct[i]),
                    "ed_frame_idx": int(batch["ed_frame_idx"][i]),
                    "es_frame_idx": int(batch["es_frame_idx"][i]),
                    "window_start_frame": int(batch["window_start_frame"][i]),
                    "window_end_frame": int(batch["window_end_frame"][i]),
                    "sampled_frame_indices": "[" + ",".join(str(int(v)) for v in batch["sampled_frame_indices"][i].tolist()) + "]",
                    "sampled_normalized_positions": "[" + ",".join(f"{float(v):.8f}" for v in batch["sampled_normalized_positions"][i].tolist()) + "]",
                }
            )
    return aggregate_ef_rows(rows, "eval"), predictions


def aggregate_ef_rows(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    total_n = sum(row["n"] for row in rows)
    pred = np.concatenate([row["pred"] for row in rows]) if rows else np.array([])
    target = np.concatenate([row["target"] for row in rows]) if rows else np.array([])
    metrics = {f"{prefix}_loss": sum(row["loss"] * row["n"] for row in rows) / max(total_n, 1)}
    metrics.update({f"{prefix}_{key}": value for key, value in ef_regression_metrics(pred, target).items()})
    return metrics


def zero_motion_clip(video: torch.Tensor, source_index: int = 0) -> torch.Tensor:
    source_index = min(max(int(source_index), 0), video.shape[2] - 1)
    return video[:, :, source_index : source_index + 1].repeat(1, 1, video.shape[2], 1, 1).contiguous()


def shuffled_clip(video: torch.Tensor, seed: int = 42) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    order = torch.randperm(video.shape[2], generator=generator).to(video.device)
    return video[:, :, order].contiguous()


def reversed_clip(video: torch.Tensor) -> torch.Tensor:
    return torch.flip(video, dims=(2,)).contiguous()


def temporally_subsampled_repeat_clip(video: torch.Tensor, step: int = 2) -> torch.Tensor:
    indices = torch.arange(0, video.shape[2], step, device=video.device)
    sampled = video[:, :, indices]
    repeat = int(np.ceil(video.shape[2] / sampled.shape[2]))
    return sampled.repeat_interleave(repeat, dim=2)[:, :, : video.shape[2]].contiguous()


def spatially_blurred_clip(video: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")
    b, c, t, h, w = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(b * t * c, 1, h, w)
    kernel = video.new_ones((1, 1, kernel_size, kernel_size)) / float(kernel_size * kernel_size)
    blurred = torch.nn.functional.conv2d(flat, kernel, padding=kernel_size // 2)
    return blurred.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4).contiguous()
