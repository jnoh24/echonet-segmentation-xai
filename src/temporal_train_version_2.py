from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.losses import DiceCELoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import ensure_dir, overlay_mask


def get_temporal_loss() -> DiceCELoss:
    """Match the original ConvLSTM baseline Dice + CE objective."""
    return DiceCELoss(
        sigmoid=True,
        squared_pred=True,
        lambda_dice=1.0,
        lambda_ce=1.0,
    )


def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> dict[str, torch.Tensor]:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, predictions.ndim))
    tp = (predictions * targets).sum(dim=dims)
    pred_area = predictions.sum(dim=dims)
    target_area = targets.sum(dim=dims)
    fp = (predictions * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - predictions) * targets).sum(dim=dims)
    union = pred_area + target_area - tp
    return {
        "dice": (2.0 * tp + eps) / (pred_area + target_area + eps),
        "iou": (tp + eps) / (union + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
    }


def dice_between_masks(mask_a: np.ndarray, mask_b: np.ndarray, eps: float = 1e-7) -> float:
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    intersection = np.logical_and(mask_a, mask_b).sum()
    total = mask_a.sum() + mask_b.sum()
    return float((2.0 * intersection + eps) / (total + eps))


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="temporal train", leave=False):
        sequences = batch["sequence"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(sequences)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * sequences.shape[0]

    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate_temporal_validation(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    values: dict[str, list[torch.Tensor]] = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }

    for batch in tqdm(loader, desc="temporal validation", leave=False):
        sequences = batch["sequence"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(sequences)
        loss = loss_fn(logits, masks)
        metrics = segmentation_metrics(logits, masks)
        total_loss += float(loss.detach().cpu()) * sequences.shape[0]
        for key in values:
            values[key].append(metrics[key].detach().cpu())

    result = {"loss": total_loss / max(len(loader.dataset), 1)}
    for key, tensors in values.items():
        all_values = torch.cat(tensors) if tensors else torch.zeros(1)
        result[key] = float(all_values.mean())
    return result


def save_temporal_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int | bool],
    output_path: str | Path,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> None:
    checkpoint: dict[str, Any] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if config is not None:
        checkpoint["config"] = config
    ensure_dir(Path(output_path).parent)
    torch.save(checkpoint, output_path)


def fit_temporal_v2(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    early_stopping_patience: int = 10,
    scheduler: Any | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Train with validation-only model selection and early stopping."""
    output_dir = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    history: list[dict[str, float | int | bool]] = []
    best_dice = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate_temporal_validation(model, val_loader, loss_fn, device)
        improved = val_metrics["dice"] > best_dice
        if improved:
            best_dice = val_metrics["dice"]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row: dict[str, float | int | bool] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "best_val_dice": best_dice,
            "is_best": improved,
            "early_stopped": False,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        if scheduler is not None:
            try:
                scheduler.step(val_metrics["loss"])
            except TypeError:
                scheduler.step()

        if improved:
            save_temporal_checkpoint(
                model,
                optimizer,
                epoch,
                row,
                checkpoint_dir / "best_model.pt",
                scheduler=scheduler,
                config=config,
            )
        save_temporal_checkpoint(
            model,
            optimizer,
            epoch,
            row,
            checkpoint_dir / "final_model.pt",
            scheduler=scheduler,
            config=config,
        )

        print(
            f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f} | no_improve={epochs_without_improvement}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            history[-1]["early_stopped"] = True
            pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
            print(f"Early stopping after {epoch} epochs.")
            break

    return pd.DataFrame(history)


def plot_temporal_history(history: pd.DataFrame, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["epoch"], history["val_dice"], label="Dice")
    axes[1].plot(history["epoch"], history["val_iou"], label="IoU")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _batch_value(batch: dict[str, Any], key: str, idx: int, default: Any = None) -> Any:
    value = batch.get(key, default)
    if value is default:
        return default
    if isinstance(value, torch.Tensor):
        selected = value[idx]
        return selected.detach().cpu().tolist() if selected.ndim > 0 else selected.detach().cpu().item()
    return value[idx] if isinstance(value, (list, tuple)) else value


@torch.no_grad()
def evaluate_temporal_test_v2(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
    output_dir: str | Path,
    threshold: float = 0.5,
    max_prediction_examples: int = 10,
    max_area_curve_videos: int = 12,
) -> dict[str, float]:
    """Run the final held-out test evaluation once after checkpoint selection."""
    output_dir = ensure_dir(output_dir)
    predictions_dir = ensure_dir(output_dir / "figures" / "predictions")
    curves_dir = ensure_dir(output_dir / "figures" / "lv_area_curves")
    model.eval()

    total_loss = 0.0
    metric_values: dict[str, list[torch.Tensor]] = {
        "dice": [],
        "iou": [],
        "precision": [],
        "recall": [],
    }
    sample_rows: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    saved_examples = 0

    for batch in tqdm(loader, desc="temporal final test", leave=False):
        sequences = batch["sequence"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(sequences)
        loss = loss_fn(logits, masks)
        metrics = segmentation_metrics(logits, masks, threshold=threshold)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).float()
        center_idx = sequences.shape[1] // 2

        total_loss += float(loss.detach().cpu()) * sequences.shape[0]
        for key in metric_values:
            metric_values[key].append(metrics[key].detach().cpu())

        for idx in range(sequences.shape[0]):
            sample_id = str(_batch_value(batch, "id", idx, f"sample_{len(sample_rows)}"))
            video_id = str(_batch_value(batch, "video_id", idx, "unknown"))
            frame_idx = int(_batch_value(batch, "frame_idx", idx, -1))
            fps = float(_batch_value(batch, "fps", idx, float("nan")))
            frame_indices = _batch_value(batch, "frame_indices", idx, [])
            window_span_frames = int(_batch_value(batch, "window_span_frames", idx, 0))
            window_span_seconds = float(_batch_value(batch, "window_span_seconds", idx, float("nan")))

            pred_mask = predictions[idx, 0].detach().cpu().numpy().astype(np.uint8)
            target_mask = masks[idx, 0].detach().cpu().numpy().astype(np.uint8)
            center = sequences[idx, center_idx, 0].detach().cpu().numpy()
            pred_area = int(pred_mask.sum())
            target_area = int(target_mask.sum())

            sample_rows.append(
                {
                    "id": sample_id,
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "fps": fps,
                    "frame_indices": " ".join(str(x) for x in frame_indices),
                    "window_span_frames": window_span_frames,
                    "window_span_seconds": window_span_seconds,
                    "prediction_area_px": pred_area,
                    "target_area_px": target_area,
                    "dice": float(metrics["dice"][idx].detach().cpu()),
                    "iou": float(metrics["iou"][idx].detach().cpu()),
                    "precision": float(metrics["precision"][idx].detach().cpu()),
                    "recall": float(metrics["recall"][idx].detach().cpu()),
                }
            )
            prediction_records.append(
                {
                    "video_id": video_id,
                    "frame_idx": frame_idx,
                    "fps": fps,
                    "prediction_area_px": pred_area,
                    "target_area_px": target_area,
                    "prediction_mask": pred_mask,
                }
            )

            if saved_examples < max_prediction_examples:
                center_uint8 = np.clip(center * 255.0, 0, 255).astype(np.uint8)
                pred_uint8 = pred_mask * 255
                fig, axes = plt.subplots(1, 4, figsize=(14, 4))
                axes[0].imshow(center_uint8, cmap="gray")
                axes[0].set_title("Center frame")
                axes[1].imshow(target_mask, cmap="gray", vmin=0, vmax=1)
                axes[1].set_title("Ground truth")
                axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
                axes[2].set_title("Prediction")
                axes[3].imshow(overlay_mask(center_uint8, pred_uint8))
                axes[3].set_title("Prediction overlay")
                for axis in axes:
                    axis.axis("off")
                fig.tight_layout()
                fig.savefig(predictions_dir / f"prediction_{sample_id}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved_examples += 1

    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(output_dir / "test_sample_metrics.csv", index=False)
    temporal_summary = summarize_temporal_consistency(
        prediction_records,
        curves_dir=curves_dir,
        max_area_curve_videos=max_area_curve_videos,
    )

    result = {"test_loss": total_loss / max(len(loader.dataset), 1)}
    for key, tensors in metric_values.items():
        all_values = torch.cat(tensors) if tensors else torch.zeros(1)
        result[f"test_{key}"] = float(all_values.mean())
    result.update(temporal_summary["aggregate_metrics"])
    result["prediction_examples_saved"] = saved_examples

    pd.DataFrame([result]).to_csv(output_dir / "test_metrics_table.csv", index=False)
    save_json(result, output_dir / "test_metrics.json")
    return result


def summarize_temporal_consistency(
    prediction_records: list[dict[str, Any]],
    curves_dir: str | Path,
    max_area_curve_videos: int = 12,
) -> dict[str, Any]:
    curves_dir = ensure_dir(curves_dir)
    per_video_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    plotted = 0

    by_video: dict[str, list[dict[str, Any]]] = {}
    for record in prediction_records:
        by_video.setdefault(str(record["video_id"]), []).append(record)

    for video_id, records in by_video.items():
        ordered = sorted(records, key=lambda item: int(item["frame_idx"]))
        frames = np.array([int(item["frame_idx"]) for item in ordered], dtype=float)
        fps_values = [float(item["fps"]) for item in ordered if np.isfinite(float(item["fps"])) and float(item["fps"]) > 0]
        fps = float(np.median(fps_values)) if fps_values else float("nan")
        time_seconds = frames / fps if np.isfinite(fps) and fps > 0 else frames
        areas = np.array([float(item["prediction_area_px"]) for item in ordered], dtype=float)
        targets = np.array([float(item["target_area_px"]) for item in ordered], dtype=float)
        masks = [item["prediction_mask"] for item in ordered]

        area_change = np.diff(areas)
        frame_delta = np.diff(frames)
        time_delta = np.diff(time_seconds)
        valid_dt = np.where(time_delta > 0, time_delta, np.nan)
        area_velocity = area_change / valid_dt if len(area_change) else np.array([])
        area_acceleration = np.diff(area_velocity) if len(area_velocity) > 1 else np.array([])
        area_jerk = np.diff(area_acceleration) if len(area_acceleration) > 1 else np.array([])
        adjacent_dice = np.array(
            [dice_between_masks(masks[i], masks[i + 1]) for i in range(len(masks) - 1)],
            dtype=float,
        )

        row = {
            "video_id": video_id,
            "n_labeled_frames": len(ordered),
            "fps": fps,
            "mean_prediction_area_px": float(np.mean(areas)) if len(areas) else float("nan"),
            "mean_target_area_px": float(np.mean(targets)) if len(targets) else float("nan"),
            "mean_abs_area_change_px": float(np.nanmean(np.abs(area_change))) if len(area_change) else float("nan"),
            "mean_abs_area_velocity_px_per_s": float(np.nanmean(np.abs(area_velocity))) if len(area_velocity) else float("nan"),
            "mean_abs_area_acceleration": float(np.nanmean(np.abs(area_acceleration))) if len(area_acceleration) else float("nan"),
            "mean_abs_area_jerk": float(np.nanmean(np.abs(area_jerk))) if len(area_jerk) else float("nan"),
            "mean_adjacent_prediction_dice": float(np.nanmean(adjacent_dice)) if len(adjacent_dice) else float("nan"),
            "mean_frame_delta": float(np.nanmean(frame_delta)) if len(frame_delta) else float("nan"),
        }
        per_video_rows.append(row)

        for frame, time_s, area, target in zip(frames, time_seconds, areas, targets):
            curve_rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": int(frame),
                    "time_seconds": float(time_s),
                    "prediction_area_px": float(area),
                    "target_area_px": float(target),
                }
            )

        if plotted < max_area_curve_videos and len(ordered) >= 2:
            fig, axis = plt.subplots(figsize=(7, 4))
            axis.plot(time_seconds, areas, marker="o", label="Prediction")
            axis.plot(time_seconds, targets, marker="x", label="Ground truth")
            axis.set_title(f"LV area curve: {video_id}")
            axis.set_xlabel("Time (s)" if np.isfinite(fps) and fps > 0 else "Frame index")
            axis.set_ylabel("LV area (pixels)")
            axis.legend()
            fig.tight_layout()
            fig.savefig(curves_dir / f"lv_area_curve_{video_id}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            plotted += 1

    per_video_df = pd.DataFrame(per_video_rows)
    curve_df = pd.DataFrame(curve_rows)
    per_video_df.to_csv(curves_dir.parent.parent / "temporal_consistency_per_video.csv", index=False)
    curve_df.to_csv(curves_dir.parent.parent / "lv_area_curve_samples.csv", index=False)

    aggregate_metrics: dict[str, float] = {}
    for column in [
        "mean_abs_area_change_px",
        "mean_abs_area_velocity_px_per_s",
        "mean_abs_area_acceleration",
        "mean_abs_area_jerk",
        "mean_adjacent_prediction_dice",
    ]:
        aggregate_metrics[f"temporal_{column}"] = (
            float(per_video_df[column].mean(skipna=True)) if column in per_video_df else float("nan")
        )
    aggregate_metrics["temporal_videos_with_multiple_labeled_frames"] = float(
        (per_video_df["n_labeled_frames"] >= 2).sum() if "n_labeled_frames" in per_video_df else 0
    )
    aggregate_metrics["temporal_area_curve_plots_saved"] = float(plotted)

    pd.DataFrame([aggregate_metrics]).to_csv(curves_dir.parent.parent / "temporal_consistency_metrics_table.csv", index=False)
    return {"per_video": per_video_df, "curves": curve_df, "aggregate_metrics": aggregate_metrics}


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_experiment_log(
    output_path: str | Path,
    config: dict[str, Any],
    best_val_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None = None,
) -> None:
    lines = [
        "# ConvLSTM U-Net Variable-Stride Experiment",
        "",
        "## Configuration",
        "",
        *[f"- `{key}`: {value}" for key, value in config.items()],
        "",
        "## Best-checkpoint validation metrics",
        "",
        *[f"- `{key}`: {value}" for key, value in best_val_metrics.items()],
    ]
    if test_metrics is not None:
        lines.extend(
            [
                "",
                "## Final held-out test metrics",
                "",
                *[f"- `{key}`: {value}" for key, value in test_metrics.items()],
            ]
        )
    ensure_dir(Path(output_path).parent)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
