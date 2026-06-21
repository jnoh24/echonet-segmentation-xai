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


def get_unetr_loss() -> DiceCELoss:
    """Use the same combined Dice and BCE-style loss as the U-Net baseline."""
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
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, predictions.ndim))
    intersection = (predictions * targets).sum(dim=dims)
    pred_area = predictions.sum(dim=dims)
    target_area = targets.sum(dim=dims)
    union = pred_area + target_area - intersection
    dice = (2.0 * intersection + eps) / (pred_area + target_area + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, desc="UNETR train", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu()) * images.shape[0]

    return total_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate_unetr(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    dice_values: list[torch.Tensor] = []
    iou_values: list[torch.Tensor] = []

    for batch in tqdm(loader, desc="UNETR eval", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        dice, iou = segmentation_metrics(logits, masks)

        total_loss += float(loss.detach().cpu()) * images.shape[0]
        dice_values.append(dice.detach().cpu())
        iou_values.append(iou.detach().cpu())

    dice_all = torch.cat(dice_values) if dice_values else torch.zeros(1)
    iou_all = torch.cat(iou_values) if iou_values else torch.zeros(1)
    return {
        "loss": total_loss / max(len(loader.dataset), 1),
        "dice": float(dice_all.mean()),
        "iou": float(iou_all.mean()),
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float | int],
    output_path: str | Path,
    scheduler: Any | None = None,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    ensure_dir(Path(output_path).parent)
    torch.save(checkpoint, output_path)


def fit_unetr(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epochs: int,
    output_dir: str | Path,
    scheduler: Any | None = None,
) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    history: list[dict[str, float | int]] = []
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate_unetr(model, val_loader, loss_fn, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
            "val_iou": val_metrics["iou"],
        }
        history.append(row)
        history_df = pd.DataFrame(history)
        history_df.to_csv(output_dir / "history.csv", index=False)

        if scheduler is not None:
            try:
                scheduler.step(val_metrics["loss"])
            except TypeError:
                scheduler.step()

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(
                model,
                optimizer,
                epoch,
                row,
                checkpoint_dir / "best_model.pt",
                scheduler,
            )

        save_checkpoint(
            model,
            optimizer,
            epoch,
            row,
            checkpoint_dir / "final_model.pt",
            scheduler,
        )
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | val_iou={val_metrics['iou']:.4f}"
        )

    return pd.DataFrame(history)


def plot_history(history: pd.DataFrame, output_path: str | Path) -> None:
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


@torch.no_grad()
def save_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    max_examples: int = 10,
    threshold: float = 0.5,
) -> int:
    output_dir = ensure_dir(output_dir)
    model.eval()
    saved = 0

    for batch in tqdm(loader, desc="UNETR predictions", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        predictions = (torch.sigmoid(model(images)) >= threshold).float()
        ids = batch.get("id", [f"sample_{index}" for index in range(images.shape[0])])

        for index in range(images.shape[0]):
            image = images[index, 0].detach().cpu().numpy()
            target = masks[index, 0].detach().cpu().numpy()
            prediction = predictions[index, 0].detach().cpu().numpy()
            image_uint8 = np.clip(image * 255.0, 0, 255).astype(np.uint8)
            prediction_uint8 = (prediction > 0).astype(np.uint8) * 255

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(image_uint8, cmap="gray")
            axes[0].set_title("Frame")
            axes[1].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            axes[2].imshow(prediction, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Prediction")
            axes[3].imshow(overlay_mask(image_uint8, prediction_uint8))
            axes[3].set_title("Prediction overlay")
            for axis in axes:
                axis.axis("off")

            sample_id = str(ids[index])
            fig.tight_layout()
            fig.savefig(
                output_dir / f"prediction_{sample_id}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            saved += 1
            if saved >= max_examples:
                return saved
    return saved


def save_json(data: dict[str, Any], output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    with Path(output_path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def write_experiment_log(
    output_path: str | Path,
    config: dict[str, Any],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> None:
    lines = [
        "# UNETR Experiment",
        "",
        "## Configuration",
        "",
        *[f"- `{key}`: {value}" for key, value in config.items()],
        "",
        "## Best-checkpoint validation metrics",
        "",
        *[f"- `{key}`: {value:.6f}" for key, value in val_metrics.items()],
        "",
        "## Held-out test metrics",
        "",
        *[f"- `{key}`: {value:.6f}" for key, value in test_metrics.items()],
        "",
    ]
    ensure_dir(Path(output_path).parent)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
