from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from monai.losses import DiceCELoss, DiceLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .utils import DEFAULT_CHECKPOINT_DIR, DEFAULT_FIGURES_DIR, ensure_dir, overlay_mask


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-7) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()
    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return float(dice.mean().detach().cpu())


def get_loss(name: str = "dice_bce"):
    """Return a binary segmentation loss.

    Dice+BCE is the default because LV masks occupy a small fraction of the
    image; BCE stabilizes early training while Dice optimizes overlap.
    """
    name = name.lower()
    if name in {"dice_bce", "dice_ce", "diceceloss"}:
        return DiceCELoss(sigmoid=True, squared_pred=True, lambda_dice=1.0, lambda_ce=1.0)
    if name in {"dice", "diceloss"}:
        return DiceLoss(sigmoid=True, squared_pred=True)
    raise ValueError(f"Unsupported loss: {name}")


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.detach().cpu()) * images.shape[0]

    return running_loss / max(len(loader.dataset), 1)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    loss_fn,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    running_loss = 0.0
    dice_scores: list[float] = []

    for batch in tqdm(loader, desc="eval", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        loss = loss_fn(logits, masks)

        running_loss += float(loss.detach().cpu()) * images.shape[0]
        dice_scores.append(dice_score_from_logits(logits, masks))

    return {
        "loss": running_loss / max(len(loader.dataset), 1),
        "dice": float(np.mean(dice_scores)) if dice_scores else 0.0,
    }


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    output_path: str | Path,
) -> None:
    ensure_dir(Path(output_path).parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        output_path,
    )


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epochs: int,
    checkpoint_dir: str | Path = DEFAULT_CHECKPOINT_DIR,
) -> dict[str, list[float]]:
    """Train a model and save best/final checkpoints."""
    checkpoint_dir = ensure_dir(checkpoint_dir)
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_dice": []}
    best_dice = -1.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_metrics = evaluate(model, val_loader, loss_fn, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_dice": val_metrics["dice"],
        }
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir / "best_unet.pt")

        save_checkpoint(model, optimizer, epoch, metrics, checkpoint_dir / "final_unet.pt")
        print(
            f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | val_dice={val_metrics['dice']:.4f}"
        )

    return history


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path = DEFAULT_FIGURES_DIR / "training_curves.png") -> None:
    ensure_dir(Path(output_path).parent)
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history.get("train_loss", []), label="train")
    axes[0].plot(epochs, history.get("val_loss", []), label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, history.get("val_dice", []), label="val Dice", color="tab:green")
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def save_prediction_examples(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path = DEFAULT_FIGURES_DIR,
    max_examples: int = 6,
    threshold: float = 0.5,
) -> None:
    """Save frame, ground truth, prediction, and overlay examples."""
    output_dir = ensure_dir(output_dir)
    model.eval()
    saved = 0

    for batch in tqdm(loader, desc="predict examples", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) >= threshold).float()
        ids: Any = batch.get("id", [f"sample_{i}" for i in range(images.shape[0])])

        for i in range(images.shape[0]):
            image_np = images[i, 0].detach().cpu().numpy()
            mask_np = masks[i, 0].detach().cpu().numpy()
            pred_np = preds[i, 0].detach().cpu().numpy()
            image_uint8 = np.clip(image_np * 255.0, 0, 255).astype(np.uint8)
            pred_uint8 = (pred_np > 0).astype(np.uint8) * 255

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(image_uint8, cmap="gray")
            axes[0].set_title("Frame")
            axes[1].imshow(mask_np, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Ground truth")
            axes[2].imshow(pred_np, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Prediction")
            axes[3].imshow(overlay_mask(image_uint8, pred_uint8))
            axes[3].set_title("Prediction overlay")

            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            sample_id = ids[i] if isinstance(ids, (list, tuple)) else f"sample_{saved}"
            fig.savefig(output_dir / f"prediction_{sample_id}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

            saved += 1
            if saved >= max_examples:
                return
