from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "EchoNet-Dynamic"
DEFAULT_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def set_seed(seed: int = 42) -> None:
    """Set common random seeds for reproducible preprocessing and training."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_echonet_tables(raw_dir: str | Path = DEFAULT_RAW_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FileList.csv and VolumeTracings.csv from an EchoNet-Dynamic folder."""
    raw_dir = Path(raw_dir)
    file_list = pd.read_csv(raw_dir / "FileList.csv")
    tracings = pd.read_csv(raw_dir / "VolumeTracings.csv")
    return file_list, tracings


def normalize_video_name(file_name: str | Path) -> str:
    """Return an EchoNet video filename with the .avi suffix."""
    name = Path(str(file_name)).name
    return name if name.lower().endswith(".avi") else f"{name}.avi"


def video_path_from_name(file_name: str | Path, raw_dir: str | Path = DEFAULT_RAW_DIR) -> Path:
    return Path(raw_dir) / "Videos" / normalize_video_name(file_name)


def read_video_frame(video_path: str | Path, frame_idx: int) -> np.ndarray:
    """Read one RGB frame from an EchoNet AVI.

    EchoNet tracing frame indices are stored as zero-based integer frame numbers.
    OpenCV returns BGR frames, so this helper converts to RGB for plotting and PNG
    export consistency.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def tracing_group_to_polygon(tracing_rows: pd.DataFrame) -> np.ndarray:
    """Convert EchoNet tracing rows for one frame into a closed LV polygon.

    Each row stores paired contour points from opposite sides of the left
    ventricle: (X1, Y1) and (X2, Y2). The polygon follows the first side in row
    order and returns along the second side in reverse row order.
    """
    required = {"X1", "Y1", "X2", "Y2"}
    missing = required.difference(tracing_rows.columns)
    if missing:
        raise ValueError(f"Tracing rows are missing columns: {sorted(missing)}")

    rows = tracing_rows.dropna(subset=["X1", "Y1", "X2", "Y2"])
    if rows.empty:
        raise ValueError("No valid tracing coordinates were provided.")

    side_a = rows[["X1", "Y1"]].to_numpy(dtype=np.float32)
    side_b = rows[["X2", "Y2"]].to_numpy(dtype=np.float32)[::-1]
    return np.vstack([side_a, side_b])


def tracing_group_to_mask(tracing_rows: pd.DataFrame, image_shape: Sequence[int]) -> np.ndarray:
    """Rasterize one traced frame into a binary uint8 mask with values {0, 255}."""
    height, width = int(image_shape[0]), int(image_shape[1])
    polygon = tracing_group_to_polygon(tracing_rows)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    polygon_int = np.round(polygon).astype(np.int32)

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_int], color=255)
    return mask


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 48, 48),
    alpha: float = 0.35,
) -> np.ndarray:
    """Create an RGB mask overlay for qualitative segmentation checks."""
    image_rgb = image
    if image_rgb.ndim == 2:
        image_rgb = np.repeat(image_rgb[..., None], 3, axis=-1)
    image_rgb = image_rgb.astype(np.float32)

    color_arr = np.array(color, dtype=np.float32)
    mask_bool = mask > 0
    overlay = image_rgb.copy()
    overlay[mask_bool] = (1 - alpha) * overlay[mask_bool] + alpha * color_arr
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_mask_sanity_figure(
    frame: np.ndarray,
    mask: np.ndarray,
    polygon: np.ndarray | None,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save frame, binary mask, and overlay panels for mask validation."""
    ensure_dir(Path(output_path).parent)
    overlay = overlay_mask(frame, mask)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(frame, cmap="gray")
    if polygon is not None:
        closed = np.vstack([polygon, polygon[:1]])
        axes[0].plot(closed[:, 0], closed[:, 1], color="yellow", linewidth=1.5)
    axes[0].set_title("Frame + tracing")
    axes[1].imshow(mask, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("Binary mask")
    axes[2].imshow(overlay)
    axes[2].set_title("Mask overlay")

    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_json(data: dict, output_path: str | Path) -> None:
    ensure_dir(Path(output_path).parent)
    with Path(output_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def dice_coefficient_np(pred_mask: np.ndarray, true_mask: np.ndarray, eps: float = 1e-7) -> float:
    pred = (pred_mask > 0).astype(np.float32)
    true = (true_mask > 0).astype(np.float32)
    intersection = float((pred * true).sum())
    return (2.0 * intersection + eps) / (float(pred.sum() + true.sum()) + eps)


def list_image_mask_pairs(
    image_dir: str | Path = DEFAULT_PROCESSED_DIR / "images",
    mask_dir: str | Path = DEFAULT_PROCESSED_DIR / "masks",
    suffixes: Iterable[str] = (".png", ".jpg", ".jpeg", ".npy"),
) -> list[dict[str, str]]:
    """Return sorted image-mask pairs with matching stems."""
    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)
    suffixes = tuple(s.lower() for s in suffixes)

    images = {p.stem: p for p in image_dir.glob("*") if p.suffix.lower() in suffixes}
    masks = {p.stem: p for p in mask_dir.glob("*") if p.suffix.lower() in suffixes}
    common = sorted(images.keys() & masks.keys())
    return [{"image": str(images[stem]), "mask": str(masks[stem]), "id": stem} for stem in common]


def preprocess_traced_frames(
    tracings: pd.DataFrame,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    output_dir: str | Path = DEFAULT_PROCESSED_DIR,
    figures_dir: str | Path = DEFAULT_FIGURES_DIR,
    max_samples: int | None = None,
    save_examples: int = 8,
) -> dict[str, int]:
    """Convert all traced EchoNet frames into image/mask PNG pairs.

    The output filename pattern is `<video_stem>_frame<frame_idx:04d>.png`.
    Rerunning is safe: existing files are overwritten with deterministic names.
    Missing or unreadable videos are skipped and counted in the returned summary.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    image_dir = ensure_dir(output_dir / "images")
    mask_dir = ensure_dir(output_dir / "masks")
    figures_dir = ensure_dir(figures_dir)

    grouped = list(tracings.groupby(["FileName", "Frame"], sort=True))
    if max_samples is not None:
        grouped = grouped[:max_samples]

    saved = 0
    skipped = 0
    examples = 0

    for (file_name, frame_idx), rows in tqdm(grouped, desc="preprocess traced frames"):
        video_path = video_path_from_name(file_name, raw_dir)
        if not video_path.exists():
            skipped += 1
            continue

        try:
            frame = read_video_frame(video_path, int(frame_idx))
            mask = tracing_group_to_mask(rows, frame.shape[:2])
        except (FileNotFoundError, ValueError):
            skipped += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        stem = f"{Path(normalize_video_name(file_name)).stem}_frame{int(frame_idx):04d}"
        cv2.imwrite(str(image_dir / f"{stem}.png"), gray)
        cv2.imwrite(str(mask_dir / f"{stem}.png"), mask)
        saved += 1

        if examples < save_examples:
            polygon = tracing_group_to_polygon(rows)
            save_mask_sanity_figure(
                frame,
                mask,
                polygon,
                figures_dir / f"preprocess_example_{stem}.png",
                title=stem,
            )
            examples += 1

    summary = {"saved": saved, "skipped": skipped, "examples": examples}
    save_json(summary, output_dir / "preprocess_summary.json")
    return summary
