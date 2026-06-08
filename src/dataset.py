from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

try:
    from monai.transforms import (
        Compose,
        EnsureChannelFirstd,
        EnsureTyped,
        LoadImaged,
        RandFlipd,
        RandRotate90d,
        Resized,
        ScaleIntensityd,
    )
except ImportError:  # Keeps lightweight exploration imports usable before MONAI install.
    Compose = None

from .utils import DEFAULT_PROCESSED_DIR, list_image_mask_pairs, normalize_video_name


@dataclass(frozen=True)
class SplitConfig:
    val_size: float = 0.15
    test_size: float = 0.15
    seed: int = 42


class EchoNetProcessedDataset(Dataset):
    """Dataset for processed EchoNet frame/mask PNG pairs.

    Images are returned as float tensors with shape [1, H, W]. Masks are binary
    float tensors with shape [1, H, W]. This simple Dataset is useful when MONAI
    dictionary transforms are not needed.
    """

    def __init__(self, samples: Sequence[dict[str, str]], image_size: tuple[int, int] | None = (112, 112)):
        self.samples = list(samples)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        image = cv2.imread(sample["image"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(sample["mask"], cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {sample['image']}")
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {sample['mask']}")

        if self.image_size is not None:
            width, height = self.image_size[1], self.image_size[0]
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return {"image": image_t, "mask": mask_t, "id": sample.get("id", Path(sample["image"]).stem)}


def get_monai_transforms(
    image_size: tuple[int, int] = (112, 112),
    augment: bool = False,
):
    """Build MONAI dictionary transforms for binary 2D segmentation."""
    if Compose is None:
        raise ImportError("MONAI is required for get_monai_transforms(). Install monai first.")

    keys = ["image", "mask"]
    transforms = [
        LoadImaged(keys=keys, image_only=True),
        EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        ScaleIntensityd(keys=keys),
        Resized(keys=keys, spatial_size=image_size, mode=("bilinear", "nearest")),
    ]
    if augment:
        transforms.extend(
            [
                RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
                RandRotate90d(keys=keys, prob=0.25, max_k=3),
            ]
        )
    transforms.append(EnsureTyped(keys=keys, dtype=torch.float32))
    return Compose(transforms)


def load_processed_samples(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> list[dict[str, str]]:
    processed_dir = Path(processed_dir)
    return list_image_mask_pairs(processed_dir / "images", processed_dir / "masks")


def split_samples(
    samples: Sequence[dict[str, str]],
    config: SplitConfig = SplitConfig(),
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Create reproducible train/validation/test splits from processed pairs."""
    samples = list(samples)
    if not samples:
        raise ValueError("No samples were provided for splitting.")
    if len(samples) < 3:
        raise ValueError("At least three samples are required for train/val/test splitting.")

    train_val, test = train_test_split(
        samples,
        test_size=config.test_size,
        random_state=config.seed,
        shuffle=True,
    )
    val_fraction_of_train_val = config.val_size / (1.0 - config.test_size)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction_of_train_val,
        random_state=config.seed,
        shuffle=True,
    )
    return list(train), list(val), list(test)


def split_by_echonet_filelist(
    samples: Sequence[dict[str, str]],
    file_list: pd.DataFrame,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    """Split processed samples using EchoNet's official TRAIN/VAL/TEST labels.

    Processed sample ids are expected to begin with the video stem, for example
    `0XABC123_frame0042`. If a sample cannot be matched to FileList.csv it is
    skipped, which keeps Kaggle reruns robust when partial preprocessing exists.
    """
    split_lookup = {
        Path(normalize_video_name(row.FileName)).stem: str(row.Split).upper()
        for row in file_list.itertuples(index=False)
    }

    buckets = {"TRAIN": [], "VAL": [], "TEST": []}
    for sample in samples:
        stem = sample.get("id", Path(sample["image"]).stem)
        video_stem = stem.split("_frame")[0]
        split = split_lookup.get(video_stem)
        if split in buckets:
            buckets[split].append(sample)
    return buckets["TRAIN"], buckets["VAL"], buckets["TEST"]
