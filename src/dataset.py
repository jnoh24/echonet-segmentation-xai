from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
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


def load_temporal_metadata(metadata_path: str | Path) -> list[dict[str, str | int]]:
    """Load processed ED/ES target-frame records created by notebook 02.

    The processed metadata stores one row per labeled EchoNet frame. Each row
    remains one training sample; temporal context is read from the source AVI at
    dataset access time.
    """
    metadata_path = Path(metadata_path)
    metadata = pd.read_csv(metadata_path)
    required = {"video_id", "frame_idx", "image_path", "mask_path"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"metadata.csv is missing columns: {sorted(missing)}")

    base_dir = metadata_path.parent
    samples: list[dict[str, str | int]] = []
    for row in metadata.itertuples(index=False):
        image_path = Path(str(row.image_path))
        mask_path = Path(str(row.mask_path))
        if not image_path.is_absolute():
            image_path = base_dir / image_path
        if not mask_path.is_absolute():
            mask_path = base_dir / mask_path

        video_id = str(row.video_id)
        frame_idx = int(row.frame_idx)
        samples.append(
            {
                "id": f"{video_id}_frame{frame_idx:04d}",
                "video_id": video_id,
                "frame_idx": frame_idx,
                "image": str(image_path),
                "mask": str(mask_path),
            }
        )
    return samples


class EchoNetTemporalDataset(Dataset):
    """Load target-aligned temporal AVI sequences and processed target masks.

    For a target frame ``t``, sequence offsets are
    ``[-num_frames_before * temporal_stride, ..., 0, ..., num_frames_after *
    temporal_stride]``. Requested frame indices outside the video range are
    clamped, so ED/ES samples are preserved even when their windows cross a
    boundary.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, str | int]],
        videos_dir: str | Path,
        num_frames_before: int = 12,
        num_frames_after: int = 12,
        temporal_stride: int = 2,
        image_size: tuple[int, int] = (112, 112),
        augment: bool = False,
    ) -> None:
        if num_frames_before < 0 or num_frames_after < 0:
            raise ValueError("num_frames_before and num_frames_after must be non-negative.")
        if temporal_stride < 1:
            raise ValueError("temporal_stride must be a positive integer.")

        self.samples = list(samples)
        self.videos_dir = Path(videos_dir)
        self.num_frames_before = int(num_frames_before)
        self.num_frames_after = int(num_frames_after)
        self.temporal_stride = int(temporal_stride)
        self.target_idx = self.num_frames_before
        self.offsets = [
            offset * self.temporal_stride
            for offset in range(-self.num_frames_before, self.num_frames_after + 1)
        ]
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sequence_length(self) -> int:
        return len(self.offsets)

    def _video_path(self, video_id: str) -> Path:
        filename = video_id if video_id.lower().endswith(".avi") else f"{video_id}.avi"
        return self.videos_dir / filename

    def _read_sequence(self, video_path: Path, target_frame_idx: int) -> tuple[np.ndarray, list[int], int]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise ValueError(f"Video reports no frames: {video_path}")

        frames: list[np.ndarray] = []
        frame_indices: list[int] = []
        for offset in self.offsets:
            frame_idx = min(max(target_frame_idx + offset, 0), frame_count - 1)
            frame_indices.append(frame_idx)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                cap.release()
                raise ValueError(f"Could not read frame {frame_idx} from {video_path}")

            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(
                frame,
                (self.image_size[1], self.image_size[0]),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(frame.astype(np.float32) / 255.0)

        cap.release()
        return np.stack(frames, axis=0), frame_indices, frame_count

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[idx]
        video_id = str(sample["video_id"])
        target_frame_idx = int(sample["frame_idx"])
        sequence, frame_indices, frame_count = self._read_sequence(
            self._video_path(video_id),
            target_frame_idx,
        )

        mask = cv2.imread(str(sample["mask"]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {sample['mask']}")
        mask = cv2.resize(
            mask,
            (self.image_size[1], self.image_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0).astype(np.float32)

        sequence_t = torch.from_numpy(sequence).unsqueeze(1)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        if self.augment:
            if random.random() < 0.5:
                sequence_t = torch.flip(sequence_t, dims=(-1,))
                mask_t = torch.flip(mask_t, dims=(-1,))
            if random.random() < 0.25:
                k = random.randint(1, 3)
                sequence_t = torch.rot90(sequence_t, k=k, dims=(-2, -1))
                mask_t = torch.rot90(mask_t, k=k, dims=(-2, -1))

        return {
            "sequence": sequence_t.contiguous(),
            "mask": mask_t.contiguous(),
            "id": str(sample["id"]),
            "video_id": video_id,
            "frame_idx": target_frame_idx,
            "target_idx": self.target_idx,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "frame_count": frame_count,
            "temporal_stride": self.temporal_stride,
        }


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
