from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_temporal_metadata(metadata_path: str | Path) -> list[dict[str, str | int]]:
    """Load processed center-frame records created by notebook 02."""
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
    """Load a five-frame AVI sequence and the processed center-frame mask.

    Neighboring frame indices are clamped at video boundaries. Spatial
    augmentation is applied identically to every frame and the target mask.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, str | int]],
        videos_dir: str | Path,
        sequence_length: int = 5,
        image_size: tuple[int, int] = (112, 112),
        augment: bool = False,
    ) -> None:
        if sequence_length < 1 or sequence_length % 2 == 0:
            raise ValueError("sequence_length must be a positive odd integer.")

        self.samples = list(samples)
        self.videos_dir = Path(videos_dir)
        self.sequence_length = sequence_length
        self.radius = sequence_length // 2
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def _video_path(self, video_id: str) -> Path:
        filename = video_id if video_id.lower().endswith(".avi") else f"{video_id}.avi"
        return self.videos_dir / filename

    def _read_sequence(self, video_path: Path, center_idx: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            cap.release()
            raise ValueError(f"Video reports no frames: {video_path}")

        frames: list[np.ndarray] = []
        for offset in range(-self.radius, self.radius + 1):
            frame_idx = min(max(center_idx + offset, 0), frame_count - 1)
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
        return np.stack(frames, axis=0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[idx]
        video_id = str(sample["video_id"])
        center_idx = int(sample["frame_idx"])
        sequence = self._read_sequence(self._video_path(video_id), center_idx)

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
            "frame_idx": center_idx,
        }
