from __future__ import annotations

import random
from pathlib import Path
from typing import Mapping, Sequence

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


def build_fps_lookup(file_list: pd.DataFrame) -> dict[str, float]:
    """Map EchoNet video stems to FPS values from FileList.csv."""
    if "FileName" not in file_list.columns or "FPS" not in file_list.columns:
        return {}

    fps_by_video: dict[str, float] = {}
    for row in file_list.itertuples(index=False):
        file_name = str(getattr(row, "FileName"))
        stem = Path(file_name).stem
        try:
            fps = float(getattr(row, "FPS"))
        except (TypeError, ValueError):
            continue
        if fps > 0:
            fps_by_video[stem] = fps
    return fps_by_video


class EchoNetTemporalVariableStrideDataset(Dataset):
    """Load an odd-length temporal window with configurable frame spacing.

    For sequence_length=5 and temporal_stride=6, the frame offsets are
    [-12, -6, 0, 6, 12]. Boundary indices are clamped to the valid video range.
    """

    def __init__(
        self,
        samples: Sequence[dict[str, str | int]],
        videos_dir: str | Path,
        sequence_length: int = 5,
        temporal_stride: int = 1,
        image_size: tuple[int, int] = (112, 112),
        augment: bool = False,
        fps_by_video: Mapping[str, float] | None = None,
    ) -> None:
        if sequence_length < 1 or sequence_length % 2 == 0:
            raise ValueError("sequence_length must be a positive odd integer.")
        if temporal_stride < 1:
            raise ValueError("temporal_stride must be a positive integer.")

        self.samples = list(samples)
        self.videos_dir = Path(videos_dir)
        self.sequence_length = sequence_length
        self.temporal_stride = temporal_stride
        self.radius = sequence_length // 2
        self.offsets = [offset * temporal_stride for offset in range(-self.radius, self.radius + 1)]
        self.image_size = image_size
        self.augment = augment
        self.fps_by_video = dict(fps_by_video or {})

    def __len__(self) -> int:
        return len(self.samples)

    def _video_path(self, video_id: str) -> Path:
        filename = video_id if video_id.lower().endswith(".avi") else f"{video_id}.avi"
        return self.videos_dir / filename

    def _fps_for_video(self, video_id: str) -> float:
        stem = Path(video_id).stem
        fps = self.fps_by_video.get(stem)
        return float(fps) if fps and fps > 0 else float("nan")

    def _read_sequence(self, video_path: Path, center_idx: int) -> tuple[np.ndarray, list[int], int]:
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
            frame_idx = min(max(center_idx + offset, 0), frame_count - 1)
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int | float]:
        sample = self.samples[idx]
        video_id = str(sample["video_id"])
        center_idx = int(sample["frame_idx"])
        sequence, frame_indices, frame_count = self._read_sequence(self._video_path(video_id), center_idx)

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

        fps = self._fps_for_video(video_id)
        span_frames = int(max(frame_indices) - min(frame_indices))
        span_seconds = span_frames / fps if np.isfinite(fps) and fps > 0 else float("nan")

        return {
            "sequence": sequence_t.contiguous(),
            "mask": mask_t.contiguous(),
            "id": str(sample["id"]),
            "video_id": video_id,
            "frame_idx": center_idx,
            "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "fps": fps,
            "frame_count": frame_count,
            "temporal_stride": self.temporal_stride,
            "window_span_frames": span_frames,
            "window_span_seconds": span_seconds,
        }
