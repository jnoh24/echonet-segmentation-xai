from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import normalize_video_name, tracing_group_to_polygon


@dataclass(frozen=True)
class CardiacCycleWindowConfig:
    clip_length: int = 32
    image_size: tuple[int, int] = (112, 112)
    cycle_scale: float = 2.0
    min_window_frames: int = 32
    force_include_ed_es: bool = True
    prefer_true_cycle: bool = True


def polygon_area(points: np.ndarray) -> float:
    x = points[:, 0].astype(np.float64)
    y = points[:, 1].astype(np.float64)
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def video_frame_count(video_path: str | Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if count <= 0:
        raise ValueError(f"Video reports no frames: {video_path}")
    return count


def tracing_phase_table(volume_tracings: pd.DataFrame) -> pd.DataFrame:
    required = {"FileName", "Frame", "X1", "Y1", "X2", "Y2"}
    missing = required.difference(volume_tracings.columns)
    if missing:
        raise ValueError(f"VolumeTracings.csv is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    grouped = volume_tracings.groupby(["FileName", "Frame"], sort=True)
    for (file_name, frame_idx), frame_rows in grouped:
        polygon = tracing_group_to_polygon(frame_rows)
        rows.append(
            {
                "video_id": Path(normalize_video_name(file_name)).stem,
                "file_name": normalize_video_name(file_name),
                "frame_idx": int(frame_idx),
                "lv_area": polygon_area(polygon),
            }
        )

    frame_table = pd.DataFrame(rows)
    if frame_table.empty:
        raise ValueError("No tracing frames were found in VolumeTracings.csv.")

    phase_rows: list[dict[str, Any]] = []
    for video_id, group in frame_table.groupby("video_id", sort=True):
        group = group.sort_values("frame_idx")
        if len(group) < 2:
            continue
        ed = group.loc[group["lv_area"].idxmax()]
        es = group.loc[group["lv_area"].idxmin()]
        phase_rows.append(
            {
                "video_id": video_id,
                "file_name": str(ed["file_name"]),
                "ed_frame_idx": int(ed["frame_idx"]),
                "es_frame_idx": int(es["frame_idx"]),
                "ed_lv_area": float(ed["lv_area"]),
                "es_lv_area": float(es["lv_area"]),
                "ed_es_distance_frames": abs(int(ed["frame_idx"]) - int(es["frame_idx"])),
                "phase_source": "VolumeTracings.csv; ED=max traced LV area, ES=min traced LV area",
            }
        )
    return pd.DataFrame(phase_rows)


def _window_bounds(ed_frame: int, es_frame: int, frame_count: int, config: CardiacCycleWindowConfig) -> tuple[int, int]:
    distance = max(abs(int(ed_frame) - int(es_frame)), 1)
    window_length = max(int(round(config.cycle_scale * distance)), int(config.min_window_frames), config.clip_length)
    ed_frame = int(ed_frame)
    es_frame = int(es_frame)

    if config.prefer_true_cycle:
        if ed_frame <= es_frame:
            preferred_start = ed_frame
            preferred_end = ed_frame + window_length - 1
            fallback_start = ed_frame - window_length + 1
            fallback_end = ed_frame
        else:
            preferred_start = ed_frame - window_length + 1
            preferred_end = ed_frame
            fallback_start = ed_frame
            fallback_end = ed_frame + window_length - 1

        preferred_contains = 0 <= preferred_start and preferred_end < frame_count and preferred_start <= es_frame <= preferred_end
        fallback_contains = 0 <= fallback_start and fallback_end < frame_count and fallback_start <= es_frame <= fallback_end
        if preferred_contains:
            start, end = preferred_start, preferred_end
        elif fallback_contains:
            start, end = fallback_start, fallback_end
        else:
            start, end = preferred_start, preferred_end
    else:
        center = 0.5 * (ed_frame + es_frame)
        start = int(round(center - window_length / 2.0))
        end = start + window_length - 1

    if start < 0:
        end -= start
        start = 0
    if end >= frame_count:
        shift = end - frame_count + 1
        start = max(0, start - shift)
        end = frame_count - 1
    return int(start), int(end)


def sample_window_indices(
    start_frame: int,
    end_frame: int,
    ed_frame: int,
    es_frame: int,
    config: CardiacCycleWindowConfig,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.linspace(0.0, 1.0, int(config.clip_length), dtype=np.float32)
    sampled = np.rint(np.linspace(start_frame, end_frame, int(config.clip_length))).astype(np.int64)
    if config.force_include_ed_es:
        for annotated_frame in [int(ed_frame), int(es_frame)]:
            nearest = int(np.argmin(np.abs(sampled - annotated_frame)))
            sampled[nearest] = annotated_frame
    sampled = np.clip(sampled, int(start_frame), int(end_frame)).astype(np.int64)
    return sampled, positions


def build_cardiac_cycle_manifest(
    file_list: pd.DataFrame,
    volume_tracings: pd.DataFrame,
    videos_dir: str | Path,
    config: CardiacCycleWindowConfig,
    max_videos: int | None = None,
) -> pd.DataFrame:
    videos_dir = Path(videos_dir)
    phases = tracing_phase_table(volume_tracings)
    echo_table = file_list.copy()
    echo_table["video_id"] = echo_table["FileName"].astype(str).map(lambda x: Path(normalize_video_name(x)).stem)
    merged = echo_table.merge(phases, on="video_id", how="inner", suffixes=("", "_phase"))
    if max_videos is not None:
        merged = merged.iloc[:max_videos].copy()

    rows: list[dict[str, Any]] = []
    for row in merged.itertuples(index=False):
        video_id = str(row.video_id)
        file_name = normalize_video_name(row.FileName)
        video_path = videos_dir / file_name
        if not video_path.exists():
            continue
        frame_count = video_frame_count(video_path)
        ed_frame = int(row.ed_frame_idx)
        es_frame = int(row.es_frame_idx)
        start, end = _window_bounds(ed_frame, es_frame, frame_count, config)
        sampled, positions = sample_window_indices(start, end, ed_frame, es_frame, config)
        rows.append(
            {
                "video_id": video_id,
                "file_name": file_name,
                "video_path": str(video_path),
                "split": str(row.Split).upper(),
                "ef": float(row.EF),
                "edv": float(row.EDV) if hasattr(row, "EDV") and pd.notna(row.EDV) else np.nan,
                "esv": float(row.ESV) if hasattr(row, "ESV") and pd.notna(row.ESV) else np.nan,
                "frame_count": int(frame_count),
                "ed_frame_idx": ed_frame,
                "es_frame_idx": es_frame,
                "ed_es_distance_frames": abs(ed_frame - es_frame),
                "window_start_frame": int(start),
                "window_end_frame": int(end),
                "window_length_frames": int(end - start + 1),
                "sampled_frame_indices": json_dumps_ints(sampled),
                "sampled_normalized_positions": json_dumps_floats(positions),
                "contains_ed_exact_sampled": bool(ed_frame in set(sampled.tolist())),
                "contains_es_exact_sampled": bool(es_frame in set(sampled.tolist())),
                "contains_both_ed_es_exact_sampled": bool(ed_frame in set(sampled.tolist()) and es_frame in set(sampled.tolist())),
                "window_contains_ed": bool(start <= ed_frame <= end),
                "window_contains_es": bool(start <= es_frame <= end),
                "window_contains_both_ed_es": bool(start <= ed_frame <= end and start <= es_frame <= end),
                "phase_source": str(row.phase_source),
            }
        )
    return pd.DataFrame(rows)


def json_dumps_ints(values: Sequence[int] | np.ndarray) -> str:
    return "[" + ",".join(str(int(v)) for v in values) + "]"


def json_dumps_floats(values: Sequence[float] | np.ndarray) -> str:
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [int(v) for v in value.strip("[]").split(",") if v != ""]
    return [int(v) for v in value]


def parse_float_list(value: str | Sequence[float]) -> list[float]:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return [float(v) for v in value.strip("[]").split(",") if v != ""]
    return [float(v) for v in value]


class EchoNetCardiacCycleDataset(Dataset):
    """Load deterministic cardiac-cycle clips for EF regression.

    Videos are returned in torchvision video format: ``[C, T, H, W]``.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        image_size: tuple[int, int] = (112, 112),
        ef_mean: float | None = None,
        ef_std: float | None = None,
        rgb: bool = True,
        imagenet_normalize: bool = True,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_size = image_size
        self.ef_mean = ef_mean
        self.ef_std = ef_std
        self.rgb = bool(rgb)
        self.imagenet_normalize = bool(imagenet_normalize)
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], dtype=torch.float32).view(3, 1, 1, 1)

    def __len__(self) -> int:
        return len(self.manifest)

    def _read_frames(self, video_path: Path, frame_indices: Sequence[int]) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        frames: list[np.ndarray] = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                cap.release()
                raise ValueError(f"Could not read frame {frame_idx} from {video_path}")
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_AREA)
            frames.append(frame.astype(np.float32) / 255.0)
        cap.release()
        array = np.stack(frames, axis=0)
        video = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous()
        if not self.rgb:
            video = video.mean(dim=0, keepdim=True).repeat(3, 1, 1, 1)
        if self.imagenet_normalize:
            video = (video - self.mean) / self.std
        return video

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]
        frame_indices = parse_int_list(row.sampled_frame_indices)
        positions = parse_float_list(row.sampled_normalized_positions)
        ef = float(row.ef)
        item = {
            "video": self._read_frames(Path(row.video_path), frame_indices),
            "ef": torch.tensor(ef, dtype=torch.float32),
            "video_id": str(row.video_id),
            "file_name": str(row.file_name),
            "split": str(row.split),
            "ed_frame_idx": torch.tensor(int(row.ed_frame_idx), dtype=torch.long),
            "es_frame_idx": torch.tensor(int(row.es_frame_idx), dtype=torch.long),
            "window_start_frame": torch.tensor(int(row.window_start_frame), dtype=torch.long),
            "window_end_frame": torch.tensor(int(row.window_end_frame), dtype=torch.long),
            "sampled_frame_indices": torch.tensor(frame_indices, dtype=torch.long),
            "sampled_normalized_positions": torch.tensor(positions, dtype=torch.float32),
        }
        if self.ef_mean is not None and self.ef_std is not None:
            item["ef_normalized"] = torch.tensor((ef - float(self.ef_mean)) / float(self.ef_std), dtype=torch.float32)
        return item
