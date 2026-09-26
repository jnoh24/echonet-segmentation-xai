from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ArtifactRemovalConfig:
    """Conservative white annotation-line removal for EchoNet frames.

    Detection is intentionally restricted to peripheral/background image
    regions. If the candidate is not bright, thin, long, and line-like, no mask
    is returned and the source frames are left unchanged.
    """

    enabled: bool = True
    bright_percentile: float = 97.0
    min_brightness: int = 130
    min_length_fraction: float = 0.08
    max_width_pixels: int = 7
    min_aspect_ratio: float = 5.0
    min_line_fill_fraction: float = 0.35
    max_removed_fraction: float = 0.015
    peripheral_top_fraction: float = 0.55
    peripheral_right_fraction: float = 0.55
    diagonal_angle_min_degrees: float = 15.0
    diagonal_angle_max_degrees: float = 75.0
    dilation_pixels: int = 2
    background_percentile: float = 5.0
    hough_threshold: int = 6
    hough_max_line_gap_fraction: float = 0.08
    hough_mask_thickness_pixels: int = 3
    line_candidate_min_brightness: int = 115
    line_candidate_threshold_fraction: float = 0.70
    upper_right_x_fraction: float = 0.60
    upper_right_y_fraction: float = 0.70
    upper_right_min_length_fraction: float = 0.06
    upper_right_max_component_area_fraction: float = 0.004
    upper_right_max_component_extent_fraction: float = 0.18
    upper_right_corridor_half_width_pixels: int = 3
    upper_right_final_dilation_pixels: int = 1
    upper_right_lenient_min_brightness: int = 70
    upper_right_lenient_threshold_fraction: float = 0.45
    upper_right_min_corridor_support_fraction: float = 0.50
    background_floor: int = 20
    upper_right_min_background_fraction: float = 0.90
    upper_right_min_final_background_fraction: float = 0.90
    debug: bool = False
    debug_output_dir: str | None = None

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        return asdict(self)


def _as_gray_uint8(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            frame = frame[..., 0]
    if frame.dtype == np.uint8:
        return frame
    arr = np.asarray(frame, dtype=np.float32)
    if arr.max(initial=0.0) <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _peripheral_roi(shape: tuple[int, int], config: ArtifactRemovalConfig) -> np.ndarray:
    height, width = shape
    roi = np.zeros((height, width), dtype=bool)
    top_limit = int(round(height * config.peripheral_top_fraction))
    right_start = int(round(width * (1.0 - config.peripheral_right_fraction)))
    roi[:top_limit, right_start:] = True

    # Also allow the upper peripheral band, excluding the central cardiac field.
    central_x0 = int(round(width * 0.22))
    central_x1 = int(round(width * 0.72))
    upper_band = np.zeros_like(roi)
    upper_band[: int(round(height * 0.28)), :] = True
    upper_band[:, central_x0:central_x1] = False
    return roi | upper_band


def _angle_from_endpoints(x1: int, y1: int, x2: int, y2: int) -> float:
    angle = abs(np.degrees(np.arctan2(float(y2 - y1), float(x2 - x1))))
    if angle > 90.0:
        angle = 180.0 - angle
    return float(angle)


def _upper_right_roi(shape: tuple[int, int], config: ArtifactRemovalConfig) -> np.ndarray:
    height, width = shape
    roi = np.zeros((height, width), dtype=bool)
    x0 = int(round(width * float(config.upper_right_x_fraction)))
    y1 = int(round(height * float(config.upper_right_y_fraction)))
    roi[:y1, x0:] = True
    return roi


def _compute_background_fraction_map(
    gray_frames: Sequence[np.ndarray],
    config: ArtifactRemovalConfig,
) -> np.ndarray:
    stack = np.stack([_as_gray_uint8(frame) for frame in gray_frames], axis=0)
    return (stack <= int(config.background_floor)).mean(axis=0).astype(np.float32)


def _background_gate(
    background_fraction: np.ndarray | None,
    shape: tuple[int, int],
    config: ArtifactRemovalConfig,
) -> np.ndarray:
    if background_fraction is None:
        warnings.warn(
            "background_fraction was not provided; upper-right artifact detection "
            "will skip the temporal background-consistency gate.",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.ones(shape, dtype=bool)
    gate = np.asarray(background_fraction, dtype=np.float32)
    if gate.shape != shape:
        raise ValueError(f"background_fraction shape {gate.shape} does not match frame shape {shape}.")
    return gate >= float(config.upper_right_min_background_fraction)


def _save_upper_right_debug_figure(
    representative: np.ndarray,
    roi: np.ndarray,
    thresholded: np.ndarray,
    retained: np.ndarray,
    corridor: np.ndarray,
    final_mask: np.ndarray,
    info: dict[str, float | int | bool | str],
    config: ArtifactRemovalConfig,
    background_fraction: np.ndarray | None = None,
) -> None:
    if not config.debug or not config.debug_output_dir:
        return
    output_dir = Path(str(config.debug_output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(output_dir.glob("upper_right_white_line_debug_*.png")))
    output_path = output_dir / f"upper_right_white_line_debug_{index:04d}.png"
    cleaned = apply_artifact_mask(representative, final_mask, config.background_percentile)
    diff = cv2.absdiff(representative, cleaned)
    background_panel = (
        _background_gate(background_fraction, representative.shape, config)
        if background_fraction is not None
        else np.zeros_like(roi)
    )
    panels = [
        ("representative", representative),
        ("upper-right ROI", roi),
        ("thresholded bright", thresholded),
        ("retained dashes", retained),
        ("background gate", background_panel),
        ("fitted corridor", corridor),
        ("final mask", final_mask),
        ("cleaned", cleaned),
        ("difference", diff),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    for ax, (title, image) in zip(axes.ravel(), panels):
        if image.dtype == bool:
            ax.imshow(image, cmap="gray")
        else:
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig.suptitle(
        "upper_right_white_line | "
        + " | ".join(f"{key}={value}" for key, value in info.items() if key in {"detected", "reason", "area", "angle_degrees"})
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _line_coordinates_from_fit(
    shape: tuple[int, int],
    roi: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(roi)
    centered_x = xx.astype(np.float32) - float(x0)
    centered_y = yy.astype(np.float32) - float(y0)
    projections = centered_x * float(vx) + centered_y * float(vy)
    distances = np.abs(centered_x * float(vy) - centered_y * float(vx))
    return yy, xx, projections, distances


def _mask_corridor(
    shape: tuple[int, int],
    roi: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    p_min: float,
    p_max: float,
    half_width: float,
    projection_margin: float,
) -> np.ndarray:
    yy, xx, projections, distances = _line_coordinates_from_fit(shape, roi, x0, y0, vx, vy)
    corridor = np.zeros(shape, dtype=bool)
    corridor[yy, xx] = (
        (distances <= float(half_width))
        & (projections >= float(p_min) - float(projection_margin))
        & (projections <= float(p_max) + float(projection_margin))
    )
    return corridor


def _estimate_line_half_width(
    representative: np.ndarray,
    roi: np.ndarray,
    lenient_pixels: np.ndarray,
    x0: float,
    y0: float,
    vx: float,
    vy: float,
    p_min: float,
    p_max: float,
    config: ArtifactRemovalConfig,
) -> float:
    projection_margin = max(2.0, float(config.hough_max_line_gap_fraction) * float(min(representative.shape)))
    yy, xx, projections, distances = _line_coordinates_from_fit(representative.shape, roi, x0, y0, vx, vy)
    in_span = (
        (projections >= float(p_min) - projection_margin)
        & (projections <= float(p_max) + projection_margin)
    )
    line_pixels = np.zeros(representative.shape, dtype=bool)
    line_pixels[yy[in_span], xx[in_span]] = True
    support = line_pixels & lenient_pixels
    if not support.any():
        return float(config.upper_right_corridor_half_width_pixels)
    _, _, _, support_distances = _line_coordinates_from_fit(representative.shape, support, x0, y0, vx, vy)
    estimated = float(np.percentile(support_distances, 95)) + 1.0
    max_half_width = max(float(config.upper_right_corridor_half_width_pixels) + 1.0, 4.0)
    return float(np.clip(estimated, 2.0, max_half_width))


def _detect_upper_right_white_line(
    representative: np.ndarray,
    config: ArtifactRemovalConfig,
    background_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, float | int | bool | str] | None, float]:
    """Detect the short dashed white annotation in the upper-right EchoNet field."""

    height, width = representative.shape
    roi = _upper_right_roi((height, width), config)
    background_consistent = _background_gate(background_fraction, (height, width), config)
    local_background_consistent = cv2.dilate(
        background_consistent.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    roi_values = representative[roi]
    if roi_values.size == 0:
        return None, {"detected": False, "method": "upper_right_white_line", "reason": "empty_roi"}, -np.inf

    threshold = max(
        float(config.line_candidate_min_brightness),
        float(np.percentile(roi_values, float(config.bright_percentile))),
    )
    thresholded = (representative >= threshold) & roi

    # Drop isolated speckle while keeping small dash fragments.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresholded.astype(np.uint8), connectivity=8)
    retained = np.zeros((height, width), dtype=bool)
    retained_component_count = 0
    max_component_area = max(10, int(round(float(config.upper_right_max_component_area_fraction) * height * width)))
    max_extent = max(6, int(round(float(config.upper_right_max_component_extent_fraction) * min(height, width))))
    for label_idx in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label_idx]]
        if area < 2:
            continue
        if area > max_component_area:
            continue
        component_thickness = float(area) / max(float(max(w, h)), 1.0)
        slender_extent_allowed = (
            max(w, h) <= int(round(max_extent * 2.0))
            and component_thickness <= max(float(config.max_width_pixels), 3.0)
        )
        if max(w, h) > max_extent and not slender_extent_allowed:
            continue
        if w > max_extent * 0.75 and h > max_extent * 0.75 and not slender_extent_allowed:
            continue
        component = labels == label_idx
        component_background_fraction = float(local_background_consistent[component].mean()) if component.any() else 0.0
        if component_background_fraction < float(config.upper_right_min_background_fraction):
            continue
        retained |= component
        retained_component_count += 1

    if retained_component_count == 0:
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "no_retained_dash_components",
            "threshold": threshold,
            "retained_component_count": 0,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, np.zeros_like(roi), np.zeros_like(roi), info, config, background_fraction
        )
        return None, info, -np.inf

    points_yx = np.column_stack(np.nonzero(retained)).astype(np.float32)
    if len(points_yx) < 4:
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "too_few_retained_pixels",
            "threshold": threshold,
            "retained_component_count": retained_component_count,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, np.zeros_like(roi), np.zeros_like(roi), info, config, background_fraction
        )
        return None, info, -np.inf

    points_xy = points_yx[:, ::-1].astype(np.float32)
    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(points_xy, cv2.DIST_WELSCH, 0, 0.01, 0.01).ravel()]
    norm = max(float(np.hypot(vx, vy)), 1e-6)
    vx /= norm
    vy /= norm
    centered_x = points_xy[:, 0] - x0
    centered_y = points_xy[:, 1] - y0
    projections = centered_x * vx + centered_y * vy
    p_min = float(np.percentile(projections, 2))
    p_max = float(np.percentile(projections, 98))

    lenient_threshold = max(
        float(config.upper_right_lenient_min_brightness),
        float(threshold) * float(config.upper_right_lenient_threshold_fraction),
    )
    lenient_pixels = (representative >= lenient_threshold) & roi & local_background_consistent
    projection_margin = max(4.0, float(config.hough_max_line_gap_fraction) * float(min(height, width)))

    # Bridge faint anti-aliased dash fragments only after the strict detection
    # has established a plausible line. This expands along the confirmed line,
    # not across the whole image.
    broad_corridor = _mask_corridor(
        (height, width),
        roi,
        x0,
        y0,
        vx,
        vy,
        p_min,
        p_max,
        max(float(config.upper_right_corridor_half_width_pixels) + 2.0, 5.0),
        projection_margin * 1.15,
    )
    bridge_pixels = broad_corridor & lenient_pixels
    if int(bridge_pixels.sum()) >= max(4, int(retained.sum())):
        bridge_points_yx = np.column_stack(np.nonzero(bridge_pixels)).astype(np.float32)
        bridge_points_xy = bridge_points_yx[:, ::-1].astype(np.float32)
        vx2, vy2, x02, y02 = [float(v) for v in cv2.fitLine(bridge_points_xy, cv2.DIST_WELSCH, 0, 0.01, 0.01).ravel()]
        norm2 = max(float(np.hypot(vx2, vy2)), 1e-6)
        vx2 /= norm2
        vy2 /= norm2
        bridge_centered_x = bridge_points_xy[:, 0] - x02
        bridge_centered_y = bridge_points_xy[:, 1] - y02
        bridge_projections = bridge_centered_x * vx2 + bridge_centered_y * vy2
        vx, vy, x0, y0 = vx2, vy2, x02, y02
        p_min = float(np.percentile(bridge_projections, 1))
        p_max = float(np.percentile(bridge_projections, 99))

    fitted_length = float(p_max - p_min)
    min_length = max(6.0, float(config.upper_right_min_length_fraction) * float(min(height, width)))
    angle = abs(float(np.degrees(np.arctan2(vy, vx))))
    if angle > 90.0:
        angle = 180.0 - angle

    if fitted_length < min_length:
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "fitted_line_too_short",
            "threshold": threshold,
            "length": fitted_length,
            "angle_degrees": angle,
            "retained_component_count": retained_component_count,
            "lenient_threshold": lenient_threshold,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, np.zeros_like(roi), np.zeros_like(roi), info, config, background_fraction
        )
        return None, info, -np.inf
    if not (float(config.diagonal_angle_min_degrees) <= angle <= float(config.diagonal_angle_max_degrees)):
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "non_diagonal_angle",
            "threshold": threshold,
            "length": fitted_length,
            "angle_degrees": angle,
            "retained_component_count": retained_component_count,
            "lenient_threshold": lenient_threshold,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, np.zeros_like(roi), np.zeros_like(roi), info, config, background_fraction
        )
        return None, info, -np.inf

    estimated_half_width = _estimate_line_half_width(
        representative,
        roi,
        lenient_pixels & local_background_consistent,
        x0,
        y0,
        vx,
        vy,
        p_min,
        p_max,
        config,
    )
    corridor = _mask_corridor(
        (height, width),
        roi,
        x0,
        y0,
        vx,
        vy,
        p_min,
        p_max,
        estimated_half_width,
        projection_margin,
    )

    corridor_lenient_count = int((corridor & lenient_pixels).sum())
    corridor_area = int(corridor.sum())
    corridor_support_fraction = float(corridor_lenient_count) / max(float(corridor_area), 1.0)
    if corridor_support_fraction < float(config.upper_right_min_corridor_support_fraction):
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "insufficient_lenient_corridor_support",
            "threshold": threshold,
            "lenient_threshold": lenient_threshold,
            "length": fitted_length,
            "angle_degrees": angle,
            "retained_component_count": retained_component_count,
            "corridor_support_fraction": corridor_support_fraction,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, corridor, np.zeros_like(roi), info, config, background_fraction
        )
        return None, info, -np.inf

    final_mask = corridor & lenient_pixels & local_background_consistent & roi
    if int(config.upper_right_final_dilation_pixels) > 0:
        radius = int(config.upper_right_final_dilation_pixels)
        final_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        final_mask = cv2.dilate(final_mask.astype(np.uint8), final_kernel).astype(bool) & roi
        final_mask &= corridor & local_background_consistent
        final_mask |= retained & corridor & local_background_consistent

    area = int(final_mask.sum())
    if area <= 0:
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "empty_final_mask",
            "threshold": threshold,
            "length": fitted_length,
            "angle_degrees": angle,
            "retained_component_count": retained_component_count,
            "lenient_threshold": lenient_threshold,
            "estimated_half_width": estimated_half_width,
            "corridor_support_fraction": corridor_support_fraction,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, corridor, final_mask, info, config, background_fraction
        )
        return None, info, -np.inf
    removed_fraction = float(area) / float(height * width)
    final_background_fraction = float(local_background_consistent[final_mask].mean()) if final_mask.any() else 0.0
    if final_background_fraction < float(config.upper_right_min_final_background_fraction):
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "final_mask_not_background_consistent",
            "threshold": threshold,
            "lenient_threshold": lenient_threshold,
            "area": area,
            "removed_fraction": removed_fraction,
            "length": fitted_length,
            "angle_degrees": angle,
            "retained_component_count": retained_component_count,
            "estimated_half_width": estimated_half_width,
            "corridor_support_fraction": corridor_support_fraction,
            "final_background_fraction": final_background_fraction,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, corridor, final_mask, info, config, background_fraction
        )
        return None, info, -np.inf
    upper_right_max_removed_fraction = max(float(config.max_removed_fraction), 0.02)
    if removed_fraction > upper_right_max_removed_fraction:
        info = {
            "detected": False,
            "method": "upper_right_white_line",
            "reason": "removed_fraction_too_large",
            "threshold": threshold,
            "area": area,
            "removed_fraction": removed_fraction,
            "length": fitted_length,
            "angle_degrees": angle,
            "retained_component_count": retained_component_count,
            "lenient_threshold": lenient_threshold,
            "estimated_half_width": estimated_half_width,
            "corridor_support_fraction": corridor_support_fraction,
            "max_removed_fraction": upper_right_max_removed_fraction,
        }
        _save_upper_right_debug_figure(
            representative, roi, thresholded, retained, corridor, final_mask, info, config, background_fraction
        )
        return None, info, -np.inf

    strict_support = final_mask & thresholded
    mean_intensity = float(representative[strict_support].mean()) if strict_support.any() else 0.0
    info = {
        "detected": True,
        "method": "upper_right_white_line",
        "threshold": threshold,
        "lenient_threshold": lenient_threshold,
        "area": area,
        "removed_fraction": removed_fraction,
        "length": fitted_length,
        "angle_degrees": angle,
        "mean_intensity": mean_intensity,
        "retained_component_count": retained_component_count,
        "estimated_half_width": estimated_half_width,
        "corridor_support_fraction": corridor_support_fraction,
        "final_background_fraction": final_background_fraction,
    }
    _save_upper_right_debug_figure(representative, roi, thresholded, retained, corridor, final_mask, info, config, background_fraction)
    score = fitted_length * max(mean_intensity, 1.0) * max(retained_component_count, 1)
    return final_mask, info, score


def _detect_hough_line_mask(
    candidate: np.ndarray,
    roi: np.ndarray,
    representative: np.ndarray,
    threshold: float,
    config: ArtifactRemovalConfig,
    background_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, float | int | bool | str] | None, float]:
    height, width = candidate.shape
    background_consistent = _background_gate(background_fraction, (height, width), config)
    local_background_consistent = cv2.dilate(
        background_consistent.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    min_length = config.min_length_fraction * float(min(height, width))
    max_gap = max(4, int(round(config.hough_max_line_gap_fraction * min(height, width))))
    thickness = max(1, int(config.hough_mask_thickness_pixels))

    # Close tiny gaps so dashed annotation segments can still vote for one line.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1)
    lines = cv2.HoughLinesP(
        (closed * 255).astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=int(config.hough_threshold),
        minLineLength=max(4, int(round(min_length))),
        maxLineGap=max_gap,
    )
    if lines is None:
        return None, None, -np.inf

    bright_dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(thickness, int(config.dilation_pixels)) + 1, 2 * max(thickness, int(config.dilation_pixels)) + 1),
    )
    lenient_threshold = max(
        float(config.upper_right_lenient_min_brightness),
        float(threshold) * float(config.upper_right_lenient_threshold_fraction),
    )
    lenient_pixels = (representative >= lenient_threshold) & roi & local_background_consistent
    bright_support = cv2.dilate(lenient_pixels.astype(np.uint8), bright_dilate_kernel).astype(bool)

    best_mask: np.ndarray | None = None
    best_info: dict[str, float | int | bool | str] | None = None
    best_score = -np.inf
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_length:
            continue
        angle = _angle_from_endpoints(x1, y1, x2, y2)
        if not (config.diagonal_angle_min_degrees <= angle <= config.diagonal_angle_max_degrees):
            continue

        line_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.line(line_mask, (x1, y1), (x2, y2), color=1, thickness=thickness)
        line_mask = line_mask.astype(bool) & roi

        # Annotation marks are often broken into several bright dashes. Once a
        # plausible peripheral diagonal line is found, collect all nearby bright
        # support along that same line, instead of only the exact Hough segment.
        yy, xx = np.nonzero(roi)
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        norm = max(float(np.hypot(dx, dy)), 1.0)
        signed_dist = np.abs(dy * (xx - x1) - dx * (yy - y1)) / norm
        projection = ((xx - x1) * dx + (yy - y1) * dy) / (norm * norm)
        projection_margin = max(0.35, float(config.hough_max_line_gap_fraction) * 2.0)
        corridor = np.zeros((height, width), dtype=bool)
        corridor[yy, xx] = (
            (signed_dist <= max(thickness + int(config.dilation_pixels) + 2, 4))
            & (projection >= -projection_margin)
            & (projection <= 1.0 + projection_margin)
        )
        # Keep only pixels with lenient nearby bright evidence. Strict pixels
        # detected the line; lenient support covers anti-aliased/faint edges.
        supported_mask = (line_mask | corridor) & bright_support & lenient_pixels
        if config.dilation_pixels > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * int(config.dilation_pixels) + 1, 2 * int(config.dilation_pixels) + 1),
            )
            supported_mask = cv2.dilate(supported_mask.astype(np.uint8), kernel).astype(bool) & roi
            supported_mask &= (line_mask | corridor) & local_background_consistent
        area = int(supported_mask.sum())
        if area <= 0:
            continue
        removed_fraction = float(area) / float(height * width)
        if removed_fraction > float(config.max_removed_fraction):
            continue
        final_background_fraction = float(local_background_consistent[supported_mask].mean()) if supported_mask.any() else 0.0
        if final_background_fraction < float(config.upper_right_min_final_background_fraction):
            continue
        strict_support = supported_mask & candidate.astype(bool)
        bright_fraction = float(strict_support.sum()) / max(float(area), 1.0)
        mean_intensity = float(representative[strict_support].mean()) if strict_support.any() else 0.0
        if bright_fraction < 0.08:
            continue
        if mean_intensity < threshold * 0.75:
            continue
        score = length * max(bright_fraction, 0.05) * max(mean_intensity, 1.0)
        if score > best_score:
            best_score = score
            best_mask = supported_mask
            best_info = {
                "detected": True,
                "method": "hough_line",
                "threshold": threshold,
                "lenient_threshold": lenient_threshold,
                "area": area,
                "removed_fraction": removed_fraction,
                "length": length,
                "width": thickness,
                "angle_degrees": angle,
                "bright_fraction": bright_fraction,
                "mean_intensity": mean_intensity,
                "final_background_fraction": final_background_fraction,
            }

    return best_mask, best_info, best_score


def _detect_sparse_line_mask(
    candidate: np.ndarray,
    roi: np.ndarray,
    representative: np.ndarray,
    threshold: float,
    config: ArtifactRemovalConfig,
    background_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, float | int | bool | str] | None, float]:
    height, width = candidate.shape
    background_consistent = _background_gate(background_fraction, (height, width), config)
    local_background_consistent = cv2.dilate(
        background_consistent.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    candidate = candidate.astype(bool) & local_background_consistent
    points_yx = np.column_stack(np.nonzero(candidate & roi)).astype(np.float32)
    if len(points_yx) < 6:
        return None, None, -np.inf

    points_xy = points_yx[:, ::-1]
    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    cov = np.cov(centered.T)
    if not np.isfinite(cov).all():
        return None, None, -np.inf
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    direction = eigvecs[:, 0]
    normal = eigvecs[:, 1]
    projections = centered @ direction
    offsets = centered @ normal
    length = float(projections.max() - projections.min())
    robust_width = float(np.percentile(np.abs(offsets), 90) * 2.0)
    min_length = config.min_length_fraction * float(min(height, width))
    aspect = length / max(robust_width, 1.0)
    angle = abs(float(np.degrees(np.arctan2(direction[1], direction[0]))))
    if angle > 90.0:
        angle = 180.0 - angle

    if length < min_length:
        return None, None, -np.inf
    if robust_width > max(float(config.max_width_pixels) * 2.0, 3.0):
        return None, None, -np.inf
    if aspect < max(float(config.min_aspect_ratio) * 0.75, 2.0):
        return None, None, -np.inf
    if not (config.diagonal_angle_min_degrees <= angle <= config.diagonal_angle_max_degrees):
        return None, None, -np.inf

    p0 = mean + direction * projections.min()
    p1 = mean + direction * projections.max()
    line_mask = np.zeros((height, width), dtype=np.uint8)
    thickness = max(1, min(int(config.hough_mask_thickness_pixels), 2))
    cv2.line(
        line_mask,
        tuple(np.round(p0).astype(int)),
        tuple(np.round(p1).astype(int)),
        color=1,
        thickness=thickness,
    )
    support_radius = max(2, int(config.dilation_pixels) + 2)
    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * support_radius + 1, 2 * support_radius + 1),
    )
    bright_support = cv2.dilate(candidate.astype(np.uint8), support_kernel).astype(bool)

    yy, xx = np.nonzero(roi)
    dx = float(p1[0] - p0[0])
    dy = float(p1[1] - p0[1])
    norm = max(float(np.hypot(dx, dy)), 1.0)
    signed_dist = np.abs(dy * (xx - p0[0]) - dx * (yy - p0[1])) / norm
    projection = ((xx - p0[0]) * dx + (yy - p0[1]) * dy) / (norm * norm)
    projection_margin = max(0.35, float(config.hough_max_line_gap_fraction) * 2.0)
    corridor = np.zeros((height, width), dtype=bool)
    corridor[yy, xx] = (
        (signed_dist <= max(thickness + int(config.dilation_pixels) + 2, 4))
        & (projection >= -projection_margin)
        & (projection <= 1.0 + projection_margin)
    )
    supported_mask = (line_mask.astype(bool) | corridor) & roi & bright_support & local_background_consistent
    # For sparse/dashed lines, avoid a second broad dilation; otherwise the
    # fitted line band can become much larger than the actual annotation.
    supported_mask = supported_mask | (candidate.astype(bool) & roi & local_background_consistent)

    area = int(supported_mask.sum())
    if area <= 0:
        return None, None, -np.inf
    removed_fraction = float(area) / float(height * width)
    if removed_fraction > float(config.max_removed_fraction):
        return None, None, -np.inf
    final_background_fraction = float(local_background_consistent[supported_mask].mean()) if supported_mask.any() else 0.0
    if final_background_fraction < float(config.upper_right_min_final_background_fraction):
        return None, None, -np.inf
    mean_intensity = float(representative[supported_mask].mean()) if area else 0.0
    bright_fraction = float(candidate[supported_mask].mean()) if area else 0.0
    if bright_fraction < 0.05:
        return None, None, -np.inf
    score = length * aspect * max(bright_fraction, 0.05) * max(mean_intensity, 1.0)
    info = {
        "detected": True,
        "method": "sparse_line_fit",
        "threshold": threshold,
        "area": area,
        "removed_fraction": removed_fraction,
        "length": length,
        "width": robust_width,
        "aspect_ratio": aspect,
        "angle_degrees": angle,
        "bright_fraction": bright_fraction,
        "mean_intensity": mean_intensity,
        "final_background_fraction": final_background_fraction,
    }
    return supported_mask, info, score


def _detect_right_annotation_dash_mask(
    candidate: np.ndarray,
    roi: np.ndarray,
    representative: np.ndarray,
    threshold: float,
    config: ArtifactRemovalConfig,
    background_fraction: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, float | int | bool | str] | None, float]:
    """Detect short dashed diagonal annotation marks in the upper-right field."""

    height, width = candidate.shape
    background_consistent = _background_gate(background_fraction, (height, width), config)
    local_background_consistent = cv2.dilate(
        background_consistent.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    right_roi = np.zeros((height, width), dtype=bool)
    right_roi[: int(round(height * 0.68)), int(round(width * 0.58)) :] = True
    right_roi &= roi
    candidate = candidate.astype(bool) & right_roi & local_background_consistent
    if int(candidate.sum()) < 4:
        return None, None, -np.inf

    # A larger horizontal close links dashed text/marker strokes well enough
    # for Hough voting, but the final mask is still restricted to bright pixels.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
    closed = cv2.morphologyEx(candidate.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel, iterations=1)
    lines = cv2.HoughLinesP(
        (closed * 255).astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=max(2, int(config.hough_threshold) // 2),
        minLineLength=max(4, int(round(config.min_length_fraction * min(height, width) * 0.45))),
        maxLineGap=max(8, int(round(config.hough_max_line_gap_fraction * min(height, width) * 2.5))),
    )
    candidate_lines: list[tuple[int, int, int, int]] = []
    if lines is not None:
        candidate_lines.extend(tuple(int(v) for v in line) for line in lines[:, 0, :])

    # If Hough misses a broken/dashed annotation, fit one line to just the
    # upper-right bright fragments. This is intentionally narrower than the
    # general sparse detector so the ultrasound wedge cannot dominate the fit.
    points_yx = np.column_stack(np.nonzero(candidate)).astype(np.float32)
    if len(points_yx) >= 4:
        points_xy = points_yx[:, ::-1]
        mean = points_xy.mean(axis=0)
        centered = points_xy - mean
        cov = np.cov(centered.T)
        if np.isfinite(cov).all():
            eigvals, eigvecs = np.linalg.eigh(cov)
            direction = eigvecs[:, int(np.argmax(eigvals))]
            projections = centered @ direction
            p0 = mean + direction * projections.min()
            p1 = mean + direction * projections.max()
            candidate_lines.append((
                int(round(p0[0])),
                int(round(p0[1])),
                int(round(p1[0])),
                int(round(p1[1])),
            ))

    if not candidate_lines:
        return None, None, -np.inf

    support_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * max(1, int(config.dilation_pixels)) + 1, 2 * max(1, int(config.dilation_pixels)) + 1),
    )
    bright_support = cv2.dilate(candidate.astype(np.uint8), support_kernel).astype(bool)

    best_mask: np.ndarray | None = None
    best_info: dict[str, float | int | bool | str] | None = None
    best_score = -np.inf
    for line in candidate_lines:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = _angle_from_endpoints(x1, y1, x2, y2)
        if length < 4:
            continue
        if not (config.diagonal_angle_min_degrees <= angle <= config.diagonal_angle_max_degrees):
            continue

        yy, xx = np.nonzero(right_roi)
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        norm = max(float(np.hypot(dx, dy)), 1.0)
        signed_dist = np.abs(dy * (xx - x1) - dx * (yy - y1)) / norm
        projection = ((xx - x1) * dx + (yy - y1) * dy) / (norm * norm)
        corridor = np.zeros((height, width), dtype=bool)
        corridor[yy, xx] = (
            (signed_dist <= max(3, int(config.hough_mask_thickness_pixels) + 1))
            & (projection >= -0.75)
            & (projection <= 1.75)
        )
        mask = corridor & bright_support & right_roi & local_background_consistent

        # Include nearby bright pixels from the original candidate so separated
        # dashes along the same annotation line are removed together.
        if mask.any():
            mask = cv2.dilate(mask.astype(np.uint8), support_kernel).astype(bool) & right_roi
            mask &= cv2.dilate(candidate.astype(np.uint8), support_kernel).astype(bool) & local_background_consistent
            mask |= candidate & corridor & local_background_consistent

        area = int(mask.sum())
        if area <= 0:
            continue
        removed_fraction = float(area) / float(height * width)
        if removed_fraction > float(config.max_removed_fraction):
            continue
        final_background_fraction = float(local_background_consistent[mask].mean()) if mask.any() else 0.0
        if final_background_fraction < float(config.upper_right_min_final_background_fraction):
            continue
        bright_pixels = mask & candidate
        mean_intensity = float(representative[bright_pixels].mean()) if bright_pixels.any() else 0.0
        bright_fraction = float(candidate[mask].mean()) if area else 0.0
        if mean_intensity < max(185.0, threshold * 0.80):
            continue
        score = length * max(mean_intensity, 1.0) * max(bright_fraction, 0.05)
        if score > best_score:
            best_score = score
            best_mask = mask
            best_info = {
                "detected": True,
                "method": "right_annotation_dash",
                "threshold": threshold,
                "area": area,
                "removed_fraction": removed_fraction,
                "length": length,
                "width": max(3, int(config.hough_mask_thickness_pixels) + 1),
                "angle_degrees": angle,
                "bright_fraction": bright_fraction,
                "mean_intensity": mean_intensity,
                "final_background_fraction": final_background_fraction,
            }

    return best_mask, best_info, best_score


def detect_annotation_artifact_mask(
    frames: Sequence[np.ndarray],
    config: ArtifactRemovalConfig | None = None,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Detect one sequence/video-level annotation-line mask.

    Parameters
    ----------
    frames:
        Representative raw-resolution grayscale or RGB frames from one video or
        sequence. The returned mask is shared across all frames.
    config:
        Detection thresholds. Disable by setting ``enabled=False``.
    """

    config = config or ArtifactRemovalConfig()
    if not frames:
        raise ValueError("At least one representative frame is required.")

    gray_frames = [_as_gray_uint8(frame) for frame in frames]
    height, width = gray_frames[0].shape
    if not config.enabled:
        return np.zeros((height, width), dtype=bool), {"detected": False, "reason": "disabled"}

    stack = np.stack(gray_frames, axis=0)
    representative = np.percentile(stack, 90, axis=0).astype(np.uint8)
    background_fraction = _compute_background_fraction_map(gray_frames, config)
    roi = _peripheral_roi((height, width), config)
    roi_values = representative[roi]
    if roi_values.size == 0:
        return np.zeros((height, width), dtype=bool), {"detected": False, "reason": "empty_roi"}

    upper_mask, upper_info, _ = _detect_upper_right_white_line(representative, config, background_fraction)
    if upper_mask is not None and upper_info is not None:
        return upper_mask.astype(bool), upper_info

    # The dedicated upper-right detector above uses the relaxed thresholds.
    # Keep legacy fallback detectors stricter so they do not erase ultrasound
    # sector boundaries or anatomy-like bright structures.
    threshold = max(205.0, float(np.percentile(roi_values, max(float(config.bright_percentile), 99.5))))
    line_threshold = max(
        145.0,
        min(float(threshold), 205.0 * float(config.line_candidate_threshold_fraction)),
    )
    candidate = (representative >= threshold) & roi
    line_candidate = (representative >= line_threshold) & roi
    candidate = candidate.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)

    min_length = config.min_length_fraction * float(min(height, width))
    best_score = -np.inf
    best_mask: np.ndarray | None = None
    best_info: dict[str, float | int | bool | str] = (
        upper_info
        if upper_info is not None
        else {"detected": False, "reason": "no_candidate"}
    )

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area <= 0:
            continue
        component = labels == label_idx
        points_yx = np.column_stack(np.nonzero(component)).astype(np.float32)
        if len(points_yx) < 2:
            continue
        points_xy = points_yx[:, ::-1]
        _, size, angle = cv2.minAreaRect(points_xy)
        rect_long = float(max(size))
        rect_short = float(max(min(size), 1.0))
        aspect = rect_long / rect_short
        diagonal_angle = abs(float(angle))
        if diagonal_angle > 90.0:
            diagonal_angle = 180.0 - diagonal_angle
        if diagonal_angle > 45.0:
            diagonal_angle = 90.0 - diagonal_angle
        # Treat both slash directions as diagonal-ish relative to horizontal.
        diagonal_angle = abs(diagonal_angle)
        fill_fraction = float(area) / max(rect_long * rect_short, 1.0)
        removed_fraction = float(area) / float(height * width)

        checks = [
            rect_long >= min_length,
            rect_short <= float(config.max_width_pixels),
            aspect >= float(config.min_aspect_ratio),
            fill_fraction >= float(config.min_line_fill_fraction),
            removed_fraction <= float(config.max_removed_fraction),
            config.diagonal_angle_min_degrees <= diagonal_angle <= config.diagonal_angle_max_degrees,
        ]
        if not all(checks):
            continue
        score = rect_long * aspect * fill_fraction
        if score > best_score:
            best_score = score
            best_mask = component
            best_info = {
                "detected": True,
                "method": "connected_component",
                "threshold": threshold,
                "area": int(area),
                "removed_fraction": removed_fraction,
                "length": rect_long,
                "width": rect_short,
                "aspect_ratio": aspect,
                "fill_fraction": fill_fraction,
                "angle_degrees": diagonal_angle,
            }

    hough_mask, hough_info, hough_score = _detect_hough_line_mask(
        candidate=line_candidate.astype(bool),
        roi=roi,
        representative=representative,
        threshold=line_threshold,
        config=config,
        background_fraction=background_fraction,
    )
    dash_mask, dash_info, dash_score = _detect_right_annotation_dash_mask(
        candidate=line_candidate.astype(bool),
        roi=roi,
        representative=representative,
        threshold=line_threshold,
        config=config,
        background_fraction=background_fraction,
    )
    sparse_mask, sparse_info, sparse_score = _detect_sparse_line_mask(
        candidate=line_candidate.astype(bool),
        roi=roi,
        representative=representative,
        threshold=line_threshold,
        config=config,
        background_fraction=background_fraction,
    )
    connected_length = float(best_info.get("length", 0.0)) if best_info else 0.0
    hough_length = float(hough_info.get("length", 0.0)) if hough_info else 0.0
    hough_is_better = hough_score > best_score or hough_length > connected_length * 1.25
    if hough_mask is not None and hough_info is not None and hough_is_better:
        best_mask = hough_mask
        best_info = hough_info
        best_score = hough_score
    current_length = float(best_info.get("length", 0.0)) if best_info else 0.0
    dash_length = float(dash_info.get("length", 0.0)) if dash_info else 0.0
    dash_is_better = dash_score > best_score or dash_length > current_length * 1.10
    if dash_mask is not None and dash_info is not None and dash_is_better:
        best_mask = dash_mask
        best_info = dash_info
        best_score = dash_score
    current_length = float(best_info.get("length", 0.0)) if best_info else 0.0
    sparse_length = float(sparse_info.get("length", 0.0)) if sparse_info else 0.0
    sparse_is_better = sparse_score > best_score or sparse_length > current_length * 1.25
    if sparse_mask is not None and sparse_info is not None and sparse_is_better:
        best_mask = sparse_mask
        best_info = sparse_info
        best_score = sparse_score

    if best_mask is None:
        return np.zeros((height, width), dtype=bool), best_info

    background_consistent = _background_gate(background_fraction, (height, width), config)
    local_background_consistent = cv2.dilate(
        background_consistent.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ).astype(bool)
    best_mask = best_mask.astype(bool) & local_background_consistent
    if not best_mask.any():
        best_info = dict(best_info)
        best_info.update({"detected": False, "reason": "selected_mask_not_background_consistent"})
        return np.zeros((height, width), dtype=bool), best_info
    final_background_fraction = float(local_background_consistent[best_mask].mean())
    if final_background_fraction < float(config.upper_right_min_final_background_fraction):
        best_info = dict(best_info)
        best_info.update(
            {
                "detected": False,
                "reason": "selected_mask_failed_background_consistency",
                "final_background_fraction": final_background_fraction,
            }
        )
        return np.zeros((height, width), dtype=bool), best_info
    best_info = dict(best_info)
    best_info["final_background_fraction"] = final_background_fraction

    if config.dilation_pixels > 0 and str(best_info.get("method", "")) not in {
        "hough_line",
        "sparse_line_fit",
        "right_annotation_dash",
    }:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * int(config.dilation_pixels) + 1, 2 * int(config.dilation_pixels) + 1),
        )
        best_mask = cv2.dilate(best_mask.astype(np.uint8), kernel).astype(bool)
        best_mask &= local_background_consistent
        best_info["dilated_removed_fraction"] = float(best_mask.mean())

    return best_mask.astype(bool), best_info


def apply_artifact_mask(
    frame: np.ndarray,
    mask: np.ndarray,
    background_percentile: float = 5.0,
) -> np.ndarray:
    """Replace artifact pixels using conservative OpenCV Telea inpainting."""

    gray = _as_gray_uint8(frame)
    if not mask.any():
        return gray.copy()
    inpaint_mask = (mask.astype(np.uint8) * 255)
    return cv2.inpaint(gray, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def clean_frames_with_shared_artifact_mask(
    frames: Sequence[np.ndarray],
    representative_frames: Sequence[np.ndarray] | None = None,
    config: ArtifactRemovalConfig | None = None,
) -> tuple[list[np.ndarray], np.ndarray, dict[str, float | int | bool | str]]:
    """Detect one mask and apply it consistently to all frames."""

    config = config or ArtifactRemovalConfig()
    reps = list(representative_frames) if representative_frames is not None else list(frames)
    mask, info = detect_annotation_artifact_mask(reps, config)
    cleaned = [apply_artifact_mask(frame, mask, config.background_percentile) for frame in frames]
    return cleaned, mask, info


def save_artifact_diagnostic_figure(
    original_frame: np.ndarray,
    cleaned_frame: np.ndarray,
    mask: np.ndarray,
    output_path: str | Path,
    title: str | None = None,
) -> None:
    """Save original/mask/cleaned/difference panels for QA."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original = _as_gray_uint8(original_frame)
    cleaned = _as_gray_uint8(cleaned_frame)
    diff = cv2.absdiff(original, cleaned)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("original")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("artifact mask")
    axes[2].imshow(cleaned, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("cleaned")
    axes[3].imshow(diff, cmap="magma")
    axes[3].set_title("difference")
    for ax in axes:
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
