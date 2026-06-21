from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GradCAMResult:
    heatmaps: np.ndarray
    logits: torch.Tensor
    layer_name: str
    n_layer_calls: int


def get_module_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if layer_name not in modules:
        available = ", ".join(modules.keys())
        raise KeyError(f"Layer {layer_name!r} was not found. Available layers: {available}")
    return modules[layer_name]


def find_last_conv2d_name(model: nn.Module) -> str:
    last_name = ""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_name = name
    if not last_name:
        raise ValueError("No nn.Conv2d layer found for Grad-CAM.")
    return last_name


def segmentation_target(logits: torch.Tensor, target_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Use logit evidence inside the target mask as the scalar CAM objective."""
    if target_mask is None:
        return logits.mean()
    mask = (target_mask > 0.5).to(dtype=logits.dtype, device=logits.device)
    if float(mask.sum().detach().cpu()) <= 0:
        return logits.mean()
    return (logits * mask).sum() / mask.sum().clamp_min(1.0)


def _tensor_from_layer_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Grad-CAM hook expected a tensor-like layer output, got {type(output)!r}")


class AppendGradCAMHook:
    """Forward hook that appends one activation/gradient entry per layer call."""

    def __init__(self, module: nn.Module) -> None:
        self.entries: list[dict[str, torch.Tensor | None]] = []
        self.handle = module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        del module, inputs
        activation = _tensor_from_layer_output(output)
        entry: dict[str, torch.Tensor | None] = {"activation": activation, "gradient": None}
        self.entries.append(entry)

        def save_gradient(gradient: torch.Tensor) -> None:
            entry["gradient"] = gradient

        activation.register_hook(save_gradient)

    def close(self) -> None:
        self.handle.remove()


def _cam_from_activation_gradient(
    activation: torch.Tensor,
    gradient: torch.Tensor,
    output_size: tuple[int, int],
) -> np.ndarray:
    weights = gradient.mean(dim=(-2, -1), keepdim=True)
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=output_size, mode="bilinear", align_corners=False)
    cam_np = cam.detach().cpu().numpy()[:, 0]
    normalized: list[np.ndarray] = []
    for item in cam_np:
        item = item.astype(np.float32)
        item -= float(item.min())
        max_value = float(item.max())
        if max_value > 0:
            item /= max_value
        normalized.append(item)
    return np.stack(normalized, axis=0)


def convlstm_gradcam(
    model: nn.Module,
    sequence: torch.Tensor,
    target_mask: torch.Tensor,
    layer_name: str,
) -> GradCAMResult:
    """Compute per-timestep Grad-CAM for a ConvLSTM U-Net target layer.

    Layers called once per frame, such as bottleneck_encoder and temporal_bottleneck,
    return one heatmap per temporal input. Decoder layers are called once after
    temporal fusion; their fused CAM is repeated across sequence positions so the
    saved output grid remains shape [T, H, W].
    """
    if sequence.ndim != 5 or sequence.shape[0] != 1:
        raise ValueError("ConvLSTM Grad-CAM expects sequence shape [1, T, C, H, W].")

    model.zero_grad(set_to_none=True)
    module = get_module_by_name(model, layer_name)
    hook = AppendGradCAMHook(module)
    try:
        logits = model(sequence)
        target = segmentation_target(logits, target_mask)
        target.backward()

        output_size = tuple(sequence.shape[-2:])
        heatmap_chunks: list[np.ndarray] = []
        for entry in hook.entries:
            activation = entry["activation"]
            gradient = entry["gradient"]
            if not isinstance(activation, torch.Tensor) or not isinstance(gradient, torch.Tensor):
                continue
            heatmap_chunks.append(_cam_from_activation_gradient(activation, gradient, output_size))
    finally:
        hook.close()

    if not heatmap_chunks:
        raise RuntimeError(f"No Grad-CAM activations/gradients were captured for layer {layer_name}.")

    heatmaps = np.concatenate(heatmap_chunks, axis=0)
    sequence_length = int(sequence.shape[1])
    if heatmaps.shape[0] == 1:
        heatmaps = np.repeat(heatmaps, sequence_length, axis=0)
    elif heatmaps.shape[0] != sequence_length:
        heatmaps = heatmaps[:sequence_length]
        if heatmaps.shape[0] < sequence_length:
            heatmaps = np.pad(
                heatmaps,
                pad_width=((0, sequence_length - heatmaps.shape[0]), (0, 0), (0, 0)),
                mode="edge",
            )

    return GradCAMResult(
        heatmaps=heatmaps.astype(np.float32),
        logits=logits.detach(),
        layer_name=layer_name,
        n_layer_calls=len(hook.entries),
    )


def unet_framewise_gradcam(
    model: nn.Module,
    sequence: torch.Tensor,
    target_mask: torch.Tensor,
    layer_name: str,
) -> GradCAMResult:
    """Run a 2D U-Net independently on each frame from a temporal sequence."""
    if sequence.ndim != 5 or sequence.shape[0] != 1:
        raise ValueError("2D U-Net framewise Grad-CAM expects sequence shape [1, T, C, H, W].")

    module = get_module_by_name(model, layer_name)
    heatmaps: list[np.ndarray] = []
    logits_all: list[torch.Tensor] = []
    layer_calls = 0

    for time_idx in range(sequence.shape[1]):
        model.zero_grad(set_to_none=True)
        hook = AppendGradCAMHook(module)
        try:
            frame = sequence[:, time_idx]
            logits = model(frame)
            target = segmentation_target(logits, target_mask)
            target.backward()
            output_size = tuple(frame.shape[-2:])
            frame_heatmaps: list[np.ndarray] = []
            for entry in hook.entries:
                activation = entry["activation"]
                gradient = entry["gradient"]
                if isinstance(activation, torch.Tensor) and isinstance(gradient, torch.Tensor):
                    frame_heatmaps.append(_cam_from_activation_gradient(activation, gradient, output_size))
            if not frame_heatmaps:
                raise RuntimeError(f"No Grad-CAM entries captured for U-Net layer {layer_name}.")
            heatmaps.append(frame_heatmaps[-1][0])
            logits_all.append(logits.detach())
            layer_calls += len(hook.entries)
        finally:
            hook.close()

    return GradCAMResult(
        heatmaps=np.stack(heatmaps, axis=0).astype(np.float32),
        logits=torch.stack(logits_all, dim=1),
        layer_name=layer_name,
        n_layer_calls=layer_calls,
    )
