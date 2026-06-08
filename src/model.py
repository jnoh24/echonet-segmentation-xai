from __future__ import annotations

import torch
from torch import nn

try:
    from monai.networks.nets import UNet
except ImportError:
    UNet = None


def build_unet(
    spatial_dims: int = 2,
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple[int, ...] = (16, 32, 64, 128, 256),
    strides: tuple[int, ...] = (2, 2, 2, 2),
    num_res_units: int = 2,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a MONAI 2D U-Net for binary LV segmentation.

    The model returns logits. Use sigmoid only for metrics, visualization, and
    inference thresholding.
    """
    if UNet is None:
        raise ImportError("MONAI is required to build the U-Net. Install monai first.")

    return UNet(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        dropout=dropout,
    )


def predict_mask(
    model: nn.Module,
    image: torch.Tensor,
    threshold: float = 0.5,
) -> torch.Tensor:
    """Return a binary mask prediction for a batched image tensor."""
    model.eval()
    with torch.no_grad():
        logits = model(image)
        probs = torch.sigmoid(logits)
        return (probs >= threshold).float()
