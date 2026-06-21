from __future__ import annotations

from torch import nn

try:
    from monai.networks.nets import UNETR
except ImportError:
    UNETR = None


def build_unetr(
    in_channels: int = 1,
    out_channels: int = 1,
    image_size: tuple[int, int] = (112, 112),
    feature_size: int = 16,
    hidden_size: int = 384,
    mlp_dim: int = 1536,
    num_heads: int = 6,
    dropout_rate: float = 0.0,
) -> nn.Module:
    """Build a MONAI 2D UNETR that returns binary-segmentation logits."""
    if UNETR is None:
        raise ImportError("MONAI is required to build UNETR. Install monai first.")
    if hidden_size % num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads.")
    if any(size % 16 != 0 for size in image_size):
        raise ValueError("Each image dimension must be divisible by UNETR's patch size of 16.")

    return UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=image_size,
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        proj_type="conv",
        norm_name="instance",
        conv_block=True,
        res_block=True,
        dropout_rate=dropout_rate,
        spatial_dims=2,
    )
