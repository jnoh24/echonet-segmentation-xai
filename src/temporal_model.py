from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell that preserves spatial feature geometry."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = torch.chunk(
            self.gates(torch.cat([x, hidden], dim=1)),
            chunks=4,
            dim=1,
        )
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)

        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell

    def init_state(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1])
        return x.new_zeros(shape), x.new_zeros(shape)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class ConvLSTMUNet(nn.Module):
    """U-Net encoder/decoder with temporal fusion at the deepest feature level."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, int, int, int] = (16, 32, 64, 128),
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = channels
        self.encoder1 = DoubleConv(in_channels, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.bottleneck_encoder = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.temporal_bottleneck = ConvLSTMCell(c4, c4)
        self.decoder3 = UpBlock(c4, c3, c3)
        self.decoder2 = UpBlock(c3, c2, c2)
        self.decoder1 = UpBlock(c2, c1, c1)
        self.output = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _encode_frame(
        self,
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip1 = self.encoder1(frame)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        bottleneck = self.bottleneck_encoder(self.pool(skip3))
        return skip1, skip2, skip3, bottleneck

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError("Expected sequence shape [B, T, C, H, W].")

        center_idx = sequence.shape[1] // 2
        center_skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        state: tuple[torch.Tensor, torch.Tensor] | None = None

        for time_idx in range(sequence.shape[1]):
            skip1, skip2, skip3, encoded = self._encode_frame(sequence[:, time_idx])
            if time_idx == center_idx:
                center_skips = (skip1, skip2, skip3)
            if state is None:
                state = self.temporal_bottleneck.init_state(encoded)
            state = self.temporal_bottleneck(encoded, state)

        if center_skips is None or state is None:
            raise RuntimeError("Temporal sequence did not contain a center frame.")

        hidden, _ = state
        skip1, skip2, skip3 = center_skips
        x = self.decoder3(hidden, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.output(x)


def build_convlstm_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple[int, int, int, int] = (16, 32, 64, 128),
) -> ConvLSTMUNet:
    return ConvLSTMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
    )
