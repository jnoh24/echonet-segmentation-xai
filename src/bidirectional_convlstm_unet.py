from __future__ import annotations

import torch
from torch import nn

from .temporal_model import ConvLSTMCell, DoubleConv, UpBlock


class BidirectionalConvLSTMUNet(nn.Module):
    """Target-aligned bidirectional ConvLSTM U-Net for frame segmentation.

    The shared U-Net encoder is applied once to each frame in an input sequence
    shaped ``[B, T, C, H, W]``. Forward and backward ConvLSTM branches are run
    only until they have consumed the target frame, where
    ``target_idx = num_frames_before``. The target-aligned hidden states are
    concatenated, projected back to the bottleneck channel count, and decoded
    with skip connections from the target frame only.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, int, int, int] = (16, 32, 64, 128),
        num_frames_before: int = 12,
        num_frames_after: int = 12,
    ) -> None:
        super().__init__()
        if len(channels) != 4:
            raise ValueError("channels must contain four encoder channel counts.")
        if num_frames_before < 0 or num_frames_after < 0:
            raise ValueError("num_frames_before and num_frames_after must be non-negative.")

        self.num_frames_before = int(num_frames_before)
        self.num_frames_after = int(num_frames_after)
        self.target_idx = self.num_frames_before
        self.expected_sequence_length = self.num_frames_before + 1 + self.num_frames_after

        c1, c2, c3, c4 = channels
        self.encoder1 = DoubleConv(in_channels, c1)
        self.encoder2 = DoubleConv(c1, c2)
        self.encoder3 = DoubleConv(c2, c3)
        self.bottleneck_encoder = DoubleConv(c3, c4)
        self.pool = nn.MaxPool2d(2)

        self.forward_temporal_bottleneck = ConvLSTMCell(c4, c4)
        self.backward_temporal_bottleneck = ConvLSTMCell(c4, c4)
        self.bidirectional_fusion = nn.Sequential(
            nn.Conv2d(2 * c4, c4, kernel_size=1, bias=False),
            nn.BatchNorm2d(c4),
            nn.ReLU(inplace=True),
        )

        self.decoder3 = UpBlock(c4, c3, c3)
        self.decoder2 = UpBlock(c3, c2, c2)
        self.decoder1 = UpBlock(c2, c1, c1)
        self.output = nn.Conv2d(c1, out_channels, kernel_size=1)

    def _validate_sequence(self, sequence: torch.Tensor) -> None:
        if sequence.ndim != 5:
            raise ValueError("Expected sequence shape [B, T, C, H, W].")
        if sequence.shape[1] != self.expected_sequence_length:
            raise ValueError(
                "Expected sequence length "
                f"{self.expected_sequence_length} "
                f"({self.num_frames_before} before + target + {self.num_frames_after} after), "
                f"got {sequence.shape[1]}."
            )

    def _encode_frame(
        self,
        frame: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        skip1 = self.encoder1(frame)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        bottleneck = self.bottleneck_encoder(self.pool(skip3))
        return skip1, skip2, skip3, bottleneck

    def _encode_sequence(
        self,
        sequence: torch.Tensor,
    ) -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        bottlenecks: list[torch.Tensor] = []
        target_skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None

        for time_idx in range(sequence.shape[1]):
            skip1, skip2, skip3, bottleneck = self._encode_frame(sequence[:, time_idx])
            bottlenecks.append(bottleneck)
            if time_idx == self.target_idx:
                target_skips = (skip1, skip2, skip3)

        if target_skips is None:
            raise RuntimeError("Temporal sequence did not contain the configured target frame.")
        return bottlenecks, target_skips

    @staticmethod
    def _run_temporal_branch(
        cell: ConvLSTMCell,
        features: list[torch.Tensor],
        indices: range,
    ) -> torch.Tensor:
        state: tuple[torch.Tensor, torch.Tensor] | None = None
        for time_idx in indices:
            encoded = features[time_idx]
            if state is None:
                state = cell.init_state(encoded)
            state = cell(encoded, state)
        if state is None:
            raise RuntimeError("Temporal branch did not process any frames.")
        hidden, _ = state
        return hidden

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        self._validate_sequence(sequence)

        bottlenecks, target_skips = self._encode_sequence(sequence)
        forward_hidden = self._run_temporal_branch(
            self.forward_temporal_bottleneck,
            bottlenecks,
            range(0, self.target_idx + 1),
        )
        backward_hidden = self._run_temporal_branch(
            self.backward_temporal_bottleneck,
            bottlenecks,
            range(self.expected_sequence_length - 1, self.target_idx - 1, -1),
        )

        hidden = self.bidirectional_fusion(torch.cat([forward_hidden, backward_hidden], dim=1))
        skip1, skip2, skip3 = target_skips
        x = self.decoder3(hidden, skip3)
        x = self.decoder2(x, skip2)
        x = self.decoder1(x, skip1)
        return self.output(x)


def build_bidirectional_convlstm_unet(
    in_channels: int = 1,
    out_channels: int = 1,
    channels: tuple[int, int, int, int] = (16, 32, 64, 128),
    num_frames_before: int = 12,
    num_frames_after: int = 12,
) -> BidirectionalConvLSTMUNet:
    """Build a target-aligned bidirectional ConvLSTM U-Net."""
    return BidirectionalConvLSTMUNet(
        in_channels=in_channels,
        out_channels=out_channels,
        channels=channels,
        num_frames_before=num_frames_before,
        num_frames_after=num_frames_after,
    )
