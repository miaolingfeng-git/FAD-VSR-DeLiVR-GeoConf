"""Frequency and motion guided video super-resolution network.

The implementation keeps the useful ideas from HFGAT and V3 in one PyTorch
model.  It predicts a high-resolution centre frame from a short LR clip.  The
flow estimator is deliberately lightweight so that the project can run
without the CUDA/JAX extension used by the original V3 implementation.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDownsample(nn.Module):
    """Fixed 2-D Haar decomposition performed independently per channel."""

    def __init__(self, channels: int):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer("weight", filters.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        h, w = x.shape[-2:]
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = F.conv2d(x, self.weight, stride=2, groups=self.channels)
        return y, (h, w)


class HaarUpsample(nn.Module):
    """Inverse Haar reconstruction for tensors produced by HaarDownsample."""

    def __init__(self, channels: int):
        super().__init__()
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1) / 2.0
        self.register_buffer("weight", filters.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        y = F.conv_transpose2d(x, self.weight, stride=2, groups=self.channels)
        return y[..., : output_size[0], : output_size[1]]


def _window_partition(x: torch.Tensor, window_size: int):
    b, c, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h or pad_w:
        # Replicate also works for very small tiles where reflect padding is
        # undefined (for example a 1-pixel feature map at the deepest scale).
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    hp, wp = x.shape[-2:]
    windows = x.view(b, c, hp // window_size, window_size, wp // window_size, window_size)
    windows = windows.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, c)
    return windows, hp, wp


def _window_reverse(windows: torch.Tensor, window_size: int, b: int, h: int, w: int):
    n_h, n_w = h // window_size, w // window_size
    c = windows.shape[-1]
    x = windows.view(b, n_h, n_w, window_size, window_size, c)
    return x.permute(0, 5, 1, 3, 2, 4).reshape(b, c, h, w)


class WindowTransformer(nn.Module):
    """Local self-attention block used at the frequency feature scale."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        windows, hp, wp = _window_partition(x, self.window_size)
        q = self.norm1(windows)
        attn, _ = self.attn(q, q, q, need_weights=False)
        windows = windows + attn
        windows = windows + self.mlp(self.norm2(windows))
        out = _window_reverse(windows, self.window_size, b, hp, wp)
        return out[..., :h, :w]


class SKFF(nn.Module):
    """Selective kernel feature fusion from HFGAT."""

    def __init__(self, channels: int, height: int, reduction: int = 8):
        super().__init__()
        self.height = height
        d = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Sequential(nn.Conv2d(channels, d, 1), nn.PReLU())
        self.fcs = nn.ModuleList([nn.Conv2d(d, channels, 1) for _ in range(height)])

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.height:
            raise ValueError(f"expected {self.height} features, got {len(features)}")
        stacked = torch.stack(features, dim=1)  # B, height, C, H, W
        summary = stacked.sum(dim=1)
        z = self.reduce(self.pool(summary))
        weights = torch.stack([fc(z) for fc in self.fcs], dim=1)
        weights = torch.softmax(weights, dim=1)
        return (stacked * weights).sum(dim=1)


class FrequencyBlock(nn.Module):
    """Haar frequency processing followed by local Transformer refinement."""

    def __init__(self, dim: int, heads: int = 4, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.down = HaarDownsample(dim)
        self.up = HaarUpsample(dim)
        self.high_gate = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim * 3, 1),
            nn.Sigmoid(),
        )
        self.in_proj = nn.Conv2d(dim * 4, dim, 1)
        self.transformer = WindowTransformer(dim, heads, window_size)
        self.out_proj = nn.Conv2d(dim, dim * 4, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.high_proj = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        subbands, size = self.down(x)
        # HaarDownsample stores four bands per input channel: LL, LH, HL, HH.
        # Reshape before separating low and high frequencies so channel groups
        # are not mixed when dim > 1.
        b, _, hs, ws = subbands.shape
        grouped = subbands.view(b, self.dim, 4, hs, ws)
        low = grouped[:, :, 0]
        high = grouped[:, :, 1:].reshape(b, self.dim * 3, hs, ws)
        high = high * self.high_gate(high)
        y = self.in_proj(torch.cat([low, high], dim=1))
        y = self.transformer(y)
        y = self.out_proj(y) + subbands
        y = self.up(y, size)
        out = x + self.refine(y)
        high_guidance = self.high_proj(high)
        high_guidance = F.interpolate(high_guidance, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out, high_guidance


class FrequencyBackbone(nn.Module):
    """A compact U-shaped frequency backbone applied to every input frame."""

    def __init__(self, dim: int, heads: int = 4, window_size: int = 8):
        super().__init__()
        self.block1 = FrequencyBlock(dim, heads, window_size)
        self.down = nn.Conv2d(dim, dim, 3, stride=2, padding=1)
        self.block2 = FrequencyBlock(dim, heads, window_size)
        self.up = nn.Sequential(nn.Conv2d(dim, dim * 4, 3, padding=1), nn.PixelShuffle(2))
        self.skip_fuse = SKFF(dim, height=2)
        self.block3 = FrequencyBlock(dim, heads, window_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, w = x.shape[-2:]
        x1, hf1 = self.block1(x)
        x2 = self.down(x1)
        x2, hf2 = self.block2(x2)
        x2 = self.up(x2)[..., :h, :w]
        fused = self.skip_fuse([x1, x2])
        out, hf3 = self.block3(fused)
        hf = hf1 + F.interpolate(hf2, size=(h, w), mode="bilinear", align_corners=False) + hf3
        return out, hf


def flow_warp(x: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Warp BCHW features with pixel displacement flow (dx, dy)."""

    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, -1, -1, -1) + flow.permute(0, 2, 3, 1)
    grid_x = 2.0 * grid[..., 0] / max(w - 1, 1) - 1.0
    grid_y = 2.0 * grid[..., 1] / max(h - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


class TinyFlowEstimator(nn.Module):
    """Trainable flow estimator used as a portable RAFT substitute."""

    def __init__(self, base: int = 32, max_flow: float = 24.0):
        super().__init__()
        self.max_flow = max_flow
        self.body = nn.Sequential(
            nn.Conv2d(6, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(base, base, 3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Conv2d(base, 2, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.max_flow * torch.tanh(self.head(self.body(torch.cat([source, target], dim=1))))


class FrequencyGuidedAlign(nn.Module):
    """Flow warp plus frequency-guided deformable residual sampling."""

    def __init__(self, dim: int, max_offset: float = 8.0):
        super().__init__()
        self.flow = TinyFlowEstimator()
        self.max_offset = max_offset
        self.offset = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 2, 3, padding=1),
        )
        self.confidence = nn.Sequential(
            nn.Conv2d(dim * 2 + 2, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 3, padding=1),
            nn.Sigmoid(),
        )
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)

    def forward(
        self,
        center_feat: torch.Tensor,
        neighbor_feat: torch.Tensor,
        center_high: torch.Tensor,
        neighbor_high: torch.Tensor,
        center_rgb: torch.Tensor,
        neighbor_rgb: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        flow = self.flow(neighbor_rgb, center_rgb)
        warped = flow_warp(neighbor_feat, flow)
        warped_high = flow_warp(neighbor_high, flow)
        delta = self.max_offset * torch.tanh(
            self.offset(torch.cat([center_feat, warped, center_high, warped_high], dim=1))
        )
        aligned = flow_warp(neighbor_feat, flow + delta)
        confidence = self.confidence(torch.cat([center_feat, aligned, flow], dim=1))
        attention = torch.sigmoid((self.q(center_feat) * self.k(aligned)).mean(dim=1, keepdim=True))
        aligned = aligned + attention * self.v(aligned)
        return aligned, confidence, flow + delta


class TemporalPropagator(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.update = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.norm = nn.GroupNorm(8, dim)

    def forward(self, sequence: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        if reverse:
            sequence = sequence.flip(1)
        state = torch.zeros_like(sequence[:, 0])
        states = []
        for t in range(sequence.shape[1]):
            state = state + self.update(torch.cat([sequence[:, t], state], dim=1))
            state = self.norm(state)
            states.append(state)
        result = torch.stack(states, dim=1)
        return result.flip(1) if reverse else result


class BidirectionalPropagation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.forward_prop = TemporalPropagator(dim)
        self.backward_prop = TemporalPropagator(dim)

    def forward(self, sequence: torch.Tensor, center: int) -> Tuple[torch.Tensor, torch.Tensor]:
        forward_states = self.forward_prop(sequence, reverse=False)
        backward_states = self.backward_prop(sequence, reverse=True)
        return forward_states[:, center], backward_states[:, center]


class ReconstructionHead(nn.Module):
    def __init__(self, dim: int, scale: int):
        super().__init__()
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        layers: List[nn.Module] = []
        for _ in range(int(math.log2(scale))):
            layers.extend([nn.Conv2d(dim, dim * 4, 3, padding=1), nn.PixelShuffle(2), nn.PReLU(dim)])
        layers.append(nn.Conv2d(dim, 3, 3, padding=1))
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class FADVSR(nn.Module):
    """Frequency-Aware Deformable Video Super-Resolution network."""

    def __init__(
        self,
        in_channels: int = 3,
        num_frames: int = 7,
        dim: int = 64,
        scale: int = 4,
        heads: int = 4,
        window_size: int = 8,
    ):
        super().__init__()
        if num_frames % 2 != 1:
            raise ValueError("num_frames must be odd")
        self.num_frames = num_frames
        self.center = num_frames // 2
        self.scale = scale
        self.input_proj = nn.Conv2d(in_channels, dim, 3, padding=1)
        self.frequency = FrequencyBackbone(dim, heads, window_size)
        self.align = FrequencyGuidedAlign(dim)
        self.propagation = BidirectionalPropagation(dim)
        self.fusion = SKFF(dim, height=5)
        self.post_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )
        self.reconstruction = ReconstructionHead(dim, scale)

    def forward(self, inputs: torch.Tensor, return_aux: bool = False):
        if inputs.ndim != 5:
            raise ValueError("inputs must have shape [B, T, C, H, W]")
        b, t, c, h, w = inputs.shape
        if t != self.num_frames:
            raise ValueError(f"expected {self.num_frames} frames, got {t}")
        flat = inputs.reshape(b * t, c, h, w)
        shallow = self.input_proj(flat)
        encoded, high = self.frequency(shallow)
        encoded = encoded.reshape(b, t, -1, h, w)
        high = high.reshape(b, t, -1, h, w)
        center_feat = encoded[:, self.center]
        center_high = high[:, self.center]
        center_rgb = inputs[:, self.center]

        aligned = [center_feat]
        confidences = [torch.ones(b, 1, h, w, device=inputs.device, dtype=inputs.dtype)]
        flows = [torch.zeros(b, 2, h, w, device=inputs.device, dtype=inputs.dtype)]
        for index in range(t):
            if index == self.center:
                continue
            item, confidence, flow = self.align(
                center_feat,
                encoded[:, index],
                center_high,
                high[:, index],
                center_rgb,
                inputs[:, index],
            )
            aligned.append(item)
            confidences.append(confidence)
            flows.append(flow)

        # Restore temporal order after processing the neighbours around the centre.
        order = [self.center] + [i for i in range(t) if i != self.center]
        aligned = [item for _, item in sorted(zip(order, aligned), key=lambda pair: pair[0])]
        confidences = [item for _, item in sorted(zip(order, confidences), key=lambda pair: pair[0])]
        flows = [item for _, item in sorted(zip(order, flows), key=lambda pair: pair[0])]
        aligned_seq = torch.stack(aligned, dim=1)
        confidence_seq = torch.stack(confidences, dim=1)
        aggregate = (aligned_seq * confidence_seq).sum(dim=1) / (confidence_seq.sum(dim=1) + 1e-6)
        forward_state, backward_state = self.propagation(aligned_seq, self.center)
        fused = self.fusion([center_feat, aggregate, forward_state, backward_state, center_high])
        fused = fused + self.post_fusion(fused)
        residual = self.reconstruction(fused)
        base = F.interpolate(center_rgb, scale_factor=self.scale, mode="bicubic", align_corners=False)
        output = base + residual
        if return_aux:
            return output, {"flows": flows, "confidence": confidence_seq, "frequency": center_high}
        return output


Network = FADVSR

