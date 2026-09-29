# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Gaussian partition of unity, without host-specific model assumptions."""

import torch

from .geometry import TilePlanData, crop, validate_plan


def working_dtype(dtype):
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"Unsupported prediction dtype: {dtype}")
    return torch.float64 if dtype == torch.float64 else torch.float32


def gaussian(hw, *, device, dtype):
    # Paper §3.1: normalized axes, variance .01 (standard deviation .1).
    # https://arxiv.org/html/2302.02412v1#S3
    # Symmetric discrete centers and per-axis scales follow the author's canvas:
    # https://github.com/albarji/mixture-of-diffusers/blob/af42292d0a8cb414f6da2eeac79be4c60afbbe48/mixdiff/canvas.py#L187-L200
    # Intentionally correct BOTH vertical midpoint and denominator, unlike the
    # denominator-only patch: https://github.com/shiimizu/ComfyUI-TiledDiffusion/pull/77
    # https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L450-L461
    h, w = hw
    dtype = working_dtype(dtype)
    y = (torch.arange(h, device=device, dtype=dtype) - (h - 1) / 2) / h
    x = (torch.arange(w, device=device, dtype=dtype) - (w - 1) / 2) / w
    return torch.exp(-50 * y.square())[:, None] * torch.exp(-50 * x.square())[None, :]


class FusionWeights:
    def __init__(self, plan: TilePlanData, *, device, dtype, rank=4):
        validate_plan(plan)
        self.plan = plan
        self.dtype = working_dtype(dtype)
        self.kernel = gaussian(plan.tile_hw, device=device, dtype=self.dtype).reshape(
            (1,) * (rank - 2) + plan.tile_hw
        )
        self.denominator = torch.zeros(
            (1,) * (rank - 2) + plan.padded_latent_hw, device=device, dtype=self.dtype
        )
        for region in plan.regions:
            crop(self.denominator, region.sampling).add_(self.kernel)
        if not torch.all(torch.isfinite(self.denominator) & (self.denominator > 0)):
            raise ValueError(
                "Gaussian denominator must be finite and positive everywhere"
            )

    def accumulate(self, output, prediction, region):
        crop(output, region.sampling).add_(prediction.to(self.dtype) * self.kernel)

    def normalize(self, output, dtype):
        # Eq.16 / Algorithm 1: D=sum E(g), whereas paper Z=1/D.
        # https://arxiv.org/html/2302.02412v1#S3
        # Normalize in >=FP32 (FP64 stays FP64); singleton non-spatial axes
        # broadcast over every batch/channel. No epsilon clamp: edge weights
        # can be tiny but are positive. This preserves constant predictions.
        h, w = self.plan.latent_hw
        # Discard padding before returning predictions to native CFG/solvers.
        return (output[..., :h, :w] / self.denominator[..., :h, :w]).to(dtype)
