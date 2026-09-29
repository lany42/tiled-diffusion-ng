# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import math

import pytest
import torch

from tiled_diffusion_ng.fusion import FusionWeights, gaussian
from tiled_diffusion_ng.geometry import crop, make_plan

from .test_geometry import spec


def fuse(weights, fields):
    out = torch.zeros(
        (*fields[0].shape[:-2], *weights.plan.padded_latent_hw), dtype=weights.dtype
    )
    for region, field in zip(weights.plan.regions, fields, strict=True):
        weights.accumulate(out, field, region)
    return weights.normalize(out, fields[0].dtype)


@pytest.mark.parametrize("hw", [(7, 11), (156, 108)])
def test_gaussian_against_scalar_formula(hw):
    h, w = hw
    actual = gaussian(hw, device="cpu", dtype=torch.float64)
    # Independent cell-center coordinates: compare the complete rectangular
    # kernel, including edges where the old vertical denominator was wrong.
    expected = torch.tensor(
        [
            [
                math.exp(
                    -(((x + 0.5) / w - 0.5) ** 2 + ((y + 0.5) / h - 0.5) ** 2) / 0.02
                )
                for x in range(w)
            ]
            for y in range(h)
        ],
        dtype=torch.float64,
    )
    # FP64 predictions keep FP64 weights, computed at full precision.
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_constant_coverage_precision_and_nonspatial_axes(dtype):
    plan = make_plan(spec((9, 13)), 8)
    weights = FusionWeights(plan, device="cpu", dtype=dtype, rank=5)
    assert weights.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    fields = [torch.full((2, 3, 2, *plan.tile_hw), 2.75, dtype=dtype)] * 4
    output = fuse(weights, fields)
    assert output.dtype == dtype
    assert output.shape == (2, 3, 2, 9, 13)
    torch.testing.assert_close(output, torch.full_like(output, 2.75))


@pytest.mark.parametrize(
    "plan",
    [
        make_plan(spec((5, 7), scale=(1, 1)), 2),
        make_plan(spec((5, 7), scale=(1, 1), alignment=(2, 2), padding="circular"), 1),
    ],
    ids=["aligned", "padded"],
)
def test_fusion_matches_scalar_oracle_and_discards_padding(plan):
    hw = plan.latent_hw
    weights = FusionWeights(plan, device="cpu", dtype=torch.float64, rank=5)
    fields = [
        torch.full((2, 3, 1, *plan.tile_hw), i + 1.0, dtype=torch.float64)
        for i in range(4)
    ]
    # Values outside the requested image must not leak into its predictions.
    for region, field in zip(plan.regions, fields, strict=True):
        x0, y0, x1, y1 = region.sampling
        if y1 > hw[0]:
            field[..., hw[0] - y0 :, :] = 1e9
        if x1 > hw[1]:
            field[..., hw[1] - x0 :] = 1e9
    actual = fuse(weights, fields)
    expected = torch.empty(hw, dtype=torch.float64)
    th, tw = plan.tile_hw
    for y in range(hw[0]):
        for x in range(hw[1]):
            total, divisor = 0.0, 0.0
            for index, region in enumerate(plan.regions):
                x0, y0, x1, y1 = region.sampling
                if x0 <= x < x1 and y0 <= y < y1:
                    weight = math.exp(
                        -50
                        * (
                            ((x - x0 + 0.5) / tw - 0.5) ** 2
                            + ((y - y0 + 0.5) / th - 0.5) ** 2
                        )
                    )
                    total += weight * (index + 1)
                    divisor += weight
            expected[y, x] = total / divisor
    assert actual.shape == (2, 3, 1, *hw)
    torch.testing.assert_close(actual, expected.expand_as(actual))
    # Each corner belongs to exactly one view.
    assert actual[0, 0, 0, 0, 0] == 1 and actual[0, 0, 0, -1, -1] == 3


def test_cfg_and_affine_representation_equivalence():
    # Fusing then guiding equals guiding then fusing, and denoised conversion
    # commutes; an epsilon or clamp in normalization would break both.
    plan = make_plan(spec((19, 27)), 32)
    weights = FusionWeights(plan, device="cpu", dtype=torch.float64)
    generator = torch.Generator().manual_seed(42)
    p, n = (
        [
            torch.randn((2, 7, *plan.tile_hw), generator=generator, dtype=torch.float64)
            for _ in range(4)
        ]
        for _ in range(2)
    )
    expected = fuse(weights, n) + 7.5 * (fuse(weights, p) - fuse(weights, n))
    actual = fuse(weights, [ni + 7.5 * (pi - ni) for pi, ni in zip(p, n)])
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    x = torch.randn((2, 7, *plan.latent_hw), generator=generator, dtype=torch.float64)
    denoised = [
        0.7 * crop(x, region.sampling) - 0.2 * noise
        for region, noise in zip(plan.regions, p)
    ]
    torch.testing.assert_close(
        fuse(weights, denoised), 0.7 * x - 0.2 * fuse(weights, p)
    )
