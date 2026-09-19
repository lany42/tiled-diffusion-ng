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
        (*fields[0].shape[:-2], *weights.plan.latent_hw), dtype=weights.dtype
    )
    for region, field in zip(weights.plan.regions, fields, strict=True):
        weights.accumulate(out, field, region)
    return weights.normalize(out, fields[0].dtype)


@pytest.mark.parametrize("hw", [(7, 11), (11, 7), (8, 8), (156, 108), (160, 112)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gaussian_against_scalar_formula(hw, dtype):
    h, w = hw
    actual = gaussian(hw, device="cpu", dtype=dtype)
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
        dtype=dtype,
    )
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual, actual.flip(0))
    assert torch.equal(actual, actual.flip(1))
    assert torch.equal(actual.T, gaussian((w, h), device="cpu", dtype=dtype))
    assert actual.min() > 0


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize(
    "hw,overlap", [((304, 208), 64), ((304, 208), 128), ((8, 12), 0), ((9, 13), 8)]
)
def test_constant_coverage_precision_and_nonspatial_axes(dtype, hw, overlap):
    plan = make_plan(spec(hw), overlap)
    weights = FusionWeights(plan, device="cpu", dtype=dtype, rank=5)
    assert weights.dtype == (torch.float64 if dtype == torch.float64 else torch.float32)
    fields = [torch.full((2, 3, 2, *plan.tile_hw), 2.75, dtype=dtype)] * 4
    output = fuse(weights, fields)
    assert output.dtype == dtype
    assert output.shape == (2, 3, 2, *hw)
    torch.testing.assert_close(output, torch.full_like(output, 2.75))
    assert torch.isfinite(weights.denominator).all() and (weights.denominator > 0).all()


def test_independent_scalar_fusion_at_all_coverage_cases():
    plan = make_plan(spec((5, 7), scale=(1, 1)), 2)
    weights = FusionWeights(plan, device="cpu", dtype=torch.float64)
    fields = [
        torch.full((1, 5, *plan.tile_hw), float(i + 1), dtype=torch.float64)
        for i in range(4)
    ]
    actual = fuse(weights, fields)
    expected = torch.empty(5, 7, dtype=torch.float64)
    h, w = plan.tile_hw
    for y in range(5):
        for x in range(7):
            total, divisor = 0, 0
            for i, region in enumerate(plan.regions):
                x0, y0, x1, y1 = region.sampling
                if x0 <= x < x1 and y0 <= y < y1:
                    weight = math.exp(
                        -50
                        * (
                            ((x - x0 + 0.5) / w - 0.5) ** 2
                            + ((y - y0 + 0.5) / h - 0.5) ** 2
                        )
                    )
                    total += weight * (i + 1)
                    divisor += weight
            expected[y, x] = total / divisor
    torch.testing.assert_close(actual, expected.expand(1, 5, 5, 7))
    assert actual[0, 0, 0, 0] == 1
    assert actual[0, 0, -1, -1] == 3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cfg_and_affine_representation_equivalence(dtype):
    plan = make_plan(spec((19, 27)), 32)
    weights = FusionWeights(plan, device="cpu", dtype=dtype)
    generator = torch.Generator().manual_seed(42)
    fields = [
        [
            torch.randn((2, 7, *plan.tile_hw), generator=generator, dtype=dtype)
            for _ in range(4)
        ]
        for _ in range(2)
    ]
    p, n = fields
    expected = fuse(weights, n) + 7.5 * (fuse(weights, p) - fuse(weights, n))
    actual = fuse(weights, [ni + 7.5 * (pi - ni) for pi, ni in zip(p, n)])
    torch.testing.assert_close(
        actual,
        expected,
        atol=1e-5 if dtype == torch.float32 else 1e-13,
        rtol=1e-5 if dtype == torch.float32 else 1e-13,
    )
    x = torch.randn((2, 7, *plan.latent_hw), generator=generator, dtype=dtype)
    denoised = [
        0.7 * crop(x, region.sampling) - 0.2 * noise
        for region, noise in zip(plan.regions, p)
    ]
    torch.testing.assert_close(
        fuse(weights, denoised), 0.7 * x - 0.2 * fuse(weights, p)
    )


def test_bad_denominator_is_not_clamped(monkeypatch):
    import tiled_diffusion_ng.fusion as module

    plan = make_plan(spec((8, 10)), 0)
    monkeypatch.setattr(module, "gaussian", lambda hw, **kwargs: torch.zeros(hw))
    with pytest.raises(ValueError, match="positive everywhere"):
        FusionWeights(plan, device="cpu", dtype=torch.float32)
