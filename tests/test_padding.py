# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native padding semantics and offline CPU contracts, not host inference."""

import math
from dataclasses import replace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.fusion import FusionWeights
from tiled_diffusion_ng.geometry import (
    image_views,
    make_plan,
    pad_spatial,
    validate_plan,
)

from .host import Model, reshape_mask
from .test_anima import arguments as anima_arguments
from .test_fusion import fuse
from .test_geometry import spec
from .test_krea2 import arguments as krea2_arguments


@pytest.fixture(params=[anima_arguments, krea2_arguments], ids=["anima", "krea2"])
def arguments(request, host):
    return request.param


@pytest.mark.parametrize("family", ["anima", "krea2"])
def test_2_5x_portrait_keeps_requested_dimensions(host, family):
    from tiled_diffusion_ng.nodes import TilePlan

    # W×H 1040×1520 -> 2600×3800. Planning only needs tensor metadata.
    model = Model(family)
    latent = {"samples": torch.empty(1, 16, 475, 325, device="meta")}
    plan = TilePlan.execute(model, latent, 64)[0]
    assert plan.latent_hw == (475, 325)
    assert plan.pixel_hw == (3800, 2600)
    assert plan.padded_latent_hw == (476, 326)
    assert plan.padded_pixel_hw == (3808, 2608)
    assert plan.tile_hw == (242, 168)
    assert plan.effective_pixel_overlap == (64, 80)
    assert tuple(region.sampling for region in plan.regions) == (
        (0, 0, 168, 242),
        (158, 0, 326, 242),
        (158, 234, 326, 476),
        (0, 234, 168, 476),
    )
    validate_plan(plan, resolve_adapter(model).describe(model, latent))
    # The original size participates in identity even with identical padding.
    other = {"samples": torch.empty(1, 16, 476, 326, device="meta")}
    with pytest.raises(ValueError, match="Stale"):
        validate_plan(plan, resolve_adapter(model).describe(model, other))
    with pytest.raises(ValueError, match="fields or bounds"):
        validate_plan(replace(plan, padded_latent_hw=(478, 326)))


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("hw", [(13, 16), (12, 15), (13, 15)])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_each_evaluation_wraps_edges_and_returns_original_shape(
    arguments, host, rank, hw, dtype
):
    args = arguments(rank=rank, batch=2, hw=hw, dtype=dtype, steps=1)
    latent, plan = args["latent_image"], args["tile_plan"]
    latent["samples"].copy_(
        torch.linspace(-1, 1, latent["samples"].numel()).reshape_as(latent["samples"])
    )
    before = latent["samples"].clone()
    captured = []

    def pre(data):
        assert all(out.shape == (2, 16, 1, *hw) for out in data["conds_out"])
        return data["conds_out"]

    def post(data):
        captured.append(data["denoised"].clone())
        assert data["input"].shape == data["denoised"].shape == (2, 16, 1, *hw)
        return data["denoised"]

    args["model"].model_options.update(
        sampler_pre_cfg_function=[pre], sampler_post_cfg_function=[post]
    )
    result = sampling.sample(**args)
    assert result["samples"].shape == (2, 16, 1, *hw)
    assert result["samples"].isfinite().all()
    assert result["note"] == "retained"
    assert len(host.common_calls) == 1 and host.common_calls[0]["latent"] is latent
    assert len(captured) == len(host.global_evaluations) > 1
    for evaluation, (current, sigma) in enumerate(host.global_evaluations):
        assert current.shape[-2:] == hw
        for index, region in enumerate(plan.regions):
            _, tile, tile_sigma, _ = host.tile_calls[evaluation * 4 + index]
            x0, y0, x1, y1 = region.sampling
            # Independent circular indexing, including the bottom-right corner.
            expected = current.index_select(-2, torch.arange(y0, y1) % hw[0])
            expected = expected.index_select(-1, torch.arange(x0, x1) % hw[1])
            torch.testing.assert_close(tile, expected, rtol=0, atol=0)
            torch.testing.assert_close(tile_sigma, sigma)
            assert tile.shape[-2:] == plan.tile_hw
    torch.testing.assert_close(latent["samples"], before, rtol=0, atol=0)
    repeated = sampling.sample(**args)["samples"]
    torch.testing.assert_close(repeated, result["samples"], rtol=0, atol=0)
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("denoise", [0, 0.4])
def test_padding_preserves_native_masks_batch_indices_and_zero_denoise(
    arguments, host, rank, denoise
):
    args = arguments(rank=rank, batch=2, hw=(13, 15), steps=1, denoise=denoise)
    latent = args["latent_image"]
    latent["samples"].fill_(0.5)
    latent["batch_index"] = [7, 7]
    baseline = sampling.sample(**args)["samples"]
    torch.testing.assert_close(baseline[0], baseline[1])
    mask = torch.zeros(2, 1, 1, 6, 8)
    mask[1, ..., 4:] = 1
    before = mask.clone()
    latent["noise_mask"] = mask
    result = sampling.sample(**args)
    assert (
        result["noise_mask"] is mask and result["batch_index"] is latent["batch_index"]
    )
    assert host.common_calls[-1]["latent"] is latent
    actual = result["samples"]
    assert actual.shape == (2, 16, 1, 13, 15)
    prepared = reshape_mask(mask, actual.shape)
    assert torch.all(actual[prepared == 0] == 0.5)
    torch.testing.assert_close(actual[prepared == 1], baseline[prepared == 1])
    if denoise == 0:
        assert torch.all(actual == 0.5)
    assert torch.all(latent["samples"] == 0.5) and latent["samples"].ndim == rank
    torch.testing.assert_close(mask, before, rtol=0, atol=0)


@pytest.mark.parametrize("hw", [(13, 16), (12, 15), (13, 15)])
@pytest.mark.parametrize("channels", [3, 4])
def test_tileview_wraps_pixels_without_resizing_or_mutating(arguments, hw, channels):
    from tiled_diffusion_ng.nodes import TileView

    plan = arguments(hw=hw)["tile_plan"]
    h, w = plan.pixel_hw
    image = torch.arange(2 * h * w * channels, dtype=torch.float64).reshape(
        2, h, w, channels
    )
    before = image.clone()
    views = TileView.execute(image, plan)[0]
    assert views.shape == (8, *plan.tile_pixel_hw, channels)
    for source in range(2):
        for index, region in enumerate(plan.regions):
            x0, y0, x1, y1 = region.pixel_sampling
            expected = image[source].index_select(0, torch.arange(y0, y1) % h)
            expected = expected.index_select(1, torch.arange(x0, x1) % w)
            torch.testing.assert_close(
                views[source * 4 + index], expected, rtol=0, atol=0
            )
    views.zero_()
    torch.testing.assert_close(image, before, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Expected image H×W"):
        image_views(image[:, :-1], plan)


@pytest.mark.parametrize("hw", [(5, 8), (6, 7), (5, 7)])
def test_padded_fusion_matches_scalar_oracle_and_discards_border(hw):
    plan = make_plan(spec(hw, scale=(1, 1), alignment=(2, 2), padding="circular"), 1)
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
    output = fuse(weights, fields)
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
    assert output.shape == (2, 3, 1, *hw)
    torch.testing.assert_close(output, expected.expand_as(output))
    assert torch.all(weights.denominator > 0)


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("mode", ["tracing", "scripting"])
def test_native_reflect_fallback_during_compilation(monkeypatch, rank, mode):
    monkeypatch.setattr(torch.jit, f"is_{mode}", lambda: True)
    shape = (2, 3) + ((1,) if rank == 5 else ()) + (5, 7)
    tensor = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    expected = tensor.index_select(-2, torch.tensor([0, 1, 2, 3, 4, 3]))
    expected = expected.index_select(-1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 5]))
    torch.testing.assert_close(pad_spatial(tensor, (6, 8)), expected, rtol=0, atol=0)
    assert pad_spatial(tensor, (5, 7)) is tensor


def test_padding_policy_is_validated():
    with pytest.raises(ValueError, match="padding policy"):
        make_plan(spec((7, 9), padding="replicate"), 0)
