# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import math
from dataclasses import replace

import pytest
import torch

from tiled_diffusion_ng.geometry import (
    GeometrySignature,
    LatentSpec,
    image_views,
    make_plan,
    pad_spatial,
    validate_plan,
)


def spec(hw, **kwargs):
    return LatentSpec(GeometrySignature("test", 1, "synthetic", **kwargs), hw)


def circular(tensor, rect, hw, axes=(-2, -1)):
    """Independent wrap-around crop of a half-open (x0, y0, x1, y1) rectangle."""
    x0, y0, x1, y1 = rect
    tensor = tensor.index_select(axes[0], torch.arange(y0, y1) % hw[0])
    return tensor.index_select(axes[1], torch.arange(x0, x1) % hw[1])


@pytest.mark.parametrize(
    "overlap,extent,rects",
    [
        (
            64,
            (156, 108),
            (
                (0, 0, 108, 156),
                (100, 0, 208, 156),
                (100, 148, 208, 304),
                (0, 148, 108, 304),
            ),
        ),
        (
            128,
            (160, 112),
            (
                (0, 0, 112, 160),
                (96, 0, 208, 160),
                (96, 144, 208, 304),
                (0, 144, 112, 304),
            ),
        ),
    ],
)
def test_portrait_handwritten(overlap, extent, rects):
    plan = make_plan(spec((304, 208)), overlap)
    assert plan.tile_hw == extent
    assert tuple(r.sampling for r in plan.regions) == rects
    assert tuple(r.pixel_sampling for r in plan.regions) == tuple(
        tuple(v * 8 for v in rect) for rect in rects
    )
    assert plan.pixel_hw == (2432, 1664)
    assert plan.regions[0].pixel_core == (0, 0, 832, 1216)
    assert plan.effective_pixel_overlap == (overlap, overlap)


def test_odd_landscape_and_rounded_overlap():
    plan = make_plan(spec((193, 257)), 128)
    assert plan.tile_hw == (105, 137)
    assert plan.effective_overlap == (17, 17)
    assert plan.effective_pixel_overlap == (136, 136)
    assert plan.regions[2].core == (128, 96, 257, 193)
    plan = make_plan(spec((192, 256)), 128)
    assert plan.tile_pixel_hw == (832, 1088)
    assert plan.regions[2].pixel_sampling == (960, 704, 2048, 1536)
    plan = make_plan(spec((9, 10)), 0)
    assert plan.effective_overlap == (1, 0)


def test_extent_against_independent_enumeration():
    for length in range(2, 33):
        for align in (1, 2, 4):
            for minimum in (1, 3, 8):
                for scale in (1, 8):
                    for overlap in (0, 1, 7, 16, 64):
                        feasible = [
                            t
                            for t in range(1, length)
                            if length % align == 0
                            and t % align == 0
                            and t >= minimum
                            and (2 * t - length) * scale >= overlap
                        ]
                        current = spec(
                            (length, length),
                            scale=(scale, scale),
                            alignment=(align, align),
                            minimum=(minimum, minimum),
                        )
                        if feasible:
                            assert (
                                make_plan(current, overlap).tile_hw
                                == (min(feasible),) * 2
                            )
                        else:
                            with pytest.raises(ValueError):
                                make_plan(current, overlap)


@pytest.mark.parametrize("overlap", [-1, True])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="nonnegative integer"):
        make_plan(spec((20, 20)), overlap)


def test_padded_plan_keeps_requested_canvas_identity():
    # A 2.5x portrait, W×H 1040×1520 -> 2600×3800, padded to 2-cell alignment.
    wan = {"alignment": (2, 2), "minimum": (2, 2), "padding": "circular"}
    plan = make_plan(spec((475, 325), **wan), 64)
    assert (plan.latent_hw, plan.pixel_hw) == ((475, 325), (3800, 2600))
    assert (plan.padded_latent_hw, plan.padded_pixel_hw) == ((476, 326), (3808, 2608))
    assert plan.tile_hw == (242, 168)
    assert plan.effective_pixel_overlap == (64, 80)
    assert tuple(region.sampling for region in plan.regions) == (
        (0, 0, 168, 242),
        (158, 0, 326, 242),
        (158, 234, 326, 476),
        (0, 234, 168, 476),
    )
    validate_plan(plan, spec((475, 325), **wan))
    # The original size participates in identity even with identical padding.
    with pytest.raises(ValueError, match="Stale"):
        validate_plan(plan, spec((476, 326), **wan))
    with pytest.raises(ValueError, match="padding policy"):
        make_plan(spec((7, 9), padding="replicate"), 0)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("schema_version", 1, "schema version 2"),
        ("tile_ids", ("TL", "TR", "BL", "BR"), "fields or bounds"),
        ("padded_latent_hw", (22, 22), "fields or bounds"),
    ],
)
def test_revalidate_untrusted_plans(field, value, error):
    plan = make_plan(spec((20, 20)), 8)
    with pytest.raises(ValueError, match=error):
        validate_plan(replace(plan, **{field: value}))


@pytest.mark.parametrize(
    "plan",
    [
        make_plan(spec((5, 7), scale=(2, 3)), 3),
        make_plan(spec((13, 15), alignment=(2, 2), padding="circular"), 16),
    ],
    ids=["aligned", "padded"],
)
def test_image_views_are_image_major_wrapped_crops(plan):
    h, w = plan.pixel_hw
    image = torch.arange(2 * h * w * 4, dtype=torch.float64).reshape(2, h, w, 4)
    before = image.clone()
    views = image_views(image, plan)
    assert views.shape == (8, *plan.tile_pixel_hw, 4)
    for source in range(2):
        for index, region in enumerate(plan.regions):
            expected = circular(image[source], region.pixel_sampling, (h, w), (0, 1))
            torch.testing.assert_close(
                views[4 * source + index], expected, rtol=0, atol=0
            )
    views.zero_()
    assert torch.equal(image, before)
    with pytest.raises(ValueError, match="Expected image H×W"):
        image_views(image[:, :-1], plan)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        image_views(image[..., :2], plan)


def test_native_reflect_fallback_during_compilation(monkeypatch):
    shape = (2, 3, 1, 5, 7)
    tensor = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    assert pad_spatial(tensor, (5, 7)) is tensor
    monkeypatch.setattr(torch.jit, "is_tracing", lambda: True)
    expected = tensor.index_select(-2, torch.tensor([0, 1, 2, 3, 4, 3]))
    expected = expected.index_select(-1, torch.tensor([0, 1, 2, 3, 4, 5, 6, 5]))
    torch.testing.assert_close(pad_spatial(tensor, (6, 8)), expected, rtol=0, atol=0)
