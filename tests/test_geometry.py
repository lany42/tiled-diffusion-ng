# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.geometry import (
    GeometrySignature,
    LatentSpec,
    image_views,
    make_plan,
    validate_plan,
)

from .host import Model


def spec(hw, **kwargs):
    return LatentSpec(GeometrySignature("test", 1, "synthetic", **kwargs), hw)


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
    with pytest.raises(FrozenInstanceError):
        plan.layout = "other"


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


@pytest.mark.parametrize("overlap", [-1, 0.5, True, "64", None])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="nonnegative integer"):
        make_plan(spec((20, 20)), overlap)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("layout", "grid"),
        ("tile_count", 3),
        ("tile_ids", ("TL", "TR", "BL", "BR")),
        ("variance", 0.1),
        ("pixel_hw", (1, 1)),
        ("effective_overlap", (0, 0)),
        ("tile_hw", (2, 2)),
        ("regions", ()),
        ("kernel", "legacy"),
        ("center", "edge"),
    ],
)
def test_revalidate_untrusted_plans(field, value):
    plan = make_plan(spec((20, 20)), 8)
    with pytest.raises(ValueError):
        validate_plan(replace(plan, **{field: value}))


@pytest.mark.parametrize("channels", [3, 4])
def test_actual_views_are_image_major_and_preserve_values(channels):
    plan = make_plan(spec((5, 7), scale=(2, 3)), 3)
    image = torch.arange(2 * 10 * 21 * channels, dtype=torch.float64).reshape(
        2, 10, 21, channels
    )
    before = image.clone()
    views = image_views(image, plan)
    for image_index in range(2):
        for index, region in enumerate(plan.regions):
            x0, y0, x1, y1 = region.pixel_sampling
            torch.testing.assert_close(
                views[4 * image_index + index], image[image_index, y0:y1, x0:x1]
            )
    assert views.shape == (8, *plan.tile_pixel_hw, channels)
    assert views.dtype == image.dtype and views.device == image.device
    views.zero_()
    assert torch.equal(image, before)
    with pytest.raises(ValueError, match="Expected image H×W"):
        image_views(image[:, :-1], plan)
    with pytest.raises(ValueError, match="RGB or RGBA"):
        image_views(image[..., :1], plan)


def test_adapter_reads_metadata_without_values_and_allows_batch_reuse(host):
    model = Model()
    adapter = resolve_adapter(model)
    latent = {"samples": torch.empty(1, 4, 17, 21, device="meta"), "note": "retained"}
    plan = make_plan(adapter.describe(model, latent), 8)
    assert latent["samples"].device.type == "meta"
    other = {"samples": torch.zeros(2, 4, 17, 21)}
    validate_plan(plan, adapter.describe(Model(), other))
    with pytest.raises(ValueError, match="Stale"):
        validate_plan(
            plan, adapter.describe(model, {"samples": torch.zeros(1, 4, 18, 21)})
        )
    for samples in (
        torch.zeros(1, 4, 1, 17, 21),
        torch.zeros(1, 8, 17, 21),
        torch.zeros(0, 4, 17, 21),
        torch.zeros(1, 4, 17, 21, dtype=torch.int64),
    ):
        with pytest.raises(ValueError, match="SDXL requires"):
            adapter.describe(model, {"samples": samples})
    for field, value in (
        ("downscale_ratio_spacial", 16),
        ("type", "video"),
        ("spatial_unknown", torch.zeros(2, 2)),
    ):
        with pytest.raises(ValueError, match=field):
            adapter.describe(model, dict(other, **{field: value}))
    model.model = object()
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_adapter(model)
