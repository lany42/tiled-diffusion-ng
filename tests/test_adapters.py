# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Model-family boundaries: resolution, latent layouts and prediction types."""

from dataclasses import replace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.adapters.anima import AnimaAdapter
from tiled_diffusion_ng.adapters.krea2 import Krea2Adapter
from tiled_diffusion_ng.adapters.sdxl import SDXLAdapter
from tiled_diffusion_ng.geometry import make_plan, validate_plan

from .host import EPS, Model, SDXLFormat, VPrediction, arguments, cond


def describe(family, latent):
    model = Model(family)
    return resolve_adapter(model).describe(model, latent)


def test_families_resolve_and_older_hosts_keep_other_families(host, monkeypatch):
    from comfy import model_base

    for family, adapter in (
        ("sdxl", SDXLAdapter),
        ("anima", AnimaAdapter),
        ("krea2", Krea2Adapter),
    ):
        assert type(resolve_adapter(Model(family))) is adapter
    mismatched = [Model(family) for family in ("anima", "krea2")]
    for model in mismatched:
        model.format = SDXLFormat()
    unrelated = Model()
    unrelated.model = object()
    for model in (*mismatched, unrelated):
        with pytest.raises(ValueError, match="Unsupported model/latent family"):
            resolve_adapter(model)
    # Older hosts may lack newer model classes entirely.
    monkeypatch.delattr(model_base, "Anima")
    monkeypatch.delattr(model_base, "Krea2")
    assert type(resolve_adapter(Model())) is SDXLAdapter
    for family in ("anima", "krea2"):
        with pytest.raises(ValueError, match="Unsupported model/latent family"):
            resolve_adapter(Model(family))


def test_sdxl_plans_read_metadata_and_allow_batch_reuse(host):
    latent = {"samples": torch.empty(1, 4, 17, 21, device="meta"), "note": "kept"}
    plan = make_plan(describe("sdxl", latent), 8)
    # Native LATENT bookkeeping and masks are accepted without inspection.
    other = {
        "samples": torch.zeros(2, 4, 17, 21),
        "noise_mask": torch.ones(2, 17, 21),
        "batch_index": [3, 4],
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 1,
    }
    validate_plan(plan, describe("sdxl", other))
    with pytest.raises(ValueError, match="Stale"):
        validate_plan(plan, describe("sdxl", {"samples": torch.zeros(1, 4, 18, 21)}))
    for samples in (torch.zeros(1, 4, 1, 17, 21), torch.zeros(1, 8, 17, 21)):
        with pytest.raises(ValueError, match="SDXL requires"):
            describe("sdxl", {"samples": samples})
    for field, value in (
        ("downscale_ratio_spacial", 16),
        ("type", "video"),
        ("spatial", torch.zeros(2)),
    ):
        with pytest.raises(ValueError, match=field):
            describe("sdxl", dict(other, **{field: value}))
    # Inpaint checkpoints resolve as SDXL but fail at planning, before loading.
    inpaint = Model()
    inpaint.model.concat_keys = ("mask", "masked_image")
    with pytest.raises(ValueError, match="concat/inpaint"):
        resolve_adapter(inpaint).describe(inpaint, other)


@pytest.mark.parametrize(
    "field,value",
    [("area", (5, 5, 0, 0)), ("unknown_tensor", torch.ones(2, 3))],
)
def test_sdxl_unsupported_condition_fields_are_named(host, field, value):
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments(positive=cond(1, **{field: value})))
    assert not host.common_calls


def test_sdxl_prepared_spatial_inputs_are_rejected(host):
    host.mutate_prepared = lambda branches: branches[0][0]["model_conds"].update(
        c_concat=object()
    )
    with pytest.raises(ValueError, match="c_concat"):
        sampling.sample(**arguments())
    assert not host.tile_calls


def test_sdxl_requires_native_epsilon_or_v_prediction(host):
    args = arguments()
    args["model"].sampling = VPrediction()
    sampling.sample(**args)
    args["model"].sampling.calculate_denoised = lambda *a: None
    with pytest.raises(ValueError, match="model_sampling"):
        sampling.sample(**args)


@pytest.mark.parametrize("family,other", [("anima", "krea2"), ("krea2", "anima")])
def test_wan_plans_ignore_layout_batch_and_depth_but_not_family_or_canvas(
    host, family, other
):
    args = arguments(family)
    plan = args["tile_plan"]
    assert plan.signature.padding == "circular" and plan.signature.alignment == (2, 2)
    # Checkpoint depth (28/40 blocks), 4D/5D input and batch share one plan.
    reused = arguments(family, rank=5, batch=2, depth=40)
    assert reused["tile_plan"] == plan
    sampling.sample(**{**reused, "tile_plan": plan})
    for stale in (
        arguments(other)["tile_plan"],
        arguments("sdxl")["tile_plan"],
        arguments(family, hw=(16, 12))["tile_plan"],
        replace(plan, signature=replace(plan.signature, adapter_version=2)),
    ):
        with pytest.raises(ValueError, match="Stale"):
            sampling.sample(**{**args, "tile_plan": stale})
    # Geometry reads tensor metadata, never values.
    meta = {"samples": torch.empty(2, 16, 1, 12, 16, device="meta")}
    validate_plan(plan, describe(family, meta))


@pytest.mark.parametrize(
    "family,samples",
    [
        ("anima", torch.zeros(1, 16, 2, 12, 16)),
        ("krea2", torch.zeros(1, 4, 12, 16)),
        ("anima", torch.zeros(1, 16, 12, 16, dtype=torch.int64)),
        ("krea2", None),
    ],
    ids=["video", "sdxl_channels", "integer", "missing"],
)
def test_wan_targets_require_single_frame_16_channel_images(host, family, samples):
    with pytest.raises(ValueError, match=f"{family.title()} requires"):
        describe(family, {"samples": samples})


@pytest.mark.parametrize(
    "family,mask,valid",
    [
        ("anima", torch.ones(1, 1, 1, 6, 8), True),
        ("krea2", torch.ones(2, 1, 1, 12, 16), True),
        ("anima", torch.ones(2, 1, 12, 16), False),
        ("krea2", torch.ones(3, 1, 1, 12, 16), False),
        ("anima", torch.ones(2, 1, 2, 12, 16), False),
        ("anima", torch.ones(2, 1, 1, 12, 16, dtype=torch.int64), False),
    ],
    ids=["broadcast", "per_image", "sub_5d", "batch", "multi_frame", "integer"],
)
def test_wan_noise_masks_require_native_5d_layout(host, family, mask, valid):
    # Native preparation treats sub-5D mask axes as time, mixing images.
    latent = {
        "samples": torch.zeros(2, 16, 12, 16),
        "noise_mask": mask,
        "batch_index": [7, 7],
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 4,
    }
    if valid:
        assert describe(family, latent).canvas == (12, 16)
    else:
        with pytest.raises(ValueError, match="noise_mask.*B×1×1×H×W"):
            describe(family, latent)


@pytest.mark.parametrize(
    "family,field,value",
    [
        ("anima", "downscale_ratio_temporal", 1),
        ("krea2", "type", "video"),
        ("anima", "latent_shapes", [(16, 1, 12, 16)]),
    ],
)
def test_wan_latent_metadata_is_named(host, family, field, value):
    args = arguments(family)
    args["latent_image"][field] = value
    with pytest.raises(ValueError, match=field):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize(
    "family,change,error",
    [
        ("anima", "calculate_input", "native CONST"),
        ("krea2", "calculate_denoised", "native CONST"),
        ("anima", "format", "Wan21 latent format metadata"),
    ],
)
def test_wan_models_require_native_format_and_const_conversion(
    host, family, change, error
):
    args = arguments(family)
    if change == "format":
        args["model"].format.temporal_downscale_ratio = 1
    else:
        setattr(args["model"].sampling, change, getattr(EPS(), change))
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.common_calls
