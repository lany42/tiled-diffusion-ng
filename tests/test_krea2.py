# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Offline CPU Krea2 contracts, not evidence of upstream inference.

Inspected source: ComfyUI c194dd00cd42aa18d9dbf27d977bf6b85d9ea565.
https://github.com/Comfy-Org/ComfyUI/tree/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565
Raw and Turbo each still require real-host generation/refinement, native text
encoding and Qwen image VAE, portrait/landscape, global/identical/distinct locals,
batch two, masks, cancellation/failure/reuse and visual tile-boundary acceptance.
Reference acceptance separately needs both methods, multiple sizes, local/global
references and the upstream style-reference model/LoRA configuration. Confirm
wrapper routing, preparation and V3 execution-list behavior on that actual host.
Comparison settings only: Turbo 8 steps/CFG 1/Euler/simple; Raw 52/4.5/Euler/simple.
These do not change public defaults. Official Raw guidance 3.5 uses P+3.5(P-N).
https://github.com/krea-ai/krea-2/blob/db3984fbc6e13b34c0064990fc2d95ac64d00058/sampling.py#L122
"""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.adapters._krea2_sampling import PREPARATION
from tiled_diffusion_ng.adapters.krea2 import Krea2Adapter
from tiled_diffusion_ng.geometry import crop, make_plan, validate_plan

from .host import EPS, ControlNet, Model, SDXLFormat, copy_containers
from .krea2_host import cond
from .test_anima import arguments as anima_arguments
from .test_sampling import arguments as sdxl_arguments


def arguments(*, rank=4, batch=1, hw=(12, 16), depth=28, dtype=torch.float32, **kwargs):
    model = Model("krea2", num_blocks=depth)
    shape = (batch, 16) + ((1,) if rank == 5 else ()) + hw
    latent = {"samples": torch.zeros(shape, dtype=dtype), "note": "retained"}
    result = {
        "model": model,
        "seed": 321,
        "steps": 1,
        "cfg": 4.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": cond(2),
        "negative": cond(-1),
        "latent_image": latent,
        "tile_plan": make_plan(resolve_adapter(model).describe(model, latent), 16),
        "denoise": 0.4,
    }
    result.update(kwargs)
    return result


@pytest.mark.parametrize(
    "rank,batch,depth", [(4, 1, 28), (4, 2, 40), (5, 1, 16), (5, 2, 28)]
)
def test_native_layout_depth_and_one_trajectory(host, rank, batch, depth):
    args = arguments(rank=rank, batch=batch, depth=depth, steps=2)
    model, latent, plan = args["model"], args["latent_image"], args["tile_plan"]
    network = model.model.diffusion_model
    embedder = network.pe_embedder
    assert isinstance(resolve_adapter(model), Krea2Adapter)
    assert (plan.signature.adapter_id, plan.signature.adapter_version) == ("krea2", 1)
    assert (plan.signature.layout, plan.signature.rank, plan.signature.channels) == (
        "BCTHW",
        5,
        16,
    )
    assert plan.signature.scale == (8, 8)
    assert plan.signature.spatial_axes == (-2, -1)
    assert plan.signature.alignment == plan.signature.minimum == (2, 2)
    result = sampling.sample(**args)
    assert result["samples"].shape == (batch, 16, 1, *plan.latent_hw)
    assert result["note"] == "retained"
    assert latent["samples"].ndim == rank and not latent["samples"].any()
    assert host.latest_clone.model.diffusion_model is network
    assert network.pe_embedder is embedder and len(network.blocks) == depth
    assert len(host.common_calls) == 1 and len(host.global_evaluations) == 6
    assert len(host.tile_calls) == 24
    assert host.common_calls[0]["latent"] is latent
    for evaluation, (x, sigma) in enumerate(host.global_evaluations):
        for index, region in enumerate(plan.regions):
            branches, tile, tiled_sigma, _ = host.tile_calls[evaluation * 4 + index]
            torch.testing.assert_close(tile, crop(x, region.sampling))
            torch.testing.assert_close(tiled_sigma, sigma)
            assert tile.shape[-2:] == plan.tile_hw
            assert all(
                entry[sampling.TAG] == region.tile_id
                for branch in branches
                for entry in branch
            )
    assert not model.wrappers


def test_raw_turbo_plan_reuse_and_cross_family_staleness(host):
    args = arguments()
    plan = args["tile_plan"]
    other = arguments(rank=5, batch=2, depth=40)
    assert plan == other["tile_plan"]
    sampling.sample(**{**other, "tile_plan": plan})
    for stale in (
        replace(plan, signature=replace(plan.signature, adapter_version=2)),
        replace(plan, signature=replace(plan.signature, layout="BCHW", rank=4)),
        anima_arguments()["tile_plan"],
        sdxl_arguments()["tile_plan"],
        arguments(hw=(16, 12))["tile_plan"],
    ):
        with pytest.raises(ValueError, match="Stale"):
            sampling.sample(**{**args, "tile_plan": stale})
    with pytest.raises(ValueError, match="Stale"):
        sampling.sample(**{**anima_arguments(), "tile_plan": plan})
    with pytest.raises(ValueError, match="fields or bounds"):
        validate_plan(replace(plan, regions=tuple(reversed(plan.regions))))
    validate_plan(
        plan,
        resolve_adapter(args["model"]).describe(
            args["model"], {"samples": torch.empty(2, 16, 1, 12, 16, device="meta")}
        ),
    )


@pytest.mark.parametrize(
    "hw,expected", [((10, 14), (6, 8)), ((14, 10), (8, 6)), ((10, 10), (6, 6))]
)
def test_aligned_views_and_overlap_rounding(host, hw, expected):
    from tiled_diffusion_ng.nodes import TilePlan, TileView

    args = arguments(hw=hw)
    plan = TilePlan.execute(args["model"], args["latent_image"], 1)[0]
    assert plan.tile_hw == expected and plan.effective_pixel_overlap == (16, 16)
    h, w = plan.pixel_hw
    image = torch.arange(2 * h * w * 3, dtype=torch.float32).reshape(2, h, w, 3)
    before = image.clone()
    views = TileView.execute(image, plan)[0]
    for source in range(2):
        for index, region in enumerate(plan.regions):
            assert all(value % 2 == 0 for value in region.sampling)
            x0, y0, x1, y1 = region.pixel_sampling
            torch.testing.assert_close(
                views[4 * source + index], image[source, y0:y1, x0:x1]
            )
    views.zero_()
    torch.testing.assert_close(image, before)


@pytest.mark.parametrize("hw,overlap", [((2, 8), 0), ((4, 4), 1), ((12, 16), 96)])
def test_impossible_four_view_plans(host, hw, overlap):
    model = Model("krea2")
    spec = resolve_adapter(model).describe(model, {"samples": torch.zeros(1, 16, *hw)})
    with pytest.raises(ValueError, match="four distinct"):
        make_plan(spec, overlap)


@pytest.mark.parametrize(
    "samples",
    [
        torch.zeros(1, 16, 2, 12, 16),
        torch.zeros(1, 4, 12, 16),
        torch.zeros(0, 16, 12, 16),
        torch.zeros(1, 16, 0, 16),
        torch.zeros(1, 16, 12, 16, dtype=torch.int64),
        torch.zeros(16, 12, 16),
        torch.zeros(1, 16, 12, 16).to_sparse(),
        None,
    ],
)
def test_target_layout_rejection_before_sampling(host, samples):
    args = arguments()
    args["latent_image"]["samples"] = samples
    with pytest.raises(ValueError, match="Krea2 requires"):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("downscale_ratio_spacial", 16),
        ("downscale_ratio_temporal", 1),
        ("type", "video"),
        ("latent_shapes", [(16, 1, 12, 16)]),
        ("spatial", torch.zeros(2, 2)),
        ("structured", {}),
    ],
)
def test_latent_metadata_rejection(host, field, value):
    args = arguments()
    args["latent_image"][field] = value
    with pytest.raises(ValueError, match=field):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("latent_channels", 4),
        ("latent_dimensions", 2),
        ("spacial_downscale_ratio", 16),
        ("temporal_downscale_ratio", 1),
    ],
)
def test_incompatible_wan_metadata(host, field, value):
    args = arguments()
    setattr(args["model"].format, field, value)
    with pytest.raises(ValueError, match="Krea2 Wan21 latent format metadata"):
        sampling.sample(**args)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("channels", 4, "architecture"),
        ("patch", 1, "patch geometry"),
        ("pe_embedder", object(), "EmbedND"),
        ("txtdim", 0, "text dimensions"),
        ("txtlayers", None, "text dimensions"),
        ("txtlayers", 1.5, "text dimensions"),
    ],
)
def test_native_architecture_guards(host, field, value, error):
    args = arguments()
    setattr(args["model"].model.diffusion_model, field, value)
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.common_calls


def test_family_backbone_and_concat_capabilities(host):
    args = arguments()
    args["model"].model.concat_keys = ("concat_latent_image",)
    with pytest.raises(ValueError, match="architecture"):
        sampling.sample(**args)
    args["model"].model.concat_keys = ()
    args["model"].model.diffusion_model = object()
    with pytest.raises(ValueError, match="architecture"):
        sampling.sample(**args)
    args["model"].format = SDXLFormat()
    with pytest.raises(ValueError, match="Unsupported model/latent family.*Krea2"):
        resolve_adapter(args["model"])
    unrelated = Model("krea2")
    unrelated.model = SimpleNamespace(diffusion_model=unrelated.model.diffusion_model)
    with pytest.raises(ValueError, match="Unsupported model/latent family"):
        resolve_adapter(unrelated)


@pytest.mark.parametrize("missing", ["wrapper", "network", "positions"])
def test_optional_krea2_apis_do_not_break_existing_families(host, monkeypatch, missing):
    from comfy import model_base
    from comfy.ldm.flux import layers

    args = arguments()
    if missing == "wrapper":
        monkeypatch.delattr(model_base, "Krea2")
    elif missing == "network":
        monkeypatch.setitem(sys.modules, "comfy.ldm.krea2.model", None)
    else:
        monkeypatch.delattr(layers, "EmbedND")
    with pytest.raises(
        ValueError, match="Unsupported model/latent family|native Krea2 APIs"
    ):
        sampling.sample(**args)
    sampling.sample(**sdxl_arguments(steps=1))
    sampling.sample(**anima_arguments(steps=1))
    assert len(host.common_calls) == 2


@pytest.mark.parametrize("conversion", ["calculate_input", "calculate_denoised"])
def test_native_const_conversion_required(host, conversion):
    args = arguments()
    setattr(args["model"].sampling, conversion, getattr(EPS(), conversion))
    with pytest.raises(ValueError, match="Krea2 model_sampling conversion.*CONST"):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize(
    "embedding",
    [
        torch.zeros(1, 2, 48, dtype=torch.int64),
        torch.zeros(1, 48),
        torch.zeros(0, 2, 48),
        torch.zeros(1, 0, 48),
        torch.zeros(1, 2, 48).to_sparse(),
        None,
    ],
)
def test_malformed_context_rejected_before_host(host, embedding):
    with pytest.raises(ValueError, match="floating rank-3 embedding"):
        sampling.sample(**arguments(positive=[[embedding, {}]]))
    assert not host.common_calls


@pytest.mark.parametrize("branch", ["positive", "negative", "local"])
def test_context_width_checked_on_every_effective_branch(host, branch):
    args = arguments(cfg=1)
    if branch == "local":
        args["local_positive"] = [cond(2)] * 3 + [cond(4, width=3)]
    else:
        args[branch] = cond(4, width=3)
    with pytest.raises(ValueError, match="requires 48 features.*got 3"):
        sampling.sample(**args)
    assert not host.common_calls and not host.discovered


@pytest.mark.parametrize("layers,width", [(12, 2560), (3, 8)])
def test_encoded_context_is_opaque_and_dimensions_come_from_model(host, layers, width):
    args = arguments(cfg=1)
    network = args["model"].model.diffusion_model
    network.txtlayers, network.txtdim = layers, width
    embedding = (
        torch.arange(layers * width * 2, dtype=torch.float32)
        .reshape(1, 2, -1)
        .transpose(0, 1)
    )
    attention, pooled = torch.tensor([[1, 0]]), torch.tensor([[17.0]])
    args["positive"] = [
        [embedding, {"attention_mask": attention, "pooled_output": pooled}]
    ]
    args["negative"] = cond(-1, width=layers * width)
    sampling.sample(**args)
    for item in args["model"].model.condition_calls:
        if item["cross_attn"] is embedding:
            assert (
                item["attention_mask"] is attention and item["pooled_output"] is pooled
            )
    for entry in host.prepared[0]:
        assert entry["model_conds"]["c_crossattn"].cond is embedding
        assert "attention_mask" not in entry["model_conds"]
    torch.testing.assert_close(
        embedding.flatten(), torch.arange(layers * width * 2, dtype=torch.float32)
    )


def test_scheduled_weighted_context_and_complete_local_replacement(host):
    args = arguments()
    baseline = sampling.sample(**args)["samples"]
    identical = sampling.sample(**{**args, "local_positive": [args["positive"]] * 4})[
        "samples"
    ]
    torch.testing.assert_close(identical, baseline)
    locals = [
        cond(2, strength=1) + cond(6, strength=3, start_percent=0, end_percent=1)
    ] * 4
    args["positive"] = cond(999, width=3)  # Replaced global width is not effective.
    replaced = sampling.sample(**{**args, "local_positive": locals})["samples"]
    expected = sampling.sample(**{**args, "positive": cond(5)})["samples"]
    torch.testing.assert_close(replaced, expected)
    inactive = cond(9, timestep_start=0.1, timestep_end=0) + cond(
        2, start_percent=0, end_percent=0.1
    )
    # All fixture evaluations have sigma .75: these entries are inactive.
    inactive_result = sampling.sample(**{**args, "positive": cond(2) + inactive})[
        "samples"
    ]
    torch.testing.assert_close(inactive_result, baseline)
    assert all(meta.get("tdng_tile_id") is None for _, meta in locals[0])


@pytest.mark.parametrize("branch", ["positive", "negative", "local_positive"])
def test_native_clip_schedule_is_preserved_and_intersected(host, branch):
    # encode_from_tokens_scheduled may bake CLIP LoRAs into multiple embeddings
    # without attaching runtime hooks. ComfyUI owns percent-to-sigma conversion.
    # https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/comfy/samplers.py#L859-L884
    condition = cond(2, clip_start_percent=0, clip_end_percent=0.5) + cond(
        7,
        clip_start_percent=0.5,
        clip_end_percent=1,
        start_percent=0.6,
        end_percent=0.8,
    )
    args = arguments()
    args[branch] = [condition] * 4 if branch == "local_positive" else condition
    captures = []
    args["model"].model_options["sampler_post_cfg_function"] = [
        lambda data: captures.append(data) or data["denoised"]
    ]

    def solve(evaluate, x, sigmas):
        for index, sigma in enumerate((0.75, 0.45, 0.3, 0.1)):
            x = evaluate(x, torch.tensor([sigma]), (0, index))
        return x

    host.dispatch["euler"].sampler_function = solve
    sampling.sample(**args)
    output_key = "uncond_denoised" if branch == "negative" else "cond_denoised"
    for data, (current, sigma), value in zip(
        captures, host.global_evaluations, (2, None, 7, None), strict=True
    ):
        expected = (
            torch.zeros_like(current)
            if value is None
            else current - sigma * (current * 0.25 + value)
        )
        torch.testing.assert_close(data[output_key], expected)
    prepared = host.prepared[1 if branch == "negative" else 0]
    assert [
        (entry["timestep_start"], entry["timestep_end"]) for entry in prepared[:2]
    ] == [(1, 0.5), (0.4, 1 - 0.8)]
    assert all("timestep_start" not in metadata for _, metadata in condition)


def test_denoised_fusion_before_global_cfg_with_per_image_sigmas(host):
    args = arguments(batch=2, dtype=torch.float64)
    values = (2, 4, 7, 11)
    args["local_positive"] = [cond(value) for value in values]
    captures, hooks = [], []

    def pre(data):
        assert data["conds_out"][0].shape == (2, 16, 1, 12, 16)
        hooks.append("pre")
        return data["conds_out"]

    def cfg(data):
        hooks.append("cfg")
        return data["uncond"] + data["cond_scale"] * (data["cond"] - data["uncond"])

    def post(data):
        hooks.append("post")
        captures.append(data)
        return data["denoised"]

    args["model"].model_options.update(
        sampler_pre_cfg_function=[pre],
        sampler_cfg_function=cfg,
        sampler_post_cfg_function=[post],
    )
    host.dispatch["euler"].sampler_function = lambda evaluate, x, sigmas: evaluate(
        x, torch.tensor([0.7, 0.2], dtype=x.dtype), (0, 0)
    )
    sampling.sample(**args)
    current, sigmas = host.global_evaluations[0]
    plan = args["tile_plan"]
    # Independent Gaussian and CONST oracle, with distinct values at every view.
    h, w = plan.tile_hw
    y = (torch.arange(h, dtype=torch.float64) - (h - 1) / 2) / h
    x = (torch.arange(w, dtype=torch.float64) - (w - 1) / 2) / w
    kernel = torch.exp(-50 * (y[:, None].square() + x[None, :].square()))
    numerator, denominator = (
        torch.zeros(plan.latent_hw, dtype=torch.float64) for _ in range(2)
    )
    for region, value in zip(plan.regions, values, strict=True):
        crop(numerator, region.sampling).add_(kernel * value)
        crop(denominator, region.sampling).add_(kernel)
    sigma = sigmas.reshape(2, 1, 1, 1, 1)
    positive = current - sigma * (current * 0.25 + numerator / denominator)
    negative = current - sigma * (current * 0.25 - 1)
    assert hooks == ["pre", "cfg", "post"] and len(host.tile_calls) == 4
    torch.testing.assert_close(captures[0]["cond_denoised"], positive)
    torch.testing.assert_close(captures[0]["uncond_denoised"], negative)
    torch.testing.assert_close(
        captures[0]["denoised"], negative + 4.5 * (positive - negative)
    )
    assert all(torch.equal(call[1], sigmas.repeat(2)) for call in host.krea2_calls)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_cfg_one_omits_negative_and_preserves_native_precision(host, dtype):
    output = sampling.sample(**arguments(cfg=1, dtype=dtype))
    assert (
        output["samples"].dtype == torch.float32
        and torch.isfinite(output["samples"]).all()
    )
    assert all(branches[1] is None for branches, *_ in host.tile_calls)


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("denoise", [0, 0.4, 1])
@pytest.mark.parametrize("mask_batch", [1, 2])
def test_refinement_masks_noise_indices_and_unchanged_inputs(
    host, rank, denoise, mask_batch
):
    args = arguments(rank=rank, batch=2, denoise=denoise)
    latent = args["latent_image"]
    latent["samples"].fill_(0.5)
    latent["batch_index"] = [7, 7]
    unmasked = sampling.sample(**args)["samples"]
    torch.testing.assert_close(unmasked[0], unmasked[1])
    mask = torch.zeros(mask_batch, 1, 1, 6, 8)
    mask[..., 4:] = 1
    if mask_batch == 2:
        mask[1] = 1
    before = mask.clone()
    latent.update(
        noise_mask=mask, downscale_ratio_spacial=8, downscale_ratio_temporal=4
    )
    output = sampling.sample(**args)
    assert (
        output["noise_mask"] is mask and output["batch_index"] is latent["batch_index"]
    )
    assert (
        "downscale_ratio_temporal" not in output
        and latent["downscale_ratio_temporal"] == 4
    )
    assert torch.all(output["samples"][0, ..., :6] == 0.5)
    if mask_batch == 2:
        torch.testing.assert_close(output["samples"][1], unmasked[1])
    if denoise == 0:
        assert torch.all(output["samples"] == 0.5) and not host.tile_calls
    torch.testing.assert_close(mask, before)
    assert latent["samples"].ndim == rank and torch.all(latent["samples"] == 0.5)


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, 12, 16),
        torch.ones(2, 1, 12, 16),
        torch.ones(2, 2, 1, 12, 16),
        torch.ones(2, 1, 2, 12, 16),
        torch.ones(3, 1, 1, 12, 16),
        torch.ones(0, 1, 1, 12, 16),
        torch.ones(2, 1, 1, 0, 16),
        torch.ones(2, 1, 1, 12, 16, dtype=torch.int64),
        torch.ones(2, 1, 1, 12, 16).to_sparse(),
        "mask",
    ],
)
def test_invalid_masks_rejected_before_host(host, mask):
    args = arguments(batch=2)
    args["latent_image"]["noise_mask"] = mask
    with pytest.raises(ValueError, match="Krea2 noise_mask.*B×1×1×H×W"):
        sampling.sample(**args)
    assert not host.common_calls and host.latest_clone is None


@pytest.mark.parametrize("branch", ["positive", "negative", "local", "replaced_global"])
@pytest.mark.parametrize("strength", [0, 1])
def test_controls_rejected_on_every_branch_before_discovery(host, branch, strength):
    args = arguments(cfg=1)
    control = ControlNet()
    control.strength = strength
    if branch in ("positive", "negative"):
        args[branch] = cond(1, control=control)
    else:
        args["local_positive"] = [cond(1)] * 4
        if branch == "local":
            args["local_positive"][-1] = cond(2, control=control)
        else:
            args["positive"] = cond(2, control=control)
    with pytest.raises(ValueError, match="Krea2 conditioning control.*ControlNet"):
        sampling.sample(**args)
    assert not host.common_calls and not host.discovered
    assert not control.pre_runs and not control.cleanups


def test_null_controls_removed_only_from_owned_metadata(host):
    source = cond(7, control=None, control_apply_to_uncond=True)
    before = source[0][1].copy()
    sampling.sample(
        **arguments(positive=source, negative=source, local_positive=[source] * 4)
    )
    assert source[0][1] == before
    assert all("control" not in entry for branch in host.prepared for entry in branch)
    assert not host.discovered


@pytest.mark.parametrize(
    "field,value",
    [
        ("area", (2, 2, 0, 0)),
        ("mask", torch.ones(1, 12, 16)),
        ("gligen", object()),
        ("hooks", object()),
        ("padding_mask", torch.zeros(12, 16)),
        ("concat_latent_image", torch.zeros(1, 16, 12, 16)),
        ("unknown_tensor", torch.zeros(2)),
    ],
)
def test_unsupported_raw_spatial_fields(host, field, value):
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments(positive=cond(1, **{field: value})))
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("control", ControlNet()),
        ("area", (2, 2, 0, 0)),
        ("mask", torch.ones(1, 12, 16)),
        ("gligen", object()),
        ("hooks", object()),
        ("concat_latent_image", object()),
        ("model_conds", {"c_concat": object()}),
    ],
)
def test_unsupported_prepared_spatial_fields(host, field, value):
    host.mutate_prepared = lambda branches: branches[0][0].update({field: value})
    with pytest.raises(ValueError, match="control|area|mask|gligen|hooks|concat"):
        sampling.sample(**arguments())
    assert not host.tile_calls


@pytest.mark.parametrize("field", ["patches", "patches_replace"])
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("lllite", [False, True])
def test_transformer_hooks_and_lllite_rejected(host, field, live, lllite):
    patch = (
        type(
            "AnimaLLLiteAttentionPatch", (), {"__module__": "comfy.ldm.anima.lllite"}
        )()
        if lllite
        else object()
    )
    args = arguments()

    def attach(options):
        options["transformer_options"][field] = (
            {"attn1_patch": [patch]}
            if field == "patches"
            else {"dit": {("single_block", 0): patch}}
        )

    if live:
        host.option_update = attach
    else:
        attach(args["model"].model_options)
    with pytest.raises(
        NotImplementedError if lllite else ValueError,
        match="Krea2 LLLite support remains TBD" if lllite else field,
    ):
        sampling.sample(**args)
    assert len(host.common_calls) == int(live) and not host.tile_calls
    assert not args["model"].wrappers


@pytest.mark.parametrize("route", ["attachment", "hook", "orphan_attachment"])
def test_tiled_lllite_hooks_and_attachments_have_krea2_error(host, route):
    from .test_anima_lllite import patched

    anima, *_ = patched()
    args = arguments()
    if route in ("attachment", "orphan_attachment"):
        args["model"].attachments = anima["model"].attachments.copy()
    if route in ("attachment", "hook"):
        args["model"].model_options = copy_containers(anima["model"].model_options)
    with pytest.raises(NotImplementedError, match="Krea2 LLLite support remains TBD"):
        sampling.sample(**args)
    assert not host.common_calls and not host.patch_models


def test_late_patch_injection_closes_invocation(host):
    args = arguments()

    def solver(evaluate, x, sigmas):
        x = evaluate(x, torch.tensor([0.6]), (0, 0))
        host.live_options["transformer_options"]["patches"] = {"post_input": [object()]}
        return evaluate(x, torch.tensor([0.4]), (0, 1))

    host.dispatch["euler"].sampler_function = solver
    with pytest.raises(ValueError, match="Krea2 transformer patches"):
        sampling.sample(**args)
    assert len(host.tile_calls) == 4
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)


def test_live_wrappers_registrations_shift_and_weight_loras_preserved(host):
    args = arguments(cfg=1)
    model = args["model"]
    model.sampling.shift = 3.125
    schedule = model.sampling
    token = object()
    model.attachments["nonspatial"] = token
    model.model_options["transformer_options"].update(
        patches={}, patches_replace={}, optimized_attention_override=token
    )
    calls = []

    def outer(executor, base, branches, x, sigma, options):
        calls.append(("outer", x.shape[-2:]))
        return executor(base, branches, x, sigma, {**options, "marker": token})

    def inner(executor, base, branches, x, sigma, options):
        calls.append(("inner", x.shape[-2:]))
        assert options["marker"] is token and options["live_marker"] is token
        return executor(base, branches, x, sigma, options)

    def native(executor, base, x, sigma, conditions, transformer):
        assert (
            transformer["sigmas"] is sigma
            and transformer["optimized_attention_override"] is token
        )
        return executor(base, x, sigma, conditions, transformer)

    def forward(executor, x, sigma, context, mask, refs, transformer, **kwargs):
        assert transformer["sigmas"] is sigma
        return executor(x, sigma, context, mask, refs, transformer, **kwargs)

    model.add_wrapper_with_key("calc_cond_batch", "outer", outer)
    model.add_wrapper_with_key("apply_model", "native", native)
    model.add_wrapper_with_key("diffusion_model", "native", forward)
    common = host.nodes.common_ksampler

    def common_with_inner(clone, *args, **kwargs):
        clone.add_wrapper_with_key("calc_cond_batch", "inner", inner)
        assert clone.sampling is schedule and schedule.shift == 3.125
        return common(clone, *args, **kwargs)

    host.nodes.common_ksampler = common_with_inner
    host.option_update = lambda options: options.update(
        live_marker=token, disable_cfg1_optimization=True
    )
    host.samplers.KSampler.SAMPLERS.append("registered")
    host.samplers.KSampler.SCHEDULERS.append("custom_schedule")
    host.dispatch["registered"] = SimpleNamespace(sampler_function=host.solve)
    host.samplers.SCHEDULER_HANDLERS["custom_schedule"] = SimpleNamespace(
        handler=lambda steps, denoise: [0.6]
    )
    sampling.sample(
        **{**args, "sampler_name": "registered", "scheduler": "custom_schedule"}
    )
    assert calls == [("outer", (12, 16)), *[("inner", (8, 10))] * 4] * 3
    assert all(branches[1] is not None for branches, *_ in host.tile_calls)
    assert host.latest_clone.sampling is schedule and schedule.shift == 3.125
    assert host.latest_clone.patches["lora"][0] is model.patches["lora"][0]
    assert host.latest_clone.attachments["nonspatial"] is token
    assert "live_marker" not in model.model_options
    assert host.latest_clone.get_wrappers("calc_cond_batch", "outer") == [outer]
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)


@pytest.mark.parametrize(
    "failure",
    [None, "prepare", "finalize", "clone", "host_prepare", "tile", "cancel", "cleanup"],
)
@pytest.mark.parametrize("hw", [(12, 16), (13, 15)])
def test_context_state_released_on_every_exit_and_subsequent_reuse(
    host, monkeypatch, failure, hw
):
    args = arguments(
        hw=hw,
        positive=cond(
            2,
            reference_latents=[torch.ones(1, 16, 1, 5, 7)],
            reference_latents_method="index",
        ),
    )
    adapter = resolve_adapter(args["model"])
    create, evaluation_type = adapter.create_sampling_context, sampling.TileEvaluation
    contexts, evaluations = [], []

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate preparation failure")

    def create_context(plan):
        context = create(plan)
        contexts.append(context)
        phase = {"prepare": "prepare_pair", "finalize": "finalize_preparation"}.get(
            failure
        )
        if phase:
            monkeypatch.setattr(context, phase, fail)
        if failure == "cleanup":
            close = context.close

            def fail_close():
                close()
                raise RuntimeError("deliberate cleanup failure")

            monkeypatch.setattr(context, "close", fail_close)
        return context

    def create_evaluation(*args):
        evaluation = evaluation_type(*args)
        evaluations.append(evaluation)
        return evaluation

    with monkeypatch.context() as patch:
        patch.setattr(adapter, "create_sampling_context", create_context)
        patch.setattr(sampling, "TileEvaluation", create_evaluation)
        if failure == "clone":
            patch.setattr(args["model"], "clone", fail)
        if failure == "host_prepare":
            patch.setattr(args["model"].model, "extra_conds", fail)
        if failure == "tile":
            host.fail_tile = 3
        if failure == "cancel":
            host.interrupt_after = 3
        if failure is None:
            sampling.sample(**args)
        else:
            with pytest.raises((RuntimeError, InterruptedError)):
                sampling.sample(**args)
    assert contexts and all(
        context.plan is context.width is context.default_method is None
        for context in contexts
    )
    assert all(
        evaluation.plan is evaluation.adapter is evaluation.context is None
        and not evaluation.weights
        for evaluation in evaluations
    )
    assert not args["model"].wrappers
    assert PREPARATION not in args["positive"][0][1]
    if host.latest_clone is not None:
        for kind in ("calc_cond_batch", "predict_noise"):
            assert not host.latest_clone.get_wrappers(kind, sampling.WRAPPER_KEY)
    context = contexts[0]
    type(context).close(context)
    with pytest.raises(RuntimeError, match="closed"):
        type(context).prepare_pair(context, [], [], args["tile_plan"].regions[0])
    host.fail_tile = host.interrupt_after = None
    sampling.sample(**args)


def test_anima_krea2_anima_repeatability(host):
    args = anima_arguments(steps=1)
    before = sampling.sample(**args)["samples"]
    sampling.sample(**arguments())
    after = sampling.sample(**args)["samples"]
    torch.testing.assert_close(before, after)


def test_public_node_local_execution_list_contract(host):
    from tiled_diffusion_ng.nodes import TileSampler

    args = arguments()
    ordinary = {key: [value] for key, value in args.items()}
    locals = [cond(value) for value in (2, 3, 5, 7)]
    output = TileSampler.execute(**ordinary, local_positive=locals)
    assert output[0]["samples"].shape == (1, 16, 1, 12, 16)
    assert len(host.common_calls) == 1
    assert [
        embedding.mean().item() for embedding, _ in host.common_calls[0]["positive"]
    ] == [2, 3, 5, 7]
    for local in (locals[:3], locals + [cond(9)], [locals]):
        with pytest.raises(ValueError, match="four complete"):
            TileSampler.execute(**ordinary, local_positive=local)
    for name, values in ordinary.items():
        with pytest.raises(ValueError, match=name):
            TileSampler.execute(**{**ordinary, name: values * 2})
