# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Offline CPU contracts, not evidence of real-host Anima or LLLite inference.

Inspected source: ComfyUI 944386c233e02eaf877b1c8d5d513fb3d3a4d5e3.
Real-host generation/refinement, portrait/landscape, local prompts, LoRAs and
seam acceptance remain pending separately for Base, Aesthetic, Turbo and 2.9B.
Tiled LLLite contracts live in test_anima_lllite.py; native chaining is deferred.
"""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.adapters.anima import AnimaAdapter
from tiled_diffusion_ng.geometry import crop, make_plan, validate_plan

from .host import EPS, ControlNet, Model, SDXLFormat, cond


def arguments(*, rank=4, batch=1, hw=(12, 16), depth=28, dtype=torch.float32, **kwargs):
    model = Model("anima", num_blocks=depth)
    shape = (batch, 16) + ((1,) if rank == 5 else ()) + hw
    latent = {"samples": torch.zeros(shape, dtype=dtype), "note": "retained"}
    result = {
        "model": model,
        "seed": 321,
        "steps": 2,
        "cfg": 4.0,
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
    "rank,batch,depth", [(4, 1, 28), (4, 2, 40), (5, 1, 40), (5, 2, 28)]
)
def test_native_layout_depth_and_one_trajectory(host, rank, batch, depth):
    args = arguments(rank=rank, batch=batch, depth=depth)
    model, latent, plan = args["model"], args["latent_image"], args["tile_plan"]
    adapter = resolve_adapter(model)
    assert isinstance(adapter, AnimaAdapter)
    assert (plan.signature.layout, plan.signature.rank, plan.signature.channels) == (
        "BCTHW",
        5,
        16,
    )
    assert plan.signature.alignment == plan.signature.minimum == (2, 2)
    assert plan.signature.scale == (8, 8)
    native_network = model.model.diffusion_model
    rope = native_network.pos_embedder
    output = sampling.sample(**args)
    assert output["samples"].shape == (batch, 16, 1, 12, 16)
    assert latent["samples"].ndim == rank and not latent["samples"].any()
    assert output["note"] == "retained"
    assert len(host.common_calls) == 1 and len(host.global_evaluations) == 6
    assert len(host.tile_calls) == 24
    assert host.common_calls[0]["latent"] is latent
    assert host.latest_clone.model.diffusion_model is native_network
    assert native_network.pos_embedder is rope
    assert native_network.num_blocks == depth
    assert native_network.rope_h_extrapolation_ratio == 4
    for evaluation, (current_x, sigma) in enumerate(host.global_evaluations):
        for index, region in enumerate(plan.regions):
            branches, view, current_sigma, _ = host.tile_calls[evaluation * 4 + index]
            torch.testing.assert_close(view, crop(current_x, region.sampling))
            torch.testing.assert_close(current_sigma, sigma)
            assert view.shape == (batch, 16, 1, *plan.tile_hw)
            assert all(
                entry[sampling.TAG] == region.tile_id
                for branch in branches
                for entry in branch
            )
    assert not model.wrappers


def test_layout_batch_depth_independent_plan_identity_and_staleness(host):
    args = arguments()
    plan = args["tile_plan"]
    other = arguments(rank=5, batch=2, depth=40)
    assert plan == other["tile_plan"]
    other["tile_plan"] = plan
    sampling.sample(**other)
    for stale in (
        replace(plan, signature=replace(plan.signature, adapter_version=2)),
        replace(plan, signature=replace(plan.signature, layout="BCHW", rank=4)),
        replace(plan, signature=replace(plan.signature, adapter_id="sdxl")),
        arguments(hw=(16, 12))["tile_plan"],
    ):
        with pytest.raises(ValueError, match="Stale"):
            sampling.sample(**{**args, "tile_plan": stale})
    with pytest.raises(ValueError, match="fields or bounds"):
        validate_plan(replace(plan, regions=tuple(reversed(plan.regions))))
    # Geometry reads metadata, never values or the checkpoint weights.
    spec = resolve_adapter(args["model"]).describe(
        args["model"], {"samples": torch.empty(2, 16, 1, 12, 16, device="meta")}
    )
    validate_plan(plan, spec)


@pytest.mark.parametrize("hw", [(10, 14), (14, 10)])
def test_aligned_rectangles_effective_overlap_and_exact_tileview_order(host, hw):
    from tiled_diffusion_ng.nodes import TilePlan, TileView

    args = arguments(hw=hw)
    plan = TilePlan.execute(args["model"], args["latent_image"], 1)[0]
    assert plan.tile_hw == tuple(n // 2 + 1 for n in hw)
    assert plan.effective_pixel_overlap == (16, 16)
    assert plan.tile_ids == ("TL", "TR", "BR", "BL")
    h, w = plan.pixel_hw
    image = torch.arange(2 * h * w * 3, dtype=torch.float32).reshape(2, h, w, 3)
    before = image.clone()
    views = TileView.execute(image, plan)[0]
    for source in range(2):
        for index, region in enumerate(plan.regions):
            assert all(coordinate % 2 == 0 for coordinate in region.sampling)
            x0, y0, x1, y1 = region.pixel_sampling
            torch.testing.assert_close(
                views[4 * source + index], image[source, y0:y1, x0:x1]
            )
    assert views.shape == (8, *plan.tile_pixel_hw, 3)
    views.zero_()
    assert image.equal(before)


@pytest.mark.parametrize("hw,overlap", [((2, 8), 0), ((4, 4), 1), ((12, 16), 96)])
def test_impossible_four_view_geometry(host, hw, overlap):
    from tiled_diffusion_ng.nodes import TilePlan

    with pytest.raises(ValueError, match="four distinct"):
        TilePlan.execute(Model("anima"), {"samples": torch.zeros(1, 16, *hw)}, overlap)


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
def test_invalid_latents(host, samples):
    model = Model("anima")
    with pytest.raises(ValueError, match="Anima requires"):
        resolve_adapter(model).describe(model, {"samples": samples})


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
    with pytest.raises(ValueError, match="Wan21 latent format metadata"):
        sampling.sample(**args)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("in_channels", 17, "architecture"),
        ("out_channels", 4, "architecture"),
        ("patch_spatial", 1, "patch geometry"),
        ("patch_temporal", 2, "patch geometry"),
        ("pos_emb_cls", "sincos", "RoPE"),
        ("pos_embedder", object(), "RoPE"),
        ("extra_per_block_abs_pos_emb", True, "absolute positions"),
    ],
)
def test_incompatible_native_architecture(host, field, value, error):
    args = arguments()
    setattr(args["model"].model.diffusion_model, field, value)
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)


def test_family_and_native_backbone_required(host, monkeypatch):
    from comfy import model_base

    args = arguments()
    args["model"].model.diffusion_model = object()
    with pytest.raises(ValueError, match="native Anima architecture"):
        sampling.sample(**args)
    args["model"].format = SDXLFormat()
    with pytest.raises(ValueError, match="SDXL base or native Anima"):
        resolve_adapter(args["model"])
    monkeypatch.delattr(model_base, "Anima")
    assert resolve_adapter(Model()).describe(
        Model(), {"samples": torch.zeros(1, 4, 8, 8)}
    )
    with pytest.raises(ValueError, match="Unsupported model"):
        resolve_adapter(Model("anima"))


@pytest.mark.parametrize("method", ["calculate_input", "calculate_denoised"])
def test_only_native_const_conversion_is_accepted(host, method):
    args = arguments()
    setattr(args["model"].sampling, method, getattr(EPS(), method))
    with pytest.raises(ValueError, match="native CONST"):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize("deferred", [False, True])
def test_host_owns_text_preparation_and_preserves_metadata(host, deferred):
    host.defer_anima_text = deferred
    metadata = {
        "t5xxl_ids": torch.tensor([1, 7, 9]),
        "t5xxl_weights": torch.tensor([0.5, 2.0, 1.0]),
        "attention_mask": torch.tensor([[1, 1]]),
        "pooled_output": torch.tensor([[3.0]]),
        "strength": 0.6,
        "start_percent": 0.1,
        "end_percent": 0.9,
        "timestep_start": 0.95,
        "timestep_end": 0.05,
        "control": None,
    }
    positive = cond(3, **metadata)
    before = copy.deepcopy(positive)
    args = arguments(positive=positive, steps=1)
    sampling.sample(**args)
    model = args["model"].model
    assert len(model.condition_calls) == 8
    for entry in host.common_calls[0]["positive"]:
        assert entry[0] is positive[0][0]
        assert "control" not in entry[1]
        for key, value in metadata.items():
            if key != "control":
                assert entry[1][key] is value
    for entry in host.prepared[0]:
        assert entry["timestep_start"] == 0.9
        assert entry["timestep_end"] == pytest.approx(0.1)
    for branches, _, _, _ in host.tile_calls:
        entry = branches[0][0]
        assert "control" not in entry
        for key, value in metadata.items():
            if key not in {"timestep_start", "timestep_end", "control"}:
                assert entry[key] is value
        fields = set(entry["model_conds"])
        assert fields == (
            {"c_crossattn", "t5xxl_ids", "t5xxl_weights"}
            if deferred
            else {"c_crossattn"}
        )
    assert len(model.diffusion_model.text_calls) == (12 if deferred else 4)
    for embedding, ids, weights in model.diffusion_model.text_calls:
        assert embedding is positive[0][0]
        torch.testing.assert_close(ids, metadata["t5xxl_ids"].unsqueeze(0))
        torch.testing.assert_close(weights, metadata["t5xxl_weights"].reshape(1, 3, 1))
    torch.testing.assert_close(before[0][0], positive[0][0])
    assert before[0][1].keys() == positive[0][1].keys()
    assert positive[0][1]["control"] is None
    for key, value in metadata.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(before[0][1][key], value)


def test_local_prompts_replace_whole_positive_and_use_execution_lists(host):
    from tiled_diffusion_ng.nodes import TileSampler

    args = arguments(steps=1)
    baseline = sampling.sample(**args)["samples"]
    same = sampling.sample(**args, local_positive=[args["positive"]] * 4)["samples"]
    torch.testing.assert_close(baseline, same)
    locals = [cond(i + 10) + cond(i + 20, strength=0.5) for i in range(4)]
    wrapped = {key: [value] for key, value in args.items()}
    start = len(host.tile_calls)
    result = TileSampler.execute(**wrapped, local_positive=locals)[0]
    assert not result["samples"].equal(baseline)
    for index, call in enumerate(host.tile_calls[start : start + 4]):
        assert [entry["cross_attn"].mean() for entry in call[0][0]] == [
            index + 10,
            index + 20,
        ]
        assert call[0][1][0]["cross_attn"].mean() == -1
    with pytest.raises(ValueError, match="CONDITIONING"):
        TileSampler.execute(**wrapped, local_positive=cond(1) * 4)


def test_native_strength_and_timestep_selection(host):
    args = arguments(
        steps=1,
        positive=cond(2, strength=0.5, timestep_start=0.8, timestep_end=0.3)
        + cond(8, strength=1.5, timestep_start=0.8, timestep_end=0.3)
        + cond(99, start_percent=0.9),
    )
    actual = sampling.sample(**args)["samples"]
    expected = sampling.sample(**{**args, "positive": cond(6.5)})["samples"]
    torch.testing.assert_close(actual, expected)


def test_custom_global_cfg_hooks_see_fused_branches_once(host):
    args = arguments(steps=1)
    baseline = sampling.sample(**args)["samples"]
    calls = []

    def pre(data):
        assert all(branch.shape == (1, 16, 1, 12, 16) for branch in data["conds_out"])
        calls.append("pre")
        return data["conds_out"]

    def cfg(data):
        assert data["input"].shape == (1, 16, 1, 12, 16)
        torch.testing.assert_close(data["cond"], data["input"] - data["cond_denoised"])
        torch.testing.assert_close(
            data["uncond"], data["input"] - data["uncond_denoised"]
        )
        calls.append("cfg")
        return data["uncond"] + data["cond_scale"] * (data["cond"] - data["uncond"])

    def post(data):
        assert data["denoised"].shape == (1, 16, 1, 12, 16)
        calls.append("post")
        return data["denoised"]

    args["model"].model_options.update(
        sampler_pre_cfg_function=[pre],
        sampler_cfg_function=cfg,
        sampler_post_cfg_function=[post],
    )
    result = sampling.sample(**args)["samples"]
    torch.testing.assert_close(result, baseline)
    assert calls == ["pre", "cfg", "post"] * 3


def test_flow_fuses_denoised_before_global_cfg_with_per_batch_sigmas(host):
    args = arguments(batch=2, dtype=torch.float64, steps=1)
    args["local_positive"] = [cond(value) for value in (2, 4, 7, 11)]
    captures = []
    args["model"].model_options["sampler_post_cfg_function"] = [
        lambda data: captures.append(data) or data["denoised"]
    ]

    def solver(evaluate, x, sigmas):
        return evaluate(x, torch.tensor([0.7, 0.2], dtype=x.dtype), (0, 0))

    host.dispatch["euler"].sampler_function = solver
    sampling.sample(**args)
    current_x, sigmas = host.global_evaluations[0]
    sigma = sigmas.reshape(2, 1, 1, 1, 1)
    plan = args["tile_plan"]
    # Independent Gaussian blend of distinct tile constants, then CONST.
    h, w = plan.tile_hw
    y = (torch.arange(h, dtype=torch.float64) - (h - 1) / 2) / h
    x = (torch.arange(w, dtype=torch.float64) - (w - 1) / 2) / w
    kernel = torch.exp(-50 * (y[:, None] ** 2 + x[None, :] ** 2))
    numerator = torch.zeros(plan.latent_hw, dtype=torch.float64)
    denominator = torch.zeros_like(numerator)
    for region, value in zip(plan.regions, (2, 4, 7, 11), strict=True):
        crop(numerator, region.sampling).add_(kernel * value)
        crop(denominator, region.sampling).add_(kernel)
    p = current_x - sigma * (current_x * 0.25 + numerator / denominator)
    n = current_x - sigma * (current_x * 0.25 - 1)
    assert len(captures) == 1
    torch.testing.assert_close(captures[0]["cond_denoised"], p)
    torch.testing.assert_close(captures[0]["uncond_denoised"], n)
    torch.testing.assert_close(captures[0]["denoised"], n + args["cfg"] * (p - n))
    assert all(torch.equal(call[1], sigmas) for call in host.native_calls)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_turbo_cfg1_omits_negative_and_preserves_host_output_dtype(host, dtype):
    args = arguments(dtype=dtype, cfg=1.0, steps=1)
    output = sampling.sample(**args)
    assert output["samples"].dtype == torch.float32
    assert torch.isfinite(output["samples"]).all()
    assert all(call[0][1] is None for call in host.tile_calls)
    assert len(host.native_calls) == len(host.tile_calls)


@pytest.mark.parametrize("rank", [4, 5])
def test_refinement_masks_batch_indices_and_zero_denoise(host, rank):
    args = arguments(rank=rank, batch=2)
    latent = args["latent_image"]
    latent["samples"].fill_(0.5)
    mask = torch.zeros(2, 1, 1, 12, 16)
    mask[..., 8:] = 1
    latent.update(
        noise_mask=mask,
        batch_index=[7, 7],
        downscale_ratio_spacial=8,
        downscale_ratio_temporal=4,
    )
    output = sampling.sample(**args)
    assert output["noise_mask"] is mask
    assert output["batch_index"] is latent["batch_index"]
    assert "downscale_ratio_temporal" not in output
    assert latent["downscale_ratio_temporal"] == 4
    assert latent["samples"].ndim == rank and torch.all(latent["samples"] == 0.5)
    assert torch.all(output["samples"][..., :8] == 0.5)
    torch.testing.assert_close(output["samples"][0], output["samples"][1])
    zero = sampling.sample(**{**args, "denoise": 0.0})
    assert zero["samples"].shape == (2, 16, 1, 12, 16)
    assert torch.all(zero["samples"] == 0.5)


@pytest.mark.parametrize("rank", [4, 5])
def test_batched_noise_masks_require_native_5d_layout(host, rank):
    args = arguments(rank=rank, batch=2)
    mask = torch.stack((torch.zeros(12, 16), torch.ones(12, 16)))
    args["latent_image"]["noise_mask"] = mask
    with pytest.raises(ValueError, match="Anima noise_mask.*B×1×1×H×W"):
        sampling.sample(**args)
    assert not host.common_calls and host.latest_clone is None
    assert args["latent_image"]["noise_mask"] is mask
    assert not mask[0].any() and mask[1].all()


@pytest.mark.parametrize(
    "mask",
    [
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
def test_invalid_noise_masks_rejected_before_host_sampling(host, mask):
    args = arguments(batch=2)
    args["latent_image"]["noise_mask"] = mask
    with pytest.raises(ValueError, match="Anima noise_mask"):
        sampling.sample(**args)
    assert not host.common_calls and host.latest_clone is None


@pytest.mark.parametrize("rank", [4, 5])
@pytest.mark.parametrize("mask_batch", [1, 2])
def test_native_noise_masks_preserve_protected_images(host, rank, mask_batch):
    args = arguments(rank=rank, batch=2, steps=1)
    latent = args["latent_image"]
    latent["samples"].fill_(0.5)
    unmasked = sampling.sample(**args)["samples"]
    # Exercise native resizing as well as an explicit shared/per-image batch.
    mask = torch.zeros(mask_batch, 1, 1, 6, 8)
    if mask_batch == 2:
        mask[1] = 1
    before = mask.clone()
    latent["noise_mask"] = mask
    output = sampling.sample(**args)
    assert output["noise_mask"] is mask
    assert host.common_calls[-1]["latent"] is latent
    assert torch.all(output["samples"][0] == 0.5)
    if mask_batch == 2:
        torch.testing.assert_close(output["samples"][1], unmasked[1])
        assert not torch.all(output["samples"][1] == 0.5)
    else:
        assert torch.all(output["samples"][1] == 0.5)
    torch.testing.assert_close(mask, before)
    assert latent["samples"].ndim == rank and torch.all(latent["samples"] == 0.5)


@pytest.mark.parametrize("branch", ["positive", "negative", "local"])
def test_null_controls_removed_before_host_discovery_without_input_mutation(
    host, branch
):
    args = arguments(steps=1)
    source = cond(7, control=None, control_apply_to_uncond=True) + cond(8)
    metadata = source[0][1].copy()
    if branch == "local":
        args["local_positive"] = [source] * 4
    else:
        args[branch] = source
    sampling.sample(**args)
    assert source[0][1].keys() == metadata.keys()
    assert all(source[0][1][key] is value for key, value in metadata.items())
    for name in ("positive", "negative"):
        assert all("control" not in entry[1] for entry in host.common_calls[0][name])
    assert all("control" not in entry for branch in host.prepared for entry in branch)
    assert not host.discovered


@pytest.mark.parametrize("branch", ["positive", "negative", "local"])
def test_condition_controls_are_rejected_before_host_discovery(host, branch):
    control = ControlNet()
    args = arguments()
    if branch == "local":
        args["local_positive"] = [cond(1)] * 3 + [cond(2, control=control)]
    else:
        args[branch] = cond(1, control=control)
    with pytest.raises(ValueError, match="Anima conditioning control.*incompatible"):
        sampling.sample(**args)
    assert not host.common_calls and not host.discovered
    assert not control.pre_runs and not control.cleanups


@pytest.mark.parametrize(
    "field,value",
    [
        ("area", (2, 2, 0, 0)),
        ("mask", torch.ones(1, 12, 16)),
        ("gligen", object()),
        ("hooks", object()),
        ("padding_mask", torch.zeros(12, 16)),
        ("concat_latent_image", torch.zeros(1, 16, 12, 16)),
        ("reference_latents", []),
        ("unknown_tensor", torch.zeros(2, 2)),
    ],
)
def test_spatial_conditioning_rejection(host, field, value):
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments(positive=cond(1, **{field: value})))
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("control", ControlNet()),
        ("area", (2, 2, 0, 0)),
        ("model_conds", {"c_concat": object()}),
    ],
)
def test_prepared_spatial_conditions_are_rejected(host, field, value):
    host.mutate_prepared = lambda branches: branches[0][0].update({field: value})
    with pytest.raises(ValueError, match="control|area|c_concat"):
        sampling.sample(**arguments())
    assert not host.tile_calls


def native_lllite(name):
    return type(name, (), {"__module__": "comfy.ldm.anima.lllite"})()


@pytest.mark.parametrize(
    "slot,name",
    [
        ("post_input", "AnimaLLLitePatch"),
        ("attn1_patch", "AnimaLLLiteAttentionPatch"),
        ("attn2_patch", "AnimaLLLiteAttentionPatch"),
        ("mlp_patch", "AnimaLLLiteMLPPatch"),
    ],
)
@pytest.mark.parametrize("live", [False, True])
def test_native_lllite_hooks_have_explicit_deferred_guard(host, slot, name, live):
    args = arguments()
    patch = native_lllite(name)

    def attach(options):
        options["transformer_options"]["patches"] = {slot: [patch]}

    if live:
        host.option_update = attach
    else:
        attach(args["model"].model_options)
    with pytest.raises(
        NotImplementedError, match="AnimaLLLiteApply chaining is unsupported"
    ):
        sampling.sample(**args)
    assert len(host.common_calls) == int(live)
    assert not host.tile_calls
    assert not args["model"].wrappers


@pytest.mark.parametrize(
    "field,patches",
    [
        ("patches", {"post_input": [object()]}),
        ("patches", {"attn1_patch": [object()]}),
        ("patches", {"mlp_patch": [object()]}),
        ("patches", {"unknown": [object()]}),
        ("patches_replace", {"dit": {("double_block", 0): object()}}),
    ],
)
def test_other_transformer_patches_are_not_silently_discarded(host, field, patches):
    args = arguments()
    args["model"].model_options["transformer_options"][field] = patches
    with pytest.raises(ValueError, match=field):
        sampling.sample(**args)
    assert not host.common_calls


def test_patch_added_after_first_evaluation_is_rejected_and_closed(host):
    args = arguments(steps=1)

    def solver(evaluate, x, sigmas):
        x = evaluate(x, torch.tensor([0.6]), (0, 0))
        live = {
            **host.live_options,
            "transformer_options": {
                **host.live_options["transformer_options"],
                "patches": {"post_input": [object()]},
            },
        }
        return evaluate(x, torch.tensor([0.4]), (0, 1), live)

    host.dispatch["euler"].sampler_function = solver
    with pytest.raises(ValueError, match="patches.post_input"):
        sampling.sample(**args)
    assert len(host.tile_calls) == 4
    for kind in ("predict_noise", "calc_cond_batch"):
        assert not host.latest_clone.get_wrappers(kind, sampling.WRAPPER_KEY)


def test_prediction_guard_uses_guider_wrappers_with_replaced_live_options(host):
    args = arguments()
    overrides = []

    def solver(evaluate, x, sigmas):
        return evaluate(
            x,
            torch.tensor([0.5]),
            (0, 0),
            {
                "transformer_options": {},
                "sampler_calc_cond_batch_function": lambda data: overrides.append(data),
            },
        )

    host.dispatch["euler"].sampler_function = solver
    with pytest.raises(ValueError, match="sampler_calc_cond_batch_function"):
        sampling.sample(**args)
    assert not overrides and not host.tile_calls
    assert not host.latest_clone.get_wrappers("predict_noise", sampling.WRAPPER_KEY)


def test_clone_live_options_continuations_sigmas_and_auxiliary_discovery(host):
    args = arguments(steps=1, cfg=1.0)
    model = args["model"]
    auxiliary = Model("anima")
    auxiliary.additional_models["nested"] = [Model()]
    model.additional_models["compatible_auxiliary"] = [auxiliary]
    attachment, optimized_attention = object(), object()
    model.attachments["nonspatial"] = attachment
    model.model_options["transformer_options"].update(
        optimized_attention_override=optimized_attention,
        patches={},
        patches_replace={},
    )
    wrapper_calls = []

    def outer(executor, model, conds, x, sigma, options):
        wrapper_calls.append(("outer", x.shape[-2:]))
        return executor(model, conds, x, sigma, {**options, "outer_marker": attachment})

    def inner(executor, model, conds, x, sigma, options):
        assert (
            options["outer_marker"] is attachment
            and options["live_marker"] is attachment
        )
        wrapper_calls.append(("inner", x.shape[-2:]))
        return executor(model, conds, x, sigma, options)

    def native(executor, base, x, sigma, conditions, transformer):
        assert transformer["sigmas"] is sigma
        assert transformer["optimized_attention_override"] is optimized_attention
        return executor(base, x, sigma, conditions, transformer)

    def native_forward(executor, x, sigma, *args, transformer_options):
        assert transformer_options["sigmas"] is sigma
        assert (
            transformer_options["optimized_attention_override"] is optimized_attention
        )
        return executor(x, sigma, *args, transformer_options=transformer_options)

    model.add_wrapper_with_key("calc_cond_batch", "outer", outer)
    model.add_wrapper_with_key("apply_model", "native", native)
    model.add_wrapper_with_key("diffusion_model", "native", native_forward)
    common = host.nodes.common_ksampler

    def common_with_inner(clone, *a, **kw):
        clone.add_wrapper_with_key("calc_cond_batch", "inner", inner)
        return common(clone, *a, **kw)

    host.nodes.common_ksampler = common_with_inner
    host.option_update = lambda options: options.update(
        live_marker=attachment, disable_cfg1_optimization=True
    )
    # Registrations remain live for this family as well as SDXL.
    host.samplers.KSampler.SAMPLERS.append("anima_registered")
    host.samplers.KSampler.SCHEDULERS.append("anima_schedule")
    host.dispatch["anima_registered"] = SimpleNamespace(sampler_function=host.solve)
    host.samplers.SCHEDULER_HANDLERS["anima_schedule"] = SimpleNamespace(
        handler=lambda steps, denoise: [0.6]
    )
    sampling.sample(
        **{**args, "sampler_name": "anima_registered", "scheduler": "anima_schedule"}
    )
    assert wrapper_calls == [("outer", (12, 16)), *[("inner", (8, 10))] * 4] * 3
    assert all(call[0][1] is not None for call in host.tile_calls)
    assert len(host.discovered_models) == 2
    assert host.discovered_models[0] is not auxiliary
    assert host.discovered_models[0].model is auxiliary.model
    assert host.latest_clone.attachments["nonspatial"] is attachment
    assert host.latest_clone.patches["lora"][0] is model.patches["lora"][0]
    assert host.latest_clone.model_options is not model.model_options
    assert "live_marker" not in model.model_options
    assert "wrappers" not in model.model_options["transformer_options"]
    assert host.latest_clone.get_wrappers("calc_cond_batch", "outer") == [outer]
    assert host.latest_clone.get_wrappers("calc_cond_batch", "inner") == [inner]
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)


@pytest.mark.parametrize(
    "failure", [None, "prepare", "finalize", "clone", "tile", "cancel", "cleanup"]
)
def test_invocation_context_and_fusion_state_always_released(
    host, monkeypatch, failure
):
    args = arguments()
    adapter = resolve_adapter(args["model"])
    create = adapter.create_sampling_context
    contexts, evaluations = [], []
    evaluation_type = sampling.TileEvaluation

    def create_context(plan):
        context = create(plan)
        contexts.append(context)
        phase = {"prepare": "prepare_pair", "finalize": "finalize_preparation"}.get(
            failure
        )
        if phase:

            def fail(*args):
                raise RuntimeError("context failure")

            monkeypatch.setattr(context, phase, fail)
        if failure == "cleanup":
            close = context.close

            def fail_close():
                close()
                raise RuntimeError("cleanup failure")

            monkeypatch.setattr(context, "close", fail_close)
        return context

    def create_evaluation(*args):
        evaluation = evaluation_type(*args)
        evaluations.append(evaluation)
        return evaluation

    monkeypatch.setattr(adapter, "create_sampling_context", create_context)
    monkeypatch.setattr(sampling, "TileEvaluation", create_evaluation)
    if failure == "clone":

        def fail_clone():
            raise RuntimeError("clone failure")

        monkeypatch.setattr(args["model"], "clone", fail_clone)
    if failure == "tile":
        host.fail_tile = 3
    if failure == "cancel":
        host.interrupt_after = 3
    if failure is None:
        first = sampling.sample(**args)["samples"]
        sampling.sample(
            **arguments(
                hw=(16, 12), depth=40, local_positive=[cond(i) for i in range(4)]
            )
        )
        repeated = sampling.sample(**args)["samples"]
        torch.testing.assert_close(first, repeated)
        assert len({id(context) for context in contexts}) == 3
    else:
        with pytest.raises((RuntimeError, InterruptedError)):
            sampling.sample(**args)
    assert all(context.plan is None for context in contexts)
    assert all(
        evaluation.plan is None
        and evaluation.adapter is None
        and not evaluation.weights
        for evaluation in evaluations
    )
    assert not args["model"].wrappers and not args["latent_image"]["samples"].any()
    if host.latest_clone is not None:
        for kind in ("predict_noise", "calc_cond_batch"):
            assert not host.latest_clone.get_wrappers(kind, sampling.WRAPPER_KEY)
    with pytest.raises(RuntimeError, match="closed"):
        type(contexts[0]).prepare_pair(
            contexts[0], [], [], args["tile_plan"].regions[0]
        )
