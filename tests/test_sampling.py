# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import copy
from types import SimpleNamespace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.geometry import crop, make_plan

from .host import ControlNet, Model, VPrediction, cond


def arguments(model=None, hw=(13, 17), dtype=torch.float32, **kwargs):
    model = model or Model()
    latent = {"samples": torch.zeros(1, 4, *hw, dtype=dtype), "note": "keep me"}
    args = {
        "model": model,
        "seed": 123,
        "steps": 2,
        "cfg": 7.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "positive": cond(2),
        "negative": cond(-1),
        "latent_image": latent,
        "tile_plan": make_plan(resolve_adapter(model).describe(model, latent), 16),
        "denoise": 0.4,
    }
    args.update(kwargs)
    return args


def test_one_global_sampler_multiple_evaluations_full_canvas_defaults_and_immutability(
    host,
):
    args = arguments()
    args["positive"] += cond(
        3, width=2048, height=1024, start_percent=0.1, end_percent=0.9, strength=0.5
    )
    before = copy.deepcopy(args["positive"])
    output = sampling.sample(**args)
    assert len(host.common_calls) == 1
    assert len(host.global_evaluations) == 6
    assert len(host.tile_calls) == 24
    assert host.common_calls[0]["latent"] is args["latent_image"]
    assert host.latest_clone is not args["model"]
    assert host.latest_clone.patches is args["model"].patches
    assert output["samples"].shape == (1, 4, 13, 17)
    assert output["note"] == "keep me"
    assert args["model"].wrappers == {}
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)
    assert not host.latest_clone.get_wrappers("predict_noise", sampling.WRAPPER_KEY)
    for before_entry, after_entry in zip(before, args["positive"]):
        torch.testing.assert_close(before_entry[0], after_entry[0])
        assert before_entry[1].keys() == after_entry[1].keys()
    assert not args["latent_image"]["samples"].any()
    assert host.prepared[0][0]["model_conds"]["y"] == (136, 104)
    assert host.prepared[0][1]["model_conds"]["y"] == (2048, 1024)
    for evaluation, (global_x, sigma) in enumerate(host.global_evaluations):
        for offset, region in enumerate(args["tile_plan"].regions):
            branches, tile_x, tile_sigma, _ = host.tile_calls[4 * evaluation + offset]
            torch.testing.assert_close(tile_x, crop(global_x, region.sampling))
            torch.testing.assert_close(tile_sigma, sigma)
            assert all(
                entry[sampling.TAG] == region.tile_id
                for branch in branches
                for entry in branch
            )
            assert len(branches[0]) == 2


def test_identical_locals_match_and_distinct_locals_replace_global(host):
    args = arguments(steps=1)
    global_result = sampling.sample(**args)
    same_result = sampling.sample(**args, local_positive=[args["positive"]] * 4)
    torch.testing.assert_close(global_result["samples"], same_result["samples"])
    start = len(host.tile_calls)
    sampling.sample(**args, local_positive=[cond(i + 10) for i in range(4)])
    for tile, call in enumerate(host.tile_calls[start : start + 4]):
        assert call[0][0][0]["cross_attn"].mean().item() == tile + 10
        assert call[0][1][0]["cross_attn"].mean().item() == -1


@pytest.mark.parametrize(
    "locals", [[], [cond(1)], [None] * 4, cond(1) * 4, [[], cond(1), cond(2), cond(3)]]
)
def test_bad_local_nesting_is_never_fallback(host, locals):
    with pytest.raises(ValueError, match="CONDITIONING"):
        sampling.sample(**arguments(), local_positive=locals)
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("area", (5, 5, 0, 0)),
        ("mask", torch.ones(1, 13, 17)),
        ("gligen", object()),
        ("hooks", object()),
        ("concat_latent_image", torch.ones(1, 4, 13, 17)),
        ("unknown_tensor", torch.ones(2, 3)),
    ],
)
def test_unsupported_condition_metadata_names_field(host, field, value):
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments(positive=cond(1, **{field: value})))
    assert not host.common_calls


@pytest.mark.parametrize(
    "field",
    ["context_handler", "model_function_wrapper", "sampler_calc_cond_batch_function"],
)
def test_incompatible_model_options(host, field):
    args = arguments()
    args["model"].model_options[field] = object()
    with pytest.raises(ValueError, match=field):
        sampling.sample(**args)


def test_native_epsilon_v_prediction_gate(host):
    args = arguments()
    args["model"].sampling = VPrediction()
    sampling.sample(**args)
    args["model"].sampling.calculate_denoised = lambda *a: None
    with pytest.raises(ValueError, match="model_sampling"):
        sampling.sample(**args)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_boundary_dtype_and_cfg1_omission(host, dtype):
    result = sampling.sample(**arguments(dtype=dtype, cfg=1, steps=1))
    assert result["samples"].dtype == dtype
    assert all(call[0][1] is None for call in host.tile_calls)
    assert torch.isfinite(result["samples"]).all()


def test_live_registrations_hooks_options_and_common_lookup(host):
    from tiled_diffusion_ng.nodes import TileSampler

    before = TileSampler.define_schema()
    host.samplers.KSampler.SAMPLERS.append("res_3s_ode")
    host.samplers.KSampler.SCHEDULERS.append("beta57")
    dispatched = []

    def registered_sampler(evaluate, x, sigmas):
        dispatched.append(("sampler", tuple(x.shape), tuple(sigmas)))
        return host.solve(evaluate, x, sigmas)

    def registered_scheduler(steps, denoise):
        dispatched.append(("scheduler", steps, denoise))
        return [0.7, 0.3]

    host.dispatch["res_3s_ode"] = SimpleNamespace(sampler_function=registered_sampler)
    host.samplers.SCHEDULER_HANDLERS["beta57"] = SimpleNamespace(
        handler=registered_scheduler
    )
    after = TileSampler.define_schema()
    inputs = {item.id: item for item in after.inputs}
    assert "res_3s_ode" in inputs["sampler_name"].options
    assert "beta57" in inputs["scheduler"].options
    assert (
        "res_3s_ode"
        not in {item.id: item for item in before.inputs}["sampler_name"].options
    )
    captures = []
    options_seen = []
    args = arguments(sampler_name="res_3s_ode", scheduler="beta57", cfg=1)

    def capture(data):
        captures.append(data)
        return data["denoised"]

    def composable(executor, model, conds, x, sigma, options):
        options_seen.append(options)
        return executor(model, conds, x, sigma, options)

    args["model"].add_wrapper_with_key("calc_cond_batch", "other", composable)
    args["model"].model_options["sampler_pre_cfg_function"] = [
        lambda data: data["conds_out"]
    ]
    host.option_update = lambda options: options.update(
        disable_cfg1_optimization=True,
        sampler_post_cfg_function=[capture],
        live_res_option="retained",
    )
    called_live = []
    old_common = host.nodes.common_ksampler

    def new_common(*a, **kw):
        called_live.append(True)
        return old_common(*a, **kw)

    host.nodes.common_ksampler = new_common
    sampling.sample(**args)
    assert called_live == [True]
    assert dispatched == [
        ("scheduler", 2, 0.4),
        ("sampler", (1, 4, 13, 17), (0.7, 0.3)),
    ]
    assert [
        float(sigma.item()) for _, sigma in host.global_evaluations
    ] == pytest.approx([0.7] * 3 + [0.3] * 3)
    assert len(captures) == 6
    assert all(c["cond_denoised"].shape == (1, 4, 13, 17) for c in captures)
    assert all(c["uncond_denoised"].shape == (1, 4, 13, 17) for c in captures)
    assert all(call[0][1] is not None for call in host.tile_calls)
    assert all(options["live_res_option"] == "retained" for options in options_seen)
    assert host.latest_clone.get_wrappers("calc_cond_batch", "other") == [composable]
    assert host.common_calls[0]["sampler_name"] == "res_3s_ode"
    assert host.common_calls[0]["scheduler"] == "beta57"
    del host.dispatch["res_3s_ode"]
    with pytest.raises(ValueError, match="Cannot resolve native"):
        sampling.sample(**args)
    args["scheduler"] = "missing"
    with pytest.raises(ValueError, match="Missing native scheduler"):
        sampling.sample(**args)


def test_masks_batch_indices_zero_denoise_and_bookkeeping(host):
    args = arguments()
    x = (
        torch.arange(4 * 13 * 17, dtype=torch.float32)
        .reshape(1, 4, 13, 17)
        .repeat(2, 1, 1, 1)
    )
    mask = torch.zeros(2, 13, 17)
    mask[:, :, 9:] = 1
    args["latent_image"] = {
        "samples": x,
        "noise_mask": mask,
        "batch_index": [7, 7],
        "downscale_ratio_spacial": 8,
        "downscale_ratio_temporal": 1,
        "note": "retained",
    }
    result = sampling.sample(**args)
    assert result["noise_mask"] is mask
    assert result["batch_index"] == [7, 7]
    assert "downscale_ratio_spacial" not in result
    assert "downscale_ratio_spacial" in args["latent_image"]
    torch.testing.assert_close(result["samples"][..., :9], x[..., :9])
    torch.testing.assert_close(result["samples"][0], result["samples"][1])
    for denoise in (0.0, 1.0):
        args["denoise"] = denoise
        output = sampling.sample(**args)
        assert host.common_calls[-1]["denoise"] == denoise
        if denoise == 0:
            torch.testing.assert_close(output["samples"], x)


def test_control_discovery_pair_sharing_chain_crop_schedule_and_cleanup(host):
    first, second = ControlNet(), ControlNet()
    first.previous_controlnet = second
    first.timestep_percent_range = (0.1, 0.8)
    first.extra_args = {"scale": 0.75}
    first.cond_hint_original = torch.arange(3 * 104 * 136, dtype=torch.float32).reshape(
        1, 3, 104, 136
    )
    second.cond_hint_original = first.cond_hint_original
    original_hint = first.cond_hint_original.clone()
    args = arguments(
        positive=cond(999, control=ControlNet()),
        local_positive=[
            cond(i, control=first, control_apply_to_uncond=True) for i in range(4)
        ],
    )
    saved_hints = []
    saved_controls = []

    def inspect_branches(branches):
        for i in range(4):
            p, n = branches[0][i], branches[1][i]
            assert p["control"] is n["control"]
            control = p["control"]
            assert control is not first
            assert control.previous_controlnet is not second
            assert control.control_model is first.control_model
            assert control.extra_args == {"scale": 0.75}
            assert control.timestep_percent_range == (0.1, 0.8)
            saved_hints.append(control.cond_hint_original.clone())
            saved_controls.extend([control, control.previous_controlnet])

    host.mutate_prepared = inspect_branches
    sampling.sample(**args)
    assert len(host.resize_calls) == 1
    assert len(host.discovered) == 4
    assert len({id(control) for control in saved_controls}) == 8
    for region, hint in zip(args["tile_plan"].regions, saved_hints):
        torch.testing.assert_close(hint, crop(original_hint, region.pixel_sampling))
    assert all(
        control.pre_runs > 0 and control.cleanups > 0 for control in saved_controls
    )
    assert all(
        control.cond_hint_original is None
        and control.cond_hint is None
        and control.previous_controlnet is None
        for control in saved_controls
    )
    assert first.previous_controlnet is second
    assert first.cleanups == 0 and first.pre_runs == 0 and first.cond_hint is None
    torch.testing.assert_close(first.cond_hint_original, original_hint)


def test_control_resize_before_crop_not_resizing_scene_per_tile(host):
    control = ControlNet()
    # Source aspect ratio differs: centered resize must happen once globally.
    control.cond_hint_original = torch.arange(3 * 8 * 24, dtype=torch.float32).reshape(
        1, 3, 8, 24
    )
    saved = []
    host.mutate_prepared = lambda branches: saved.extend(
        c["control"].cond_hint_original.clone() for c in branches[0]
    )
    args = arguments(positive=cond(1, control=control, control_apply_to_uncond=True))
    sampling.sample(**args)
    assert len(host.resize_calls) == 1
    normalized = host.resize(
        control.cond_hint_original, 136, 104, "nearest-exact", "center"
    )
    for hint, region in zip(saved, args["tile_plan"].regions):
        torch.testing.assert_close(hint, crop(normalized, region.pixel_sampling))
    assert not torch.equal(saved[0], saved[2])


def test_controls_do_not_cross_tile_pairs_and_global_controls_are_replaced(host):
    a, b, global_control = ControlNet(), ControlNet(), ControlNet()
    locals = [
        cond(1, control=a, control_apply_to_uncond=True),
        cond(2),
        cond(3, control=b, control_apply_to_uncond=True),
        cond(4),
    ]
    args = arguments(positive=cond(99, control=global_control), local_positive=locals)
    sampling.sample(**args)
    assert ["control" in c for c in host.prepared[1]] == [True, False, True, False]
    assert global_control.pre_runs == 0
    assert len(host.discovered) == 2


def test_explicit_negative_controls_block_propagation_and_multi_entry_pairing(host):
    a, b, n = ControlNet(), ControlNet(), ControlNet()
    args = arguments(
        positive=cond(1, control=a, control_apply_to_uncond=True)
        + cond(2, control=b, control_apply_to_uncond=True),
        negative=cond(-1, control=n),
    )
    saved = []
    host.mutate_prepared = lambda branches: saved.extend(
        entry["control"].control_model for entry in branches[1]
    )
    sampling.sample(**args)
    assert saved == [n.control_model] * 4
    host.discovered.clear()
    args["negative"] = cond(-1)
    sampling.sample(**args)
    assert len(host.prepared[1]) == 4
    for name in ("TL", "TR", "BR", "BL"):
        positives = [c for c in host.prepared[0] if c[sampling.TAG] == name]
        negatives = [c for c in host.prepared[1] if c[sampling.TAG] == name]
        assert len(negatives) == 1
        assert negatives[0]["control"] is positives[-1]["control"]


def test_zero_strength_and_inactive_schedule_equal_control_free(host):
    args = arguments(steps=1)
    baseline = sampling.sample(**args)["samples"]
    control = ControlNet()
    control.cond_hint_original.fill_(1)
    control.strength = 0
    args["positive"] = cond(2, control=control)
    torch.testing.assert_close(sampling.sample(**args)["samples"], baseline)
    control.strength = 1
    control.timestep_percent_range = (0.9, 1.0)
    torch.testing.assert_close(sampling.sample(**args)["samples"], baseline)


@pytest.mark.parametrize("failure", ["tile", "cancel", "tag", "prepared_spatial"])
def test_cleanup_on_failures_and_repeated_stages(host, failure):
    control = ControlNet()
    args = arguments(positive=cond(1, control=control))
    if failure == "tile":
        host.fail_tile = 3
    elif failure == "cancel":
        host.interrupt_after = 3
    elif failure == "tag":
        host.mutate_prepared = lambda branches: branches[0][0].pop(sampling.TAG)
    else:
        host.mutate_prepared = lambda branches: branches[0][0]["model_conds"].update(
            c_concat=object()
        )
    with pytest.raises((RuntimeError, InterruptedError, ValueError)):
        sampling.sample(**args)
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)
    assert not host.latest_clone.get_wrappers("predict_noise", sampling.WRAPPER_KEY)
    assert all(
        c.cond_hint_original is None and c.cond_hint is None for c in host.discovered
    )
    assert args["model"].wrappers == {} and control.cond_hint_original is not None
    host.fail_tile = host.interrupt_after = host.mutate_prepared = None
    result_a = sampling.sample(**args)
    b = arguments(
        model=args["model"], hw=(17, 23), local_positive=[cond(i) for i in range(4)]
    )
    sampling.sample(**b)
    result_a_again = sampling.sample(**args)
    torch.testing.assert_close(result_a["samples"], result_a_again["samples"])
    b["tile_plan"] = args["tile_plan"]
    with pytest.raises(ValueError, match="Stale"):
        sampling.sample(**b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vae", object()),
        ("latent_format", object()),
        ("concat_mask", True),
        ("extra_concat_orig", [torch.ones(1, 1, 16, 16)]),
        ("extra_hooks", object()),
        ("multigpu_clones", {"other_device": object()}),
        ("preprocess_image", lambda x: x * 2),
        ("compression_ratio", 4),
        ("extra_conds", ["image"]),
        ("extra_args", {"spatial": torch.ones(2, 2)}),
    ],
)
@pytest.mark.parametrize("union", [False, True])
def test_control_capability_errors_before_host_sampling(host, union, field, value):
    control = ControlNet(union=union)
    setattr(control, field, value)
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls


def test_control_chain_cycle_and_failure_during_preparation(host):
    control = ControlNet()
    control.previous_controlnet = control
    with pytest.raises(ValueError, match="Cyclic"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert control.previous_controlnet is control
    assert not host.common_calls


def test_sampler_starts_denominator_validation_before_first_tile(host, monkeypatch):
    # The continuation must never receive an invalid normalization state.
    from tiled_diffusion_ng import fusion

    monkeypatch.setattr(fusion, "gaussian", lambda hw, **kw: torch.zeros(hw))
    with pytest.raises(ValueError, match="positive everywhere"):
        sampling.sample(**arguments())
    assert not host.tile_calls


@pytest.mark.parametrize("after_first_evaluation", [False, True])
@pytest.mark.parametrize("replace_options", [False, True])
def test_native_sampler_late_override_fails_before_untiled_forward(
    host, after_first_evaluation, replace_options
):
    override_calls = []
    control = ControlNet()
    args = arguments(positive=cond(1, control=control), sampler_name="late_override")
    host.samplers.KSampler.SAMPLERS.append("late_override")

    def override(call):
        override_calls.append(call)
        return [call["input"], call["input"]]

    def registered_sampler(evaluate, x, sigmas):
        sigma = torch.tensor([sigmas[0]], dtype=x.dtype)
        if after_first_evaluation:
            x = evaluate(x, sigma, (0, 0))
        live = host.live_options.copy() if replace_options else host.live_options
        live["sampler_calc_cond_batch_function"] = override
        return evaluate(x, sigma, (0, 1), model_options=live)

    host.dispatch["late_override"] = SimpleNamespace(
        sampler_function=registered_sampler
    )
    with pytest.raises(
        ValueError, match="sampler_calc_cond_batch_function.*bypasses tiling"
    ):
        sampling.sample(**args)
    assert not override_calls
    assert len(host.tile_calls) == (4 if after_first_evaluation else 0)
    assert args["model"].wrappers == {}
    assert "sampler_calc_cond_batch_function" not in args["model"].model_options
    for kind in ("predict_noise", "calc_cond_batch"):
        assert not host.latest_clone.get_wrappers(kind, sampling.WRAPPER_KEY)
    assert all(
        c.cond_hint_original is None and c.cond_hint is None for c in host.discovered
    )
    assert control.cond_hint_original is not None


def test_prediction_guard_preserves_wrappers_live_options_and_seed(host):
    args = arguments()
    seen = []

    def outer(executor, x, timestep, model_options, seed):
        seen.append((tuple(x.shape), seed))
        return executor(
            x, timestep, {**model_options, "prediction_wrapper": True}, seed
        )

    def inner(executor, model, conds, x, timestep, model_options):
        assert model_options["prediction_wrapper"] is True
        return executor(model, conds, x, timestep, model_options)

    args["model"].add_wrapper_with_key("predict_noise", "other", outer)
    args["model"].add_wrapper_with_key("calc_cond_batch", "other", inner)
    sampling.sample(**args)
    assert seen == [((1, 4, 13, 17), 123)] * 6
    assert host.latest_clone.get_wrappers("predict_noise", "other") == [outer]
    assert host.latest_clone.get_wrappers("calc_cond_batch", "other") == [inner]
    assert len(host.tile_calls) == 24


def test_override_installed_by_outer_prediction_wrapper_is_rejected(host):
    args = arguments()
    called = []

    def outer(executor, x, timestep, model_options, seed):
        live = {
            **model_options,
            "sampler_calc_cond_batch_function": lambda args: called.append(args),
        }
        return executor(x, timestep, live, seed)

    args["model"].add_wrapper_with_key("predict_noise", "other", outer)
    with pytest.raises(ValueError, match="sampler_calc_cond_batch_function"):
        sampling.sample(**args)
    assert not called and not host.tile_calls
    assert host.latest_clone.get_wrappers("predict_noise", "other") == [outer]
