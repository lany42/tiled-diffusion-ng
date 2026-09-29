# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Shared orchestration contracts: one trajectory, routing, guards and cleanup."""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.geometry import (
    GeometrySignature,
    LatentSpec,
    crop,
    make_plan,
)

from . import krea2_host
from .host import ControlNet, Model, arguments, cond
from .test_geometry import circular

FAMILIES = ["sdxl", "anima", "krea2"]
WRAPPER_KINDS = ("predict_noise", "calc_cond_batch")


def family_arguments(family, **kwargs):
    # Anima pads an unaligned 4D target; Krea2 keeps an aligned 5D batch of two.
    layout = {"sdxl": {}, "anima": {"hw": (13, 15)}, "krea2": {"rank": 5, "batch": 2}}
    return arguments(family, **{**layout[family], **kwargs})


def family_cond(family, value, **metadata):
    return (krea2_host.cond if family == "krea2" else cond)(value, **metadata)


def assert_detached(host, model):
    for candidate in (model, host.latest_clone):
        for kind in WRAPPER_KINDS:
            assert candidate is None or not candidate.get_wrappers(
                kind, sampling.WRAPPER_KEY
            )


@pytest.mark.parametrize("family", FAMILIES)
def test_one_trajectory_routes_four_views_per_evaluation(host, family):
    args = family_arguments(family)
    args["positive"] += family_cond(
        family, 3, start_percent=0.1, end_percent=0.9, strength=0.5
    )
    before = copy.deepcopy(args["positive"])
    latent, plan = args["latent_image"], args["tile_plan"]
    output = sampling.sample(**args)
    assert len(host.common_calls) == 1 and host.common_calls[0]["latent"] is latent
    assert output["note"] == "retained"
    expected_shape = latent["samples"].shape
    if family != "sdxl":
        expected_shape = (expected_shape[0], 16, 1, *plan.latent_hw)
    assert output["samples"].shape == expected_shape
    assert host.global_evaluations
    assert len(host.tile_calls) == 4 * len(host.global_evaluations)
    for evaluation, (current, sigma) in enumerate(host.global_evaluations):
        for index, region in enumerate(plan.regions):
            branches, tile, tile_sigma, _ = host.tile_calls[4 * evaluation + index]
            # Unaligned canvases wrap circularly, like native patch padding.
            expected = circular(current, region.sampling, plan.latent_hw)
            torch.testing.assert_close(tile, expected, rtol=0, atol=0)
            torch.testing.assert_close(tile_sigma, sigma)
            assert len(branches[0]) == 2
            assert all(
                entry[sampling.TAG] == region.tile_id
                for branch in branches
                for entry in branch
            )
    assert not latent["samples"].any()
    for before_entry, after_entry in zip(before, args["positive"], strict=True):
        torch.testing.assert_close(before_entry[0], after_entry[0])
        assert before_entry[1].keys() == after_entry[1].keys()
    assert_detached(host, args["model"])


@pytest.mark.parametrize("family", FAMILIES)
def test_identical_locals_match_global_and_distinct_locals_replace_it(host, family):
    args = family_arguments(family)
    global_result = sampling.sample(**args)["samples"]
    same = sampling.sample(**args, local_positive=[args["positive"]] * 4)["samples"]
    torch.testing.assert_close(global_result, same, rtol=0, atol=0)
    start = len(host.tile_calls)
    locals = [family_cond(family, index + 10) for index in range(4)]
    sampling.sample(**args, local_positive=locals)
    for tile, call in enumerate(host.tile_calls[start : start + 4]):
        assert call[0][0][0]["cross_attn"].mean().item() == tile + 10
        assert call[0][1][0]["cross_attn"].mean().item() == -1
    assert all(sampling.TAG not in meta for local in locals for _, meta in local)


@pytest.mark.parametrize(
    "locals,error",
    [
        ([cond(1)], "four complete"),
        ([[], cond(1), cond(2), cond(3)], "complete, nonempty CONDITIONING"),
        (cond(1) * 4, "invalid CONDITIONING nesting"),
    ],
    ids=["count", "empty", "nesting"],
)
def test_malformed_local_positives_never_fall_back(host, locals, error):
    with pytest.raises(ValueError, match=error):
        sampling.sample(**arguments(), local_positive=locals)
    assert not host.common_calls


def test_sdxl_size_conditioning_stays_full_canvas(host):
    args = arguments()
    args["positive"] += cond(3, width=2048, height=1024)
    sampling.sample(**args)
    # SDXL encodes y on the full canvas; views must not crop or rewrite it.
    assert {
        entry["model_conds"]["y"]
        for branches, *_ in host.tile_calls
        for entry in branches[0]
    } == {(136, 104), (2048, 1024)}


@pytest.mark.parametrize(
    "field",
    [
        "context_handler",
        "model_function_wrapper",
        "multigpu_clones",
        "sampler_calc_cond_batch_function",
        "multigpu",
    ],
)
def test_incompatible_model_options(host, field):
    # Upstream TiledDiffusion installs model_function_wrapper; multi-device
    # clones and replacement evaluators route around the tiled continuation.
    args = arguments()
    if field == "multigpu":
        args["model"].additional_models["multigpu"] = [Model()]
    else:
        args["model"].model_options[field] = object()
    with pytest.raises(
        ValueError, match="Multi-device" if field == "multigpu" else field
    ):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize(
    "damage,error",
    [
        ("branches", "unexpected prediction branch count"),
        ("batch", "incompatible tile prediction"),
        ("dtype", "dtype changed between tiles"),
    ],
)
def test_host_tile_predictions_must_match_their_view(host, damage, error):
    args = arguments(batch=2)
    tiles = []

    def inner(executor, model, conds, x, sigma, options):
        predictions = executor(model, conds, x, sigma, options)
        tiles.append(x)
        if damage == "branches":
            return predictions[:1]
        if damage == "batch":
            return [prediction[:1] for prediction in predictions]
        return [p.double() for p in predictions] if len(tiles) == 2 else predictions

    common = host.nodes.common_ksampler

    def common_with_inner(clone, *a, **kw):
        clone.add_wrapper_with_key("calc_cond_batch", "inner", inner)
        return common(clone, *a, **kw)

    host.nodes.common_ksampler = common_with_inner
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert_detached(host, args["model"])


@pytest.mark.parametrize(
    "family,latent_dtype,prediction_dtype",
    [("sdxl", torch.float16, torch.float16), ("anima", torch.float32, torch.float64)],
)
def test_fused_predictions_keep_the_host_prediction_dtype(
    host, family, latent_dtype, prediction_dtype
):
    # The host may promote predictions beyond the latent dtype, here through
    # float64 CONST sigmas. Fusion weights follow each prediction's dtype.
    captures = []
    args = arguments(family, dtype=latent_dtype, cfg=1)
    args["model"].model_options["sampler_post_cfg_function"] = [
        lambda data: captures.append(data["cond_denoised"]) or data["denoised"]
    ]
    host.dispatch["euler"].sampler_function = lambda evaluate, x, sigmas: evaluate(
        x, torch.tensor([0.6], dtype=prediction_dtype), (0, 0)
    )
    result = sampling.sample(**args)
    assert captures and all(c.dtype == prediction_dtype for c in captures)
    assert all(torch.isfinite(c).all() for c in captures)
    assert torch.isfinite(result["samples"]).all()
    assert host.tile_calls and all(call[0][1] is None for call in host.tile_calls)


def test_live_registrations_and_global_cfg_hooks(host):
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
    args = arguments(sampler_name="res_3s_ode", scheduler="beta57", cfg=1)
    captures = []
    args["model"].model_options["sampler_pre_cfg_function"] = [
        lambda data: captures.append(("pre", data["conds_out"])) or data["conds_out"]
    ]
    host.option_update = lambda options: options.update(
        disable_cfg1_optimization=True,
        sampler_post_cfg_function=[
            lambda data: captures.append(("post", data)) or data["denoised"]
        ],
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
        ("scheduler", 1, 0.4),
        ("sampler", (1, 4, 13, 17), (0.7, 0.3)),
    ]
    assert [
        float(sigma.item()) for _, sigma in host.global_evaluations
    ] == pytest.approx([0.7] * 3 + [0.3] * 3)
    # Global hooks run once per evaluation on fused full-canvas branches.
    assert [kind for kind, _ in captures] == ["pre", "post"] * 6
    for kind, data in captures:
        branches = (
            data if kind == "pre" else [data["cond_denoised"], data["uncond_denoised"]]
        )
        assert all(branch.shape == (1, 4, 13, 17) for branch in branches)
    assert all(call[0][1] is not None for call in host.tile_calls)
    assert host.common_calls[0]["sampler_name"] == "res_3s_ode"
    assert host.common_calls[0]["scheduler"] == "beta57"
    del host.dispatch["res_3s_ode"]
    with pytest.raises(ValueError, match="Cannot resolve native"):
        sampling.sample(**args)
    args["scheduler"] = "missing"
    with pytest.raises(ValueError, match="Missing native scheduler"):
        sampling.sample(**args)


def test_foreign_wrappers_compose_with_live_options_and_seed(host):
    args = arguments()
    seen, events = [], []

    def outer_prediction(executor, x, timestep, model_options, seed):
        seen.append((tuple(x.shape), seed))
        return executor(x, timestep, {**model_options, "prediction": True}, seed)

    def outer(executor, model, conds, x, sigma, options):
        events.append(("outer", tuple(x.shape[-2:])))
        return executor(model, conds, x, sigma, {**options, "outer": True})

    def inner(executor, model, conds, x, sigma, options):
        assert options["prediction"] and options["outer"] and options["live"]
        events.append(("inner", tuple(x.shape[-2:])))
        return executor(model, conds, x, sigma, options)

    args["model"].add_wrapper_with_key("predict_noise", "other", outer_prediction)
    args["model"].add_wrapper_with_key("calc_cond_batch", "outer", outer)
    common = host.nodes.common_ksampler

    def common_with_inner(clone, *a, **kw):
        # Registered after tiling, so the host runs it inside each view.
        clone.add_wrapper_with_key("calc_cond_batch", "inner", inner)
        return common(clone, *a, **kw)

    host.nodes.common_ksampler = common_with_inner
    host.option_update = lambda options: options.update(live=True)
    sampling.sample(**args)
    evaluations = len(host.global_evaluations)
    tile = args["tile_plan"].tile_hw
    assert seen == [((1, 4, 13, 17), 123)] * evaluations
    assert events == [("outer", (13, 17)), *[("inner", tile)] * 4] * evaluations
    assert host.latest_clone.get_wrappers("predict_noise", "other") == [
        outer_prediction
    ]
    assert host.latest_clone.get_wrappers("calc_cond_batch", "outer") == [outer]
    assert host.latest_clone.get_wrappers("calc_cond_batch", "inner") == [inner]
    assert "live" not in args["model"].model_options
    assert_detached(host, args["model"])


@pytest.mark.parametrize("after_first_evaluation", [False, True])
def test_native_sampler_late_override_fails_before_untiled_forward(
    host, after_first_evaluation
):
    override_calls = []
    args = arguments(sampler_name="late_override")
    host.samplers.KSampler.SAMPLERS.append("late_override")

    def override(call):
        override_calls.append(call)
        return [call["input"], call["input"]]

    def registered_sampler(evaluate, x, sigmas):
        sigma = torch.tensor([sigmas[0]], dtype=x.dtype)
        if after_first_evaluation:
            x = evaluate(x, sigma, (0, 0))
        live = {**host.live_options, "sampler_calc_cond_batch_function": override}
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
    assert "sampler_calc_cond_batch_function" not in args["model"].model_options
    assert_detached(host, args["model"])


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


@pytest.mark.parametrize("replacement", ["missing", "foreign"])
def test_live_options_cannot_remove_or_replace_this_invocations_tiling(
    host, replacement
):
    args = arguments()

    def solver(evaluate, x, sigmas):
        x = evaluate(x, torch.tensor([0.6]), (0, 0))
        wrappers = host.live_options["transformer_options"]["wrappers"]
        entries = (
            [] if replacement == "missing" else [lambda executor, *a: executor(*a)]
        )
        live = {
            **host.live_options,
            "transformer_options": {
                **host.live_options["transformer_options"],
                "wrappers": {
                    **wrappers,
                    "calc_cond_batch": {sampling.WRAPPER_KEY: entries},
                },
            },
        }
        return evaluate(x, torch.tensor([0.4]), (0, 1), live)

    host.dispatch["euler"].sampler_function = solver
    with pytest.raises(
        ValueError, match="removed or replaced.*tiled conditioning wrapper"
    ):
        sampling.sample(**args)
    assert len(host.tile_calls) == 4
    assert_detached(host, args["model"])


@pytest.mark.parametrize(
    "damage,error",
    [("tag", "Missing or unknown tile tag"), ("tile", "missing a tile")],
)
def test_prepared_conditioning_must_keep_every_tile_tag(host, damage, error):
    def mutate(branches):
        if damage == "tag":
            branches[0][0].pop(sampling.TAG)
        else:
            branches[0][:] = [e for e in branches[0] if e[sampling.TAG] != "BR"]

    host.mutate_prepared = mutate
    args = arguments()
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.tile_calls
    assert_detached(host, args["model"])


def test_shared_evaluation_delegates_embedding_and_layout_semantics(host):
    # A synthetic adapter contract, not an additional supported model family.
    plan = make_plan(
        LatentSpec(
            GeometrySignature(
                "synthetic",
                1,
                "fixture",
                layout="BTCHW",
                rank=5,
                channels=7,
                scale=(1, 1),
            ),
            (5, 9),
        ),
        2,
    )
    x = torch.arange(2 * 3 * 7 * 5 * 9, dtype=torch.float64).reshape(2, 3, 7, 5, 9)
    validations = []

    class Adapter:
        def validate_model_options(self, options):
            validations.append("options")

        def validate_condition(self, embedding, metadata):
            assert embedding == {"tokens": (1, 2)}
            validations.append("embedding")

        def validate_evaluation(self, samples, tile_plan):
            assert samples.shape == (2, 3, 7, *tile_plan.latent_hw)
            validations.append("layout")

        def adapt_spatial_condition(self, condition, region):
            return {**condition, "adapter_region": region.tile_id}

    adapter = Adapter()
    sampling.validate_conditioning([[{"tokens": (1, 2)}, {}]], adapter, "positive")

    class Context:
        def tile_options(self, options, region):
            return nullcontext(options)

    evaluation = sampling.TileEvaluation(plan, adapter, Context())
    regions = []

    def next_wrapper(model, branches, tile, sigma, options):
        regions.append(branches[0][0]["adapter_region"])
        return [tile * 0.5 + 2, torch.zeros_like(tile)]

    branches = [[{sampling.TAG: region.tile_id} for region in plan.regions], None]
    try:
        output = evaluation(next_wrapper, None, branches, x, torch.tensor([0.5]), {})
    finally:
        evaluation.close()
    torch.testing.assert_close(output[0], x * 0.5 + 2)
    assert validations == ["embedding", "options", "layout"]
    assert regions == ["TL", "TR", "BR", "BL"]
    with pytest.raises(RuntimeError, match="already closed"):
        evaluation(next_wrapper, None, branches, x, torch.tensor([0.5]), {})


def test_context_prepares_every_pair_before_host_sampling(host, monkeypatch):
    control = ControlNet()
    args = arguments(positive=cond(1, control=control, control_apply_to_uncond=True))
    adapter = resolve_adapter(args["model"])
    create = adapter.create_sampling_context
    events = []

    class Context:
        def __init__(self, plan):
            self.inner = create(plan)

        def prepare_model(self, model, latent):
            events.append("model")
            self.inner.prepare_model(model, latent)

        def tile_options(self, options, region):
            return self.inner.tile_options(options, region)

        def prepare_pair(self, positive, negative, region):
            events.append(region.tile_id)
            # Host propagation happened before adapter-specific preparation.
            assert positive[0]["control"] is negative[0]["control"]
            assert sampling.TAG not in positive[0]
            return self.inner.prepare_pair(positive, negative, region)

        def finalize_preparation(self):
            assert not host.common_calls
            events.append("finalize")
            self.inner.finalize_preparation()

        def close(self):
            events.append("close")
            self.inner.close()

    monkeypatch.setattr(adapter, "create_sampling_context", Context)
    sampling.sample(**args)
    assert events == ["model", "TL", "TR", "BR", "BL", "finalize", "close"]


def test_denoised_fusion_precedes_global_cfg_with_per_image_sigmas(host):
    values = (2, 4, 7, 11)
    args = arguments(
        "krea2",
        batch=2,
        dtype=torch.float64,
        local_positive=[krea2_host.cond(value) for value in values],
    )
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


LIFECYCLE = [
    ("sdxl", None),
    ("anima", None),
    ("krea2", None),
    ("sdxl", "cancel"),
    ("anima", "cancel"),
    ("krea2", "cancel"),
    ("sdxl", "clone"),
    ("krea2", "prepare"),
    ("anima", "cleanup"),
]


@pytest.mark.parametrize("family,failure", LIFECYCLE)
def test_invocation_state_released_on_every_exit(host, monkeypatch, family, failure):
    control = ControlNet()
    spatial = {
        "sdxl": {"control": control, "control_apply_to_uncond": True},
        "anima": {"control": None},
        "krea2": {
            "reference_latents": [torch.ones(1, 16, 1, 5, 7)],
            "reference_latents_method": "index",
        },
    }[family]
    args = family_arguments(family, positive=family_cond(family, 2, **spatial))
    metadata = args["positive"][0][1].copy()
    adapter = resolve_adapter(args["model"])
    create, evaluation_type = adapter.create_sampling_context, sampling.TileEvaluation
    contexts, evaluations = [], []

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate failure")

    def create_evaluation(*args):
        evaluations.append(evaluation_type(*args))
        return evaluations[-1]

    monkeypatch.setattr(sampling, "TileEvaluation", create_evaluation)
    with monkeypatch.context() as patch:

        def create_context(plan):
            context = create(plan)
            contexts.append(context)
            if failure == "prepare":
                patch.setattr(context, "prepare_pair", fail)
            elif failure == "cleanup":
                close = context.close

                def close_then_fail():
                    close()
                    fail()

                patch.setattr(context, "close", close_then_fail)
            return context

        patch.setattr(adapter, "create_sampling_context", create_context)
        if failure == "clone":
            patch.setattr(args["model"], "clone", fail)
        elif failure == "cancel":
            host.interrupt_after = 2
        if failure is None:
            first = sampling.sample(**args)["samples"]
            other = family_arguments(
                family,
                model=args["model"],
                hw=(16, 12),
                local_positive=[family_cond(family, i) for i in range(4)],
            )
            sampling.sample(**other)
            repeated = sampling.sample(**args)["samples"]
            torch.testing.assert_close(first, repeated, rtol=0, atol=0)
            assert len({id(context) for context in contexts}) == 3
            with pytest.raises(ValueError, match="Stale"):
                sampling.sample(**{**other, "tile_plan": args["tile_plan"]})
        else:
            with pytest.raises((RuntimeError, InterruptedError)):
                sampling.sample(**args)
    assert contexts or failure == "clone"
    assert all(context.plan is None for context in contexts)
    assert all(
        evaluation.plan is evaluation.context is None and not evaluation.weights
        for evaluation in evaluations
    )
    assert_detached(host, args["model"])
    assert not args["latent_image"]["samples"].any()
    assert args["positive"][0][1] == metadata
    if family == "sdxl":
        assert control.cond_hint_original is not None and control.cleanups == 0
        assert all(
            c.cond_hint_original is None and c.cond_hint is None
            for c in host.discovered
        )
    elif contexts:
        with pytest.raises(RuntimeError, match="closed"):
            type(contexts[0]).prepare_pair(
                contexts[0], [], [], args["tile_plan"].regions[0]
            )
    # Inputs remain usable after any exit.
    host.interrupt_after = None
    assert sampling.sample(**args)["samples"].isfinite().all()
