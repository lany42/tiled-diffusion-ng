# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Offline CPU contracts for tiled native LLLite, not real-host acceptance."""

import copy
import gc
import weakref
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.adapters._anima_lllite import (
    ATTACHMENT_KEY,
    DISPATCH_KEY,
    LLLiteInvocation,
    TiledLLLiteDispatcher,
    apply_lllite,
    get_attachment,
)
from tiled_diffusion_ng.geometry import image_views, make_plan

from .host import CONST, Model, cond, copy_containers, isolated_host
from .lllite_host import (
    TARGETS,
    AnimaLLLite,
    AnimaLLLiteAttentionPatch,
    AnimaLLLitePatch,
)
from .test_anima import arguments


def references(plan, groups=1, channels=3):
    h, w = plan.pixel_hw
    y, x = torch.meshgrid(
        torch.linspace(-0.3, 0.7, h), torch.linspace(0, 0.8, w), indexing="ij"
    )
    image = torch.stack((x, y, x * y), -1).unsqueeze(0)
    image = torch.cat([image + index * 0.25 for index in range(groups)])
    if channels == 4:
        image = torch.cat((image, torch.full_like(image[..., :1], 100)), -1)
    return image_views(image, plan)


def patched(args=None, *, groups=None, channels=3, weights=None, **settings):
    args = arguments(steps=1) if args is None else args.copy()
    source = args["model"]
    groups = args["latent_image"]["samples"].shape[0] if groups is None else groups
    refs = references(args["tile_plan"], groups, channels)
    model_patch = Model()
    model_patch.model = AnimaLLLite() if weights is None else weights
    args["model"] = apply_lllite(
        source, model_patch, args["tile_plan"], refs, **settings
    )
    return args, source, refs, model_patch


@pytest.mark.parametrize("combined", [False, True])
@pytest.mark.parametrize(
    "batch,groups,channels,cfg,entries",
    [(1, 1, 3, 4, 1), (2, 2, 4, 4, 2), (2, 1, 3, 1, 2)],
)
def test_crop_routing_batches_branches_and_native_hooks(
    host, combined, batch, groups, channels, cfg, entries
):
    args = arguments(batch=batch, cfg=cfg, steps=1)
    args["positive"] = cond(2) + (cond(3, strength=0.5) if entries == 2 else [])
    host.combine_anima_conditions = combined
    host.evaluations_per_step = 1
    args, source, refs, patch = patched(args, groups=groups, channels=channels)
    before = refs.clone()
    attachment = get_attachment(args["model"])
    assert (
        args["model"] is not source and not source.model_options["transformer_options"]
    )
    assert not source.attachments
    assert attachment.config.reference_tiles is refs
    assert attachment.post_input.models() == [patch]
    assert attachment.post_input.to(torch.float16) is attachment.post_input
    assert args["model"].model_patches_models() == [patch]
    assert (
        attachment.attn1.patch
        is attachment.attn2.patch
        is attachment.mlp.patch
        is attachment.post_input
    )
    assert type(attachment.attn1) is AnimaLLLiteAttentionPatch
    assert args["model"].clone().get_attachment(ATTACHMENT_KEY) is attachment
    with pytest.raises(FrozenInstanceError):
        attachment.config.strength = 9
    with pytest.raises(FrozenInstanceError):
        attachment.post_input.config = None
    sampling.sample(**args)
    assert len(host.common_calls) == 1 and host.patch_models == [patch]
    assert len(host.tile_calls) == 4
    branches = entries + int(cfg != 1)
    calls_per_tile = 1 if combined else branches
    assert len(host.lllite_forwards) == 4 * calls_per_tile
    for call_index, record in enumerate(host.lllite_forwards):
        index = call_index // calls_per_tile
        expected = refs[index::4, ..., :3].movedim(-1, 1).clamp(0, 1) * 2 - 1
        torch.testing.assert_close(patch.model.encodings[call_index], expected)
        embedding = (
            F.avg_pool2d(expected.mean(1, keepdim=True), 16).flatten(2).transpose(1, 2)
        )
        assert record.keys == (attachment.post_input,)
        torch.testing.assert_close(record.embedding[0], embedding)
        assert all(k and v for k, v in record.cross_kv)
        # Native per-forward reset, not a dispatcher or embedding cache.
        # The trace keeps native data alive; the invocation does not own it.
        assert tuple(record.options["model_patch_data"]) == record.keys
        assert record.options[DISPATCH_KEY].invocation is None
        assert record.x.shape[0] == batch * (branches if combined else 1)
        for _, target, repeated, strength in patch.model.calls[
            call_index * 5 : (call_index + 1) * 5
        ]:
            assert target in TARGETS and strength == 1
            torch.testing.assert_close(
                repeated, embedding.repeat(record.x.shape[0] // groups, 1, 1)
            )
    assert len(
        {id(r.options["model_patch_data"]) for r in host.lllite_forwards}
    ) == len(host.lllite_forwards)
    torch.testing.assert_close(refs, before)
    assert not args["latent_image"]["samples"].any()
    assert DISPATCH_KEY not in args["model"].model_options["transformer_options"]


@pytest.mark.parametrize(
    "strength,start,end,sigmas,active",
    [
        (0, 0, 1, [0.5, 0.9], False),
        (1, 0.2, 0.4, [0.5, 0.55], False),
        (1, 0.2, 0.4, [0.6, 0.6], True),
        (1, 0.2, 0.4, [0.4, 0.8], True),
        (-2, 0.2, 0.4, [0.3, 0.7], True),
        (1, 0.2, 0.4, [0.7, 0.9], False),
        (1, 0.8, 0.2, [0.5, 0.5], False),
    ],
)
def test_native_max_sigma_gate_boundaries_repeated_evaluations(
    host, strength, start, end, sigmas, active
):
    args, source, _, patch = patched(
        arguments(batch=2, steps=1),
        strength=strength,
        start_percent=start,
        end_percent=end,
    )
    predictions = []

    def solver(evaluate, x, schedule):
        for i in range(3):
            predictions.append(
                evaluate(x, torch.tensor(sigmas, dtype=torch.float64), (0, i))
            )
        return predictions[-1]

    host.dispatch["euler"].sampler_function = solver
    result = sampling.sample(**args)["samples"]
    assert len(patch.model.encodings) == (24 if active else 0)
    for prediction in predictions:
        torch.testing.assert_close(result, prediction.float())
    baseline = sampling.sample(**{**args, "model": source})["samples"]
    if active:
        assert not torch.equal(result, baseline)
    else:
        torch.testing.assert_close(result, baseline)


def test_thresholds_recomputed_after_sampling_settings_change(host, monkeypatch):
    args, source, _, patch = patched(start_percent=0.25, end_percent=0.75)
    thresholds = []
    original = AnimaLLLitePatch.__init__

    def init(self, *a):
        original(self, *a)
        thresholds.append((self.sigma_start, self.sigma_end))

    monkeypatch.setattr(AnimaLLLitePatch, "__init__", init)
    host.samplers.SCHEDULER_HANDLERS["normal"].handler = lambda *_: [0.6]
    sampling.sample(**args)

    class Shifted(CONST):
        def percent_to_sigma(self, percent):
            return 3 * (1 - percent)

    args["model"] = args["model"].clone()
    args["model"].sampling = Shifted()
    patch.model.encodings.clear()
    result = sampling.sample(**args)["samples"]
    assert thresholds == [(0.75, 0.25)] * 4 + [(2.25, 0.75)] * 4
    assert not patch.model.encodings
    torch.testing.assert_close(
        result, sampling.sample(**{**args, "model": source})["samples"]
    )


@pytest.mark.parametrize("strength", [0, 1])
@pytest.mark.parametrize("clone", [False, True])
def test_duplicate_application_including_clones(host, strength, clone):
    args, _, refs, patch = patched(strength=strength)
    model = args["model"].clone() if clone else args["model"]
    with pytest.raises(ValueError, match="already applied"):
        apply_lllite(model, patch, args["tile_plan"], refs, strength=0)


@pytest.mark.parametrize("native_first", [False, True])
@pytest.mark.parametrize("strength", [0, 1])
def test_native_chaining_rejected_in_either_order(host, native_first, strength):
    args, source, refs, patch = patched(strength=strength)
    target = source if native_first else args["model"].clone()
    target.set_model_patch(AnimaLLLitePatch(patch, refs, None, 0, 1, 0), "post_input")
    with pytest.raises(NotImplementedError, match="Native AnimaLLLiteApply chaining"):
        if native_first:
            apply_lllite(target, patch, args["tile_plan"], refs, strength=0)
        else:
            sampling.sample(**{**args, "model": target})
    assert not host.common_calls


@pytest.mark.parametrize(
    "change,error",
    [
        (lambda refs: refs[:3], "four crops"),
        (lambda refs: refs[:0], "four crops"),
        (lambda refs: refs[..., 0], "RGB/RGBA"),
        (lambda refs: refs[..., :2], "RGB/RGBA"),
        (lambda refs: refs.int(), "floating"),
        (lambda refs: refs[:, :-1], "dimensions"),
        (lambda refs: refs[:, :, :-16], "dimensions"),
    ],
)
def test_malformed_references_rejected_before_native_resize(host, change, error):
    args, source, refs, patch = patched()
    with pytest.raises(ValueError, match=error):
        apply_lllite(source, patch, args["tile_plan"], change(refs), strength=0)
    assert not host.resize_calls


@pytest.mark.parametrize("batch,groups", [(1, 2), (2, 3), (4, 2)])
def test_reference_groups_are_not_arbitrarily_repeated(host, batch, groups):
    args, _, _, _ = patched(arguments(batch=batch), groups=groups, strength=0)
    with pytest.raises(ValueError, match="source batch"):
        sampling.sample(**args)
    assert not host.common_calls


def test_complete_plan_identity_includes_requested_overlap(host):
    args, _, _, _ = patched(strength=0)
    old = args["tile_plan"]
    other = make_plan(
        resolve_adapter(args["model"]).describe(args["model"], args["latent_image"]), 1
    )
    assert old.tile_hw == other.tile_hw and old != other
    with pytest.raises(ValueError, match="same complete TILE_PLAN"):
        sampling.sample(**{**args, "tile_plan": other})
    assert not host.common_calls


@pytest.mark.parametrize(
    "mutation,error",
    [
        (lambda weights: setattr(weights, "cond_in_channels", 4), "RGB weights"),
        (lambda weights: setattr(weights, "model_dim", 8), "width"),
        (lambda weights: setattr(weights, "module_names", set()), "nonempty"),
        (
            lambda weights: weights.module_names.add("lllite_dit_blocks_28_mlp_layer1"),
            "depth",
        ),
        (
            lambda weights: weights.module_names.add(
                "lllite_dit_blocks_0_cross_attn_k_proj"
            ),
            "target",
        ),
        (
            lambda weights: weights.module_names.add("lllite_dit_blocks_00_mlp_layer1"),
            "target",
        ),
        (lambda weights: setattr(weights, "block_count", 9), "block_count"),
        (
            lambda weights: delattr(weights, "lllite_dit_blocks_0_self_attn_q_proj"),
            "Missing",
        ),
        (
            lambda weights: setattr(
                weights.lllite_dit_blocks_0_mlp_layer1.down, "in_features", 8
            ),
            "width",
        ),
    ],
)
def test_invalid_weights_rejected_at_zero_strength(host, mutation, error):
    args, source, refs, patch = patched()
    mutation(patch.model)
    with pytest.raises(ValueError, match=error):
        apply_lllite(source, patch, args["tile_plan"], refs, strength=0)


def test_native_model_and_loader_required(host):
    args, source, refs, patch = patched()
    with pytest.raises(ValueError, match="native Anima MODEL"):
        apply_lllite(Model(), patch, args["tile_plan"], refs)
    with pytest.raises(ValueError, match="native-loaded"):
        apply_lllite(
            source, SimpleNamespace(model=patch.model), args["tile_plan"], refs
        )
    patch.model = torch.nn.Linear(3, 4)
    with pytest.raises(ValueError, match="native-loaded"):
        apply_lllite(source, patch, args["tile_plan"], refs)


@pytest.mark.parametrize("depth", [28, 40])
def test_sparse_coverage_within_actual_backbone_depth(host, depth):
    name = f"lllite_dit_blocks_{depth - 1}_mlp_layer1"
    args, _, _, patch = patched(
        arguments(depth=depth, steps=1), weights=AnimaLLLite(targets=[name])
    )
    sampling.sample(**args)
    assert {call[:2] for call in patch.model.calls} == {(depth - 1, "mlp_layer1")}
    args["model"].model.diffusion_model.blocks.pop()
    with pytest.raises(ValueError, match="backbone depth"):
        sampling.sample(**args)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "kind",
        "unknown_version",
        "empty",
        "orphan",
        "no_hooks",
        "different_config",
    ],
)
def test_bad_or_orphaned_declarations_rejected(host, change):
    args, _, refs, patch = patched(strength=0)
    model = args["model"]
    attachment = get_attachment(model)
    if change in ("schema", "kind", "different_config"):
        fields = {
            "schema": {"schema_version": 2},
            "kind": {"kind": "unknown"},
            "different_config": {"config": replace(attachment.config, strength=1)},
        }[change]
        model.attachments[ATTACHMENT_KEY] = replace(attachment, **fields)
    elif change == "unknown_version":
        model.attachments[ATTACHMENT_KEY + ".v2"] = attachment
    elif change == "empty":
        model.attachments[ATTACHMENT_KEY] = None
    elif change == "orphan":
        model.attachments.clear()
    else:
        model.model_options["transformer_options"].pop("patches")
    with pytest.raises(ValueError, match="declaration|Orphaned|Incomplete"):
        sampling.sample(**args)
    with pytest.raises(ValueError):
        apply_lllite(model, patch, args["tile_plan"], refs, strength=0)
    assert not host.common_calls


@pytest.mark.parametrize(
    "change",
    [
        "remove_all",
        "remove_one",
        "replace",
        "duplicate",
        "targets",
        "native",
        "swap",
        "attachment",
    ],
)
@pytest.mark.parametrize("later", [False, True])
def test_altered_live_hooks_rejected(host, change, later):
    args, _, _, patch = patched(strength=0)

    def mutate(options):
        hooks = options["transformer_options"]["patches"]
        if change == "remove_all":
            hooks.clear()
        elif change == "remove_one":
            hooks.pop("post_input")
        elif change == "replace":
            old = hooks["attn1_patch"][0]
            hooks["attn1_patch"] = [AnimaLLLiteAttentionPatch(old.patch, old.targets)]
        elif change == "duplicate":
            hooks["mlp_patch"] *= 2
        elif change == "targets":
            hooks["attn2_patch"][0].targets = {"k": "cross_attn_q_proj"}
        elif change == "native":
            hooks["post_input"].append(AnimaLLLitePatch(patch, None, None, 0, 1, 0))
        elif change == "attachment":
            host.latest_clone.attachments.clear()
        else:
            hooks["attn1_patch"], hooks["attn2_patch"] = (
                hooks["attn2_patch"],
                hooks["attn1_patch"],
            )

    if later:

        def solve(evaluate, x, sigmas):
            evaluate(x, torch.tensor([0.7]), (0, 0))
            live = copy_containers(host.live_options)
            mutate(live)
            return evaluate(x, torch.tensor([0.5]), (0, 1), live)

        host.dispatch["euler"].sampler_function = solve
    else:
        host.option_update = mutate
    with pytest.raises((ValueError, NotImplementedError), match="LLLite|declaration"):
        sampling.sample(**args)
    assert len(host.tile_calls) == (4 if later else 0)
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)


@pytest.mark.parametrize("stage", ["apply_model", "diffusion_model"])
@pytest.mark.parametrize("remove_all", [False, True])
@pytest.mark.parametrize("strength,start", [(1.0, 0.0), (0.0, 0.0), (1.0, 0.9)])
def test_hooks_removed_inside_native_wrapper_are_rejected(
    host, stage, remove_all, strength, start
):
    args, _, _, patch = patched(strength=strength, start_percent=start)

    def strip_hooks(transformer):
        patches = (
            {}
            if remove_all
            else {
                name: hooks
                for name, hooks in transformer["patches"].items()
                if name != "post_input"
            }
        )
        return {**transformer, "patches": patches}

    def apply_wrapper(executor, model, x, sigma, conditions, transformer):
        return executor(model, x, sigma, conditions, strip_hooks(transformer))

    def forward_wrapper(executor, *args, transformer_options):
        return executor(*args, transformer_options=strip_hooks(transformer_options))

    wrapper = apply_wrapper if stage == "apply_model" else forward_wrapper
    args["model"].add_wrapper_with_key(stage, "strip_lllite_hooks", wrapper)
    with pytest.raises(ValueError, match="LLLite"):
        sampling.sample(**args)
    assert not patch.model.encodings


def test_completed_forward_releases_embedding_before_next_forward(host):
    args, _, _, _ = patched()
    context = LLLiteInvocation(args["tile_plan"])
    context.prepare_model(args["model"], args["latent_image"])
    dispatcher = context.attachment.post_input
    height, width = args["tile_plan"].tile_hw

    def forward(options):
        # No trace retains the forward's data dictionary or its embedding.
        transformer = dict(
            options["transformer_options"],
            model_patch_data={},
            sigmas=torch.tensor([0.5]),
            cond_or_uncond=[0],
        )
        dispatcher(
            {
                "x": torch.zeros(1, 16, 1, height, width),
                "img": torch.zeros(1, 1, height // 2, width // 2, 4),
                "transformer_options": transformer,
            }
        )
        return weakref.ref(transformer["model_patch_data"][dispatcher])

    try:
        with context.tile_options(
            args["model"].model_options, args["tile_plan"].regions[0]
        ) as options:
            for _ in range(6):
                embedding = forward(options)
                gc.collect()
                assert embedding() is None
    finally:
        context.close()


@pytest.mark.parametrize("strength,start,end", [(1, 0, 1), (0, 0, 1), (1, 0.9, 1)])
def test_ordinary_ksampler_requires_dispatch_even_when_inactive(
    host, strength, start, end
):
    args, _, _, _ = patched(strength=strength, start_percent=start, end_percent=end)
    native_args = {
        key: value
        for key, value in args.items()
        if key not in ("tile_plan", "latent_image")
    }
    with pytest.raises(ValueError, match="ordinary KSampler"):
        host.common_ksampler(**native_args, latent=args["latent_image"])
    assert not host.resize_calls


@pytest.mark.parametrize(
    "change,error",
    [
        ("geometry", "geometry"),
        ("batch", "latent batch"),
        ("branches", "batch mapping"),
        ("img", "embedded image"),
        ("no_data", "fresh model_patch_data"),
        ("references", "reference.*dimensions"),
        ("wrong_dispatcher", "dispatcher"),
        ("nested", "Nested"),
    ],
)
def test_forward_validation_precedes_zero_strength(host, change, error):
    args, _, _, _ = patched(arguments(batch=2), strength=0)
    context = LLLiteInvocation(args["tile_plan"])
    context.prepare_model(args["model"], args["latent_image"])
    dispatcher = context.attachment.post_input
    region = args["tile_plan"].regions[0]
    h, w = args["tile_plan"].tile_hw
    try:
        with context.tile_options(args["model"].model_options, region) as options:
            transformer = dict(
                options["transformer_options"],
                sigmas=torch.tensor([0.8]),
                model_patch_data={},
                cond_or_uncond=[0],
            )
            call = {
                "x": torch.zeros(2, 16, 1, h, w),
                "img": torch.zeros(2, 1, h // 2, w // 2, 4),
                "transformer_options": transformer,
            }
            if change == "geometry":
                call["x"] = call["x"][..., :-1]
            elif change == "batch":
                call["x"] = call["x"][:1]
            elif change == "branches":
                transformer["cond_or_uncond"] = [0, 1]
            elif change == "img":
                call["img"] = call["img"][:, :, :-1]
            elif change == "no_data":
                transformer.pop("model_patch_data")
            elif change == "references":
                context.delegates[0].image = context.delegates[0].image[:, :-1]
            elif change == "wrong_dispatcher":
                dispatcher = TiledLLLiteDispatcher(dispatcher.config)
            with pytest.raises(ValueError, match=error):
                if change == "nested":
                    with context.tile_options(options, region):
                        pass
                else:
                    dispatcher(call)
        with pytest.raises(ValueError, match="missing active tiled dispatch"):
            dispatcher(call)
    finally:
        context.close()
    assert not host.resize_calls


def test_bad_encoded_token_count_is_rejected_and_removed(host, monkeypatch):
    args, _, _, patch = patched()
    encode = patch.model.encode_conditioning
    monkeypatch.setattr(
        patch.model, "encode_conditioning", lambda image: encode(image)[:, :-1]
    )
    with pytest.raises(ValueError, match="encoded token count"):
        sampling.sample(**args)


@pytest.mark.parametrize("combined", [False, True])
def test_local_prompts_global_cfg_wrappers_loras_and_live_options(host, combined):
    args, source, _, patch = patched(
        arguments(batch=2, steps=1, local_positive=[cond(i) for i in range(4)])
    )
    host.combine_anima_conditions = combined
    marker = object()
    args["model"].attachments["nonspatial"] = marker
    args["model"].model_options["transformer_options"][
        "optimized_attention_override"
    ] = marker
    events, global_shapes = [], []

    def outer(executor, model, conditions, x, sigma, options):
        events.append(("outer", x.shape[-2:]))
        return executor(model, conditions, x, sigma, {**options, "compatible": marker})

    def inner(executor, model, conditions, x, sigma, options):
        assert options["compatible"] is marker and options["live"] is marker
        assert options["transformer_options"]["optimized_attention_override"] is marker
        events.append(("inner", x.shape[-2:]))
        return executor(model, conditions, x, sigma, options)

    def native(executor, model, x, sigma, conditions, transformer):
        assert transformer["sigmas"] is sigma
        return executor(model, x, sigma, conditions, transformer)

    def native_forward(executor, x, sigma, *args, transformer_options):
        assert transformer_options["sigmas"] is sigma
        return executor(x, sigma, *args, transformer_options=transformer_options)

    def cfg(data):
        global_shapes.append(data["input"].shape)
        return data["cond"]

    args["model"].add_wrapper_with_key("calc_cond_batch", "outer", outer)
    args["model"].add_wrapper_with_key("apply_model", "native", native)
    args["model"].add_wrapper_with_key("diffusion_model", "native", native_forward)
    args["model"].model_options["sampler_cfg_function"] = cfg
    common = host.nodes.common_ksampler

    def common_with_inner(clone, *a, **kw):
        clone.add_wrapper_with_key("calc_cond_batch", "inner", inner)
        return common(clone, *a, **kw)

    host.nodes.common_ksampler = common_with_inner
    host.option_update = lambda options: options.update(live=marker)
    sampling.sample(**args)
    assert events == [("outer", (12, 16)), *[("inner", (8, 10))] * 4] * 3
    assert global_shapes == [(2, 16, 1, 12, 16)] * 3
    assert len(host.common_calls) == 1 and host.patch_models == [patch]
    for index, (branches, _, _, _) in enumerate(host.tile_calls):
        assert branches[0][0]["cross_attn"].mean() == index % 4
    assert host.latest_clone.patches["lora"][0] is source.patches["lora"][0]
    assert host.latest_clone.attachments["nonspatial"] is marker
    assert "live" not in args["model"].model_options
    assert not args["model"].get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)
    assert len(patch.model.encodings) == (12 if combined else 24)


@pytest.mark.parametrize("failure", [None, "prepare", "tile", "native", "cancel"])
def test_lifecycle_closes_handles_delegates_and_aba_reuse(host, monkeypatch, failure):
    args, _, refs, _ = patched()
    original_refs = refs.clone()
    attachment = get_attachment(args["model"])
    graph_hook_state = copy.copy(attachment.post_input.__dict__)
    contexts, delegates = [], []
    prepare = LLLiteInvocation.prepare_model

    def track_prepare(self, *a):
        contexts.append(self)
        return prepare(self, *a)

    monkeypatch.setattr(LLLiteInvocation, "prepare_model", track_prepare)
    init = AnimaLLLitePatch.__init__

    def track_delegate(self, *a):
        init(self, *a)
        delegates.append(weakref.ref(self))
        if failure == "prepare" and len(delegates) == 2:
            raise RuntimeError("delegate preparation failed")

    monkeypatch.setattr(AnimaLLLitePatch, "__init__", track_delegate)
    if failure == "tile":
        host.fail_tile = 2
    elif failure == "cancel":
        host.interrupt_after = 2
    elif failure == "native":

        def fail(*a):
            raise RuntimeError("native encoding failed")

        monkeypatch.setattr(
            attachment.config.model_patch.model, "encode_conditioning", fail
        )
    if failure:
        with pytest.raises((RuntimeError, InterruptedError)):
            sampling.sample(**args)
    else:
        first = sampling.sample(**args)["samples"]
        b, _, _, _ = patched(arguments(hw=(16, 12), batch=2, steps=1))
        sampling.sample(**b)
        repeated = sampling.sample(**args)["samples"]
        torch.testing.assert_close(first, repeated)
        assert len({id(context) for context in contexts}) == 3
    gc.collect()
    assert all(ref() is None for ref in delegates)
    assert all(
        context.plan is context.model is context.attachment is context.batch is None
        and not context.delegates
        and not context.handles
        for context in contexts
    )
    for record in host.lllite_forwards:
        handle = record.options[DISPATCH_KEY]
        assert handle.invocation is handle.region is None
        assert tuple(record.options["model_patch_data"]) == record.keys
    assert attachment.post_input.__dict__ == graph_hook_state
    torch.testing.assert_close(refs, original_refs)
    assert not args["latent_image"]["samples"].any()


def test_lllite_host_isolation_repeated_in_process():
    results = []
    for _ in range(2):
        with isolated_host():
            args, _, _, _ = patched()
            results.append(sampling.sample(**args)["samples"])
    torch.testing.assert_close(*results)
