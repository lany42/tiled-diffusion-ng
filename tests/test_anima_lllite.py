# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Offline CPU contracts for tiled native LLLite, not real-host acceptance.

The native activation gate, encoding and residual math belong to the host;
these doubles only observe which crop and dispatcher each tile receives.
"""

import copy
import sys
import weakref
from dataclasses import replace
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
    apply_lllite,
    get_attachment,
)
from tiled_diffusion_ng.geometry import image_views, make_plan

from .host import CONST, Model, arguments, cond, copy_containers
from .lllite_host import (
    TARGETS,
    AnimaLLLite,
    AnimaLLLiteAttentionPatch,
    AnimaLLLitePatch,
)


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
    args = arguments("anima") if args is None else args.copy()
    source = args["model"]
    groups = args["latent_image"]["samples"].shape[0] if groups is None else groups
    refs = references(args["tile_plan"], groups, channels)
    model_patch = Model()
    model_patch.model = AnimaLLLite() if weights is None else weights
    args["model"] = apply_lllite(
        source, model_patch, args["tile_plan"], refs, **settings
    )
    return args, source, refs, model_patch


@pytest.mark.parametrize(
    "combined,hw,batch,groups,channels,cfg,entries",
    [
        (False, (12, 16), 1, 1, 3, 4, 1),
        (True, (13, 15), 2, 2, 4, 4, 2),
        (False, (13, 15), 2, 1, 3, 1, 2),
    ],
    ids=["single", "batched_rgba", "broadcast_cfg1"],
)
def test_crop_routing_batches_branches_and_native_hooks(
    host, combined, hw, batch, groups, channels, cfg, entries
):
    args = arguments("anima", batch=batch, cfg=cfg, hw=hw)
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
    assert args["model"].model_patches_models() == [patch]
    # ModelPatcher moves patches with .to(); the graph-owned hook stays in place.
    assert attachment.post_input.to(torch.float16) is attachment.post_input
    assert (
        attachment.attn1.patch
        is attachment.attn2.patch
        is attachment.mlp.patch
        is attachment.post_input
    )
    assert type(attachment.attn1) is AnimaLLLiteAttentionPatch
    assert args["model"].clone().get_attachment(ATTACHMENT_KEY) is attachment
    sampling.sample(**args)
    assert len(host.common_calls) == 1 and host.patch_models == [patch]
    assert len(host.tile_calls) == 4
    branches = entries + int(cfg != 1)
    calls_per_tile = 1 if combined else branches
    assert len(host.lllite_forwards) == 4 * calls_per_tile
    for call_index, record in enumerate(host.lllite_forwards):
        index = call_index // calls_per_tile
        # Each view receives its own overlap-inclusive RGB crop per group.
        expected = refs[index::4, ..., :3].movedim(-1, 1).clamp(0, 1) * 2 - 1
        torch.testing.assert_close(patch.model.encodings[call_index], expected)
        embedding = (
            F.avg_pool2d(expected.mean(1, keepdim=True), 16).flatten(2).transpose(1, 2)
        )
        assert record.keys == (attachment.post_input,)
        torch.testing.assert_close(record.embedding[0], embedding)
        assert all(k and v for k, v in record.cross_kv)
        # Native per-forward reset, not a dispatcher or embedding cache.
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
    torch.testing.assert_close(refs, before)
    assert not args["latent_image"]["samples"].any()
    assert DISPATCH_KEY not in args["model"].model_options["transformer_options"]


def test_native_delegates_get_strength_and_live_sigma_thresholds(host, monkeypatch):
    args, _, _, _ = patched(strength=-2, start_percent=0.25, end_percent=0.75)
    delegates = []
    original = AnimaLLLitePatch.__init__

    def init(self, *a):
        original(self, *a)
        delegates.append((self.strength, self.sigma_start, self.sigma_end))

    monkeypatch.setattr(AnimaLLLitePatch, "__init__", init)
    sampling.sample(**args)

    class Shifted(CONST):
        def percent_to_sigma(self, percent):
            return 3 * (1 - percent)

    # Thresholds follow each invocation's model sampling, not apply time.
    args["model"] = args["model"].clone()
    args["model"].sampling = Shifted()
    sampling.sample(**args)
    assert delegates == [(-2, 0.75, 0.25)] * 4 + [(-2, 2.25, 0.75)] * 4


def test_duplicate_application_is_rejected_through_clones(host):
    args, _, refs, patch = patched()
    with pytest.raises(ValueError, match="already applied"):
        apply_lllite(args["model"].clone(), patch, args["tile_plan"], refs, strength=0)


@pytest.mark.parametrize("native_first", [False, True])
def test_native_chaining_rejected_in_either_order(host, native_first):
    args, source, refs, patch = patched(strength=0)
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
        (lambda refs: refs[..., :2], "RGB/RGBA"),
        (lambda refs: refs[:, :-1], "dimensions"),
    ],
    ids=["count", "channels", "size"],
)
def test_malformed_references_rejected_before_native_resize(host, change, error):
    args, source, refs, patch = patched()
    with pytest.raises(ValueError, match=error):
        apply_lllite(source, patch, args["tile_plan"], change(refs), strength=0)
    assert not host.resize_calls


def test_reference_groups_are_not_arbitrarily_repeated(host):
    args, _, _, _ = patched(arguments("anima", batch=2), groups=3, strength=0)
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
        (lambda weights: setattr(weights, "model_dim", 8), "model width"),
        (lambda weights: setattr(weights, "module_names", set()), "nonempty"),
        (
            lambda weights: weights.module_names.add("lllite_dit_blocks_28_mlp_layer1"),
            "backbone depth",
        ),
        (
            lambda weights: weights.module_names.add(
                "lllite_dit_blocks_0_cross_attn_k_proj"
            ),
            "module target",
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
            "module width",
        ),
    ],
    ids=[
        "mask",
        "width",
        "empty",
        "depth",
        "unknown_target",
        "block_count",
        "missing",
        "module",
    ],
)
def test_invalid_weights_rejected_at_zero_strength(host, mutation, error):
    args, source, refs, patch = patched()
    mutation(patch.model)
    with pytest.raises(ValueError, match=error):
        apply_lllite(source, patch, args["tile_plan"], refs, strength=0)


def test_plain_anima_works_on_hosts_without_native_lllite(host, monkeypatch):
    args, source, refs, patch = patched()
    monkeypatch.delattr(sys.modules["comfy.ldm.anima"], "lllite")
    monkeypatch.setitem(sys.modules, "comfy.ldm.anima.lllite", None)
    sampling.sample(**arguments("anima"))
    with pytest.raises(ValueError, match="native Anima LLLite APIs are required"):
        apply_lllite(source, patch, args["tile_plan"], refs)


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


def test_sparse_coverage_within_actual_backbone_depth(host):
    name = "lllite_dit_blocks_39_mlp_layer1"
    args, _, _, patch = patched(
        arguments("anima", depth=40), weights=AnimaLLLite(targets=[name])
    )
    sampling.sample(**args)
    assert {call[:2] for call in patch.model.calls} == {(39, "mlp_layer1")}
    args["model"].model.diffusion_model.blocks.pop()
    with pytest.raises(ValueError, match="backbone depth"):
        sampling.sample(**args)


@pytest.mark.parametrize("change", ["unknown_version", "schema", "orphan", "no_hooks"])
def test_bad_or_orphaned_declarations_rejected(host, change):
    args, _, refs, patch = patched(strength=0)
    model = args["model"]
    attachment = get_attachment(model)
    if change == "unknown_version":
        model.attachments[ATTACHMENT_KEY + ".v2"] = attachment
    elif change == "schema":
        model.attachments[ATTACHMENT_KEY] = replace(attachment, schema_version=2)
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
    "change,later",
    [
        ("remove_one", False),
        ("native", False),
        ("duplicate", False),
        ("targets", False),
        ("replace", True),
        ("attachment", True),
    ],
)
def test_altered_live_hooks_rejected(host, change, later):
    args, _, _, patch = patched(strength=0)

    def mutate(options):
        hooks = options["transformer_options"]["patches"]
        if change == "remove_one":
            hooks.pop("post_input")
        elif change == "native":
            hooks["post_input"].append(AnimaLLLitePatch(patch, None, None, 0, 1, 0))
        elif change == "duplicate":
            # A repeated hook would apply its residual twice.
            hooks["mlp_patch"] *= 2
        elif change == "targets":
            hooks["attn2_patch"][0].targets = {"k": "cross_attn_q_proj"}
        elif change == "replace":
            old = hooks["attn1_patch"][0]
            hooks["attn1_patch"] = [AnimaLLLiteAttentionPatch(old.patch, old.targets)]
        else:
            host.latest_clone.attachments.clear()

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


@pytest.mark.parametrize(
    "stage,remove_all,strength",
    [("apply_model", True, 0.0), ("diffusion_model", False, 1.0)],
)
def test_hooks_removed_inside_native_wrapper_are_rejected(
    host, stage, remove_all, strength
):
    # The forward guard does not depend on whether the patch is active.
    args, _, _, patch = patched(strength=strength)

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
            for _ in range(2):
                assert forward(options)() is None
    finally:
        context.close()


def test_ordinary_ksampler_requires_dispatch_even_when_inactive(host):
    args, _, _, _ = patched(strength=0)
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
        ("branches", "batch mapping"),
        ("img", "embedded image"),
        ("no_data", "fresh model_patch_data"),
        ("nested", "Nested"),
    ],
)
def test_forward_validation_precedes_zero_strength(host, change, error):
    args, _, _, _ = patched(arguments("anima", batch=2), strength=0)
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
            elif change == "branches":
                transformer["cond_or_uncond"] = [0, 1]
            elif change == "img":
                call["img"] = call["img"][:, :, :-1]
            elif change == "no_data":
                transformer.pop("model_patch_data")
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


def test_bad_encoded_token_count_is_rejected(host, monkeypatch):
    args, _, _, patch = patched()
    encode = patch.model.encode_conditioning
    monkeypatch.setattr(
        patch.model, "encode_conditioning", lambda image: encode(image)[:, :-1]
    )
    with pytest.raises(ValueError, match="encoded token count"):
        sampling.sample(**args)


def test_local_prompts_route_per_view_while_the_model_patch_stays_active(host):
    args, _, _, patch = patched(
        arguments("anima", batch=2, local_positive=[cond(i) for i in range(4)])
    )
    host.combine_anima_conditions = True
    sampling.sample(**args)
    for index, (branches, _, _, _) in enumerate(host.tile_calls):
        assert branches[0][0]["cross_attn"].mean() == index % 4
    # One combined forward per view and evaluation, each with its crop.
    assert len(patch.model.encodings) == len(host.tile_calls)
    assert len(host.common_calls) == 1 and host.patch_models == [patch]


@pytest.mark.parametrize("failure", [None, "prepare", "native"])
def test_lifecycle_closes_handles_delegates_and_aba_reuse(host, monkeypatch, failure):
    args, _, refs, _ = patched(arguments("anima", hw=(13, 15)))
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
    if failure == "native":

        def fail(*a):
            raise RuntimeError("native encoding failed")

        monkeypatch.setattr(
            attachment.config.model_patch.model, "encode_conditioning", fail
        )
    if failure:
        with pytest.raises(RuntimeError):
            sampling.sample(**args)
    else:
        first = sampling.sample(**args)["samples"]
        b, _, _, _ = patched(arguments("anima", hw=(16, 12), batch=2))
        sampling.sample(**b)
        repeated = sampling.sample(**args)["samples"]
        torch.testing.assert_close(first, repeated)
        assert len({id(context) for context in contexts}) == 3
    assert delegates and all(ref() is None for ref in delegates)
    assert all(
        context.plan is context.model is context.attachment is context.batch is None
        and not context.delegates
        and not context.handles
        for context in contexts
    )
    for record in host.lllite_forwards:
        handle = record.options[DISPATCH_KEY]
        assert handle.invocation is handle.region is None
    assert attachment.post_input.__dict__ == graph_hook_state
    torch.testing.assert_close(refs, original_refs)
    assert not args["latent_image"]["samples"].any()
