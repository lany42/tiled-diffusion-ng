# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native SDXL ControlNet routing, hint storage and lifetime contracts.

These CPU doubles do not establish real-host Union or Tile ControlNet inference.
"""

import weakref
from types import SimpleNamespace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import _native_control as native
from tiled_diffusion_ng.adapters import _sdxl_sampling as sdxl
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.geometry import crop, make_plan

from .host import ControlNet, ControlNetwork, arguments, cond, resize_hint


def control_with_hint(hint, algorithm="nearest-exact"):
    control = ControlNet()
    control.cond_hint_original = hint
    control.upscale_algorithm = algorithm
    return control


def storage_ids(tensors):
    # Count backing allocations once, including those retained by small views.
    return {(tensor.device, tensor.untyped_storage().data_ptr()) for tensor in tensors}


def small_plan(overlap=16):
    args = arguments(hw=(6, 8))
    return make_plan(
        resolve_adapter(args["model"]).describe(args["model"], args["latent_image"]),
        overlap,
    )


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
        # Like native KSampler, a positive-only control leaves the negative free.
        cond(4, control=ControlNet()),
    ]
    args = arguments(positive=cond(99, control=global_control), local_positive=locals)
    sampling.sample(**args)
    assert ["control" in c for c in host.prepared[1]] == [True, False, True, False]
    assert global_control.pre_runs == 0
    assert len(host.discovered) == 3


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


@pytest.mark.parametrize(
    "field,value",
    [
        # VAE-encoded, inpaint and hooked ControlNets need their own paths.
        ("vae", object()),
        ("latent_format", object()),
        ("concat_mask", True),
        ("extra_concat_orig", [torch.ones(1, 1, 16, 16)]),
        ("extra_hooks", object()),
        ("multigpu_clones", {"other_device": object()}),
        ("compression_ratio", 4),
        ("extra_conds", ["image"]),
        ("preprocess_image", lambda x: x * 2),
        ("extra_args", {"spatial": torch.ones(2, 2)}),
    ],
)
def test_control_capability_errors_before_host_sampling(host, field, value):
    control = ControlNet()
    setattr(control, field, value)
    error = "compression_ratio/extra_conds" if field == "extra_conds" else field
    with pytest.raises(ValueError, match=error):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls


@pytest.mark.parametrize(
    "extra_args",
    [
        {"control_type": [True]},
        {"control_type": (6,)},
        {"control_type": [6], "other": []},
    ],
)
def test_only_native_mode_lists_are_added_to_supported_metadata(host, extra_args):
    control = ControlNet(union=True)
    control.extra_args = extra_args
    with pytest.raises(ValueError, match="extra_args"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_classes", None),
        ("label_emb", [[SimpleNamespace(in_features=768)]]),
    ],
)
def test_controls_require_sdxl_architecture(host, field, value):
    control = ControlNet()
    setattr(control.control_model, field, value)
    with pytest.raises(ValueError, match="architecture"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls


def test_control_chain_cycle_is_rejected(host):
    control = ControlNet()
    control.previous_controlnet = control
    with pytest.raises(ValueError, match="Cyclic"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert control.previous_controlnet is control
    assert not host.common_calls


@pytest.mark.parametrize("implementation", ["wrapper", "network"])
def test_unknown_control_in_chain_is_rejected_and_partial_clones_closed(
    host, monkeypatch, implementation
):
    if implementation == "wrapper":
        # Even a subclass with native-looking fields needs its own lifecycle.
        class OtherControl(ControlNet):
            pass

        unsupported = OtherControl()
    else:
        # Native wrappers also carry non-UNet networks, such as MMDiT controls.
        class OtherNetwork(ControlNetwork):
            pass

        unsupported = ControlNet()
        unsupported.control_model = OtherNetwork()
    control = ControlNet(union=True)
    control.previous_controlnet = unsupported
    clones = []
    copy_control = ControlNet.copy

    def copy(original):
        clones.append(copy_control(original))
        return clones[-1]

    monkeypatch.setattr(ControlNet, "copy", copy)
    with pytest.raises(ValueError, match="only native image-hint ControlNet"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls
    assert len(clones) == 1
    assert clones[0].cleanups == 1 and clones[0].cond_hint_original is None
    assert control.previous_controlnet is unsupported
    assert control.cleanups == unsupported.cleanups == 0


def test_modes_and_prepared_caches_are_isolated_while_hint_pixels_are_shared(host):
    args = arguments(hw=(6, 8))
    plan = args["tile_plan"]
    first = ControlNet(union=True)
    first.cond_hint_original = torch.rand(1, 3, 17, 29)
    first.extra_args = {"control_type": [6, 1], "scale": 0.75}
    second = first.copy()
    second.extra_args = {"control_type": [1]}
    second.cond_hint_original = first.cond_hint_original.view_as(
        first.cond_hint_original
    )
    first.previous_controlnet = second
    source = first.cond_hint_original.clone()
    context = resolve_adapter(args["model"]).create_sampling_context(plan)
    clones = []
    try:
        for region in plan.regions:
            positive, negative = context.prepare_pair(
                [{"control": first}, {"control": second}], [{"control": first}], region
            )
            a, b = [entry["control"] for entry in positive]
            assert a is negative[0]["control"]
            assert a.previous_controlnet is b
            clones.extend((a, b))
        context.finalize_preparation()
        assert len(host.resize_calls) == 1
        assert len({id(c.extra_args["control_type"]) for c in clones}) == 8
        for a, b in zip(clones[::2], clones[1::2], strict=True):
            assert a.cond_hint_original is b.cond_hint_original
            assert a.control_model is b.control_model is first.control_model
            assert a.extra_args == first.extra_args
            assert b.extra_args == second.extra_args
            a.pre_run()
            a.predict(torch.zeros(1, 4, *plan.tile_hw), torch.tensor([0.5]))
        assert len({c.cond_hint.untyped_storage().data_ptr() for c in clones}) == 8
        caches = [c.cond_hint.clone() for c in clones]
        hint = clones[0].cond_hint_original.clone()
        clones[0].extra_args["control_type"].append(7)
        clones[0].cond_hint.fill_(42)
        assert first.extra_args == {"control_type": [6, 1], "scale": 0.75}
        assert all(c.extra_args == first.extra_args for c in clones[2::2])
        for clone, cache in zip(clones[1:], caches[1:], strict=True):
            torch.testing.assert_close(clone.cond_hint, cache)
        torch.testing.assert_close(clones[0].cond_hint_original, hint)
        torch.testing.assert_close(first.cond_hint_original, source)
    finally:
        context.close()
    assert all(
        c.cleanups == 1
        and c.previous_controlnet is None
        and c.cond_hint_original is None
        and c.cond_hint is None
        and c.timestep_range is None
        and c.model_sampling_current is None
        for c in clones
    )
    assert first.previous_controlnet is second
    assert first.cleanups == second.cleanups == 0


def test_local_mixed_chains_preserve_each_mode_and_pair(host, monkeypatch):
    source = torch.rand(1, 3, 17, 29)
    modes = [{}, {"control_type": []}, {"control_type": [6]}, {"control_type": [1]}]
    originals, locals, expected = [], [], {}
    for i, mode_args in enumerate(modes):
        union, traditional = ControlNet(union=True), ControlNet()
        union.extra_args = mode_args
        for control in (union, traditional):
            control.cond_hint_original = source.view_as(source)
            expected[control.control_model] = control.extra_args.copy()
        union.previous_controlnet = traditional
        originals.extend((union, traditional))
        locals.append(cond(i + 1, control=union, control_apply_to_uncond=True))
    saved, evaluated = [], set()
    predict = ControlNet.predict

    def inspect(branches):
        for positive, negative in zip(*branches, strict=True):
            first = positive["control"]
            assert first is negative["control"]
            second = first.previous_controlnet
            assert first.cond_hint_original is second.cond_hint_original
            saved.extend((first, second))
        assert len(set(saved)) == 8
        for clone, original in zip(saved, originals, strict=True):
            assert clone is not original
            assert clone.control_model is original.control_model

    def evaluate(clone, x, sigma):
        assert clone.extra_args == expected[clone.control_model]
        evaluated.add(clone)
        return predict(clone, x, sigma)

    monkeypatch.setattr(ControlNet, "predict", evaluate)
    host.mutate_prepared = inspect
    # An unsupported global control must be wholly replaced by local positives.
    args = arguments(
        hw=(6, 8), positive=cond(99, control=object()), local_positive=locals
    )
    sampling.sample(**args)
    assert len(host.common_calls) == len(host.resize_calls) == 1
    assert evaluated == set(saved)
    for clone, original in zip(saved, originals, strict=True):
        assert clone.extra_args == original.extra_args == expected[clone.control_model]
        assert clone.cleanups > 0 and clone.cond_hint_original is None
        assert original.cleanups == 0 and original.cond_hint_original is not None


@pytest.mark.parametrize("count,overlap,canvas_retained", [(1, 0, False), (4, 0, True)])
def test_storage_sharing_and_duplicate_rectangles(
    host, monkeypatch, count, overlap, canvas_retained
):
    from comfy import utils

    canvases = []

    def normalize(*args):
        canvas = host.resize(*args)
        canvases.append(weakref.ref(canvas))
        return canvas

    monkeypatch.setattr(utils, "common_upscale", normalize)
    plan = small_plan(overlap)
    source = torch.arange(2 * 3 * 17 * 29, dtype=torch.float32).reshape(2, 3, 17, 29)
    a = control_with_hint(source)
    b = control_with_hint(source.view_as(source))
    b.strength = 0.25
    b.timestep_percent_range = (0.3, 0.8)
    context = sdxl.SDXLSamplingContext(plan)
    clones = []
    try:
        for region in plan.regions[:count]:
            p, n = context.prepare_pair(
                [{"control": a}, {"control": b}], [{"control": a}], region
            )
            assert p[0]["control"] is n[0]["control"]
            clones.extend(entry["control"] for entry in p)
        assert not host.resize_calls
        assert all(c.cond_hint_original is None for c in clones)
        context.finalize_preparation()
        assert len(host.resize_calls) == 1
        assert not context.hint_groups
        hints = [c.cond_hint_original for c in clones]
        for first, second in zip(clones[::2], clones[1::2], strict=True):
            assert first is not second
            assert first.cond_hint_original is second.cond_hint_original
            assert first.control_model is a.control_model
            assert second.control_model is b.control_model
            assert second.strength == 0.25
            assert second.timestep_percent_range == (0.3, 0.8)
        # Smaller crops are compacted; views covering the canvas, including
        # an exact partition, keep one shared canvas.
        storages = storage_ids(hints)
        if canvas_retained:
            assert canvases[0]() is not None
            assert storages == storage_ids([canvases[0]()])
        else:
            assert len(storages) == count
            assert canvases[0]() is None
        ph, pw = plan.pixel_hw
        normalized = resize_hint(source, pw, ph, a.upscale_algorithm, "center")
        for region, hint in zip(plan.regions, hints[::2]):
            torch.testing.assert_close(
                hint, crop(normalized, region.pixel_sampling), rtol=0, atol=0
            )
    finally:
        context.close()


def test_source_view_and_resize_configurations_remain_distinct(host):
    plan = small_plan()
    base = torch.arange(2 * 3 * 18 * 18, dtype=torch.float16).reshape(2, 3, 18, 18)
    hints = [
        base[..., :17, :17],
        base[..., 1:, :17],  # Same storage, different offset.
        base.transpose(-1, -2)[..., :17, :17],  # Same shape/offset, different strides.
        base.view(torch.bfloat16)[..., :17, :17],  # Same view, different dtype.
        base[..., :16, :17],  # Different shape.
        base[..., :17, :17].clone(),  # Equal pixels, independent storage.
    ]
    controls = [control_with_hint(hint) for hint in hints]
    controls.append(control_with_hint(hints[0].view_as(hints[0]), "bilinear"))
    context = sdxl.SDXLSamplingContext(plan)
    try:
        context.prepare_pair([{"control": c} for c in controls], [], plan.regions[0])
        context.finalize_preparation()
        assert len(host.resize_calls) == len(controls)
        assert len(storage_ids(c.cond_hint_original for c in context.controls)) == len(
            controls
        )
        ph, pw = plan.pixel_hw
        for original, clone in zip(controls, context.controls, strict=True):
            expected = crop(
                resize_hint(
                    original.cond_hint_original,
                    pw,
                    ph,
                    original.upscale_algorithm,
                    "center",
                ),
                plan.regions[0].pixel_sampling,
            )
            torch.testing.assert_close(
                clone.cond_hint_original, expected, rtol=0, atol=0
            )
    finally:
        context.close()


def test_matching_dimensions_still_apply_host_normalization(host, monkeypatch):
    from comfy import utils

    plan = small_plan()
    source = torch.zeros(1, 3, *plan.pixel_hw)
    control = control_with_hint(source)
    calls = []

    def transform(hint, width, height, algorithm, policy):
        calls.append((width, height, algorithm, policy))
        return resize_hint(hint, width, height, algorithm, policy) + 0.125

    monkeypatch.setattr(utils, "common_upscale", transform)
    context = sdxl.SDXLSamplingContext(plan)
    try:
        p, _ = context.prepare_pair([{"control": control}], [], plan.regions[0])
        context.finalize_preparation()
        assert calls == [(64, 48, "nearest-exact", "center")]
        assert (p[0]["control"].cond_hint_original == 0.125).all()
        assert not source.any()
    finally:
        context.close()


def test_discovery_pins_source_until_materialization_then_releases_it(
    host, monkeypatch
):
    from comfy import utils

    # The recording double would retain the source; use the plain resize.
    monkeypatch.setattr(utils, "common_upscale", resize_hint)
    context = sdxl.SDXLSamplingContext(small_plan())
    source = torch.rand(1, 3, 17, 29)
    ref = weakref.ref(source)
    control = control_with_hint(source)
    try:
        context.prepare_pair([{"control": control}], [], context.plan.regions[0])
        del source, control
        assert ref() is not None
        context.finalize_preparation()
        assert ref() is None
        assert not context.hint_groups
    finally:
        context.close()


@pytest.mark.parametrize("scenario", ["global", "aliased_locals", "distinct_locals"])
def test_sampling_matches_copy_reference_without_repreparing_caches(
    host, monkeypatch, scenario
):
    source = torch.rand(1, 3, 17, 29, generator=torch.Generator().manual_seed(8))
    originals = []
    locals = []
    for i in range(1 if scenario == "global" else 4):
        hint = (
            source + i / 10 if scenario == "distinct_locals" else source.view_as(source)
        )
        first = control_with_hint(hint)
        second = control_with_hint(hint.view_as(hint))
        first.previous_controlnet = second
        first.strength = 0.8
        second.strength = -0.4
        first.timestep_percent_range = (0.0, 0.8)
        second.timestep_percent_range = (0.3, 1.0)
        first.control_model.weight = torch.arange(4.0)
        second.control_model.weight = torch.arange(4.0) + 2
        first.cond_hint = torch.tensor([17.0])
        first.timestep_range = (9.0, 8.0)
        originals.extend((first, second))
        locals.append(cond(i + 1, control=first, control_apply_to_uncond=True))
    args = arguments(hw=(6, 8), steps=3, positive=locals[0])
    args["latent_image"]["samples"] = torch.zeros(3, 4, 6, 8)
    if scenario != "global":
        args["local_positive"] = locals
    snapshots = [
        (
            c.cond_hint_original.clone(),
            c.control_model.weight.clone(),
            c.cond_hint,
            c.timestep_range,
            c.previous_controlnet,
        )
        for c in originals
    ]
    clones = []

    def inspect(branches):
        for positive, negative in zip(*branches, strict=True):
            first = positive["control"]
            assert first is negative["control"]
            assert (
                first.cond_hint_original is first.previous_controlnet.cond_hint_original
            )
            clones.extend((first, first.previous_controlnet))

    host.mutate_prepared = inspect
    actual = sampling.sample(**args)["samples"]
    assert len(host.resize_calls) == (4 if scenario == "distinct_locals" else 1)
    assert len(clones) == 8
    assert all(c.hint_preparations == 1 for c in clones)

    def normalize_then_copy(group):
        ph, pw = group.pixel_hw
        normalized = resize_hint(
            group.source, pw, ph, group.algorithm, group.crop_policy
        )
        for rect, controls in group.controls.items():
            for control in controls:
                control.cond_hint_original = crop(normalized, rect).clone()

    monkeypatch.setattr(native._HintGroup, "materialize", normalize_then_copy)
    host.mutate_prepared = None
    expected = sampling.sample(**args)["samples"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not args["latent_image"]["samples"].any()
    for original, (hint, weight, cache, timestep, previous) in zip(
        originals, snapshots, strict=True
    ):
        torch.testing.assert_close(original.cond_hint_original, hint, rtol=0, atol=0)
        torch.testing.assert_close(
            original.control_model.weight, weight, rtol=0, atol=0
        )
        assert original.cond_hint is cache and original.timestep_range == timestep
        assert original.previous_controlnet is previous
        assert original.pre_runs == original.cleanups == original.hint_preparations == 0
    assert args["model"].wrappers == {}


@pytest.mark.parametrize("failure", [None, "normalize", "model", "cleanup"])
def test_groups_do_not_accumulate_canvases_and_cleanup_releases_hints(
    host, monkeypatch, failure
):
    from comfy import utils

    controls = [control_with_hint(torch.rand(1, 3, 17, 29)) for _ in range(4)]
    args = arguments(
        hw=(6, 8), local_positive=[cond(i, control=c) for i, c in enumerate(controls)]
    )
    adapter = resolve_adapter(args["model"])
    contexts, canvases, owned_hints = [], [], []
    create = adapter.create_sampling_context

    def create_context(plan):
        context = create(plan)
        contexts.append(context)
        close = context.close

        def capture_and_close():
            for control in context.controls:
                for hint in (control.cond_hint_original, control.cond_hint):
                    if hint is not None:
                        owned_hints.append(weakref.ref(hint))
            close()

        context.close = capture_and_close
        return context

    def normalize(*a):
        # Local groups retain only compact crops, not their temporary canvases.
        assert all(ref() is None for ref in canvases)
        if failure == "normalize" and len(canvases) == 1:
            raise RuntimeError("normalization failed")
        canvas = resize_hint(*a)
        canvases.append(weakref.ref(canvas))
        return canvas

    monkeypatch.setattr(adapter, "create_sampling_context", create_context)
    monkeypatch.setattr(utils, "common_upscale", normalize)
    if failure == "model":
        host.fail_tile = 3
    elif failure == "cleanup":
        # Host cleanup runs on success; a failing repeat must not stop release.
        cleanup, calls = ControlNet.cleanup, {}

        def fail_on_repeat(control):
            cleanup(control)
            calls[id(control)] = calls.get(id(control), 0) + 1
            if calls[id(control)] == 2:
                raise RuntimeError("cleanup failed")

        monkeypatch.setattr(ControlNet, "cleanup", fail_on_repeat)
    if failure in (None, "cleanup"):
        sampling.sample(**args)
    else:
        with pytest.raises(RuntimeError):
            sampling.sample(**args)
    assert canvases and owned_hints
    assert all(ref() is None for ref in canvases + owned_hints)
    assert all(
        not c.controls and not c.hint_groups and c.plan is None for c in contexts
    )
    if failure == "normalize":
        assert not host.common_calls
    assert all(c.cond_hint_original is not None and c.cleanups == 0 for c in controls)
