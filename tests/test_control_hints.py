# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""CPU storage/lifetime contracts, separate from real ComfyUI acceptance."""

import gc
import weakref

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import _sdxl_sampling as sdxl
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.geometry import crop, make_plan

from .host import ControlNet, cond, resize_hint
from .test_sampling import arguments


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


@pytest.mark.parametrize(
    "count,overlap,canvas_retained",
    [
        (1, 0, False),
        (2, 0, False),
        (4, 0, True),
        (1, 32, False),
        (2, 32, True),
        (4, 16, True),
    ],
)
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
    assert a.cond_hint_original is not b.cond_hint_original
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
        storages = storage_ids(hints)
        ph, pw = plan.pixel_hw
        if canvas_retained:
            assert canvases[0]() is not None
            assert storages == storage_ids([canvases[0]()])
        else:
            assert len(storages) == count
            assert canvases[0]() is None
        normalized = resize_hint(source, pw, ph, a.upscale_algorithm, "center")
        for region, hint in zip(plan.regions, hints[::2]):
            torch.testing.assert_close(
                hint, crop(normalized, region.pixel_sampling), rtol=0, atol=0
            )
    finally:
        context.close()


@pytest.mark.parametrize("algorithm", ["nearest-exact", "bilinear", "bicubic", "area"])
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("channels_last", [False, True])
@pytest.mark.parametrize("source_hw", [(17, 29), (48, 64)])
def test_canvas_views_match_normalize_then_copy_and_prepared_layout(
    host, algorithm, dtype, channels_last, source_hw
):
    plan = small_plan()
    source = torch.rand(
        2, 3, *source_hw, generator=torch.Generator().manual_seed(42)
    ).to(dtype)
    if channels_last:
        source = source.contiguous(memory_format=torch.channels_last)
    before = source.clone()
    control = control_with_hint(source, algorithm)
    context = sdxl.SDXLSamplingContext(plan)
    try:
        for region in plan.regions:
            context.prepare_pair([{"control": control}], [], region)
        context.finalize_preparation()
        assert len(host.resize_calls) == 1  # Even when source dimensions match.
        ph, pw = plan.pixel_hw
        normalized = resize_hint(source, pw, ph, algorithm, "center")
        for region, clone in zip(plan.regions, context.controls, strict=True):
            expected = crop(normalized, region.pixel_sampling).clone()
            actual = clone.cond_hint_original
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            h, w = expected.shape[-2:]
            # Ordinary host preparation still resizes each clone's hint. Its
            # resulting layout must match the previous compact-crop input path.
            torch.testing.assert_close(
                resize_hint(actual, w, h, algorithm, "center"),
                resize_hint(expected, w, h, algorithm, "center"),
                rtol=0,
                atol=0,
                check_stride=True,
            )
        torch.testing.assert_close(source, before, rtol=0, atol=0, check_stride=True)
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
            ).clone()
            torch.testing.assert_close(
                clone.cond_hint_original, expected, rtol=0, atol=0
            )
        key = sdxl._hint_key(hints[0], "nearest-exact", "center", plan.pixel_hw)
        assert key == sdxl._hint_key(
            hints[0].view_as(hints[0]), "nearest-exact", "center", plan.pixel_hw
        )
        assert key != sdxl._hint_key(
            hints[0], "nearest-exact", "disabled", plan.pixel_hw
        )
        assert key != sdxl._hint_key(hints[0], "nearest-exact", "center", (64, 48))
        assert key != sdxl._hint_key(
            torch._neg_view(hints[0]), "nearest-exact", "center", plan.pixel_hw
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

    monkeypatch.setattr(utils, "common_upscale", resize_hint)
    context = sdxl.SDXLSamplingContext(small_plan())
    source = torch.rand(1, 3, 17, 29)
    ref = weakref.ref(source)
    control = control_with_hint(source)
    try:
        context.prepare_pair([{"control": control}], [], context.plan.regions[0])
        del source, control
        gc.collect()
        assert ref() is not None
        context.finalize_preparation()
        gc.collect()
        assert ref() is None
        assert not context.hint_groups
    finally:
        context.close()


@pytest.mark.parametrize("scenario", ["global", "aliased_locals", "distinct_locals"])
@pytest.mark.parametrize("hint_batch", [1, 2, 4])
def test_sampling_matches_copy_reference_without_repreparing_caches(
    host, monkeypatch, scenario, hint_batch
):
    source = torch.rand(
        hint_batch, 3, 17, 29, generator=torch.Generator().manual_seed(8)
    )
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
    assert len(host.tile_calls) == 3 * 3 * 4

    def normalize_then_copy(group):
        ph, pw = group.pixel_hw
        normalized = resize_hint(
            group.source, pw, ph, group.algorithm, group.crop_policy
        )
        for rect, controls in group.controls.items():
            for control in controls:
                control.cond_hint_original = crop(normalized, rect).clone()

    monkeypatch.setattr(sdxl._HintGroup, "materialize", normalize_then_copy)
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


@pytest.mark.parametrize(
    "failure", [None, "normalize", "crop", "host_prepare", "model", "cancel"]
)
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

    def failing_crop(tensor, rect):
        if len(canvases) == 2:
            raise RuntimeError("crop failed")
        return crop(tensor, rect)

    def fail_preparation(branches):
        raise RuntimeError("host preparation failed")

    monkeypatch.setattr(adapter, "create_sampling_context", create_context)
    monkeypatch.setattr(utils, "common_upscale", normalize)
    if failure == "crop":
        monkeypatch.setattr(sdxl, "crop", failing_crop)
    elif failure == "host_prepare":
        host.mutate_prepared = fail_preparation
    elif failure == "model":
        host.fail_tile = 3
    elif failure == "cancel":
        host.interrupt_after = 3
    if failure is None:
        sampling.sample(**args)
    else:
        with pytest.raises((RuntimeError, InterruptedError)):
            sampling.sample(**args)
    gc.collect()
    assert canvases and owned_hints
    assert all(ref() is None for ref in canvases + owned_hints)
    assert all(
        not c.controls and not c.hint_groups and c.plan is None for c in contexts
    )
    if failure in ("normalize", "crop"):
        assert not host.common_calls
    assert all(c.cond_hint_original is not None and c.cleanups == 0 for c in controls)
