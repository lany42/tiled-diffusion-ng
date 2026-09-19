# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native control ownership contracts, not real-host Union inference."""

from types import SimpleNamespace

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter

from .host import ControlNet, ControlNetwork, cond
from .test_sampling import arguments


@pytest.mark.parametrize("union", [False, True])
@pytest.mark.parametrize(
    "mode_args",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"control_type": []}, id="auto"),
        pytest.param({"control_type": [6]}, id="tile"),
        pytest.param({"control_type": [1]}, id="depth"),
        pytest.param({"control_type": [1, 6, 1]}, id="ordered_with_duplicates"),
        pytest.param({"control_type": [99, -1]}, id="host_validated_ids"),
    ],
)
def test_mode_metadata_reaches_control_evaluation(host, monkeypatch, union, mode_args):
    control = ControlNet(union=union)
    expected = {"scale": 0.75, **mode_args}
    control.extra_args = expected
    evaluated = []
    predict = ControlNet.predict

    def evaluate(clone, x, sigma):
        assert clone.control_model is control.control_model
        assert clone.extra_args == expected
        assert clone.extra_args is not control.extra_args
        if "control_type" in expected:
            assert clone.extra_args["control_type"] is not expected["control_type"]
        evaluated.append(clone)
        # This double observes metadata only. Mode interpretation and supported
        # ID validation belong to the real host network, not tiled sampling.
        return predict(clone, x, sigma)

    monkeypatch.setattr(ControlNet, "predict", evaluate)
    args = arguments(
        hw=(6, 8), positive=cond(1, control=control, control_apply_to_uncond=True)
    )
    sampling.sample(**args)
    assert len(host.common_calls) == 1
    assert len(evaluated) == args["steps"] * host.evaluations_per_step * 4 * 2
    assert len(set(evaluated)) == 4
    assert control.extra_args is expected
    assert control.extra_args == {"scale": 0.75, **mode_args}
    assert control.cond_hint_original is not None and control.cleanups == 0


@pytest.mark.parametrize("modes", [[], [6, 1]])
def test_modes_and_prepared_caches_are_isolated_while_hint_pixels_are_shared(
    host, modes
):
    args = arguments(hw=(6, 8))
    plan = args["tile_plan"]
    first = ControlNet(union=True)
    first.cond_hint_original = torch.rand(1, 3, 17, 29)
    first.extra_args = {"control_type": modes.copy()}
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
        assert len({id(c.extra_args) for c in clones}) == 8
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
        clones[0].extra_args["scale"] = 0.5
        clones[0].cond_hint.fill_(42)
        assert first.extra_args == {"control_type": modes}
        assert all(c.extra_args == first.extra_args for c in clones[2::2])
        assert all(c.extra_args == second.extra_args for c in clones[1::2])
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


@pytest.mark.parametrize("union_first", [False, True])
def test_local_mixed_chains_preserve_each_mode_and_pair(host, monkeypatch, union_first):
    source = torch.rand(1, 3, 17, 29)
    modes = [{}, {"control_type": []}, {"control_type": [6]}, {"control_type": [1]}]
    originals, locals, expected = [], [], {}
    for i, mode_args in enumerate(modes):
        union, traditional = ControlNet(union=True), ControlNet()
        union.extra_args = mode_args
        for control in (union, traditional):
            control.cond_hint_original = source.view_as(source)
            expected[control.control_model] = control.extra_args.copy()
        first, second = (union, traditional) if union_first else (traditional, union)
        first.previous_controlnet = second
        originals.extend((first, second))
        locals.append(cond(i + 1, control=first, control_apply_to_uncond=True))
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


def test_host_mode_validation_error_propagates_and_cleans_clones(host, monkeypatch):
    control = ControlNet(union=True)
    control.control_model.num_control_type = 6
    control.extra_args = {"control_type": [6]}

    def reject_mode(clone, x, sigma):
        assert clone.extra_args == {"control_type": [6]}
        raise ValueError("host rejected this network's mode")

    monkeypatch.setattr(ControlNet, "predict", reject_mode)
    with pytest.raises(ValueError, match="host rejected"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert len(host.common_calls) == 1
    assert all(
        c.cond_hint_original is None and c.cleanups == 1 for c in host.discovered
    )
    assert control.extra_args == {"control_type": [6]}
    assert control.cleanups == 0 and control.cond_hint_original is not None


@pytest.mark.parametrize(
    "extra_args",
    [
        {"control_type": [6.0]},
        {"control_type": [True]},
        {"control_type": ["6"]},
        {"control_type": [[6]]},
        {"control_type": (6,)},
        {"control_type": torch.tensor([6])},
        {"control_type": {"mode": 6}},
        {"control_type": [6], "other": []},
        {"control_type": [6], "spatial": torch.ones(2, 2)},
    ],
)
def test_only_native_mode_lists_are_added_to_supported_metadata(host, extra_args):
    control = ControlNet(union=True)
    control.extra_args = extra_args
    with pytest.raises(ValueError, match="extra_args"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls


@pytest.mark.parametrize("union", [False, True])
@pytest.mark.parametrize(
    "field,value",
    [
        ("dims", 3),
        ("in_channels", 9),
        ("num_classes", None),
        ("label_emb", [[SimpleNamespace(in_features=768)]]),
        ("input_hint_block", [SimpleNamespace(in_channels=4)]),
    ],
)
def test_native_controls_still_require_sdxl_architecture(host, union, field, value):
    control = ControlNet(union=union)
    setattr(control.control_model, field, value)
    with pytest.raises(ValueError, match="architecture"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls


@pytest.mark.parametrize("implementation", ["wrapper", "network"])
def test_unknown_implementations_in_chains_are_rejected_and_partial_clones_closed(
    host, monkeypatch, implementation
):
    control = ControlNet(union=True)
    if implementation == "wrapper":
        # Even a subclass with native-looking fields needs its own lifecycle.
        class OtherControl(ControlNet):
            pass

        unsupported = OtherControl()
    else:

        class OtherNetwork(ControlNetwork):
            pass

        unsupported = ControlNet()
        unsupported.control_model = OtherNetwork()
    control.previous_controlnet = unsupported
    clones = []
    copy_control = ControlNet.copy

    def copy(original):
        clone = copy_control(original)
        clones.append(clone)
        return clone

    monkeypatch.setattr(ControlNet, "copy", copy)
    with pytest.raises(ValueError, match="only native image-hint ControlNet"):
        sampling.sample(**arguments(positive=cond(1, control=control)))
    assert not host.common_calls
    assert len(clones) == 1
    assert clones[0].cleanups == 1 and clones[0].cond_hint_original is None
    assert control.previous_controlnet is unsupported
    assert control.cleanups == unsupported.cleanups == 0
