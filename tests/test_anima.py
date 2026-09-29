# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Offline CPU contracts, not evidence of real-host Anima or LLLite inference.

Inspected source: ComfyUI 944386c233e02eaf877b1c8d5d513fb3d3a4d5e3.
Real-host generation/refinement, portrait/landscape, local prompts, LoRAs and
seam acceptance remain pending separately for Base, Aesthetic, Turbo and 2.9B.
Shared routing, plan identity and cleanup live in test_sampling.py and
test_adapters.py; tiled LLLite contracts live in test_anima_lllite.py.
"""

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling

from .host import ControlNet, arguments, cond


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("in_channels", 17, "architecture"),
        ("patch_spatial", 1, "patch geometry"),
        ("pos_emb_cls", "sincos", "RoPE"),
        ("extra_per_block_abs_pos_emb", True, "absolute positions"),
    ],
)
def test_native_architecture_and_tile_local_rope_are_required(
    host, field, value, error
):
    args = arguments("anima")
    setattr(args["model"].model.diffusion_model, field, value)
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize("deferred", [False, True])
def test_text_metadata_reaches_host_unchanged_and_null_controls_are_dropped(
    host, deferred
):
    # Outside inference mode, native preparation defers token ids and weights
    # to forward as prepared model_conds, which each view must keep.
    host.defer_anima_text = deferred
    metadata = {
        "t5xxl_ids": torch.tensor([1, 7, 9]),
        "t5xxl_weights": torch.tensor([0.5, 2.0, 1.0]),
        "attention_mask": torch.tensor([[1, 1]]),
        "strength": 0.6,
        "start_percent": 0.1,
        "end_percent": 0.9,
        "timestep_start": 0.95,
        "timestep_end": 0.05,
        "control": None,
    }
    positive = cond(3, **metadata)
    negative = cond(-1, control=None, control_apply_to_uncond=True)
    before = [dict(positive[0][1]), dict(negative[0][1])]
    sampling.sample(**arguments("anima", positive=positive, negative=negative))
    # Text preprocessing, token weights and schedules stay host-owned.
    for embedding, meta in host.common_calls[0]["positive"]:
        assert embedding is positive[0][0] and "control" not in meta
        for key, value in metadata.items():
            if key != "control":
                assert meta[key] is value
    assert all("control" not in meta for _, meta in host.common_calls[0]["negative"])
    assert not host.discovered
    deferred_fields = {"t5xxl_ids", "t5xxl_weights"} if deferred else set()
    for branches, *_ in host.tile_calls:
        assert set(branches[0][0]["model_conds"]) == {"c_crossattn"} | deferred_fields
    # Invocation-owned copies drop null controls; caller inputs keep them.
    assert [positive[0][1], negative[0][1]] == before


def test_conditioning_controls_are_rejected_before_host_discovery(host):
    control = ControlNet()
    args = arguments("anima", local_positive=[cond(1)] * 3 + [cond(2, control=control)])
    with pytest.raises(ValueError, match="Anima conditioning control.*incompatible"):
        sampling.sample(**args)
    assert not host.common_calls and not host.discovered
    assert not control.pre_runs and not control.cleanups


@pytest.mark.parametrize(
    "field,value",
    [("area", (2, 2, 0, 0)), ("reference_latents", [])],
)
def test_unsupported_conditioning_fields_are_named(host, field, value):
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments("anima", positive=cond(1, **{field: value})))
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value",
    [("area", (2, 2, 0, 0)), ("model_conds", {"c_concat": object()})],
)
def test_prepared_spatial_conditions_are_rejected(host, field, value):
    host.mutate_prepared = lambda branches: branches[0][0].update({field: value})
    with pytest.raises(ValueError, match="area|c_concat"):
        sampling.sample(**arguments("anima"))
    assert not host.tile_calls


def native_lllite(name):
    return type(name, (), {"__module__": "comfy.ldm.anima.lllite"})()


@pytest.mark.parametrize(
    "slot,name,live",
    [
        ("attn1_patch", "AnimaLLLiteAttentionPatch", False),
        ("post_input", "AnimaLLLitePatch", True),
    ],
)
def test_native_lllite_hooks_have_explicit_deferred_guard(host, slot, name, live):
    args = arguments("anima")
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
        ("patches", {"attn1_patch": [object()]}),
        ("patches_replace", {"dit": {("double_block", 0): object()}}),
    ],
)
def test_other_transformer_patches_are_not_silently_discarded(host, field, patches):
    args = arguments("anima")
    args["model"].model_options["transformer_options"][field] = patches
    with pytest.raises(ValueError, match=field):
        sampling.sample(**args)
    assert not host.common_calls


def test_patch_added_after_first_evaluation_is_rejected_and_closed(host):
    args = arguments("anima")

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
