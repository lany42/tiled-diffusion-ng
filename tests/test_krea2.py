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
Shared routing, plan identity and cleanup live in test_sampling.py and
test_adapters.py.
"""

import sys

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling

from .host import ControlNet, arguments, copy_containers
from .krea2_host import cond


@pytest.mark.parametrize(
    "owner,field,value,error",
    [
        ("model", "concat_keys", ("concat_latent_image",), "architecture"),
        ("network", "patch", 1, "patch geometry"),
        ("network", "pe_embedder", object(), "EmbedND"),
        ("network", "txtlayers", 1.5, "text dimensions"),
    ],
)
def test_native_architecture_guards(host, owner, field, value, error):
    args = arguments("krea2")
    model = args["model"].model
    setattr(model if owner == "model" else model.diffusion_model, field, value)
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.common_calls


@pytest.mark.parametrize("missing", ["wrapper", "network"])
def test_optional_krea2_apis_do_not_break_existing_families(host, monkeypatch, missing):
    from comfy import model_base

    args = arguments("krea2")
    if missing == "wrapper":
        monkeypatch.delattr(model_base, "Krea2")
    else:
        monkeypatch.setitem(sys.modules, "comfy.ldm.krea2.model", None)
    with pytest.raises(
        ValueError, match="Unsupported model/latent family|native Krea2 APIs"
    ):
        sampling.sample(**args)
    sampling.sample(**arguments("sdxl"))
    sampling.sample(**arguments("anima"))
    assert len(host.common_calls) == 2


@pytest.mark.parametrize(
    "embedding",
    [torch.zeros(1, 2, 48, dtype=torch.int64), torch.zeros(1, 48)],
    ids=["integer", "rank2"],
)
def test_malformed_context_rejected_before_host(host, embedding):
    with pytest.raises(ValueError, match="floating rank-3 embedding"):
        sampling.sample(**arguments("krea2", positive=[[embedding, {}]]))
    assert not host.common_calls


def test_context_width_is_checked_on_every_effective_branch(host):
    args = arguments("krea2", cfg=1)
    # A replaced global positive is not effective and needs no native width.
    sampling.sample(
        **{**args, "positive": cond(9, width=3), "local_positive": [cond(2)] * 4}
    )
    for override in (
        {"local_positive": [cond(2)] * 3 + [cond(4, width=3)]},
        {"negative": cond(4, width=3)},
    ):
        with pytest.raises(ValueError, match="requires 48 features.*got 3"):
            sampling.sample(**{**args, **override})
    assert len(host.common_calls) == 1 and not host.discovered


def test_encoded_context_is_opaque_and_dimensions_come_from_model(host):
    args = arguments("krea2", cfg=1)
    network = args["model"].model.diffusion_model
    network.txtlayers, network.txtdim = 12, 2560
    # A noncontiguous two-image batch; features are never inspected or copied.
    embedding = (
        torch.arange(12 * 2560 * 2, dtype=torch.float32)
        .reshape(1, 2, -1)
        .transpose(0, 1)
    )
    attention, pooled = torch.tensor([[1, 0]]), torch.tensor([[17.0]])
    args["positive"] = [
        [embedding, {"attention_mask": attention, "pooled_output": pooled}]
    ]
    args["negative"] = cond(-1, width=12 * 2560)
    sampling.sample(**args)
    for embedding_seen, meta in host.common_calls[0]["positive"]:
        assert embedding_seen is embedding
        assert meta["attention_mask"] is attention and meta["pooled_output"] is pooled
    torch.testing.assert_close(
        embedding.flatten(), torch.arange(12 * 2560 * 2, dtype=torch.float32)
    )


def test_baked_clip_schedules_reach_the_host_unchanged(host):
    # encode_from_tokens_scheduled may bake CLIP LoRAs into several entries
    # without runtime hooks. ComfyUI owns percent-to-sigma conversion.
    condition = cond(2, clip_start_percent=0, clip_end_percent=0.5) + cond(
        7,
        clip_start_percent=0.5,
        clip_end_percent=1,
        start_percent=0.6,
        timestep_start=0.9,
        timestep_end=0.1,
    )
    before = [meta.copy() for _, meta in condition]
    sampling.sample(**arguments("krea2", positive=condition, negative=condition))
    for branch in ("positive", "negative"):
        prepared = [meta for _, meta in host.common_calls[0][branch][:2]]
        for meta, original in zip(prepared, before, strict=True):
            assert all(meta[key] == value for key, value in original.items())
    assert [meta for _, meta in condition] == before


@pytest.mark.parametrize("branch", ["negative", "replaced_global"])
def test_controls_rejected_before_discovery(host, branch):
    args = arguments("krea2", cfg=1)
    control = ControlNet()
    if branch == "negative":
        args["negative"] = cond(1, control=control)
    else:
        args["local_positive"] = [cond(1)] * 4
        args["positive"] = cond(2, control=control)
    with pytest.raises(ValueError, match="Krea2 conditioning control.*ControlNet"):
        sampling.sample(**args)
    assert not host.common_calls and not host.discovered
    assert not control.pre_runs and not control.cleanups


def test_null_controls_removed_only_from_owned_metadata(host):
    source = cond(7, control=None, control_apply_to_uncond=True)
    before = source[0][1].copy()
    sampling.sample(
        **arguments(
            "krea2", positive=source, negative=source, local_positive=[source] * 4
        )
    )
    assert source[0][1] == before
    assert all("control" not in entry for branch in host.prepared for entry in branch)
    assert not host.discovered


@pytest.mark.parametrize(
    "field,value",
    [("area", (2, 2, 0, 0)), ("concat_latent_image", torch.zeros(1, 16, 12, 16))],
)
def test_unsupported_raw_spatial_fields(host, field, value):
    with pytest.raises(ValueError, match=field):
        sampling.sample(**arguments("krea2", positive=cond(1, **{field: value})))
    assert not host.common_calls


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("area", (2, 2, 0, 0), "conditioning field: area"),
        ("model_conds", {"c_concat": object()}, "model_conds field: c_concat"),
    ],
)
def test_unsupported_prepared_spatial_fields(host, field, value, error):
    host.mutate_prepared = lambda branches: branches[0][0].update({field: value})
    with pytest.raises(ValueError, match="prepared Krea2 " + error):
        sampling.sample(**arguments("krea2"))
    assert not host.tile_calls


@pytest.mark.parametrize(
    "field,live,lllite",
    [
        ("patches", False, True),
        ("patches_replace", False, False),
        ("patches", True, False),
    ],
)
def test_transformer_hooks_and_lllite_rejected(host, field, live, lllite):
    patch = (
        type(
            "AnimaLLLiteAttentionPatch", (), {"__module__": "comfy.ldm.anima.lllite"}
        )()
        if lllite
        else object()
    )
    args = arguments("krea2")

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


@pytest.mark.parametrize("route", ["attachment", "hook"])
def test_tiled_lllite_hooks_and_attachments_have_krea2_error(host, route):
    from .test_anima_lllite import patched

    anima, *_ = patched()
    args = arguments("krea2")
    # Unrelated nonspatial attachments remain compatible.
    args["model"].attachments["nonspatial"] = object()
    sampling.sample(**args)
    host.common_calls.clear()
    if route == "attachment":
        args["model"].attachments.update(anima["model"].attachments)
    else:
        args["model"].model_options = copy_containers(anima["model"].model_options)
    with pytest.raises(NotImplementedError, match="Krea2 LLLite support remains TBD"):
        sampling.sample(**args)
    assert not host.common_calls and not host.patch_models


def test_late_patch_injection_closes_invocation(host):
    args = arguments("krea2")

    def solver(evaluate, x, sigmas):
        x = evaluate(x, torch.tensor([0.6]), (0, 0))
        host.live_options["transformer_options"]["patches"] = {"post_input": [object()]}
        return evaluate(x, torch.tensor([0.4]), (0, 1))

    host.dispatch["euler"].sampler_function = solver
    with pytest.raises(ValueError, match="Krea2 transformer patches"):
        sampling.sample(**args)
    assert len(host.tile_calls) == 4
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)
