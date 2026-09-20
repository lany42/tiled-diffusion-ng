# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Whole-reference CPU routing contracts; real-host acceptance is still pending.

Pinned conditioning/normalization containers: ComfyUI
c194dd00cd42aa18d9dbf27d977bf6b85d9ea565, comfy/model_base.py#L2724,
comfy/latent_formats.py#L729 and comfy/conds.py. Reference positions, padding
and zero-timestep network math remain upstream responsibilities, not fixture
inference evidence. See test_krea2.py for separate Raw/Turbo/reference checks.
"""

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.adapters._krea2_sampling import PREPARATION

from .krea2_host import (
    CONDConstant,
    CONDList,
    CONDRegular,
    cond,
    normalize_reference,
    zero_out,
)
from .test_krea2 import arguments


def reference(value=4, *, batch=1, hw=(5, 7)):
    return torch.full((batch, 16, 1, *hw), float(value))


@pytest.mark.parametrize("refs", [None, [], ()])
def test_no_references_need_no_method_and_match_text_only(host, refs):
    args = arguments()
    baseline = sampling.sample(**args)["samples"]
    result = sampling.sample(**{**args, "positive": cond(2, reference_latents=refs)})[
        "samples"
    ]
    torch.testing.assert_close(result, baseline)
    for entry in host.prepared[0]:
        prepared = entry["model_conds"].get("ref_latents")
        assert prepared is None if refs is None else prepared.cond == []
        assert "ref_latents_method" not in entry["model_conds"]


@pytest.mark.parametrize("method", [None, "index", "index_timestep_zero"])
@pytest.mark.parametrize("locals_kind", [None, "identical", "distinct"])
@pytest.mark.parametrize("evaluate_negative", [False, True])
def test_turbo_zeroed_negative_preserves_native_conditioning_paths(
    host, method, locals_kind, evaluate_negative
):
    # Both upstream Turbo blueprints use ConditioningZeroOut; the style path
    # passes TextEncodeQwenImageEditPlus through Edit Model Reference Method
    # before zeroing the negative. Vision encoding itself stays upstream.
    # https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/blueprints/Image%20Style%20Reference%20%28Krea-2%20Turbo%29.json
    mask, pooled = torch.tensor([[1, 1, 0]]), torch.ones(1, 4)
    refs = [reference(4), reference(8, hw=(9, 3))]
    metadata = {"pooled_output": pooled, "attention_mask": mask}
    if method is not None:
        metadata.update(reference_latents=refs, reference_latents_method=method)
    positive = cond(2, width=30720, tokens=3, **metadata)
    negative = zero_out(positive)
    args = arguments(batch=2, cfg=1, positive=positive, negative=negative)
    args["model"].model.diffusion_model.txtdim = 2560
    if locals_kind == "identical":
        args["local_positive"] = [positive] * 4
    elif locals_kind == "distinct":
        args["local_positive"] = [
            cond(
                3 + index,
                width=30720,
                tokens=index + 1,
                **(
                    {
                        "reference_latents": [reference(index)],
                        "reference_latents_method": "index",
                    }
                    if index % 2 and method is not None
                    else {}
                ),
            )
            for index in range(4)
        ]
    captures = []
    args["model"].model_options.update(
        disable_cfg1_optimization=evaluate_negative,
        sampler_post_cfg_function=[
            lambda data: captures.append(data) or data["denoised"]
        ],
    )
    host.evaluations_per_step = 1
    result = sampling.sample(**args)["samples"]
    assert len(host.common_calls) == 1 and len(host.tile_calls) == 4
    torch.testing.assert_close(result, captures[0]["cond_denoised"])
    assert all(
        (branches[1] is not None) == evaluate_negative
        for branches, *_ in host.tile_calls
    )
    for embedding, copied in host.common_calls[0]["negative"]:
        assert embedding is negative[0][0] and not embedding.any()
        assert copied["pooled_output"] is negative[0][1]["pooled_output"]
        assert copied["attention_mask"] is mask
        if method is not None:
            assert (
                copied["reference_latents"] is refs
                and copied["reference_latents_method"] == method
            )
        else:
            assert "reference_latents" not in copied
    positives = args.get("local_positive", [positive] * 4)
    for actual, expected in zip(
        host.common_calls[0]["positive"], positives, strict=True
    ):
        assert actual[0] is expected[0][0]
        assert actual[1].get("reference_latents") is expected[0][1].get(
            "reference_latents"
        )
    if evaluate_negative:
        current, sigma = host.global_evaluations[0]
        ref_value = (
            sum(
                index * normalize_reference(ref).mean()
                for index, ref in enumerate(refs, 1)
            )
            if method is not None
            else 0
        )
        torch.testing.assert_close(
            captures[0]["uncond_denoised"],
            current - sigma * (current * 0.25 + ref_value),
        )
        for entry in host.prepared[1]:
            assert not entry["model_conds"]["c_crossattn"].cond.any()
    else:
        assert not captures[0]["uncond_denoised"].any()
    assert torch.all(positive[0][0] == 2) and pooled.all()
    assert not negative[0][0].any() and not negative[0][1]["pooled_output"].any()
    assert PREPARATION not in positive[0][1] and PREPARATION not in negative[0][1]


@pytest.mark.parametrize("method", ["index", "index_timestep_zero"])
@pytest.mark.parametrize("collection", [list, tuple])
def test_whole_odd_references_native_containers_and_metadata_identity(
    host, method, collection
):
    refs = collection([reference(2, hw=(5, 7)), reference(6, batch=3, hw=(9, 3))])
    before = [value.clone() for value in refs]
    attention, pooled = torch.tensor([[1, 0]]), torch.tensor([[5.0]])
    source = cond(
        2,
        reference_latents=refs,
        reference_latents_method=method,
        attention_mask=attention,
        pooled_output=pooled,
    )
    original_meta = source[0][1].copy()
    args = arguments(batch=2, positive=source, cfg=1)
    sampling.sample(**args)
    for embedding, meta in host.common_calls[0]["positive"]:
        assert embedding is source[0][0] and meta["reference_latents"] is refs
        assert meta["attention_mask"] is attention and meta["pooled_output"] is pooled
    for branches, *_ in host.tile_calls:
        for entry in branches[0]:
            prepared = entry["model_conds"]
            source_entry = next(
                item
                for item in host.prepared[0]
                if item[sampling.TAG] == entry[sampling.TAG]
            )
            assert entry is not source_entry and prepared is source_entry["model_conds"]
            assert type(prepared["c_crossattn"]) is CONDRegular
            assert prepared["c_crossattn"].cond is source[0][0]
            assert type(prepared["ref_latents"]) is CONDList
            assert type(prepared["ref_latents_method"]) is CONDConstant
            assert prepared["ref_latents_method"].cond == method
            for raw, normalized in zip(refs, prepared["ref_latents"].cond, strict=True):
                assert (
                    normalized.shape == raw.shape
                )  # Odd sizes stay native, never tile-cropped.
                torch.testing.assert_close(normalized, normalize_reference(raw))
    for _, sigma, _, routed, routed_method, options in host.krea2_calls:
        assert [value.shape for value in routed] == [(2, 16, 1, 5, 7), (2, 16, 1, 9, 3)]
        assert routed_method == method
        torch.testing.assert_close(options["sigmas"], sigma)
        torch.testing.assert_close(
            routed[0], normalize_reference(refs[0]).expand(2, -1, -1, -1, -1)
        )
        torch.testing.assert_close(routed[1], normalize_reference(refs[1])[:2])
    assert source[0][1].keys() == original_meta.keys()
    assert all(source[0][1][key] is value for key, value in original_meta.items())
    for ref, original in zip(refs, before, strict=True):
        torch.testing.assert_close(ref, original)


@pytest.mark.parametrize(
    "reference_batch,target_batch", [(1, 2), (2, 1), (2, 3), (3, 2)]
)
def test_reference_batch_repetition_truncation_stays_native(
    host, reference_batch, target_batch
):
    refs = torch.cat([reference(value) for value in range(reference_batch)])
    args = arguments(
        batch=target_batch,
        cfg=1,
        positive=cond(2, reference_latents=[refs], reference_latents_method="index"),
    )
    sampling.sample(**args)
    expected = normalize_reference(refs)[torch.arange(target_batch) % reference_batch]
    assert host.krea2_calls
    for _, _, _, routed, *_ in host.krea2_calls:
        torch.testing.assert_close(routed[0], expected)


@pytest.mark.parametrize("default", ["index", "index_timestep_zero"])
def test_model_default_materialized_only_in_owned_conditioning(host, default):
    source = cond(3, reference_latents=[reference()])
    args = arguments(positive=source)
    network = args["model"].model.diffusion_model
    network.default_ref_method = default
    sampling.sample(**args)
    assert "reference_latents_method" not in source[0][1]
    assert network.default_ref_method == default
    assert all(
        meta["reference_latents_method"] == default
        for _, meta in host.common_calls[0]["positive"]
    )
    assert all(
        entry["model_conds"]["ref_latents_method"].cond == default
        for entry in host.prepared[0]
    )
    assert all(
        "reference_latents_method" not in meta
        for _, meta in host.common_calls[0]["negative"]
    )


def test_model_object_patch_default_resolved_before_loading(host, monkeypatch):
    args = arguments(positive=cond(2, reference_latents=[reference()]))
    get = args["model"].get_model_object

    def patched(name):
        return (
            "index_timestep_zero"
            if name == "diffusion_model.default_ref_method"
            else get(name)
        )

    monkeypatch.setattr(args["model"], "get_model_object", patched)
    sampling.sample(**args)
    assert args["model"].model.diffusion_model.default_ref_method is None
    assert all(
        entry["model_conds"]["ref_latents_method"].cond == "index_timestep_zero"
        for entry in host.prepared[0]
    )


@pytest.mark.parametrize("default", [None, "offset", "uxo", "index_timestep_zero"])
def test_explicit_method_takes_priority_over_model_default(host, default):
    args = arguments(
        positive=cond(
            2, reference_latents=[reference()], reference_latents_method="index"
        )
    )
    args["model"].model.diffusion_model.default_ref_method = default
    sampling.sample(**args)
    assert all(
        entry["model_conds"]["ref_latents_method"].cond == "index"
        for entry in host.prepared[0]
    )


@pytest.mark.parametrize("default", [None, "offset", "uxo", "unknown"])
def test_no_supported_method_fails_with_upstream_node_instruction(host, default):
    args = arguments(positive=cond(2, reference_latents=[reference()]))
    args["model"].model.diffusion_model.default_ref_method = default
    with pytest.raises(ValueError, match="Edit Model Reference Method"):
        sampling.sample(**args)
    assert (
        not host.common_calls
        and "reference_latents_method" not in args["positive"][0][1]
    )


@pytest.mark.parametrize(
    "method", ["offset", "uxo", "uxo/uno", "unknown", "", [], torch.tensor(1)]
)
def test_unsupported_explicit_reference_methods(host, method):
    with pytest.raises(
        ValueError, match="index.*index_timestep_zero.*Edit Model Reference Method"
    ):
        sampling.sample(
            **arguments(
                positive=cond(
                    2, reference_latents=[reference()], reference_latents_method=method
                )
            )
        )
    assert not host.common_calls


@pytest.mark.parametrize(
    "refs",
    [
        reference(),
        "reference",
        [torch.zeros(1, 16, 5, 7)],
        [torch.zeros(2, 16, 5, 7)],
        [torch.zeros(1, 16, 2, 5, 7)],
        [torch.zeros(1, 4, 1, 5, 7)],
        [torch.zeros(0, 16, 1, 5, 7)],
        [torch.zeros(1, 16, 1, 0, 7)],
        [torch.zeros(1, 16, 1, 5, 7, dtype=torch.int64)],
        [reference().to_sparse()],
        [None],
    ],
)
def test_invalid_reference_layouts_fail_before_host_normalization(host, refs):
    args = arguments(
        positive=cond(2, reference_latents=refs, reference_latents_method="index")
    )
    with pytest.raises(ValueError, match="Krea2 reference_latents"):
        sampling.sample(**args)
    assert not host.common_calls and not args["model"].model.condition_calls


def test_noncontiguous_target_context_and_references_are_supported(host):
    args = arguments()
    args["latent_image"]["samples"] = torch.zeros(1, 16, 24, 32)[..., ::2, ::2]
    ref = reference(hw=(10, 14))[..., ::2, ::2]
    embedding = torch.full((1, 4, 96), 2.0)[:, ::2, ::2]
    assert not ref.is_contiguous() and not embedding.is_contiguous()
    args["positive"] = [
        [embedding, {"reference_latents": [ref], "reference_latents_method": "index"}]
    ]
    sampling.sample(**args)
    assert host.prepared[0][0]["model_conds"]["c_crossattn"].cond is embedding
    torch.testing.assert_close(
        host.prepared[0][0]["model_conds"]["ref_latents"].cond[0],
        normalize_reference(ref),
    )


def test_local_replacement_keeps_distinct_references_and_negative_independent(host):
    global_refs, negative_refs = [reference(99)], [reference(-3, hw=(3, 9))]
    local_refs = [
        None,
        [reference(2, hw=(5, 7))],
        [reference(4, hw=(9, 5)), reference(6, hw=(3, 3))],
        [],
    ]
    locals = [
        cond(index + 1, reference_latents=refs, reference_latents_method="index")
        for index, refs in enumerate(local_refs)
    ]
    negative = cond(
        -1,
        reference_latents=negative_refs,
        reference_latents_method="index_timestep_zero",
    )
    args = arguments(
        positive=cond(
            100,
            reference_latents=global_refs,
            reference_latents_method="index_timestep_zero",
        ),
        negative=negative,
        local_positive=locals,
    )
    sampling.sample(**args)
    for index, (_, meta) in enumerate(host.common_calls[0]["positive"]):
        assert meta["reference_latents"] is local_refs[index]
        assert meta["reference_latents_method"] == "index"
    for embedding, meta in host.common_calls[0]["negative"]:
        assert (
            embedding is negative[0][0] and meta["reference_latents"] is negative_refs
        )
        assert meta["reference_latents_method"] == "index_timestep_zero"
    assert all(
        kwargs.get("reference_latents") is not global_refs
        for kwargs in args["model"].model.condition_calls
    )
    assert all(PREPARATION not in local[0][1] for local in locals)


def test_local_without_reference_fields_does_not_inherit_global(host):
    args = arguments(
        positive=cond(
            2, reference_latents=[reference(100)], reference_latents_method="index"
        )
    )
    result = sampling.sample(**{**args, "local_positive": [cond(2)] * 4})["samples"]
    assert all(
        "reference_latents" not in meta for _, meta in host.common_calls[0]["positive"]
    )
    expected = sampling.sample(**{**args, "positive": cond(2)})["samples"]
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "variation", ["same", "length", "shape", "method", "tokens", "absent"]
)
def test_native_conditional_batching_preserves_reference_routing(host, variation):
    first = cond(2, reference_latents=[reference(4)], reference_latents_method="index")
    second = cond(6, reference_latents=[reference(8)], reference_latents_method="index")
    if variation == "length":
        second[0][1]["reference_latents"].append(reference(10, hw=(3, 5)))
    elif variation == "shape":
        second[0][1]["reference_latents"] = [reference(8, hw=(3, 9))]
    elif variation == "method":
        second[0][1]["reference_latents_method"] = "index_timestep_zero"
    elif variation == "tokens":
        second[0][0] = torch.full((1, 3, 48), 6.0)
    elif variation == "absent":
        second = cond(6)
    captures = []
    args = arguments(batch=2, cfg=1, positive=first + second)
    args["model"].model_options["sampler_post_cfg_function"] = [
        lambda data: captures.append(data) or data["denoised"]
    ]
    host.evaluations_per_step = 1
    sampling.sample(**args)
    text_and_refs = []
    for embedding, metadata in first + second:
        value = embedding.mean().item()
        for index, ref in enumerate(metadata.get("reference_latents", []), 1):
            value += index * normalize_reference(ref).mean().item()
        text_and_refs.append(value)
    current, sigma = host.global_evaluations[0]
    expected = current - sigma.reshape(-1, 1, 1, 1, 1) * (
        current * 0.25 + sum(text_and_refs) / 2
    )
    torch.testing.assert_close(captures[0]["cond_denoised"], expected)
    assert len(host.krea2_calls) == (4 if variation == "same" else 8)
    assert all(
        call[0].shape[0] == (4 if variation == "same" else 2)
        for call in host.krea2_calls
    )


@pytest.mark.parametrize(
    "damage,error",
    [
        ("missing_refs", "disappeared"),
        ("empty_refs", "disappeared"),
        ("missing_method", "method disappeared"),
        ("changed_method", "method disappeared"),
        ("unknown_method", "Edit Model Reference Method"),
        ("bad_layout", "B×16×1×H×W"),
        ("bad_refs_container", "CONDList"),
        ("bad_method_container", "CONDConstant"),
        ("bad_context_container", "CONDRegular"),
        ("changed_context_width", "requires 48 features"),
        ("missing_metadata", "preparation metadata"),
    ],
)
def test_prepared_reference_contract_fails_explicitly_and_detaches(host, damage, error):
    args = arguments(
        positive=cond(
            2, reference_latents=[reference()], reference_latents_method="index"
        )
    )

    def mutate(branches):
        entry = branches[0][0]
        prepared = entry["model_conds"]
        if damage == "missing_refs":
            prepared.pop("ref_latents")
        elif damage == "empty_refs":
            prepared["ref_latents"] = CONDList([])
        elif damage == "missing_method":
            prepared.pop("ref_latents_method")
        elif damage == "changed_method":
            prepared["ref_latents_method"] = CONDConstant("index_timestep_zero")
        elif damage == "unknown_method":
            prepared["ref_latents_method"] = CONDConstant("offset")
        elif damage == "bad_layout":
            prepared["ref_latents"] = CONDList([torch.zeros(1, 16, 5, 7)])
        elif damage == "bad_refs_container":
            prepared["ref_latents"] = [reference()]
        elif damage == "bad_method_container":
            prepared["ref_latents_method"] = "index"
        elif damage == "bad_context_container":
            prepared["c_crossattn"] = torch.ones(1, 2, 48)
        elif damage == "changed_context_width":
            prepared["c_crossattn"] = CONDRegular(torch.ones(1, 2, 3))
        else:
            entry.pop(PREPARATION)

    host.mutate_prepared = mutate
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.tile_calls
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)
    assert PREPARATION not in args["positive"][0][1]


def test_context_keeps_only_tensor_free_preparation_metadata(host):
    args = arguments()
    context = resolve_adapter(args["model"]).create_sampling_context(args["tile_plan"])
    context.prepare_model(args["model"], args["latent_image"])
    refs = [reference()]
    entries = [
        {
            "tdng_embedding": cond(2)[0][0],
            "reference_latents": refs,
            "reference_latents_method": "index",
        }
    ]
    context.prepare_pair(entries, [], args["tile_plan"].regions[0])
    contract = entries[0][PREPARATION]
    assert contract.reference_shapes == ((1, 16, 1, 5, 7),)
    assert vars(context) == {
        "plan": args["tile_plan"],
        "width": 48,
        "default_method": None,
    }
    context.close()
    context.close()
    assert all(value is None for value in vars(context).values())
