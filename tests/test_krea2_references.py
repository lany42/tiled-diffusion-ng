# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Whole-reference CPU routing contracts; real-host acceptance is still pending.

Pinned conditioning/normalization containers: ComfyUI
c194dd00cd42aa18d9dbf27d977bf6b85d9ea565, comfy/model_base.py#L2724,
comfy/latent_formats.py#L729 and comfy/conds.py. Reference positions, padding,
zero-timestep network math and batch repetition remain upstream
responsibilities, not fixture inference evidence.
"""

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters._krea2_sampling import PREPARATION

from . import krea2_host
from .host import arguments
from .krea2_host import (
    CONDConstant,
    CONDList,
    CONDRegular,
    cond,
    normalize_reference,
    zero_out,
)


def reference(value=4, *, batch=1, hw=(5, 7)):
    return torch.full((batch, 16, 1, *hw), float(value))


def test_empty_references_need_no_method(host):
    sampling.sample(**arguments("krea2", positive=cond(2, reference_latents=[])))
    for entry in host.prepared[0]:
        assert entry["model_conds"]["ref_latents"].cond == []
        assert "ref_latents_method" not in entry["model_conds"]


@pytest.mark.parametrize(
    "method,locals_kind", [("index", None), ("index_timestep_zero", "distinct")]
)
def test_zeroed_negative_preserves_native_conditioning_paths(host, method, locals_kind):
    # Both upstream Turbo blueprints use ConditioningZeroOut; the style path
    # passes TextEncodeQwenImageEditPlus through Edit Model Reference Method
    # before zeroing the negative. Vision encoding itself stays upstream.
    # https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/blueprints/Image%20Style%20Reference%20%28Krea-2%20Turbo%29.json
    mask, pooled = torch.tensor([[1, 1, 0]]), torch.ones(1, 4)
    refs = [reference(4), reference(8, hw=(9, 3))]
    positive = cond(
        2,
        width=30720,
        tokens=3,
        pooled_output=pooled,
        attention_mask=mask,
        reference_latents=refs,
        reference_latents_method=method,
    )
    negative = zero_out(positive)
    args = arguments("krea2", batch=2, cfg=1, positive=positive, negative=negative)
    args["model"].model.diffusion_model.txtdim = 2560
    if locals_kind == "distinct":
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
                    if index % 2
                    else {}
                ),
            )
            for index in range(4)
        ]
    # Turbo samples at CFG 1; ask the host to evaluate the negative anyway.
    args["model"].model_options["disable_cfg1_optimization"] = True
    host.evaluations_per_step = 1
    sampling.sample(**args)
    assert len(host.common_calls) == 1 and len(host.tile_calls) == 4
    for embedding, copied in host.common_calls[0]["negative"]:
        assert embedding is negative[0][0] and not embedding.any()
        assert copied["pooled_output"] is negative[0][1]["pooled_output"]
        assert copied["attention_mask"] is mask
        assert copied["reference_latents"] is refs
        assert copied["reference_latents_method"] == method
    positives = args.get("local_positive", [positive] * 4)
    for actual, expected in zip(
        host.common_calls[0]["positive"], positives, strict=True
    ):
        assert actual[0] is expected[0][0]
        assert actual[1].get("reference_latents") is expected[0][1].get(
            "reference_latents"
        )
    # The zeroed negative still routes its whole references to the model.
    negative_calls = [c for c in host.krea2_calls if 1 in c[5]["cond_or_uncond"]]
    assert negative_calls
    for _, _, _, routed, routed_method, _ in negative_calls:
        assert [ref.shape[1:] for ref in routed] == [(16, 1, 5, 7), (16, 1, 9, 3)]
        assert routed_method == method
    assert torch.all(positive[0][0] == 2) and pooled.all()
    assert not negative[0][0].any() and not negative[0][1]["pooled_output"].any()
    assert PREPARATION not in positive[0][1] and PREPARATION not in negative[0][1]


def test_whole_odd_references_native_containers_and_metadata_identity(host):
    method = "index_timestep_zero"
    refs = (reference(2, hw=(5, 7)), reference(6, batch=3, hw=(9, 3)))
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
    args = arguments("krea2", batch=2, positive=source, cfg=1, hw=(13, 15))
    sampling.sample(**args)
    for embedding, meta in host.common_calls[0]["positive"]:
        assert embedding is source[0][0] and meta["reference_latents"] is refs
        assert meta["attention_mask"] is attention and meta["pooled_output"] is pooled
    for branches, *_ in host.tile_calls:
        for entry in branches[0]:
            prepared = entry["model_conds"]
            # Views copy the entry shallowly, sharing every native container.
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
            # Odd reference sizes stay whole; they are never tile-cropped.
            for raw, normalized in zip(refs, prepared["ref_latents"].cond, strict=True):
                torch.testing.assert_close(normalized, normalize_reference(raw))
    for *_, routed, routed_method, _ in host.krea2_calls:
        assert [value.shape[1:] for value in routed] == [(16, 1, 5, 7), (16, 1, 9, 3)]
        assert routed_method == method
    assert source[0][1].keys() == original_meta.keys()
    assert all(source[0][1][key] is value for key, value in original_meta.items())
    for ref, original in zip(refs, before, strict=True):
        torch.testing.assert_close(ref, original)


@pytest.mark.parametrize(
    "explicit,default,expected",
    [
        (None, "index_timestep_zero", "index_timestep_zero"),
        ("index", "index_timestep_zero", "index"),
        (None, None, None),
        (None, "offset", None),
    ],
    ids=["model_default", "explicit_wins", "missing", "unsupported_default"],
)
def test_reference_method_resolution(host, explicit, default, expected):
    metadata = {} if explicit is None else {"reference_latents_method": explicit}
    source = cond(3, reference_latents=[reference()], **metadata)
    before = source[0][1].copy()
    args = arguments("krea2", positive=source)
    args["model"].model.diffusion_model.default_ref_method = default
    if expected is None:
        with pytest.raises(ValueError, match="Edit Model Reference Method"):
            sampling.sample(**args)
        assert not host.common_calls
    else:
        sampling.sample(**args)
        assert all(
            meta["reference_latents_method"] == expected
            for _, meta in host.common_calls[0]["positive"]
        )
        assert all(
            entry["model_conds"]["ref_latents_method"].cond == expected
            for entry in host.prepared[0]
        )
        assert all(
            "reference_latents_method" not in meta
            for _, meta in host.common_calls[0]["negative"]
        )
    # The default materializes only in invocation-owned conditioning.
    assert source[0][1] == before
    assert args["model"].model.diffusion_model.default_ref_method == default


def test_models_without_a_default_method_need_an_explicit_one(host, monkeypatch):
    # get_model_object raises AttributeError when the network has no default.
    monkeypatch.delattr(krea2_host.Krea2Network, "default_ref_method")
    sampling.sample(**arguments("krea2"))
    refs = [reference()]
    with pytest.raises(ValueError, match="Edit Model Reference Method"):
        sampling.sample(**arguments("krea2", positive=cond(2, reference_latents=refs)))
    explicit = cond(2, reference_latents=refs, reference_latents_method="index")
    sampling.sample(**arguments("krea2", positive=explicit))
    assert len(host.common_calls) == 2


def test_model_object_patch_default_resolved_before_loading(host, monkeypatch):
    args = arguments("krea2", positive=cond(2, reference_latents=[reference()]))
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


@pytest.mark.parametrize("method", ["offset", []])
def test_unsupported_explicit_reference_methods(host, method):
    with pytest.raises(
        ValueError, match="index.*index_timestep_zero.*Edit Model Reference Method"
    ):
        sampling.sample(
            **arguments(
                "krea2",
                positive=cond(
                    2, reference_latents=[reference()], reference_latents_method=method
                ),
            )
        )
    assert not host.common_calls


@pytest.mark.parametrize(
    "refs",
    [
        reference(),
        [torch.zeros(1, 16, 5, 7)],
        [torch.zeros(1, 16, 2, 5, 7)],
    ],
    ids=["bare_tensor", "4d", "video"],
)
def test_invalid_reference_layouts_fail_before_host_normalization(host, refs):
    args = arguments(
        "krea2",
        positive=cond(2, reference_latents=refs, reference_latents_method="index"),
    )
    with pytest.raises(ValueError, match="Krea2 reference_latents"):
        sampling.sample(**args)
    assert not host.common_calls and not args["model"].model.condition_calls


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
        "krea2",
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
        "krea2",
        positive=cond(
            2, reference_latents=[reference(100)], reference_latents_method="index"
        ),
    )
    result = sampling.sample(**{**args, "local_positive": [cond(2)] * 4})["samples"]
    assert all(
        "reference_latents" not in meta for _, meta in host.common_calls[0]["positive"]
    )
    expected = sampling.sample(**{**args, "positive": cond(2)})["samples"]
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "damage,error",
    [
        ("missing_refs", "disappeared"),
        ("changed_method", "method disappeared"),
        ("bad_refs_container", "CONDList"),
        ("bad_method_container", "CONDConstant"),
        ("bad_context_container", "CONDRegular"),
        ("missing_metadata", "preparation metadata"),
    ],
)
def test_prepared_reference_contract_fails_explicitly_and_detaches(host, damage, error):
    # Defends against host preparation changes; the doubles cannot detect them.
    args = arguments(
        "krea2",
        positive=cond(
            2, reference_latents=[reference()], reference_latents_method="index"
        ),
    )

    def mutate(branches):
        entry = branches[0][0]
        prepared = entry["model_conds"]
        if damage == "missing_refs":
            prepared.pop("ref_latents")
        elif damage == "changed_method":
            prepared["ref_latents_method"] = CONDConstant("index_timestep_zero")
        elif damage == "bad_refs_container":
            prepared["ref_latents"] = [reference()]
        elif damage == "bad_method_container":
            prepared["ref_latents_method"] = "index"
        elif damage == "bad_context_container":
            prepared["c_crossattn"] = torch.ones(1, 2, 48)
        else:
            entry.pop(PREPARATION)

    host.mutate_prepared = mutate
    with pytest.raises(ValueError, match=error):
        sampling.sample(**args)
    assert not host.tile_calls
    assert not host.latest_clone.get_wrappers("calc_cond_batch", sampling.WRAPPER_KEY)
    assert PREPARATION not in args["positive"][0][1]
