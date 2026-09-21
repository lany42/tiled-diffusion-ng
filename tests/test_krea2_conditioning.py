# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Offline CPU encoding/routing contracts, not real-host Krea2 acceptance.

Raw and Turbo still need native image-only/prompted tile inference, strength
comparisons, cutoff transitions and downsizing checks. Record the host revision,
encoder/checkpoint precision and actual visual outcomes during those runs.
These doubles cannot test native prefix removal, vision preprocessing or model
adherence; they test delegation without recreating or patching those operations.
"""

import asyncio
import gc
import sys
import weakref

import pytest
import torch
import torch.nn.functional as F

from tiled_diffusion_ng import comfy_entrypoint
from tiled_diffusion_ng._krea2_conditioning import encode_tiles
from tiled_diffusion_ng.geometry import TILE_IDS

from . import krea2_conditioning_host as encoder_host
from .krea2_host import cond
from .test_krea2 import arguments as sampler_arguments

VISION = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture
def clip(host, monkeypatch):
    encoder_host.install(monkeypatch)
    return encoder_host.Clip()


def tiles(height=6, width=10, channels=3):
    return (
        torch.arange(1, 5, dtype=torch.float32)
        .reshape(4, 1, 1, 1)
        .expand(4, height, width, channels)
        / 10
    )


def node_arguments(clip, **overrides):
    return {
        name: [value]
        for name, value in {
            "clip": clip,
            "reference_tiles": tiles(),
            "strength": 1.0,
            "end_percent": 1.0,
            "downsize_to_1mp": False,
            **overrides,
        }.items()
    }


def test_schema_and_execution_list_output(clip):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    schema = TileKrea2Conditioning.define_schema()
    assert schema.node_id == "TiledDiffusionNG_TileKrea2Conditioning"
    assert schema.display_name == "TileKrea2Conditioning"
    assert schema.category == "Tiled Diffusion NG" and schema.is_input_list
    inputs = {item.id: item for item in schema.inputs}
    assert list(inputs) == [
        "clip",
        "reference_tiles",
        "prompts",
        "strength",
        "end_percent",
        "downsize_to_1mp",
    ]
    assert inputs["prompts"].optional and inputs["prompts"].force_input
    assert inputs["prompts"].dynamic_prompts is False
    for name, expected in (
        ("strength", (1, 0, 3, 0.05)),
        ("end_percent", (1, 0, 1, 0.001)),
    ):
        item = inputs[name]
        assert (item.default, item.min, item.max, item.step) == expected
    assert inputs["downsize_to_1mp"].default is False
    assert len(schema.outputs) == 1
    assert schema.outputs[0].id == "local_positive"
    assert schema.outputs[0].is_output_list
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5},
        {"clip_start_percent": 0.5, "clip_end_percent": 1},
    ]
    output = TileKrea2Conditioning.execute(**node_arguments(clip, end_percent=0.6))
    assert len(output) == 1 and len(output[0]) == 4
    assert all(len(conditioning) == 4 for conditioning in output[0])


@pytest.mark.parametrize(
    "name", ["clip", "reference_tiles", "strength", "end_percent", "downsize_to_1mp"]
)
def test_ordinary_inputs_require_singleton_execution_lists(clip, name):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    args = node_arguments(clip)
    for bad in ([], args[name] * 2, tuple(args[name]), args[name][0]):
        with pytest.raises(
            ValueError, match=name + " requires one execution-list item"
        ):
            TileKrea2Conditioning.execute(**{**args, name: bad})
    assert not clip.tokenized


@pytest.mark.parametrize(
    "prompts",
    [
        None,
        [],
        [""] * 4,
        ["red", "green", "blue", "white"],
        ["a {literal|brace}", "", "{{still literal}}", ""],
    ],
)
@pytest.mark.parametrize("channels", [3, 4])
def test_four_joint_encodes_keep_order_rgb_and_literal_prompts(clip, prompts, channels):
    source = tiles(channels=channels)
    if channels == 4:
        source[..., 3] = 100
    before = source.clone()
    result = encode_tiles(clip, source, prompts=prompts)
    expected_prompts = prompts or [""] * 4
    assert len(result) == len(clip.encoded) == len(clip.tokenized) == 4
    for index, (tokens, conditioning, native) in enumerate(
        zip(clip.tokenized, result, clip.outputs, strict=True)
    ):
        assert tokens is clip.encoded[index]
        assert tokens["text"] == VISION + expected_prompts[index]
        assert tokens["template"] == encoder_host.KREA2_TEMPLATE
        assert tokens["rendered"] == encoder_host.KREA2_TEMPLATE.format(
            VISION + expected_prompts[index]
        )
        assert len(tokens["images"]) == 1
        torch.testing.assert_close(
            tokens["images"][0], source[index : index + 1, ..., :3]
        )
        assert conditioning[0][0] is native[0][0]
        assert conditioning[0][0].shape[0] == 1 and conditioning[0][0].shape[2] == 30720
        assert conditioning is not native and conditioning[0] is not native[0]
        assert conditioning[0][1] is not native[0][1]
        assert conditioning[0][1].keys() == native[0][1].keys()
        assert all(
            conditioning[0][1][key] is value for key, value in native[0][1].items()
        )
        assert "reference_latents" not in conditioning[0][1]
    torch.testing.assert_close(source, before)


@pytest.mark.parametrize("strength", [0, 0.25, 1, 1.75, 3])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_strength_scales_every_entry_without_mutation(clip, strength, dtype):
    clip.dtype = dtype
    marker = object()
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5, "opaque": marker},
        {"clip_start_percent": 0.5, "clip_end_percent": 1, "strength": 0.7},
    ]
    baseline = encode_tiles(clip, tiles())
    result = encode_tiles(clip, tiles(), strength=strength)
    for expected, actual, native in zip(
        baseline, result, clip.outputs[4:], strict=True
    ):
        assert len(actual) == 2
        for before, after, original in zip(expected, actual, native, strict=True):
            torch.testing.assert_close(original[0], before[0])
            torch.testing.assert_close(after[0], before[0] * strength)
            assert after[0].dtype == dtype
            assert (after[0] is original[0]) == (strength == 1)
            assert after[1] is not original[1]
            assert after[1].keys() == original[1].keys()
            assert all(after[1][key] is value for key, value in original[1].items())
            if strength == 0:
                assert not after[0].any()


@pytest.mark.parametrize("cutoff", [0, 0.35, 1])
def test_cutoff_shared_unscaled_fallback_and_independent_containers(clip, cutoff):
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5},
        {"clip_start_percent": 0.5, "clip_end_percent": 1},
    ]
    output = encode_tiles(
        clip, tiles(), ["tile text"] * 4, strength=0, end_percent=cutoff
    )
    assert len(clip.encoded) == (1 if cutoff == 0 else 4 if cutoff == 1 else 5)
    assert len(output) == 4
    for conditioning in output:
        assert len(conditioning) == (4 if 0 < cutoff < 1 else 2)
        if cutoff > 0:
            assert not conditioning[0][0].any() and not conditioning[1][0].any()
        if cutoff < 1:
            fallback = conditioning[-2:]
            assert (
                clip.tokenized[-1]["text"] == "" and clip.tokenized[-1]["images"] == []
            )
            assert clip.tokenized[-1]["template"] == encoder_host.KREA2_TEMPLATE
            for (embedding, meta), (native, original_meta) in zip(
                fallback, clip.outputs[-1], strict=True
            ):
                assert embedding is native and embedding.any()
                assert meta["clip_start_percent"] == original_meta["clip_start_percent"]
                assert meta["clip_end_percent"] == original_meta["clip_end_percent"]
                if cutoff > 0:
                    assert (meta["start_percent"], meta["end_percent"]) == (cutoff, 1)
            if cutoff > 0:
                for _, meta in conditioning[:2]:
                    assert (meta["start_percent"], meta["end_percent"]) == (0, cutoff)
    for position in range(len(output[0])):
        assert len({id(conditioning[position]) for conditioning in output}) == 4
        assert len({id(conditioning[position][1]) for conditioning in output}) == 4
    output[0][0][1]["changed"] = True
    output[0].append([None, {}])
    assert all("changed" not in conditioning[0][1] for conditioning in output[1:])
    assert all("changed" not in entries[0][1] for entries in clip.outputs)


def test_cutoff_intersects_existing_ranges_without_rewriting_clip_schedule(clip):
    clip.schedules = [
        {
            "start_percent": 0.2,
            "end_percent": 0.8,
            "clip_start_percent": 0.3,
            "clip_end_percent": 0.7,
        }
    ]
    result = encode_tiles(clip, tiles(), end_percent=0.5)
    for conditioning in result:
        assert [
            (meta["start_percent"], meta["end_percent"]) for _, meta in conditioning
        ] == [(0.2, 0.5), (0.5, 0.8)]
        assert all(
            (meta["clip_start_percent"], meta["clip_end_percent"]) == (0.3, 0.7)
            for _, meta in conditioning
        )
    assert all(
        (meta["start_percent"], meta["end_percent"]) == (0.2, 0.8)
        for entries in clip.outputs
        for _, meta in entries
    )


@pytest.mark.parametrize(
    "height,width,expected",
    [
        (2048, 1024, (1448, 724)),
        (1024, 2048, (724, 1448)),
        (2048, 2048, (1024, 1024)),
        (1111, 1777, (810, 1295)),
        (1, 4194304, (1, 2097152)),
        (1024, 1024, None),
        (64, 32, None),
    ],
)
@pytest.mark.parametrize("enabled", [False, True])
def test_downsize_dispatch_preserves_aspect_without_enlarging(
    clip, monkeypatch, height, width, expected, enabled
):
    calls = []

    def resize(image, *, size, mode, align_corners, antialias):
        calls.append((image, size, mode, align_corners, antialias))
        # Isolate dimension/transport checks, including extreme aspect ratios.
        # Real PyTorch interpolation and pixel precision are exercised below.
        new_height, new_width = size
        return image[..., :1, :1].expand(1, 3, new_height, new_width)

    monkeypatch.setattr(F, "interpolate", resize)
    source = tiles(1, 1, 4).expand(4, height, width, 4)
    encode_tiles(clip, source, downsize_to_1mp=enabled)
    target = expected if enabled and expected is not None else (height, width)
    assert len(calls) == (4 if enabled and expected is not None else 0)
    for image, size, mode, align_corners, antialias in calls:
        assert image.shape == (1, 3, height, width)
        assert size == expected
        assert (mode, align_corners, antialias) == ("bicubic", False, True)
        new_height, new_width = size
        assert new_width <= width and new_height <= height
    for index, tokens in enumerate(clip.tokenized):
        assert tokens["images"][0].shape == (1, *target, 3)
        torch.testing.assert_close(
            tokens["images"][0][0, 0, 0], source[index, 0, 0, :3]
        )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_real_bicubic_downsize_preserves_precision_and_source(clip, host, dtype):
    # These colors cannot survive a uint8 or float16 round trip unchanged.
    colors = torch.tensor([0.1234567, 0.2345678, 0.3456789, 0.9], dtype=dtype)
    source = colors.reshape(1, 1, 1, 4).expand(4, 1025, 1024, 4)
    before = colors.clone()
    result = encode_tiles(clip, source, downsize_to_1mp=True)
    assert len(result) == len(clip.tokenized) == 4
    assert not host.resize_calls
    resized_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    tolerance = 1e-12 if resized_dtype == torch.float64 else 1e-6
    for tokens in clip.tokenized:
        image = tokens["images"][0]
        assert image.shape == (1, 1024, 1024, 3)
        assert image.dtype == resized_dtype and image.device == source.device
        expected = before[:3].to(resized_dtype).expand_as(image)
        torch.testing.assert_close(image, expected, rtol=0, atol=tolerance)
    assert source.dtype == dtype
    torch.testing.assert_close(colors, before, rtol=0, atol=0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("enabled,size", [(False, 1025), (True, 8)])
def test_unresized_images_keep_exact_pixels_and_dtype(
    clip, monkeypatch, dtype, enabled, size
):
    def unexpected_resize(*args, **kwargs):
        pytest.fail("An unchanged image must not be resized")

    monkeypatch.setattr(F, "interpolate", unexpected_resize)
    colors = torch.tensor([0.1234567, 0.2345678, 0.3456789], dtype=dtype)
    source = colors.reshape(1, 1, 1, 3).expand(4, size, size, 3)
    encode_tiles(clip, source, downsize_to_1mp=enabled)
    for index, tokens in enumerate(clip.tokenized):
        torch.testing.assert_close(
            tokens["images"][0], source[index : index + 1], rtol=0, atol=0
        )


def test_real_bicubic_keeps_rgb_range_at_sharp_edges(clip):
    # Cubic interpolation can overshoot at a black/white boundary.
    source = torch.zeros(4, 1025, 1024, 3)
    source[:, 512:] = 1
    before = source.clone()
    encode_tiles(clip, source, downsize_to_1mp=True)
    for tokens in clip.tokenized:
        image = tokens["images"][0]
        assert image.dtype == torch.float32
        assert image.min() == 0 and image.max() == 1
        assert torch.any((image > 0) & (image < 1))
    torch.testing.assert_close(source, before, rtol=0, atol=0)


def test_zero_cutoff_skips_tile_resize_and_encoding(clip, monkeypatch):
    def unexpected_resize(*args, **kwargs):
        pytest.fail("Zero cutoff must only encode the empty-prompt fallback")

    monkeypatch.setattr(F, "interpolate", unexpected_resize)
    encode_tiles(
        clip, tiles(1, 1).expand(4, 2048, 2048, 3), end_percent=0, downsize_to_1mp=True
    )
    assert len(clip.tokenized) == 1 and not clip.tokenized[0]["images"]


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda: None,
        lambda: [tiles()],
        lambda: torch.empty((3, 6, 10, 3)),
        lambda: torch.empty((5, 6, 10, 3)),
        lambda: torch.empty((8, 6, 10, 3)),
        lambda: torch.empty((0, 6, 10, 3)),
        lambda: torch.empty((4, 0, 10, 3)),
        lambda: torch.empty((4, 6, 0, 3)),
        lambda: torch.empty((4, 6, 10, 2)),
        lambda: torch.empty((4, 6, 10, 5)),
        lambda: torch.empty((4, 3, 6, 10)),
        lambda: torch.empty((4, 6, 10)),
        lambda: torch.empty((4, 1, 6, 10, 3)),
        lambda: tiles().to(torch.int64),
        lambda: tiles().to(torch.complex64),
        lambda: tiles().to_sparse(),
    ],
)
def test_invalid_images_fail_before_encoding(clip, make_bad):
    with pytest.raises(ValueError, match="reference_tiles"):
        encode_tiles(clip, make_bad(), end_percent=0)
    assert not clip.tokenized


@pytest.mark.parametrize(
    "prompts",
    ["text", ["one"], [""] * 3, [""] * 5, ("",) * 4, ["", "", "", None], [[""]] * 4],
)
def test_invalid_prompts_fail_before_encoding_even_at_zero_cutoff(clip, prompts):
    with pytest.raises(ValueError, match="prompts requires exactly four"):
        encode_tiles(clip, tiles(), prompts=prompts, end_percent=0)
    assert not clip.tokenized


@pytest.mark.parametrize(
    "name,value",
    [
        (name, value)
        for name in ("strength", "end_percent")
        for value in (
            float("nan"),
            float("inf"),
            -float("inf"),
            -0.1,
            3.1,
            True,
            "1",
            None,
            torch.tensor(1.0),
        )
    ]
    + [
        ("end_percent", 1.001),
        ("downsize_to_1mp", 1),
        ("downsize_to_1mp", "false"),
        ("downsize_to_1mp", None),
    ],
)
def test_invalid_settings_fail_before_encoding(clip, name, value):
    with pytest.raises(ValueError, match=name):
        encode_tiles(clip, tiles(), **{name: value})
    assert not clip.tokenized


@pytest.mark.parametrize(
    "missing", ["module", "KREA2_TEMPLATE", "Krea2Tokenizer", "Krea2TEModel"]
)
def test_optional_apis_fail_only_when_executing_helper(clip, monkeypatch, missing):
    native = sys.modules["comfy.text_encoders.krea2"]
    if missing == "module":
        monkeypatch.setitem(sys.modules, "comfy.text_encoders.krea2", None)
    else:
        monkeypatch.delattr(native, missing)
    extension = asyncio.run(comfy_entrypoint())
    nodes = asyncio.run(extension.get_node_list())
    assert len([node.define_schema() for node in nodes]) == 5
    with pytest.raises(ValueError, match="native Krea2 text encoder APIs"):
        encode_tiles(clip, tiles())
    assert not clip.tokenized


@pytest.mark.parametrize(
    "field",
    ["cond_stage_model", "tokenizer", "tokenize", "encode_from_tokens_scheduled"],
)
def test_incompatible_clip_rejected(clip, field):
    setattr(clip, field, object())
    with pytest.raises(ValueError, match="native Krea2 CLIP"):
        encode_tiles(clip, tiles())
    assert not clip.tokenized


def test_native_tokenizer_and_model_subclasses_are_accepted(clip):
    clip.tokenizer = type("TokenizerSubclass", (encoder_host.Krea2Tokenizer,), {})()
    assert len(encode_tiles(clip, tiles())) == 4


@pytest.mark.parametrize("template", [None, "", "{}{}", "missing placeholder"])
def test_missing_template_capability_rejected(clip, monkeypatch, template):
    monkeypatch.setattr(
        sys.modules["comfy.text_encoders.krea2"], "KREA2_TEMPLATE", template
    )
    with pytest.raises(ValueError, match="conditioning template"):
        encode_tiles(clip, tiles())
    assert not clip.tokenized


@pytest.mark.parametrize("bad", [None, [], {}, [[None]], [[None, []]]])
def test_malformed_native_conditioning_rejected(clip, monkeypatch, bad):
    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", lambda tokens: bad)
    with pytest.raises(ValueError, match="Native Krea2 encoding must return"):
        encode_tiles(clip, tiles())


@pytest.mark.parametrize(
    "make_bad",
    [
        lambda: None,
        lambda: torch.empty((1, 2, 2560)),
        lambda: torch.empty((1, 12, 2, 2560)),
        lambda: torch.empty((2, 2, 30720)),
        lambda: torch.empty((1, 0, 30720)),
        lambda: torch.empty((1, 2, 30720), dtype=torch.int64),
        lambda: torch.empty((1, 2, 30720)).to_sparse(),
    ],
)
@pytest.mark.parametrize("cutoff", [0, 1])
def test_native_output_dimensions_checked_including_fallback(
    clip, monkeypatch, make_bad, cutoff
):
    monkeypatch.setattr(
        clip, "encode_from_tokens_scheduled", lambda tokens: [[make_bad(), {}]]
    )
    with pytest.raises(ValueError, match=r"\[1, tokens, 30720\]"):
        encode_tiles(clip, tiles(), end_percent=cutoff)


@pytest.mark.parametrize(
    "method,fail_at",
    [
        ("tokenize", 3),
        ("encode_from_tokens_scheduled", 3),
        ("encode_from_tokens_scheduled", 5),
    ],
)
@pytest.mark.parametrize(
    "exception", [RuntimeError, InterruptedError, asyncio.CancelledError]
)
def test_failure_and_cancellation_leave_inputs_reusable(
    clip, monkeypatch, method, fail_at, exception
):
    source = tiles()
    before = source.clone()
    original = getattr(clip, method)
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == fail_at:
            raise exception("deliberate encode failure")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(clip, method, fail)
        with pytest.raises(exception, match="deliberate encode failure"):
            encode_tiles(clip, source, end_percent=0.5)
    result = encode_tiles(clip, source, end_percent=0.5)
    repeat = encode_tiles(clip, source, end_percent=0.5)
    for first, second in zip(result, repeat, strict=True):
        for (left, _), (right, _) in zip(first, second, strict=True):
            torch.testing.assert_close(left, right)
    torch.testing.assert_close(source, before)
    assert all(
        "start_percent" not in meta for entries in clip.outputs for _, meta in entries
    )


def test_host_cancellation_is_checked_between_encodes(clip, host):
    host.interrupt_after = 5
    with pytest.raises(InterruptedError, match="cancelled"):
        encode_tiles(clip, tiles())
    assert len(clip.encoded) == 2
    host.interrupt_after = None
    assert len(encode_tiles(clip, tiles())) == 4


@pytest.mark.parametrize("fail", [False, True])
def test_invocation_tensors_are_released(clip, monkeypatch, fail):
    references = []
    calls = 0

    def tokenize(text, *, images, llama_template):
        references.extend(weakref.ref(image) for image in images)
        return images

    def encode(tokens):
        nonlocal calls
        calls += 1
        if fail and calls == 3:
            raise RuntimeError("failure")
        embedding = torch.ones(1, 1, 30720)
        references.append(weakref.ref(embedding))
        return [[embedding, {}]]

    monkeypatch.setattr(clip, "tokenize", tokenize)
    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", encode)
    try:
        result = encode_tiles(clip, tiles(), end_percent=0.5)
        del result
    except RuntimeError:
        assert fail
    gc.collect()
    assert references and all(reference() is None for reference in references)


def test_tileview_to_helper_to_sampler_has_one_trajectory(clip, host):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning, TileSampler, TileView

    args = sampler_arguments(cfg=1, negative=cond(0, width=30720))
    args["model"].model.diffusion_model.txtdim = 2560
    args["positive"] = cond(
        999,
        width=30720,
        reference_latents=[torch.ones(1, 16, 1, 3, 3)],
        reference_latents_method="index",
    )
    height, width = args["tile_plan"].pixel_hw
    canvas = torch.arange(height * width * 3, dtype=torch.float32).reshape(
        1, height, width, 3
    ) / (height * width * 3)
    references = TileView.execute(canvas, args["tile_plan"])[0]
    local = TileKrea2Conditioning.execute(
        **node_arguments(clip, reference_tiles=references),
        prompts=["top left", "top right", "bottom right", "bottom left"],
    )[0]
    result = TileSampler.execute(
        **{key: [value] for key, value in args.items()}, local_positive=local
    )
    assert len(host.common_calls) == 1
    assert result[0]["samples"].shape == (1, 16, 1, 12, 16)
    assert len({conditioning[0][0].mean().item() for conditioning in local}) == 4
    for index, (embedding, meta) in enumerate(host.common_calls[0]["positive"]):
        assert embedding is local[index][0][0]
        assert meta["tdng_tile_id"] == TILE_IDS[index]
        assert "reference_latents" not in meta
        torch.testing.assert_close(
            clip.tokenized[index]["images"][0], references[index : index + 1]
        )
    for index, (branches, *_rest) in enumerate(host.tile_calls):
        tile_index = index % 4
        assert branches[0][0]["cross_attn"] is local[tile_index][0][0]
        assert branches[0][0]["tdng_tile_id"] == TILE_IDS[tile_index]
    assert all(refs is None for _, _, _, refs, _, _ in host.krea2_calls)


def test_native_cutoff_boundary_and_clip_schedule_intersections(clip, host):
    from tiled_diffusion_ng.nodes import TileSampler

    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.25},
        {"clip_start_percent": 0.25, "clip_end_percent": 1},
    ]
    local = encode_tiles(clip, tiles(), strength=0.5, end_percent=0.5)
    args = sampler_arguments(cfg=1, negative=cond(0, width=30720))
    args["model"].model.diffusion_model.txtdim = 2560

    def solve(evaluate, x, sigmas):
        for index, sigma in enumerate((0.875, 0.625, 0.5, 0.375)):
            x = evaluate(x, torch.tensor([sigma]), (0, index))
        return x

    host.dispatch["euler"].sampler_function = solve
    TileSampler.execute(
        **{key: [value] for key, value in args.items()}, local_positive=local
    )
    assert len(host.common_calls) == 1 and len(host.tile_calls) == 16
    for step, indices in enumerate(((0,), (1,), (1, 3), (3,))):
        for tile in range(4):
            branches, _, sigma, _ = host.tile_calls[step * 4 + tile]
            active = [
                entry["cross_attn"]
                for entry in branches[0]
                if entry["timestep_end"] <= sigma[0] <= entry["timestep_start"]
            ]
            assert len(active) == len(indices)
            assert all(
                embedding is local[tile][index][0]
                for embedding, index in zip(active, indices, strict=True)
            )
    # Inclusive cutoff admits both the scaled image/text and unscaled fallback.
    assert all(
        "timestep_start" not in meta
        for conditioning in local
        for _, meta in conditioning
    )
