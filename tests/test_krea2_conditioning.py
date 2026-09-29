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
import sys

import pytest
import torch
import torch.nn.functional as F

from tiled_diffusion_ng import comfy_entrypoint
from tiled_diffusion_ng._krea2_conditioning import encode_tiles
from tiled_diffusion_ng.geometry import TILE_IDS

from . import krea2_conditioning_host as encoder_host
from .host import arguments
from .krea2_host import cond

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


@pytest.mark.parametrize(
    "prompts,channels",
    [(None, 4), (["red", "green", "blue", "white"], 3)],
    ids=["image_only_rgba", "prompted_rgb"],
)
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
        assert len(tokens["images"]) == 1
        torch.testing.assert_close(
            tokens["images"][0], source[index : index + 1, ..., :3]
        )
        assert conditioning[0][0] is native[0][0]
        assert conditioning is not native and conditioning[0] is not native[0]
        assert conditioning[0][1] is not native[0][1]
        assert conditioning[0][1].keys() == native[0][1].keys()
        assert all(
            conditioning[0][1][key] is value for key, value in native[0][1].items()
        )
        assert "reference_latents" not in conditioning[0][1]
    torch.testing.assert_close(source, before)


@pytest.mark.parametrize(
    "prompts,baseline",
    [(None, []), (["red", "", "lake", "bird"], ["same", "same", " same", "same "])],
    ids=["defaults", "literal_deduplication"],
)
def test_baseline_list_transport_and_independent_literal_encoding(
    clip, prompts, baseline
):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    prompts_before = None if prompts is None else prompts.copy()
    baseline_before = None if baseline is None else baseline.copy()
    output = TileKrea2Conditioning.execute(
        **node_arguments(clip, strength=0.25, end_percent=0.5),
        prompts=prompts,
        baseline=baseline,
    )[0]
    texts = baseline or [""] * 4
    # Whitespace is significant; only identical literals share an encode.
    distinct_texts = list(dict.fromkeys(texts))
    assert len(clip.encoded) == 4 + len(distinct_texts)
    assert [tokens["text"] for tokens in clip.tokenized[:4]] == [
        VISION + prompt for prompt in (prompts or [""] * 4)
    ]
    assert [tokens["text"] for tokens in clip.tokenized[4:]] == distinct_texts
    natives = dict(zip(distinct_texts, clip.outputs[4:], strict=True))
    for tokens in clip.tokenized[4:]:
        assert tokens["images"] == []
        assert tokens["template"] == encoder_host.KREA2_TEMPLATE
    for index, (conditioning, text) in enumerate(zip(output, texts, strict=True)):
        assert len(conditioning) == 3
        assert conditioning[0][0] is clip.outputs[index][0][0]
        assert conditioning[1][0] is conditioning[2][0] is natives[text][0][0]
        assert [meta.get("strength", 1) for _, meta in conditioning] == [0.25, 0.75, 1]
        assert [
            (meta["start_percent"], meta["end_percent"]) for _, meta in conditioning
        ] == [(0, 0.5), (0, 0.5), (0.5, 1)]
    assert prompts == prompts_before and baseline == baseline_before


def test_baseline_cache_is_per_invocation_and_containers_are_independent(clip):
    baseline = ["red", "bed", "red", "bed"]
    original_attributes = set(clip.__dict__)
    first = encode_tiles(
        clip, tiles(), strength=0.5, end_percent=0.5, baseline=baseline
    )
    second = encode_tiles(
        clip, tiles(), strength=0.5, end_percent=0.5, baseline=baseline
    )
    assert len(clip.encoded) == 12
    assert [tokens["text"] for tokens in clip.encoded if not tokens["images"]] == [
        "red",
        "bed",
        "red",
        "bed",
    ]
    assert set(clip.__dict__) == original_attributes
    torch.testing.assert_close(first, second)
    for index in range(4):
        assert first[index][1][0] is not second[index][1][0]
        assert first[index][1][0] is first[index][2][0] is first[(index + 2) % 4][1][0]
    entries = [entry for result in (first, second) for tile in result for entry in tile]
    assert len({id(entry) for entry in entries}) == len(entries)
    assert len({id(entry[1]) for entry in entries}) == len(entries)
    first[0][1][1]["changed"] = True
    assert all(
        "changed" not in entry[1] for entry in entries if entry is not first[0][1]
    )
    assert all("changed" not in meta for output in clip.outputs for _, meta in output)


def test_prediction_weights_preserve_native_tensors_and_metadata(clip):
    marker = object()
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5, "opaque": marker},
        {"clip_start_percent": 0.5, "clip_end_percent": 1, "strength": 0.7},
    ]
    result = encode_tiles(clip, tiles(), strength=0.25, baseline=["baseline"] * 4)
    assert len(clip.encoded) == 5
    for index, actual in enumerate(result):
        sources = [(clip.outputs[index], 0.25), (clip.outputs[-1], 0.75)]
        assert len(actual) == 4
        for group, (native, weight) in enumerate(sources):
            for (embedding, meta), (original, original_meta) in zip(
                actual[group * 2 : group * 2 + 2], native, strict=True
            ):
                # Weights multiply native entry strengths; tensors stay shared.
                assert embedding is original and meta is not original_meta
                expected = original_meta.copy()
                expected["strength"] = original_meta.get("strength", 1) * weight
                assert meta.keys() == expected.keys()
                for key, value in expected.items():
                    if key == "strength":
                        assert meta[key] == value
                    else:
                        assert meta[key] is value
    assert all(
        "strength" not in native[0][1] and native[1][1]["strength"] == 0.7
        for native in clip.outputs
    )


@pytest.mark.parametrize("cutoff,strength", [(0.35, 0.25), (0.35, 1), (1, 0.25)])
def test_cutoff_baselines_have_independent_containers(clip, cutoff, strength):
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5},
        {"clip_start_percent": 0.5, "clip_end_percent": 1},
    ]
    output = encode_tiles(
        clip, tiles(), ["tile text"] * 4, strength=strength, end_percent=cutoff
    )
    assert len(clip.encoded) == 5
    assert clip.tokenized[-1]["text"] == "" and clip.tokenized[-1]["images"] == []
    for index, conditioning in enumerate(output):
        interval = (0, cutoff) if cutoff < 1 else None
        groups = [(clip.outputs[index], strength, interval)]
        if strength < 1:
            groups.append((clip.outputs[-1], 1 - strength, interval))
        if cutoff < 1:
            groups.append((clip.outputs[-1], 1, (cutoff, 1)))
        assert len(conditioning) == 2 * len(groups)
        for group, (native, weight, interval) in enumerate(groups):
            for (embedding, meta), (original, original_meta) in zip(
                conditioning[group * 2 : group * 2 + 2], native, strict=True
            ):
                assert embedding is original
                assert meta.get("strength", 1) == weight
                assert meta["clip_start_percent"] == original_meta["clip_start_percent"]
                assert meta["clip_end_percent"] == original_meta["clip_end_percent"]
                if interval is None:
                    assert "start_percent" not in meta and "end_percent" not in meta
                else:
                    assert (meta["start_percent"], meta["end_percent"]) == interval
    entries = [entry for conditioning in output for entry in conditioning]
    assert len({id(conditioning) for conditioning in output}) == 4
    assert len({id(entry) for entry in entries}) == len(entries)
    assert len({id(entry[1]) for entry in entries}) == len(entries)
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
    result = encode_tiles(clip, tiles(), strength=0.5, end_percent=0.5)
    for conditioning in result:
        assert [
            (meta["start_percent"], meta["end_percent"]) for _, meta in conditioning
        ] == [(0.2, 0.5), (0.2, 0.5), (0.5, 0.8)]
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
        (1111, 1777, (810, 1295)),
        (1, 4194304, (1, 2097152)),
        (1024, 1024, None),
    ],
)
def test_downsize_dispatch_preserves_aspect_without_enlarging(
    clip, monkeypatch, height, width, expected
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
    encode_tiles(clip, source, downsize_to_1mp=True)
    target = expected or (height, width)
    assert len(calls) == (4 if expected else 0)
    for image, size, mode, align_corners, antialias in calls:
        assert image.shape == (1, 3, height, width)
        assert size == expected
        assert (mode, align_corners, antialias) == ("bicubic", False, True)
    for index, tokens in enumerate(clip.tokenized):
        assert tokens["images"][0].shape == (1, *target, 3)
        torch.testing.assert_close(
            tokens["images"][0][0, 0, 0], source[index, 0, 0, :3]
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float64])
def test_real_bicubic_downsize_preserves_precision_and_source(clip, host, dtype):
    # These colors cannot survive a uint8 round trip unchanged.
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
    "enabled,size,dtype",
    [(False, 1025, torch.bfloat16), (True, 8, torch.float64)],
    ids=["disabled", "small"],
)
def test_unresized_images_keep_exact_pixels_and_dtype(
    clip, monkeypatch, enabled, size, dtype
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


@pytest.mark.parametrize(
    "strength,cutoff,baseline",
    [(0, 0.5, ["red", "bed", "red", ""]), (1, 0, None)],
    ids=["zero_strength", "zero_cutoff"],
)
def test_baseline_endpoints_skip_tile_preparation(
    clip, monkeypatch, strength, cutoff, baseline
):
    from tiled_diffusion_ng import _krea2_conditioning

    def unexpected_resize(*args, **kwargs):
        pytest.fail("Baseline-only endpoints must skip image preparation")

    monkeypatch.setattr(_krea2_conditioning, "_image", unexpected_resize)
    result = encode_tiles(
        clip,
        tiles(1, 1).expand(4, 2048, 2048, 3),
        strength=strength,
        end_percent=cutoff,
        downsize_to_1mp=True,
        baseline=baseline,
    )
    assert len(clip.tokenized) == len(set(baseline or [""]))
    assert all(not tokens["images"] for tokens in clip.tokenized)
    assert all(len(conditioning) == 1 for conditioning in result)
    assert all(
        "strength" not in meta
        and "start_percent" not in meta
        and "end_percent" not in meta
        for conditioning in result
        for _, meta in conditioning
    )


def test_full_reference_endpoint_skips_custom_baseline_encodes(clip):
    encode_tiles(clip, tiles(), baseline=["unused"] * 4)
    assert len(clip.encoded) == 4
    assert all(tokens["images"] for tokens in clip.encoded)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        torch.empty((3, 6, 10, 3)),
        torch.empty((4, 6, 10, 2)),
        tiles().to(torch.int64),
    ],
    ids=["missing", "three_tiles", "two_channels", "integer"],
)
def test_invalid_images_fail_before_encoding(clip, bad):
    with pytest.raises(ValueError, match="reference_tiles"):
        encode_tiles(clip, bad)
    assert not clip.tokenized


@pytest.mark.parametrize(
    "name,value",
    [
        ("prompts", "text"),
        ("prompts", [""] * 3),
        ("baseline", [1] * 4),
        ("baseline", ("",) * 4),
    ],
)
def test_invalid_string_lists_fail_before_encoding(clip, name, value):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    with pytest.raises(ValueError, match=name + " requires exactly four"):
        TileKrea2Conditioning.execute(**node_arguments(clip), **{name: value})
    assert not clip.tokenized


@pytest.mark.parametrize(
    "name,value",
    [
        ("strength", float("nan")),
        ("strength", 1.001),
        ("strength", True),
        ("end_percent", -0.1),
        ("end_percent", 1.5),
        ("end_percent", "1"),
        ("downsize_to_1mp", 1),
    ],
)
def test_invalid_settings_fail_before_encoding(clip, name, value):
    with pytest.raises(ValueError, match=name):
        encode_tiles(clip, tiles(), **{name: value})
    assert not clip.tokenized


@pytest.mark.parametrize("settings", [{"strength": 0}, {"end_percent": 0}])
def test_validation_precedes_endpoint_shortcuts(clip, settings):
    with pytest.raises(ValueError, match="reference_tiles"):
        encode_tiles(clip, tiles()[:3], **settings)
    with pytest.raises(ValueError, match="prompts requires exactly four"):
        encode_tiles(clip, tiles(), prompts=["one"], **settings)
    assert not clip.tokenized


@pytest.mark.parametrize("missing", ["module", "KREA2_TEMPLATE"])
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


@pytest.mark.parametrize("field,strength", [("cond_stage_model", 0), ("tokenize", 1)])
def test_incompatible_clip_rejected(clip, field, strength):
    setattr(clip, field, object())
    with pytest.raises(ValueError, match="native Krea2 CLIP"):
        encode_tiles(clip, tiles(), strength=strength, baseline=["custom"] * 4)
    assert not clip.tokenized


def test_native_tokenizer_and_model_subclasses_are_accepted(clip):
    clip.tokenizer = type("TokenizerSubclass", (encoder_host.Krea2Tokenizer,), {})()
    assert len(encode_tiles(clip, tiles())) == 4


@pytest.mark.parametrize("template", [None, "{}{}"])
def test_missing_template_capability_rejected(clip, monkeypatch, template):
    monkeypatch.setattr(
        sys.modules["comfy.text_encoders.krea2"], "KREA2_TEMPLATE", template
    )
    with pytest.raises(ValueError, match="conditioning template"):
        encode_tiles(clip, tiles())
    assert not clip.tokenized


@pytest.mark.parametrize("bad", [[], [[None]]], ids=["empty", "nesting"])
def test_malformed_native_conditioning_rejected(clip, monkeypatch, bad):
    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", lambda tokens: bad)
    with pytest.raises(ValueError, match="Native Krea2 encoding must return"):
        encode_tiles(clip, tiles())


@pytest.mark.parametrize(
    "bad",
    [
        torch.empty((1, 2, 2560)),
        torch.empty((2, 2, 30720)),
        torch.empty((1, 2, 30720), dtype=torch.int64),
    ],
    ids=["one_layer", "batch", "integer"],
)
def test_native_output_dimensions_checked(clip, monkeypatch, bad):
    monkeypatch.setattr(
        clip, "encode_from_tokens_scheduled", lambda tokens: [[bad, {}]]
    )
    with pytest.raises(ValueError, match=r"\[1, tokens, 30720\]"):
        encode_tiles(clip, tiles())


def test_invalid_baseline_output_is_checked_after_reference_encodes(clip, monkeypatch):
    encode = clip.encode_from_tokens_scheduled

    def invalid_baseline(tokens):
        return encode(tokens) if tokens["images"] else [[torch.ones(1, 1, 2560), {}]]

    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", invalid_baseline)
    with pytest.raises(ValueError, match=r"\[1, tokens, 30720\]"):
        encode_tiles(clip, tiles(), strength=0.5, baseline=["custom"] * 4)
    assert len(clip.encoded) == 4


def test_host_cancellation_is_checked_between_encodes(clip, host):
    host.interrupt_after = 10
    settings = {"strength": 0.5, "baseline": ["red", "bed", "lake", "bird"]}
    with pytest.raises(InterruptedError, match="cancelled"):
        encode_tiles(clip, tiles(), **settings)
    # Checks surround each encode; the fifth finished before cancellation.
    assert len(clip.encoded) == 5


def test_tileview_to_helper_to_sampler_has_one_trajectory(clip, host):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning, TileSampler, TileView

    args = arguments("krea2", cfg=1, negative=cond(0, width=30720), hw=(13, 15))
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
    assert result[0]["samples"].shape == (1, 16, 1, 13, 15)
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
