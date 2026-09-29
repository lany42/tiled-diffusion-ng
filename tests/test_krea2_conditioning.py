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
import math
import sys
import weakref

import pytest
import torch
import torch.nn.functional as F

from tiled_diffusion_ng import comfy_entrypoint
from tiled_diffusion_ng._krea2_conditioning import encode_tiles
from tiled_diffusion_ng.geometry import TILE_IDS

from . import krea2_conditioning_host as encoder_host
from . import krea2_host
from .krea2_host import cond
from .test_krea2 import arguments as sampler_arguments

VISION = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture
def clip(host, monkeypatch):
    encoder_host.install(monkeypatch)
    return encoder_host.Clip()


@pytest.fixture
def predictions(host, monkeypatch):
    from tiled_diffusion_ng.nodes import TileSampler

    forward = krea2_host.forward

    def nonlinear(host, x, sigma, context, *args, **kwargs):
        output = forward(host, x, sigma, context, *args, **kwargs)
        text = context.mean((1, 2)).to(x).reshape(-1, 1, 1, 1, 1)
        return output + text.square() / 32

    monkeypatch.setattr(krea2_host, "forward", nonlinear)

    def run(local, *, batch=1, cfg=1, sigmas=(0.625,)):
        args = sampler_arguments(batch=batch, cfg=cfg, negative=cond(-3, width=30720))
        args["model"].model.diffusion_model.txtdim = 2560
        # Complete replacement excludes the global embedding and its references.
        args["positive"] = cond(
            999,
            width=30720,
            reference_latents=[torch.ones(1, 16, 1, 3, 3)],
            reference_latents_method="index",
        )
        hooks, captures, tile_outputs, tile_model_calls = [], [], [], []

        def pre(data):
            hooks.append("pre")
            assert data["conds_out"][0].shape == (batch, 16, 1, 12, 16)
            return data["conds_out"]

        def guide(data):
            hooks.append("cfg")
            return data["uncond"] + data["cond_scale"] * (data["cond"] - data["uncond"])

        def post(data):
            hooks.append("post")
            captures.append(data)
            return data["denoised"] + 0.125

        args["model"].model_options.update(
            sampler_pre_cfg_function=[pre],
            sampler_cfg_function=guide,
            sampler_post_cfg_function=[post],
        )

        def solve(evaluate, x, schedule):
            # Each comparison evaluates the same current latent at a chosen
            # sigma, never a trajectory whose evolving samples could diverge.
            for index, sigma in enumerate(sigmas):
                result = evaluate(x, torch.full((batch,), sigma), (0, index))
            return result

        leaf = host.leaf

        def record_tiles(*args):
            model_start = len(host.krea2_calls)
            result = leaf(*args)
            tile_outputs.append(result)
            tile_model_calls.append(host.krea2_calls[model_start:])
            return result

        common_start, model_start, tile_start = (
            len(host.common_calls),
            len(host.krea2_calls),
            len(host.tile_calls),
        )
        with monkeypatch.context() as patch:
            patch.setattr(host, "leaf", record_tiles)
            patch.setattr(host.dispatch["euler"], "sampler_function", solve)
            output = TileSampler.execute(
                **{key: [value] for key, value in args.items()}, local_positive=local
            )[0]
        assert len(host.common_calls) == common_start + 1
        assert hooks == ["pre", "cfg", "post"] * len(sigmas)
        for data in captures:
            torch.testing.assert_close(
                data["denoised"],
                data["uncond_denoised"]
                + cfg * (data["cond_denoised"] - data["uncond_denoised"]),
            )
        torch.testing.assert_close(output["samples"], captures[-1]["denoised"] + 0.125)
        assert not args["model"].wrappers
        assert not args["latent_image"]["samples"].any()
        assert args["model"].model_options["sampler_cfg_function"] is guide
        assert args["model"].model_options["sampler_post_cfg_function"] == [post]
        assert all(
            "tdng_tile_id" not in meta for entries in local for _, meta in entries
        )
        assert all(
            refs is None for _, _, _, refs, _, _ in host.krea2_calls[model_start:]
        )
        return {
            "captures": captures,
            "model_calls": host.krea2_calls[model_start:],
            "tile_calls": host.tile_calls[tile_start:],
            "tile_outputs": tile_outputs,
            "tile_model_calls": tile_model_calls,
        }

    return run


def evaluated_contexts(run, sigma, branch, batch):
    # Native batching can evaluate several condition entries in one forward.
    # Count batch examples after schedule filtering, including CFG negatives.
    examples = []
    for x, timestep, context, _refs, _method, options in run["model_calls"]:
        if timestep[0].item() != sigma:
            continue
        assert len(context) == len(x) == batch * len(options["cond_or_uncond"])
        for index, branch_index in enumerate(options["cond_or_uncond"]):
            if branch_index == branch:
                examples.extend(context[index * batch : (index + 1) * batch])
    return examples


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
        "baseline",
    ]
    for name in ("prompts", "baseline"):
        assert inputs[name].optional and inputs[name].force_input
        assert inputs[name].dynamic_prompts is False
    for name, expected in (
        ("strength", (1, 0, 1, 0.05)),
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


@pytest.mark.parametrize("prompts", [None, ["  {red|blue}  ", "", "lake", "{{bird}}"]])
@pytest.mark.parametrize(
    "baseline",
    [
        None,
        [],
        [""] * 4,
        ["red", "bed", "lake", "bird"],
        ["", "  {literal|braces} \n", "{{literal}}", ""],
        ["same", "same", " same", "same "],
    ],
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
        assert tokens["rendered"] == encoder_host.KREA2_TEMPLATE.format(tokens["text"])
    for index, (conditioning, text) in enumerate(zip(output, texts, strict=True)):
        assert len(conditioning) == 3
        assert conditioning[0][0] is clip.outputs[index][0][0]
        assert conditioning[1][0] is conditioning[2][0] is natives[text][0][0]
        assert [meta.get("strength", 1) for _, meta in conditioning] == [0.25, 0.75, 1]
        assert (
            conditioning[1][1]["start_percent"],
            conditioning[1][1]["end_percent"],
        ) == (
            0,
            0.5,
        )
        assert (
            conditioning[2][1]["start_percent"],
            conditioning[2][1]["end_percent"],
        ) == (
            0.5,
            1,
        )
    assert len({entries[0][0].mean().item() for entries in natives.values()}) == len(
        distinct_texts
    )
    assert prompts == prompts_before and baseline == baseline_before


def test_same_list_can_supply_prompts_and_baseline_positionally(clip):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    texts = ["sun", "sea", "oak", "sky"]
    source = tiles()
    # The added optional argument follows every existing positional argument.
    direct = encode_tiles(clip, source, texts, 0.5, 0.5, False, texts)
    output = TileKrea2Conditioning.execute(
        [clip], [source], [0.5], [0.5], [False], texts, texts
    )[0]
    torch.testing.assert_close(direct, output)
    for offset in (0, 8):
        assert [tokens["text"] for tokens in clip.tokenized[offset : offset + 8]] == [
            *(VISION + text for text in texts),
            *texts,
        ]
    assert texts == ["sun", "sea", "oak", "sky"]


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


@pytest.mark.parametrize("strength", [0, 0.25, 0.5, 1])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_prediction_weights_preserve_native_tensors_and_metadata(clip, strength, dtype):
    clip.dtype = dtype
    marker = object()
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5, "opaque": marker},
        {"clip_start_percent": 0.5, "clip_end_percent": 1, "strength": 0.7},
    ]
    result = encode_tiles(clip, tiles(), strength=strength, baseline=["baseline"] * 4)
    assert len(clip.encoded) == (1 if strength == 0 else 4 if strength == 1 else 5)
    for index, actual in enumerate(result):
        sources = (
            [(clip.outputs[0], 1)]
            if strength == 0
            else [(clip.outputs[index], 1)]
            if strength == 1
            else [(clip.outputs[index], strength), (clip.outputs[-1], 1 - strength)]
        )
        assert len(actual) == 2 * len(sources)
        for group, (native, weight) in enumerate(sources):
            for (embedding, meta), (original, original_meta) in zip(
                actual[group * 2 : group * 2 + 2], native, strict=True
            ):
                assert embedding is original and embedding.dtype == dtype
                assert embedding.any()
                assert meta is not original_meta
                expected = original_meta.copy()
                if weight != 1:
                    expected["strength"] = original_meta.get("strength", 1) * weight
                assert meta.keys() == expected.keys()
                for key, value in expected.items():
                    if key == "strength":
                        assert meta[key] == value
                        if weight == 1:
                            assert meta[key] is original_meta[key]
                    else:
                        assert meta[key] is value
    assert all(
        "strength" not in native[0][1] and native[1][1]["strength"] == 0.7
        for native in clip.outputs
    )


@pytest.mark.parametrize("cutoff", [0, 0.35, 1])
@pytest.mark.parametrize("strength", [0, 0.25, 1])
def test_cutoff_baselines_have_independent_containers(clip, cutoff, strength):
    clip.schedules = [
        {"clip_start_percent": 0, "clip_end_percent": 0.5},
        {"clip_start_percent": 0.5, "clip_end_percent": 1},
    ]
    output = encode_tiles(
        clip, tiles(), ["tile text"] * 4, strength=strength, end_percent=cutoff
    )
    baseline_only = cutoff == 0 or strength == 0
    has_baseline = baseline_only or strength < 1 or cutoff < 1
    assert len(clip.encoded) == (1 if baseline_only else 4 + has_baseline)
    assert len(output) == 4
    if has_baseline:
        assert clip.tokenized[-1]["text"] == ""
        assert clip.tokenized[-1]["images"] == []
        assert clip.tokenized[-1]["template"] == encoder_host.KREA2_TEMPLATE
    for index, conditioning in enumerate(output):
        if baseline_only:
            groups = [(clip.outputs[-1], 1, None)]
        else:
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
                assert embedding is original and embedding.any()
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


@pytest.mark.parametrize("cutoff", [0.5, 1])
def test_cutoff_intersects_existing_ranges_without_rewriting_clip_schedule(
    clip, cutoff
):
    clip.schedules = [
        {
            "start_percent": 0.2,
            "end_percent": 0.8,
            "clip_start_percent": 0.3,
            "clip_end_percent": 0.7,
        }
    ]
    result = encode_tiles(clip, tiles(), strength=0.5, end_percent=cutoff)
    for conditioning in result:
        assert [
            (meta["start_percent"], meta["end_percent"]) for _, meta in conditioning
        ] == ([(0.2, 0.5), (0.2, 0.5), (0.5, 0.8)] if cutoff < 1 else [(0.2, 0.8)] * 2)
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


@pytest.mark.parametrize(
    "strength,cutoff", [(0, 0), (0, 0.5), (0, 1), (0.5, 0), (1, 0)]
)
@pytest.mark.parametrize("baseline", [None, ["red", "bed", "red", ""]])
def test_baseline_endpoints_skip_tile_preparation(
    clip, monkeypatch, strength, cutoff, baseline
):
    from tiled_diffusion_ng import _krea2_conditioning

    def unexpected_resize(*args, **kwargs):
        pytest.fail("Baseline-only endpoints must skip image preparation")

    monkeypatch.setattr(F, "interpolate", unexpected_resize)
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
    "strength", [math.nextafter(0.0, 1.0), math.nextafter(1.0, 0.0)]
)
def test_near_endpoint_strengths_are_not_snapped(clip, strength):
    output = encode_tiles(clip, tiles(), strength=strength)
    assert len(clip.encoded) == 5
    for conditioning in output:
        assert len(conditioning) == 2
        assert conditioning[0][1]["strength"] == strength
        assert conditioning[1][1].get("strength", 1) == 1 - strength


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
@pytest.mark.parametrize("settings", [{}, {"strength": 0}, {"end_percent": 0}])
def test_invalid_images_fail_before_encoding(clip, make_bad, settings):
    with pytest.raises(ValueError, match="reference_tiles"):
        encode_tiles(clip, make_bad(), **settings)
    assert not clip.tokenized


@pytest.mark.parametrize(
    "value",
    [
        "",
        "text",
        (),
        ["one"],
        [""] * 3,
        [""] * 5,
        ("",) * 4,
        ["", "", "", None],
        [1] * 4,
        [[""]] * 4,
    ],
)
@pytest.mark.parametrize("name", ["prompts", "baseline"])
@pytest.mark.parametrize("settings", [{}, {"strength": 0}, {"end_percent": 0}])
def test_invalid_string_lists_fail_before_encoding_even_at_endpoints(
    clip, name, value, settings
):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    with pytest.raises(ValueError, match=name + " requires exactly four"):
        TileKrea2Conditioning.execute(
            **node_arguments(clip, **settings), **{name: value}
        )
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
            1.001,
            1.75,
            3,
            3.1,
            True,
            "1",
            None,
            torch.tensor(1.0),
        )
    ]
    + [
        ("downsize_to_1mp", 1),
        ("downsize_to_1mp", "false"),
        ("downsize_to_1mp", None),
    ],
)
@pytest.mark.parametrize("settings", [{}, {"strength": 0}, {"end_percent": 0}])
def test_invalid_settings_fail_before_encoding(clip, name, value, settings):
    with pytest.raises(ValueError, match=name):
        encode_tiles(clip, tiles(), **{**settings, name: value})
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
@pytest.mark.parametrize("strength", [0, 1])
def test_incompatible_clip_rejected(clip, field, strength):
    setattr(clip, field, object())
    with pytest.raises(ValueError, match="native Krea2 CLIP"):
        encode_tiles(clip, tiles(), strength=strength, baseline=["custom"] * 4)
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
@pytest.mark.parametrize("strength", [0, 1])
def test_malformed_native_conditioning_rejected(clip, monkeypatch, bad, strength):
    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", lambda tokens: bad)
    with pytest.raises(ValueError, match="Native Krea2 encoding must return"):
        encode_tiles(clip, tiles(), strength=strength, baseline=["custom"] * 4)


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
        encode_tiles(clip, tiles(), end_percent=cutoff, baseline=["custom"] * 4)


def test_invalid_baseline_output_is_checked_after_reference_encodes(clip, monkeypatch):
    encode = clip.encode_from_tokens_scheduled

    def invalid_baseline(tokens):
        return encode(tokens) if tokens["images"] else [[torch.ones(1, 1, 2560), {}]]

    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", invalid_baseline)
    with pytest.raises(ValueError, match=r"\[1, tokens, 30720\]"):
        encode_tiles(clip, tiles(), strength=0.5, baseline=["custom"] * 4)
    assert len(clip.encoded) == 4


@pytest.mark.parametrize(
    "method,fail_at",
    [
        ("tokenize", 3),
        ("tokenize", 6),
        ("encode_from_tokens_scheduled", 3),
        ("encode_from_tokens_scheduled", 5),
        ("encode_from_tokens_scheduled", 6),
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
    prompts = ["reference"] * 4
    baseline = ["custom", "{literal} ", "custom", ""]
    settings = {
        "prompts": prompts,
        "strength": 0.5,
        "end_percent": 0.5,
        "baseline": baseline,
    }
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
            encode_tiles(clip, source, **settings)
    encodes_before = len(clip.encoded)
    result = encode_tiles(clip, source, **settings)
    repeat = encode_tiles(clip, source, **settings)
    assert len(clip.encoded) == encodes_before + 14
    torch.testing.assert_close(result, repeat)
    torch.testing.assert_close(source, before)
    assert baseline == ["custom", "{literal} ", "custom", ""]
    assert prompts == ["reference"] * 4
    assert all(
        "start_percent" not in meta for entries in clip.outputs for _, meta in entries
    )


@pytest.mark.parametrize("interrupt_after", [5, 9, 10, 11, 12])
def test_host_cancellation_is_checked_between_encodes(clip, host, interrupt_after):
    host.interrupt_after = interrupt_after
    settings = {"strength": 0.5, "baseline": ["red", "bed", "lake", "bird"]}
    with pytest.raises(InterruptedError, match="cancelled"):
        encode_tiles(clip, tiles(), **settings)
    assert len(clip.encoded) == interrupt_after // 2
    host.interrupt_after = None
    encodes_before = len(clip.encoded)
    assert len(encode_tiles(clip, tiles(), **settings)) == 4
    assert len(clip.encoded) == encodes_before + 8


@pytest.mark.parametrize(
    "failure", [None, RuntimeError, InterruptedError, asyncio.CancelledError]
)
@pytest.mark.parametrize("fail_at", [3, 6])
def test_invocation_tensors_are_released(clip, monkeypatch, failure, fail_at):
    references = []
    calls = 0

    def tokenize(text, *, images, llama_template):
        references.extend(weakref.ref(image) for image in images)
        return images

    def encode(tokens):
        nonlocal calls
        calls += 1
        if failure and calls == fail_at:
            raise failure("failure")
        embedding = torch.ones(1, 1, 30720)
        pooled = torch.ones(1, 30720)
        mask = torch.ones(1, 1)
        references.extend(weakref.ref(tensor) for tensor in (embedding, pooled, mask))
        return [[embedding, {"pooled_output": pooled, "attention_mask": mask}]]

    monkeypatch.setattr(clip, "tokenize", tokenize)
    monkeypatch.setattr(clip, "encode_from_tokens_scheduled", encode)
    try:
        result = encode_tiles(
            clip,
            tiles(),
            strength=0.5,
            end_percent=0.5,
            baseline=["red", "bed", "lake", "bird"],
        )
        del result
    except (RuntimeError, InterruptedError, asyncio.CancelledError):
        assert failure is not None
    assert calls == (8 if failure is None else fail_at)
    gc.collect()
    assert references and all(reference() is None for reference in references)


@pytest.mark.parametrize("hw", [(12, 16), (13, 15)])
def test_tileview_to_helper_to_sampler_has_one_trajectory(clip, host, hw):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning, TileSampler, TileView

    args = sampler_arguments(cfg=1, negative=cond(0, width=30720), hw=hw)
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
    assert result[0]["samples"].shape == (1, 16, 1, *hw)
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


@pytest.mark.parametrize("strength", [0, 0.25, 0.5, 1])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("cfg", [1, 4.5])
@pytest.mark.parametrize("baseline", [None, ["", "bed", "red", "lake"]])
def test_blend_matches_independent_nonlinear_predictions(
    clip, predictions, strength, batch, cfg, baseline
):
    settings = {"prompts": ["red", "reed", "tall", "blue"], "baseline": baseline}
    reference = encode_tiles(clip, tiles(), **settings)
    text_only = encode_tiles(clip, tiles(), strength=0, **settings)
    local = encode_tiles(clip, tiles(), strength=strength, **settings)
    assert all(
        reference[index][0][0].shape[1] != text_only[index][0][0].shape[1]
        for index in range(4)
    )
    p = predictions(reference, batch=batch, cfg=cfg)
    b = predictions(text_only, batch=batch, cfg=cfg)
    actual = predictions(local, batch=batch, cfg=cfg)
    for index in range(4):
        torch.testing.assert_close(
            actual["tile_outputs"][index][0],
            strength * p["tile_outputs"][index][0]
            + (1 - strength) * b["tile_outputs"][index][0],
        )
        for source in (p, b):
            torch.testing.assert_close(
                actual["tile_calls"][index][1], source["tile_calls"][index][1]
            )
            torch.testing.assert_close(
                actual["tile_calls"][index][2], source["tile_calls"][index][2]
            )
    for key in ("cond_denoised", "denoised"):
        torch.testing.assert_close(
            actual["captures"][0][key],
            strength * p["captures"][0][key] + (1 - strength) * b["captures"][0][key],
        )
    for model_calls in actual["tile_model_calls"]:
        per_tile = {"model_calls": model_calls}
        assert len(evaluated_contexts(per_tile, 0.625, 0, batch)) == batch * (
            2 if 0 < strength < 1 else 1
        )
        assert len(evaluated_contexts(per_tile, 0.625, 1, batch)) == (
            batch if cfg > 1 else 0
        )


@pytest.mark.parametrize("strength", [0.25, 0.5, 1])
@pytest.mark.parametrize("batch,cfg", [(1, 1), (2, 4.5)])
@pytest.mark.parametrize("scheduled", [False, True])
def test_cutoff_filters_model_examples_and_matches_baseline_afterward(
    clip, predictions, strength, batch, cfg, scheduled
):
    if scheduled:
        clip.schedules = [
            {"clip_start_percent": 0, "clip_end_percent": 0.25},
            {"clip_start_percent": 0.25, "clip_end_percent": 1},
        ]
    settings = {
        "prompts": ["red", "reed", "tall", "blue"],
        "baseline": ["", "bed", "red", "lake"],
    }
    reference = encode_tiles(clip, tiles(), **settings)
    text_only = encode_tiles(clip, tiles(), strength=0, **settings)
    local = encode_tiles(clip, tiles(), strength=strength, end_percent=0.5, **settings)
    sigmas = (0.875, 0.75, 0.625, 0.5, 0.375)
    p = predictions(reference, batch=batch, cfg=cfg, sigmas=sigmas)
    b = predictions(text_only, batch=batch, cfg=cfg, sigmas=sigmas)
    actual = predictions(local, batch=batch, cfg=cfg, sigmas=sigmas)
    assert len(actual["tile_calls"]) == 4 * len(sigmas)
    for step, sigma in enumerate(sigmas):
        # Both intervals include the cutoff: s*P + (1-s)*B + B is
        # normalized by 2. The extra baseline may batch with the active one.
        coefficient = strength if sigma > 0.5 else strength / 2 if sigma == 0.5 else 0
        entries = (
            (1 + (strength < 1))
            if sigma > 0.5
            else (2 + (strength < 1))
            if sigma == 0.5
            else 1
        )
        if scheduled and sigma == 0.75:
            entries *= 2  # Both baked CLIP entries also include their boundary.
        for key in ("cond_denoised", "denoised"):
            torch.testing.assert_close(
                actual["captures"][step][key],
                coefficient * p["captures"][step][key]
                + (1 - coefficient) * b["captures"][step][key],
            )
        for tile in range(4):
            index = step * 4 + tile
            torch.testing.assert_close(
                actual["tile_outputs"][index][0],
                coefficient * p["tile_outputs"][index][0]
                + (1 - coefficient) * b["tile_outputs"][index][0],
            )
            per_tile = {"model_calls": actual["tile_model_calls"][index]}
            positive = evaluated_contexts(per_tile, sigma, 0, batch)
            assert len(positive) == entries * batch
            assert len(evaluated_contexts(per_tile, sigma, 1, batch)) == (
                batch if cfg > 1 else 0
            )
            if sigma < 0.5:
                baseline_calls = {"model_calls": b["tile_model_calls"][index]}
                expected = evaluated_contexts(baseline_calls, sigma, 0, batch)
                torch.testing.assert_close(positive, expected, rtol=0, atol=0)
                for context in positive:
                    assert not any(
                        torch.equal(context, embedding[0])
                        for embedding, _ in reference[tile]
                    )
                assert sum(len(call[0]) for call in per_tile["model_calls"]) == sum(
                    len(call[0]) for call in baseline_calls["model_calls"]
                )
    if strength < 1:
        assert any(
            len(x) > batch for x, sigma, *_ in actual["model_calls"] if sigma[0] == 0.5
        )
    assert all(
        "timestep_start" not in meta
        for conditioning in local
        for _, meta in conditioning
    )


@pytest.mark.parametrize(
    "strength,cutoff", [(0, 0), (0, 0.5), (0, 1), (0.5, 0), (1, 0), (1, 1)]
)
def test_endpoint_optimizations_evaluate_only_selected_source(
    clip, predictions, strength, cutoff
):
    local = encode_tiles(
        clip,
        tiles(),
        strength=strength,
        end_percent=cutoff,
        baseline=["red", "bed", "red", ""],
    )
    reference_only = strength == cutoff == 1
    assert len(clip.encoded) == (4 if reference_only else 3)
    assert all(bool(tokens["images"]) == reference_only for tokens in clip.encoded)
    actual = predictions(local, batch=2, sigmas=(0.75, 0.5, 0.25))
    for sigma in (0.75, 0.5, 0.25):
        contexts = evaluated_contexts(actual, sigma, 0, 2)
        assert len(contexts) == 8
        assert all(
            any(torch.equal(context, entries[0][0][0]) for entries in local)
            for context in contexts
        )
        assert not evaluated_contexts(actual, sigma, 1, 2)
