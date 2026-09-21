# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Four native Krea2 image/text conditionings, with no persistent encode state.

Inspected capabilities, not a revision requirement: ComfyUI
c194dd00cd42aa18d9dbf27d977bf6b85d9ea565, comfy/text_encoders/krea2.py,
comfy/text_encoders/qwen3vl.py, comfy/sd.py and comfy/samplers.py.
Prefix removal, layer extraction and baked CLIP schedules remain host-owned.
"""

import math

import torch
import torch.nn.functional as F

VISION_TURN = "<|vision_start|><|image_pad|><|vision_end|>"


def _native_template(clip):
    # Krea2 is optional at pack load time, including on older ComfyUI hosts.
    try:
        from comfy.text_encoders.krea2 import (
            KREA2_TEMPLATE,
            Krea2TEModel,
            Krea2Tokenizer,
        )
    except ImportError as exc:
        raise ValueError("ComfyUI native Krea2 text encoder APIs are required") from exc
    if (
        not isinstance(getattr(clip, "cond_stage_model", None), Krea2TEModel)
        or not isinstance(getattr(clip, "tokenizer", None), Krea2Tokenizer)
        or not callable(getattr(clip, "tokenize", None))
        or not callable(getattr(clip, "encode_from_tokens_scheduled", None))
    ):
        raise ValueError(  # noqa: TRY004
            "clip must be a native Krea2 CLIP with image tokenization and scheduled "
            "encoding; load the text encoder with CLIPLoader type 'krea2'"
        )
    if not isinstance(KREA2_TEMPLATE, str) or KREA2_TEMPLATE.count("{}") != 1:
        raise ValueError("ComfyUI native Krea2 conditioning template is unavailable")
    return KREA2_TEMPLATE


def _strings(values, name):
    if values is None or (isinstance(values, list) and not values):
        return [""] * 4
    if (
        not isinstance(values, list)
        or len(values) != 4
        or any(not isinstance(value, str) for value in values)
    ):
        raise ValueError(
            f"{name} requires exactly four STRING execution-list items in "
            "TL/TR/BR/BL order, or an absent/empty list for four empty strings"
        )
    return values


def _validate_inputs(
    reference_tiles, prompts, strength, end_percent, downsize_to_1mp, baseline
):
    if (
        not isinstance(reference_tiles, torch.Tensor)
        or reference_tiles.is_nested
        or reference_tiles.layout != torch.strided
        or not reference_tiles.is_floating_point()
        or reference_tiles.ndim != 4
        or reference_tiles.shape[0] != 4
        or min(reference_tiles.shape[1:3]) < 1
        or reference_tiles.shape[-1] not in (3, 4)
    ):
        raise ValueError(
            "reference_tiles requires one floating RGB/RGBA IMAGE tensor with "
            "exactly four nonempty tiles in TL/TR/BR/BL order"
        )
    prompts = _strings(prompts, "prompts")
    baseline = _strings(baseline, "baseline")
    for name, value, upper in (
        ("strength", strength, 1),
        ("end_percent", end_percent, 1),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0 <= value <= upper
        ):
            raise ValueError(f"{name} must be finite and in 0…{upper}")
    if not isinstance(downsize_to_1mp, bool):
        raise ValueError("downsize_to_1mp must be a boolean")  # noqa: TRY004
    return prompts, baseline


def _image(tile, downsize_to_1mp):
    tile = tile[..., :3]
    height, width = tile.shape[1:3]
    if downsize_to_1mp and height * width > 1024**2:
        scale = math.sqrt(1024**2 / (height * width))
        # Keep pixels in floating point; native PIL Lanczos quantizes to uint8.
        # Antialiased interpolation needs at least float32 on supported hosts.
        dtype = torch.float64 if tile.dtype == torch.float64 else torch.float32
        tile = (
            F.interpolate(
                tile.movedim(-1, 1).to(dtype=dtype),
                size=(max(1, round(height * scale)), max(1, round(width * scale))),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .clamp(0, 1)
            .movedim(1, -1)
        )
    return tile


def _encode(clip, template, text, images):
    from comfy.model_management import throw_exception_if_processing_interrupted

    throw_exception_if_processing_interrupted()
    tokens = clip.tokenize(text, images=images, llama_template=template)
    conditioning = clip.encode_from_tokens_scheduled(tokens)
    throw_exception_if_processing_interrupted()
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError("Native Krea2 encoding must return nonempty CONDITIONING")
    for entry in conditioning:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not isinstance(entry[1], dict)
        ):
            raise ValueError(
                "Native Krea2 encoding must return [embedding, metadata] entries"
            )
        embedding = entry[0]
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.is_nested
            or embedding.layout != torch.strided
            or not embedding.is_floating_point()
            or embedding.ndim != 3
            or embedding.shape[0] != 1
            or embedding.shape[1] < 1
            or embedding.shape[2] != 30720
        ):
            raise ValueError(
                "Native Krea2 encoding requires floating [1, tokens, 30720] "
                "embeddings from the 12-layer Qwen3-VL encoder"
            )
    return conditioning


def _copy_conditioning(conditioning, weight=1.0, interval=None):
    result = []
    for embedding, metadata in conditioning:
        metadata = metadata.copy()
        if weight != 1:
            # The host weights separate predictions, then normalizes their sum.
            # Scaling embeddings or weighting a lone entry cannot form a blend.
            metadata["strength"] = metadata.get("strength", 1) * weight
        if interval is not None:
            # Native percent-to-sigma conversion intersects these values with
            # clip_start/end_percent and includes both endpoints at the cutoff.
            metadata["start_percent"] = max(
                metadata.get("start_percent", 0), interval[0]
            )
            metadata["end_percent"] = min(metadata.get("end_percent", 1), interval[1])
        result.append([embedding, metadata])
    return result


def encode_tiles(
    clip,
    reference_tiles,
    prompts=None,
    strength=1.0,
    end_percent=1.0,
    downsize_to_1mp=False,
    baseline=None,
):
    prompts, baseline = _validate_inputs(
        reference_tiles, prompts, strength, end_percent, downsize_to_1mp, baseline
    )
    template = _native_template(clip)
    baselines = {}

    def text_conditioning(text):
        # Invocation-local, keyed by literal text. Only tensors are shared;
        # every tile and interval receives fresh conditioning containers below.
        if text not in baselines:
            baselines[text] = _encode(clip, template, text, [])
        return baselines[text]

    if strength == 0 or end_percent == 0:
        return [_copy_conditioning(text_conditioning(text)) for text in baseline]

    result = []
    interval = (0, end_percent) if end_percent < 1 else None
    for index, prompt in enumerate(prompts):
        image = _image(reference_tiles[index : index + 1], downsize_to_1mp)
        conditioning = _encode(clip, template, VISION_TURN + prompt, [image])
        result.append(_copy_conditioning(conditioning, strength, interval))

    if strength < 1 or end_percent < 1:
        for conditioning, text in zip(result, baseline, strict=True):
            fallback = text_conditioning(text)
            if strength < 1:
                conditioning.extend(
                    _copy_conditioning(fallback, 1 - strength, interval)
                )
            if end_percent < 1:
                conditioning.extend(
                    _copy_conditioning(fallback, interval=(end_percent, 1))
                )
    return result
