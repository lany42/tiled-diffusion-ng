# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Wan21 image validation shared without sharing family conditioning state."""

import torch

from ..geometry import GeometrySignature, LatentSpec


def validate_target(model, latent, family):
    if not isinstance(latent, dict):
        raise TypeError("LATENT must be a dictionary containing samples")
    samples = latent.get("samples")
    if (
        not isinstance(samples, torch.Tensor)
        or samples.is_nested
        or samples.layout != torch.strided
        or samples.ndim not in (4, 5)
        or samples.shape[1] != 16
        or (samples.ndim == 5 and samples.shape[2] != 1)
        or any(n < 1 for n in samples.shape)
        or not samples.is_floating_point()
    ):
        raise ValueError(
            f"{family} requires nonempty floating B×16×H×W or B×16×1×H×W "
            "samples; video is unsupported"
        )
    # The host treats sub-5D masks as time, mixing images before broadcasting.
    # Require explicit batch/channel/time axes; spatial resizing stays native.
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/utils.py#L1350-L1369
    mask = latent.get("noise_mask")
    if mask is not None and (
        not isinstance(mask, torch.Tensor)
        or mask.is_nested
        or mask.layout != torch.strided
        or mask.ndim != 5
        or mask.shape[0] not in (1, samples.shape[0])
        or mask.shape[1:3] != (1, 1)
        or any(n < 1 for n in mask.shape)
        or not mask.is_floating_point()
    ):
        raise ValueError(
            f"{family} noise_mask requires nonempty floating B×1×1×H×W masks "
            "with B=1 or the samples batch size"
        )
    fmt = model.get_model_object("latent_format")
    # Wan21 inherits spatial scale 8 and declares temporal scale 4.
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/latent_formats.py#L729-L732
    if tuple(
        getattr(fmt, field, None)
        for field in (
            "latent_channels",
            "latent_dimensions",
            "spacial_downscale_ratio",
            "temporal_downscale_ratio",
        )
    ) != (16, 3, 8, 4):
        raise ValueError(f"Unsupported {family} Wan21 latent format metadata")
    for field, expected in (
        ("downscale_ratio_spacial", 8),
        ("downscale_ratio_temporal", 4),
    ):
        if field in latent and latent[field] != expected:
            raise ValueError(f"Unsupported LATENT {field}; convert it before planning")
    for key, value in latent.items():
        if key in {
            "samples",
            "noise_mask",
            "batch_index",
            "downscale_ratio_spacial",
            "downscale_ratio_temporal",
        }:
            continue
        if key == "type" or isinstance(value, (torch.Tensor, dict, list, tuple)):
            raise ValueError(f"Unsupported LATENT metadata field: {key}")
    return samples


def image_spec(samples, family, adapter_id, adapter_version):
    if any(n % 2 for n in samples.shape[-2:]):
        raise ValueError(
            f"{family} canvas dimensions must be divisible by 16 pixels "
            "(latent alignment 2); resizing and padding are unsupported"
        )
    # Both input layouts have one canonical evaluation signature. Only the
    # host introduces T=1, and its native 5D output remains unchanged.
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/sample.py#L45-L71
    signature = GeometrySignature(
        adapter_id,
        adapter_version,
        "comfy.latent_formats.Wan21",
        layout="BCTHW",
        rank=5,
        channels=16,
        scale=(8, 8),
        alignment=(2, 2),
        minimum=(2, 2),
    )
    return LatentSpec(signature, tuple(samples.shape[-2:]))


def validate_const(model, family):
    from comfy import model_sampling

    sampling = model.get_model_object("model_sampling")
    # The host owns input and denoised conversion. For shared x/sigma,
    # normalized fusion commutes with CONST: F(x_i-sigma*v_i)=x-sigma*F(v_i).
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_sampling.py#L86-L92
    if (
        getattr(getattr(sampling, "calculate_input", None), "__func__", None)
        is not model_sampling.CONST.calculate_input
        or getattr(getattr(sampling, "calculate_denoised", None), "__func__", None)
        is not model_sampling.CONST.calculate_denoised
    ):
        raise ValueError(
            f"Unsupported {family} model_sampling conversion; expected native CONST"
        )


def validate_evaluation(samples, plan, family):
    if (
        not isinstance(samples, torch.Tensor)
        or samples.is_nested
        or samples.layout != torch.strided
        or samples.ndim != 5
        or samples.shape[1:3] != (16, 1)
        or samples.shape[0] < 1
        or not samples.is_floating_point()
        or tuple(samples.shape[-2:]) != plan.latent_hw
    ):
        raise ValueError(f"{family} model evaluation canvas does not match TILE_PLAN")
