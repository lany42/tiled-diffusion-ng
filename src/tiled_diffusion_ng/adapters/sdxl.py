# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""SDXL image geometry, conditioning, and prediction capabilities."""

import torch

from ..geometry import GeometrySignature, LatentSpec, validate_plan


class SDXLAdapter:
    def accepts(self, model):
        from comfy import latent_formats, model_base

        return (
            type(getattr(model, "model", None)) is model_base.SDXL
            and type(model.get_model_object("latent_format")) is latent_formats.SDXL
        )

    def describe(self, model, latent):
        if not isinstance(latent, dict):
            raise TypeError("LATENT must be a dictionary containing samples")
        samples = latent.get("samples")
        if (
            not isinstance(samples, torch.Tensor)
            or samples.is_nested
            or samples.layout != torch.strided
            or samples.ndim != 4
            or samples.shape[1] != 4
            or any(n < 1 for n in samples.shape)
            or not samples.is_floating_point()
        ):
            raise ValueError(
                "SDXL requires nonempty floating BCHW samples with 4 channels"
            )
        fmt = model.get_model_object("latent_format")
        # Host latent metadata uses the spelling 'spacial'. Read it, do not
        # infer scale from channel count or scan pixel/weight values.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/latent_formats.py#L6-L50
        if (
            fmt.latent_channels,
            fmt.latent_dimensions,
            fmt.spacial_downscale_ratio,
            fmt.temporal_downscale_ratio,
        ) != (4, 2, 8, 1):
            raise ValueError("Unsupported SDXL latent format metadata")
        for field, expected in (
            ("downscale_ratio_spacial", 8),
            ("downscale_ratio_temporal", 1),
        ):
            if field in latent and latent[field] != expected:
                raise ValueError(
                    f"Unsupported LATENT {field}; convert it before planning"
                )
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
        config = model.model.model_config.unet_config
        if (
            config.get("in_channels") != 4
            or config.get("out_channels") != 4
            or model.model.concat_keys
        ):
            raise ValueError("SDXL concat/inpaint model inputs are not supported")
        # Padded stride-2 convolution accepts one cell; upsampling uses each
        # skip's explicit target shape, so origins/extents have a one-cell grid.
        # This is an architectural minimum, not a useful image-quality size.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/ldm/modules/diffusionmodules/openaimodel.py#L92-L145
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/ldm/modules/diffusionmodules/openaimodel.py#L919-L922
        signature = GeometrySignature(
            "sdxl",
            1,
            "comfy.latent_formats.SDXL",
            scale=(fmt.spacial_downscale_ratio,) * 2,
        )
        return LatentSpec(signature, tuple(samples.shape[-2:]))

    def validate_sampling(self, model, latent, plan):
        from comfy import model_sampling

        validate_plan(plan, self.describe(model, latent))
        sampling = model.get_model_object("model_sampling")
        # These two inspected affine conversions commute with normalized fusion:
        # F(a*x_i+b*epsilon_i)=a*x+b*F(epsilon). Host outputs denoised, not epsilon.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L208-L257
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_sampling.py#L30-L55
        if (
            getattr(sampling.calculate_denoised, "__func__", None)
            not in (
                model_sampling.EPS.calculate_denoised,
                model_sampling.V_PREDICTION.calculate_denoised,
            )
            or getattr(sampling.calculate_input, "__func__", None)
            is not model_sampling.EPS.calculate_input
        ):
            raise ValueError(
                "Unsupported SDXL model_sampling conversion; expected native EPS or V_PREDICTION"
            )

    def validate_condition(self, metadata):
        known = {
            "pooled_output",
            "cross_attn_controlnet",
            "pooled_output_controlnet",
            "width",
            "height",
            "crop_w",
            "crop_h",
            "target_width",
            "target_height",
            "aesthetic_score",
            "noise_augmentation",
            "strength",
            "start_percent",
            "end_percent",
            "timestep_start",
            "timestep_end",
            "control",
            "control_apply_to_uncond",
        }
        for key in metadata:
            if key not in known:
                raise ValueError(f"Unsupported SDXL conditioning field: {key}")

    def adapt_spatial_condition(self, condition, region):
        # SDXL text/y conditions have no spatial crop; explicit size values and
        # missing-size defaults were encoded on the FULL canvas by extra_conds.
        # Only our ControlNet clones carry spatial data, already pixel-cropped.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py#L945-L964
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L515-L538
        for key in condition.get("model_conds", {}):
            if key not in {"c_crossattn", "y", "crossattn_controlnet"}:
                raise ValueError(f"Unsupported prepared SDXL model_conds field: {key}")
        return condition.copy()
