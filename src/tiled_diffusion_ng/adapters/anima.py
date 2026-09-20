# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native Anima image semantics with aligned views and tile-local RoPE."""

import torch

from ..geometry import GeometrySignature, LatentSpec, validate_plan
from ._anima_sampling import AnimaSamplingContext, validate_control


class AnimaAdapter:
    def accepts(self, model):
        from comfy import latent_formats, model_base

        return type(getattr(model, "model", None)) is getattr(
            model_base, "Anima", None
        ) and type(model.get_model_object("latent_format")) is getattr(
            latent_formats, "Wan21", None
        )

    def describe(self, model, latent):
        from comfy.ldm.anima.model import Anima
        from comfy.ldm.cosmos.position_embedding import VideoRopePosition3DEmb

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
                "Anima requires nonempty floating B×16×H×W or B×16×1×H×W "
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
                "Anima noise_mask requires nonempty floating B×1×1×H×W masks "
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
            raise ValueError("Unsupported Anima Wan21 latent format metadata")
        for field, expected in (
            ("downscale_ratio_spacial", 8),
            ("downscale_ratio_temporal", 4),
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

        network = model.get_model_object("diffusion_model")
        # Detection reads depth from weights; no checkpoint name or block count
        # participates in support decisions for Base/Aesthetic/Turbo/2.9B.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_detection.py#L848-L881
        if (
            type(network) is not Anima
            or getattr(network, "in_channels", None) != 16
            or getattr(network, "out_channels", None) != 16
            or getattr(network, "patch_spatial", None) != 2
            or getattr(network, "patch_temporal", None) != 1
            or model.model.concat_keys
        ):
            raise ValueError("Unsupported native Anima architecture or patch geometry")
        # Native RoPE is generated from each view's shape. Keep its extrapolation
        # settings and native zero padding-mask channel; add no global offsets.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/ldm/cosmos/predict2.py#L739-L827
        if (
            getattr(network, "pos_emb_cls", None) != "rope3d"
            or type(getattr(network, "pos_embedder", None))
            is not VideoRopePosition3DEmb
            or getattr(network, "extra_per_block_abs_pos_emb", None) is not False
        ):
            raise ValueError(
                "Anima requires native tile-local RoPE without extra absolute positions"
            )
        if any(n % 2 for n in samples.shape[-2:]):
            raise ValueError(
                "Anima canvas dimensions must be divisible by 16 pixels "
                "(latent alignment 2); resizing and padding are unsupported"
            )
        # Both input layouts have one canonical evaluation signature. Only the
        # host introduces T=1, and its native 5D output remains unchanged.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/sample.py#L45-L71
        signature = GeometrySignature(
            "anima",
            1,
            "comfy.latent_formats.Wan21",
            layout="BCTHW",
            rank=5,
            channels=16,
            scale=(8, 8),
            alignment=(2, 2),
            minimum=(2, 2),
        )
        return LatentSpec(signature, tuple(samples.shape[-2:]))

    def validate_sampling(self, model, latent, plan):
        from comfy import model_sampling

        validate_plan(plan, self.describe(model, latent))
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
                "Unsupported Anima model_sampling conversion; expected native CONST"
            )

    def validate_model_options(self, options):
        transformer = options.get("transformer_options", {})
        attached = []
        for field in ("patches", "patches_replace"):
            groups = transformer.get(field, {})
            if not isinstance(groups, dict):
                raise ValueError(f"Unsupported Anima transformer {field}")  # noqa: TRY004
            for name, patches in groups.items():
                if field == "patches_replace" and isinstance(patches, dict):
                    patches = list(patches.values())
                if not isinstance(patches, (list, tuple)):
                    raise ValueError(  # noqa: TRY004
                        f"Unsupported Anima transformer {field}.{name}"
                    )
                attached.extend((f"{field}.{name}", patch) for patch in patches)
        # Recognize even partial native LLLite hook sets without importing the
        # optional implementation or creating any patch/attachment state.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy_extras/nodes_model_patch.py#L384-L425
        for _, patch in attached:
            if any(
                cls.__module__ == "comfy.ldm.anima.lllite"
                and cls.__name__
                in {
                    "AnimaLLLitePatch",
                    "AnimaLLLiteAttentionPatch",
                    "AnimaLLLiteMLPPatch",
                }
                for cls in type(patch).__mro__
            ):
                raise NotImplementedError(
                    "Native AnimaLLLite attachments require tiled reference routing; "
                    "tiled LLLite integration is deferred"
                )
        if attached:
            raise ValueError(f"Unsupported Anima transformer {attached[0][0]}")

    def validate_condition(self, embedding, metadata):
        validate_control(metadata)
        if (
            not isinstance(embedding, torch.Tensor)
            or embedding.is_nested
            or embedding.layout != torch.strided
            or embedding.ndim != 3
            or embedding.numel() == 0
            or not embedding.is_floating_point()
        ):
            raise ValueError(
                "Anima CONDITIONING requires a nonempty floating rank-3 embedding tensor"
            )
        # Text preprocessing, token weights and optional metadata stay native.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_base.py#L1482-L1505
        known = {
            "t5xxl_ids",
            "t5xxl_weights",
            "attention_mask",
            "pooled_output",
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
                raise ValueError(f"Unsupported Anima conditioning field: {key}")

    def validate_evaluation(self, samples, plan):
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
            raise ValueError("Anima model evaluation canvas does not match TILE_PLAN")

    def create_sampling_context(self, plan):
        return AnimaSamplingContext(plan)

    def adapt_spatial_condition(self, condition, region):
        validate_control(condition)
        for key in ("area", "mask", "gligen", "hooks"):
            if condition.get(key) is not None:
                raise ValueError(
                    f"Unsupported prepared Anima conditioning field: {key}"
                )
        for key in condition.get("model_conds", {}):
            if key not in {
                "c_crossattn",
                "t5xxl_ids",
                "t5xxl_weights",
                "attention_mask",
            }:
                raise ValueError(f"Unsupported prepared Anima model_conds field: {key}")
        return condition.copy()
