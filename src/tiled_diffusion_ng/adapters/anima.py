# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native Anima image semantics with aligned views and tile-local RoPE."""

import torch

from ..geometry import validate_plan
from ._anima_lllite import validate_hook_options
from ._anima_sampling import AnimaSamplingContext, validate_control
from ._wan_image import (
    image_spec,
    validate_const,
    validate_evaluation,
    validate_target,
)


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

        samples = validate_target(model, latent, "Anima")

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
        return image_spec(samples, "anima", 1)

    def validate_sampling(self, model, latent, plan):
        validate_plan(plan, self.describe(model, latent))
        validate_const(model, "Anima")

    def validate_model_options(self, options):
        validate_hook_options(options)

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
        validate_evaluation(samples, plan, "Anima")

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
