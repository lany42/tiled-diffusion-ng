# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Native Krea2 Raw/Turbo image geometry and whole-reference conditioning.

Inspected capabilities, not a runtime revision requirement: ComfyUI
c194dd00cd42aa18d9dbf27d977bf6b85d9ea565, comfy/model_base.py (Krea2),
comfy/ldm/krea2/model.py (SingleStreamDiT), comfy/supported_models.py (Krea2).
Raw and Turbo share these boundaries; checkpoint names and depth are irrelevant.
Real-host inference and visual acceptance remain separate from CPU contracts.
"""

from ..geometry import validate_plan
from ._krea2_sampling import (
    Krea2SamplingContext,
    adapt_condition,
    context_width,
    validate_condition,
    validate_hook_options,
)
from ._wan_image import (
    image_spec,
    validate_const,
    validate_evaluation,
    validate_target,
)


class Krea2Adapter:
    def accepts(self, model):
        from comfy import latent_formats, model_base

        return type(getattr(model, "model", None)) is getattr(
            model_base, "Krea2", None
        ) and type(model.get_model_object("latent_format")) is getattr(
            latent_formats, "Wan21", None
        )

    def describe(self, model, latent):
        # Older hosts must still be able to import the registry and use SDXL or
        # Anima. Only a Krea2 invocation requires these optional native APIs.
        try:
            from comfy.ldm.flux.layers import EmbedND
            from comfy.ldm.krea2.model import SingleStreamDiT
        except ImportError as exc:
            raise ValueError("ComfyUI native Krea2 APIs are required") from exc

        samples = validate_target(model, latent, "Krea2")
        network = model.get_model_object("diffusion_model")
        if (
            type(network) is not SingleStreamDiT
            or getattr(network, "channels", None) != 16
            or getattr(network, "patch", None) != 2
            or model.model.concat_keys
        ):
            raise ValueError("Unsupported native Krea2 architecture or patch geometry")
        if type(getattr(network, "pe_embedder", None)) is not EmbedND:
            raise ValueError("Krea2 requires native tile-local EmbedND positions")
        context_width(network)
        # Each target view and whole reference gets native per-image positions;
        # do not introduce full-canvas offsets or crop reference latents.
        # https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/comfy/ldm/krea2/model.py#L283-L403
        return image_spec(samples, "Krea2", "krea2", 1)

    def validate_sampling(self, model, latent, plan):
        validate_plan(plan, self.describe(model, latent))
        # Retain the existing FLUX sampling object, including canvas-based shift.
        validate_const(model, "Krea2")

    def validate_model_options(self, options):
        validate_hook_options(options)

    def validate_condition(self, embedding, metadata):
        validate_condition(embedding, metadata)

    def validate_evaluation(self, samples, plan):
        validate_evaluation(samples, plan, "Krea2")

    def adapt_spatial_condition(self, condition, region):
        return adapt_condition(condition)

    def create_sampling_context(self, plan):
        return Krea2SamplingContext(plan)
