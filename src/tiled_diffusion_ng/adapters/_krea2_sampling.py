# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Krea2 preparation metadata; reference tensors stay in native conditioning.

Inspected ComfyUI c194dd00cd42aa18d9dbf27d977bf6b85d9ea565 contracts:
comfy/model_base.py#L2724 (normalization and containers), comfy/conds.py
(batch repetition/concatenation), comfy/ldm/krea2/model.py#L283 (whole-image
positions, reference padding and zero timestep), comfy_extras/nodes_flux.py#L153
(Edit Model Reference Method). CPU doubles cannot establish host compatibility.
"""

from contextlib import contextmanager
from dataclasses import dataclass

import torch

METHODS = ("index", "index_timestep_zero")
PREPARATION = "tdng_krea2_preparation"


def validate_control(condition):
    if condition.get("control") is not None:
        raise ValueError(
            "Krea2 conditioning control is unsupported; ControlNet residuals "
            "are incompatible with the native Krea2 transformer"
        )


def validate_embedding(embedding, width=None):
    if (
        not isinstance(embedding, torch.Tensor)
        or embedding.is_nested
        or embedding.layout != torch.strided
        or embedding.ndim != 3
        or embedding.numel() == 0
        or not embedding.is_floating_point()
    ):
        raise ValueError(
            "Krea2 CONDITIONING requires a nonempty floating rank-3 embedding tensor"
        )
    if width is not None and embedding.shape[-1] != width:
        raise ValueError(
            f"Krea2 CONDITIONING requires {width} features from the native "
            f"encoder; got {embedding.shape[-1]}. Use CLIPLoader type 'krea2'."
        )


def context_width(network):
    dimensions = [getattr(network, key, None) for key in ("txtlayers", "txtdim")]
    if any(type(n) is not int or n < 1 for n in dimensions):
        raise ValueError("Unsupported native Krea2 text dimensions")
    return dimensions[0] * dimensions[1]


def validate_references(references):
    if references is None:
        return ()
    if not isinstance(references, (list, tuple)):
        raise ValueError("Krea2 reference_latents must be a list or tuple")  # noqa: TRY004
    for reference in references:
        # Wan21 normalization has 5D per-channel statistics. A 4D reference can
        # silently broadcast its batch/channel axes into an invalid video.
        if (
            not isinstance(reference, torch.Tensor)
            or reference.is_nested
            or reference.layout != torch.strided
            or reference.ndim != 5
            or reference.shape[1:3] != (16, 1)
            or any(n < 1 for n in reference.shape)
            or not reference.is_floating_point()
        ):
            raise ValueError(
                "Krea2 reference_latents requires nonempty floating B×16×1×H×W "
                "tensors; 4D references and video are unsupported"
            )
    return references


def validate_method(method):
    if not isinstance(method, str) or method not in METHODS:
        raise ValueError(
            "Krea2 references require 'index' or 'index_timestep_zero'; use "
            "upstream Edit Model Reference Method to set reference_latents_method"
        )


def validate_condition(embedding, metadata):
    validate_control(metadata)
    validate_embedding(embedding)
    known = {
        "pooled_output",
        "attention_mask",
        "strength",
        "start_percent",
        "end_percent",
        # Baked CLIP schedules are nonspatial; the host intersects these with
        # start/end_percent. Runtime conditioning hooks remain unsupported.
        # https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/comfy/sd.py#L339-L400
        "clip_start_percent",
        "clip_end_percent",
        "timestep_start",
        "timestep_end",
        "reference_latents",
        "reference_latents_method",
        "control",
        "control_apply_to_uncond",
    }
    for key in metadata:
        if key not in known:
            raise ValueError(f"Unsupported Krea2 conditioning field: {key}")
    validate_references(metadata.get("reference_latents"))
    if metadata.get("reference_latents_method") is not None:
        validate_method(metadata["reference_latents_method"])


def _is_lllite(value):
    # Recognition must not require optional Anima LLLite host APIs.
    return any(
        (cls.__module__ == "comfy.ldm.anima.lllite" and "LLLite" in cls.__name__)
        or (
            cls.__module__ == f"{__package__}._anima_lllite"
            and cls.__name__ in {"TiledLLLiteDispatcher", "LLLiteAttachment"}
        )
        for cls in type(value).__mro__
    )


def _reject_lllite():
    raise NotImplementedError("Krea2 LLLite support remains TBD")


def _hooks(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _hooks(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _hooks(item)
    else:
        yield value


def validate_hook_options(options):
    transformer = options.get("transformer_options", {})
    for field in ("patches", "patches_replace"):
        groups = transformer.get(field, {})
        if not isinstance(groups, dict):
            raise ValueError(f"Unsupported Krea2 transformer {field}")  # noqa: TRY004
        hooks = list(_hooks(groups))
        if any(_is_lllite(hook) for hook in hooks):
            _reject_lllite()
        if hooks:
            raise ValueError(f"Unsupported Krea2 transformer {field}")


def validate_attachments(model):
    for key, attachment in getattr(model, "attachments", {}).items():
        if attachment is not None and (
            key == "tiled_diffusion_ng.anima_lllite.v1" or _is_lllite(attachment)
        ):
            _reject_lllite()


@dataclass(frozen=True)
class Preparation:
    width: int
    reference_shapes: tuple[tuple[int, ...], ...]
    method: str | None


def adapt_condition(condition):
    from comfy.conds import CONDConstant, CONDList, CONDRegular

    validate_control(condition)
    for key in (
        "area",
        "mask",
        "gligen",
        "hooks",
        "concat_latent_image",
        "concat_mask",
        "mask_strength",
        "set_area_to_bounds",
    ):
        if condition.get(key) is not None:
            raise ValueError(f"Unsupported prepared Krea2 conditioning field: {key}")
    prepared = condition.get("model_conds", {})
    for key in prepared:
        if key not in {"c_crossattn", "ref_latents", "ref_latents_method"}:
            raise ValueError(f"Unsupported prepared Krea2 model_conds field: {key}")
    contract = condition.get(PREPARATION)
    if type(contract) is not Preparation:
        raise ValueError("Missing Krea2 conditioning preparation metadata")
    context = prepared.get("c_crossattn")
    if type(context) is not CONDRegular:
        raise ValueError("Krea2 requires native prepared c_crossattn CONDRegular")
    validate_embedding(context.cond, contract.width)
    references = prepared.get("ref_latents")
    if references is not None and type(references) is not CONDList:
        raise ValueError("Krea2 requires native prepared ref_latents CONDList")
    refs = validate_references(None if references is None else references.cond)
    if tuple(tuple(ref.shape) for ref in refs) != contract.reference_shapes:
        raise ValueError(
            "Krea2 reference_latents disappeared or changed during preparation"
        )
    method = prepared.get("ref_latents_method")
    if method is not None:
        if type(method) is not CONDConstant:
            raise ValueError(
                "Krea2 requires native prepared ref_latents_method CONDConstant"
            )
        validate_method(method.cond)
    if refs and (method is None or method.cond != contract.method):
        raise ValueError(
            "Krea2 reference_latents_method disappeared or changed during preparation"
        )
    # Shallow copies preserve every native container and normalized reference.
    return condition.copy()


class Krea2SamplingContext:
    def __init__(self, plan):
        self.plan = plan
        self.width = None
        self.default_method = None

    def _require_open(self):
        if self.plan is None:
            raise RuntimeError("Krea2 sampling invocation has already closed")

    def prepare_model(self, model, latent):
        self._require_open()
        if self.width is not None:
            raise RuntimeError("Krea2 sampling invocation is already prepared")
        validate_hook_options(model.model_options)
        validate_attachments(model)
        self.width = context_width(model.get_model_object("diffusion_model"))
        # get_model_object also resolves ModelPatcher object patches before load.
        try:
            self.default_method = model.get_model_object(
                "diffusion_model.default_ref_method"
            )
        except AttributeError:
            self.default_method = None

    @contextmanager
    def tile_options(self, options, region):
        self._require_open()
        validate_hook_options(options)
        yield options

    def prepare_pair(self, positive, negative, region):
        self._require_open()
        if self.width is None:
            raise RuntimeError("Krea2 sampling invocation is unprepared")
        for branch in (positive, negative):
            for entry in branch:
                validate_control(entry)
                validate_embedding(entry["tdng_embedding"], self.width)
                references = validate_references(entry.get("reference_latents"))
                method = entry.get("reference_latents_method")
                if references:
                    if method is None:
                        method = self.default_method
                    validate_method(method)
                    entry["reference_latents_method"] = method
                # Only invocation-owned dictionaries reach this method.
                entry.pop("control", None)
                entry[PREPARATION] = Preparation(
                    self.width, tuple(tuple(ref.shape) for ref in references), method
                )
        return positive, negative

    def finalize_preparation(self):
        self._require_open()
        if self.width is None:
            raise RuntimeError("Krea2 sampling invocation is unprepared")

    def close(self):
        self.plan = None
        self.width = None
        self.default_method = None
