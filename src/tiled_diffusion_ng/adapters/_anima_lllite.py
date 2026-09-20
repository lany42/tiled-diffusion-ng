# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Stable native hooks with invocation-owned, overlap-inclusive RGB references.

Inspected contracts (not an exact-version requirement): ComfyUI
944386c233e02eaf877b1c8d5d513fb3d3a4d5e3, comfy/ldm/anima/lllite.py,
comfy/ldm/cosmos/predict2.py and comfy_extras/nodes_model_patch.py.
Preparation, encoding, activation and attention/MLP math remain host-owned.
"""

import math
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType

import torch

from ..geometry import TilePlanData, validate_plan

ATTACHMENT_KEY = "tiled_diffusion_ng.anima_lllite.v1"
DISPATCH_KEY = "tiled_diffusion_ng.anima_lllite.dispatch.v1"
SELF_TARGETS = MappingProxyType(
    {"q": "self_attn_q_proj", "k": "self_attn_k_proj", "v": "self_attn_v_proj"}
)
CROSS_TARGETS = MappingProxyType({"q": "cross_attn_q_proj"})


def _native():
    try:
        from comfy.ldm.anima import lllite

        for name in (
            "AnimaLLLite",
            "AnimaLLLitePatch",
            "AnimaLLLiteAttentionPatch",
            "AnimaLLLiteMLPPatch",
            "MODULE_PATTERN",
        ):
            getattr(lllite, name)
        return lllite
    except (ImportError, AttributeError) as exc:
        raise ValueError("ComfyUI native Anima LLLite APIs are required") from exc


def validate_references(references, plan, batch=None):
    if (
        not isinstance(references, torch.Tensor)
        or references.is_nested
        or references.layout != torch.strided
        or references.ndim != 4
        or not references.is_floating_point()
        or references.shape[-1] not in (3, 4)
        or references.shape[0] < 4
        or references.shape[0] % 4
    ):
        raise ValueError(
            "reference_tiles requires floating RGB/RGBA IMAGE batches with exactly "
            "four crops per source image in image-major TL/TR/BR/BL order"
        )
    if tuple(references.shape[1:3]) != plan.tile_pixel_hw:
        raise ValueError(
            "reference_tiles dimensions must match the TILE_PLAN's "
            f"overlap-inclusive crops {plan.tile_pixel_hw}"
        )
    groups = references.shape[0] // 4
    if batch is not None and groups not in (1, batch):
        raise ValueError(
            "reference_tiles source batch must equal the latent batch, "
            "or contain one reference group for broadcast"
        )
    return groups


def _validate_settings(strength, start_percent, end_percent):
    for name, value, lower, upper in (
        ("strength", strength, -10, 10),
        ("start_percent", start_percent, 0, 1),
        ("end_percent", end_percent, 0, 1),
    ):
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not (lower <= value <= upper)
        ):
            raise ValueError(f"LLLite {name} must be finite and in {lower}…{upper}")


def validate_weights(model, model_patch):
    from comfy.model_patcher import ModelPatcher

    native = _native()
    weights = getattr(model_patch, "model", None)
    if (
        not isinstance(model_patch, ModelPatcher)
        or type(weights) is not native.AnimaLLLite
    ):
        raise ValueError("MODEL_PATCH must contain native-loaded Anima LLLite weights")
    if weights.cond_in_channels != 3:
        raise ValueError(
            "Tiled Anima LLLite requires RGB weights; masks are unsupported"
        )
    backbone = model.get_model_object("diffusion_model")
    width = getattr(backbone, "model_channels", None)
    blocks = getattr(backbone, "blocks", ())
    if not isinstance(width, int) or width < 1 or weights.model_dim != width:
        raise ValueError("Anima LLLite model width does not match the native backbone")
    names = weights.module_names
    if not isinstance(names, (set, frozenset)) or not names or not len(blocks):
        raise ValueError("Anima LLLite requires nonempty module target coverage")
    indices = []
    for name in names:
        match = native.MODULE_PATTERN.fullmatch(name) if isinstance(name, str) else None
        if (
            match is None
            or int(match[1]) >= len(blocks)
            or name != f"lllite_dit_blocks_{int(match[1])}_{match[2]}"
        ):
            raise ValueError(
                f"Invalid Anima LLLite module target for backbone depth: {name}"
            )
        try:
            module = weights.get_submodule(name)
            if module.down.in_features != width or module.up.out_features != width:
                raise ValueError(f"Anima LLLite module width mismatch: {name}")
        except AttributeError as exc:
            raise ValueError(f"Missing Anima LLLite module target: {name}") from exc
        indices.append(int(match[1]))
    # Sparse checkpoints are supported: native apply skips missing targets.
    if weights.block_count != max(indices) + 1:
        raise ValueError("Anima LLLite block_count disagrees with its module targets")


@dataclass(frozen=True, eq=False)
class LLLiteConfig:
    plan: TilePlanData
    reference_tiles: torch.Tensor
    model_patch: object
    strength: float
    start_percent: float
    end_percent: float


@dataclass(frozen=True, eq=False)
class TiledLLLiteDispatcher:
    """Graph-owned hook: immutable configuration, no active tile or embeddings."""

    config: LLLiteConfig

    @property
    def model_patch(self):
        return self.config.model_patch

    @property
    def strength(self):
        return self.config.strength

    def models(self):
        # ModelPatcher discovers this before sampling/loading and owns offload.
        return [self.model_patch]

    def to(self, device_or_dtype):
        return self

    def __call__(self, args):
        handle = args.get("transformer_options", {}).get(DISPATCH_KEY)
        if type(handle) is not TileDispatch or handle.invocation is None:
            raise ValueError(
                "TiledAnimaLLLiteApply requires TileSampler with the same TILE_PLAN; "
                "missing active tiled dispatch (ordinary KSampler is unsupported)"
            )
        return handle.invocation.dispatch(self, handle, args)


@dataclass(frozen=True, eq=False)
class LLLiteAttachment:
    config: LLLiteConfig
    post_input: TiledLLLiteDispatcher
    attn1: object
    attn2: object
    mlp: object
    schema_version: int = 1
    kind: str = "anima_lllite"

    @property
    def hooks(self):
        return (
            ("post_input", self.post_input),
            ("attn1_patch", self.attn1),
            ("attn2_patch", self.attn2),
            ("mlp_patch", self.mlp),
        )


def get_attachment(model):
    attachments = getattr(model, "attachments", {})
    keys = [
        key for key in attachments if key.startswith("tiled_diffusion_ng.anima_lllite")
    ]
    if not keys:
        return None
    if keys != [ATTACHMENT_KEY]:
        raise ValueError("Unsupported or duplicate tiled Anima LLLite declarations")
    attachment = attachments[ATTACHMENT_KEY]
    if (
        type(attachment) is not LLLiteAttachment
        or attachment.schema_version != 1
        or attachment.kind != "anima_lllite"
        or type(attachment.config) is not LLLiteConfig
        or type(attachment.post_input) is not TiledLLLiteDispatcher
        or attachment.post_input.config is not attachment.config
    ):
        raise ValueError("Incomplete or unsupported tiled Anima LLLite declaration")
    return attachment


def _installed(options):
    transformer = options.get("transformer_options", {})
    attached = []
    for field_name in ("patches", "patches_replace"):
        groups = transformer.get(field_name, {})
        if not isinstance(groups, dict):
            raise ValueError(f"Unsupported Anima transformer {field_name}")  # noqa: TRY004
        for name, patches in groups.items():
            if field_name == "patches_replace" and isinstance(patches, dict):
                patches = list(patches.values())
            if not isinstance(patches, (list, tuple)):
                raise ValueError(f"Unsupported Anima transformer {field_name}.{name}")  # noqa: TRY004
            attached.extend((field_name, name, patch) for patch in patches)
    return attached


def validate_hook_options(options, attachment=None):
    attached = _installed(options)
    dispatchers = [
        patch for _, _, patch in attached if type(patch) is TiledLLLiteDispatcher
    ]
    expected = {} if attachment is None else dict(attachment.hooks)
    for field_name, slot, hook in attached:
        # Recognize native hooks without importing optional APIs on bare Anima.
        native = any(
            cls.__module__ == "comfy.ldm.anima.lllite"
            and cls.__name__
            in {"AnimaLLLitePatch", "AnimaLLLiteAttentionPatch", "AnimaLLLiteMLPPatch"}
            for cls in type(hook).__mro__
        )
        if native and type(getattr(hook, "patch", None)) is not TiledLLLiteDispatcher:
            raise NotImplementedError(
                "Native AnimaLLLiteApply chaining is unsupported; use "
                "TiledAnimaLLLiteApply for tiled reference routing. Native chaining is deferred"
            )
        if field_name != "patches" or (
            not native and type(hook) is not TiledLLLiteDispatcher
        ):
            raise ValueError(f"Unsupported Anima transformer {field_name}.{slot}")
        if attachment is not None and expected.get(slot) is not hook:
            raise ValueError("Live options replaced or added tiled LLLite hooks")
    if not attached and attachment is None:
        return None
    if len(dispatchers) != 1 or len(attached) != 4:
        raise ValueError("Incomplete or duplicate tiled LLLite hooks")
    slots = {slot: hook for _, slot, hook in attached}
    dispatcher = dispatchers[0]
    native = _native()
    if (
        slots.get("post_input") is not dispatcher
        or type(slots.get("attn1_patch")) is not native.AnimaLLLiteAttentionPatch
        or type(slots.get("attn2_patch")) is not native.AnimaLLLiteAttentionPatch
        or type(slots.get("mlp_patch")) is not native.AnimaLLLiteMLPPatch
    ):
        raise ValueError("Incomplete or mismatched tiled LLLite hook slots")
    if (
        any(
            slots[name].patch is not dispatcher
            for name in ("attn1_patch", "attn2_patch", "mlp_patch")
        )
        or slots["attn1_patch"].targets != SELF_TARGETS
        or slots["attn2_patch"].targets != CROSS_TARGETS
    ):
        raise ValueError("Mismatched tiled LLLite hook identities or targets")
    return dispatcher


def apply_lllite(
    model,
    model_patch,
    plan,
    references,
    strength=1.0,
    start_percent=0.0,
    end_percent=1.0,
):
    from .._comfy_sampling import validate_options
    from .anima import AnimaAdapter

    # Check both routes even at strength zero and through ModelPatcher clones.
    if get_attachment(model) is not None:
        raise ValueError(
            "Tiled Anima LLLite is already applied; duplicate application is unsupported"
        )
    if validate_hook_options(model.model_options) is not None:
        raise ValueError(
            "Orphaned tiled LLLite hooks; duplicate application is unsupported"
        )
    validate_options(model.model_options)
    validate_plan(plan)
    adapter = AnimaAdapter()
    if not adapter.accepts(model):
        raise ValueError("TiledAnimaLLLiteApply requires a native Anima MODEL")
    # Describe metadata without allocating a full latent or reading any weights.
    metadata = {"samples": torch.empty(1, 16, *plan.latent_hw, device="meta")}
    validate_plan(plan, adapter.describe(model, metadata))
    validate_weights(model, model_patch)
    validate_references(references, plan)
    _validate_settings(strength, start_percent, end_percent)
    if not callable(getattr(model, "set_attachments", None)):
        raise ValueError("ComfyUI ModelPatcher attachments API is required")  # noqa: TRY004
    native = _native()
    config = LLLiteConfig(
        plan, references, model_patch, strength, start_percent, end_percent
    )
    dispatcher = TiledLLLiteDispatcher(config)
    attachment = LLLiteAttachment(
        config,
        dispatcher,
        native.AnimaLLLiteAttentionPatch(dispatcher, SELF_TARGETS),
        native.AnimaLLLiteAttentionPatch(dispatcher, CROSS_TARGETS),
        native.AnimaLLLiteMLPPatch(dispatcher),
    )
    clone = model.clone()
    for slot, hook in attachment.hooks:
        clone.set_model_patch(hook, slot)
    clone.set_attachments(ATTACHMENT_KEY, attachment)
    validate_hook_options(clone.model_options, attachment)
    return clone


@dataclass(eq=False)
class TileDispatch:
    invocation: object
    region: object

    def guard_forward(self, executor, *args, **kwargs):
        transformer = kwargs.get("transformer_options", {})
        if self.invocation is None or transformer.get(DISPATCH_KEY) is not self:
            raise ValueError("LLLite forward requires active matching tiled dispatch")
        self.invocation.validate_live({"transformer_options": transformer})
        return executor(*args, **kwargs)

    def close(self):
        self.invocation = None
        self.region = None


class LLLiteInvocation:
    def __init__(self, plan):
        self.plan = plan
        self.model = None
        self.attachment = None
        self.delegates = []
        self.batch = None
        self.handles = set()

    def prepare_model(self, model, latent):
        if self.plan is None or self.model is not None:
            raise RuntimeError("LLLite invocation is closed or already prepared")
        self.model = model
        self.batch = latent["samples"].shape[0]
        self.attachment = get_attachment(model)
        self.validate_live(model.model_options)
        if self.attachment is None:
            return
        config = self.attachment.config
        validate_plan(config.plan)
        if config.plan != self.plan:
            raise ValueError(
                "Tiled Anima LLLite requires the same complete TILE_PLAN at sampling"
            )
        validate_references(config.reference_tiles, self.plan, self.batch)
        validate_weights(model, config.model_patch)
        _validate_settings(config.strength, config.start_percent, config.end_percent)
        sampling = model.get_model_object("model_sampling")
        if not callable(getattr(sampling, "percent_to_sigma", None)):
            raise ValueError("Native Anima LLLite requires percent_to_sigma")  # noqa: TRY004
        start = float(sampling.percent_to_sigma(config.start_percent))
        end = float(sampling.percent_to_sigma(config.end_percent))
        native = _native()
        for index in range(4):
            self.delegates.append(
                native.AnimaLLLitePatch(
                    config.model_patch,
                    config.reference_tiles[index::4, ..., :3],
                    None,
                    config.strength,
                    start,
                    end,
                )
            )

    def validate_live(self, options):
        if self.plan is None or self.model is None:
            raise RuntimeError("LLLite invocation is closed or unprepared")
        if get_attachment(self.model) is not self.attachment:
            raise ValueError("Tiled LLLite declaration changed during sampling")
        dispatcher = validate_hook_options(options, self.attachment)
        if self.attachment is None and dispatcher is not None:
            raise ValueError("Orphaned tiled LLLite hooks without an attachment")

    @contextmanager
    def tile_options(self, options, region):
        from comfy.patcher_extension import WrappersMP

        self.validate_live(options)
        if self.attachment is None:
            yield options
            return
        if region not in self.plan.regions:
            raise ValueError("LLLite tile region does not match TILE_PLAN")
        transformer = options.get("transformer_options", {})
        wrappers = transformer.get("wrappers", {})
        forward_wrappers = wrappers.get(WrappersMP.DIFFUSION_MODEL, {})
        if DISPATCH_KEY in transformer or DISPATCH_KEY in forward_wrappers:
            raise ValueError("Nested or stale tiled LLLite dispatch")
        handle = TileDispatch(self, region)
        self.handles.add(handle)
        # Run after existing forward wrappers, independently of post_input: a
        # removed hook cannot validate its own removal. Copy only owned groups.
        # Native MiniTrainDIT.forward supplies transformer_options by keyword.
        forward_wrappers = {**forward_wrappers, DISPATCH_KEY: [handle.guard_forward]}
        tiled = {
            **options,
            "transformer_options": {
                **transformer,
                DISPATCH_KEY: handle,
                "wrappers": {**wrappers, WrappersMP.DIFFUSION_MODEL: forward_wrappers},
            },
        }
        try:
            yield tiled
        finally:
            handle.close()
            self.handles.discard(handle)
            tiled["transformer_options"].pop(DISPATCH_KEY, None)
            forward_wrappers.pop(DISPATCH_KEY, None)

    def dispatch(self, dispatcher, handle, args):
        if handle not in self.handles or dispatcher is not self.attachment.post_input:
            raise ValueError("Mismatched tiled LLLite invocation or dispatcher")
        transformer = args["transformer_options"]
        self.validate_live({"transformer_options": transformer})
        region = handle.region
        if region not in self.plan.regions:
            raise ValueError("LLLite tile region does not match TILE_PLAN")
        x, img = args["x"], args["img"]
        height, width = self.plan.tile_hw
        if (
            not isinstance(x, torch.Tensor)
            or x.ndim != 5
            or not x.is_floating_point()
            or x.shape[1:] != (16, 1, height, width)
            or x.shape[0] < 1
            or x.shape[0] % self.batch
        ):
            raise ValueError(
                "LLLite forward geometry or latent batch does not match tiled dispatch"
            )
        branches = transformer.get("cond_or_uncond")
        if branches is not None and x.shape[0] != self.batch * len(branches):
            raise ValueError(
                "LLLite conditioning batch mapping does not match the latent batch"
            )
        if not isinstance(img, torch.Tensor) or img.shape != (
            x.shape[0],
            1,
            height // 2,
            width // 2,
            self.attachment.config.model_patch.model.model_dim,
        ):
            raise ValueError("LLLite embedded image geometry does not match TILE_PLAN")
        config = self.attachment.config
        groups = validate_references(config.reference_tiles, self.plan, self.batch)
        delegate = self.delegates[region.index]
        if tuple(delegate.image.shape) != (groups, height * 8, width * 8, 3):
            raise ValueError(
                "LLLite reference dimensions changed before native preparation"
            )
        data = transformer.get("model_patch_data")
        if not isinstance(data, dict):
            raise ValueError("Native Anima forward must provide fresh model_patch_data")  # noqa: TRY004
        data.pop(dispatcher, None)
        data.pop(delegate, None)
        try:
            result = delegate(args)
        finally:
            embedding = data.pop(delegate, None)
        if embedding is not None:
            if (
                not isinstance(embedding, torch.Tensor)
                or embedding.ndim != 3
                or not embedding.is_floating_point()
                or embedding.shape[:2] != (groups, (height // 2) * (width // 2))
            ):
                raise ValueError(
                    "Native LLLite encoded token count or reference batch mismatch"
                )
            data[dispatcher] = embedding
        return result

    def close(self):
        for handle in self.handles:
            handle.close()
        self.handles.clear()
        self.delegates.clear()
        self.attachment = None
        self.model = None
        self.plan = None
        self.batch = None
