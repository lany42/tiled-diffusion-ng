# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Pinned ComfyUI compatibility boundary; everything here is invocation-local."""

import inspect
import logging

import torch

from .adapters import resolve_adapter
from .fusion import FusionWeights, working_dtype
from .geometry import TILE_IDS, crop

TAG = "tdng_tile_id"
WRAPPER_KEY = "tiled_diffusion_ng.v1"
logger = logging.getLogger(__name__)


def validate_registrations(sampler_name, scheduler):
    from comfy import samplers

    if sampler_name not in samplers.KSampler.SAMPLERS:
        raise ValueError(f"Missing native sampler registration: {sampler_name}")
    if scheduler not in samplers.KSampler.SCHEDULERS:
        raise ValueError(f"Missing native scheduler registration: {scheduler}")
    try:
        sampler = samplers.sampler_object(sampler_name)
        if not callable(sampler.sampler_function):
            raise TypeError("sampler callable is missing")
        if not callable(samplers.SCHEDULER_HANDLERS[scheduler].handler):
            raise TypeError("scheduler callable is missing")
    except (AttributeError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Cannot resolve native registration {sampler_name}/{scheduler}: {exc}"
        ) from exc


def validate_options(options, *, allow_tiled=False):
    for key in (
        "sampler_calc_cond_batch_function",
        "context_handler",
        "model_function_wrapper",
        "multigpu_clones",
    ):
        if options.get(key) is not None:
            raise ValueError(f"Unsupported model option: {key}")
    transformer = options.get("transformer_options", {})
    if transformer.get("context_handler") is not None:
        raise ValueError("Unsupported transformer context_handler")
    if not allow_tiled and transformer.get("wrappers", {}).get(
        "calc_cond_batch", {}
    ).get(WRAPPER_KEY):
        raise ValueError("Model is already tiled by Tiled Diffusion NG")


def validate_conditioning(conditioning, adapter, label):
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError(f"{label} must be a complete, nonempty CONDITIONING")
    for entry in conditioning:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not isinstance(entry[0], torch.Tensor)
            or entry[0].ndim != 3
            or entry[0].numel() == 0
            or not isinstance(entry[1], dict)
        ):
            raise ValueError(
                f"{label} has invalid CONDITIONING nesting; expected [embedding, metadata] entries"
            )
        adapter.validate_condition(entry[1])


class ControlCopies:
    def __init__(self, plan):
        self.plan = plan
        self.controls = []
        self.normalized = {}

    def clone(self, original, region, memo, visiting):
        from comfy import controlnet, utils
        from comfy.cldm.cldm import ControlNet as ControlNetwork

        if original is None:
            return None
        key = id(original)
        if key in visiting:
            raise ValueError("Cyclic previous_controlnet chain")
        if key in memo:
            return memo[key]
        if (
            type(original) is not controlnet.ControlNet
            or type(original.control_model) is not ControlNetwork
        ):
            raise ValueError(
                "Unsupported control: only ordinary SDXL image-hint ControlNet is supported"
            )
        network = original.control_model
        if hasattr(network, "num_control_type"):
            raise ValueError("Unsupported ControlNet capability: Union control types")
        # Ordinary SDXL controls use sequential ADM with 2816 inputs. Checking
        # architecture attributes does not load or inspect parameter values.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/cldm/cldm.py#L119-L186
        if (
            network.dims != 2
            or network.in_channels != 4
            or network.num_classes != "sequential"
            or network.label_emb[0][0].in_features != 2816
            or network.input_hint_block[0].in_channels != 3
        ):
            raise ValueError(
                "ControlNet architecture is incompatible with standard SDXL RGB hints"
            )
        for field in (
            "vae",
            "latent_format",
            "concat_mask",
            "extra_concat_orig",
            "extra_hooks",
            "multigpu_clones",
        ):
            if getattr(original, field, None):
                raise ValueError(f"Unsupported ControlNet capability: {field}")
        if original.compression_ratio != 8 or set(original.extra_conds) - {"y"}:
            raise ValueError("Unsupported ControlNet compression_ratio/extra_conds")
        default_preprocess = (
            inspect.signature(controlnet.ControlNet)
            .parameters["preprocess_image"]
            .default
        )
        if original.preprocess_image is not default_preprocess:
            raise ValueError(
                "Unsupported ControlNet preprocess_image; requires an explicit full-canvas handler"
            )
        if any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in original.extra_args.values()
        ):
            raise ValueError("Unsupported ControlNet spatial extra_args")
        hint = original.cond_hint_original
        if (
            not isinstance(hint, torch.Tensor)
            or hint.ndim != 4
            or hint.shape[1] != 3
            or min(hint.shape) < 1
        ):
            raise ValueError(
                "ControlNet hint must be a nonempty BCHW RGB tensor in full-canvas coordinates"
            )

        visiting.add(key)
        clone = original.copy()
        if clone is original:
            raise ValueError("ControlNet.copy() must return an independent control")
        self.controls.append(clone)
        memo[key] = clone
        clone.previous_controlnet = None
        clone.cond_hint = None
        clone.timestep_range = None
        clone.extra_concat = None
        clone.model_sampling_current = None
        # Hints are BCHW. Host center-resize to full pixel dimensions FIRST,
        # then use the identical pixel rectangles used by TileView.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/controlnet.py#L269-L303
        # Coordinate comparison: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L330-L447
        hint_key = (id(hint), original.upscale_algorithm)
        if hint_key not in self.normalized:
            ph, pw = self.plan.pixel_hw
            self.normalized[hint_key] = utils.common_upscale(
                hint, pw, ph, original.upscale_algorithm, "center"
            )
        clone.cond_hint_original = crop(
            self.normalized[hint_key], region.pixel_sampling
        ).clone()
        clone.previous_controlnet = self.clone(
            original.previous_controlnet, region, memo, visiting
        )
        visiting.remove(key)
        return clone

    def close(self):
        # Sever clone-only chains before cleanup to visit each clone exactly once.
        # Host cleanup is idempotent; its success path may have run already.
        for control in self.controls:
            control.previous_controlnet = None
        for control in self.controls:
            try:
                control.cleanup()
            except Exception:
                logger.exception("Tiled Diffusion NG ControlNet cleanup failed")
            finally:
                control.cond_hint_original = None
                control.cond_hint = None
                control.extra_concat = None
                control.model_sampling_current = None
        self.controls.clear()
        self.normalized.clear()
        self.plan = None


def prepare_pairs(positives, negative, plan, controls):
    from comfy import samplers

    positive_out, negative_out = [], []
    for region, positive in zip(plan.regions, positives, strict=True):
        # The helper consumes dicts. Keep each embedding under a temporary key,
        # so duplicate negative entries retain the helper's exact pairing rules.
        p = [dict(meta, tdng_embedding=embedding) for embedding, meta in positive]
        n = [dict(meta, tdng_embedding=embedding) for embedding, meta in negative]
        # Resolve before flattening, never across tiles. The host will prepare
        # all discovered controls, but may not propagate them again globally.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py#L902-L940
        samplers.apply_empty_x_to_equal_area(
            [entry for entry in p if entry.get("control_apply_to_uncond", False)],
            n,
            "control",
            lambda found, index: found[index],
        )
        memo = {}
        for entries, out in ((p, positive_out), (n, negative_out)):
            for meta in entries:
                embedding = meta.pop("tdng_embedding")
                if meta.get("control") is not None:
                    meta["control"] = controls.clone(
                        meta["control"], region, memo, set()
                    )
                meta["control_apply_to_uncond"] = False
                meta[TAG] = region.tile_id
                out.append([embedding, meta])
    return positive_out, negative_out


class TileEvaluation:
    def __init__(self, plan, adapter):
        self.plan = plan
        self.adapter = adapter
        self.weights = {}

    def __call__(self, executor, model, conds, x_in, timestep, model_options):
        from comfy import model_management

        if self.plan is None:
            raise RuntimeError("Tiled sampling invocation has already closed")
        validate_options(model_options, allow_tiled=True)
        if (
            x_in.ndim != self.plan.signature.rank
            or x_in.shape[1] != self.plan.signature.channels
            or tuple(x_in.shape[-2:]) != self.plan.latent_hw
        ):
            raise ValueError("Model evaluation canvas does not match TILE_PLAN")
        for branch in conds:
            if branch is not None:
                if not branch or any(
                    entry.get(TAG) not in TILE_IDS for entry in branch
                ):
                    raise ValueError(
                        "Missing or unknown tile tag after host conditioning preparation"
                    )
                if {entry[TAG] for entry in branch} != set(TILE_IDS):
                    raise ValueError("Prepared conditioning is missing a tile")

        key = (x_in.device, working_dtype(x_in.dtype), x_in.ndim)
        if key not in self.weights:
            self.weights[key] = FusionWeights(
                self.plan, device=x_in.device, dtype=x_in.dtype, rank=x_in.ndim
            )

        accumulators = [None] * len(conds)
        output_dtypes = [None] * len(conds)
        weight_sets = [None] * len(conds)
        for region in self.plan.regions:
            model_management.throw_exception_if_processing_interrupted()
            tile_conds = [
                None
                if branch is None
                else [
                    self.adapter.adapt_spatial_condition(entry, region)
                    for entry in branch
                    if entry[TAG] == region.tile_id
                ]
                for branch in conds
            ]
            # All views read this same x/sigma. Call the continuation (never the
            # outer calc_cond_batch), preserving this evaluation's live options.
            tile_x = crop(x_in, region.sampling)
            predictions = executor(model, tile_conds, tile_x, timestep, model_options)
            if len(predictions) != len(conds):
                raise ValueError("Host returned an unexpected prediction branch count")
            for index, prediction in enumerate(predictions):
                # Native calc_cond_batch returns zeros_like(x) for an omitted
                # branch, not None. Preserve that boundary convention.
                if (
                    not isinstance(prediction, torch.Tensor)
                    or prediction.shape != tile_x.shape
                    or prediction.device != x_in.device
                ):
                    raise ValueError("Host returned an incompatible tile prediction")
                if accumulators[index] is None:
                    dtype = working_dtype(prediction.dtype)
                    key = (prediction.device, dtype, prediction.ndim)
                    if key not in self.weights:
                        self.weights[key] = FusionWeights(
                            self.plan,
                            device=prediction.device,
                            dtype=dtype,
                            rank=prediction.ndim,
                        )
                    weights = self.weights[key]
                    accumulators[index] = torch.zeros(
                        x_in.shape, dtype=dtype, device=x_in.device
                    )
                    output_dtypes[index] = prediction.dtype
                    weight_sets[index] = weights
                elif prediction.dtype != output_dtypes[index]:
                    raise ValueError("Host prediction dtype changed between tiles")
                weight_sets[index].accumulate(accumulators[index], prediction, region)
        # Paper eq.15 and linearity: F(N+c(P-N))=F(N)+c(F(P)-F(N)).
        # https://arxiv.org/html/2302.02412v1#S3
        # Let the host apply its shared CFG and full-canvas pre/post hooks once:
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py#L592-L632
        return [
            weights.normalize(acc, dtype)
            for weights, acc, dtype in zip(
                weight_sets, accumulators, output_dtypes, strict=True
            )
        ]

    def close(self):
        self.weights.clear()
        self.plan = None
        self.adapter = None


def _detach(clone, wrapper_type):
    clone.remove_wrappers_with_key(wrapper_type, WRAPPER_KEY)
    # Host preparation may copy wrappers to either of these private dictionaries.
    for options in (
        clone.model_options.get("transformer_options", {}),
        clone.model_options.get("to_load_options", {}),
    ):
        options.get("wrappers", {}).get(wrapper_type, {}).pop(WRAPPER_KEY, None)


def sample(
    model,
    seed,
    steps,
    cfg,
    sampler_name,
    scheduler,
    positive,
    negative,
    latent_image,
    tile_plan,
    denoise=1.0,
    local_positive=None,
):
    import nodes
    from comfy.patcher_extension import WrappersMP

    adapter = resolve_adapter(model)
    adapter.validate_sampling(model, latent_image, tile_plan)
    validate_registrations(sampler_name, scheduler)
    validate_options(model.model_options)
    if model.get_wrappers(WrappersMP.CALC_COND_BATCH, WRAPPER_KEY):
        raise ValueError("Model is already tiled by Tiled Diffusion NG")
    if getattr(model, "additional_models", {}).get("multigpu"):
        raise ValueError(
            "Multi-device model clones are outside the supported SDXL path"
        )
    validate_conditioning(positive, adapter, "positive")
    validate_conditioning(negative, adapter, "negative")
    if local_positive is not None:
        if not isinstance(local_positive, (list, tuple)) or len(local_positive) != 4:
            raise ValueError(
                "local_positive must contain four complete CONDITIONING values in TL/TR/BR/BL order"
            )
        for name, cond in zip(TILE_IDS, local_positive, strict=True):
            validate_conditioning(cond, adapter, f"local_positive {name}")
    positives = [positive] * 4 if local_positive is None else local_positive
    controls = ControlCopies(tile_plan)
    evaluation = TileEvaluation(tile_plan, adapter)
    clone = None
    try:
        positives, negatives = prepare_pairs(positives, negative, tile_plan, controls)
        clone = model.clone()
        clone.add_wrapper_with_key(WrappersMP.CALC_COND_BATCH, WRAPPER_KEY, evaluation)
        # Live lookup preserves extensions' dispatch. Noise/masks/batch indices,
        # schedules, callbacks, solver state and LATENT metadata stay host-owned.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/nodes.py#L1572-L1598
        return nodes.common_ksampler(
            clone,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positives,
            negatives,
            latent_image,
            denoise=denoise,
        )[0]
    finally:
        evaluation.close()
        controls.close()
        if clone is not None:
            _detach(clone, WrappersMP.CALC_COND_BATCH)
