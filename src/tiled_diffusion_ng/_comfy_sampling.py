# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Shared ComfyUI orchestration with invocation-local sampling state."""

import torch

from .adapters import resolve_adapter
from .fusion import FusionWeights, working_dtype
from .geometry import TILE_IDS, crop

TAG = "tdng_tile_id"
WRAPPER_KEY = "tiled_diffusion_ng.v1"


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
    if "sampler_calc_cond_batch_function" in options:
        raise ValueError(
            "Unsupported model option: sampler_calc_cond_batch_function "
            "replaces ComfyUI conditioning evaluation and bypasses tiling"
        )
    for key in (
        "context_handler",
        "model_function_wrapper",
        "multigpu_clones",
    ):
        if options.get(key) is not None:
            raise ValueError(f"Unsupported model option: {key}")
    transformer = options.get("transformer_options", {})
    if transformer.get("context_handler") is not None:
        raise ValueError("Unsupported transformer context_handler")
    if not allow_tiled:
        for kind in ("calc_cond_batch", "predict_noise"):
            if transformer.get("wrappers", {}).get(kind, {}).get(WRAPPER_KEY):
                raise ValueError("Model is already tiled by Tiled Diffusion NG")


def guard_prediction(executor, x, timestep, model_options, seed=None):
    # Native samplers can change these options after common_ksampler starts.
    # Guard the live prediction call BEFORE sampling_function dispatches its
    # optional override, which otherwise skips CALC_COND_BATCH and our wrapper.
    # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py#L609-L623
    # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py#L1210-L1218
    validate_options(model_options, allow_tiled=True)
    return executor(x, timestep, model_options, seed)


def validate_conditioning(conditioning, adapter, label):
    if not isinstance(conditioning, (list, tuple)) or not conditioning:
        raise ValueError(f"{label} must be a complete, nonempty CONDITIONING")
    for entry in conditioning:
        if (
            not isinstance(entry, (list, tuple))
            or len(entry) != 2
            or not isinstance(entry[1], dict)
        ):
            raise ValueError(
                f"{label} has invalid CONDITIONING nesting; expected [embedding, metadata] entries"
            )
        adapter.validate_condition(entry[0], entry[1])


def prepare_pairs(positives, negative, plan, context):
    from comfy import samplers

    positive_out, negative_out = [], []
    for region, positive in zip(plan.regions, positives, strict=True):
        # The helper consumes dicts. Keep each embedding under a temporary key,
        # so replaced negative entries retain their original embeddings.
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
        p, n = context.prepare_pair(p, n, region)
        for entries, out in ((p, positive_out), (n, negative_out)):
            for meta in entries:
                embedding = meta.pop("tdng_embedding")
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
        self.adapter.validate_evaluation(x_in, self.plan)
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
    wrapper_types = (WrappersMP.PREDICT_NOISE, WrappersMP.CALC_COND_BATCH)
    if any(model.get_wrappers(kind, WRAPPER_KEY) for kind in wrapper_types):
        raise ValueError("Model is already tiled by Tiled Diffusion NG")
    if getattr(model, "additional_models", {}).get("multigpu"):
        raise ValueError(
            "Multi-device model clones are outside the supported tiled sampling path"
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
    # Locals replace each entire positive, including its controls.
    positives = [positive] * 4 if local_positive is None else local_positive
    context = adapter.create_sampling_context(tile_plan)
    evaluation = TileEvaluation(tile_plan, adapter)
    clone = None
    try:
        positives, negatives = prepare_pairs(positives, negative, tile_plan, context)
        context.finalize_preparation()
        clone = model.clone()
        clone.add_wrapper_with_key(
            WrappersMP.PREDICT_NOISE, WRAPPER_KEY, guard_prediction
        )
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
        try:
            context.close()
        finally:
            if clone is not None:
                for kind in wrapper_types:
                    _detach(clone, kind)
