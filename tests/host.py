# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""CPU contract doubles; these do not exercise ComfyUI inference.

Source baselines and pending host checks: docs/comfyui-compatibility.md.
"""

import copy
import math
import sys
from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@contextmanager
def isolated_host():
    """Restore host imports and cached node classes, including nested imports."""
    import tiled_diffusion_ng

    prefixes = (
        "comfy",
        "comfy_api",
        "nodes",
        "tdng_clone",
        "tiled_diffusion_ng.nodes",
        "tiled_diffusion_ng.extension",
    )

    def temporary(name):
        return any(
            name == prefix or name.startswith(prefix + ".") for prefix in prefixes
        )

    modules = {name: module for name, module in sys.modules.items() if temporary(name)}
    missing = object()
    attributes = {
        name: getattr(tiled_diffusion_ng, name, missing)
        for name in ("nodes", "extension")
    }
    for name in ("tiled_diffusion_ng.nodes", "tiled_diffusion_ng.extension"):
        sys.modules.pop(name, None)
        tiled_diffusion_ng.__dict__.pop(name.rsplit(".", 1)[1], None)
    try:
        with pytest.MonkeyPatch.context() as monkeypatch:
            yield Host(monkeypatch)
    finally:
        for name in tuple(sys.modules):
            if temporary(name):
                del sys.modules[name]
        sys.modules.update(modules)
        for name, value in attributes.items():
            if value is missing:
                tiled_diffusion_ng.__dict__.pop(name, None)
            else:
                setattr(tiled_diffusion_ng, name, value)


def cond(value, **metadata):
    return [
        [
            torch.full((1, 2, 3), float(value)),
            {"pooled_output": torch.tensor([[value]]), **metadata},
        ]
    ]


def reshape_mask(mask, shape):
    # Native 5D preparation treats a sub-5D mask's leading axes as time,
    # interpolates that axis, then repeats the result across the image batch.
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/utils.py#L1350-L1369
    if len(shape) == 4:
        mask = mask.reshape(-1, 1, *mask.shape[-2:])
        mode = "bilinear"
    else:
        if mask.ndim < 5:
            mask = mask.reshape(1, 1, -1, *mask.shape[-2:])
        mode = "trilinear"
    mask = F.interpolate(mask, size=shape[2:], mode=mode)
    if mask.shape[1] < shape[1]:
        mask = mask.repeat((1, shape[1]) + (1,) * (len(shape) - 2))[:, : shape[1]]
    return mask.repeat(
        (math.ceil(shape[0] / mask.shape[0]),) + (1,) * (len(shape) - 1)
    )[: shape[0]]


def copy_containers(value):
    # ModelPatcher clones list/dict containers, preserving tensor and callable
    # identities. deepcopy would hide mutations to graph-owned patch objects.
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_patcher.py#L449-L483
    if isinstance(value, dict):
        return {key: copy_containers(item) for key, item in value.items()}
    if isinstance(value, list):
        return [copy_containers(item) for item in value]
    return value


class SDXLFormat:
    latent_channels = 4
    latent_dimensions = 2
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 1


class Wan21Format:
    latent_channels = 16
    latent_dimensions = 3
    spacial_downscale_ratio = 8
    temporal_downscale_ratio = 4


class CONST:
    def calculate_input(self, sigma, x):
        return x

    def calculate_denoised(self, sigma, prediction, x):
        return x - sigma.reshape((-1,) + (1,) * (x.ndim - 1)) * prediction


class VideoRopePosition3DEmb:
    pass


class AnimaNetwork:
    in_channels = out_channels = 16
    patch_spatial = 2
    patch_temporal = 1
    pos_emb_cls = "rope3d"
    extra_per_block_abs_pos_emb = False
    concat_padding_mask = True
    rope_h_extrapolation_ratio = rope_w_extrapolation_ratio = 4.0
    rope_t_extrapolation_ratio = 1.0

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.pos_embedder = VideoRopePosition3DEmb()
        self.text_calls = []

    def preprocess_text_embeds(self, embedding, ids, t5xxl_weights=None):
        # Distinct, small deterministic text math, not an LLM adapter substitute.
        # Preserve native token weighting, padding and preparation ownership.
        self.text_calls.append((embedding, ids, t5xxl_weights))
        output = embedding.mean(1, keepdim=True) + ids.unsqueeze(-1) * 0.01
        if t5xxl_weights is not None:
            output = output * t5xxl_weights
        return F.pad(output, (0, 0, 0, max(0, 512 - output.shape[1])))


class Anima:
    def __init__(self, num_blocks=28):
        self.diffusion_model = AnimaNetwork(num_blocks)
        self.model_config = SimpleNamespace(
            unet_config={"image_model": "anima", "num_blocks": num_blocks}
        )
        self.concat_keys = ()
        self.condition_calls = []

    def extra_conds(self, **kwargs):
        # Inference prepares text once; the non-inference native path carries
        # ids/weights into forward. These contracts deliberately use no host code.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_base.py#L1482-L1505
        self.condition_calls.append(kwargs)
        embedding = kwargs["cross_attn"]
        prepared = {}
        if kwargs.get("t5xxl_ids") is not None:
            ids = kwargs["t5xxl_ids"].unsqueeze(0)
            weights = kwargs["t5xxl_weights"].unsqueeze(0).unsqueeze(-1).to(embedding)
            if torch.is_inference_mode_enabled():
                embedding = self.diffusion_model.preprocess_text_embeds(
                    embedding, ids, weights
                )
            else:
                prepared.update(t5xxl_ids=ids, t5xxl_weights=weights)
        return dict(prepared, c_crossattn=embedding)


class EPS:
    def calculate_input(self, sigma, x):
        return x / (sigma.square() + 1).sqrt()

    def calculate_denoised(self, sigma, prediction, x):
        return x - sigma * prediction


class VPrediction(EPS):
    def calculate_denoised(self, sigma, prediction, x):
        return (
            x / (sigma.square() + 1) - prediction * sigma / (sigma.square() + 1).sqrt()
        )


class SDXL:
    def __init__(self):
        self.model_config = SimpleNamespace(
            unet_config={"in_channels": 4, "out_channels": 4}
        )
        self.concat_keys = ()


class Model:
    def __init__(self, family="sdxl", *, num_blocks=28):
        self.model = SDXL() if family == "sdxl" else Anima(num_blocks)
        self.format = SDXLFormat() if family == "sdxl" else Wan21Format()
        self.sampling = EPS() if family == "sdxl" else CONST()
        self.model_options = {"transformer_options": {}}
        self.wrappers = {}
        self.additional_models = {}
        self.attachments = {}
        self.patches = {"lora": [object()]}

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.model.diffusion_model
        return {"latent_format": self.format, "model_sampling": self.sampling}[name]

    def clone(self):
        result = copy.copy(self)
        result.model_options = copy_containers(self.model_options)
        result.patches = copy_containers(self.patches)
        result.attachments = self.attachments.copy()
        result.additional_models = {
            key: [model.clone() for model in models]
            for key, models in self.additional_models.items()
        }
        result.wrappers = {
            kind: {key: list(items) for key, items in groups.items()}
            for kind, groups in self.wrappers.items()
        }
        return result

    def get_nested_additional_models(self):
        result = []
        for models in self.additional_models.values():
            for model in models:
                result.extend([model, *model.get_nested_additional_models()])
        return result

    def add_wrapper_with_key(self, kind, key, wrapper):
        self.wrappers.setdefault(kind, {}).setdefault(key, []).append(wrapper)

    def remove_wrappers_with_key(self, kind, key):
        self.wrappers.get(kind, {}).pop(key, None)

    def get_wrappers(self, kind, key):
        return self.wrappers.get(kind, {}).get(key, [])


def identity_hint(x):
    return x


def resize_hint(hint, width, height, algorithm, crop):
    # Ordinary BCHW interpolation contract; no host imports or checkpoints.
    h, w = hint.shape[-2:]
    if crop == "center":
        x = round((w - h * width / height) / 2) if w / h > width / height else 0
        y = round((h - w * height / width) / 2) if w / h < width / height else 0
        hint = hint[..., y : h - y, x : w - x]
    return F.interpolate(hint, size=(height, width), mode=algorithm)


class ControlNetwork:
    dims = 2
    in_channels = 4
    num_classes = "sequential"

    def __init__(self, *, union=False):
        self.label_emb = [[SimpleNamespace(in_features=2816)]]
        self.input_hint_block = [SimpleNamespace(in_channels=3)]
        if union:
            self.num_control_type = 8


class ControlNet:
    def __init__(self, preprocess_image=identity_hint, *, union=False):
        self.control_model = ControlNetwork(union=union)
        self.control_model_wrapped = object()
        self.cond_hint_original = torch.zeros(1, 3, 16, 16)
        self.cond_hint = None
        self.previous_controlnet = None
        self.compression_ratio = 8
        self.extra_conds = ["y"]
        self.extra_args = {}
        self.preprocess_image = preprocess_image
        self.upscale_algorithm = "nearest-exact"
        self.strength = 1.0
        self.timestep_percent_range = (0.0, 1.0)
        self.timestep_range = None
        self.extra_concat = None
        self.extra_concat_orig = []
        self.extra_hooks = None
        self.vae = self.latent_format = None
        self.concat_mask = False
        self.multigpu_clones = {}
        self.model_sampling_current = None
        self.cleanups = 0
        self.pre_runs = 0
        self.hint_preparations = 0

    def copy(self):
        result = ControlNet(self.preprocess_image)
        for field in (
            "control_model",
            "control_model_wrapped",
            "cond_hint_original",
            "compression_ratio",
            "strength",
            "timestep_percent_range",
            "upscale_algorithm",
        ):
            setattr(result, field, getattr(self, field))
        result.extra_args = self.extra_args.copy()
        result.extra_conds = self.extra_conds.copy()
        return result

    def pre_run(self):
        assert self.cond_hint_original is not None
        self.pre_runs += 1
        self.timestep_range = (
            1 - self.timestep_percent_range[0],
            1 - self.timestep_percent_range[1],
        )
        self.model_sampling_current = object()
        if self.previous_controlnet:
            self.previous_controlnet.pre_run()

    def cleanup(self):
        self.cleanups += 1
        self.cond_hint = self.model_sampling_current = self.timestep_range = None
        if self.previous_controlnet:
            self.previous_controlnet.cleanup()

    def predict(self, x, sigma):
        out = torch.zeros_like(x)
        if self.previous_controlnet:
            out += self.previous_controlnet.predict(x, sigma)
        if self.timestep_range[1] <= sigma.item() <= self.timestep_range[0]:
            pixel_hw = tuple(n * self.compression_ratio for n in x.shape[-2:])
            if self.cond_hint is None or self.cond_hint.shape[-2:] != pixel_hw:
                self.cond_hint = self.preprocess_image(
                    resize_hint(
                        self.cond_hint_original,
                        pixel_hw[1],
                        pixel_hw[0],
                        self.upscale_algorithm,
                        "center",
                    )
                ).to(x)
                self.hint_preparations += 1
            # A singleton broadcasts at the network; larger batches repeat or
            # truncate. This double evaluates one conditioning branch at a time.
            if self.cond_hint.shape[0] not in (1, x.shape[0]):
                self.cond_hint = self.cond_hint.repeat(
                    (x.shape[0] + self.cond_hint.shape[0] - 1)
                    // self.cond_hint.shape[0],
                    1,
                    1,
                    1,
                )[: x.shape[0]]
            pixels = F.interpolate(
                self.cond_hint.mean(1, keepdim=True),
                size=x.shape[-2:],
                mode="nearest-exact",
            )
            if pixels.shape[0] != x.shape[0]:
                pixels = pixels.repeat(
                    (x.shape[0] + pixels.shape[0] - 1) // pixels.shape[0], 1, 1, 1
                )[: x.shape[0]]
            out += pixels * self.strength
        return out


class Input:
    def __init__(self, id, **kwargs):
        self.id = id
        self.__dict__.update(kwargs)


class ComfyType:
    Input = Input
    Output = Input


class NodeOutput(tuple):
    def __new__(cls, *values):
        return super().__new__(cls, values)


class Host:
    def __init__(self, monkeypatch):
        self.common_calls = []
        self.tile_calls = []
        self.global_evaluations = []
        self.discovered = []
        self.prepared = None
        self.resize_calls = []
        self.interrupt_after = None
        self.interrupt_checks = 0
        self.fail_tile = None
        self.evaluations_per_step = 3
        self.latest_clone = None
        self.mutate_prepared = None
        self.option_update = None
        self.live_options = None
        self.defer_anima_text = False
        self.native_calls = []
        self.discovered_models = []

        def module(name, **attrs):
            item = ModuleType(name)
            item.__dict__.update(attrs)
            monkeypatch.setitem(sys.modules, name, item)
            if "." in name:
                parent, key = name.rsplit(".", 1)
                setattr(sys.modules[parent], key, item)
            return item

        module("comfy")
        module("comfy.model_base", SDXL=SDXL, Anima=Anima)
        module("comfy.latent_formats", SDXL=SDXLFormat, Wan21=Wan21Format)
        module("comfy.model_sampling", EPS=EPS, V_PREDICTION=VPrediction, CONST=CONST)
        module("comfy.ldm")
        module("comfy.ldm.anima")
        module("comfy.ldm.anima.model", Anima=AnimaNetwork)
        module("comfy.ldm.cosmos")
        module(
            "comfy.ldm.cosmos.position_embedding",
            VideoRopePosition3DEmb=VideoRopePosition3DEmb,
        )
        module("comfy.controlnet", ControlNet=ControlNet)
        module("comfy.cldm")
        module("comfy.cldm.cldm", ControlNet=ControlNetwork)
        module("comfy.utils", common_upscale=self.resize)
        module(
            "comfy.model_management",
            throw_exception_if_processing_interrupted=self.interrupt,
        )
        module(
            "comfy.patcher_extension",
            WrappersMP=SimpleNamespace(
                CALC_COND_BATCH="calc_cond_batch",
                PREDICT_NOISE="predict_noise",
                APPLY_MODEL="apply_model",
                DIFFUSION_MODEL="diffusion_model",
            ),
        )
        self.samplers = module(
            "comfy.samplers",
            KSampler=type(
                "KSampler",
                (),
                {"SAMPLERS": ["euler", "heun"], "SCHEDULERS": ["normal", "karras"]},
            ),
            SCHEDULER_HANDLERS={
                key: SimpleNamespace(handler=self.schedule)
                for key in ("normal", "karras")
            },
            sampler_object=lambda name: self.dispatch[name],
            apply_empty_x_to_equal_area=self.propagate,
        )
        self.dispatch = {
            key: SimpleNamespace(sampler_function=self.solve)
            for key in ("euler", "heun")
        }
        self.nodes = module("nodes", common_ksampler=self.common_ksampler)
        module("comfy_api")
        io = SimpleNamespace(
            ComfyNode=type("ComfyNode", (), {}),
            Schema=lambda **kwargs: SimpleNamespace(**kwargs),
            NodeOutput=NodeOutput,
            Custom=lambda name: ComfyType,
        )
        for name in (
            "Model",
            "Latent",
            "Image",
            "Conditioning",
            "Int",
            "Float",
            "Combo",
        ):
            setattr(io, name, ComfyType)
        module("comfy_api.latest", io=io, ComfyExtension=type("ComfyExtension", (), {}))

    @staticmethod
    def schedule(steps, denoise):
        return [0.75 - step / (steps + 1) * 0.5 for step in range(steps)]

    def solve(self, evaluate, x, sigmas):
        for step, sigma in enumerate(sigmas):
            for substep in range(self.evaluations_per_step):
                x = evaluate(x, torch.tensor([sigma], dtype=x.dtype), (step, substep))
        return x

    def interrupt(self):
        self.interrupt_checks += 1
        if self.interrupt_after == self.interrupt_checks:
            raise InterruptedError("cancelled")

    def resize(self, hint, width, height, algorithm, crop):
        self.resize_calls.append((hint, width, height, algorithm, crop))
        return resize_hint(hint, width, height, algorithm, crop)

    @staticmethod
    def propagate(positives, negatives, field, fill):
        # Pinned host contract: explicit negative control blocks autofill.
        # Repeated modulo slots are overwritten using the original metadata,
        # so the final positive control assigned to each slot wins.
        if any(entry.get(field) is not None for entry in negatives):
            return
        controls = [entry[field] for entry in positives if entry.get(field) is not None]
        originals = list(negatives)
        for index, _ in enumerate(controls):
            slot = index % len(originals)
            entry = dict(originals[slot], **{field: fill(controls, index)})
            negatives[slot] = entry

    def leaf(self, model, branches, x, sigma, options):
        self.tile_calls.append((branches, x.clone(), sigma.clone(), options))
        if self.fail_tile == len(self.tile_calls):
            raise RuntimeError("deliberate tile failure")
        outputs = []
        for branch in branches:
            active = []
            for entry in branch or []:
                if (
                    entry.get("timestep_start", 1.0) < sigma[0]
                    or entry.get("timestep_end", 0.0) > sigma[0]
                ):
                    continue
                prediction = x * 0.25 + entry["cross_attn"].mean().to(x)
                if isinstance(model, Anima):
                    # Native conditional evaluation forwards the live sigma in
                    # transformer options and leaves prediction conversion in
                    # BaseModel. Neither responsibility belongs to the adapter.
                    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/samplers.py#L309-L334
                    transformer = copy_containers(options["transformer_options"])
                    transformer["sigmas"] = sigma
                    prediction = self.call_wrappers(
                        transformer,
                        "apply_model",
                        self.anima_apply_model,
                        model,
                        x,
                        sigma,
                        entry["model_conds"],
                        transformer,
                    )
                if entry.get("control") is not None:
                    prediction = prediction + entry["control"].predict(x, sigma)
                active.append((prediction, entry.get("strength", 1.0)))
            if active:
                outputs.append(
                    sum(p * w for p, w in active) / sum(w for _, w in active)
                )
            else:
                outputs.append(torch.zeros_like(x))
        return outputs

    @staticmethod
    def call_wrappers(options, kind, original, *args):
        wrappers = [
            wrapper
            for group in options.get("wrappers", {}).get(kind, {}).values()
            for wrapper in group
        ]

        def continuation(index, *a, **kwargs):
            if index == len(wrappers):
                return original(*a, **kwargs)
            return wrappers[index](
                lambda *next_args, **next_kwargs: continuation(
                    index + 1, *next_args, **next_kwargs
                ),
                *a,
                **kwargs,
            )

        return continuation(0, *args)

    def anima_apply_model(self, model, x, sigma, conditions, transformer):
        native_sampling = self.latest_clone.sampling
        model_input = native_sampling.calculate_input(sigma, x)
        prediction = self.call_wrappers(
            transformer,
            "diffusion_model",
            self.anima_forward,
            model,
            model_input,
            sigma,
            conditions,
            transformer,
        )
        # BaseModel converts the native prediction to float32 before denoising.
        # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_base.py#L251-L255
        return native_sampling.calculate_denoised(sigma, prediction.float(), x)

    def anima_forward(self, model, x, sigma, conditions, transformer):
        self.native_calls.append((x.clone(), sigma.clone(), transformer))
        embedding = conditions["c_crossattn"]
        if "t5xxl_ids" in conditions:
            embedding = model.diffusion_model.preprocess_text_embeds(
                embedding, conditions["t5xxl_ids"], conditions["t5xxl_weights"]
            )
        return x * 0.25 + embedding.mean().to(x)

    def common_ksampler(
        self,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent,
        denoise=1.0,
    ):
        self.latest_clone = model
        self.discovered_models.extend(model.get_nested_additional_models())
        self.common_calls.append(
            {
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "positive": positive,
                "negative": negative,
                "latent": latent,
                "denoise": denoise,
            }
        )
        branches = []
        for raw in (positive, negative):
            prepared = []
            for embedding, metadata in raw:
                item = dict(metadata, cross_attn=embedding)
                if isinstance(model.model, Anima):
                    with torch.inference_mode(not self.defer_anima_text):
                        item["model_conds"] = model.model.extra_conds(**item)
                else:
                    item["model_conds"] = {
                        "c_crossattn": embedding,
                        "y": (
                            metadata.get("width", latent["samples"].shape[-1] * 8),
                            metadata.get("height", latent["samples"].shape[-2] * 8),
                        ),
                    }
                # Percent settings override explicit timesteps when present.
                # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/samplers.py#L859-L882
                if "start_percent" in metadata:
                    item["timestep_start"] = 1 - metadata["start_percent"]
                if "end_percent" in metadata:
                    item["timestep_end"] = 1 - metadata["end_percent"]
                # Native hook discovery checks key presence, including None.
                # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/sampler_helpers.py#L31-L50
                if "control" in metadata:
                    control = metadata["control"]
                    if control not in self.discovered:
                        self.discovered.append(control)
                    control.pre_run()
                prepared.append(item)
            branches.append(prepared)
        self.prepared = branches
        if self.mutate_prepared:
            self.mutate_prepared(branches)
        options = copy_containers(model.model_options)
        options["transformer_options"]["wrappers"] = model.wrappers
        self.live_options = options
        if self.option_update:
            self.option_update(options)

        x = latent["samples"].clone()
        if model.format.latent_dimensions == 3 and x.ndim == 4:
            x = x.unsqueeze(2)
        original_x = x.clone()
        batch_indices = latent.get("batch_index", list(range(x.shape[0])))
        noise = torch.cat(
            [
                torch.randn(
                    (1, *x.shape[1:]),
                    dtype=x.dtype,
                    generator=torch.Generator().manual_seed(seed + index),
                )
                for index in batch_indices
            ]
        )
        if denoise:
            x = x + noise * denoise

            def predict_noise(current_x, sigma, model_options, seed):
                uncond = (
                    None
                    if cfg == 1 and not model_options.get("disable_cfg1_optimization")
                    else branches[1]
                )
                # The pinned host dispatches an override BEFORE calc_cond_batch.
                # Tests must reproduce that bypass instead of routing overrides
                # through the tiling continuation unconditionally.
                if "sampler_calc_cond_batch_function" in model_options:
                    out = model_options["sampler_calc_cond_batch_function"](
                        {
                            "conds": [branches[0], uncond],
                            "input": current_x,
                            "sigma": sigma,
                            "model": model.model,
                            "model_options": model_options,
                        }
                    )
                else:
                    out = self.call_wrappers(
                        model_options["transformer_options"],
                        "calc_cond_batch",
                        self.leaf,
                        model.model,
                        [branches[0], uncond],
                        current_x,
                        sigma,
                        model_options,
                    )
                for fn in model_options.get("sampler_pre_cfg_function", []):
                    out = fn({"conds_out": out, "input": current_x})
                if "sampler_cfg_function" in model_options:
                    result = current_x - model_options["sampler_cfg_function"](
                        {
                            "cond": current_x - out[0],
                            "uncond": current_x - out[1],
                            "cond_denoised": out[0],
                            "uncond_denoised": out[1],
                            "cond_scale": cfg,
                            "input": current_x,
                            "sigma": sigma,
                            "model_options": model_options,
                        }
                    )
                else:
                    result = out[1] + cfg * (out[0] - out[1])
                for fn in model_options.get("sampler_post_cfg_function", []):
                    result = fn(
                        {
                            "cond_denoised": out[0],
                            "uncond_denoised": out[1],
                            "denoised": result,
                            "input": current_x,
                        }
                    )
                return result

            def evaluate(current_x, sigma, evaluation, model_options=None):
                live = self.live_options if model_options is None else model_options
                live["evaluation"] = evaluation
                self.global_evaluations.append((current_x.clone(), sigma.clone()))
                # PREDICT_NOISE comes from guider-owned options, even when a
                # sampler replaces the options passed to this prediction call.
                # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/samplers.py#L1210-L1218
                result = self.call_wrappers(
                    {"wrappers": model.wrappers},
                    "predict_noise",
                    predict_noise,
                    current_x,
                    sigma,
                    live,
                    seed,
                )
                if latent.get("noise_mask") is not None:
                    mask = reshape_mask(latent["noise_mask"], current_x.shape)
                    result = result * mask + original_x * (1 - mask)
                return result

            sigmas = self.samplers.SCHEDULER_HANDLERS[scheduler].handler(steps, denoise)
            x = self.samplers.sampler_object(sampler_name).sampler_function(
                evaluate, x, sigmas
            )
            if isinstance(model.model, Anima):
                # Native output conversion is host-owned, independent of the
                # input/model precision. Wan21 scaling itself is not simulated.
                # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/samplers.py#L1237-L1238
                x = x.float()
        for control in self.discovered:
            control.cleanup()
        result = dict(latent, samples=x)
        result.pop("downscale_ratio_spacial", None)
        result.pop("downscale_ratio_temporal", None)
        return (result,)
