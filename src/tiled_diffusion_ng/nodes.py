# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Public V3 nodes; local positives use execution-list transport."""

from comfy_api.latest import io

from . import _comfy_sampling
from ._krea2_conditioning import encode_tiles
from .adapters import resolve_adapter
from .adapters._anima_lllite import apply_lllite
from .geometry import image_views, make_plan

PLAN = io.Custom("TILE_PLAN")
CATEGORY = "Tiled Diffusion NG"


class TiledAnimaLLLiteApply(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TiledDiffusionNG_TiledAnimaLLLiteApply",
            display_name="TiledAnimaLLLiteApply",
            category=CATEGORY,
            description="Apply one native-loaded RGB Anima LLLite MODEL_PATCH for tiled refinement. Create TilePlan from the base MODEL and target LATENT, crop the matching canvas with TileView, then apply and sample with that same complete TilePlan. Reference order is image 0 TL/TR/BR/BL, then image 1 TL/TR/BR/BL; one group may broadcast to the latent batch. Local positives change text guidance while this MODEL patch remains active on whichever branches native sampling evaluates. Use TileSampler. Repeated application, native AnimaLLLiteApply chaining, masks and four-channel checkpoints are unsupported.",
            inputs=[
                io.Model.Input("model"),
                io.Custom("MODEL_PATCH").Input("model_patch"),
                PLAN.Input("tile_plan"),
                io.Image.Input("reference_tiles"),
                io.Float.Input("strength", default=1.0, min=-10.0, max=10.0, step=0.01),
                io.Float.Input(
                    "start_percent", default=0.0, min=0.0, max=1.0, step=0.001
                ),
                io.Float.Input(
                    "end_percent", default=1.0, min=0.0, max=1.0, step=0.001
                ),
            ],
            outputs=[io.Model.Output("model")],
        )

    @classmethod
    def execute(
        cls,
        model,
        model_patch,
        tile_plan,
        reference_tiles,
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
    ):
        return io.NodeOutput(
            apply_lllite(
                model,
                model_patch,
                tile_plan,
                reference_tiles,
                strength,
                start_percent,
                end_percent,
            )
        )


class TilePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TiledDiffusionNG_TilePlan",
            display_name="TilePlan",
            category=CATEGORY,
            description="Four views for SDXL, native Anima (Base, Aesthetic, Turbo, 2.9B), or Krea2 (Raw, Turbo). Anima and Krea2 require 16-channel image latents and canvas dimensions divisible by 16 pixels; overlap rounds to aligned views without padding.",
            inputs=[
                io.Model.Input("model"),
                io.Latent.Input("latent"),
                io.Int.Input(
                    "tile_overlap",
                    default=64,
                    min=0,
                    step=8,
                    tooltip="Shared overlap in pixels, not a per-side halo.",
                ),
            ],
            outputs=[PLAN.Output("tile_plan")],
        )

    @classmethod
    def execute(cls, model, latent, tile_overlap=64):
        return io.NodeOutput(
            make_plan(resolve_adapter(model).describe(model, latent), tile_overlap)
        )


class TileView(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TiledDiffusionNG_TileView",
            display_name="TileView",
            category=CATEGORY,
            description="Crop the plan's overlap-inclusive sampling rectangles from an image at the exact canvas size. Returns one IMAGE batch: image 0 TL/TR/BR/BL, then image 1 TL/TR/BR/BL, and so on.",
            inputs=[io.Image.Input("image"), PLAN.Input("tile_plan")],
            outputs=[io.Image.Output("tiles")],
        )

    @classmethod
    def execute(cls, image, tile_plan):
        return io.NodeOutput(image_views(image, tile_plan))


class TileKrea2Conditioning(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TiledDiffusionNG_TileKrea2Conditioning",
            display_name="TileKrea2Conditioning",
            category=CATEGORY,
            description=(
                "Jointly encode exactly four TileView images and optional texts with native Krea2 CLIP. "
                "Returns four complete local positives in TL/TR/BR/BL order for TileSampler. "
                "These replace the global positive, including its reference metadata. "
                "This node produces vision conditioning without VAE reference latents."
            ),
            is_input_list=True,
            inputs=[
                io.Clip.Input("clip"),
                io.Image.Input("reference_tiles"),
                io.String.Input(
                    "prompts",
                    optional=True,
                    force_input=True,
                    dynamic_prompts=False,
                    tooltip="Four literal strings in TL/TR/BR/BL execution-list order. Empty strings omit tile text; an absent or empty list encodes images only.",
                ),
                io.Float.Input(
                    "strength",
                    default=1.0,
                    min=0.0,
                    max=3.0,
                    step=0.05,
                    tooltip="Scale the complete image/text embedding amplitude. Visual effects depend on the model; 0 zeros the embeddings, 1 preserves native output.",
                ),
                io.Float.Input(
                    "end_percent",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    tooltip="After this diffusion percentage, use unscaled empty-prompt conditioning with no tile image or text. Native scheduling includes both entries at the exact cutoff.",
                ),
                io.Boolean.Input(
                    "downsize_to_1mp",
                    default=False,
                    tooltip="Downsize tiles larger than 1024² pixels with antialiased bicubic, preserving aspect ratio and floating-point pixels. Native Qwen preprocessing still applies.",
                ),
            ],
            outputs=[io.Conditioning.Output("local_positive", is_output_list=True)],
        )

    @classmethod
    def execute(
        cls, clip, reference_tiles, strength, end_percent, downsize_to_1mp, prompts=None
    ):
        ordinary = {
            "clip": clip,
            "reference_tiles": reference_tiles,
            "strength": strength,
            "end_percent": end_percent,
            "downsize_to_1mp": downsize_to_1mp,
        }
        for name, values in ordinary.items():
            if not isinstance(values, list) or len(values) != 1:
                raise ValueError(
                    f"{name} requires one execution-list item; list sweeps are unsupported"
                )
            ordinary[name] = values[0]
        return io.NodeOutput(encode_tiles(**ordinary, prompts=prompts))


class TileSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from comfy.samplers import KSampler

        # GET_SCHEMA in the pinned V3 host calls define_schema on each request;
        # no module-level snapshot can hide a later RES4LYF registration.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy_api/latest/_io.py#L2239-L2250
        return io.Schema(
            node_id="TiledDiffusionNG_TileSampler",
            display_name="TileSampler",
            category=CATEGORY,
            description=(
                "One native trajectory for SDXL, Anima, or Krea2 Raw/Turbo generation and refinement. "
                "Anima/Krea2 accept 16-channel 4D or single-frame 5D targets, require B×1×1×H×W noise masks, and return 5D. "
                "Krea2 keeps whole B×16×1×H×W references; choose index or index_timestep_zero with Edit Model Reference Method. "
                "Anima/Krea2 conditioning ControlNets are unsupported; Krea2 LLLite remains TBD. "
                "For Anima RGB refinement use TiledAnimaLLLiteApply with the same plan; native LLLite chaining is unsupported. "
                "Choose checkpoint-appropriate sampler/CFG settings; Turbo supports CFG 1 with ConditioningZeroOut."
            ),
            is_input_list=True,
            inputs=[
                io.Model.Input("model"),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input(
                    "cfg", default=8.0, min=0.0, max=100.0, step=0.1, round=0.01
                ),
                io.Combo.Input("sampler_name", options=list(KSampler.SAMPLERS)),
                io.Combo.Input("scheduler", options=list(KSampler.SCHEDULERS)),
                io.Conditioning.Input(
                    "positive",
                    tooltip="Used for every view unless local_positive supplies four complete replacements.",
                ),
                io.Conditioning.Input(
                    "negative",
                    tooltip="Shared across all views. ConditioningZeroOut is supported and preserves reference metadata. ComfyUI skips negative evaluation at CFG 1 unless a hook requests it.",
                ),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                PLAN.Input("tile_plan"),
                io.Conditioning.Input(
                    "local_positive",
                    optional=True,
                    tooltip="Four complete positives in TL/TR/BR/BL execution-list order. Each replaces the entire global positive, including Krea2 embeddings/references/method or SDXL controls. The supplied negative stays independent. Anima's MODEL LLLite patch remains active on evaluated branches. Use a list producer, not ConditioningCombine.",
                ),
            ],
            outputs=[io.Latent.Output("latent")],
        )

    @classmethod
    def execute(
        cls,
        model,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        positive,
        negative,
        latent_image,
        denoise,
        tile_plan,
        local_positive=None,
    ):
        ordinary = {
            "model": model,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "positive": positive,
            "negative": negative,
            "latent_image": latent_image,
            "denoise": denoise,
            "tile_plan": tile_plan,
        }
        for name, values in ordinary.items():
            if not isinstance(values, list) or len(values) != 1:
                raise ValueError(
                    f"{name} requires one execution-list item; list sweeps are unsupported"
                )
            ordinary[name] = values[0]
        return io.NodeOutput(
            _comfy_sampling.sample(**ordinary, local_positive=local_positive)
        )
