# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""The three public V3 nodes; local positives use execution-list transport."""

from comfy_api.latest import io

from . import _comfy_sampling
from .adapters import resolve_adapter
from .geometry import image_views, make_plan

PLAN = io.Custom("TILE_PLAN")
CATEGORY = "Tiled Diffusion NG"


class TilePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="TiledDiffusionNG_TilePlan",
            display_name="Prepare Four Tile Plan",
            category=CATEGORY,
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
            display_name="Extract Four Vision Views",
            category=CATEGORY,
            inputs=[io.Image.Input("image"), PLAN.Input("tile_plan")],
            outputs=[io.Image.Output("tiles")],
        )

    @classmethod
    def execute(cls, image, tile_plan):
        return io.NodeOutput(image_views(image, tile_plan))


class TileSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        from comfy.samplers import KSampler

        # GET_SCHEMA in the pinned V3 host calls define_schema on each request;
        # no module-level snapshot can hide a later RES4LYF registration.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy_api/latest/_io.py#L2239-L2250
        return io.Schema(
            node_id="TiledDiffusionNG_TileSampler",
            display_name="Sample Four Tiles",
            category=CATEGORY,
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
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Latent.Input("latent_image"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
                PLAN.Input("tile_plan"),
                io.Conditioning.Input(
                    "local_positive",
                    optional=True,
                    tooltip="Execution list of four complete positives: TL, TR, BR, BL. Each replaces the entire global positive, including its controls. Attach controls to locals when needed; use a list producer, not ConditioningCombine.",
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
