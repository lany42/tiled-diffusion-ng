# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Invocation ownership for Anima text and tiled LLLite conditioning."""

from ._anima_lllite import LLLiteInvocation


def validate_control(condition):
    # The native transformer accepts **kwargs but does not consume UNet control
    # residuals. Do not let an incompatible control silently become a no-op.
    # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/ldm/cosmos/predict2.py#L852-L929
    if condition.get("control") is not None:
        raise ValueError(
            "Anima conditioning control is unsupported; SDXL/UNet ControlNet "
            "residuals are incompatible with the native Anima transformer"
        )


class AnimaSamplingContext:
    def __init__(self, plan):
        self.plan = plan
        self.lllite = LLLiteInvocation(plan)

    def prepare_model(self, model, latent):
        self.lllite.prepare_model(model, latent)

    def tile_options(self, options, region):
        return self.lllite.tile_options(options, region)

    def prepare_pair(self, positive, negative, region):
        if self.plan is None:
            raise RuntimeError("Anima sampling invocation has already closed")
        for branch in (positive, negative):
            for entry in branch:
                validate_control(entry)
                # prepare_pairs owns these dicts; native hooks test key presence.
                # https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/samplers.py#L1076-L1085
                entry.pop("control", None)
        return positive, negative

    def finalize_preparation(self):
        if self.plan is None:
            raise RuntimeError("Anima sampling invocation has already closed")

    def close(self):
        self.lllite.close()
        self.plan = None
