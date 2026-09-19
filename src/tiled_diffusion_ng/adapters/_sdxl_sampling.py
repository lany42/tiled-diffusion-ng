# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""SDXL architecture requirements for native image-hint ControlNets."""

from ._native_control import NativeControlContext


def _validate_network(network):
    # Traditional and Union SDXL controls use sequential ADM with 2816 inputs.
    # Inspect architecture attributes without loading parameter values.
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


class SDXLSamplingContext(NativeControlContext):
    """Supply SDXL validation to the native control lifecycle."""

    def __init__(self, plan):
        super().__init__(plan, _validate_network)
