# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Four-view diffusion. Pure geometry remains importable without ComfyUI."""


async def comfy_entrypoint():
    from .extension import TiledDiffusionNGExtension

    return TiledDiffusionNGExtension()


__all__ = ["comfy_entrypoint"]
