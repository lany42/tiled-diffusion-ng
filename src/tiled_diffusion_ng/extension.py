# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

from comfy_api.latest import ComfyExtension

from .nodes import (
    TiledAnimaLLLiteApply,
    TileKrea2Conditioning,
    TilePlan,
    TileSampler,
    TileView,
)


class TiledDiffusionNGExtension(ComfyExtension):
    async def get_node_list(self):
        return [
            TilePlan,
            TileView,
            TileSampler,
            TiledAnimaLLLiteApply,
            TileKrea2Conditioning,
        ]
