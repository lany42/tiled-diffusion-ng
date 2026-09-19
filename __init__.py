# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""ComfyUI entry point for clone and ZIP installations."""

from .src.tiled_diffusion_ng import comfy_entrypoint

__all__ = ["comfy_entrypoint"]
