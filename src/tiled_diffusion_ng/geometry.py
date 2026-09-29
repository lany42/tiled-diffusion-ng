# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Immutable, tensor-free geometry shared by every spatial consumer."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

TILE_IDS = ("TL", "TR", "BR", "BL")
type Rect = tuple[int, int, int, int]
type HW = tuple[int, int]


@dataclass(frozen=True)
class GeometrySignature:
    """Model geometry in latent cells, with pixel-per-cell scales."""

    adapter_id: str
    adapter_version: int
    latent_format: str
    layout: str = "BCHW"
    rank: int = 4
    channels: int = 4
    spatial_axes: tuple[int, int] = (-2, -1)
    scale: HW = (8, 8)
    alignment: HW = (1, 1)
    minimum: HW = (1, 1)
    padding: str = "none"


@dataclass(frozen=True)
class LatentSpec:
    signature: GeometrySignature
    canvas: HW


@dataclass(frozen=True)
class TileRegion:
    """Half-open (x0, y0, x1, y1) bounds in latent and pixel coordinates."""

    index: int
    tile_id: str
    core: Rect
    sampling: Rect
    pixel_core: Rect
    pixel_sampling: Rect


@dataclass(frozen=True)
class TilePlanData:
    """Requested and padded (H, W) canvases; regions use the padded canvas."""

    signature: GeometrySignature
    latent_hw: HW
    pixel_hw: HW
    requested_overlap: int
    effective_overlap: HW
    effective_pixel_overlap: HW
    tile_hw: HW
    tile_pixel_hw: HW
    regions: tuple[TileRegion, ...]
    padded_latent_hw: HW
    padded_pixel_hw: HW
    schema_version: int = 2
    layout: str = "quadrants_2x2"
    tile_count: int = 4
    tile_ids: tuple[str, ...] = TILE_IDS
    kernel: str = "symmetric_axis_gaussian_v1"
    variance: float = 0.01
    center: str = "symmetric_cell_centers"


def _positive_pair(value, name):
    if (
        type(value) is not tuple
        or len(value) != 2
        or any(type(x) is not int or x < 1 for x in value)
    ):
        raise ValueError(f"{name} must contain two positive integers")


def make_plan(spec: LatentSpec, overlap: int) -> TilePlanData:
    if type(overlap) is not int or overlap < 0:
        raise ValueError("tile_overlap must be a nonnegative integer in pixels")
    s = spec.signature
    if type(s) is not GeometrySignature or s.spatial_axes != (-2, -1):
        raise ValueError("Unsupported geometry signature/spatial axes")
    for value, name in (
        (spec.canvas, "canvas"),
        (s.scale, "scale"),
        (s.alignment, "alignment"),
        (s.minimum, "minimum"),
    ):
        _positive_pair(value, name)
    h, w = spec.canvas
    sy, sx = s.scale
    if s.padding not in ("none", "circular"):
        raise ValueError("Unsupported canvas padding policy")
    if s.padding == "circular":
        h += -h % s.alignment[0]
        w += -w % s.alignment[1]

    # Our fixed-four coverage derivation: 2*t - length >= ceil(p/scale).
    # Round that lower bound and the minimum upward to the crop lattice.
    # Unlike the count-driven reference splitter, these are sampling extents,
    # not quadrant cores: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L68-L86
    def extent(length, scale, alignment, minimum):
        if length % alignment:
            raise ValueError(f"Canvas extent {length} violates alignment {alignment}")
        cells = (overlap + scale - 1) // scale
        tile = (
            max(
                (length + cells + 2 * alignment - 1) // (2 * alignment),
                (minimum + alignment - 1) // alignment,
            )
            * alignment
        )
        if not 0 < tile < length:
            raise ValueError("Canvas/overlap cannot produce four distinct valid views")
        return tile

    th, tw = (
        extent(n, scale, align, minimum)
        for n, scale, align, minimum in zip(
            (h, w), s.scale, s.alignment, s.minimum, strict=True
        )
    )
    samples = (
        (0, 0, tw, th),
        (w - tw, 0, w, th),
        (w - tw, h - th, w, h),
        (0, h - th, tw, h),
    )
    cores = (
        (0, 0, w // 2, h // 2),
        (w // 2, 0, w, h // 2),
        (w // 2, h // 2, w, h),
        (0, h // 2, w // 2, h),
    )

    def pixels(r):
        return (r[0] * sx, r[1] * sy, r[2] * sx, r[3] * sy)

    regions = tuple(
        TileRegion(i, name, core, sample, pixels(core), pixels(sample))
        for i, (name, core, sample) in enumerate(
            zip(TILE_IDS, cores, samples, strict=True)
        )
    )
    return TilePlanData(
        s,
        spec.canvas,
        (spec.canvas[0] * sy, spec.canvas[1] * sx),
        overlap,
        (2 * th - h, 2 * tw - w),
        ((2 * th - h) * sy, (2 * tw - w) * sx),
        (th, tw),
        (th * sy, tw * sx),
        regions,
        (h, w),
        (h * sy, w * sx),
    )


def validate_plan(plan, spec: LatentSpec | None = None) -> TilePlanData:
    if type(plan) is not TilePlanData or plan.schema_version != 2:
        raise ValueError("Expected TILE_PLAN schema version 2")
    try:
        expected = make_plan(
            LatentSpec(plan.signature, plan.latent_hw), plan.requested_overlap
        )
        if plan != expected:
            raise ValueError("TILE_PLAN fields or bounds do not match its geometry")
    except (TypeError, AttributeError) as exc:
        raise ValueError("Malformed TILE_PLAN geometry") from exc
    if spec is not None and (
        plan.signature != spec.signature or plan.latent_hw != spec.canvas
    ):
        raise ValueError(
            "Stale or incompatible TILE_PLAN; create a plan for this model and latent"
        )
    return plan


def crop(tensor: torch.Tensor, rect: Rect) -> torch.Tensor:
    x0, y0, x1, y1 = rect
    return tensor[..., y0:y1, x0:x1]


def pad_spatial(tensor: torch.Tensor, hw: HW) -> torch.Tensor:
    """Match native right/bottom patch padding, without importing host APIs."""
    if tuple(tensor.shape[-2:]) == hw:
        return tensor
    # ComfyUI's common_dit.pad_to_patch_size uses circular padding, falling
    # back to reflect while tracing/scripting. Leave time and channels intact.
    # https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/common_dit.py
    mode = (
        "reflect" if torch.jit.is_tracing() or torch.jit.is_scripting() else "circular"
    )
    padding = (0, hw[1] - tensor.shape[-1], 0, hw[0] - tensor.shape[-2])
    return F.pad(tensor, padding + (0, 0) * (tensor.ndim - 4), mode=mode)


def image_views(image: torch.Tensor, plan: TilePlanData) -> torch.Tensor:
    validate_plan(plan)
    if (
        not isinstance(image, torch.Tensor)
        or image.ndim != 4
        or image.shape[0] < 1
        or image.shape[-1] not in (3, 4)
    ):
        raise ValueError("image must be a nonempty BHWC RGB or RGBA tensor")
    if tuple(image.shape[1:3]) != plan.pixel_hw:
        raise ValueError(
            f"Expected image H×W {plan.pixel_hw}; received {tuple(image.shape[1:3])}"
        )
    image = pad_spatial(image.movedim(-1, 1), plan.padded_pixel_hw).movedim(1, -1)
    views = []
    for region in plan.regions:
        x0, y0, x1, y1 = region.pixel_sampling
        views.append(image[:, y0:y1, x0:x1, :])
    return torch.stack(views, dim=1).flatten(0, 1)
