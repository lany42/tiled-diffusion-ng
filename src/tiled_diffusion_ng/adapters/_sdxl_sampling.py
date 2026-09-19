# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Ordinary SDXL image-hint ControlNet support; no sampler or solver logic."""

import inspect
import logging
from dataclasses import dataclass, field

import torch

from ..geometry import HW, Rect, crop

logger = logging.getLogger(__name__)


def _hint_key(hint, algorithm, crop_policy, pixel_hw):
    # Tensor identity differs between apply nodes even for the same image view.
    # The group keeps the source alive, preventing storage pointer reuse.
    return (
        hint.untyped_storage().data_ptr(),
        hint.device,
        hint.storage_offset(),
        tuple(hint.shape),
        hint.stride(),
        hint.dtype,
        hint.is_conj(),
        hint.is_neg(),
        algorithm,
        crop_policy,
        pixel_hw,
    )


@dataclass
class _HintGroup:
    source: torch.Tensor | None
    algorithm: str
    pixel_hw: HW
    crop_policy: str = "center"
    controls: dict[Rect, list] = field(default_factory=dict)

    def materialize(self):
        from comfy import utils

        canvas = hint = None
        try:
            ph, pw = self.pixel_hw
            # Always normalize, even at matching dimensions: host resize modes
            # can transform pixels. Preserve full-canvas resize BEFORE cropping.
            canvas = utils.common_upscale(
                self.source, pw, ph, self.algorithm, self.crop_policy
            )
            crop_bytes = (
                canvas.shape[0]
                * canvas.shape[1]
                * canvas.element_size()
                * sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in self.controls)
            )
            compact = crop_bytes < canvas.untyped_storage().nbytes()
            for rect, controls in self.controls.items():
                hint = crop(canvas, rect)
                if compact:
                    hint = hint.clone()
                # Unprepared pixels are read-only. Host preparation still owns
                # each clone's separate mutable cond_hint/device cache.
                for control in controls:
                    control.cond_hint_original = hint
        finally:
            # Do not retain unused canvases between groups, even in a traceback.
            canvas = hint = None
            self.source = None
            self.controls.clear()


class SDXLSamplingContext:
    """SDXL spatial preparation and caches owned by one sampler invocation."""

    def __init__(self, plan):
        self.plan = plan
        self.controls = []
        self.hint_groups = {}

    def prepare_pair(self, positive, negative, region):
        memo = {}
        pair = []
        for entries in (positive, negative):
            prepared = []
            for metadata in entries:
                metadata = metadata.copy()
                if metadata.get("control") is not None:
                    metadata["control"] = self._clone_control(
                        metadata["control"], region, memo, set()
                    )
                prepared.append(metadata)
            pair.append(prepared)
        return pair[0], pair[1]

    def _clone_control(self, original, region, memo, visiting):
        from comfy import controlnet
        from comfy.cldm.cldm import ControlNet as ControlNetwork

        if original is None:
            return None
        key = id(original)
        if key in visiting:
            raise ValueError("Cyclic previous_controlnet chain")
        if key in memo:
            return memo[key]
        if (
            type(original) is not controlnet.ControlNet
            or type(original.control_model) is not ControlNetwork
        ):
            raise ValueError(
                "Unsupported control: only ordinary SDXL image-hint ControlNet is supported"
            )
        network = original.control_model
        if hasattr(network, "num_control_type"):
            raise ValueError("Unsupported ControlNet capability: Union control types")
        # Ordinary SDXL controls use sequential ADM with 2816 inputs. Checking
        # architecture attributes does not load or inspect parameter values.
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
        for capability in (
            "vae",
            "latent_format",
            "concat_mask",
            "extra_concat_orig",
            "extra_hooks",
            "multigpu_clones",
        ):
            if getattr(original, capability, None):
                raise ValueError(f"Unsupported ControlNet capability: {capability}")
        if original.compression_ratio != 8 or set(original.extra_conds) - {"y"}:
            raise ValueError("Unsupported ControlNet compression_ratio/extra_conds")
        default_preprocess = (
            inspect.signature(controlnet.ControlNet)
            .parameters["preprocess_image"]
            .default
        )
        if original.preprocess_image is not default_preprocess:
            raise ValueError(
                "Unsupported ControlNet preprocess_image; requires an explicit full-canvas handler"
            )
        if any(
            not isinstance(value, (str, int, float, bool, type(None)))
            for value in original.extra_args.values()
        ):
            raise ValueError("Unsupported ControlNet spatial extra_args")
        hint = original.cond_hint_original
        if (
            not isinstance(hint, torch.Tensor)
            or hint.ndim != 4
            or hint.shape[1] != 3
            or min(hint.shape) < 1
        ):
            raise ValueError(
                "ControlNet hint must be a nonempty BCHW RGB tensor in full-canvas coordinates"
            )

        visiting.add(key)
        clone = original.copy()
        if clone is original:
            raise ValueError("ControlNet.copy() must return an independent control")
        self.controls.append(clone)
        memo[key] = clone
        clone.previous_controlnet = None
        clone.cond_hint = None
        clone.timestep_range = None
        clone.extra_concat = None
        clone.model_sampling_current = None
        clone.cond_hint_original = None
        # Discover all required rectangles before choosing their backing store.
        # Hint pixels depend on the source view and resize policy, not on the
        # independent control networks, strengths or timestep schedules.
        # https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/controlnet.py#L269-L303
        hint_key = _hint_key(
            hint, original.upscale_algorithm, "center", self.plan.pixel_hw
        )
        if hint_key not in self.hint_groups:
            self.hint_groups[hint_key] = _HintGroup(
                hint, original.upscale_algorithm, self.plan.pixel_hw
            )
        self.hint_groups[hint_key].controls.setdefault(
            region.pixel_sampling, []
        ).append(clone)
        clone.previous_controlnet = self._clone_control(
            original.previous_controlnet, region, memo, visiting
        )
        visiting.remove(key)
        return clone

    def finalize_preparation(self):
        try:
            while self.hint_groups:
                key = next(iter(self.hint_groups))
                self.hint_groups.pop(key).materialize()
        finally:
            self.hint_groups.clear()

    def close(self):
        # Sever clone-only chains before cleanup to visit each clone exactly once.
        # Host cleanup is idempotent; its success path may have run already.
        for control in self.controls:
            control.previous_controlnet = None
        for control in self.controls:
            try:
                control.cleanup()
            except Exception:
                logger.exception("Tiled Diffusion NG ControlNet cleanup failed")
            finally:
                control.cond_hint_original = None
                control.cond_hint = None
                control.timestep_range = None
                control.extra_concat = None
                control.model_sampling_current = None
        self.controls.clear()
        self.hint_groups.clear()
        self.plan = None
