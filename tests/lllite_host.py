# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Small CPU doubles for native LLLite boundaries, not native inference.

ComfyUI 944386c233e02eaf877b1c8d5d513fb3d3a4d5e3 supplies the contract:
comfy/ldm/anima/lllite.py and comfy/ldm/cosmos/predict2.py. Encoding and
residual math below are deliberately simple and project-written.
"""

import re
from types import SimpleNamespace

import torch
import torch.nn.functional as F

MODULE_PATTERN = re.compile(
    r"lllite_dit_blocks_(\d+)_(self_attn_[qkv]_proj|cross_attn_q_proj|mlp_layer1)$"
)
TARGETS = (
    "self_attn_q_proj",
    "self_attn_k_proj",
    "self_attn_v_proj",
    "cross_attn_q_proj",
    "mlp_layer1",
)


class AnimaLLLite(torch.nn.Module):
    def __init__(self, width=4, channels=3, targets=None):
        super().__init__()
        self.model_dim = width
        self.cond_in_channels = channels
        self.module_names = set(
            targets
            if targets is not None
            else (f"lllite_dit_blocks_0_{target}" for target in TARGETS)
        )
        self.block_count = max(
            (int(name.split("_")[3]) + 1 for name in self.module_names), default=0
        )
        for name in self.module_names:
            module = torch.nn.Module()
            module.down = torch.nn.Linear(width, 1)
            module.up = torch.nn.Linear(1, width)
            self.add_module(name, module)
        self.encodings = []
        self.calls = []

    def encode_conditioning(self, image):
        self.encodings.append(image.clone())
        return F.avg_pool2d(image.mean(1, keepdim=True), 16).flatten(2).transpose(1, 2)

    def apply(self, x, embedding, block_index, target, strength):
        if f"lllite_dit_blocks_{block_index}_{target}" not in self.module_names:
            return x
        shape = x.shape
        tokens = x.flatten(1, 3) if x.ndim == 5 else x
        assert tokens.shape[0] % embedding.shape[0] == 0
        repeated = embedding.repeat(tokens.shape[0] // embedding.shape[0], 1, 1)
        assert repeated.shape[:2] == tokens.shape[:2]
        self.calls.append((block_index, target, repeated.clone(), strength))
        return (tokens + repeated * strength / 10).reshape(shape)


class AnimaLLLitePatch:
    __module__ = "comfy.ldm.anima.lllite"

    def __init__(self, model_patch, image, mask, strength, sigma_start, sigma_end):
        self.model_patch, self.image, self.mask = model_patch, image, mask
        self.strength, self.sigma_start, self.sigma_end = (
            strength,
            sigma_start,
            sigma_end,
        )

    def __call__(self, args):
        from comfy.utils import common_upscale

        options, x = args["transformer_options"], args["x"]
        sigmas = options.get("sigmas")
        if self.strength == 0 or (
            sigmas is not None
            and not self.sigma_end <= float(sigmas.max()) <= self.sigma_start
        ):
            return args
        assert x.shape[2] == 1 and self.mask is None
        image = common_upscale(
            self.image.movedim(-1, 1),
            x.shape[-1] * 8,
            x.shape[-2] * 8,
            "bicubic",
            "center",
        )
        prepared = image.clamp(0, 1).to(x) * 2 - 1
        options["model_patch_data"][self] = self.model_patch.model.encode_conditioning(
            prepared
        )
        return args

    def models(self):
        return [self.model_patch]

    def to(self, device_or_dtype):
        return self


class AnimaLLLiteAttentionPatch:
    __module__ = "comfy.ldm.anima.lllite"

    def __init__(self, patch, targets):
        self.patch, self.targets = patch, targets

    def __call__(self, q, k, v, pe=None, attn_mask=None, extra_options=None):
        result = {"q": q, "k": k, "v": v, "pe": pe, "attn_mask": attn_mask}
        embedding = extra_options["model_patch_data"].get(self.patch)
        if embedding is not None:
            for name, target in self.targets.items():
                result[name] = self.patch.model_patch.model.apply(
                    result[name],
                    embedding,
                    extra_options["block_index"],
                    target,
                    self.patch.strength,
                )
        return result


class AnimaLLLiteMLPPatch:
    __module__ = "comfy.ldm.anima.lllite"

    def __init__(self, patch):
        self.patch = patch

    def __call__(self, args):
        options = args["transformer_options"]
        embedding = options["model_patch_data"].get(self.patch)
        if embedding is not None:
            args["x"] = self.patch.model_patch.model.apply(
                args["x"],
                embedding,
                options["block_index"],
                "mlp_layer1",
                self.patch.strength,
            )
        return args


def forward_hooks(host, model, x, transformer):
    """Reset per-forward data and exercise native hook slots before projection."""
    patches = transformer.get("patches", {})
    if "post_input" not in patches:
        return torch.zeros_like(x)
    local = {**transformer, "model_patch_data": {}}
    b, _, _, h, w = x.shape
    img = torch.zeros(
        b, 1, h // 2, w // 2, model.diffusion_model.model_channels, dtype=x.dtype
    )
    for patch in patches["post_input"]:
        img = patch({"x": x, "img": img, "transformer_options": local})["img"]
    record = SimpleNamespace(
        options=local,
        x=x.clone(),
        keys=tuple(local["model_patch_data"]),
        embedding=[v.clone() for v in local["model_patch_data"].values()],
        cross_kv=[],
    )
    host.lllite_forwards.append(record)
    delta = torch.zeros_like(img)
    for index in range(len(model.diffusion_model.blocks)):
        local["block_index"] = index
        for slot in ("attn1_patch", "attn2_patch"):
            # Text K/V have a different sequence length and must stay untouched.
            kv = img if slot == "attn1_patch" else torch.ones(b, 3, img.shape[-1])
            for patch in patches.get(slot, []):
                out = patch(img, kv, kv, extra_options=local)
                delta += out["q"] - img
                if slot == "attn1_patch":
                    delta += out["k"] - img + out["v"] - img
                else:
                    record.cross_kv.append((out["k"] is kv, out["v"] is kv))
        for patch in patches.get("mlp_patch", []):
            delta += patch({"x": img, "transformer_options": local})["x"] - img
    delta = delta.mean(-1).repeat_interleave(2, -1).repeat_interleave(2, -2)
    return delta.unsqueeze(1).expand_as(x)
