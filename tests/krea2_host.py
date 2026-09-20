# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Small Krea2 CPU doubles, independent of the Anima fixture path.

Contracts inspected at ComfyUI c194dd00cd42aa18d9dbf27d977bf6b85d9ea565:
comfy/model_base.py#L2724, comfy/conds.py, comfy/ldm/krea2/model.py#L283,
comfy/samplers.py#L214. This synthetic arithmetic is not Krea2 inference.
Raw, Turbo and reference conditioning each still need real-host acceptance
with native encoders/checkpoints, wrapper routing, V3 execution and seam review.
"""

import math

import torch


def cond(value, *, width=48, tokens=2, batch=1, **metadata):
    return [[torch.full((batch, tokens, width), float(value)), metadata]]


def zero_out(conditioning):
    # Native ConditioningZeroOut zeros embeddings/pooled_output, preserving
    # attention masks, references, reference methods and schedule metadata.
    # https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/nodes.py#L272-L298
    result = []
    for embedding, metadata in conditioning:
        metadata = metadata.copy()
        if metadata.get("pooled_output") is not None:
            metadata["pooled_output"] = torch.zeros_like(metadata["pooled_output"])
        result.append([torch.zeros_like(embedding), metadata])
    return result


def repeat_batch(value, batch):
    return value.repeat((math.ceil(batch / len(value)),) + (1,) * (value.ndim - 1))[
        :batch
    ]


class CONDRegular:
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, **kwargs):
        return type(self)(repeat_batch(self.cond, batch_size))

    def can_concat(self, other):
        return (
            self.cond.shape == other.cond.shape
            and self.cond.device == other.cond.device
        )

    def concat(self, others):
        return torch.cat([self.cond, *(other.cond for other in others)])


class CONDConstant(CONDRegular):
    def process_cond(self, batch_size, **kwargs):
        return type(self)(self.cond)

    def can_concat(self, other):
        return self.cond == other.cond

    def concat(self, others):
        return self.cond


class CONDList(CONDRegular):
    def process_cond(self, batch_size, **kwargs):
        return type(self)([repeat_batch(value, batch_size) for value in self.cond])

    def can_concat(self, other):
        return len(self.cond) == len(other.cond) and all(
            left.shape == right.shape for left, right in zip(self.cond, other.cond)
        )

    def concat(self, others):
        return [
            torch.cat([value, *(other.cond[index] for other in others)])
            for index, value in enumerate(self.cond)
        ]


class EmbedND:
    pass


class Krea2Network:
    channels = 16
    patch = 2
    txtlayers = 12
    txtdim = 4
    default_ref_method = None

    def __init__(self, num_blocks):
        self.blocks = [object() for _ in range(num_blocks)]
        self.pe_embedder = EmbedND()


def normalize_reference(value):
    # Deliberately non-identity channel statistics with native-shaped broadcasting.
    # These are not Wan21's real constants or an inference implementation.
    mean = torch.arange(16, device=value.device, dtype=value.dtype).reshape(
        1, 16, 1, 1, 1
    )
    return (value - mean * 0.01) / 2


class Krea2:
    concat_keys = ()

    def __init__(self, num_blocks):
        self.diffusion_model = Krea2Network(num_blocks)
        self.condition_calls = []

    def extra_conds(self, **kwargs):
        self.condition_calls.append(kwargs)
        result = {"c_crossattn": CONDRegular(kwargs["cross_attn"])}
        references = kwargs.get("reference_latents")
        if references is not None:
            result["ref_latents"] = CONDList(
                [normalize_reference(ref) for ref in references]
            )
            if kwargs.get("reference_latents_method") is not None:
                result["ref_latents_method"] = CONDConstant(
                    kwargs["reference_latents_method"]
                )
        return result


def leaf(host, model, branches, x, sigma, options):
    # Group using the native condition containers after batch processing. This
    # exercises references with different lengths/shapes/methods across entries.
    groups = []
    for branch_index, branch in enumerate(branches):
        for entry in branch or []:
            if (
                not entry.get("timestep_end", 0)
                <= sigma[0]
                <= entry.get("timestep_start", 1)
            ):
                continue
            conditions = {
                key: value.process_cond(x.shape[0])
                for key, value in entry["model_conds"].items()
            }
            for group in groups:
                first = group[0][2]
                if first.keys() == conditions.keys() and all(
                    value.can_concat(first[key]) for key, value in conditions.items()
                ):
                    group.append((branch_index, entry, conditions))
                    break
            else:
                groups.append([(branch_index, entry, conditions)])
    active = [[] for _ in branches]
    for group in groups:
        combined = {
            key: value.concat([item[2][key] for item in group[1:]])
            for key, value in group[0][2].items()
        }
        model_input = torch.cat([x] * len(group))
        combined_sigma = sigma.expand(x.shape[0]).repeat(len(group))
        transformer = dict(
            options["transformer_options"],
            sigmas=combined_sigma,
            cond_or_uncond=[item[0] for item in group],
        )
        prediction = host.call_wrappers(
            transformer,
            "apply_model",
            lambda *args: apply_model(host, *args),
            model,
            model_input,
            combined_sigma,
            combined,
            transformer,
        )
        for (index, entry, _), chunk in zip(
            group, prediction.split(x.shape[0]), strict=True
        ):
            active[index].append((chunk, entry.get("strength", 1)))
    return [
        sum(prediction * weight for prediction, weight in entries)
        / sum(weight for _, weight in entries)
        if entries
        else torch.zeros_like(x)
        for entries in active
    ]


def apply_model(host, model, x, sigma, conditions, transformer):
    native_sampling = host.latest_clone.sampling
    model_input = native_sampling.calculate_input(sigma, x)
    # Krea2's DIFFUSION_MODEL wrapper receives transformer_options positionally.
    prediction = host.call_wrappers(
        transformer,
        "diffusion_model",
        lambda *args, **kw: forward(host, *args, **kw),
        model_input,
        sigma,
        conditions["c_crossattn"],
        None,
        conditions.get("ref_latents"),
        transformer,
        ref_latents_method=conditions.get("ref_latents_method"),
    )
    return native_sampling.calculate_denoised(sigma, prediction.float(), x)


def forward(
    host, x, sigma, context, attention_mask, refs, transformer, *, ref_latents_method
):
    host.krea2_calls.append(
        (x.clone(), sigma.clone(), context, refs, ref_latents_method, transformer)
    )
    text = context.mean((1, 2)).to(x).reshape(-1, 1, 1, 1, 1)
    reference_value = sum(
        index * ref.mean((1, 2, 3, 4)).to(x).reshape(-1, 1, 1, 1, 1)
        for index, ref in enumerate(refs or [], 1)
    )
    return x * 0.25 + text + reference_value
