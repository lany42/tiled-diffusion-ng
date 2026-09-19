# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import pytest
import torch

from tiled_diffusion_ng import _comfy_sampling as sampling
from tiled_diffusion_ng.adapters import resolve_adapter
from tiled_diffusion_ng.geometry import GeometrySignature, LatentSpec, make_plan

from .host import ControlNet, cond
from .test_sampling import arguments


def test_shared_evaluation_delegates_embedding_and_layout_semantics(host):
    # A synthetic adapter contract, not an additional supported model family.
    plan = make_plan(
        LatentSpec(
            GeometrySignature(
                "synthetic",
                1,
                "fixture",
                layout="BTCHW",
                rank=5,
                channels=7,
                scale=(1, 1),
            ),
            (5, 9),
        ),
        2,
    )
    x = torch.arange(2 * 3 * 7 * 5 * 9, dtype=torch.float64).reshape(2, 3, 7, 5, 9)
    validations = []

    class Adapter:
        def validate_condition(self, embedding, metadata):
            assert embedding == {"tokens": (1, 2)}
            validations.append("embedding")

        def validate_evaluation(self, samples, tile_plan):
            assert samples.shape == (2, 3, 7, *tile_plan.latent_hw)
            validations.append("layout")

        def adapt_spatial_condition(self, condition, region):
            return {**condition, "adapter_region": region.tile_id}

    adapter = Adapter()
    sampling.validate_conditioning([[{"tokens": (1, 2)}, {}]], adapter, "positive")
    evaluation = sampling.TileEvaluation(plan, adapter)
    regions = []

    def next_wrapper(model, branches, tile, sigma, options):
        regions.append(branches[0][0]["adapter_region"])
        return [tile * 0.5 + 2, torch.zeros_like(tile)]

    branches = [[{sampling.TAG: region.tile_id} for region in plan.regions], None]
    try:
        output = evaluation(next_wrapper, None, branches, x, torch.tensor([0.5]), {})
    finally:
        evaluation.close()
    torch.testing.assert_close(output[0], x * 0.5 + 2)
    assert validations == ["embedding", "layout"]
    assert regions == ["TL", "TR", "BR", "BL"]


@pytest.mark.parametrize(
    "failure", [None, "prepare", "finalize", "model", "cancel", "cleanup"]
)
def test_adapter_context_is_per_invocation_and_always_closed(
    host, monkeypatch, failure
):
    control = ControlNet()
    args = arguments(positive=cond(1, control=control, control_apply_to_uncond=True))
    adapter = resolve_adapter(args["model"])
    create = adapter.create_sampling_context
    contexts = []

    class Context:
        def __init__(self, plan):
            self.inner = create(plan)
            self.regions = []
            self.finalized = 0
            self.closed = 0

        def prepare_pair(self, positive, negative, region):
            self.regions.append(region.tile_id)
            # Host propagation happened before adapter-specific preparation.
            assert positive[0]["control"] is negative[0]["control"]
            assert sampling.TAG not in positive[0]
            if failure == "prepare":
                raise RuntimeError("context preparation failure")
            return self.inner.prepare_pair(positive, negative, region)

        def finalize_preparation(self):
            assert self.regions == ["TL", "TR", "BR", "BL"]
            assert not host.common_calls
            self.finalized += 1
            if failure == "finalize":
                raise RuntimeError("context finalization failure")
            self.inner.finalize_preparation()

        def close(self):
            self.closed += 1
            self.inner.close()
            if failure == "cleanup":
                raise RuntimeError("context cleanup failure")

    def create_context(plan):
        context = Context(plan)
        contexts.append(context)
        return context

    monkeypatch.setattr(adapter, "create_sampling_context", create_context)
    if failure == "model":
        host.fail_tile = 2
    elif failure == "cancel":
        host.interrupt_after = 2
    if failure is None:
        first = sampling.sample(**args)
        host.common_calls.clear()
        second = sampling.sample(**args)
        torch.testing.assert_close(first["samples"], second["samples"])
        assert len(contexts) == 2 and contexts[0] is not contexts[1]
    else:
        with pytest.raises((RuntimeError, InterruptedError)):
            sampling.sample(**args)
    for context in contexts:
        assert context.closed == 1
        assert context.inner.plan is None and not context.inner.controls
        assert not context.inner.hint_groups
        assert context.finalized == (failure != "prepare")
        if failure != "prepare":
            assert context.regions == ["TL", "TR", "BR", "BL"]
    if host.latest_clone is not None:
        for kind in ("predict_noise", "calc_cond_batch"):
            assert not host.latest_clone.get_wrappers(kind, sampling.WRAPPER_KEY)
    assert args["model"].wrappers == {}
    assert control.cond_hint_original is not None and control.cleanups == 0
