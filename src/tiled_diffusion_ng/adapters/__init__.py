# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Shared adapter contract and explicit model-family registry."""

from contextlib import AbstractContextManager
from typing import Protocol

from ..geometry import LatentSpec, TilePlanData, TileRegion
from .anima import AnimaAdapter
from .krea2 import Krea2Adapter
from .sdxl import SDXLAdapter


class SamplingContext(Protocol):
    """Spatial state owned by one invocation and closed on every exit.

    Discover all pairs before finalizing preparation for the host sampler.
    """

    def prepare_model(self, model, latent) -> None: ...
    def tile_options(
        self, options: dict, region: TileRegion
    ) -> AbstractContextManager[dict]: ...
    def prepare_pair(
        self, positive: list[dict], negative: list[dict], region: TileRegion
    ) -> tuple[list[dict], list[dict]]: ...
    def finalize_preparation(self) -> None: ...
    def close(self) -> None: ...


class LatentAdapter(Protocol):
    """Stateless model semantics with invocation-owned spatial preparation."""

    def accepts(self, model) -> bool: ...
    def describe(self, model, latent) -> LatentSpec: ...
    def validate_sampling(self, model, latent, plan) -> None: ...
    def validate_model_options(self, options: dict) -> None: ...
    def validate_condition(self, embedding, metadata: dict) -> None: ...
    def validate_evaluation(self, samples, plan: TilePlanData) -> None: ...
    def adapt_spatial_condition(self, condition: dict, region) -> dict: ...
    def create_sampling_context(self, plan: TilePlanData) -> SamplingContext: ...


ADAPTERS: tuple[LatentAdapter, ...] = (SDXLAdapter(), AnimaAdapter(), Krea2Adapter())


def resolve_adapter(model) -> LatentAdapter:
    for adapter in ADAPTERS:
        if adapter.accepts(model):
            return adapter
    raise ValueError(
        "Unsupported model/latent family; expected standard SDXL base or native Anima or Krea2 image models"
    )
