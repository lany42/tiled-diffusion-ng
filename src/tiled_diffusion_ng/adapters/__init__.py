# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

"""Shared adapter contract and explicit model-family registry."""

from typing import Protocol

from ..geometry import LatentSpec
from .sdxl import SDXLAdapter


class LatentAdapter(Protocol):
    def accepts(self, model) -> bool: ...
    def describe(self, model, latent) -> LatentSpec: ...
    def validate_sampling(self, model, latent, plan) -> None: ...
    def validate_condition(self, metadata: dict) -> None: ...
    def adapt_spatial_condition(self, condition: dict, region) -> dict: ...


ADAPTERS: tuple[LatentAdapter, ...] = (SDXLAdapter(),)


def resolve_adapter(model) -> LatentAdapter:
    for adapter in ADAPTERS:
        if adapter.accepts(model):
            return adapter
    raise ValueError(
        "Unsupported model/latent family; this release accepts standard SDXL base image models"
    )
