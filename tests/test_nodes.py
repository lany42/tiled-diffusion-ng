# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from tiled_diffusion_ng import comfy_entrypoint

from .host import isolated_host
from .test_sampling import arguments

ROOT = Path(__file__).resolve().parents[1]


def test_host_imports_and_registries_are_restored_between_runs():
    import tiled_diffusion_ng

    original_modules = {
        name: module
        for name, module in sys.modules.items()
        if name.startswith(("comfy.", "comfy_api."))
        or name
        in {
            "comfy",
            "comfy_api",
            "nodes",
            "tiled_diffusion_ng.nodes",
            "tiled_diffusion_ng.extension",
        }
    }
    original_attributes = dict(tiled_diffusion_ng.__dict__)
    results = []
    for _ in range(2):
        with isolated_host() as installed:
            from tiled_diffusion_ng.nodes import TileSampler

            assert installed.samplers.KSampler.SAMPLERS == ["euler", "heun"]
            installed.samplers.KSampler.SAMPLERS.append("temporary_registration")
            wrapped = {key: [value] for key, value in arguments().items()}
            results.append(TileSampler.execute(**wrapped)[0]["samples"])
        assert all(
            sys.modules.get(name) is module for name, module in original_modules.items()
        )
        for name in ("nodes", "extension"):
            assert tiled_diffusion_ng.__dict__.get(name) is original_attributes.get(
                name
            )
        assert not any(
            name not in original_modules
            and (
                name in {"comfy", "comfy_api"}
                or name.startswith(("comfy.", "comfy_api."))
            )
            for name in sys.modules
        )
    assert results[0].equal(results[1])


def test_three_node_extension_and_clone_loader(host, monkeypatch):
    extension = asyncio.run(comfy_entrypoint())
    classes = asyncio.run(extension.get_node_list())
    assert [c.define_schema().node_id for c in classes] == [
        "TiledDiffusionNG_TilePlan",
        "TiledDiffusionNG_TileView",
        "TiledDiffusionNG_TileSampler",
    ]
    loader = importlib.util.spec_from_file_location(
        "tdng_clone", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, "tdng_clone", module)
    loader.loader.exec_module(module)
    nodes = asyncio.run(asyncio.run(module.comfy_entrypoint()).get_node_list())
    assert len(nodes) == 3


def test_schema_defaults_and_singleton_execution_transport(host):
    from tiled_diffusion_ng.nodes import TilePlan, TileSampler, TileView

    plan_schema = TilePlan.define_schema()
    assert len(plan_schema.outputs) == 1
    overlap = {i.id: i for i in plan_schema.inputs}["tile_overlap"]
    assert (overlap.default, overlap.min, overlap.step) == (64, 0, 8)
    assert [i.id for i in TileView.define_schema().inputs] == ["image", "tile_plan"]
    schema = TileSampler.define_schema()
    assert schema.is_input_list
    inputs = {i.id: i for i in schema.inputs}
    assert (inputs["seed"].default, inputs["seed"].min, inputs["seed"].max) == (
        0,
        0,
        2**64 - 1,
    )
    assert inputs["seed"].control_after_generate
    assert (inputs["steps"].default, inputs["steps"].min, inputs["steps"].max) == (
        20,
        1,
        10000,
    )
    assert (inputs["cfg"].default, inputs["cfg"].step) == (8, 0.1)
    assert (
        inputs["denoise"].default,
        inputs["denoise"].min,
        inputs["denoise"].max,
    ) == (1, 0, 1)
    assert inputs["local_positive"].optional
    args = arguments()
    planned = TilePlan.execute(args["model"], args["latent_image"], 16)
    assert len(planned) == 1 and planned[0] == args["tile_plan"]
    args["positive"] *= 4  # four entries still comprise ONE global CONDITIONING
    wrapped = {key: [value] for key, value in args.items()}
    result = TileSampler.execute(**wrapped)
    assert result[0]["samples"].shape == args["latent_image"]["samples"].shape
    assert all(len(call[0][0]) == 4 for call in host.tile_calls)
    for key, value in wrapped.items():
        malformed = {**wrapped, key: value * 2}
        with pytest.raises(ValueError, match=key):
            TileSampler.execute(**malformed)
