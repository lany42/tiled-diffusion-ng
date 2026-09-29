# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>

import asyncio
import importlib.util
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from tiled_diffusion_ng import comfy_entrypoint

from . import krea2_conditioning_host as encoder_host
from .host import arguments, cond

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("archive", [False, True])
def test_extension_and_clone_loader(host, monkeypatch, tmp_path, archive):
    extension = asyncio.run(comfy_entrypoint())
    classes = asyncio.run(extension.get_node_list())
    assert [c.define_schema().node_id for c in classes] == [
        "TiledDiffusionNG_TilePlan",
        "TiledDiffusionNG_TileView",
        "TiledDiffusionNG_TileSampler",
        "TiledDiffusionNG_TiledAnimaLLLiteApply",
        "TiledDiffusionNG_TileKrea2Conditioning",
    ]
    root = ROOT
    if archive:
        path = tmp_path / "source.zip"
        with zipfile.ZipFile(path, "w") as zipped:
            for source in [ROOT / "__init__.py", *(ROOT / "src").rglob("*.py")]:
                zipped.write(source, source.relative_to(ROOT))
        root = tmp_path / "extracted"
        with zipfile.ZipFile(path) as zipped:
            zipped.extractall(root)
    loader = importlib.util.spec_from_file_location(
        "tdng_clone", root / "__init__.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(loader)
    monkeypatch.setitem(sys.modules, "tdng_clone", module)
    loader.loader.exec_module(module)
    nodes = asyncio.run(asyncio.run(module.comfy_entrypoint()).get_node_list())
    assert [node.define_schema().node_id for node in nodes] == [
        node.define_schema().node_id for node in classes
    ]


def test_saved_workflow_bindings(host):
    from tiled_diffusion_ng.nodes import (
        TiledAnimaLLLiteApply,
        TileKrea2Conditioning,
        TilePlan,
        TileSampler,
        TileView,
    )

    # Saved workflows bind inputs by name and order; list flags pick transport.
    expected = {
        TilePlan: (["model", "latent", "tile_overlap"], False),
        TileView: (["image", "tile_plan"], False),
        TileSampler: (
            [
                "model",
                "seed",
                "steps",
                "cfg",
                "sampler_name",
                "scheduler",
                "positive",
                "negative",
                "latent_image",
                "denoise",
                "tile_plan",
                "local_positive",
            ],
            True,
        ),
        TiledAnimaLLLiteApply: (
            [
                "model",
                "model_patch",
                "tile_plan",
                "reference_tiles",
                "strength",
                "start_percent",
                "end_percent",
            ],
            False,
        ),
        TileKrea2Conditioning: (
            [
                "clip",
                "reference_tiles",
                "prompts",
                "strength",
                "end_percent",
                "downsize_to_1mp",
                "baseline",
            ],
            True,
        ),
    }
    schemas = {}
    for node, (inputs, input_list) in expected.items():
        schema = schemas[node] = node.define_schema()
        assert [item.id for item in schema.inputs] == inputs
        assert bool(getattr(schema, "is_input_list", False)) is input_list
        assert len(schema.outputs) == 1
    optional = {
        (node, item.id)
        for node, schema in schemas.items()
        for item in schema.inputs
        if getattr(item, "optional", False)
    }
    assert optional == {
        (TileSampler, "local_positive"),
        (TileKrea2Conditioning, "prompts"),
        (TileKrea2Conditioning, "baseline"),
    }
    # The seed's control widget occupies a saved widget-value position.
    sampler = {item.id: item for item in schemas[TileSampler].inputs}
    assert sampler["seed"].control_after_generate
    krea2 = {item.id: item for item in schemas[TileKrea2Conditioning].inputs}
    for name in ("prompts", "baseline"):
        assert krea2[name].force_input and krea2[name].dynamic_prompts is False
    assert schemas[TileKrea2Conditioning].outputs[0].is_output_list


def test_plan_and_sampler_nodes_use_singleton_execution_lists(host):
    from tiled_diffusion_ng.nodes import TilePlan, TileSampler

    args = arguments()
    planned = TilePlan.execute(args["model"], args["latent_image"], 16)
    assert len(planned) == 1 and planned[0] == args["tile_plan"]
    args["positive"] *= 4  # four entries still comprise ONE global CONDITIONING
    wrapped = {key: [value] for key, value in args.items()}
    result = TileSampler.execute(**wrapped)
    assert result[0]["samples"].shape == args["latent_image"]["samples"].shape
    assert all(len(call[0][0]) == 4 for call in host.tile_calls)
    locals = [cond(value) for value in (3, 5, 7, 11)]
    TileSampler.execute(**wrapped, local_positive=locals)
    assert [
        embedding.mean().item() for embedding, _ in host.common_calls[-1]["positive"]
    ] == [3, 5, 7, 11]
    for key, value in wrapped.items():
        with pytest.raises(ValueError, match=key):
            TileSampler.execute(**{**wrapped, key: value * 2})


def test_krea2_conditioning_node_uses_singleton_execution_lists(host, monkeypatch):
    from tiled_diffusion_ng.nodes import TileKrea2Conditioning

    encoder_host.install(monkeypatch)
    clip = encoder_host.Clip()
    args = {
        "clip": [clip],
        "reference_tiles": [torch.full((4, 6, 10, 3), 0.5)],
        "strength": [1.0],
        "end_percent": [1.0],
        "downsize_to_1mp": [False],
    }
    output = TileKrea2Conditioning.execute(**args)
    assert len(output) == 1 and len(output[0]) == 4
    for key, value in args.items():
        for bad in (value * 2, value[0]):
            with pytest.raises(ValueError, match=key + " requires one execution-list"):
                TileKrea2Conditioning.execute(**{**args, key: bad})
    assert len(clip.tokenized) == 4


def test_lllite_apply_node_returns_patched_clone(host):
    from tiled_diffusion_ng.adapters._anima_lllite import get_attachment
    from tiled_diffusion_ng.nodes import TiledAnimaLLLiteApply

    from .test_anima_lllite import patched

    args, source, refs, patch = patched()
    result = TiledAnimaLLLiteApply.execute(source, patch, args["tile_plan"], refs)
    assert len(result) == 1 and result[0] is not source
    assert get_attachment(result[0]).config.reference_tiles is refs
    assert get_attachment(source) is None
