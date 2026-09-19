# Tiled Diffusion NG

Three ComfyUI nodes for SDXL tiled sampling, blending four overlapping views into
one full-resolution latent through a single KSampler trajectory.

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://git.colorized.life/tiled-diffusion-ng.git tiled-diffusion-ng
# Restart ComfyUI.
```

Requires Python 3.13+ and ComfyUI's V3 extension/wrapper APIs. No additional runtime
dependencies. Currently targets SDXL base models; real-host GPU validation is
pending. See [compatibility](docs/comfyui-compatibility.md) and
[validation status](docs/manual-validation.md).

## Nodes

| Display name | Node ID | Inputs → output |
| --- | --- | --- |
| Prepare Four Tile Plan | `TiledDiffusionNG_TilePlan` | `model`, `latent`, `tile_overlap` → TILE_PLAN |
| Extract Four Vision Views | `TiledDiffusionNG_TileView` | `image`, `tile_plan` → IMAGE batch |
| Sample Four Tiles | `TiledDiffusionNG_TileSampler` | KSampler inputs, `tile_plan`, optional `local_positive` → LATENT |

## Workflow

Connect the same model and target latent to the planner and sampler. The plan
creates four overlapping views in clockwise order: **TL, TR, BR, BL**. Overlap
is the shared width in pixels, default **64**. Gaussian weights blend predictions
at every model evaluation.

```mermaid
flowchart LR
    A[Model and Latent] --> B[Tile Plan] --> C["Tile View (optional)"] --> D[TileSampler]
```

Vision views are optional and require a reference image matching the plan's full
pixel dimensions. External nodes can turn those views into an execution list of
four complete positive conditionings in tile order. Each replaces the global
positive for its tile; the negative stays shared. Without local positives, all
four tiles use the global positive.

Upscale the output latent externally and create a fresh plan to refine again,
or decode it to an image.

## License

Copyright © 2026 Lany Atwood <lany@colorized.life>. The project's Python source
and tests are licensed under [AGPL-3.0-only](LICENSE).

Based on the Mixture of Diffusers equations by Álvaro Barbero Jiménez, with an
independently written Gaussian and fusion implementation. See
[COPYRIGHT](COPYRIGHT) for algorithm references and attribution.
