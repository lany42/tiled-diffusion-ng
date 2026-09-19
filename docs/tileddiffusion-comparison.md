# Comparison with ComfyUI-TiledDiffusion

This comparison retains the source baseline at
[`a155b1bac39147381aeaa52b9be42e545626a44f`][upstream]. Both implementations
combine overlapping predictions while ComfyUI advances a global sample.
Tiled Diffusion NG is an independent, narrower implementation of Mixture of
Diffusers. Real-host and visual comparisons remain [pending](manual-validation.md).

## Geometry, workflow and integration

| Topic | Pinned upstream | Tiled Diffusion NG |
| --- | --- | --- |
| Geometry | [Tile dimensions and overlap][splitter] determine the tile count; order is row-major TL/TR/BL/BR for a 2×2 grid. | Exactly four equal sampling views, clockwise TL/TR/BR/BL. The model geometry and requested shared pixel overlap determine extents; tiles grow with the canvas. |
| Workflow interface | The [basic node][node] returns a patched MODEL for a downstream sampler. | TilePlan reads MODEL/LATENT and returns geometry only; TileView returns overlapping image crops; TileSampler returns an ordinary LATENT from one KSampler call. |
| Conditioning | The basic shared-prompt setup is the global comparison reference. | Optional four complete local positives replace the global positive per tile, including controls; negative conditioning and CFG stay shared. Distinct locals need their own routing check. |
| Host integration | [Global function replacements][upstream-utils] and a UNet wrapper adapt sampling. | Wrappers attach to a private model clone. Pairing, tagged full-canvas preparation and continuation routing depend on host contracts, including V3 execution-list transport. |
| Supported breadth | [The package][upstream] includes more model families, diffusion methods, tile batching and tiled VAE. | Standard SDXL base image sampling, native EPS/V conversions, and ordinary RGB SDXL ControlNet. Other spatial capabilities require explicit handlers; [Anima/Krea2 remain research](model-adapter-research.md). |

Both pixel and latent upscale workflows compose through ordinary external nodes.
This project's iterative stages use fresh plans when geometry changes and one
trajectory per stage. VAE, upscaling, text encoding and VLM calls remain external.
This interface difference does not imply that upstream requires pixel upscaling.
For a fair comparison, match actual sampling boxes and effective overlap; the
[portrait settings](manual-validation.md#reproducible-comparison) produce four
views at overlaps 64 and 128. Nominal quadrant cores are smaller than those
sampling boxes.

## Gaussian differences

The [paper's centered Gaussian][paper] uses distances normalized independently
by width and height. These inspected kernels differ in their discrete vertical
center and scale; all use horizontal center `(w-1)/2` and scale `w²`:

| Kernel | Vertical center | Vertical squared scale |
| --- | --- | --- |
| Author's [canvas implementation][canvas] | `(h-1)/2` | `h²` |
| Author's [tiling implementation][tiling] | `h/2` | `h²` |
| Pinned [TiledDiffusion kernel][gaussian] | `h/2` | `w²` |
| [PR #77][correction], head `1cac48dfcb1c5a17d15c3112a00e5b508dc90ad2` | `h/2` | `h²` |
| This project's [kernel](../src/tiled_diffusion_ng/fusion.py) | `(h-1)/2` | `h²` |

PR #77's inspected patch corrects only the denominator. This project uses both
symmetric cell centers and independent axis scales, matching the author's canvas
implementation. Square tiles can differ from stock upstream because of the center;
rectangular tiles also expose the scale difference. The [mathematical reference](mathematics.md)
explains normalization and precision. There is no legacy-kernel option; an
isolated corrected upstream checkout is an optional comparison reference.

## Performance limits

Four sequential views reduce each denoiser's spatial extent, but every stage
still retains full-canvas latents, noise, branch accumulators and a denominator,
as well as tile activations and control caches. Fixed tile count cannot promise
lower VRAM for arbitrarily large canvases. Host batching within a tile does not
provide this project's own tile batching.

Hint sharing reduces duplicate retained unprepared image storage while keeping
clone-local prepared caches. It does not establish a measured speed, memory or
compatibility advantage over upstream. For any measurement, match inputs,
settings, checkpoints, precision, device/backend and warmup. Report CPU/CUDA
allocated peaks and the method separately from process RSS and CUDA reserved
memory. Caller-owned images, weights, activations, prepared caches, resize
temporaries and allocator reserves are outside the retained hint payload.
No measured GPU savings or general speedup is claimed before real-host results.

[upstream]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/tree/a155b1bac39147381aeaa52b9be42e545626a44f
[splitter]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L68-L86
[node]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L814-L883
[upstream-utils]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/utils.py#L14-L103
[paper]: https://arxiv.org/html/2302.02412v1#S3
[canvas]: https://github.com/albarji/mixture-of-diffusers/blob/af42292d0a8cb414f6da2eeac79be4c60afbbe48/mixdiff/canvas.py#L187-L200
[tiling]: https://github.com/albarji/mixture-of-diffusers/blob/af42292d0a8cb414f6da2eeac79be4c60afbbe48/mixdiff/tiling.py#L174-L229
[gaussian]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L450-L461
[correction]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/pull/77
