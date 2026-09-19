# Model adapter research

**Unimplemented:** Anima, Krea2 Raw and Krea2 Turbo remain follow-on work after
[SDXL manual acceptance](manual-validation.md). The sources below are research
starting points at ComfyUI
[`3c80da7f87ee359b2d06f107cb3c0797079dfbbb`][comfy], not support claims.

## Pinned source map

| Family / boundary | Source observation | Unresolved work |
| --- | --- | --- |
| Latent metadata and host preparation | [Supported models][supported] select `Wan21` for Anima and Krea2; [latent formats][latents] declare 16 channels and three latent dimensions. [Sample preparation][sample] can change empty-latent channels, size or rank. | Establish actual stored and evaluated 4D/5D image layouts, spatial axes, singleton temporal handling and pixel scales. Planning must match the evaluation tensor without silently coercing it. |
| Anima | The [model wrapper][anima] uses flow sampling and model-specific text preparation; its [network][anima-network] builds on [Cosmos Predict2][cosmos]. | Verify patch alignment/minimum extents, conditioning preparation, spatial positions, timestep semantics and denoised-output conversion. |
| Krea2 Raw | The [wrapper][krea] prepares `reference_latents`; the [network][krea-network] patchifies images and builds position IDs. | Verify patch grid, image layout, reference roles, positional offsets and the prediction conversion before evaluating refinement. |
| Krea2 Turbo | The same integration family is only a starting point. | Validate its own checkpoint, native scheduler, steps, CFG and refinement behavior; Raw acceptance would not establish Turbo support. |

## Questions for an adapter

- **Layout and patch grid:** which axes are spatial at storage and model-call
  boundaries, and which crop origins/extents are valid? Latent scale alone does
  not establish a transformer patch lattice. Preserve batch and temporal axes;
  admitting a singleton frame must not imply video support.
- **Positions:** should each view reset coordinates or retain global offsets?
  Inspect position-ID construction and compare the chosen policy in the host.
  SDXL translation behavior is not sufficient evidence for a transformer.
- **Reference conditioning:** is each reference registered to the target canvas
  or an independent image? Preserve or transform `reference_latents` according
  to their meaning; do not crop every entry merely because it is a tensor.
- **Prediction conversion:** does the host return a shared affine function of
  the current latent and model output at each timestep? Reuse verified host
  conversion and check the [fusion identity](mathematics.md#affine-prediction-conversion)
  before admitting a flow/velocity path.

Keep model semantics behind the [adapter contract](../src/tiled_diffusion_ng/adapters/__init__.py)
and create spatial state per invocation. Share geometry, Gaussian fusion,
clockwise routing and the sampling lifecycle; extend layout helpers only after
verifying the new contract. The current registry contains SDXL alone.

[Upstream PR #62][rank-proposal], inspected head
`265d5a4c33ee9e70cd9e569f2965df66368cb269`, is a reference for preserving leading
axes when cropping and broadcasting spatial weights. It does not demonstrate
compatibility with these models. Video and temporal tiling remain outside scope.
External VLM workflows, tile batching and configurable layouts remain separate
later work; none is implemented by retaining this research.

[comfy]: https://github.com/Comfy-Org/ComfyUI/tree/3c80da7f87ee359b2d06f107cb3c0797079dfbbb
[supported]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/supported_models.py
[latents]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/latent_formats.py
[sample]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/sample.py
[anima]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L1482-L1505
[anima-network]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/ldm/anima/model.py
[cosmos]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/ldm/cosmos/predict2.py
[krea]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L2702-L2728
[krea-network]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/ldm/krea2/model.py
[rank-proposal]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/pull/62
