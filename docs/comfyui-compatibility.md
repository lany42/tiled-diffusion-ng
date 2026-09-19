# ComfyUI compatibility

Compatibility across releases and development HEAD is best effort. Support
requires the APIs and semantics below, not an exact revision or runtime SHA
check. Host and extension updates require renewed inspection and execution.

## Evidence and source baselines

The retained source inspection baseline is **2026-09-19**:

| Component | Inspected revision | Actual host execution |
| --- | --- | --- |
| ComfyUI | [`3c80da7f87ee359b2d06f107cb3c0797079dfbbb`][comfy] | Pending |
| ComfyUI-TiledDiffusion | [`a155b1bac39147381aeaa52b9be42e545626a44f`][tiled] | Pending |
| RES4LYF | [`e8437efef69cacf3f08fbd9f90fcc517868c5cb8`][res] | Pending |

Source inspection establishes expected contracts. Offline tests use real CPU
tensors and project-written [host doubles](../tests/host.py); they cannot detect
upstream changes by themselves. The recorded isolated CPU probes of
`convert_cond`, `apply_empty_x_to_equal_area`, `encode_model_conds`,
`sampling_function` and `WrapperExecutor` exercised selected pinned functions,
not a complete host. Real SDXL, ControlNet, RES4LYF and visual acceptance remain
pending; no row below is completed by those CPU results.

## Contract matrix

All host checks in the final column are **pending**. Implementation and test
pointers identify offline coverage, not proof of actual host compatibility.

| Contract | Pinned source boundary | Implementation / CPU coverage | Outstanding host check |
| --- | --- | --- | --- |
| V3 loading and execution lists | [V3 schema and list API][io] | [nodes.py](../src/tiled_diffusion_ng/nodes.py), [root loader](../__init__.py); [node tests](../tests/test_nodes.py) | Load all three namespaced IDs and execute graphs from clone and source ZIP. Four complete local positives must enter one invocation; each ordinary socket carries one execution-list item. |
| Geometry and crop routing | [latent metadata][latents], [UNet spatial sizes][unet], [latent normalization][sample] | [geometry.py](../src/tiled_diffusion_ng/geometry.py), [SDXL adapter](../src/tiled_diffusion_ng/adapters/sdxl.py); [geometry tests](../tests/test_geometry.py) | Verify portrait overlaps 64/128, square/landscape, zero/small/large overlap, odd latent sizes such as 13×17, and the one-cell architectural minimum. Check RGB/RGBA image batches, overlap-inclusive crops and TL/TR/BR/BL image-major order. Probe checkpoint, attention and ControlNet shape limits. |
| Evaluation routing and global CFG | [sampling dispatch][samplers], [wrapper continuations][wrappers] | `guard_prediction`, `TileEvaluation` in [sampling](../src/tiled_diffusion_ng/_comfy_sampling.py); [sampling tests](../tests/test_sampling.py) for late overrides and wrapper composition | Confirm four tile continuation calls per conditional-batch evaluation from the same global x/sigma, including repeated sigma and changed live options. Check omitted branches, global pre/post-CFG hooks and rejection of bypass routes. |
| Local conditioning and preparation | [condition preparation][samplers], [resource discovery][helpers], [SDXL size encoding][sdxl] | `prepare_pairs` and `adapt_spatial_condition`; [sampling tests](../tests/test_sampling.py) for replacement, full-canvas defaults and pair propagation | Verify tag survival, global versus identical locals, distinct local routing and complete replacement including controls. Preserve explicit sizes and full-canvas defaults; discover local-only control chains. Confirm pair-local negative propagation, explicit negative precedence and multiple entries. |
| Ordinary SDXL controls | [ControlNet state and hints][controlnet], [architecture][cldm], [resize helper][utils] | [SDXL sampling context](../src/tiled_diffusion_ng/adapters/_sdxl_sampling.py); [control sampling tests](../tests/test_sampling.py), [hint tests](../tests/test_control_hints.py) | Run an ordinary RGB SDXL Tile ControlNet with global and local positives. Verify resize-before-crop alignment, chains, batch broadcasting, partial ranges, zero strength, read-only hint sharing and separate prepared caches. |
| Latent and prediction capabilities | [model boundary][model-base], [EPS/V conversions][model-sampling], [latent formats][latents] | [SDXL adapter](../src/tiled_diffusion_ng/adapters/sdxl.py); [adapter tests](../tests/test_adapters.py), [fusion tests](../tests/test_fusion.py), [sampling tests](../tests/test_sampling.py) | Confirm BCHW/scale-8 geometry and native denoised EPS/V behavior, branch order, device and dtype. Reject geometry-changing coercion, unknown spatial fields and incompatible conversions explicitly. |
| Native registrations and solver behavior | [common_ksampler][ksampler], [sampler dispatch][samplers], RES4LYF [registration][res-registration], [exports][res-exports] and [model calls][res-calls] | Live schema/dispatch in [nodes](../src/tiled_diffusion_ng/nodes.py) and [sampling](../src/tiled_diffusion_ng/_comfy_sampling.py); [sampling tests](../tests/test_sampling.py) execute instrumented registrations | Match native menus after extension loading and reject missing saved names. Run Euler, a multi-evaluation solver, an ancestral solver and two schedulers. Probe native `res_2m`/`res_3m`, `res_3s_ode`, available stochastic variants, `beta57` and `bong_tangent`, retaining post-CFG captures and global solver history. |
| Masks, batches and metadata | [common_ksampler][ksampler], [sample preparation][sample] | `sample`; [sampling tests](../tests/test_sampling.py) for masks, batch indices and bookkeeping | Check denoise 0/partial/1, LATENT noise masks, batch two with batch indices, ordinary metadata and unchanged inputs against native KSampler. |
| Iterative latent upscaling | [native sampling entry][ksampler] | `sample`, plan validation; [sampling tests](../tests/test_sampling.py) for repeated stages and [geometry tests](../tests/test_geometry.py) for stale plans | Execute two external latent-upscale/refinement stages with fresh plans, ordinary LATENT links and no implicit VAE round trip. Each stage starts its own trajectory. |
| Iterative pixel upscaling | [native sampling entry][ksampler] | Same sampling and plan contracts; external VAE/upscaler execution is outside CPU doubles | Execute two explicit decode/upscale/encode/refinement stages with fresh plans and caller-selected VAE boundaries. |
| Invocation isolation and cleanup | [model cloning][patcher], [wrappers][wrappers], [control cleanup][controlnet] | `sample`/`TileEvaluation.close`, `SDXLSamplingContext.close`; [adapter](../tests/test_adapters.py), [sampling](../tests/test_sampling.py) and [hint lifetime tests](../tests/test_control_hints.py) | Check compatible-plan reuse, stale-plan rejection, A→B→A repeatability, cancellation and deliberate failure. Original inputs must remain usable, project wrappers must detach, and hints/caches/stage state must be released. |
| Visual acceptance | [upstream Mixture of Diffusers][tiled-mod] | [Mathematics](mathematics.md) and numerical tests establish bounded identities only | Complete the [reproducible comparison](manual-validation.md#reproducible-comparison), save settings/images and obtain the user's favorable portrait review without unresolved material regressions. |

The SDXL adapter owns model/layout validation and spatial preparation; shared
orchestration owns pairing, routing, fusion and one `common_ksampler` trajectory.
Unknown conditioning areas/masks, spatial inputs, context handlers and replacement
evaluation routes require explicit semantic support. Compatible global CFG hooks
remain available; the [CFG identity](mathematics.md#shared-linear-cfg) only proves
shared linear guidance. RES4LYF support covers native KSampler registrations,
not its custom sampler nodes, OPTIONS/GUIDES objects or separate tiling modes.

## Hint-sharing assumptions

All four pairs are discovered before hint materialization and host preparation.
Grouping identifies a source by storage pointer/device, offset, shape, strides,
dtype and logical view flags, plus resize algorithm, crop policy and target
size. A source reference prevents pointer reuse until materialization finishes;
image contents are not compared. In the supported ordinary ControlNet path,
network identity, strength and schedule do not change unprepared hint pixels.

Normalize each group to the full canvas once, even at matching dimensions, then
crop. Retain either canvas views or compact copies of distinct rectangles,
whichever uses less backing storage; ties retain the canvas. Release source
descriptors and unused canvases as groups finish. Shared `cond_hint_original`
pixels must remain read-only. Each clone owns mutable `cond_hint`, timestep state
and chain links, reusing prepared caches across evaluations. An upstream in-place
hint write or new preprocessing dependency requires revisiting this contract.
See the [control path][controlnet] and [resize helper][utils].

## Rechecking an update

1. Record candidate host and extension revisions and inspect changes to every
   affected matrix boundary, including alternate dispatch paths, before changing
   pinned references or relaxing capability checks. Start with the
   [ComfyUI diff][comfy-diff] and [TiledDiffusion diff][tiled-diff], replacing the
   destination branch with the candidate revision.
2. Run the [development checks](../AGENTS.md). When changing host isolation or
   invocation lifetime, also exercise repeated suite runs in one process.
3. Execute affected matrix cases in the actual host, including globals, locals,
   chained controls, multiple evaluations, repeatability, cancellation and failure.
   Retain revisions, settings, outputs and failures; keep unexecuted cases pending.

[Upstream differences and performance limits](tileddiffusion-comparison.md) and
[future adapter research](model-adapter-research.md) are separate references.

[comfy]: https://github.com/Comfy-Org/ComfyUI/tree/3c80da7f87ee359b2d06f107cb3c0797079dfbbb
[tiled]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/tree/a155b1bac39147381aeaa52b9be42e545626a44f
[res]: https://github.com/ClownsharkBatwing/RES4LYF/tree/e8437efef69cacf3f08fbd9f90fcc517868c5cb8
[io]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy_api/latest/_io.py
[latents]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/latent_formats.py
[unet]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/ldm/modules/diffusionmodules/openaimodel.py
[sample]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/sample.py
[samplers]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py
[wrappers]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/patcher_extension.py
[helpers]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/sampler_helpers.py
[sdxl]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L515-L538
[controlnet]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/controlnet.py
[cldm]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/cldm/cldm.py#L119-L186
[utils]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/utils.py#L1013-L1044
[model-base]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L208-L257
[model-sampling]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_sampling.py#L30-L55
[ksampler]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/nodes.py#L1572-L1624
[res-registration]: https://github.com/ClownsharkBatwing/RES4LYF/blob/e8437efef69cacf3f08fbd9f90fcc517868c5cb8/__init__.py
[res-exports]: https://github.com/ClownsharkBatwing/RES4LYF/blob/e8437efef69cacf3f08fbd9f90fcc517868c5cb8/beta/__init__.py#L92-L202
[res-calls]: https://github.com/ClownsharkBatwing/RES4LYF/blob/e8437efef69cacf3f08fbd9f90fcc517868c5cb8/beta/rk_method_beta.py
[patcher]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_patcher.py
[tiled-mod]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/blob/a155b1bac39147381aeaa52b9be42e545626a44f/tiled_diffusion.py#L706-L821
[comfy-diff]: https://github.com/Comfy-Org/ComfyUI/compare/3c80da7f87ee359b2d06f107cb3c0797079dfbbb...master
[tiled-diff]: https://github.com/shiimizu/ComfyUI-TiledDiffusion/compare/a155b1bac39147381aeaa52b9be42e545626a44f...main
