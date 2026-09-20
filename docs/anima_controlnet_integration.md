# Tiled Anima LLLite integration plan

Status: LLLite remains proposed and unimplemented. Updated on 2026-09-19 after
reviewing the base Anima implementation and ComfyUI source at
[`944386c233e02eaf877b1c8d5d513fb3d3a4d5e3`][anima-host]. Source inspection and
offline contracts are separate from real-host acceptance, which remains pending.

## Preconditions and assumptions about Anima support

The [Anima adapter](../src/tiled_diffusion_ng/adapters/anima.py) now supplies
the base sampling contract through the existing TilePlan, TileView and
TileSampler nodes. Preserve these established choices when adding LLLite:

- **Family:** native Anima with Wan21 metadata: 16 latent channels, three latent
  dimensions, spatial scale 8 and temporal scale 4. One adapter covers Base,
  Aesthetic, Turbo and 2.9B; [host detection][anima-detection] reads transformer
  depth from weights. This does not establish LLLite compatibility across depths.
- **Layout:** floating, nonempty `B×16×H×W` and `B×16×1×H×W` inputs share a
  canonical `BCTHW` plan signature. [Native sampling][anima-sample] introduces the
  singleton temporal axis and returns 5D. Video and structured spatial LATENT
  metadata are rejected. Output dtype and Wan21 normalization remain host-owned;
  the [sampled output path][anima-samplers] converts to float32 before undoing
  latent normalization. Do not require the input dtype to survive sampling.
- **Geometry:** signature `anima`, adapter version 1, scale `(8, 8)`, alignment
  `(2, 2)` and minimum extent `(2, 2)` in latent cells. Canvas dimensions must be
  divisible by 16 pixels. Origins and extents of sampling rectangles are aligned;
  impossible four-view geometry fails without resizing or padding. TILE_PLAN
  remains schema 1; preserve its complete value, including overlap-inclusive
  `pixel_sampling` rectangles, as the future attachment's plan identity.
- **Positions and predictions:** retain native tile-local RoPE and extrapolation
  settings, without extra absolute positions or global offsets. Require native
  `CONST` input and denoised conversion, `denoised = x - sigma * prediction`.
  Conversion stays in the host; fuse denoised branches before global CFG.
- **Text:** preserve embeddings, T5 IDs/weights, attention-mask and pooled
  metadata, strength and timestep settings. [Anima extra_conds][anima-base]
  preprocesses text during inference; its other path passes IDs/weights into
  forward. Metadata preservation does not imply every field is consumed by the
  backbone. Keep global positive/negative and four complete local positives in
  TL/TR/BR/BL order, with text preparation owned by the host.
- **Routing and lifetime:** one `common_ksampler` call owns the full trajectory,
  masks, batch indices and solver history. All tiles read the same current latent
  and sigma. Preserve native continuations, live options, ordinary weight LoRAs,
  compatible wrappers, optimized attention and global CFG hooks. Invocation
  context and fusion state close on success, failure and cancellation.

Current guards deliberately reject non-`None` conditioning controls with
`ValueError`: the native Anima transformer does not consume UNet ControlNet
residuals. Recognized native AnimaLLLite hooks raise `NotImplementedError` for
deferred tiled routing; other transformer patches/replacements raise `ValueError`.
Recognition is validation only, including when strength is zero. Future LLLite
support must admit only its validated attachment/hook set through these guards.

[Offline tests](../tests/test_anima.py) cover the base contracts, including
28/40-layer configurations, live options and cleanup. They neither execute
native LLLite hooks nor establish checkpoint or visual compatibility. Base,
Aesthetic, Turbo and 2.9B each still require real-host generation/refinement,
portrait/landscape, local-prompt, LoRA and seam acceptance with recorded revisions
and settings. The [adapter research](model-adapter-research.md) is background,
not a substitute for that evidence.

## Initial objective and scope

Add a tiled adaptation of native ComfyUI `AnimaLLLiteApply`, provisionally named
`TiledAnimaLLLiteApply`, for **increasing detail during high-resolution tiled
upscaling refinement**.

The node receives reference crops generated using the same TilePlan as the
sampler. It returns one MODEL containing a recognized spatial patch. During
sampling, each latent tile receives the corresponding reference crop and its
global or local text conditioning.

Initial scope:

- One supported RGB Anima LLLite patch per tiled MODEL.
- Four overlap-inclusive reference tiles per source image, with explicit batch
  mapping and shared strength/start/end settings.
- Existing global conditioning and optional local positives.
- One native sampling trajectory and existing prediction fusion.
- Explicit validation and invocation-local preparation.

Deferred: masks and four-channel inpainting patches, multiple tiled LLLite
patches, per-tile strengths/schedules, separate positive/negative patch controls,
arbitrary spatial MODEL patch support, video, and depth/canny generation claims.
Compatible ordinary upstream weight LoRAs should remain preserved, but this
feature does not establish universal compatibility with other MODEL patchers.

## Analogy with existing SDXL ControlNet support

The common operation is to associate a latent view with spatial guidance in the
same canvas coordinates. Where that guidance is attached differs:

| Concern | Existing SDXL native ControlNet | Proposed tiled Anima LLLite |
| --- | --- | --- |
| Carrier | Control objects referenced by positive/negative CONDITIONING metadata | Recognized hooks and metadata attached to MODEL |
| Reference input | Full-canvas control hint | Already-cropped reference IMAGE batch plus TilePlan |
| Spatial preparation | Normalize full hint to canvas, then crop each sampling rectangle | Validate crop-to-plan correspondence, then select the reference for the current rectangle |
| Per-tile state | Control copies with separate mutable caches | Invocation-owned patch preparation and tile routing |
| Local positive behavior | Replaces the entire global positive, including its controls | Replaces text conditioning while the MODEL patch remains active |
| Sampling | One trajectory; tiled predictions fused | Same |

Current SDXL preparation pairs each region's positive and negative conditioning
before host preparation. It resolves requested control propagation to the
negative branch, copies supported control chains, and assigns cropped hints.
Hint pixels can be shared read-only while mutable prepared control state remains
separate. See [pair preparation](../src/tiled_diffusion_ng/_comfy_sampling.py)
and [native control preparation](../src/tiled_diffusion_ng/adapters/_native_control.py).

Local positives can contain SDXL controls; they are not inherently text-only.
However, a text-only local positive does not inherit controls from the global
positive. The supplied negative remains separate and may itself carry controls.

Do not describe external Apply ControlNet nodes receiving already-cropped hints
as an equivalent supported SDXL workflow today. Our current preparation treats
those hints as full-canvas inputs and would normalize and crop them again.
External local positives can carry full-canvas hints for our existing path.

For Anima, document the intended difference prominently in node help:

> Local positives change each tile's text guidance. The tiled LLLite patch
> remains active for every tile during its configured range, including when
> local positives contain only text conditioning.

Do not add special positive-only gating initially. Preserve the native patch's
behavior on whichever branches the host evaluates and verify it in testing.

## Proposed workflow and node surface

```text
Upscaled reference image at the target canvas size
                       |
                       v
Anima MODEL ------> TilePlan ------> TileView
     |                 |               |
     |                 |               v
     |                 |        reference IMAGE batch
     |                 |               |
     +------> TiledAnimaLLLiteApply <---+
                       ^
                       |
                 MODEL_PATCH loader

Tiled MODEL + same TilePlan + full-canvas refinement LATENT
             + global prompts + optional local positives
                       |
                       v
                  TileSampler
                       |
                       v
              refined full-canvas LATENT
```

The diagram omits ordinary strength/schedule inputs and external VAE operations.
The caller supplies the refinement latent and reference in matching canvas
coordinates. This integration introduces no implicit VAE round trip or upscaler.
Create the plan from the compatible base model and target latent, then reuse it
for TileView, the tiled apply node, and TileSampler; no graph cycle is needed.

| Input | Proposed contract |
| --- | --- |
| `model` | Supported Anima MODEL, retaining compatible existing patches |
| `model_patch` | Compatible native-loaded Anima LLLite weights; initially RGB only |
| `tile_plan` | Exact plan governing reference crops and sampling |
| `reference_tiles` | TileView IMAGE batch with four entries per reference source image |
| `strength` | One shared patch strength; preserve native semantics |
| `start_percent`, `end_percent` | One shared activation window, using host sigma conversion |

Output: one MODEL. Final names and node ID should follow repository conventions.
Do not expose an optional mask socket in the first version unless an accepted
initial checkpoint demonstrably requires it; such a requirement reopens scope.

### IMAGE batches are not execution lists

Current [TileView](../src/tiled_diffusion_ng/nodes.py) returns one IMAGE tensor,
not a ComfyUI execution list. Its flattened batch order is:

```text
image 0: TL, TR, BR, BL; image 1: TL, TR, BR, BL; ...
```

TileView requires the input image to match the plan's pixel canvas and crops
`pixel_sampling`, including overlap. It does not normalize an arbitrary source
image first. Preserve this explicit upstream sizing requirement.

The proposed node interprets a batch of length `4 * B` as B reference groups.
Tile i selects entries `i, i + 4, i + 8, ...`, not a contiguous quarter of the
flattened batch. At sampling time, require B to equal the latent image batch, or
explicitly support B=1 broadcast. Do not introduce arbitrary repetition rules.
Host conditional/unconditional batching is a separate dimension of evaluation
that must retain native semantics after selecting the tile's source batch.
The [native LLLite module][anima-lllite] repeats the whole reference batch when
the model batch is a multiple of it. This matches concatenated conditioning
batches; it is not per-image `repeat_interleave`. Test multiple conditioning
entries and separate/combined positive-negative calls, including CFG 1's omitted
negative branch. Do not rely on divisibility alone to validate source mapping.

A plain IMAGE tensor does not carry provenance. Matching shape and count cannot
prove it came from the specified TileView. Document the ordering contract and
validate all observable geometry. A dedicated reference-bundle type carrying
provenance can be considered later if the public workflow needs stronger checks.

## Reimplementation boundary

Use native `AnimaLLLiteApply` as the source-level template for the new node.
Prefer reusing the native loader, model weights, and compatible attention/MLP
classes while specializing input preparation and reference selection. Copy only
what is necessary; inspect licensing and preserve required attribution before
copying upstream implementation text.

The pinned [apply node][anima-apply] clones MODEL and installs `post_input`,
`attn1_patch`, `attn2_patch` and `mlp_patch`. Its input hook derives reference size
from the received latent and handles optional four-channel inpainting masks.
The following source contracts constrain the specialization:

- **Hook data:** [native forward][anima-backbone] calls post-input hooks after
  latent padding and embedding. `x` is the padded `BCTHW` latent; `img` is the
  embedded `BTHWD` sequence. The host creates a fresh `model_patch_data` dictionary
  whenever post-input patches are present. Put invocation/tile dispatch in a
  separate namespaced transformer option; it would be overwritten in
  `model_patch_data`. The native input hook keys prepared embeddings by its own
  object, and the attention/MLP hooks must reference that same object.
- **Hook targets:** self-attention hooks modify Q/K/V inputs before projection;
  cross-attention modifies only Q, whose tokens follow the image grid. Text K/V
  have a different sequence length. The MLP hook runs before `mlp.layer1`.
  Preserve these native routes rather than replacing forward.
- **Checkpoint coverage:** the native loader admits named-key v2 LLLite weights.
  Validate RGB `cond_in_channels == 3`, `model_dim` against the actual backbone,
  and every targeted block index. `block_count` is the highest referenced index
  plus one, not proof of complete coverage: `apply` silently skips absent
  modules. Decide and validate supported sparse coverage explicitly; loading a
  checkpoint alone does not establish compatibility with a 28- or 40-layer model.
- **Activation:** the apply node converts percentages with `percent_to_sigma`.
  The input hook uses the inclusive range `sigma_end <= max(sigmas) <= sigma_start`
  for the whole forward, not a separate gate for each image. Preserve and test
  that behavior with mixed sigmas and repeated solver evaluations. Preparation
  must account for changes to MODEL sampling settings after patch application,
  rather than silently using stale thresholds. Require tiled dispatch before
  returning early for zero strength or an inactive window.
- **Discovery and ownership:** [ModelPatcher cloning][anima-patcher] copies
  option containers while preserving hook objects; attachments are shared unless
  they implement `on_model_patcher_clone`. The input hook's `models()` exposes
  MODEL_PATCH through [model_patches_models()][anima-patch-discovery] to
  [native loading][anima-loading].
  Install that discovery route before `common_ksampler`; adding it only during a
  tile evaluation is too late. Explicit `additional_models` discovery in the
  current CPU doubles does not test this patch-owned loading route.

Passing our entire TileView batch straight into the unmodified native node does
not implement routing. It is interpreted as an image batch, not four spatial
regions. Passing a full reference unchanged to each tile instead would fit the
whole reference into each tile. Neither is the intended spatial operation.

Reinspect these pinned contracts when changing the host baseline; the revision
is evidence, not an exact-version runtime requirement. The
[node documentation](https://docs.comfy.org/built-in-nodes/AnimaLLLiteApply) is
secondary to the code. Do not infer compatibility with the Kohya custom node:
its wrapper-based integration is a separate implementation.

## MODEL attachment and automatic sampler behavior

Attach a namespaced, versioned declaration of tiled spatial-patch capability.
An illustrative name is `tiled_diffusion_ng.anima_lllite.v1`; finalize the name
after inspecting current host attachment APIs.

Durable configuration should include:

- Schema version and recognized patch kind.
- Full plan identity by value: adapter/geometry signature, layout, canvas,
  tile identities, and sampling rectangles. Equal tile shapes alone are not
  enough to distinguish plans with different coordinates.
- Reference source batch count and the image-major mapping convention.
- Reference tensors and auxiliary MODEL_PATCH reference.
- Strength and activation settings, plus a stable association between the
  declaration and the installed hooks.

The sampler should validate that the declaration and hook objects agree. An
attachment without its hooks, duplicate installations, unsupported schema, or
stale plan must fail explicitly. Do not infer tiled capability from an arbitrary
attention patch or its tensor dimensions.

The model adapter still selects Anima semantics. The attachment selects spatial
patch preparation. Together these implement the proposed automatic "Anima
patched by tiles" behavior without creating a second sampler or solver path.

Current SamplingContext handles conditioning pairs and does not receive MODEL
or per-tile model options. Extend that contract deliberately, or introduce a
small adjacent spatial-patch preparation interface. Keep the shared loop free
of Anima-specific tensor preparation. `validate_model_options` already runs
before host sampling and at each tiled conditional evaluation. The existing
per-region continuation is the insertion point for tile options; the base
implementation intentionally has no attachment schema, dispatch API or cache.
The LLLite implementation should support:

1. Validating and preparing recognized MODEL patches for one invocation.
2. Producing per-tile options or dispatch context before calling the existing
   conditional-evaluation continuation.
3. Closing all prepared state in the existing unconditional cleanup path.

The specialized hooks must require this dispatch context. Feeding the tiled
MODEL to an ordinary KSampler should fail clearly rather than silently use all
references or one arbitrary crop. Generic sampler compatibility is not initial
scope.

## Evaluation and lifecycle contract

For each native conditional evaluation:

1. Read the current full-canvas latent and sigma.
2. Select the region's latent view and its positive/negative conditioning.
3. Create invocation-local model options carrying the tile identity and prepared
   patch context, while preserving unrelated live options and wrappers.
4. The input hook selects the corresponding reference batch and prepares its
   conditioning for that tile's model call.
5. Native attention/MLP hooks consume the selected conditioning.
6. Fuse predictions using the existing weights, then return full-canvas branches
   to native CFG and solver execution.

The host resolves PREDICT_NOISE wrappers from guider-owned options and
CALC_COND_BATCH wrappers from the sampler's live options. Preserve the exact
invocation's tiled conditional wrapper in any per-tile copy. The prediction
guard now rejects removed, replaced or duplicate tiled wrappers before native
dispatch; otherwise replacement live options could silently bypass tiling and
its patch validation. Re-entering the outer conditional evaluator would also
re-enter tiling, so call only the supplied continuation.

Do not switch four arbitrary MODEL objects inside one host trajectory. Retain
one model configuration with tile-aware spatial patch dispatch. Avoid mutating
an upstream hook's image field or storing a mutable `current_tile` on a shared
patch object; clone semantics may preserve references to that object.

Treat attachment configuration as read-only. Invocation-owned objects hold
prepared embeddings, device/dtype caches, and dispatch state. Preserve auxiliary
model discovery and native loading/offloading instead of manually moving weights
outside host management. Clear invocation references after success, exception,
or cancellation; upstream graph-owned source references remain valid.

Start with correct per-tile preparation. If embeddings are cached, include tile
identity, source association, effective geometry/padding, device, and dtype in
their ownership/keying rules. Equal-size tiles must never share conditioning
merely because their shapes match. Reuse across repeated evaluations within an
invocation is optional; cross-invocation caches are not part of this design.

## Encoding and padding policy

Initial policy: encode each reference crop as the guidance for that tile. This
mirrors the existing ControlNet approach of cropping pixels before the control
network evaluates them. Do not claim equivalence to encoding the full canvas
and cropping its embeddings; the native encoder's normalization and optional
global pooling make that a different computation.

The base adapter has selected aligned geometry. With temporal patch size 1,
spatial patch size 2 and T=1, every evaluated tile already fits the native patch
grid, so transformer padding adds no image area. A reference crop must match
`pixel_sampling` exactly. The native encoder's two stride-4 convolutions produce
one token per 16×16 pixels, matching one Anima token per 2×2 latent cells. Check
that token count explicitly; do not resize a malformed reference to make it fit.
Retain native RGB clamping and normalization when specializing preparation.
The backbone's separate zero padding-mask channel does not enlarge the rectangle.
Supporting unaligned geometry later would change this contract and require a
new adapter policy and matching reference handling.

## Deferred masks and inpainting

Native Anima LLLite's optional mask is not a general control-influence mask. It
is an additional conditioning channel for compatible four-channel checkpoints,
with checkpoint-dependent masked-image preparation. Initial RGB refinement does
not need it, and inpainting is low-value for the primary detail-enhancement use
case. Reject four-channel checkpoints clearly in the first implementation.

If added later, masks must share the reference crops' canvas, region order,
batch mapping, and overlap. A future UI could accept a full-canvas mask and crop
it by the plan, or accept an explicitly ordered mask batch. Do not assume
TileView currently provides a MASK output. A mask that gates patch influence is
a separate feature with different semantics and must not reuse an inpainting
mask socket ambiguously.

## Implementation sequence

1. Preserve the established Anima contract above and track base real-host
   acceptance separately from source inspection and offline tests.
2. Confirm the pinned native apply, LLLite, clone, discovery and hook contracts
   against the chosen host revision and checkpoint. Select the reuse/copy boundary.
3. Define the configuration object, schema version, plan comparison, and IMAGE
   batch validation. Define ordinary-sampler rejection behavior.
4. Implement and register the tiled apply node and recognized hook set. Preserve
   compatible upstream MODEL state and reject duplicate tiled patch application.
5. Extend invocation preparation and per-tile dispatch in the adapter/context
   layer; preserve one `common_ksampler` call and existing fusion/CFG behavior.
6. Implement aligned reference preparation and lifecycle handling. Add caching
   only if its benefit and isolation are demonstrated.
7. Add offline contract coverage, run the repository's required Python checks,
   and perform separate real-host refinement validation.
8. Publish node help explaining reference order, same-plan requirements, local
   positives, supported refinement scope, and deferred masks. Any further
   documentation changes should follow the explicit scope of the implementation
   task.

## Validation and acceptance criteria

Offline CPU tests should use real small tensors and minimal host doubles:

- Distinct asymmetric references prove TL/TR/BR/BL selection, including overlap;
  same-shaped references must remain distinguishable.
- Batch one and batch two prove image-major indexing, supported broadcast, and
  host conditional/unconditional batch handling.
- Global and distinct local positives use the same spatial patch routing;
  replacing text conditioning does not remove the MODEL patch.
- Stale plans, wrong counts/shapes, unsupported schema, duplicate patches,
  unsupported checkpoint channels/widths/block coverage, and missing dispatch
  context fail clearly, including at zero strength or an inactive window.
- Strength zero and inactive sigma windows preserve baseline behavior. Repeated
  sigma evaluations do not advance a tile-call-based schedule.
- Compatible patches, live options, auxiliary-model discovery, one sampling
  trajectory, and full-canvas CFG routing remain intact.
- Exercise patch-owned `models()` discovery and loading separately from explicit
  `additional_models`. Preserve hook identity across clone and option copies;
  test the native per-forward reset of `model_patch_data` and rejection when live
  options lose the invocation's tiled wrapper.
- Success, injected failure, cancellation, and A→B→A invocations leave original
  MODEL/reference inputs usable and release prepared state.
- Padding/alignment behavior matches the explicitly chosen Anima policy.

Real-host acceptance must use an identified compatible RGB Tile/refinement
LLLite checkpoint and record host/checkpoint/node revisions and settings. Check
high-resolution detail, preservation of structure, overlap seams, portrait and
landscape canvases, activation windows, ordinary LoRA composition, local prompts,
and repeated execution. A run without exceptions is not spatial validation.
Offline doubles cannot substitute for these checks or establish upstream
compatibility across releases.

## Follow-on possibilities, not initial promises

The same dispatch design could later pair tiled depth/canny references with local
prompts and a full-canvas noise latent for high-resolution guided generation.
This would still be one sampling trajectory with several tile evaluations per
model evaluation, not one model forward or independently generated tiles.

Prefer a coherent full-canvas control map before cropping when the control type
requires consistent global values. Shared overlap pixels do not guarantee shared
semantics: different local prompts and independently encoded views can disagree,
and fusion does not restore full-canvas attention. Validate each new behavior
before claiming support. The first implementation remains focused on tiled
upscaling refinement.

[anima-host]: https://github.com/Comfy-Org/ComfyUI/tree/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3
[anima-detection]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_detection.py#L848-L881
[anima-sample]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/sample.py#L45-L71
[anima-samplers]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/samplers.py#L1210-L1238
[anima-base]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_base.py#L1482-L1505
[anima-apply]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy_extras/nodes_model_patch.py#L384-L425
[anima-lllite]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/ldm/anima/lllite.py
[anima-backbone]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/ldm/cosmos/predict2.py
[anima-patcher]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_patcher.py#L431-L483
[anima-patch-discovery]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_patcher.py#L794-L816
[anima-loading]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/model_management.py#L936-L955
