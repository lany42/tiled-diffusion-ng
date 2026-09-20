# Tiled Anima LLLite integration plan

Status: proposed implementation, not a compatibility claim. This note records
the design discussion on 2026-09-19 so work can resume after basic Anima support
lands. It authorizes no change to the existing sampling contract by itself.

## Preconditions and assumptions about Anima support

This integration assumes the node pack already supports ordinary Anima image
sampling. At the time of this design discussion, the checked-in adapter registry
still contains SDXL alone. Implement and validate the following prerequisites
before adding LLLite support:

- **Explicit model adapter:** recognize supported Anima models by their required
  APIs and semantics. Keep architecture-specific validation and preparation out
  of the shared geometry and fusion implementation.
- **Latent layout:** establish the actual stored LATENT layout and model-call
  layout, including 16-channel latent handling and the singleton temporal axis
  expected by the inspected native LLLite path. Admit image sampling with T=1;
  do not imply video or temporal tiling support.
- **Geometry:** establish latent-to-pixel scale, transformer patch alignment,
  minimum extents, and padding rules. TilePlan, TileView, and the model evaluation
  must agree on overlap-inclusive sampling rectangles. Resolve odd dimensions
  and padding without silently stretching the spatial reference.
- **Positions:** choose and validate the positional-coordinate policy for tiled
  Anima evaluations. LLLite integration must use that established policy rather
  than introducing a separate one.
- **Prediction semantics:** validate Anima's native flow/prediction conversion
  and its compatibility with prediction fusion. Do not reuse SDXL EPS/V checks
  merely by relaxing their rejection conditions.
- **Conditioning:** preserve Anima text preparation and support global positive,
  global negative, and four complete local positives in TL/TR/BR/BL order.
- **Trajectory ownership:** retain one `common_ksampler` call, one full-canvas
  latent trajectory, and native solver history. All tile views in an evaluation
  read the same current latent and sigma. Fuse branch predictions before the
  host applies compatible global CFG behavior.
- **Patch routing:** preserve compatible MODEL patches through cloning and
  native evaluation. Verify native post-input, attention, and MLP hook routing,
  live model options, sigma propagation, and auxiliary-model discovery.
- **Lifecycle:** leave caller inputs usable and release invocation-owned state
  on success, failure, and cancellation. Anima support must not depend on a
  sibling host checkout or GPU for its offline contract tests.

The existing [model adapter research](model-adapter-research.md) is a starting
point. Its inspected revisions and CPU doubles do not establish real-host Anima
compatibility. Record real-host evidence separately.

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

Inspected native behavior relevant to this design:

- The apply node clones MODEL and installs post-input, self-attention,
  cross-attention, and MLP patches.
- Its image preparation derives target dimensions from the received latent.
- Native mask handling is conditioning for four-channel inpainting checkpoints.
- The host supplies per-forward patch data and invokes post-input hooks after
  latent padding.

Passing our entire TileView batch straight into the unmodified native node does
not implement routing. It is interpreted as an image batch, not four spatial
regions. Passing a full reference unchanged to each tile instead would fit the
whole reference into each tile. Neither is the intended spatial operation.

Use these upstream source locations when resuming:

- [Native apply node and loader](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy_extras/nodes_model_patch.py)
- [Anima LLLite implementation](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/anima/lllite.py)
- [Host forward and patch routing](https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/ldm/cosmos/predict2.py)
- [Node documentation](https://docs.comfy.org/built-in-nodes/AnimaLLLiteApply)

These are moving source links from the discussion, not a pinned compatibility
baseline. Reinspect and record an exact revision before implementation. The
documentation is secondary to the code. Do not infer compatibility with the
Kohya custom node: its wrapper-based integration is a separate implementation.

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
of Anima-specific tensor preparation. The implementation should support:

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

Resolve padded tile dimensions explicitly with the Anima adapter. A reference
crop corresponds to the actual sampling rectangle, not an enlarged rectangle
invented by resizing to padded dimensions. Either constrain initial geometry to
validated aligned sizes or define and test matching reference padding. Reject
unsupported cases rather than silently distort coordinates. The precise padding
policy is an implementation gate, not settled by this note.

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

1. Complete the Anima prerequisites and record their validated host baseline.
2. Reinspect native apply, LLLite, cloning/attachments, auxiliary-model discovery,
   and host hook routing at that baseline. Select the reuse/copy boundary.
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
  unsupported checkpoint channels, and missing dispatch context fail clearly.
- Strength zero and inactive sigma windows preserve baseline behavior. Repeated
  sigma evaluations do not advance a tile-call-based schedule.
- Compatible patches, live options, auxiliary-model discovery, one sampling
  trajectory, and full-canvas CFG routing remain intact.
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
