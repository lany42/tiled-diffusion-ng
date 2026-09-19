# SDXL manual validation

**Pending:** real ComfyUI execution and visual acceptance have not been completed.
Source inspection and CPU tests do not satisfy this gate. Remaining host cases
are tracked in the [compatibility matrix](comfyui-compatibility.md#contract-matrix);
work beyond SDXL remains gated by acceptance.

## Reproducible comparison

Build the comparison graphs in ComfyUI using standard SDXL base conditioning and
linear CFG. No example workflows are bundled. Match the checkpoint/VAE, input
latent, prompts, seed, steps, CFG, denoise, sampler, scheduler, decode path, device,
precision and attention backend. Keep the same RES4LYF installation for both
runs when testing its native registrations, and establish that the chosen native
KSampler configuration works before adding tiling.

Begin with the **1664×2432 portrait**, 20 steps, seed 12345, CFG 7, denoise 0.35,
Euler/normal. These are reproducible starting settings, not an optimum. Compare
with [pinned upstream TiledDiffusion](tileddiffusion-comparison.md) using Mixture
of Diffusers and tile batch size 1. Set sampling extents as follows; all values
in this table are pixels:

| Shared overlap | Sampling width×height | Right origin | Bottom origin |
| --- | --- | --- | --- |
| 64 | 864×1248 | 800 | 1184 |
| 128 | 896×1280 | 768 | 1152 |

Verify the actual sampling rectangles in both implementations: exactly two
columns and two rows, with matching coverage and effective overlap. The nominal
832×1216 core sizes would produce nine upstream views at overlap 64 if used as
sampling sizes. Upstream evaluates row-major TL/TR/BL/BR; this API uses clockwise
TL/TR/BR/BL. Allow for numerical accumulation-order differences.

Compare stock upstream first. Optionally use an isolated corrected reference
changing only Gaussian `h/2` to `(h-1)/2` and vertical `w*w` to `h*h`. Save its
revision and reference-only diff; [PR #77](https://github.com/shiimizu/ComfyUI-TiledDiffusion/pull/77)
alone fixes only the scale. Keep this reference separate from the user's ordinary
upstream installation.

Save queued graphs, package/host/extension revisions and dirty diffs, checkpoint,
VAE and input hashes, all settings, plan rectangles and effective overlaps, paired
images, crop previews, and observations including failures. For control runs,
include the control checkpoint, hint, chain, strength and start/end settings.
For discrepancies, compare one captured evaluation at identical `x`, sigma and
prepared conditions before comparing stochastic trajectories. Latent differences
are diagnostics; choose tolerances for the dtype and device.

Inspect seams, the central intersection, texture continuity, detail, composition
and unintended repetition, with portrait, square and landscape cases. For local
prompts, use distinct quadrant content to verify routing and four identical
locals to check agreement with the global-only result. The stock shared-prompt
node is not a direct reference for four distinct local positives.

Acceptance requires successful global, local, ordinary SDXL Tile ControlNet,
native RES4LYF and both iterative workflows, plus the pending routing and
lifecycle checks in the compatibility matrix. The user's portrait review must
find results at least comparable to upstream in seams, detail and composition,
with no unresolved material regressions. Explain differences from stock and
corrected references; neither bitwise equality nor a speedup threshold is a gate.
