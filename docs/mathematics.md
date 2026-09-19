# Mathematics

These desk-checked identities describe the implemented four-view policy and
[Mixture of Diffusers v1, §§3.1–3.2, equations 15–16 and Algorithm 1][paper].
Geometry and fusion are independent of host APIs. Source assumptions and pending
execution checks belong in [ComfyUI compatibility](comfyui-compatibility.md).

## Sampling rectangles

For an axis of `L` latent cells, pixel-per-cell scale `s`, crop alignment `a`,
minimum extent `m`, and requested shared pixel overlap `p`:

```text
o = ceil(p/s)
t = max(a*ceil((L+o)/(2*a)), a*ceil(m/a))
origins = (0, L-t)
effective overlap = 2*t-L
```

Coverage requires `2*t-L >= o`; rounding this bound and `m` upward to the
alignment lattice gives the smallest feasible `t`. Require `L % a == 0` and
`0 < t < L`. Rounding can increase overlap, including a zero request on an odd
axis. This fixed-four policy is a project choice, not a formula from the paper.

Combining the axes gives clockwise TL/TR/BR/BL sampling rectangles. Bounds are
half-open `(x0, y0, x1, y1)`; dimension pairs are `(H, W)`. Pixel bounds multiply
by each axis's scale. Diagnostic cores split at `floor(L/2)`; the sampling
rectangles include overlap. See [geometry.py](../src/tiled_diffusion_ng/geometry.py)
and the [portrait comparison](manual-validation.md#reproducible-comparison).

## Regional predictions and Gaussian fusion

At one evaluation, `R_i(x)` crops the same global latent `x` at the same sigma
for every region. `E_i` inserts a tile into a zero global canvas. Let `P_i` and
`N_i` be its positive and negative predictions. A local positive replaces the
complete global positive for that tile; it adds no fifth contribution.

For integer cell indices `u` and `v` in a tile of width `w` and height `h`:

```text
g(u,v) = exp(-50 * (((u-(w-1)/2)/w)^2 + ((v-(h-1)/2)/h)^2))
D = sum_i E_i(g)
F(Q) = sum_i E_i(g * Q_i) / D
```

Variance is `0.01`, standard deviation `0.1`. The separable kernel spans each
sampling rectangle. Its symmetric discrete centers and independent axis scales
follow the author's [canvas kernel][canvas]. The common density prefactor cancels;
per-tile gains would change the contract. Both vertical differences in pinned
upstream are described in the [Gaussian comparison](tileddiffusion-comparison.md#gaussian-differences).

The paper's `Z` is `1/D`. Normalized contributions sum to one, preserving
constants and prediction scale wherever the canvas is covered. At exactly zero
effective overlap, the sole contributor's weight cancels; weights cannot smooth
a boundary without shared coverage. See [fusion.py](../src/tiled_diffusion_ng/fusion.py).

## Shared linear CFG

With one shared guidance scale `c` and identical weights for both branches:

```text
G_i = N_i + c*(P_i-N_i)
F(G) = F(N) + c*(F(P)-F(N))
```

Linearity permits separate branch fusion followed by one full-canvas CFG
operation. It does not establish equivalence for nonlinear regional guidance
or different per-tile scales. The host applies CFG to the fused branches at its
[sampling boundary][samplers].

## Affine prediction conversion

The inspected [model boundary][model-base] returns denoised predictions.
Native EPS uses `R_i(x)-sigma*epsilon_i`; native V-prediction is another shared
affine conversion. For shared coefficients `a(sigma)` and `b(sigma)`:

```text
Q_i = a*R_i(x) + b*epsilon_i
F(Q) = a*x + b*F(epsilon)
```

This follows from the partition of unity. The SDXL adapter reuses the inspected
[host conversions][model-sampling] and rejects unknown replacements. A different
prediction family needs its own conversion and timestep check.

## Numerical precision and checks

Kernels, denominators, accumulation and division use at least FP32; FP64 boundary
predictions retain FP64. Weights are never formed in FP16. Singleton non-spatial
axes allow denominator broadcasting, and results return to the prediction dtype.
Normalized coordinates have magnitude below `0.5`, so exact weights exceed
`exp(-25)`. The denominator must be finite and positive; an epsilon clamp would
change valid tiny edge weights.

[Geometry tests](../tests/test_geometry.py) compare against feasible-extent
enumeration and handwritten rectangles. [Fusion tests](../tests/test_fusion.py)
check scalar kernels, symmetry, coverage, constant preservation, CFG and affine
identities, and precision. These CPU checks establish numerical contracts;
[visual acceptance](manual-validation.md) remains pending.

[paper]: https://arxiv.org/html/2302.02412v1#S3
[canvas]: https://github.com/albarji/mixture-of-diffusers/blob/af42292d0a8cb414f6da2eeac79be4c60afbbe48/mixdiff/canvas.py#L187-L200
[samplers]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/samplers.py#L592-L632
[model-base]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_base.py#L208-L257
[model-sampling]: https://github.com/Comfy-Org/ComfyUI/blob/3c80da7f87ee359b2d06f107cb3c0797079dfbbb/comfy/model_sampling.py#L30-L55
