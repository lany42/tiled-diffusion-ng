# Image resizing and precision

Research recorded on 2026-09-20. The source references below describe inspected
behavior, not exact runtime version requirements. CPU checks used Python 3.13
and PyTorch 2.14.0+cpu; real-host Krea2 Raw/Turbo visual acceptance remains pending.

## ComfyUI's native Lanczos path

The native [ImageScale and ImageScaleBy nodes][image-nodes] delegate to
`comfy.utils.common_upscale`. In the inspected [Lanczos implementation][utils],
pixels take this route for both upscaling and downscaling:

1. Move to CPU, multiply by 255, clip, and cast to NumPy `uint8`.
2. Resize with Pillow's Lanczos filter.
3. Convert to float32, divide by 255, and restore the input device and dtype.

A float32 output therefore does not imply float32 pixel precision throughout
the operation. Quantization discards fractional color information even when the
input was float32. The NumPy conversion also rejects bfloat16 input before the
resize can run. Promoting bfloat16 to float32 fixes that conversion error but
does not remove the subsequent uint8 quantization. This is host behavior; the
pack does not patch ComfyUI's resize nodes.

## PyTorch's floating-point resize API

[`torch.nn.functional.interpolate`][interpolate] supplies bicubic, bilinear,
area and nearest-neighbor resizing directly on float32 tensors. The image
layout is BCHW; ComfyUI IMAGE tensors use BHWC, so `movedim` converts the layout
without changing precision. Bicubic and bilinear support `antialias=True` with
`align_corners=False` for downscaling. Cubic filtering can overshoot the image
range, so normalized RGB output can be clamped to `[0, 1]` in floating point.

ComfyUI's inspected `common_upscale` already delegates bicubic, bilinear and
area to PyTorch, but does not enable antialiasing. Its Lanczos branch is separate.
PyTorch 2.14 also documents a CPU-only Lanczos mode requiring antialiasing;
availability on the user's host must be checked before relying on that mode.

## TileKrea2Conditioning policy

The [helper](../src/tiled_diffusion_ng/_krea2_conditioning.py) uses PyTorch's
built-in bicubic interpolation with antialiasing when `downsize_to_1mp` is enabled
and a tile exceeds `1024²` pixels. It preserves aspect ratio, rounds dimensions
to positive integers, performs no crop, and never enlarges smaller tiles.

Float32 pixels remain float32; float64 remains float64. Lower-precision inputs,
including float16 and bfloat16, are promoted to float32 for this resize and kept
at float32 afterward. The result is clamped to `[0, 1]` without integer
quantization. When no resize is needed, RGB values and dtype remain unchanged.
RGBA inputs use their RGB channels. Source tensors are never modified.

This removes the reviewed bfloat16-to-NumPy failure path. The
[CPU regression tests](../tests/test_krea2_conditioning.py) exercise real PyTorch
interpolation for all four dtypes, check fractional colors that an 8-bit round
trip would lose, and verify source preservation. They separately cover resize
dimensions, the disabled/small-image paths, cutoff zero, and RGB bounds.
Native Qwen preprocessing and encoder precision remain host-owned.

## Other image paths in the pack

| Path | Precision handling |
| --- | --- |
| [TileView](../src/tiled_diffusion_ng/geometry.py) | Crops and stacks without changing pixel values or dtype. |
| [Anima LLLite routing](../src/tiled_diffusion_ng/adapters/_anima_lllite.py) | Passes RGB references to the native patch without casting. The [native patch][lllite] resizes with bicubic, then converts to the model input's device and dtype for encoding. |
| [SDXL ControlNet hints](../src/tiled_diffusion_ng/adapters/_native_control.py) | Delegates full-canvas resizing to the control's selected native algorithm, then crops without a pack-imposed dtype conversion. Native Lanczos retains the quantization described above if selected. |
| [Sampler fusion](../src/tiled_diffusion_ng/fusion.py) | Accumulates predictions in at least float32, preserves float64 accumulation, and returns the native prediction dtype. These are latent predictions, not IMAGE pixels. |

[image-nodes]: https://github.com/Comfy-Org/ComfyUI/blob/c194dd00cd42aa18d9dbf27d977bf6b85d9ea565/nodes.py#L1883-L1933
[utils]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/utils.py#L1097-L1136
[interpolate]: https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.interpolate.html
[lllite]: https://github.com/Comfy-Org/ComfyUI/blob/944386c233e02eaf877b1c8d5d513fb3d3a4d5e3/comfy/ldm/anima/lllite.py#L206-L235
