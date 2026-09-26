If a push is requested, only push to `origin`. Never push to a mirror remote unless the user explicitly overrides this rule.

Do NOT update the README unless explicitly requested.
Do NOT add or update documention in docs/ unless explicitly requested.

# Development workflow

Run from the repository root. Use Python 3.13+ and uv; `uv sync --locked`
installs the CPU development environment. Update pyproject.toml and uv.lock
together. After Python changes, run these checks in order:

```sh
uv run --offline --locked ruff check --select I --fix .
uv run --offline --locked ruff check --fix .
uv run --offline --locked ruff format .
uv run --offline --locked ruff check .
uv run --locked pytest
uv lock --check
uv build
```

ComfyUI supplies runtime PyTorch and comfy_api. Keep runtime dependencies empty
unless an actual new dependency is needed; do not replace the host's CUDA build.
Development PyTorch uses the explicit CPU index. Tests use real CPU tensors and
small host doubles, with no host installation, checkpoint, GPU, network, or
sibling checkout. Keep implementation in src/tiled_diffusion_ng and the root
loader for clone/ZIP loading. Include the loader, tests, lockfile, and guides
in source distributions; include LICENSE and COPYRIGHT in both builds.

Keep geometry and fusion independent of host APIs. 
One common_ksampler call owns each trajectory. Read live sampler registrations 
and preserve compatible global CFG hooks. Reject unsupported spatial capabilities 
explicitly. Keep inputs unchanged and release all invocation state on success, failure, and cancellation.

ComfyUI compatibility is best effort across releases and development HEAD.
Commit-pinned host references document inspected behavior; they are not an
exact-version requirement. Base support decisions on required APIs and semantic
capabilities. Recheck host wrapper routing, conditioning preparation, control
state, and V3 execution-list behavior when updating the compatibility baseline.
Keep the offline CPU contract tests separate from evidence of real-host
compatibility; their doubles cannot detect upstream changes by themselves.
See [ComfyUI compatibility](docs/comfyui-compatibility.md) for inspected revisions,
the contract matrix, hint-sharing assumptions, and pending host checks.

Every Python source and test starts with:

```python
# SPDX-License-Identifier: AGPL-3.0-only
# SPDX-FileCopyrightText: 2026 Lany Atwood <lany@colorized.life>
```

Use Conventional Commits: `<type>[optional scope][!]: <summary>`. Write an
imperative subject of at most 50 characters, with no trailing period. Follow
it with one blank line and a single short paragraph explaining what changed
and why in plain language. Limit the body to four lines, each at most 72
characters; avoid lists and exhaustive change logs.
