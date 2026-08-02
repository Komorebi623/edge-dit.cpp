# Python Server Runtime Configuration

This page configures the managed Python Server used by the browser console. It does not require changing any tracked profile JSON file.

## One-time environment

Run these commands after activating the Python virtual environment and building `build-cuda-shared`:

`@bash
cd /absolute/path/to/edge-dit.cpp
export EDGE_DIT_REPO_ROOT="$PWD"
export EDGE_DIT_PYTHON_BIN="$PWD/.venv/bin/python"
export EDGE_DIT_LIBRARY="$PWD/build-cuda-shared/bin/libedgedit.so"
export EDGE_DIT_DEPENDENCY_DIRS="$PWD/build-cuda-shared/bin"
`@

Set these in your shell profile only after confirming the paths work. Keep model paths machine-local; do not commit them into `runtime/profiles/*.json`.

## Select a model profile

| Profile argument | Required variable | Expected directory |
| --- | --- | --- |
| `flux-dev` | `EDGE_DIT_FLUX_MODEL_PATH` | FLUX.1-dev Diffusers directory |
| `flux-kontext` | `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH` | FLUX.1-Kontext-dev Diffusers directory |
| `qwen-image` | `EDGE_DIT_QWEN_IMAGE_MODEL_PATH` | Qwen-Image Diffusers directory |
| `qwen-image-edit` | `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen-Image-Edit Diffusers directory |
| `sd3-medium` | `EDGE_DIT_SD3_MODEL_PATH` | Stable Diffusion 3 Medium Diffusers directory |
| `wan-t2v` | `EDGE_DIT_WAN_VIDEO_MODEL_PATH` | Wan2.1 T2V 1.3B Diffusers directory |

Every value should be absolute and should satisfy:

`@bash
test -f "$EDGE_DIT_FLUX_MODEL_PATH/model_index.json"
`@

For example:

`@bash
export EDGE_DIT_FLUX_MODEL_PATH=/models/FLUX.1-dev
export EDGE_DIT_WAN_VIDEO_MODEL_PATH=/models/Wan2.1-T2V-1.3B-Diffusers
`@

## Start and switch

Start the default `flux-dev` profile:

`@bash
cd bindings/python/frontend/server-console
npm run dev:managed
`@

Start another profile immediately:

`@bash
npm run dev:managed -- --auto-start-profile wan-t2v
`@

After the console starts, the Local Runtime panel can start, stop, or switch among profiles whose model variables are set. It shows the child process log tail and health result, which is the first place to inspect a failed load.

## Network bindings

The default addresses are local only:

| Variable | Default | Meaning |
| --- | --- | --- |
| `EDGE_DIT_FRONTEND_HOST` | `127.0.0.1` | Vite browser console address |
| `EDGE_DIT_RUNTIME_MANAGER_HOST` | `127.0.0.1` | Runtime manager address |
| `EDGE_DIT_MANAGED_BACKEND_HOST` | `127.0.0.1` | Python Server address |
| `EDGE_DIT_RUNTIME_MANAGER_PORT` | `8090` | Runtime manager port |
| `EDGE_DIT_MANAGED_BACKEND_PORT` | `8080` | Python Server port |

Run `npm run dev:managed:network` to bind all three to `0.0.0.0`. Treat that as trusted-LAN development only. The built-in server has no authentication, authorization, TLS, rate limiting, or persistent job storage.

## Runtime behavior

The profile files specify model family and conservative defaults. Most profiles use CPU parameter offload and an 8 GiB VRAM cap so they can load on smaller GPUs. This reduces VRAM use but requires substantial system RAM and makes startup slower.

The Python Server starts listening only after the model has loaded. Confirm readiness with:

`@bash
curl http://127.0.0.1:8090/runtime/v1/status
curl http://127.0.0.1:8080/ed/v2/health
`@

The manager restarts an unexpectedly exited backend a limited number of times. Fix the reported model path, native library, CUDA, or memory problem before repeatedly retrying.

For complete setup and generation parameters, return to [the Python guide](../../README.md).
