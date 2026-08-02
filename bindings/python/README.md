# Python Bindings and Python Server

This directory contains two ways to use edge-dit.cpp from Python:

- **Python bindings**: call `Engine` directly from a Python program.
- **Python Server**: keep one loaded model in a process and expose image/video jobs over HTTP. The browser console uses this server.

The shortest path for a new machine is: build the shared native library, install the Python package in a virtual environment, set one model path, and start the managed console.

## 1. Requirements

You need:

- Linux with Python 3.10 or newer.
- A C++17 compiler, CMake 3.20 or newer, Git, and the project submodules.
- CUDA Toolkit and a CUDA-capable GPU for `backend=cuda`.
- Node.js and npm for the browser console.
- A Diffusers model directory. The directory should contain `model_index.json` and the model component folders/files. Do not point at a single `.safetensors` file unless you use the separate-component options described below.

Fetch the submodules from an existing checkout:

`@bash
cd /absolute/path/to/edge-dit.cpp
git submodule update --init --recursive
`@

## 2. Build the library Python needs

The normal CUDA build produces `build-cuda/bin/ed-cli`. Python needs a shared library instead. From the repository root run:

`@bash
ED_BUILD_SHARED_LIBS=ON \
BUILD_DIR=build-cuda-shared \
bash scripts/build_cuda.sh
`@

After a successful build, these files should exist:

`@text
build-cuda-shared/bin/libedgedit.so
build-cuda-shared/bin/libggml.so
build-cuda-shared/bin/libggml-base.so
build-cuda-shared/bin/libggml-cpu.so
build-cuda-shared/bin/libggml-cuda.so
`@

The CUDA build script can install compatible cuDNN Python wheels in user space when cuDNN is not already available. CUDA Toolkit and the NVIDIA driver remain system dependencies.

## 3. Install the Python environment

Create a virtual environment in the repository and install the bindings in editable mode:

`@bash
cd /absolute/path/to/edge-dit.cpp
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e './bindings/python[dev]'
`@

The base package installs Pillow. The `dev` extra also installs NumPy and pytest. Use `./bindings/python[numpy]` instead of `./bindings/python[dev]` when you only need NumPy output.

Set the native library variables once in the same shell:

`@bash
export EDGE_DIT_REPO_ROOT="$PWD"
export EDGE_DIT_PYTHON_BIN="$PWD/.venv/bin/python"
export EDGE_DIT_LIBRARY="$PWD/build-cuda-shared/bin/libedgedit.so"
export EDGE_DIT_DEPENDENCY_DIRS="$PWD/build-cuda-shared/bin"
export PYTHONPATH="$PWD/bindings/python/src:$PYTHONPATH"
`@

`EDGE_DIT_LIBRARY` is the exact shared library to load. `EDGE_DIT_DEPENDENCY_DIRS` is a colon-separated list of directories containing CUDA, cuDNN, and sibling ggml shared libraries. The loader also searches common CUDA and NVIDIA Python-wheel locations.

## 4. Configure a model

The managed console reads a model path from an environment variable. The value must be an absolute path to the model directory, not a path to `model_index.json` itself.

| Profile | Environment variable | Model type | Safe starting settings |
| --- | --- | --- | --- |
| `flux-dev` | `EDGE_DIT_FLUX_MODEL_PATH` | FLUX.1-dev image model | CPU parameter offload, text-encoder offload, 8 GiB VRAM cap |
| `flux-kontext` | `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH` | FLUX Kontext image-edit model | Same offload settings as FLUX.1-dev |
| `qwen-image` | `EDGE_DIT_QWEN_IMAGE_MODEL_PATH` | Qwen-Image | `q4_k` weights, CPU parameter/text/VAE offload, 8 GiB VRAM cap |
| `qwen-image-edit` | `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH` | Qwen-Image-Edit | `q4_k` weights, CPU parameter/text/VAE offload, 8 GiB VRAM cap |
| `sd3-medium` | `EDGE_DIT_SD3_MODEL_PATH` | Stable Diffusion 3 Medium | CPU parameter/text-encoder offload, T5 skipped, 8 GiB VRAM cap |
| `wan-t2v` | `EDGE_DIT_WAN_VIDEO_MODEL_PATH` | Wan2.1 T2V 1.3B | CPU parameter/text/VAE offload, 8 GiB VRAM cap |

Example for the default FLUX profile:

`@bash
export EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev
test -f "$EDGE_DIT_FLUX_MODEL_PATH/model_index.json"
`@

The `test` command must succeed. If it fails, fix the path before starting the server. To use another profile, set its variable and pass the profile name to the start command.

## 5. Start the complete frontend and backend

Install the frontend dependencies once:

`@bash
cd "$EDGE_DIT_REPO_ROOT/bindings/python/frontend/server-console"
npm install
`@

Start the frontend, runtime manager, and managed Python Server together:

`@bash
EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev \
npm run dev:managed
`@

The command starts:

| Service | URL | Purpose |
| --- | --- | --- |
| Browser console | `http://127.0.0.1:5173` | React/Vite user interface |
| Runtime manager | `http://127.0.0.1:8090/runtime/v1` | Starts, stops, and monitors model profiles |
| Python Server | `http://127.0.0.1:8080/ed/v2` | Job API used by the console |

For a different model, replace the variable and profile:

`@bash
EDGE_DIT_QWEN_IMAGE_MODEL_PATH=/absolute/path/to/Qwen-Image \
npm run dev:managed -- --auto-start-profile qwen-image
`@

For access from another device on the same network, bind all three services:

`@bash
EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev \
npm run dev:managed:network
`@

This exposes the services on `0.0.0.0`; use the host machine's IP address in the browser. Do not expose these development endpoints to the public Internet without adding authentication and TLS.

Check the services from another terminal:

`@bash
curl http://127.0.0.1:8090/runtime/v1/status
curl http://127.0.0.1:8080/ed/v2/health
`@

The Python Server health response is `status: ok` only after the model has finished loading. Loading a large model can take several minutes and uses system RAM while the model is being prepared.

Stop the complete stack with `Ctrl-C` in the terminal running `npm run dev:managed`.

## 6. Start only the Python Server

Use this when you do not need the browser console:

`@bash
cd "$EDGE_DIT_REPO_ROOT"
. .venv/bin/activate
export EDGE_DIT_LIBRARY="$PWD/build-cuda-shared/bin/libedgedit.so"
export EDGE_DIT_DEPENDENCY_DIRS="$PWD/build-cuda-shared/bin"
export PYTHONPATH="$PWD/bindings/python/src:$PYTHONPATH"

edge-dit-server \
  --model /absolute/path/to/FLUX.1-dev \
  --backend cuda \
  --host 127.0.0.1 \
  --port 8080 \
  --offload-to-cpu \
  --keep-text-encoder-on-cpu \
  --max-vram 8
`@

The equivalent module command is `python -m edge_dit.server`. The server loads one model at startup, accepts jobs, and executes them serially on one worker thread.

## 7. Use the Python binding directly

`@python
from edge_dit import Engine

with Engine(
    model_path="/absolute/path/to/FLUX.1-dev",
    backend="cuda",
    offload_params_to_cpu=True,
    text_encoder_offload=True,
    max_vram_gb=8.0,
) as engine:
    images = engine.generate_image(
        prompt="a glass teapot on a wooden table",
        width=256,
        height=256,
        steps=4,
        seed=42,
    )
    images[0].save("output.png")
`@

`model_path` is the model directory. `backend` selects `cuda`, `cpu`, `vulkan`, `metal`, or `auto` when that backend is available. `offload_params_to_cpu` saves VRAM by keeping parameters in system RAM. `text_encoder_offload` and `vae_offload` apply the same idea to those components. `max_vram_gb` limits the compute placement budget; it does not reduce the model's system-RAM footprint.

## 8. Parameters that matter first

| Parameter | Meaning | Practical first value |
| --- | --- | --- |
| `width`, `height` | Output size in pixels | `256` for a smoke test; use the model's native size for quality |
| `steps` | Denoising iterations | `1` only for wiring checks; `20` is a normal starting point |
| `seed` | Reproducible random seed | `42`; omit it for a random result |
| `guidance` | FLUX distilled guidance | Leave unset unless the model recommends a value |
| `cfg_scale` | Classifier-free guidance for supported pipelines | Model-specific; often `1` or `5` |
| `frames` | Number of video frames | `9` for a smoke test |
| `weight_type` | On-the-fly weight format such as `q4_k` | `auto`; use `q4_k` when VRAM is tight |
| `vae_tiling` | Decode in tiles to reduce peak VRAM | `auto` |
| `cache_mode` | Optional computation reuse method | `disabled` until the baseline works |

Do not lower `steps` permanently to solve an out-of-memory error. First use CPU/offload options, reduce the image or video size, and select a supported quantized weight type.

## 9. HTTP jobs at a glance

The Python Server uses the `/ed/v2` protocol. `v2` is the HTTP contract version; it is not the product name. A request creates a job immediately, then the client polls that job until it succeeds, fails, or is cancelled.

`@bash
curl -s http://127.0.0.1:8080/ed/v2/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a glass teapot","width":256,"height":256,"steps":4,"seed":42}'
`@

The response contains `id`, `status_url`, and `result_url`. Poll `status_url`; when `status` is `succeeded`, fetch `result_url` and decode `data[].b64_png`. See [docs/api.md](../../docs/api.md) for every endpoint, field, error, and lifecycle state.

## 10. Troubleshooting

- **`ModuleNotFoundError: edge_dit`**: activate the virtual environment and set `PYTHONPATH` as shown above.
- **Cannot load `libedgedit.so`**: rebuild with `ED_BUILD_SHARED_LIBS=ON` and check `EDGE_DIT_LIBRARY`.
- **`model_index.json` not found**: point the profile variable at the model directory, not its parent or one weight file.
- **Backend stays `starting`**: wait for model loading; inspect the runtime manager status and its log tail before restarting.
- **CUDA or cuDNN errors**: verify `nvcc --version`, the NVIDIA driver, and `EDGE_DIT_DEPENDENCY_DIRS`. A Python package install cannot replace the system CUDA compiler or driver.
- **Out of memory**: use the profile's offload settings, reduce size/frames, then select `q4_k` where supported.

Run the fast local checks with:

`@bash
PYTHONPATH=bindings/python/src python -m unittest discover -s bindings/python/tests -p 'test_server*.py' -v
`@
