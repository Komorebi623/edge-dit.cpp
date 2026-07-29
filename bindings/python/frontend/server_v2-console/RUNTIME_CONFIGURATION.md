# Runtime Configuration Guide

This document covers just one thing:

- when migrating the `server_v2` Web Console to another machine,
- how to complete the configuration with the fewest changes,
- and distinguishing between the simple approach and the complex approach.

The recommended order is simple:

1. Use an existing profile first
2. Change environment variables first
3. Add a local profile only if it still doesn't run reliably

## 1. What can currently be configured

In the current implementation, the runtime configuration mainly comes from two places:

- `runtime/profiles/*.json`
- local environment variables

Their division of labor is:

- the profile handles the model type and default engine parameters
- environment variables handle machine-local path overrides

The most commonly used scripts are:

- [scripts/runtime-env.sh](scripts/runtime-env.sh)
- [scripts/run-managed-profile.sh](scripts/run-managed-profile.sh)

## 2. Simple approach: change only environment variables

This is the most recommended first step.

Applicable when:

- the model hasn't changed
- only the paths have changed
- the new machine differs little from the current validation machine

### 2.1 What you need to prepare

First confirm the machine already has:

- the repository code
- `libedgedit.so`
- a usable Python
- CUDA / cuDNN dependencies
- the model directory

### 2.2 Variables you need to set

First set the base variables:

```bash
export EDGE_DIT_REPO_ROOT=/path/to/edge-dit.cpp
export EDGE_DIT_PYTHON_BIN=/usr/bin/python3
export EDGE_DIT_LIBRARY=/path/to/edge-dit.cpp/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_DEPENDENCY_DIRS=/path/to/cudnn/lib:/path/to/cuda_nvrtc/lib:/path/to/cublas/lib:/path/to/cuda_runtime/lib:/path/to/edge-dit.cpp/build-cuda-shared/bin
```

Then set the model path variables. The common correspondences are:

- `flux-dev` -> `EDGE_DIT_FLUX_MODEL_PATH`
- `flux-kontext` -> `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH`
- `qwen-image` -> `EDGE_DIT_QWEN_IMAGE_MODEL_PATH`
- `qwen-image-edit` -> `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH`
- `sd3-medium` -> `EDGE_DIT_SD3_MEDIUM_MODEL_PATH`
- `wan-t2v` -> `EDGE_DIT_WAN_VIDEO_MODEL_PATH`

For example:

```bash
export EDGE_DIT_FLUX_KONTEXT_MODEL_PATH=/models/FLUX.1-Kontext-dev
export EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH=/models/Qwen-Image-Edit
export EDGE_DIT_WAN_VIDEO_MODEL_PATH=/models/Wan2.1-T2V-1.3B-Diffusers
```

### 2.3 How to start

Run in the `bindings/python/frontend/server_v2-console` directory:

```bash
npm run dev:managed
```

If you want to directly start a specific profile:

```bash
npm run dev:managed -- --auto-start-profile flux-kontext
```

### 2.4 How to operate in the frontend

You don't need to hand-write these parameters in the frontend.

You only need to:

1. Open the console
2. Select a model under `Local Runtime`
3. Click start or switch

If the simple approach works, there's no need to keep tinkering.

## 3. Complex approach: add a local profile

Only in the following cases is the complex approach recommended:

- the new machine has noticeably less or more VRAM
- an existing profile OOMs
- you need to change the offload strategy
- you need to change `weight_type` or `max_vram_gb`

### 3.1 How to do it

Add a new local profile file under the following directory:

```text
bindings/python/frontend/server_v2-console/runtime/profiles/
```

Recommended naming:

- `flux-kontext-12gb-local.json`
- `qwen-image-edit-24gb-local.json`

### 3.2 A minimal example

```json
{
  "slug": "qwen-image-edit-12gb-local",
  "name": "Qwen-Image-Edit (12GB Local)",
  "kind": "image",
  "model_env": "EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH",
  "engine": {
    "model_path": "/models/Qwen-Image-Edit",
    "backend": "cuda",
    "weight_type": "q4_k",
    "offload_params_to_cpu": true,
    "keep_text_encoder_on_cpu": true,
    "keep_vae_on_cpu": true,
    "max_vram_gb": 12.0
  }
}
```

It's recommended to change only the fields that are truly related to machine resources:

- `max_vram_gb`
- `offload_params_to_cpu`
- `keep_text_encoder_on_cpu`
- `keep_vae_on_cpu`
- `weight_type`
- `backend`

### 3.3 Recommended rules

- Prefer copying the closest existing profile
- Keep `model_env` unchanged as much as possible
- Do not edit machine-private paths into a shared profile
- Keep local profiles untracked, or add them to `.git/info/exclude`

## 4. Complete flow for users on a new machine

The recommended order is:

1. Set the environment variables first
2. Reuse an existing profile first
3. Run `npm run dev:managed`
4. Do a minimal smoke test in the frontend
5. If it's only a path issue, you're done here
6. If you hit VRAM or loading-strategy issues, then create a local profile

In one sentence:

> Solve path issues with environment variables first, then solve machine differences with a local profile.

## 5. Boundaries of the current approach

What already exists:

- existing profiles
- environment-variable overrides for model paths
- the runtime manager starting up by profile

What doesn't exist yet:

- editing startup parameters directly in the browser
- automatically saving local overrides
- one-click import/export of machine configurations

So the most reliable approach for now is still:

> Get the simple approach working first, then customize with the complex approach.
