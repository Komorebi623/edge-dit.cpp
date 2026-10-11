# Third-Party Notices

edge-dit.cpp depends on and includes code from third-party projects. The
project license applies to edge-dit.cpp code; third-party components remain
under their own licenses.

## Bundled Source Components

| Component | Location | License | Notes |
|---|---|---|---|
| ggml | `third_party/ggml` | MIT | Included as a Git submodule. See `third_party/ggml/LICENSE`. |
| nlohmann/json | `third_party/json.hpp` | MIT | Single-header JSON library. |
| cpp-httplib | `third_party/httplib.h` | MIT | Single-header HTTP server/client library. |
| stb_image_write | `third_party/stb_image_write.h` | Public domain or MIT | Single-header image writer. |

## Optional / Fetched Components

| Component | Location | License | Notes |
|---|---|---|---|
| NVIDIA cuDNN frontend | `third_party/cudnn-frontend` or CMake FetchContent | MIT | Used only when cuDNN SDPA support is enabled. See `third_party/cudnn-frontend/LICENSE.txt` when present. |
| NVIDIA CUDA, cuDNN, NCCL, MPI runtimes | System or Python environment | Vendor-specific | Not distributed by this repository. Users must install and comply with the relevant vendor licenses. |

## Derived or Referenced Implementations

- `src/utils/rng_mt19937.hpp` includes attribution to PyTorch for the original
  implementation reference.
- `src/dit_models/components/text_encoders/tokenizers/t5_unigram_tokenizer.cpp`
  includes attribution to SentencePiece for the original license reference.

Please preserve upstream notices when redistributing source or binaries.

## Model Weights

This repository does not include model weights. Supported model families such
as SD3/SD3.5, FLUX, Qwen-Image, Qwen-Image-Edit, FLUX-Kontext, and Wan are
distributed by their respective owners under separate terms. Users are
responsible for reviewing and complying with those model licenses and usage
policies.

## MiniMax-H3 Optimization References

The H3-only optimization implementations and CPU mathematical oracles reference
Apache-2.0 code in SGLang, Cache-DiT, FastVideo, OpenVDN and SageAttention.
Fixed revisions and upstream license locations are recorded in
`benchmark/h3_cuda/reference-lock.json` and
`docs/research/minimax-h3-optimization-sources.md`. The optional local selected
source snapshots retain upstream LICENSE files; they are not runtime model
weights or a complete redistributable upstream installation. Preserve per-file
notices when redistributing derived code. MiniMax-H3, FastH3, Turbo adapters and
VDN model weights have separate model-card/license requirements; a code license
does not establish permission to use or redistribute a weight artifact.
