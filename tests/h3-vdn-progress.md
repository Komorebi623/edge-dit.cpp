# VDN-H3 native ggml component delivery — 2026-10-11

**Status: CPU-testable native components, NOT completed VDN-H3 deployment.**

Write scope is limited to:

- `src/dit_models/models/minimax_h3_vdn_graph.hpp`
- `tests/h3_vdn_graph_test.cpp`
- `tests/h3-vdn-progress.md`

No model, pipeline, existing reference header/test, CMake, GPU code or checkpoint was changed by this component task. Main-thread CMake already registers `h3-vdn_graph-test` / CTest `h3-vdn_graph`. No GPU execution, process termination, or model download was performed.

## Actual implemented math

All feature/statistic/factor/recurrence/readout/softmax arithmetic is expressed using real ggml ops, with no custom host-reference callback and no copied oracle outputs. The test executes the CPU graph and independently compares its tensors to `VDNReference` or explicit scalar expected values.

- `frame_statistics`: `A[k,l] = sum_s K[s,k] beta[s] K[s,l]`; `B[v,k] = sum_s V[s,v] beta[s] K[s,k]`. F32 operands/results; explicit contraction-axis repacks support strided frame/head input views.
- `factor(SanaScaled)`: `T = Diag(alpha) (I - A/S)`, `J = B/sqrt(S)`. Row scaling of transition is distinct from key-column bridge scaling.
- `apply`: actual `state @ T + J`, including rectangular `dv != dk` states.
- `outside_window_states`: both forward/reverse state banks; query reads only `prefix[lo-1]` / `suffix[hi+1]`, never the window injection. Alpha bridge uses clamped log-prefix sums and exponentiated differences; `bridge=none` also supported.
- Full-window double-count guard: if **every** query window covers the clip, `linear_active=false`, no scan banks are built, and outside tensors are exact zero. This guard also disables a text-conditioned linear branch, matching official hybrid dispatch semantics. A single full-cover query does not deactivate other queries.
- Optional `full_text_injection`: both scans start from half the full prompt injection; absent boundary neighbors read that same half state. This argument corresponds to official `_text_chunk_state`, **not** already-halved `_text_state`.
- `window_bounds`: saturating frame-radius and official chunk-aligned bounds, including a short final chunk.
- `readout`: `state[v,k] @ query[k]` returns value channels.
- `linear_features`: conv-less SiLU and optional Q/K L2 normalization with official `eps=1e-6`, including zero and sub-epsilon rows. Input must be pre-RoPE raw Q/K/V.
- `norm_gate`: weighted per-head RMS norm and supplied output gate; shared channel norm weights; RMS epsilon must be explicit, never guessed.
- `local_softmax`: exact stable ggml softmax. Global prefix/suffix (text/audio) query rows remain dense. Every video query sees all global keys plus its inclusive local video window exactly once. Full coverage takes one dense graph path.

## Layout and request contracts

`MiniMaxH3::VDNGraph::LayoutContract` uses ggml fastest-first axes:

| Tensor | ggml axes |
| --- | --- |
| Q / K prepared linear features | `[dk, S, H, F]` |
| V | `[dv, S, H, F]` |
| beta | `[1, S, H, F]` |
| alpha | `[dk, 1, H, F]` |
| A / transition | `[dk, dk, H, F]` |
| B / injection / state bank | `[dk, dv, H, F]` |
| one frame's readout | `[dv, S, H, 1]` |
| flattened local-softmax Q/K | `[d, total, H, 1]` |
| flattened local-softmax V/output | `[dv, total, H, 1]` |

Shapes/type and contiguous channel-axis requirements fail with `invalid_argument` before unsuitable ggml operations. Frame/head and token strides may be non-contiguous and are repacked. Runtime values are **not** copied to the host to validate: caller must supply finite sigmoid beta and alpha, and prepared linear features. The F32 tiny oracle cases validate correctness; mixed BF16 production numerical equivalence is not established.

`GraphRequest.requireweights` defaults to **true**. `build_frame_factors` validates the explicit checkpoint rule, named payload, LoRA-applied preflight, and matching checkpoint/layout dimensions. `requireweights=false` is exclusively the explicit math-test mode; it does **not** bypass unsupported-rule refusal. No inference/render entrypoint exists in this header.

## Exact solve refusal — not a Sana replacement

Official released checkpoints use `delta_rule=vdn_solve`:

`T = Diag(alpha) (I+A)^-1`, `J = B (I+A)^-1`.

Official code uses batched Cholesky, triangular solves and matrix multiplication (cuSOLVER/cuBLAS in the CUDA stack). This ggml build lacks the matching Cholesky/triangular-solve graph backend. `require_native_backend`, `factor`, `validate_request`, and `build_frame_factors` therefore reject `vdn_solve` and `vdn_scaled`. An unknown or missing rule also rejects. There is no fake capability toggle, inverse approximation, token-wise delta surrogate, or fallback to Sana, including when `requireweights=false` or a schema-complete payload is supplied. A real matching cuSOLVER-backed graph implementation remains required for the original released solve checkpoints.

## Public named-payload adapter

C++ type: `MiniMaxH3::VDNGraph::PublicVdnPayloadAdapter`.

Exposed adapter identifier: `OpenVDN.vdn-minimax-h3.public-named-payload.v1` (a **local component identifier**, not an upstream file-format version).

Official artifact locations: `linear_branch/`, `adapters/default/`, optional `adapters/turbo/`. The adapter consumes caller-provided named tensors after a loader has decoded artifacts; it does not read files. Exact checkpoint attention prefix and LoRA targets/adapters/rank must come from checkpoint/config metadata. The schema covers:

- `linear_attention.alpha.{down.weight,up.weight,A_log,dt_bias}`
- `linear_attention.beta_proj.weight`
- `linear_attention.output_gate.{down.weight,up.weight,up.bias}`
- `linear_attention.norm.weight`
- `to_out_linear.weight`
- Optional `softmax_gate.up.{weight,bias}`
- Paired PEFT `.lora_A.<adapter>.weight` / `.lora_B.<adapter>.weight` for configured backbone or token-refiner Q/K/V/O projection targets.

The schema checks required names, reversed PyTorch-to-ggml axes, float storage types, paired LoRA ranks, duplicate configurations, and rejects extra/missing keys. Current LoRA shape support is square hidden-width Q/K/V/O; unsupported targets reject. `short_conv=true` explicitly rejects rather than ignoring its spatial/temporal weights.

**Schema validity does not prove trained provenance, verify an artifact digest, load or merge LoRA, or perform model inference.** Shape-only fixtures are never computed as model weights. There is no random-weight inference path. A loader must establish checkpoint provenance, and actual feature/gate/projection computation remains an integration task.

## CPU verification performed

After the main thread renamed the oracle function for GCC13 `std::apply` ADL compatibility, all test calls use `R::apply_state`. Graph recurrence calls explicitly use `::MiniMaxH3::VDNGraph::apply`. No reference-header edit was made by this task.

1. Host standalone GCC12: `-std=c++17 -O2 -Wall -Wextra -Werror`; links **only** existing CPU `libggml-cpu` and `libggml-base`.
2. Existing development container GCC **13.3.0**: rebuilt main-thread CMake target `h3-vdn_graph-test`, then ran CTest `h3-vdn_graph`; `ED_GGML_CUDA=OFF`, `GGML_CUDA=OFF`, `CUDA_VISIBLE_DEVICES=""`.

Actual result (both):

```text
PASS h3_vdn_graph: 36 CPU actual graphs, 2928 native nodes, 27159 checks; max_abs_error=2.38419e-07
Math/schema components only; VdnSolve requires cuSOLVER; NOT completed deployment.
```

CTest graph target: **1/1 passed**; combined `h3-vdn` + `h3-vdn_graph`: **2/2 passed**.

GCC13 standalone AddressSanitizer + UndefinedBehaviorSanitizer (`-O1 -g -fsanitize=address,undefined`, leak checking enabled) also passed with the same result. The container preloads tcmalloc; the sanitizer run removed `LD_PRELOAD` for that test process only so ASan could initialize first. Prebuilt ggml libraries were not sanitizer-instrumented. Host sanitizer linking was unavailable; the successful sanitizer result is from the existing development container. Runtime dependency inspection of both standalone and CMake executables showed no CUDA library.

The test checks for native math operators and rejects `GGML_OP_CUSTOM` nodes. Tolerance is `3e-5 + 3e-5 * abs(expected)`. Cases include two heads, five frames, signed/non-symmetric B, continuous and interleaved strides, nonuniform channel alpha, bridges on/off, clipping boundaries, rectangular states, local injection perturbation, full cover with/without text, uneven windows, clamp behavior, zero/sub-epsilon features, norm/gate epilogue, protected prefixes/suffixes, chunk-aligned windows, schema failures and solve refusal.

### Reproduce standalone without changing CMake

From the repository root, using the already-built CPU libraries:

```bash
export CUDA_VISIBLE_DEVICES=""
/usr/bin/g++ -std=c++17 -O2 -Wall -Wextra -Werror \
  -Isrc -Ithird_party/ggml/include tests/h3_vdn_graph_test.cpp \
  -Lbuild-h3-cpu/bin -Wl,-rpath,"$PWD/build-h3-cpu/bin" \
  -lggml-cpu -lggml-base -o /tmp/h3_vdn_graph_test
/tmp/h3_vdn_graph_test
```

Main-thread CMake target:

```bash
CUDA_VISIBLE_DEVICES="" cmake --build build-h3-cpu --target h3-vdn_graph-test -j 2
CUDA_VISIBLE_DEVICES="" ctest --test-dir build-h3-cpu -R '^h3-vdn_graph$' --output-on-failure -V
```

## Honest remaining gaps

- Original `vdn_solve` / `vdn_scaled` deployment: blocked on a real exact solve backend, not solved by the native Sana test arm.
- Released short-conv K/V spatial+temporal graph and schema support, anchor `both` input pruning/rebasing/scatter and anchor-column softmax exclusion: **not implemented**. Current local softmax contract is anchor `none`; upstream anchor configurations must not be treated as supported by these primitives.
- Full public-artifact loader/metadata validation/checksum, stage-B/default and DMD/turbo LoRA application, actual learned alpha/beta/output-gate projections, output projection and model/pipeline integration: **not implemented**.
- No GPU latency, capture, memory scalability, large-checkpoint fidelity, full model render or deployment validation. CPU graph uses out-of-place scans; no claim of production inference-state bank optimization.

Source audit is the local pinned snapshot at `_scratch/h3-reference-src/vdn/e262cb5770b3dc992d4f44f251cd17bbfe0c4d8b/`, specifically `src/models/hybrid_attention.py`, `linear_attention/{branch.py,scan.py,delta_rule.py,features.py,layers.py}`, `attention_gates.py`, `hybrid_transform.py`, stage-B config and README. The existing oracle is comparison-only and is not included by the graph helper.
