# H3 LoRA candidate recipes — 2026-10-14

## Decision / limits

**No production changes or inference runs. Stop at recipe selection until campus authentication is restored or the user supplies local weights.** No weights or weight headers were downloaded; no authentication workaround was attempted. All successful new source/metadata requests used verified TLS. Raw GitHub certificate failures were recorded, not bypassed; small publisher sources were subsequently obtained through the verified-TLS GitHub API. No further network access after the user's stop instruction.

**Prefer Larry v4-step600 EMA at 4 NFE for static/small-motion content, provided the complete adapter including adaLN is supported; use 8 NFE for the same checkpoint if fast motion smears.** These quality preferences are the publisher's claims, not local Q4 results. **LightX2V v0.1 at 4 NFE is the simpler attention/MLP-only integration alternative.** Do not ship a partial Larry adapter as the documented recipe. Do not select v1.2 merely by filename. [S1–S6]

## Candidate settings

| Candidate | NFE / video shift / audio shift | Sampler and guidance | LoRA strength / alpha | Required subset and qualification |
|---|---|---|---|---|
| Larry `minimax_h3_turbo_v4_step600_ema.safetensors` | **4 / 12 / 3**; **8 / 12 / 3** quality fallback | Euler, `simple`, denoise 1, one conditional forward per step (CFG-equivalent 1) | **strength 1.0; alpha = rank**, hence coefficient 1 on `B @ A` | FL2VA base, T2VA/I2VA. Apply every released pair, including block and final-layer adaLN projections. Numeric ranks and exact key inventory remain header-unverified. [S1–S3] |
| LightX2V `minimax_h3_fl2v_turbo_4step_v0.1.safetensors` | **4 / 12 / 3** | Euler / shared-grid dual-sigma `simple` equivalent; no extra unconditional forward | **runtime scale 1.0; alpha 8** for the original Diffusers/PEFT file; coefficient **8/rank**, NOT automatically 1 | FL2VA/T2VA. Published loader targets `to_q`, `to_k`, `to_v`, `to_out.0`, `ff.net.0.proj`, `ff.net.2` in the active transformer. Exact ranks/keys and any converted-file alpha require the actual header. [S4–S7] |
| LightX2V `minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors` (also publisher ComfyUI variant) | **Unresolved.** “4step / 768p” is confirmed as a filename, not a complete validated recipe | Unresolved for this revision | Unresolved for this revision | Present in the official ModelScope file listing on October 14, 2026. Saved/current-pinned publisher guide still documents **v1.0**, not v1.2. Do not silently carry over v1.0's 6/3 shifts and alpha 128. [S4–S6,S9] |

Larry: README specifies useful range 4–8, minimum 4, strength 1.0; 6–8 improves fast-motion smear. It recommends runtime application (`low_vram=false`) rather than merging on quantized bases. It says BF16 and alpha=rank, but does not establish this local Q4 artifact's compatibility. Its saved example workflow still names an older ckpt500 and uses 6 steps: use the explicit v4 README recommendation, not that stale preset. [S1–S3]

Light v0.1: publisher model table specifies 544p mixed-aspect training, 4 recommended NFE and shifts 12/3. Its loader derives rank from actual tensors and rejects unsupported/missing targets. Preserve split/fused QKV and FFN mappings without dropping trained updates; no source-backed smaller subset is offered. For immutable Q4, runtime residual application is an implementation choice, not a publisher-validated Q4 result. [S4–S6]

## Exact sigma grids (terminal zero included; zero is not a forward)

Define `q_i=(N-i)/N`, `sigma_s(q)=s*q/(1+(s-1)*q)`, for `i=0..N`. Video and audio use the **same q grid**, each shifted independently. [S2,S4,S7]

- **4 NFE, shifts 12/3** (Larry and Light v0.1):
  - video: `[1, 36/37, 12/13, 4/5, 0]`
  - audio: `[1, 9/10, 3/4, 1/2, 0]`
- **8 NFE, shifts 12/3** (Larry fallback):
  - video: `[1, 84/85, 36/37, 20/21, 12/13, 36/41, 4/5, 12/19, 0]`
  - audio: `[1, 21/22, 9/10, 5/6, 3/4, 9/14, 1/2, 3/10, 0]`

C++ `steps=N` already produces N forwards with CFG 1 and N+1 grid points. The saved Light Diffusers launcher passes **N+1** to its scheduler API because that API counts grid points; do **not** increase C++ steps to 5/9. [S6:1139–1143,S8:284–291,1808–1813]

## Sampler audit: preserve coordinates, not a second audio correction

**No critical mismatch found in the existing Euler/simple math.** There are two internally consistent representations; their superficial difference is not a Larry-versus-Light incompatibility. [S2:93–108,S3:85–156,S7:152–189,S8]

1. **Ordinary C++ Euler path** (`turbo_enabled_ == false`, not RES): updates video by `delta_sigma_v * velocity_v` and audio by `delta_sigma_a * (velocity_a / audio_slope)`. This directly matches Larry standalone's legacy dual-clock sampler. The C++ model already negates the data-ward prediction and multiplies audio by the slope. Light's native scheduler uses data-ward predictions with `(sigma - sigma_next)`, so its opposite-looking sign is consistent. [S2:93–108,S7:152–189,S8:1921–1926,S10:1212–1213]
2. **Existing Turbo Euler path**: rescales audio input by `a/s`, transforms raw audio velocity, steps the packed state on video sigma, and divides final audio by `k=video_shift/audio_shift`. This matches the *kind* of scaled representation described by Larry's recent native-AV branch. **Do not add Larry's legacy audio correction to it:** that double-corrects the clock. [S3:85–126,S8:1815–1819,1883–1889,1917–1920,1937–1939]

Algebra check (audit-derived, not GPU parity): let `D(s)=k-(k-1)*s`, `a=s/D`, physical audio `y`, packed audio `w=D*y`, and raw sigma-a velocity `u`. Existing Turbo uses `w_next=w+delta_s*((1-k)*y+(1+(k-1)*a)*u)`. Converting back gives **exactly** `y_next=y+(a_next-a)*u`, including endpoint `y_final=w_final/k`. Exact-rational scalar assertions passed for 4/8 steps at 12/3 and 4 steps at 6/3; results saved in `schedule-and-scalar-check.json` [S11]. This is not floating-point/model/Q4 validation.

**Actual integration hazards:** explicitly select **Euler + simple + CFG 1**; the non-Turbo scheduler default is `discrete`, and RES is not the publisher recipe. Do not switch on the existing offline-merged Turbo artifact gate merely to attach dynamic LoRA: it requires checksummed merged shards, not an immutable Q4 adapter. The ordinary Euler path already has the necessary dual-clock update. Larry's adaLN updates must reach the real time-conditioned computation (special handling for pruned/curve bases); an attention/MLP-only loader is incomplete for Larry. [S2:218–235,S3:378–423,491–523,S8:654–692,1430–1438,1965–1966]

## Bounded next action (main task; not performed here)

After authentication/local weights: inspect the real header and verify target/alpha coverage; keep the Q4 base immutable. Reuse the existing baseline, same prompt/seed/geometry/audio settings, for **one 4-NFE adapter run**. Larry 8 NFE is a conditional follow-up for motion quality, not a new baseline or a sweep. No Q4 quality claim is possible yet. FastVideo's saved card was read; VSA adapters require its backend and exact-delta/gate tensors, so they are outside this minimal-change choice. Main's BF16 dense-file observation does not establish plain-LoRA completeness. [S12]

## Sources and provenance

All paths below are absolute; line references above refer to saved source snapshots. `fetch-receipts.jsonl` contains exact request URLs, timestamps, TLS verification status and SHA-256; `local-source-manifest.json` hashes the input snapshots. GitHub code commit identity does **not** verify a current Hugging Face weights revision.

- **[S1]** [larryvrh-api-README.md](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/larryvrh-api-README.md) — Extracted Data.ReadMeContent from supplied official ModelScope larryvrh-api.json; steps/strength, Weights, low_vram.
- **[S2]** [generate.py](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/larry/generate.py) — Official ModelScope repo?Revision=master&FilePath=generate.py; exact fetched content hash in receipts.
- **[S3]** [__init__.py](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/larry-node/__init__.py) — Publisher Larryvrh/ComfyUI-MiniMax-H3-Turbo @ 4274783a23afcfdbea3b4876cb79effd6c510785; GitHub contents API.
- **[S4]** [Minimax-H3-Turbo-README.md](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/snapshots/Minimax-H3-Turbo-README.md) — ModelTC/Minimax-H3-Turbo @ 02e26d591f7a04d5d1a074c9566d5dd4f22f6225, model table and Note on shift. Head rechecked; no v1.2 table row.
- **[S5]** [DIFFUSERS_SETUP_AND_INFERENCE.md](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/snapshots/DIFFUSERS_SETUP_AND_INFERENCE.md) — Same ModelTC commit; v0.1 4-NFE and v1.0 768p commands.
- **[S6]** [inference_minimax_h3.py](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/snapshots/inference_minimax_h3.py) — Same ModelTC commit; constants 76–86, rank/target validation 680–746, scale 781–783, defaults 819–829, NFE 1139–1170.
- **[S7]** [scheduler.py](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/snapshots/scheduler.py) — Existing ModelTC/LightX2V source @ 8e86555809efd20138d3a4e1f937ee306cdf0ee0; _make_schedule, _step, step_post. Not claimed newly current.
- **[S8]** [minimax_h3_pipeline.cpp](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/snapshots/minimax_h3_pipeline.cpp) — Local worktree snapshot at audit; no edits to production source.
- **[S9]** [light-files-verified.json](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/light-files-verified.json) — Official lightx2v/Minimax-h3-Turbo ModelScope master file listing; v1.2 BF16 and ComfyUI variants present. Not a weights/header/HF-revision verification.
- **[S10]** [minimax_h3_full.hpp](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/snapshots/minimax_h3_full.hpp) — Local model output sign/slope snapshot; no edits.
- **[S11]** [schedule-and-scalar-check.json](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/schedule-and-scalar-check.json) — Audit-derived exact-rational schedule/equivalence assertions; CPU scalar math only.
- **[S12]** [FastVideo-api-README.md](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/FastVideo-api-README.md) — Extracted supplied official ModelScope Data.ReadMeContent; variant/backend limitations, strength default 1.0.

Research receipts: [/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/fetch-receipts.jsonl](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/fetch-receipts.jsonl); local snapshot hashes: [/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/local-source-manifest.json](/home/public/edge-dit.cpp/_scratch/h3-cuda-continuation/_scratch/h3-lora-implementation-20261014/recipes/local-source-manifest.json).
