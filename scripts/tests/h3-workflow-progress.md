# H3 workflow handoff — 2026-10-11

## Delivered scope
Only this task's allowed files were written:
- scripts/h3_cuda_workflow.py
- scripts/tests/test_h3_cuda_workflow.py
- benchmark/h3_cuda/example.json
- benchmark/h3_cuda/validation.json
- scripts/tests/h3-workflow-progress.md (explicitly authorized)

## Safety / precision
- No actual GPU query/generation, downloads, conversion, process termination, or background jobs.
- Existing full Q4 FL2VA/Ref2VA, Q4 Qwen, F16 video VAE, F32 audio VAE from docs/research/h3-precision-lock.json; no pruned model.
- Header/tensor/size provenance gated; explicit full-file SHA256 lock mandatory before real runs or imports. Real full-weight hashing not performed in this task.
- Baseline source fixed 97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2, not old August binaries.
- Baseline clears ED_/GGML_/SD_ legacy overrides and GPU visibility, then sets seven ED_H3 flags zero and native H3 profile on. Candidate starts from same defaults.

## Workflow / validation
- CLI fixtures / lock / baseline / candidate / workflow / compare, host/container paths overrideable even after lock.
- All eight modes, mixed Ref2VA first; first/last conditioning uses FL2VA.
- Immutable fixed reports and imported artifacts; candidate bound to baseline report SHA256 and same weights/assets/shape/sampling/placement.
- Timeout/nonzero/partial logs and fail-closed read-only GPU prechecks before future launches; no eviction or group termination. Precheck is not scheduler reservation.
- Warmup/repeats are independent ed-cli processes, not load-once steady-state. Native H3 profile + existing phase/placement/CPU-stage parsers; mean/median exclude warmups.
- Full ffmpeg video/audio decode; frame count/dim/fps/stereo 32kHz/finite validation. Exact per-frame SSIM/PSNR and audio RMSE/relative L2. No resizing/truncation/resampling for shape mismatch.
- Artifact imports and dry-runs never contain performance values. Metrics never claim semantic quality or runtime activation.

## Method catalog integration contract
- example.json method_catalog keyed by ED_H3 env names: status=implemented/pending/unsupported, modes, required_build, optional requires_separate_manifest.
- Fusion implemented; CacheDiT/AdaLN/Sage/Sparse pending and individually dry-run plannable. VDN/Turbo unsupported by default.
- Real execution requires catalog status=implemented OR h3-build-provenance.txt implemented_methods=ED_H3_CACHE_DIT,ED_H3_ADALN_CACHE,... (JSON list also accepted), plus build requirements and policy parser in binary.
- Sage catalog requires ED_ENABLE_CUDA_SAGE_ATTN=ON and build provenance cuda_sage_attention=ON for that candidate; untouched baseline still OFF.
- Turbo never bypasses this full-Q4 precision lock; a separate experiment manifest/baseline needs separate review. No step/weight substitution in main A/B.
- Main thread can expand catalog statuses/required_build after actual model wiring review. No Python hard-coded fusion-only supported list remains.

## Final tests
- 47/47 passed; 8 CPU ffmpeg/ffprobe integration tests, 39 stdlib/mock tests.
- Covers argument/legal-frame validation, asset/checkpoint/pairing mismatch, precision source, full hashes, environment isolation, occupied/low-memory GPUs (mock), nonzero/timeout/empty output, warmup stats, build gating, pending-method dry-runs, artifact binding, full compare, audio shape mismatch, no overwrites, portable paths.
- CLI help, CPU fixtures, host/container and catalog workflow dry-runs exit 0.
- Earlier test-only failures corrected: 128x512 is legal; one in-memory test accidentally shared a mutable manifest/contract.
- Performance results remain null. Real H3 GPU execution, semantic assessment and full production-weight SHA256 remain pending outside this task.

## Retained logs
- Final suite: /tmp/h3-workflow-final-dhhkcf4o/tests-catalog-final.log
- Dry-run: /tmp/h3-workflow-final-dhhkcf4o/catalog-dryrun.log
- Fixture generation: /tmp/h3-workflow-final-dhhkcf4o/fixtures.log
- All earlier CPU artifacts/logs retained under /tmp/h3-workflow-unit-* and /tmp/h3-workflow-media-*; nothing deleted.

## Reproduce
python scripts/h3_cuda_workflow.py --help
python -m unittest discover -s scripts/tests -p test_h3_cuda_workflow.py -v

Task handoff complete. No continuing task or GPU job was left running.
