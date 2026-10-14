#!/usr/bin/env python3
"""Compile-only companion to h3_subblock_perf_probe.cpp; reuse pinned build flags/libraries."""
import json, pathlib, shlex, subprocess, sys
import argparse
parser = argparse.ArgumentParser(description="Compile the opt-in H3 single-layer probe; never discover/run a GPU")
parser.add_argument("--build", type=pathlib.Path, required=True, help="existing pinned CUDA candidate build")
parser.add_argument("--output", type=pathlib.Path, required=True, help="fresh executable path (no overwrite)")
opts = parser.parse_args()
root = pathlib.Path(__file__).resolve().parents[1]
build, output = opts.build.resolve(), opts.output.resolve()
if output.exists(): raise SystemExit('refusing to overwrite existing executable: ' + str(output))
entries = json.loads((build / 'compile_commands.json').read_text())
entry = next(e for e in entries if e['file'].endswith('/src/edge_dit.cpp'))
args = entry.get('arguments') or shlex.split(entry['command'])
required = {'-DGGML_MAX_NAME=128', '-DGGML_USE_CUDA', '-DED_ENABLE_CUDA_H3_SUBBLOCK', '-DED_ENABLE_CUDNN_SDPA'}
if not required.issubset(args): raise SystemExit('candidate lacks required ABI/CUDA/SubBlock defines')
snapshot = pathlib.Path(entry['file']).parents[1]
# Native + ggml headers come ONLY from the candidate's frozen source snapshot.
# The current worktree may contain concurrent kernel changes not in the binary.
# Reusing a changed Fixture/guard would require an explicit compatibility audit.
if (root / 'tests/h3_subblock_test.cpp').read_bytes() != (snapshot / 'tests/h3_subblock_test.cpp').read_bytes():
    raise SystemExit('tiny Fixture/guard differs from the candidate snapshot')
flags = []; i = 1
while i < len(args):
    a = args[i]
    if a in ('-o', '-c', '-MF', '-MT', '-MQ'): i += 2; continue
    if a.startswith('-O') or a in ('-MD', '-MMD', '-MP'): i += 1; continue
    flags.append(a); i += 1
libdir = build / 'bin'
for name in ('ggml-cuda', 'ggml-cpu', 'ggml-base'):
    if not (libdir / ('lib' + name + '.so')).is_file(): raise SystemExit('missing library: ' + name)
edgedit = build / 'lib/libedgedit.a'
if not edgedit.is_file(): raise SystemExit('missing matching libedgedit.a')
output.parent.mkdir(parents=True, exist_ok=True)
cmd = [args[0], *flags, '-O2', '-Wall', '-Wextra',
       str(root / 'tests/h3_subblock_perf_probe.cpp'), '-L' + str(libdir),
       '-Wl,-rpath,' + str(libdir), '-Wl,-rpath-link,' + str(libdir),
       str(edgedit), '-lggml', '-lggml-cuda', '-lggml-cpu', '-lggml-base', '-pthread', '-ldl',
       '-Wl,--export-dynamic-symbol=_Z21ed_cudnn_sdpa_computeP11ggml_tensorPv',
       '-Wl,--export-dynamic-symbol=_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor',
       '-o', str(output)]
print(shlex.join(cmd), flush=True)
subprocess.run(cmd, cwd=entry['directory'], check=True)
print('Compiled only (no CUDA execution): ' + str(output))
