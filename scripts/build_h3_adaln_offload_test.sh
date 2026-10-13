#!/usr/bin/env bash
# Compile only. Never run a binary, query GPUs, alter production/CMake, or download.
# Usage: bash scripts/build_h3_adaln_offload_test.sh [matching-candidate-build] [fresh-output]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/_scratch/rtx4090-20261013/build-candidate/candidate}"
OUTPUT="${2:-$BUILD/bin/h3-cuda-adaln-offload-test}"
export CUDA_VISIBLE_DEVICES=''
python3 - "$ROOT" "$BUILD" "$OUTPUT" <<'PY'
import hashlib, json, pathlib, re, shlex, subprocess, sys
root, build, output = map(lambda p: pathlib.Path(p).resolve(), sys.argv[1:])
sys.dont_write_bytecode = True
sys.path.insert(0, str(root/'scripts'))
# Reuse the exact production manifest algorithm (src/**, include/**, CMakeLists).
# Import only; the build-config module's CLI/build entry point is not invoked.
from h3_cuda_build_config import source_manifest

def verify_candidate_sources(source_root, candidate):
    try:
        raw = (candidate/'h3-source-manifest.json').read_bytes()
        locked = json.loads(raw)
        entries = locked['files']
        if not isinstance(entries, list) or not entries:
            raise ValueError('empty/malformed production file list')
        expected = {}
        for item in entries:
            if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
                raise ValueError('malformed production file entry')
            name, digest = item['path'], item['sha256']
            if not isinstance(name, str) or not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
                raise ValueError('invalid production path/digest')
            path = pathlib.PurePosixPath(name)
            if (path.is_absolute() or '..' in path.parts or str(path) != name or name in expected or
                not (name == 'CMakeLists.txt' or name.startswith(('src/', 'include/')))):
                raise ValueError('duplicate/unsafe/non-production path: '+name)
            expected[name] = digest
        tree = hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if locked.get('source_tree_sha256') != tree:
            raise ValueError('manifest source_tree_sha256 does not match its entries')
        provenance = dict(line.split('=', 1) for line in (candidate/'h3-build-provenance.txt').read_text().splitlines() if '=' in line)
        if provenance.get('source_tree_sha256') != tree:
            raise ValueError('manifest differs from candidate build provenance')
        current_entries, current_tree = source_manifest(source_root)
        current = {item['path']: item['sha256'] for item in current_entries}
        missing = sorted(expected.keys() - current.keys())
        added = sorted(current.keys() - expected.keys())
        changed = sorted(name for name in expected.keys() & current.keys() if expected[name] != current[name])
        if missing or added or changed:
            raise ValueError('production source drift: missing='+repr(missing)+' added='+repr(added)+' changed='+repr(changed))
        if current_tree != tree:
            raise ValueError('production manifest ordering/algorithm mismatch')
        return {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                'source_tree_sha256': tree, 'verified_production_files': len(entries)}
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit('Candidate source validation FAILED (no validated helper): '+str(error))

if output.exists(): raise SystemExit('Refusing to overwrite an existing binary; supply a fresh output path')
candidate_sources = verify_candidate_sources(root, build)
print('VERIFIED production files='+str(candidate_sources['verified_production_files'])+
      ' source_tree_sha256='+candidate_sources['source_tree_sha256'], flush=True)
cache = (build/'CMakeCache.txt').read_text()
for required in ('ED_GGML_CUDA:BOOL=ON', 'GGML_BACKEND_DL:BOOL=OFF'):
    if required not in cache: raise SystemExit('Matching shared CUDA candidate required: '+required)
commands = json.loads((build/'compile_commands.json').read_text())
entry = next(e for e in commands if e['file'].endswith('/minimax_h3_pipeline.cpp'))
# Match candidate compile-time class layouts and custom CUDA extension macros,
# not just the CPU regression's include paths (ASYNC_OFFLOAD affects Runner ABI).
args = entry.get('arguments') or shlex.split(entry['command'])
flags=[]
i=1
while i < len(args):
    arg=args[i]
    if arg in ('-o', '-c'):
        i += 2; continue
    if arg.startswith('-O') or arg == '-DNDEBUG':
        i += 1; continue
    flags.append(arg); i += 1
# Relocate candidate source paths only when they refer to this same checkout.
if pathlib.Path(entry['file']).resolve() != root/'src/dit_models/pipelines/minimax_h3_pipeline.cpp':
    raise SystemExit('Candidate is from a different source root; do not mix headers and libraries')
libdir=build/'bin'
libs=[build/'lib/libedgedit.a']+[libdir/('lib'+name+'.so') for name in ('ggml','ggml-cuda','ggml-cpu','ggml-base')]
for p in libs:
    if not p.is_file(): raise SystemExit('Missing candidate library '+str(p))
source=root/'tests/h3_cuda_adaln_offload_test.cpp'
source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
output.parent.mkdir(parents=True, exist_ok=True)
cmd=[args[0],*flags,'-O1','-UNDEBUG',str(source),'-o',str(output),
     '-Wl,-rpath,'+str(libdir),'-Wl,-rpath-link,'+str(libdir),*[str(p) for p in libs],'-pthread']
print('COMPILE ONLY: '+shlex.join(cmd),flush=True)
subprocess.run(cmd,check=True,cwd=entry['directory'])
# Recheck after compilation too. A concurrent edit must not produce a success
# record with libraries from one snapshot and headers from another.
if verify_candidate_sources(root, build) != candidate_sources:
    raise SystemExit('Candidate source manifest changed during compilation; output retained but NOT validated')
if hashlib.sha256(source.read_bytes()).hexdigest() != source_sha256:
    raise SystemExit('Helper source changed during compilation; output retained but NOT validated')
manifest={'source_sha256':source_sha256,
          'candidate_sources':candidate_sources,
          'binary_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
          'candidate_build':str(build),'command':cmd,'executed':False}
output.with_suffix(output.suffix+'.build.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('BUILT (NOT RUN): '+str(output))
PY
