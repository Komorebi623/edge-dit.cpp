#!/usr/bin/env bash
# Build ONLY; never query or run a GPU, load weights, download or edit production.
set -euo pipefail
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    printf '%s\n' 'Build ONLY: build_h3_adaln_denoiser_probe.sh [matching-candidate-build] [fresh-output]' \
        'Defaults: four-methods-build-04/candidate; bin/h3-cuda-adaln-denoiser-probe.' \
        'Derives production headers from the candidate compile database, NOT the current changing checkout.' \
        'Requires matching source manifest/provenance, pinned ggml, shared CUDA/CPU backends.' \
        'Run later ONLY via test_h3_gpu_parity.sh with H3_ADALN_PROBE_MODEL and an idle GPU UUID.'
    exit 0
fi
(( $# <= 2 )) || { printf '%s\n' 'Too many arguments (see --help)' >&2; exit 2; }
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${1:-$ROOT/_scratch/h3-offload-20261014/four-methods-build-04/candidate}"
OUTPUT="${2:-$BUILD/bin/h3-cuda-adaln-denoiser-probe}"
export CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
python3 - "$ROOT" "$BUILD" "$OUTPUT" <<'PY'
import hashlib, json, os, pathlib, re, shlex, subprocess, sys
root, build, output = [pathlib.Path(p).absolute() for p in sys.argv[1:]]
source = root/'tests/h3_cuda_adaln_denoiser_probe.cpp'
script = root/'scripts/build_h3_adaln_denoiser_probe.sh'
def fail(text): raise SystemExit('NOT BUILT / NOT VALIDATED: '+text)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def fresh(p):
    if p.exists() or p.is_symlink(): fail('refusing existing output '+str(p))
fresh(output); receipt=pathlib.Path(str(output)+'.build.json'); fresh(receipt)
cache={}
for line in (build/'CMakeCache.txt').read_text().splitlines():
    if line and not line.startswith(('#','//')) and '=' in line:
        k,v=line.split('=',1); cache[k.split(':',1)[0]]=v
for k,v in {'ED_GGML_CUDA':'ON','GGML_BACKEND_DL':'OFF','BUILD_SHARED_LIBS':'ON'}.items():
    if cache.get(k)!=v: fail('matching shared CUDA build required: '+k+'='+v)
snapshot=pathlib.Path(cache['CMAKE_HOME_DIRECTORY']).resolve()
entries=json.loads((build/'compile_commands.json').read_text())
hits=[e for e in entries if pathlib.Path(e['file']).resolve()==snapshot/'src/dit_models/pipelines/minimax_h3_pipeline.cpp']
if len(hits)!=1: fail('missing/ambiguous matching pipeline compile command')
entry=hits[0]; args=entry.get('arguments') or shlex.split(entry['command'])
if not {'-DGGML_MAX_NAME=128','-DGGML_USE_CUDA','-DGGML_USE_CPU'}.issubset(args): fail('missing ABI/backend defines')
libdir=build/'bin'
libs=[build/'lib/libedgedit.a']+[libdir/('lib'+n+'.so') for n in ('ggml','ggml-cuda','ggml-cpu','ggml-base')]
for p in libs:
    if not p.is_file(): fail('missing library '+str(p))
def proof():
    raw=(build/'h3-source-manifest.json').read_bytes(); manifest=json.loads(raw)
    locked=manifest['files']; seen=set()
    for e in locked:
        if set(e)!={'path','sha256'}: fail('malformed source entry')
        name=e['path']; path=pathlib.PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or str(path)!=name or name in seen or not (name=='CMakeLists.txt' or name.startswith(('src/','include/'))): fail('unsafe/duplicate source path')
        if not re.fullmatch('[a-f0-9]{64}',e['sha256']): fail('invalid source digest')
        seen.add(name)
    current=[{'path':p.relative_to(snapshot).as_posix(),'sha256':sha(p)}
             for folder in ('src','include') for p in sorted((snapshot/folder).rglob('*')) if p.is_file()]
    current.append({'path':'CMakeLists.txt','sha256':sha(snapshot/'CMakeLists.txt')})
    tree=hashlib.sha256(json.dumps(current,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    prov=dict(line.split('=',1) for line in (build/'h3-build-provenance.txt').read_text().splitlines() if '=' in line)
    if not locked or locked!=current or tree!=manifest['source_tree_sha256'] or tree!=prov.get('source_tree_sha256'): fail('snapshot source drift or wrong provenance')
    ggml=(snapshot/'third_party/ggml').resolve()
    def git(*a): return subprocess.check_output(['git','-C',str(ggml),*a],text=True).strip()
    if pathlib.Path(git('rev-parse','--show-toplevel')).resolve()!=ggml or git('rev-parse','HEAD')!=prov.get('ggml_revision') or git('status','--porcelain','--untracked-files=all'): fail('changed/unpinned ggml headers')
    return {'snapshot':str(snapshot),'source_tree_sha256':tree,'verified_production_files':len(current),
            'manifest_sha256':hashlib.sha256(raw).hexdigest(),'ggml_revision':prov['ggml_revision'],
            'source_sha256':sha(source),'script_sha256':sha(script),
            'reused_weight_probe_sha256':sha(root/'tests/h3_cuda_adaln_weight_probe.cpp'),
            'compile_commands_sha256':sha(build/'compile_commands.json'),'cmake_cache_sha256':sha(build/'CMakeCache.txt'),
            'libraries_sha256':{str(p.resolve()):sha(p) for p in libs}}
before=proof()
# Keep ALL pipeline macros/include paths (ASYNC_OFFLOAD affects C++ ABI).
# Only replace the source/output and lower optimization for this diagnostic TU.
flags=[]; i=1
while i<len(args):
    a=args[i]
    if a in ('-o','-c','-MF','-MT','-MQ'): i+=2; continue
    if a.startswith('-O') or a in ('-DNDEBUG','-MD','-MMD','-MP'): i+=1; continue
    flags.append(a); i+=1
if any(pathlib.Path(a).suffix in ('.cpp','.cc','.cxx') for a in flags): fail('unexpected leftover compiler input')
output.parent.mkdir(parents=True,exist_ok=True)
# Reserve our fresh filename; no successful receipt will be written on failure.
with output.open('xb'): pass
cmd=[args[0],*flags,'-O1','-UNDEBUG',str(source),'-o',str(output),
     '-Wl,-rpath,'+str(libdir),'-Wl,-rpath-link,'+str(libdir),*[str(p) for p in libs],'-pthread']
print('VERIFIED SOURCE: '+json.dumps(before),flush=True)
print('COMPILE ONLY: '+shlex.join(cmd),flush=True)
subprocess.run(cmd,check=True,cwd=entry['directory'])
if proof()!=before: fail('source/library drift DURING compilation; binary retained but NOT validated')
record={'candidate_build':str(build),'proof':before,'command':cmd,'binary_sha256':sha(output),
        'executed':False,'gpu_executed':False,'model_loaded':False}
with receipt.open('x') as f: json.dump(record,f,indent=2); f.write('\n')
print('BUILT (NOT RUN): '+str(output))
print('BUILD RECEIPT: '+str(receipt))
PY
