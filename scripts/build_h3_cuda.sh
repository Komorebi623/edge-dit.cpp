#!/usr/bin/env bash
# Build an immutable baseline and/or candidate without changing existing builds.
# Run in the existing CUDA development environment (host or container).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODE="${1:-all}"
BASELINE_REF="${H3_BASELINE_REF:-97cc585d9dbd9ac5e30b78ef2e2c4a0d8bfbe9f2}"
JOBS="${H3_BUILD_JOBS:-4}"
case "$MODE" in baseline|candidate|all) ;; *) echo 'usage: build_h3_cuda.sh [baseline|candidate|all]' >&2; exit 2 ;; esac
cd "$ROOT"
for dependency in ggml onednn; do
    EXPECTED="$(git ls-tree "$BASELINE_REF" "third_party/$dependency" | awk '{print $3}')"
    ACTUAL="$(git -C "third_party/$dependency" rev-parse HEAD)"
    if [[ "$EXPECTED" != "$ACTUAL" ]] || [[ -n "$(git -C "third_party/$dependency" status --porcelain)" ]]; then
        echo "Refusing a contaminated baseline: third_party/$dependency must be clean at $EXPECTED" >&2
        exit 2
    fi
done
FRONTEND="${H3_CUDNN_FRONTEND_SOURCE:-$ROOT/build-h200-current-20260814/_deps/cudnn_frontend-src}"
if [[ ! -f "$FRONTEND/CMakeLists.txt" ]]; then
    echo 'Set H3_CUDNN_FRONTEND_SOURCE to the existing pinned cudnn-frontend checkout (07b368a); no automatic download.' >&2
    exit 2
fi
COMMON=(-G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
    -DED_GGML_CUDA=ON -DED_ENABLE_CUDNN_SDPA=ON -DED_FETCH_CUDNN_FRONTEND=ON
    "-DFETCHCONTENT_SOURCE_DIR_CUDNN_FRONTEND=$FRONTEND"
    -DGGML_CUDA_FA_ALL_QUANTS=OFF -DED_ENABLE_CUDA_SAGE_ATTN=OFF)
build() {
    local source="$1" destination="$2" revision="$3"
    local sage=OFF
    if [[ "$source" == "$ROOT" ]]; then sage=ON; fi
    cmake -S "$source" -B "$destination" "${COMMON[@]}" "-DED_ENABLE_CUDA_SAGE_ATTN=$sage"
    cmake --build "$destination" --target ed-cli -j "$JOBS"
    # git archive is not a checkout: annotate the fixed source revision and real
    # dependency provenance separately instead of trusting Git's parent search.
    python3 - "$destination/build-config.txt" "$revision" "$(git -C "$ROOT/third_party/ggml" rev-parse HEAD)" <<'PY_BUILD'
from pathlib import Path
import sys
path = Path(sys.argv[1])
lines = path.read_text().splitlines()
values = {'git_commit': sys.argv[2], 'ggml_commit': sys.argv[3]}
path.write_text('\n'.join(key+'='+values[key] if key in values else line for line in lines for key in [line.split('=',1)[0]])+'\n')
PY_BUILD
    {
        echo "source_revision=$revision"
        echo "ggml_revision=$(git -C "$ROOT/third_party/ggml" rev-parse HEAD)"
        echo "onednn_revision=$(git -C "$ROOT/third_party/onednn" rev-parse HEAD)"
        echo "cuda_architectures=90"
        echo "cudnn_frontend_revision=$(git -C "$FRONTEND" rev-parse HEAD)"
        echo "cuda_sage_attention=$sage"
        if [[ "$source" == "$ROOT" ]]; then
            echo 'implemented_methods=ED_H3_FUSED_OPS,ED_H3_CACHE_DIT,ED_H3_ADALN_CACHE,ED_H3_SAGE_ATTN,ED_H3_SPARSE_ATTN'
        fi
        git diff --binary -- src include CMakeLists.txt | sha256sum | awk '{print "tracked_diff_sha256=" $1}'
    } > "$destination/h3-build-provenance.txt"
    python3 - "$source" "$destination" <<'PY_ID'
from pathlib import Path
import hashlib, json, sys
root, destination = map(Path, sys.argv[1:])
entries = []
for folder in ('src', 'include'):
    for path in sorted((root / folder).rglob('*')):
        if path.is_file():
            entries.append({'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
path = root / 'CMakeLists.txt'
entries.append({'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
encoded = json.dumps(entries, sort_keys=True, separators=(',', ':')).encode()
identity = hashlib.sha256(encoded).hexdigest()
(destination / 'h3-source-manifest.json').write_text(json.dumps({'source_tree_sha256': identity, 'files': entries}, indent=2)+'\n')
with (destination / 'h3-build-provenance.txt').open('a') as f:
    f.write('source_tree_sha256='+identity+'\n')
    f.write('binary_sha256='+hashlib.sha256((destination/'bin/ed-cli').read_bytes()).hexdigest()+'\n')
PY_ID
}
if [[ "$MODE" == baseline || "$MODE" == all ]]; then
    SNAPSHOT="$ROOT/_scratch/h3-baseline-97cc585"
    if [[ ! -d "$SNAPSHOT" ]]; then
        mkdir -p "$SNAPSHOT"
        git archive "$BASELINE_REF" | tar -x -C "$SNAPSHOT"
        python3 - "$ROOT" "$SNAPSHOT" <<'PY'
from pathlib import Path
import sys
root, snapshot = map(Path, sys.argv[1:])
for name in ('ggml', 'onednn'):
    for entry in (root / 'third_party' / name).iterdir():
        if entry.name not in ('.git', 'build'):
            (snapshot / 'third_party' / name / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
PY
    fi
    # Never quietly reuse a modified baseline snapshot. Tracked source content
    # is checked directly against the archived original before compilation.
    python3 - "$ROOT" "$SNAPSHOT" "$BASELINE_REF" <<'PY'
from pathlib import Path
import hashlib, subprocess, sys, tarfile, io
root, snapshot, revision = sys.argv[1:]
archive = subprocess.check_output(['git', '-C', root, 'archive', revision])
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
    for entry in stream:
        if entry.isfile():
            disk = Path(snapshot) / entry.name
            if not disk.is_file() or disk.read_bytes() != stream.extractfile(entry).read():
                raise SystemExit('Refusing modified baseline source: ' + entry.name)
PY
    build "$SNAPSHOT" "$ROOT/build-h3-baseline-97cc585" "$BASELINE_REF"
fi
if [[ "$MODE" == candidate || "$MODE" == all ]]; then
    build "$ROOT" "$ROOT/build-h3-candidate" "$(git rev-parse HEAD)"
fi
