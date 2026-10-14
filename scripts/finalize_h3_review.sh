#!/usr/bin/env bash
# CPU-only report finalizer. Invoke with a NEW absolute destination directory.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:?new output directory required}"
STATE="${2:-${H3_NIGHT_STATUS:-}}"
[[ "$OUT" == /* && ! -e "$OUT" ]] || { echo 'Refusing relative/existing report directory' >&2; exit 2; }
mkdir -p "$OUT"
source "$ROOT/_scratch/rtx4090-20261013/environment.sh"
ARGS=(--output "$OUT/report-data.json" --label "汇报版：截至生成时的真实状态")
if [[ -n "$STATE" ]]; then ARGS+=(--night-state "$STATE"); fi
python3 "$ROOT/scripts/prepare_h3_review_data.py" "${ARGS[@]}"
# Render environment is isolated from all benchmark tasks.
source "$ROOT/_scratch/h3-report-runtime/env.sh"
node "$ROOT/scripts/build_h3_review_deck.js" --data "$OUT/report-data.json" --output "$OUT/H3-CUDA-优化汇报.pptx"
python "$ROOT/scripts/tests/deck_structure_check.py" "$OUT/H3-CUDA-优化汇报.pptx" --data "$OUT/report-data.json" > "$OUT/structure.log" 2>&1
pptx-validate "$OUT/H3-CUDA-优化汇报.pptx" > "$OUT/validation.log" 2>&1
mkdir "$OUT/preview"
lo-render --headless --nologo --nodefault --nofirststartwizard --convert-to 'pdf:impress_pdf_Export' --outdir "$OUT/preview" "$OUT/H3-CUDA-优化汇报.pptx" > "$OUT/render.log" 2>&1
test -s "$OUT/preview/H3-CUDA-优化汇报.pdf"
pdftoppm -jpeg -scale-to 1600 "$OUT/preview/H3-CUDA-优化汇报.pdf" "$OUT/preview/slide" > "$OUT/pdftoppm.log" 2>&1
pdftotext -layout "$OUT/preview/H3-CUDA-优化汇报.pdf" "$OUT/preview/slides.txt"
python "$ROOT/scripts/check_h3_review.py" --dir "$OUT"
echo "REPORT_READY=$OUT/H3-CUDA-优化汇报.pptx"
