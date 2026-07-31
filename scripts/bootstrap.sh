#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "error: this directory is not a Git checkout." >&2
  echo "hint: GitHub auto-generated Source ZIP archives usually do not include submodule contents." >&2
  echo "recommended:" >&2
  echo "  git clone --recursive <repository-url>" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

git submodule update --init --recursive

# Verify every submodule declared in .gitmodules actually populated. A partial
# checkout (e.g. a GitHub source ZIP, or one submodule failing to fetch) otherwise
# passes silently and only blows up later at CMake/compile time.
missing=()
while IFS= read -r sub_path; do
  # A populated submodule dir is non-empty; treat empty/absent as missing.
  if [[ -z "$(ls -A "${sub_path}" 2>/dev/null)" ]]; then
    missing+=("${sub_path}")
  fi
done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}')

if (( ${#missing[@]} > 0 )); then
  echo "error: missing submodule content after update: ${missing[*]}" >&2
  echo "hint: GitHub auto-generated Source ZIP archives usually do not include submodule contents." >&2
  echo "recommended:" >&2
  echo "  git clone --recursive <repository-url>" >&2
  echo "  git submodule update --init --recursive" >&2
  exit 1
fi

echo "bootstrap complete: submodules present ($(git config -f .gitmodules --get-regexp '^submodule\..*\.path$' | awk '{print $2}' | tr '\n' ' '))."
