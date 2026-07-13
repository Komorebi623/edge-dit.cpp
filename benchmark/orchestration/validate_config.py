#!/usr/bin/env python3
"""Validate benchmark contract files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

try:
    from .config import has_private_path, load_json, load_suite_graph, load_yaml, repo_root
except ImportError:  # pragma: no cover - direct script execution
    from config import has_private_path, load_json, load_suite_graph, load_yaml, repo_root


def validate(root: Path) -> list[str]:
    errors: list[str] = []
    local_output_dirs = {
        (root / "benchmark" / "results").resolve(),
        (root / "benchmark" / "cache").resolve(),
        (root / "benchmark" / "downloads").resolve(),
        (root / "benchmark" / "tmp").resolve(),
    }

    for path in sorted((root / "benchmark" / "schemas").glob("*.json")):
        try:
            load_json(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: invalid JSON: {exc}")

    for path in sorted((root / "benchmark" / "configs").rglob("*.yaml")):
        try:
            load_yaml(path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: invalid YAML: {exc}")

    for path in sorted((root / "benchmark").rglob("*")):
        if not path.is_file():
            continue
        resolved = path.resolve()
        if any(resolved == local_dir or local_dir in resolved.parents for local_dir in local_output_dirs):
            continue
        if "benchmark/configs/local" in path.as_posix():
            continue
        if path.suffix not in {".yaml", ".md", ".json", ".jsonl"}:
            continue
        patterns = has_private_path(path)
        if patterns:
            errors.append(f"{path}: private path pattern in public benchmark file: {patterns}")

    for suite in sorted((root / "benchmark" / "configs" / "suites").glob("*.yaml")):
        site = root / "benchmark" / "configs" / "local" / "site-h200.yaml"
        try:
            load_suite_graph(suite, site)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{suite}: cannot resolve suite graph: {exc}")

    gitignore = (root / ".gitignore").read_text(encoding="utf-8", errors="replace")
    for ignore in [
        "/benchmark/results/",
        "/benchmark/cache/",
        "/benchmark/downloads/",
        "/benchmark/tmp/",
    ]:
        if ignore not in gitignore:
            errors.append(f".gitignore missing {ignore}")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=repo_root())
    args = parser.parse_args()

    errors = validate(args.root.resolve())
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("benchmark config validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
