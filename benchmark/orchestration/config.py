"""Config loading and suite expansion helpers."""

from __future__ import annotations

from pathlib import Path
import itertools
import json
from typing import Any

import yaml


PRIVATE_PATH_PATTERNS = ("/export/home/", "/mnt/cfs/", "/home/")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected mapping in {path}")
    return data


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected object in {path}")
    return data


def resolve_config_ref(base_file: Path, ref: str) -> Path:
    return (base_file.parent / ref).resolve()


def load_prompt(prompt_file: Path, prompt_id: str) -> dict[str, Any]:
    with prompt_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("prompt_id") == prompt_id:
                return item
    raise ValueError(f"prompt_id {prompt_id!r} not found in {prompt_file}")


def load_prompt_set(prompt_file: Path) -> dict[str, dict[str, Any]]:
    prompts: dict[str, dict[str, Any]] = {}
    with prompt_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            prompt_id = item.get("prompt_id")
            if isinstance(prompt_id, str) and prompt_id:
                prompts[prompt_id] = item
    return prompts


def has_private_path(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return [pattern for pattern in PRIVATE_PATH_PATTERNS if pattern in text]


def load_suite_graph(suite_path: Path, site_path: Path | None) -> dict[str, Any]:
    suite = load_yaml(suite_path)
    site = load_yaml(site_path) if site_path else {"paths": {}}

    systems = {}
    for group in suite.get("systems", {}).values():
        for ref in group:
            system_path = resolve_config_ref(suite_path, ref)
            system = load_yaml(system_path)
            systems[system["system_id"]] = {
                "path": system_path,
                "config": system,
            }

    workloads = {}
    for group in suite.get("workloads", {}).values():
        for ref in group:
            workload_path = resolve_config_ref(suite_path, ref)
            workload = load_yaml(workload_path)
            prompt_path = resolve_config_ref(workload_path, workload["prompt_set"])
            workload["resolved_prompt"] = load_prompt(prompt_path, workload["prompt_id"])
            workload["resolved_prompt_set"] = load_prompt_set(prompt_path)
            workloads[workload["workload_id"]] = {
                "path": workload_path,
                "config": workload,
            }

    return {
        "suite_path": suite_path,
        "site_path": site_path,
        "suite": suite,
        "site": site,
        "systems": systems,
        "workloads": workloads,
    }


def expand_runs(graph: dict[str, Any]) -> list[dict[str, Any]]:
    suite = graph["suite"]
    runs: list[dict[str, Any]] = []
    scenarios = suite_run_options(suite)

    for ref in suite.get("workloads", {}).get("single_gpu", suite.get("workloads", {}).get("release_gate", [])):
        workload = load_yaml(resolve_config_ref(graph["suite_path"], ref))
        for system_ref in suite.get("systems", {}).get("single_gpu", []):
            system = load_yaml(resolve_config_ref(graph["suite_path"], system_ref))
            for scenario in scenarios:
                runs.append(
                    {
                        "system_id": system["system_id"],
                        "workload_id": workload["workload_id"],
                        "gpu_count": 1,
                        "parallel_mode": None,
                        "scenario_id": scenario["scenario_id"],
                        "run_options": scenario["options"],
                    }
                )

    parallel_workload_refs = suite.get("workloads", {}).get("parallel", [])
    if not parallel_workload_refs:
        parallel_workload_refs = suite.get("workloads", {}).get("release_gate", [])
    gpu_counts = suite.get("parallel_matrix", {}).get("gpu_counts", [])
    modes = suite.get("parallel_matrix", {}).get("modes", {})
    for ref in parallel_workload_refs:
        workload = load_yaml(resolve_config_ref(graph["suite_path"], ref))
        for system_ref in suite.get("systems", {}).get("parallel", []):
            system = load_yaml(resolve_config_ref(graph["suite_path"], system_ref))
            for gpu_count in gpu_counts:
                for mode in modes.get(system["system_id"], [None]):
                    runs.append(
                        {
                            "system_id": system["system_id"],
                            "workload_id": workload["workload_id"],
                            "gpu_count": gpu_count,
                            "parallel_mode": mode,
                            "scenario_id": "default",
                            "run_options": {},
                        }
                    )

    # Preserve order but drop exact duplicates.
    deduped = []
    seen = set()
    for run in runs:
        key = (
            run["system_id"],
            run["workload_id"],
            run["gpu_count"],
            run["parallel_mode"],
            run["scenario_id"],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(run)
    return deduped


def suite_run_options(suite: dict[str, Any]) -> list[dict[str, Any]]:
    if suite.get("scenarios"):
        return [
            {
                "scenario_id": scenario["scenario_id"],
                "options": scenario.get("options", {}),
            }
            for scenario in suite["scenarios"]
        ]
    if suite.get("ablation_stacks"):
        return [
            {
                "scenario_id": stack["stack_id"],
                "options": stack.get("options", {}),
            }
            for stack in suite["ablation_stacks"]
        ]
    if suite.get("scenario_matrix"):
        axes = suite["scenario_matrix"].get("axes", [])
        rows: list[dict[str, Any]] = []
        for combination in itertools.product(*(axis.get("values", []) for axis in axes)):
            ids: list[str] = []
            options: dict[str, Any] = {}
            for value in combination:
                value_id = str(value["id"])
                ids.append(value_id)
                options.update(value.get("options", {}))
            rows.append(
                {
                    "scenario_id": "_".join(ids),
                    "options": options,
                }
            )
        return rows
    return [{"scenario_id": "default", "options": {}}]
