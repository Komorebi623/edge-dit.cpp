from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from edge_dit.config import EngineConfig
from edge_dit.server_v2 import serve


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _load_profile(path: str | os.PathLike[str]) -> dict[str, Any]:
    profile_path = Path(path)
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("profile JSON must be a top-level object")
    return _expand(payload)


def _resolve_engine_payload(payload: dict[str, Any]) -> dict[str, Any]:
    engine_payload = payload.get("engine")
    if not isinstance(engine_payload, dict):
        raise ValueError("profile JSON must include an `engine` object")

    resolved = dict(engine_payload)
    model_env = payload.get("model_env")
    if isinstance(model_env, str) and model_env:
        model_override = os.environ.get(model_env)
        if model_override:
            resolved["model_path"] = model_override
    return resolved


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch a managed local server_v2 profile")
    parser.add_argument("--profile", required=True, help="Path to a runtime profile JSON file")
    parser.add_argument("--host", default=os.environ.get("EDGE_DIT_MANAGED_BACKEND_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EDGE_DIT_MANAGED_BACKEND_PORT", "8080")),
    )
    parser.add_argument(
        "--job-ttl-seconds",
        type=float,
        default=float(os.environ.get("EDGE_DIT_MANAGED_JOB_TTL_SECONDS", "3600")),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = _load_profile(args.profile)
    engine_payload = _resolve_engine_payload(profile)
    engine_config = EngineConfig(**engine_payload)
    job_ttl_seconds = None if args.job_ttl_seconds < 0 else args.job_ttl_seconds
    serve(engine_config, host=args.host, port=args.port, job_ttl_seconds=job_ttl_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
