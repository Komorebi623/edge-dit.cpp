from __future__ import annotations

import io
import runpy
import tempfile
import textwrap
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "server_smoke.py"
SCRIPT_GLOBALS = runpy.run_path(str(SCRIPT_PATH), run_name="__test__")

build_parser = SCRIPT_GLOBALS["build_parser"]
load_config_file = SCRIPT_GLOBALS["load_config_file"]
main = SCRIPT_GLOBALS["main"]
resolve_output_path = SCRIPT_GLOBALS["resolve_output_path"]

EXPECTED_HELP = textwrap.dedent(
    """\
    usage: server_smoke.py [-h] --config CONFIG [--kind {image,video}]
                           [--output OUTPUT] [--timeout-seconds TIMEOUT_SECONDS]
                           [--job-ttl-seconds JOB_TTL_SECONDS]

    Run a real smoke test through server

    options:
      -h, --help            show this help message and exit
      --config CONFIG       Path to an image or video smoke JSON config
      --kind {image,video}
      --output OUTPUT       Optional output path override
      --timeout-seconds TIMEOUT_SECONDS
      --job-ttl-seconds JOB_TTL_SECONDS
    """
)


class ServerSmokeExampleTests(unittest.TestCase):
    def test_help_output_snapshot(self) -> None:
        parser = build_parser()
        parser.prog = "server_smoke.py"
        self.assertEqual(parser.format_help(), EXPECTED_HELP)

    def test_load_config_file_expands_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                (
                    "{"
                    '"engine": {"model_path": "${EDGE_DIT_MODEL_PATH}"},'
                    '"request": {"prompt": "teapot"}'
                    "}"
                ),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"EDGE_DIT_MODEL_PATH": "/models/flux"}):
                payload = load_config_file(config_path)
        self.assertEqual(payload["engine"]["model_path"], "/models/flux")

    def test_resolve_output_path_prefers_override(self) -> None:
        payload = {"output": "/tmp/from-config.png", "kind": "image"}
        self.assertEqual(resolve_output_path(payload, "/tmp/override.png"), "/tmp/override.png")
        self.assertEqual(resolve_output_path({"kind": "video"}), "/tmp/edge_dit_server_smoke.gif")

    def test_main_runs_minimal_image_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(
                (
                    "{"
                    '"engine": {"model_path": "/models/flux"},'
                    '"request": {"prompt": "teapot"},'
                    '"output": "/tmp/from-config.png"'
                    "}"
                ),
                encoding="utf-8",
            )

            captured: dict[str, object] = {}

            class FakeEngine:
                def __init__(self, config) -> None:
                    captured["engine_config"] = config

            class FakeService:
                def __init__(self, engine, *, model_name=None, job_ttl_seconds=None) -> None:
                    captured["engine"] = engine
                    captured["model_name"] = model_name
                    captured["job_ttl_seconds"] = job_ttl_seconds

                def close(self) -> None:
                    captured["service_closed"] = True

            class FakeServer:
                server_address = ("127.0.0.1", 4321)

                def shutdown(self) -> None:
                    captured["server_shutdown"] = True

                def server_close(self) -> None:
                    captured["server_closed"] = True

                def serve_forever(self) -> None:
                    captured["serve_forever_called"] = True

            saved: dict[str, object] = {}

            def fake_request_json(base_url, method, path, payload=None):
                captured.setdefault("requests", []).append((base_url, method, path, payload))
                if method == "POST":
                    return 202, {"status_url": "/ed/v2/jobs/job-1", "result_url": "/ed/v2/jobs/job-1/result"}
                return 200, {"object": "edge_dit.image_generation", "data": [{"b64_png": "x"}]}

            def fake_wait_for_terminal_job(base_url, status_url, timeout_seconds):
                captured["wait_args"] = (base_url, status_url, timeout_seconds)
                return {"status": "succeeded", "result_url": "/ed/v2/jobs/job-1/result"}

            def fake_save_image_result(result, output):
                saved["result"] = result
                saved["output"] = output

            args = build_parser().parse_args(
                [
                    "--config",
                    str(config_path),
                    "--kind",
                    "image",
                    "--output",
                    "/tmp/override.png",
                    "--job-ttl-seconds",
                    "12",
                ]
            )

            stdout = io.StringIO()
            globals_patch = {
                "Engine": FakeEngine,
                "ImageJobService": FakeService,
                "create_http_server": lambda address, service: FakeServer(),
                "request_json": fake_request_json,
                "wait_for_terminal_job": fake_wait_for_terminal_job,
                "save_image_result": fake_save_image_result,
                "build_parser": lambda: build_parser(),
            }
            with patch.dict(main.__globals__, globals_patch):
                with patch.object(build_parser().__class__, "parse_args", return_value=args):
                    with redirect_stdout(stdout):
                        result = main()

        self.assertEqual(result, 0)
        self.assertEqual(saved["output"], "/tmp/override.png")
        self.assertEqual(captured["job_ttl_seconds"], 12.0)
        self.assertEqual(captured["wait_args"], ("http://127.0.0.1:4321", "/ed/v2/jobs/job-1", 600.0))
        self.assertTrue(captured["service_closed"])
        self.assertTrue(captured["server_shutdown"])
        self.assertTrue(captured["server_closed"])
        self.assertIn("saved server image smoke output to /tmp/override.png", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
