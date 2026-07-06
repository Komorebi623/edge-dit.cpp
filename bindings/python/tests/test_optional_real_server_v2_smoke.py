from __future__ import annotations

import base64
import json
import os
import threading
import time
import unittest
import urllib.error
import urllib.request
from io import BytesIO
from pathlib import Path

from PIL import Image

from edge_dit import Engine
from edge_dit.config import EngineConfig
from edge_dit.server_v2 import ImageJobService, create_http_server


def _request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    data = None
    headers = {"X-Request-ID": "optional-real-server-v2-smoke"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(base_url + path, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8"))


def _wait_for_terminal_job(base_url: str, status_url: str, timeout_seconds: float) -> dict[str, object]:
    deadline = time.time() + timeout_seconds
    last_body: dict[str, object] | None = None
    while time.time() < deadline:
        status_code, body = _request_json(base_url, "GET", status_url)
        if status_code != 200:
            raise AssertionError(f"unexpected HTTP {status_code} while polling job: {body}")
        last_body = body
        if body.get("status") in {"succeeded", "failed", "cancelled"}:
            return body
        time.sleep(1.0)
    raise AssertionError(f"timed out waiting for terminal job state; last body was {last_body}")


@unittest.skipUnless(
    os.environ.get("EDGE_DIT_RUN_INTEGRATION") == "1",
    "set EDGE_DIT_RUN_INTEGRATION=1 to run real native smoke tests",
)
class OptionalRealServerV2SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.library_path = os.environ.get("EDGE_DIT_LIBRARY")
        cls.model_path = os.environ.get("EDGE_DIT_MODEL_PATH")
        if not cls.library_path:
            raise unittest.SkipTest("EDGE_DIT_LIBRARY is required for server_v2 integration smoke tests")
        if not cls.model_path:
            raise unittest.SkipTest("EDGE_DIT_MODEL_PATH is required for server_v2 integration smoke tests")

    def _start_server(self, *, model_path: str, keep_vae_on_cpu: bool = False) -> None:
        config = EngineConfig(
            model_path=model_path,
            backend=os.environ.get("EDGE_DIT_BACKEND", "cuda"),
            offload_params_to_cpu=True,
            keep_text_encoder_on_cpu=True,
            keep_vae_on_cpu=keep_vae_on_cpu or None,
            max_vram_gb=float(os.environ.get("EDGE_DIT_MAX_VRAM_GB", "8.0")),
        )
        engine = Engine(config, _library_path=self.library_path)
        self.service = ImageJobService(engine, model_name=model_path, job_ttl_seconds=3600.0)
        self.httpd = create_http_server(("127.0.0.1", 0), self.service)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self) -> None:
        httpd = getattr(self, "httpd", None)
        if httpd is not None:
            httpd.shutdown()
            httpd.server_close()
        service = getattr(self, "service", None)
        if service is not None:
            service.close()
        thread = getattr(self, "thread", None)
        if thread is not None:
            thread.join(timeout=5.0)

    def test_generate_image_through_real_server_v2(self) -> None:
        self._start_server(model_path=self.model_path)

        output_path = Path(
            os.environ.get("EDGE_DIT_SERVER_V2_INTEGRATION_OUTPUT", "/tmp/edge_dit_server_v2_integration.png")
        )

        status_code, job = _request_json(
            self.base_url,
            "POST",
            "/ed/v2/images/generations",
            {
                "prompt": os.environ.get("EDGE_DIT_PROMPT", "server v2 integration smoke teapot"),
                "width": int(os.environ.get("EDGE_DIT_WIDTH", "256")),
                "height": int(os.environ.get("EDGE_DIT_HEIGHT", "256")),
                "steps": int(os.environ.get("EDGE_DIT_STEPS", "1")),
                "seed": int(os.environ.get("EDGE_DIT_SEED", "42")),
            },
        )
        self.assertEqual(status_code, 202, job)
        self.assertEqual(job["kind"], "image")

        terminal = _wait_for_terminal_job(
            self.base_url,
            str(job["status_url"]),
            timeout_seconds=float(os.environ.get("EDGE_DIT_SERVER_V2_TIMEOUT_SECONDS", "600")),
        )
        self.assertEqual(terminal["status"], "succeeded", terminal)

        result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
        self.assertEqual(result_status, 200, result)
        self.assertEqual(result["object"], "edge_dit.image_generation")

        payload = result["data"][0]["b64_png"]
        image = Image.open(BytesIO(base64.b64decode(payload)))
        self.assertEqual(
            image.size,
            (
                int(os.environ.get("EDGE_DIT_WIDTH", "256")),
                int(os.environ.get("EDGE_DIT_HEIGHT", "256")),
            ),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(output_path)
        self.assertTrue(output_path.exists())

    def test_generate_video_through_real_server_v2_when_enabled(self) -> None:
        if os.environ.get("EDGE_DIT_RUN_SERVER_V2_VIDEO") != "1":
            self.skipTest("set EDGE_DIT_RUN_SERVER_V2_VIDEO=1 to run the real server_v2 video smoke test")

        video_model_path = os.environ.get("EDGE_DIT_VIDEO_MODEL_PATH")
        if not video_model_path:
            self.skipTest("EDGE_DIT_VIDEO_MODEL_PATH is required for the real server_v2 video smoke test")

        self._start_server(model_path=video_model_path, keep_vae_on_cpu=True)

        output_path = Path(
            os.environ.get(
                "EDGE_DIT_SERVER_V2_VIDEO_OUTPUT",
                "/tmp/edge_dit_server_v2_integration.gif",
            )
        )

        status_code, job = _request_json(
            self.base_url,
            "POST",
            "/ed/v2/videos/generations",
            {
                "prompt": os.environ.get(
                    "EDGE_DIT_VIDEO_PROMPT",
                    "a small robot walking through a rainy neon street",
                ),
                "width": int(os.environ.get("EDGE_DIT_VIDEO_WIDTH", "416")),
                "height": int(os.environ.get("EDGE_DIT_VIDEO_HEIGHT", "240")),
                "frames": int(os.environ.get("EDGE_DIT_VIDEO_FRAMES", "9")),
                "steps": int(os.environ.get("EDGE_DIT_VIDEO_STEPS", "1")),
                "cfg_scale": float(os.environ.get("EDGE_DIT_VIDEO_CFG_SCALE", "5.0")),
                "flow_shift": float(os.environ.get("EDGE_DIT_VIDEO_FLOW_SHIFT", "5.0")),
                "seed": int(os.environ.get("EDGE_DIT_VIDEO_SEED", "42")),
            },
        )
        self.assertEqual(status_code, 202, job)
        self.assertEqual(job["kind"], "video")

        terminal = _wait_for_terminal_job(
            self.base_url,
            str(job["status_url"]),
            timeout_seconds=float(os.environ.get("EDGE_DIT_SERVER_V2_VIDEO_TIMEOUT_SECONDS", "900")),
        )
        self.assertEqual(terminal["status"], "succeeded", terminal)

        result_status, result = _request_json(self.base_url, "GET", str(terminal["result_url"]))
        self.assertEqual(result_status, 200, result)
        self.assertEqual(result["object"], "edge_dit.video_generation")
        self.assertTrue(result["frames"])

        frames: list[Image.Image] = []
        for item in result["frames"]:
            frames.append(Image.open(BytesIO(base64.b64decode(item["b64_png"]))).copy())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
