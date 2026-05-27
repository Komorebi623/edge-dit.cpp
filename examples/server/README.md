# light-dit Server

`ld-server` is a small HTTP wrapper around the public light-dit C API. It keeps one
model context loaded in-process and serializes generation calls through that
context.

## Start

```bash
./build-cuda/bin/ld-server \
  --backend cuda \
  --model /mnt/cfs/9n-das-admin/llm_models/flux-dev/ \
  --host 0.0.0.0 \
  --port 8080 \
  -W 1024 -H 1024 \
  --steps 50 \
  --guidance 3.5
```

Canonical endpoints use the `ld` prefix, matching `ld-cli` and the `ld_*` C API:

- `GET /ld/v1/health`
- `GET /ld/v1/models`
- `GET /ld/v1/capabilities`
- `POST /ld/v1/images/generations`

Aliases are also registered for `/lightdit/v1/...` and `/light-dit/v1/...`.

## Generate An Image

```bash
curl -s http://127.0.0.1:8080/ld/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "a cinematic photo of a glass teapot on a wooden table, soft morning light",
    "width": 1024,
    "height": 1024,
    "steps": 50,
    "seed": 0,
    "distilled_guidance": 3.5,
    "cfg_scale": 1.0,
    "cache": {
      "mode": "dbcache",
      "Fn_compute_blocks": 8,
      "Bn_compute_blocks": 0,
      "residual_diff_threshold": 0.08,
      "max_warmup_steps": 8
    }
  }' | jq -r '.data[0].b64_png' | base64 -d > output.png
```

The response includes `elapsed_ms`, resolved generation parameters, and one
base64-encoded PNG per generated image.
