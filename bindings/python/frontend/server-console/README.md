# Python Server Console

The Python Server Console is the local browser interface for `edge_dit.server`. It starts one selected model, sends job requests, shows progress, and displays image or video results.

For the complete installation, shared-library build, virtual environment, and model-path setup, read [the Python guide](../../README.md) first.

## Start the local stack

From this directory:

`@bash
npm install
EDGE_DIT_FLUX_MODEL_PATH=/absolute/path/to/FLUX.1-dev \
npm run dev:managed
`@

Open `http://127.0.0.1:5173`. The first model load can take several minutes. The console is usable once the runtime panel reports that the backend is running.

The command starts three local services:

| Service | Address | What it does |
| --- | --- | --- |
| Console | `http://127.0.0.1:5173` | Browser interface and development proxy |
| Runtime manager | `http://127.0.0.1:8090/runtime/v1` | Starts/stops profiles and retains a log tail |
| Python Server | `http://127.0.0.1:8080/ed/v2` | Runs generation jobs |

Use another built-in model profile by setting its model variable and forwarding the profile name:

`@bash
EDGE_DIT_WAN_VIDEO_MODEL_PATH=/absolute/path/to/Wan2.1-T2V-1.3B-Diffusers \
npm run dev:managed -- --auto-start-profile wan-t2v
`@

The available profiles and their variables are documented in [RUNTIME_CONFIGURATION.md](RUNTIME_CONFIGURATION.md).

## Useful commands

`@bash
npm run dev
npm run runtime:manager -- --auto-start-profile flux-dev
npm run dev:managed:network
npm run build
npm test
npm run test:e2e
`@

`npm run dev` starts only the UI; a separately running Python Server is required. `npm run runtime:manager` starts only the model manager. `npm run dev:managed:network` binds all services to `0.0.0.0` for a trusted local network.

Stop the managed stack with `Ctrl-C`. The runtime manager also stops the Python Server it created.

## Connection targets

The console uses `/ed/v2` by default. This is the Python Server HTTP protocol version. The native C++ server has a different `/ed/v1` contract, so do not point this console at a native server unless it implements the Python Server job endpoints.

The Console's Connection panel can probe a server running on another machine. For a remote target, use its reachable base URL and the `/ed/v2` prefix. The browser must be allowed to reach that address, and the server must be configured for the desired network binding.

## Troubleshooting

- **The UI opens but the backend is starting**: model weights are still loading. Check `http://127.0.0.1:8090/runtime/v1/status`.
- **The manager exits immediately**: ensure `EDGE_DIT_PYTHON_BIN` points to the virtual environment Python and `EDGE_DIT_LIBRARY` points to `libedgedit.so`.
- **A profile cannot find its model**: set the profile's exact `EDGE_DIT_*_MODEL_PATH` variable to the directory containing `model_index.json`.
- **Port already in use**: stop the previous managed stack or override `EDGE_DIT_RUNTIME_MANAGER_PORT` and `EDGE_DIT_MANAGED_BACKEND_PORT`. The frontend port is passed through Vite with `npm run dev -- --port <port>`.
