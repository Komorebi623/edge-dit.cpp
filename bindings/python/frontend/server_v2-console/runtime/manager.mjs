import { spawn } from 'node:child_process'
import { createServer } from 'node:http'
import { readFileSync, readdirSync } from 'node:fs'
import path from 'node:path'
import readline from 'node:readline'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const APP_ROOT = path.resolve(__dirname, '..')
const PROFILES_DIR = path.join(__dirname, 'profiles')
const RUN_PROFILE_SCRIPT = path.join(APP_ROOT, 'scripts', 'run-managed-profile.sh')

const DEFAULT_MANAGER_HOST = process.env.EDGE_DIT_RUNTIME_MANAGER_HOST ?? '127.0.0.1'
const DEFAULT_MANAGER_PORT = Number.parseInt(process.env.EDGE_DIT_RUNTIME_MANAGER_PORT ?? '8090', 10)
const DEFAULT_BACKEND_HOST = process.env.EDGE_DIT_MANAGED_BACKEND_HOST ?? '127.0.0.1'
const DEFAULT_BACKEND_PORT = Number.parseInt(process.env.EDGE_DIT_MANAGED_BACKEND_PORT ?? '8080', 10)
const DEFAULT_JOB_TTL_SECONDS = Number.parseFloat(process.env.EDGE_DIT_MANAGED_JOB_TTL_SECONDS ?? '3600')
const HEALTH_INTERVAL_MS = Number.parseInt(process.env.EDGE_DIT_RUNTIME_HEALTH_INTERVAL_MS ?? '2500', 10)
const HEALTH_TIMEOUT_MS = Number.parseInt(process.env.EDGE_DIT_RUNTIME_HEALTH_TIMEOUT_MS ?? '1500', 10)
const STOP_TIMEOUT_MS = Number.parseInt(process.env.EDGE_DIT_RUNTIME_STOP_TIMEOUT_MS ?? '5000', 10)
const AUTO_RESTART_LIMIT = Number.parseInt(process.env.EDGE_DIT_RUNTIME_AUTO_RESTART_LIMIT ?? '3', 10)
const AUTO_RESTART_BASE_DELAY_MS = Number.parseInt(
  process.env.EDGE_DIT_RUNTIME_AUTO_RESTART_BASE_DELAY_MS ?? '2000',
  10,
)
const AUTO_RESTART_MAX_DELAY_MS = Number.parseInt(
  process.env.EDGE_DIT_RUNTIME_AUTO_RESTART_MAX_DELAY_MS ?? '10000',
  10,
)
const LOG_TAIL_LIMIT = Number.parseInt(process.env.EDGE_DIT_RUNTIME_LOG_TAIL_LIMIT ?? '120', 10)
const EVENT_LIMIT = Number.parseInt(process.env.EDGE_DIT_RUNTIME_EVENT_LIMIT ?? '80', 10)

const { values } = parseArgs({
  allowPositionals: true,
  options: {
    'auto-start-profile': {
      type: 'string',
    },
    'backend-host': {
      type: 'string',
    },
    'backend-port': {
      type: 'string',
    },
    host: {
      type: 'string',
    },
    help: {
      type: 'boolean',
    },
    'job-ttl-seconds': {
      type: 'string',
    },
    port: {
      type: 'string',
    },
  },
  strict: false,
})

if (values.help) {
  console.log(`edge-dit runtime manager

Options:
  --auto-start-profile <slug>   Start a verified profile after the manager boots
  --host <host>                 Runtime manager bind host (default: ${DEFAULT_MANAGER_HOST})
  --port <port>                 Runtime manager bind port (default: ${DEFAULT_MANAGER_PORT})
  --backend-host <host>         Managed server_v2 bind host (default: ${DEFAULT_BACKEND_HOST})
  --backend-port <port>         Managed server_v2 bind port (default: ${DEFAULT_BACKEND_PORT})
  --job-ttl-seconds <seconds>   Override managed job TTL (default: ${DEFAULT_JOB_TTL_SECONDS})
`)
  process.exit(0)
}

const managerHost = values.host ?? DEFAULT_MANAGER_HOST
const managerPort = values.port ? Number.parseInt(values.port, 10) : DEFAULT_MANAGER_PORT
const managedBackendHost = values['backend-host'] ?? DEFAULT_BACKEND_HOST
const managedBackendPort = values['backend-port']
  ? Number.parseInt(values['backend-port'], 10)
  : DEFAULT_BACKEND_PORT
const jobTtlSeconds = values['job-ttl-seconds']
  ? Number.parseFloat(values['job-ttl-seconds'])
  : DEFAULT_JOB_TTL_SECONDS
const autoStartProfileSlug = values['auto-start-profile'] ?? null

function nowMs() {
  return Date.now()
}

function delay(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms)
  })
}

function clampPositiveInteger(value, fallback) {
  return Number.isFinite(value) && value > 0 ? Math.trunc(value) : fallback
}

function createRingBuffer(limit) {
  return {
    items: [],
    limit,
    push(item) {
      this.items.push(item)
      if (this.items.length > this.limit) {
        this.items.splice(0, this.items.length - this.limit)
      }
    },
  }
}

function loadProfiles() {
  return readdirSync(PROFILES_DIR)
    .filter((name) => name.endsWith('.json'))
    .map((name) => {
      const filePath = path.join(PROFILES_DIR, name)
      const payload = JSON.parse(readFileSync(filePath, 'utf8'))
      return {
        description: typeof payload.description === 'string' ? payload.description : '',
        engine: payload.engine ?? {},
        kind: payload.kind === 'video' ? 'video' : 'image',
        model_env: typeof payload.model_env === 'string' ? payload.model_env : null,
        name: typeof payload.name === 'string' ? payload.name : path.basename(name, '.json'),
        notes: Array.isArray(payload.notes) ? payload.notes.filter((item) => typeof item === 'string') : [],
        request_example: payload.request_example ?? null,
        slug: typeof payload.slug === 'string' ? payload.slug : path.basename(name, '.json'),
      }
    })
    .sort((left, right) => left.name.localeCompare(right.name))
}

const profiles = loadProfiles()
const profilesBySlug = new Map(profiles.map((profile) => [profile.slug, profile]))

function serializeProfile(profile) {
  if (!profile) {
    return null
  }
  return {
    description: profile.description,
    kind: profile.kind,
    model_env: profile.model_env,
    name: profile.name,
    notes: profile.notes,
    request_example: profile.request_example,
    slug: profile.slug,
  }
}

const logTail = createRingBuffer(clampPositiveInteger(LOG_TAIL_LIMIT, 120))
const events = createRingBuffer(clampPositiveInteger(EVENT_LIMIT, 80))

const state = {
  capabilities: null,
  child: null,
  childExitPromise: null,
  childExitResolver: null,
  currentProfile: null,
  everHealthy: false,
  expectedExit: null,
  healthFailureStreak: 0,
  lastExit: null,
  lastHealth: null,
  lastReadyAtMs: null,
  nextRestartAtMs: null,
  pendingRestartTimer: null,
  restartCountConsecutive: 0,
  restartCountTotal: 0,
  startedAtMs: null,
  status: 'idle',
  wasHealthy: false,
}

let controlQueue = Promise.resolve()
let lastCapabilitiesFetchAtMs = 0
let managerShuttingDown = false

function pushLog(stream, line) {
  logTail.push({
    line,
    stream,
    time_ms: nowMs(),
  })
}

function pushEvent(level, type, message, detail = null) {
  events.push({
    detail,
    id: `${type}_${nowMs()}_${Math.random().toString(16).slice(2, 8)}`,
    level,
    message,
    time_ms: nowMs(),
    type,
  })
}

function snapshotStatus() {
  return {
    backend: {
      auto_restart_enabled: true,
      auto_restart_limit: AUTO_RESTART_LIMIT,
      base_url: `http://${managedBackendHost}:${managedBackendPort}`,
      capabilities: state.capabilities,
      health_failure_streak: state.healthFailureStreak,
      last_exit: state.lastExit,
      last_health: state.lastHealth,
      last_ready_at_ms: state.lastReadyAtMs,
      next_restart_at_ms: state.nextRestartAtMs,
      pid: state.child?.pid ?? null,
      profile: serializeProfile(state.currentProfile),
      restart_count_consecutive: state.restartCountConsecutive,
      restart_count_total: state.restartCountTotal,
      started_at_ms: state.startedAtMs,
      status: state.status,
      uptime_ms: state.startedAtMs ? Math.max(0, nowMs() - state.startedAtMs) : null,
    },
    manager: {
      app_root: APP_ROOT,
      host: managerHost,
      port: managerPort,
      started_at_ms: managerStartedAtMs,
      status: 'online',
      version: 'local-dev-runtime-manager',
    },
    profiles: profiles.map((profile) => serializeProfile(profile)),
    recent_events: [...events.items],
    recommended_connection_target: {
      baseUrl: `http://${managedBackendHost}:${managedBackendPort}`,
      prefix: '/ed/v2',
    },
    log_tail: [...logTail.items],
    request_id: `runtime_${nowMs()}`,
  }
}

function enqueueControl(task) {
  const next = controlQueue.then(task, task)
  controlQueue = next.catch(() => {})
  return next
}

function clearPendingRestart() {
  if (state.pendingRestartTimer) {
    clearTimeout(state.pendingRestartTimer)
    state.pendingRestartTimer = null
  }
  state.nextRestartAtMs = null
}

function createExitWaiter() {
  let resolve
  const promise = new Promise((innerResolve) => {
    resolve = innerResolve
  })
  return { promise, resolve }
}

function attachStream(stream, streamName) {
  const reader = readline.createInterface({ input: stream })
  reader.on('line', (line) => {
    pushLog(streamName, line)
  })
}

function spawnManagedBackend(profile, launchReason) {
  clearPendingRestart()
  state.capabilities = null
  state.everHealthy = false
  state.healthFailureStreak = 0
  state.lastHealth = null
  state.lastReadyAtMs = null
  state.wasHealthy = false
  state.startedAtMs = nowMs()
  state.status = launchReason === 'auto_restart' ? 'restarting' : 'starting'
  state.currentProfile = profile

  const exitWaiter = createExitWaiter()
  state.childExitPromise = exitWaiter.promise
  state.childExitResolver = exitWaiter.resolve

  const child = spawn(
    'bash',
    [
      RUN_PROFILE_SCRIPT,
      profile.slug,
      '--host',
      managedBackendHost,
      '--port',
      String(managedBackendPort),
      '--job-ttl-seconds',
      String(jobTtlSeconds),
    ],
    {
      cwd: APP_ROOT,
      env: process.env,
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  )

  state.child = child
  pushEvent('info', 'backend_spawn', `Starting managed backend for ${profile.name}.`, {
    launch_reason: launchReason,
    pid: child.pid,
    profile_slug: profile.slug,
  })

  attachStream(child.stdout, 'stdout')
  attachStream(child.stderr, 'stderr')

  child.on('error', (error) => {
    pushEvent('error', 'backend_spawn_error', `Failed to start ${profile.name}.`, {
      message: error instanceof Error ? error.message : String(error),
      profile_slug: profile.slug,
    })
  })

  child.on('exit', (code, signal) => {
    const expectedExit = state.expectedExit
    state.expectedExit = null
    state.child = null
    state.childExitResolver?.({
      code,
      expected: Boolean(expectedExit),
      signal,
    })
    state.childExitPromise = null
    state.childExitResolver = null

    state.lastExit = {
      code,
      intentional: Boolean(expectedExit),
      observed_at_ms: nowMs(),
      reason: expectedExit?.reason ?? 'unexpected_exit',
      signal,
    }

    if (expectedExit) {
      pushEvent('info', 'backend_exit', `${profile.name} stopped.`, {
        code,
        reason: expectedExit.reason,
        signal,
      })
      state.everHealthy = false
      state.healthFailureStreak = 0
      state.wasHealthy = false
      state.status = 'idle'
      if (expectedExit.clearProfile) {
        state.currentProfile = null
        state.capabilities = null
      }
      return
    }

    pushEvent('error', 'backend_crash', `${profile.name} exited unexpectedly.`, {
      code,
      profile_slug: profile.slug,
      signal,
    })
    state.wasHealthy = false

    if (!state.currentProfile) {
      state.status = 'crashed'
      return
    }

    if (state.restartCountConsecutive >= AUTO_RESTART_LIMIT || managerShuttingDown) {
      state.status = 'crashed'
      if (!managerShuttingDown) {
        pushEvent(
          'error',
          'auto_restart_exhausted',
          `Auto-restart budget exhausted for ${profile.name}.`,
          {
            auto_restart_limit: AUTO_RESTART_LIMIT,
            profile_slug: profile.slug,
          },
        )
      }
      return
    }

    const restartDelayMs = Math.min(
      AUTO_RESTART_BASE_DELAY_MS * (state.restartCountConsecutive + 1),
      AUTO_RESTART_MAX_DELAY_MS,
    )
    state.status = 'restarting'
    state.nextRestartAtMs = nowMs() + restartDelayMs
    pushEvent('warning', 'auto_restart_scheduled', `Scheduling auto-restart for ${profile.name}.`, {
      delay_ms: restartDelayMs,
      profile_slug: profile.slug,
    })

    state.pendingRestartTimer = setTimeout(() => {
      state.pendingRestartTimer = null
      state.nextRestartAtMs = null
      void enqueueControl(async () => {
        state.restartCountTotal += 1
        state.restartCountConsecutive += 1
        spawnManagedBackend(profile, 'auto_restart')
      })
    }, restartDelayMs)
  })
}

async function stopManagedBackend(reason, { clearProfile } = { clearProfile: true }) {
  clearPendingRestart()
  if (!state.child) {
    state.status = 'idle'
    state.wasHealthy = false
    if (clearProfile) {
      state.currentProfile = null
      state.capabilities = null
    }
    return
  }

  const child = state.child
  const profile = state.currentProfile
  state.expectedExit = {
    clearProfile,
    reason,
  }
  state.status = 'stopping'
  pushEvent('info', 'backend_stop_requested', `Stopping managed backend${profile ? ` for ${profile.name}` : ''}.`, {
    clear_profile: clearProfile,
    pid: child.pid,
    reason,
  })

  child.kill('SIGTERM')

  const exited = await Promise.race([
    state.childExitPromise,
    delay(STOP_TIMEOUT_MS).then(() => null),
  ])

  if (!exited && state.child) {
    pushEvent('warning', 'backend_stop_timeout', 'Managed backend did not exit after SIGTERM, sending SIGKILL.', {
      pid: state.child.pid,
      reason,
      timeout_ms: STOP_TIMEOUT_MS,
    })
    state.child.kill('SIGKILL')
    await state.childExitPromise
  }
}

async function startProfile(profileSlug, reason = 'manual_start') {
  const profile = profilesBySlug.get(profileSlug)
  if (!profile) {
    throw new Error(`Unknown managed profile: ${profileSlug}`)
  }

  if (state.child && state.currentProfile?.slug === profile.slug && state.status === 'running') {
    pushEvent('info', 'backend_start_noop', `${profile.name} is already running.`, {
      profile_slug: profile.slug,
    })
    return
  }

  if (reason !== 'auto_restart') {
    state.restartCountConsecutive = 0
  }

  if (state.child) {
    await stopManagedBackend(reason === 'manual_restart' ? 'manual_restart' : 'profile_switch', {
      clearProfile: false,
    })
  }

  spawnManagedBackend(profile, reason)
}

async function restartProfile(reason = 'manual_restart') {
  return restartProfileWithOptions(reason, { resetConsecutive: true })
}

async function restartProfileWithOptions(reason, { resetConsecutive }) {
  if (!state.currentProfile) {
    throw new Error('No managed profile selected yet.')
  }
  if (resetConsecutive) {
    state.restartCountConsecutive = 0
  }
  if (state.child) {
    await stopManagedBackend(reason, { clearProfile: false })
  }
  spawnManagedBackend(state.currentProfile, reason)
}

async function probeBackend(pathname) {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), HEALTH_TIMEOUT_MS)
  try {
    const response = await fetch(`http://${managedBackendHost}:${managedBackendPort}${pathname}`, {
      headers: {
        Accept: 'application/json',
        'X-Request-ID': `runtime-probe-${nowMs()}`,
      },
      signal: controller.signal,
    })
    const payload = await response.json().catch(() => null)
    return {
      ok: response.ok,
      payload,
      status: response.status,
    }
  } catch (error) {
    return {
      error: error instanceof Error ? error.message : String(error),
      ok: false,
      payload: null,
      status: null,
    }
  } finally {
    clearTimeout(timeout)
  }
}

async function runHealthLoop() {
  if (!state.child || state.status === 'stopping') {
    return
  }

  const startedAt = nowMs()
  const health = await probeBackend('/ed/v2/health')
  state.lastHealth = {
    checked_at_ms: nowMs(),
    error: health.ok ? null : health.error ?? (health.status ? `HTTP ${health.status}` : 'request failed'),
    ok: health.ok,
    request_id: health.payload?.request_id ?? null,
    response_ms: Math.max(0, nowMs() - startedAt),
    status: health.status,
  }

  if (!health.ok) {
    if (!state.everHealthy) {
      state.status = 'starting'
      return
    }

    state.healthFailureStreak += 1
    state.wasHealthy = false
    state.status = 'unhealthy'
    if (state.healthFailureStreak === 1) {
      pushEvent('warning', 'backend_unhealthy', 'Managed backend health probe failed.', {
        error: state.lastHealth.error,
      })
    }
    if (state.healthFailureStreak >= 3 && state.child) {
      pushEvent('error', 'backend_health_restart', 'Restarting managed backend after repeated health probe failures.', {
        health_failure_streak: state.healthFailureStreak,
      })
      void enqueueControl(async () => {
        if (!state.currentProfile) {
          return
        }
        state.restartCountTotal += 1
        state.restartCountConsecutive += 1
        await restartProfileWithOptions('health_probe_restart', { resetConsecutive: false })
      })
    }
    return
  }

  state.healthFailureStreak = 0
  state.everHealthy = true
  state.status = 'running'
  state.lastReadyAtMs = nowMs()
  if (!state.wasHealthy) {
    pushEvent(
      'info',
      state.restartCountTotal > 0 ? 'backend_recovered' : 'backend_ready',
      state.restartCountTotal > 0
        ? `Managed backend recovered and is serving again.`
        : 'Managed backend is healthy and ready.',
      state.currentProfile ? { profile_slug: state.currentProfile.slug } : null,
    )
  }
  state.wasHealthy = true

  const shouldRefreshCapabilities =
    !state.capabilities || nowMs() - lastCapabilitiesFetchAtMs > Math.max(HEALTH_INTERVAL_MS * 4, 10_000)
  if (!shouldRefreshCapabilities) {
    return
  }

  const capabilities = await probeBackend('/ed/v2/capabilities')
  if (capabilities.ok) {
    state.capabilities = capabilities.payload
    lastCapabilitiesFetchAtMs = nowMs()
  }
}

function writeJson(response, statusCode, payload) {
  response.statusCode = statusCode
  response.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Request-ID')
  response.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
  response.setHeader('Access-Control-Allow-Origin', '*')
  response.setHeader('Content-Type', 'application/json; charset=utf-8')
  response.end(JSON.stringify(payload, null, 2))
}

async function readJsonBody(request) {
  const chunks = []
  for await (const chunk of request) {
    chunks.push(chunk)
  }
  if (chunks.length === 0) {
    return {}
  }
  const raw = Buffer.concat(chunks).toString('utf8')
  return raw.trim() ? JSON.parse(raw) : {}
}

function notFound(response) {
  writeJson(response, 404, {
    error: {
      code: 'not_found',
      message: 'unknown runtime manager endpoint',
      status: 404,
    },
    request_id: `runtime_${nowMs()}`,
  })
}

const managerStartedAtMs = nowMs()

async function shutdownManager(signal) {
  if (managerShuttingDown) {
    return
  }
  managerShuttingDown = true
  pushEvent('warning', 'manager_shutdown', `Runtime manager received ${signal}.`, null)
  try {
    await enqueueControl(() => stopManagedBackend('manager_shutdown', { clearProfile: false }))
  } finally {
    server.close(() => {
      process.exit(0)
    })
    setTimeout(() => {
      process.exit(0)
    }, 1_000).unref()
  }
}

setInterval(() => {
  void runHealthLoop()
}, Math.max(1000, HEALTH_INTERVAL_MS))

const server = createServer(async (request, response) => {
  if (!request.url || !request.method) {
    notFound(response)
    return
  }

  if (request.method === 'OPTIONS') {
    response.statusCode = 204
    response.setHeader('Access-Control-Allow-Headers', 'Content-Type, X-Request-ID')
    response.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    response.setHeader('Access-Control-Allow-Origin', '*')
    response.end()
    return
  }

  const url = new URL(request.url, `http://${request.headers.host ?? '127.0.0.1'}`)

  if (request.method === 'GET' && url.pathname === '/runtime/v1/health') {
    writeJson(response, 200, {
      request_id: `runtime_${nowMs()}`,
      service: 'edge-dit-runtime-manager',
      status: 'ok',
    })
    return
  }

  if (request.method === 'GET' && url.pathname === '/runtime/v1/profiles') {
    writeJson(response, 200, {
      data: profiles.map((profile) => serializeProfile(profile)),
      object: 'list',
      request_id: `runtime_${nowMs()}`,
    })
    return
  }

  if (request.method === 'GET' && url.pathname === '/runtime/v1/status') {
    writeJson(response, 200, snapshotStatus())
    return
  }

  if (request.method === 'POST' && url.pathname === '/runtime/v1/backend/start') {
    try {
      const body = await readJsonBody(request)
      const profileSlug = typeof body.profile_slug === 'string' ? body.profile_slug : ''
      if (!profileSlug) {
        writeJson(response, 400, {
          error: {
            code: 'invalid_request',
            message: '`profile_slug` is required',
            status: 400,
          },
          request_id: `runtime_${nowMs()}`,
        })
        return
      }
      await enqueueControl(() => startProfile(profileSlug, 'manual_start'))
      writeJson(response, 202, snapshotStatus())
    } catch (error) {
      writeJson(response, 500, {
        error: {
          code: 'runtime_error',
          message: error instanceof Error ? error.message : String(error),
          status: 500,
        },
        request_id: `runtime_${nowMs()}`,
      })
    }
    return
  }

  if (request.method === 'POST' && url.pathname === '/runtime/v1/backend/restart') {
    try {
      await enqueueControl(() => restartProfile('manual_restart'))
      writeJson(response, 202, snapshotStatus())
    } catch (error) {
      writeJson(response, 409, {
        error: {
          code: 'runtime_error',
          message: error instanceof Error ? error.message : String(error),
          status: 409,
        },
        request_id: `runtime_${nowMs()}`,
      })
    }
    return
  }

  if (request.method === 'POST' && url.pathname === '/runtime/v1/backend/stop') {
    await enqueueControl(() => stopManagedBackend('manual_stop', { clearProfile: false }))
    writeJson(response, 202, snapshotStatus())
    return
  }

  notFound(response)
})

server.listen(managerPort, managerHost, () => {
  pushEvent('info', 'manager_started', `Runtime manager listening on http://${managerHost}:${managerPort}.`, {
    managed_backend_base_url: `http://${managedBackendHost}:${managedBackendPort}`,
  })
})

if (autoStartProfileSlug) {
  void enqueueControl(() => startProfile(autoStartProfileSlug, 'manager_autostart'))
}

process.on('SIGINT', () => {
  void shutdownManager('SIGINT')
})

process.on('SIGTERM', () => {
  void shutdownManager('SIGTERM')
})
