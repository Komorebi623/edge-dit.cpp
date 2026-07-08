export type ManagedBackendStatus =
  | 'crashed'
  | 'idle'
  | 'restarting'
  | 'running'
  | 'starting'
  | 'stopping'
  | 'unhealthy'

export interface ManagedRuntimeProfile {
  description: string
  kind: 'image' | 'video'
  model_env: string | null
  name: string
  notes: string[]
  request_example: Record<string, unknown> | null
  slug: string
}

export interface ManagedRuntimeStatusResponse {
  request_id: string
  backend: {
    auto_restart_enabled: boolean
    auto_restart_limit: number
    base_url: string
    capabilities: Record<string, unknown> | null
    health_failure_streak: number
    last_exit: {
      code: number | null
      intentional: boolean
      observed_at_ms: number
      reason: string
      signal: string | null
    } | null
    last_health: {
      checked_at_ms: number
      error: string | null
      ok: boolean
      request_id: string | null
      response_ms: number
      status: number | null
    } | null
    last_ready_at_ms: number | null
    next_restart_at_ms: number | null
    pid: number | null
    profile: ManagedRuntimeProfile | null
    restart_count_consecutive: number
    restart_count_total: number
    started_at_ms: number | null
    status: ManagedBackendStatus
    uptime_ms: number | null
  }
  manager: {
    app_root: string
    host: string
    port: number
    started_at_ms: number
    status: string
    version: string
  }
  profiles: ManagedRuntimeProfile[]
  recent_events: Array<{
    detail: Record<string, unknown> | null
    id: string
    level: 'error' | 'info' | 'warning'
    message: string
    time_ms: number
    type: string
  }>
  recommended_connection_target: {
    baseUrl: string
    prefix: '/ed/v2'
  }
  log_tail: Array<{
    line: string
    stream: 'stderr' | 'stdout'
    time_ms: number
  }>
}
