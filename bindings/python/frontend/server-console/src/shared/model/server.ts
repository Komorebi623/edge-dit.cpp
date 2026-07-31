export const API_PREFIXES = ['/ed/v2', '/edgedit/v2', '/edge-dit/v2'] as const

export type ApiPrefix = (typeof API_PREFIXES)[number]

export interface ConnectionTarget {
  baseUrl: string
  prefix: ApiPrefix
}

export interface HealthResponse {
  status: string
  service: string
  model: string
  request_id: string
}

export interface CapabilitiesResponse {
  service: string
  package_version: string
  model: string
  pipeline_name: string | null
  version_name: string | null
  supports: {
    image: boolean
    video: boolean
  }
  defaults: {
    sampler: number | string | null
    scheduler: number | string | null
  }
  endpoints: string[]
  aliases: string[]
  semantics: {
    progress: string
    cancellation: string
    results: string
    job_ttl_ms: number | null
  }
  request_id: string
}

export type ConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'connected'
  | 'health_failed'
  | 'capabilities_failed'

export interface ActivityLogEntry {
  id: string
  time: string
  method: string
  path: string
  status: string
  requestId?: string
}
