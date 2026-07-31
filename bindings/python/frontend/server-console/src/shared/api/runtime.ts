import { ApiNetworkError, parseApiError } from './errors'

import type { ManagedRuntimeStatusResponse } from '@/shared/model/runtime'

const RUNTIME_API_PREFIX = '/runtime/v1'

function requestId() {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID()
  }

  return `runtime_${Date.now()}_${Math.random().toString(16).slice(2)}`
}

async function requestRuntimeJson<T>(suffix: string, init?: RequestInit): Promise<T> {
  const headers = new Headers(init?.headers)
  headers.set('Accept', 'application/json')
  headers.set('X-Request-ID', requestId())

  if (init?.body && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  let response: Response
  try {
    response = await fetch(`${RUNTIME_API_PREFIX}${suffix}`, {
      ...init,
      headers,
    })
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Runtime manager request failed'
    throw new ApiNetworkError(message)
  }

  const responseRequestId = response.headers.get('X-Request-ID')
  const payload = (await response.json().catch(() => null)) as unknown

  if (!response.ok) {
    throw parseApiError(response.status, payload, responseRequestId)
  }

  return payload as T
}

export function getManagedRuntimeStatus() {
  return requestRuntimeJson<ManagedRuntimeStatusResponse>('/status')
}

export function startManagedRuntimeProfile(profileSlug: string) {
  return requestRuntimeJson<ManagedRuntimeStatusResponse>('/backend/start', {
    body: JSON.stringify({ profile_slug: profileSlug }),
    method: 'POST',
  })
}

export function restartManagedRuntime() {
  return requestRuntimeJson<ManagedRuntimeStatusResponse>('/backend/restart', {
    method: 'POST',
  })
}

export function stopManagedRuntime() {
  return requestRuntimeJson<ManagedRuntimeStatusResponse>('/backend/stop', {
    method: 'POST',
  })
}
