import { API_PREFIXES, type ApiPrefix, type ConnectionTarget } from './server'

const STORAGE_KEY = 'edge-dit.server-v2-console.connection-target'

export const DEFAULT_CONNECTION_TARGET: ConnectionTarget = {
  baseUrl: 'http://127.0.0.1:8080',
  prefix: '/ed/v2',
}

export function isApiPrefix(value: string): value is ApiPrefix {
  return API_PREFIXES.includes(value as ApiPrefix)
}

export function normalizeBaseUrl(value: string) {
  return value.trim().replace(/\/+$/, '')
}

export function normalizeConnectionTarget(target: ConnectionTarget): ConnectionTarget {
  return {
    baseUrl: normalizeBaseUrl(target.baseUrl) || DEFAULT_CONNECTION_TARGET.baseUrl,
    prefix: target.prefix,
  }
}

export function loadConnectionTarget(): ConnectionTarget {
  if (typeof window === 'undefined') {
    return DEFAULT_CONNECTION_TARGET
  }

  const raw = window.localStorage.getItem(STORAGE_KEY)
  if (!raw) {
    return DEFAULT_CONNECTION_TARGET
  }

  try {
    const parsed = JSON.parse(raw) as Partial<ConnectionTarget>
    const prefix = typeof parsed.prefix === 'string' && isApiPrefix(parsed.prefix) ? parsed.prefix : DEFAULT_CONNECTION_TARGET.prefix
    const baseUrl = typeof parsed.baseUrl === 'string' ? parsed.baseUrl : DEFAULT_CONNECTION_TARGET.baseUrl
    return normalizeConnectionTarget({ baseUrl, prefix })
  } catch {
    return DEFAULT_CONNECTION_TARGET
  }
}

export function saveConnectionTarget(target: ConnectionTarget) {
  if (typeof window === 'undefined') {
    return
  }

  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalizeConnectionTarget(target)))
}
