interface ApiErrorPayload {
  error?: {
    message?: string
    type?: string
    code?: string
    status?: number
    request_id?: string
  }
  request_id?: string
}

export class ApiClientError extends Error {
  readonly code?: string
  readonly requestId?: string
  readonly status: number
  readonly type?: string

  constructor(message: string, options: { code?: string; requestId?: string; status: number; type?: string }) {
    super(message)
    this.name = 'ApiClientError'
    this.code = options.code
    this.requestId = options.requestId
    this.status = options.status
    this.type = options.type
  }
}

export class ApiNetworkError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiNetworkError'
  }
}

export function parseApiError(status: number, payload: unknown, responseRequestId: string | null) {
  const body = payload as ApiErrorPayload
  const error = body?.error
  return new ApiClientError(error?.message ?? `HTTP ${status}`, {
    code: error?.code,
    requestId: error?.request_id ?? body?.request_id ?? responseRequestId ?? undefined,
    status: error?.status ?? status,
    type: error?.type,
  })
}

export function describeApiFailure(error: unknown) {
  if (error instanceof ApiClientError) {
    return `${error.status}${error.code ? ` ${error.code}` : ''}: ${error.message}`
  }
  if (error instanceof ApiNetworkError) {
    return error.message
  }
  if (error instanceof Error) {
    return error.message
  }
  return 'Unknown error'
}
