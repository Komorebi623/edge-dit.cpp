export interface ImageDraft {
  prompt: string
  negativePrompt: string
  width: string
  height: string
  steps: string
  seed: string
  guidance: string
  batchCount: string
  cfgScale: string
  imageCfgScale: string
  eta: string
  flowShift: string
  sampler: string
  scheduler: string
}

export type ImageDraftErrors = Partial<Record<keyof ImageDraft, string>>

export interface ImagePayloadSafetyNotice {
  description: string
  isSafe: boolean
  title: string
  tone: 'info' | 'warning'
}

export const LOCAL_SD3_SMOKE_LIMITS = {
  batchCount: 1,
  height: 256,
  steps: 1,
  width: 256,
} as const

export const DEFAULT_IMAGE_DRAFT: ImageDraft = {
  prompt: '',
  negativePrompt: '',
  width: String(LOCAL_SD3_SMOKE_LIMITS.width),
  height: String(LOCAL_SD3_SMOKE_LIMITS.height),
  steps: String(LOCAL_SD3_SMOKE_LIMITS.steps),
  seed: '',
  guidance: '',
  batchCount: '',
  cfgScale: '',
  imageCfgScale: '',
  eta: '',
  flowShift: '',
  sampler: '',
  scheduler: '',
}

function parseOptionalInteger(value: string) {
  if (!value.trim()) {
    return undefined
  }

  return Number.parseInt(value, 10)
}

function parseOptionalNumber(value: string) {
  if (!value.trim()) {
    return undefined
  }

  return Number(value)
}

function compactObject(input: Record<string, unknown>) {
  return Object.fromEntries(
    Object.entries(input).filter(([, value]) => {
      if (value === undefined || value === null) {
        return false
      }
      if (typeof value === 'string') {
        return value.trim().length > 0
      }
      if (typeof value === 'object' && !Array.isArray(value)) {
        return Object.keys(value as Record<string, unknown>).length > 0
      }
      return true
    }),
  )
}

export function validateImageDraft(draft: ImageDraft): ImageDraftErrors {
  const errors: ImageDraftErrors = {}

  if (!draft.prompt.trim()) {
    errors.prompt = 'Please enter `prompt`.'
  }

  const positiveIntegerFields: Array<[keyof ImageDraft, string]> = [
    ['width', '`width` must be greater than 0.'],
    ['height', '`height` must be greater than 0.'],
    ['steps', '`steps` must be greater than 0.'],
    ['batchCount', '`batch_count` must be greater than 0.'],
  ]

  for (const [field, message] of positiveIntegerFields) {
    const raw = draft[field].trim()
    if (!raw) {
      continue
    }
    const parsed = Number.parseInt(raw, 10)
    if (!Number.isInteger(parsed) || parsed <= 0) {
      errors[field] = message
    }
  }

  if (draft.seed.trim()) {
    const parsed = Number.parseInt(draft.seed, 10)
    if (!Number.isInteger(parsed)) {
      errors.seed = '`seed` must be an integer.'
    }
  }

  const numericFields: Array<[keyof ImageDraft, string]> = [
    ['guidance', '`guidance` must be a number.'],
    ['cfgScale', '`cfg_scale` must be a number.'],
    ['imageCfgScale', '`image_cfg_scale` must be a number.'],
    ['eta', '`eta` must be a number.'],
    ['flowShift', '`flow_shift` must be a number.'],
  ]

  for (const [field, message] of numericFields) {
    const raw = draft[field].trim()
    if (!raw) {
      continue
    }
    const parsed = Number(raw)
    if (Number.isNaN(parsed)) {
      errors[field] = message
    }
  }

  return errors
}

export function buildImagePayload(draft: ImageDraft) {
  return compactObject({
    batch_count: parseOptionalInteger(draft.batchCount),
    cfg_scale: parseOptionalNumber(draft.cfgScale),
    eta: parseOptionalNumber(draft.eta),
    flow_shift: parseOptionalNumber(draft.flowShift),
    guidance: parseOptionalNumber(draft.guidance),
    height: parseOptionalInteger(draft.height),
    image_cfg_scale: parseOptionalNumber(draft.imageCfgScale),
    negative_prompt: draft.negativePrompt.trim() || undefined,
    prompt: draft.prompt.trim(),
    sampler: draft.sampler.trim() || undefined,
    scheduler: draft.scheduler.trim() || undefined,
    seed: parseOptionalInteger(draft.seed),
    steps: parseOptionalInteger(draft.steps),
    width: parseOptionalInteger(draft.width),
  })
}

export function stringifyPayload(payload: Record<string, unknown>) {
  return JSON.stringify(payload, null, 2)
}

function parsePositiveInteger(value: unknown) {
  if (typeof value === 'number') {
    return Number.isInteger(value) && value > 0 ? value : undefined
  }

  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) {
      return undefined
    }
    const parsed = Number.parseInt(trimmed, 10)
    return Number.isInteger(parsed) && parsed > 0 ? parsed : undefined
  }

  return undefined
}

export function assessImagePayloadSafety(payload: Record<string, unknown>): ImagePayloadSafetyNotice {
  const width = parsePositiveInteger(payload.width)
  const height = parsePositiveInteger(payload.height)
  const steps = parsePositiveInteger(payload.steps)
  const batchCount = parsePositiveInteger(payload.batch_count)

  if (width === undefined || height === undefined || steps === undefined) {
    return {
      description:
        'Current image payload omits `width`, `height`, or `steps`, so the console cannot verify it against the validated local SD3 smoke profile.',
      isSafe: false,
      title: 'Local backend caution',
      tone: 'warning',
    }
  }

  if (
    width > LOCAL_SD3_SMOKE_LIMITS.width ||
    height > LOCAL_SD3_SMOKE_LIMITS.height ||
    steps > LOCAL_SD3_SMOKE_LIMITS.steps ||
    (batchCount !== undefined && batchCount > LOCAL_SD3_SMOKE_LIMITS.batchCount)
  ) {
    return {
      description:
        'Current image payload is above the validated local SD3 smoke profile (`256 x 256`, `1` step, batch `1`). Larger jobs may destabilize the local `8080` backend and surface as `502` after refresh.',
      isSafe: false,
      title: 'Local backend caution',
      tone: 'warning',
    }
  }

  return {
    description:
      'This workspace has been validated for local SD3 smoke runs at `256 x 256` and `1` step on the `8080` backend. Start here, then scale up deliberately.',
    isSafe: true,
    title: 'Local SD3 smoke preset',
    tone: 'info',
  }
}
