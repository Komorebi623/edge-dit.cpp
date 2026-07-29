export interface VideoDraft {
  prompt: string
  negativePrompt: string
  width: string
  height: string
  frames: string
  steps: string
  seed: string
  guidance: string
  cfgScale: string
  eta: string
  flowShift: string
  sampler: string
  scheduler: string
}

export type VideoDraftErrors = Partial<Record<keyof VideoDraft, string>>

export const DEFAULT_VIDEO_DRAFT: VideoDraft = {
  prompt: '',
  negativePrompt: '',
  width: '416',
  height: '240',
  frames: '9',
  steps: '20',
  seed: '',
  guidance: '',
  cfgScale: '',
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

export function validateVideoDraft(draft: VideoDraft): VideoDraftErrors {
  const errors: VideoDraftErrors = {}

  if (!draft.prompt.trim()) {
    errors.prompt = 'Please enter `prompt`.'
  }

  const positiveIntegerFields: Array<[keyof VideoDraft, string]> = [
    ['width', '`width` must be greater than 0.'],
    ['height', '`height` must be greater than 0.'],
    ['frames', '`frames` must be greater than 0.'],
    ['steps', '`steps` must be greater than 0.'],
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

  const numericFields: Array<[keyof VideoDraft, string]> = [
    ['guidance', '`guidance` must be a number.'],
    ['cfgScale', '`cfg_scale` must be a number.'],
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

export function buildVideoPayload(draft: VideoDraft) {
  return compactObject({
    cfg_scale: parseOptionalNumber(draft.cfgScale),
    eta: parseOptionalNumber(draft.eta),
    flow_shift: parseOptionalNumber(draft.flowShift),
    frames: parseOptionalInteger(draft.frames),
    guidance: parseOptionalNumber(draft.guidance),
    height: parseOptionalInteger(draft.height),
    negative_prompt: draft.negativePrompt.trim() || undefined,
    prompt: draft.prompt.trim(),
    sampler: draft.sampler.trim() || undefined,
    scheduler: draft.scheduler.trim() || undefined,
    seed: parseOptionalInteger(draft.seed),
    steps: parseOptionalInteger(draft.steps),
    width: parseOptionalInteger(draft.width),
  })
}
