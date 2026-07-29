export type EditImageBindingField = 'init_image_b64' | 'ref_images_b64'

export interface EditImageDraft {
  prompt: string
  negativePrompt: string
  width: string
  height: string
  steps: string
  seed: string
  guidance: string
  cfgScale: string
  flowShift: string
  inputImage: string
  inputImageName: string
}

export type EditImageDraftErrors = Partial<Record<keyof EditImageDraft, string>>

export const DEFAULT_EDIT_IMAGE_DRAFT: EditImageDraft = {
  prompt: '',
  negativePrompt: '',
  width: '256',
  height: '256',
  steps: '1',
  seed: '',
  guidance: '',
  cfgScale: '',
  flowShift: '',
  inputImage: '',
  inputImageName: '',
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
      if (Array.isArray(value)) {
        return value.length > 0
      }
      return true
    }),
  )
}

function payloadString(value: unknown) {
  if (typeof value === 'number' || typeof value === 'string') {
    return String(value)
  }
  return ''
}

function firstRefImage(payload: Record<string, unknown>) {
  const refImages = payload.ref_images_b64
  if (!Array.isArray(refImages) || typeof refImages[0] !== 'string') {
    return ''
  }
  return refImages[0]
}

function looksLikeInlineImagePayload(value: string) {
  return value.startsWith('data:image/') || /^[A-Za-z0-9+/=]+$/.test(value)
}

export function validateEditImageDraft(draft: EditImageDraft): EditImageDraftErrors {
  const errors: EditImageDraftErrors = {}

  if (!draft.prompt.trim()) {
    errors.prompt = 'Please enter `prompt`.'
  }

  if (!draft.inputImage.trim()) {
    errors.inputImage = 'Please provide an input image first.'
  }

  const positiveIntegerFields: Array<[keyof EditImageDraft, string]> = [
    ['width', '`width` must be greater than 0.'],
    ['height', '`height` must be greater than 0.'],
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

  const numericFields: Array<[keyof EditImageDraft, string]> = [
    ['guidance', '`guidance` must be a number.'],
    ['cfgScale', '`cfg_scale` must be a number.'],
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

export function buildEditImagePayload(draft: EditImageDraft, bindingField: EditImageBindingField) {
  return compactObject({
    cfg_scale: parseOptionalNumber(draft.cfgScale),
    flow_shift: parseOptionalNumber(draft.flowShift),
    guidance: parseOptionalNumber(draft.guidance),
    height: parseOptionalInteger(draft.height),
    negative_prompt: draft.negativePrompt.trim() || undefined,
    prompt: draft.prompt.trim(),
    seed: parseOptionalInteger(draft.seed),
    steps: parseOptionalInteger(draft.steps),
    width: parseOptionalInteger(draft.width),
    [bindingField]: draft.inputImage.trim()
      ? bindingField === 'ref_images_b64'
        ? [draft.inputImage.trim()]
        : draft.inputImage.trim()
      : undefined,
  })
}

export function buildEditImageDraftFromPayload(payload: Record<string, unknown>): EditImageDraft {
  const initImage = typeof payload.init_image_b64 === 'string' ? payload.init_image_b64 : ''
  const refImage = firstRefImage(payload)
  const candidateImage = initImage || refImage
  const inputImage = looksLikeInlineImagePayload(candidateImage) ? candidateImage : ''

  return {
    ...DEFAULT_EDIT_IMAGE_DRAFT,
    cfgScale: payloadString(payload.cfg_scale),
    flowShift: payloadString(payload.flow_shift),
    guidance: payloadString(payload.guidance),
    height: payloadString(payload.height) || DEFAULT_EDIT_IMAGE_DRAFT.height,
    inputImage,
    inputImageName: inputImage ? 'payload image' : '',
    negativePrompt: typeof payload.negative_prompt === 'string' ? payload.negative_prompt : '',
    prompt: typeof payload.prompt === 'string' ? payload.prompt : '',
    seed: payloadString(payload.seed),
    steps: payloadString(payload.steps) || DEFAULT_EDIT_IMAGE_DRAFT.steps,
    width: payloadString(payload.width) || DEFAULT_EDIT_IMAGE_DRAFT.width,
  }
}
