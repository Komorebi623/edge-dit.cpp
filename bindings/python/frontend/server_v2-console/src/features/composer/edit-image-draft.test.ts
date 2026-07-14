import {
  DEFAULT_EDIT_IMAGE_DRAFT,
  buildEditImageDraftFromPayload,
  buildEditImagePayload,
  validateEditImageDraft,
} from './edit-image-draft'

describe('edit image draft helpers', () => {
  it('builds an init-image payload for qwen image edit style requests', () => {
    const payload = buildEditImagePayload(
      {
        ...DEFAULT_EDIT_IMAGE_DRAFT,
        cfgScale: '4.5',
        height: '512',
        inputImage: 'data:image/png;base64,AAAA',
        negativePrompt: 'blurry',
        prompt: 'Replace the mug with a teapot',
        steps: '20',
        width: '512',
      },
      'init_image_b64',
    )

    expect(payload).toEqual({
      cfg_scale: 4.5,
      height: 512,
      init_image_b64: 'data:image/png;base64,AAAA',
      negative_prompt: 'blurry',
      prompt: 'Replace the mug with a teapot',
      steps: 20,
      width: 512,
    })
  })

  it('builds a ref-image payload for flux kontext style requests', () => {
    const payload = buildEditImagePayload(
      {
        ...DEFAULT_EDIT_IMAGE_DRAFT,
        flowShift: '1.15',
        guidance: '3.5',
        inputImage: 'data:image/png;base64,BBBB',
        prompt: 'Turn this desk shot into an editorial product frame',
      },
      'ref_images_b64',
    )

    expect(payload).toEqual({
      flow_shift: 1.15,
      guidance: 3.5,
      prompt: 'Turn this desk shot into an editorial product frame',
      ref_images_b64: ['data:image/png;base64,BBBB'],
      height: 256,
      steps: 1,
      width: 256,
    })
  })

  it('requires an input image when validating the edit draft', () => {
    const errors = validateEditImageDraft({
      ...DEFAULT_EDIT_IMAGE_DRAFT,
      prompt: 'Stylize this photo',
    })

    expect(errors.inputImage).toBeDefined()
  })

  it('can hydrate the draft back from a flux kontext payload example', () => {
    const draft = buildEditImageDraftFromPayload({
      prompt: 'Polish the lighting',
      ref_images_b64: ['data:image/png;base64,CCCC'],
      steps: 8,
      width: 384,
      height: 384,
    })

    expect(draft.prompt).toBe('Polish the lighting')
    expect(draft.inputImage).toBe('data:image/png;base64,CCCC')
    expect(draft.width).toBe('384')
    expect(draft.height).toBe('384')
    expect(draft.steps).toBe('8')
  })
})
