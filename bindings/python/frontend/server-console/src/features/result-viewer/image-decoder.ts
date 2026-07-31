import type { EdgeDitImageGenerationResult } from '@/shared/model/jobs'

export interface DecodedImageItem {
  metadata: EdgeDitImageGenerationResult['data'][number]['metadata']
  src: string
}

export function decodeImageResult(result?: EdgeDitImageGenerationResult) {
  if (!result) {
    return []
  }

  return result.data.map((item) => {
    // Validate the payload before constructing a data URL so local decode errors surface clearly.
    atob(item.b64_png)
    return {
      metadata: item.metadata,
      src: `data:image/png;base64,${item.b64_png}`,
    }
  }) satisfies DecodedImageItem[]
}
