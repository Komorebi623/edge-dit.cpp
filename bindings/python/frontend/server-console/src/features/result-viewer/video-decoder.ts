import type { EdgeDitVideoGenerationResult } from '@/shared/model/jobs'

export interface DecodedVideoFrame {
  metadata: EdgeDitVideoGenerationResult['frames'][number]['metadata']
  src: string
}

export function decodeVideoResult(result?: EdgeDitVideoGenerationResult) {
  if (!result) {
    return []
  }

  return result.frames.map((frame) => {
    atob(frame.b64_png)
    return {
      metadata: frame.metadata,
      src: `data:image/png;base64,${frame.b64_png}`,
    }
  }) satisfies DecodedVideoFrame[]
}
