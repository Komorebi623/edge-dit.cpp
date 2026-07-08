import { DEFAULT_CONNECTION_TARGET } from '@/shared/model/connection-target'

import { buildApiUrl, getCapabilities, getHealth } from './client'

describe('api client', () => {
  it('uses the dev proxy path for the default local target', () => {
    expect(buildApiUrl(DEFAULT_CONNECTION_TARGET, '/health')).toBe('/ed/v2/health')
  })

  it('returns a full URL for non-default local targets', () => {
    expect(buildApiUrl({ baseUrl: 'http://127.0.0.1:9090', prefix: '/edge-dit/v2' }, '/capabilities')).toBe(
      'http://127.0.0.1:9090/edge-dit/v2/capabilities',
    )
  })

  it('fetches health and capabilities', async () => {
    await expect(getHealth(DEFAULT_CONNECTION_TARGET)).resolves.toMatchObject({
      model: 'edge-dit-model',
      status: 'ok',
    })

    await expect(getCapabilities(DEFAULT_CONNECTION_TARGET)).resolves.toMatchObject({
      model: 'edge-dit-model',
      supports: {
        image: true,
        video: true,
      },
    })
  })
})
