import { render, screen } from '@testing-library/react'

import { LogTailViewer } from './log-tail-viewer'

describe('LogTailViewer', () => {
  it('prefers parsed log level over raw stream when rendering badges', () => {
    render(
      <LogTailViewer
        entries={[
          {
            line: '/tmp/qwen.cpp:814 [info] qwen-image-edit step 20/20 sigma=0.020000 next=0.000000',
            stream: 'stderr',
            time_ms: new Date('2026-07-08T14:07:42+08:00').getTime(),
          },
        ]}
      />,
    )

    expect(screen.getByText('info')).toBeInTheDocument()
    expect(screen.getByText('stderr')).toBeInTheDocument()
    expect(screen.queryByText(/^err$/i)).not.toBeInTheDocument()
  })
})
