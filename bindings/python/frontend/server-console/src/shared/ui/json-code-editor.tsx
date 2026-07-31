import CodeMirror from '@uiw/react-codemirror'
import { githubLightInit } from '@uiw/codemirror-theme-github'
import { json } from '@codemirror/lang-json'
import { EditorView } from '@codemirror/view'
import { useMemo } from 'react'

import { cn } from '@/shared/lib/cn'

const jsonEditorTheme = githubLightInit({
  settings: {
    background: 'transparent',
    foreground: 'var(--text-primary)',
    caret: 'var(--accent)',
    selection: 'rgba(42, 96, 220, 0.14)',
    selectionMatch: 'rgba(42, 96, 220, 0.1)',
    lineHighlight: 'rgba(42, 96, 220, 0.04)',
    gutterBackground: '#f4f7f4',
    gutterForeground: 'var(--text-muted)',
  },
})

export function JsonCodeEditor({
  ariaLabel,
  className,
  maxHeight = 420,
  minHeight = 220,
  onChange,
  readOnly = false,
  value,
}: {
  ariaLabel: string
  className?: string
  maxHeight?: number
  minHeight?: number
  onChange?: (value: string) => void
  readOnly?: boolean
  value: string
}) {
  const extensions = useMemo(() => [json(), EditorView.lineWrapping], [])
  const basicSetup = useMemo(
    () => ({
      foldGutter: true,
      highlightActiveLine: !readOnly,
      highlightActiveLineGutter: !readOnly,
      lineNumbers: true,
    }),
    [readOnly],
  )

  return (
    <div
      className={cn(
        'overflow-hidden rounded-[22px] border border-[var(--border-strong)] bg-[linear-gradient(180deg,#ffffff,#f6f8f5)] shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]',
        readOnly ? 'bg-[linear-gradient(180deg,#fbfcfa,#f3f5f1)]' : '',
        className,
      )}
    >
      <CodeMirror
        aria-label={ariaLabel}
        basicSetup={basicSetup}
        className="json-code-editor text-sm"
        editable={!readOnly}
        extensions={extensions}
        maxHeight={`${maxHeight}px`}
        minHeight={`${minHeight}px`}
        onChange={onChange}
        readOnly={readOnly}
        theme={jsonEditorTheme}
        value={value}
      />
    </div>
  )
}
