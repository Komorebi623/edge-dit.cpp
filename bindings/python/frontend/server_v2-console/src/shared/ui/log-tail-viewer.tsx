import type { ReactNode } from 'react'

import { cn } from '@/shared/lib/cn'

const LOG_TOKEN_PATTERN =
  /(\[(?:error|warning|warn|info|debug)\]|'(?:[^']*)'|\/[A-Za-z0-9._/+:-]+|\b\d+(?:\.\d+)?(?:MB|MiB|GiB|ms|s)\b|\b(?:CUDA\d+|CPU|RAM|VMM|VRAM)\b)/gi

export function LogTailViewer({
  entries,
}: {
  entries: Array<{
    line: string
    stream: 'stderr' | 'stdout'
    time_ms: number
  }>
}) {
  return (
    <div className="overflow-hidden rounded-[22px] border border-[var(--border-strong)] bg-[linear-gradient(180deg,#fbfcfa,#f3f6f1)] shadow-[inset_0_1px_0_rgba(255,255,255,0.55)]">
      <div className="max-h-[280px] overflow-auto">
        {entries.map((entry, index) => {
          const logMeta = deriveLogMeta(entry.line, entry.stream)

          return (
            <div
              className={cn(
                'grid grid-cols-[72px_minmax(0,1fr)] gap-2.5 border-b border-[rgba(31,42,56,0.06)] px-3 py-2.5 last:border-b-0',
                logMeta.tone === 'error' && 'bg-[rgba(189,44,60,0.04)]',
                logMeta.tone === 'warning' && 'bg-[rgba(169,101,0,0.05)]',
                logMeta.tone === 'info' && 'bg-[rgba(42,96,220,0.035)]',
              )}
              key={`${entry.stream}:${entry.time_ms}:${index}`}
            >
              <div className="flex min-w-0 flex-col gap-1 pt-0.5">
                <div className="flex items-center gap-1.5">
                  <span className="font-mono text-[10px] uppercase tracking-[0.12em] text-[var(--text-muted)]">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <span
                    className={cn(
                      'inline-flex h-fit items-center justify-center rounded-full border px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-[0.12em]',
                      logMeta.tone === 'error' && 'border-[var(--error-border)] bg-[var(--error-soft)] text-[var(--error)]',
                      logMeta.tone === 'warning' &&
                        'border-[var(--warning-border)] bg-[var(--warning-soft)] text-[var(--warning)]',
                      logMeta.tone === 'info' && 'border-[var(--info-border)] bg-[var(--info-soft)] text-[var(--info)]',
                      logMeta.tone === 'neutral' &&
                        'border-[var(--border-subtle)] bg-[var(--surface-card)] text-[var(--text-muted)]',
                    )}
                  >
                    {logMeta.levelLabel}
                  </span>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-[0.08em] text-[var(--text-muted)]">
                  {formatLogTime(entry.time_ms)}
                </span>
                <span className="font-mono text-[10px] lowercase tracking-[0.04em] text-[var(--text-muted)]">
                  {entry.stream}
                </span>
              </div>
              <p className="min-w-0 whitespace-pre-wrap break-words pr-1 font-mono text-[11px] leading-5 text-[var(--text-secondary)]">
                {entry.line.length === 0 ? (
                  <span className="italic text-[var(--text-muted)]">empty line</span>
                ) : (
                  renderHighlightedLogLine(entry.line)
                )}
              </p>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function deriveLogMeta(line: string, stream: 'stderr' | 'stdout') {
  const lower = line.toLowerCase()
  if (lower.includes('[error]') || /\berror\b/.test(lower)) {
    return {
      levelLabel: 'err',
      tone: 'error' as const,
    }
  }
  if (lower.includes('[warning]') || /\bwarn(?:ing)?\b/.test(lower)) {
    return {
      levelLabel: 'warn',
      tone: 'warning' as const,
    }
  }
  if (lower.includes('[info]') || /\bloaded\b|\bloading\b|\binitialized\b|\bready\b/.test(lower)) {
    return {
      levelLabel: 'info',
      tone: 'info' as const,
    }
  }
  if (lower.includes('[debug]') || /\bdebug\b/.test(lower)) {
    return {
      levelLabel: 'dbg',
      tone: 'neutral' as const,
    }
  }
  if (stream === 'stderr') {
    return {
      levelLabel: 'warn',
      tone: 'warning' as const,
    }
  }
  return {
    levelLabel: 'log',
    tone: 'neutral' as const,
  }
}

function renderHighlightedLogLine(line: string) {
  const nodes: ReactNode[] = []
  let lastIndex = 0

  for (const match of line.matchAll(LOG_TOKEN_PATTERN)) {
    const token = match[0]
    const start = match.index ?? 0

    if (start > lastIndex) {
      nodes.push(line.slice(lastIndex, start))
    }

    nodes.push(
      <span className={tokenClassName(token)} key={`${start}:${token}`}>
        {token}
      </span>,
    )
    lastIndex = start + token.length
  }

  if (lastIndex < line.length) {
    nodes.push(line.slice(lastIndex))
  }

  return nodes.length > 0 ? nodes : line
}

function tokenClassName(token: string) {
  const lower = token.toLowerCase()

  if (lower.includes('[error]')) {
    return 'font-semibold text-[var(--error)]'
  }
  if (lower.includes('[warning]') || lower.includes('[warn]')) {
    return 'font-semibold text-[var(--warning)]'
  }
  if (lower.includes('[info]')) {
    return 'font-semibold text-[var(--info)]'
  }
  if (lower.includes('[debug]')) {
    return 'font-semibold text-[var(--text-muted)]'
  }
  if (token.startsWith('/')) {
    return 'text-[var(--text-muted)]'
  }
  if (token.startsWith("'")) {
    return 'font-medium text-[var(--accent)]'
  }
  if (/\b(?:CUDA\d+|CPU|RAM|VMM|VRAM)\b/i.test(token)) {
    return 'font-medium text-[var(--accent-2)]'
  }
  return 'font-medium text-[var(--accent)]'
}

function formatLogTime(timeMs: number) {
  return new Date(timeMs).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}
