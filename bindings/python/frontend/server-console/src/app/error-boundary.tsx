import { AlertTriangle, RefreshCcw } from 'lucide-react'
import { Component, type ErrorInfo, type ReactNode } from 'react'

interface AppErrorBoundaryProps {
  children: ReactNode
}

interface AppErrorBoundaryState {
  error: Error | null
}

export class AppErrorBoundary extends Component<AppErrorBoundaryProps, AppErrorBoundaryState> {
  state: AppErrorBoundaryState = {
    error: null,
  }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('Python Server Console crashed', error, info)
  }

  render() {
    if (!this.state.error) {
      return this.props.children
    }

    return (
      <main className="grid min-h-screen place-items-center bg-[var(--bg-app)] px-6 text-[var(--text-primary)]">
        <section className="w-full max-w-xl rounded-[28px] border border-[var(--error-border)] bg-[var(--surface-shell)]/92 p-6 shadow-[var(--shadow-soft)] backdrop-blur-xl">
          <div className="flex items-start gap-4">
            <div className="grid size-14 shrink-0 place-items-center rounded-3xl border border-[var(--error-border)] bg-[var(--error-soft)] text-[var(--error)]">
              <AlertTriangle className="size-7" />
            </div>
            <div className="min-w-0">
              <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-[var(--text-muted)]">
                console recovery guard
              </p>
              <h1 className="mt-2 text-2xl font-semibold tracking-[-0.03em]">The console hit a frontend error.</h1>
              <p className="mt-3 text-sm leading-6 text-[var(--text-secondary)]">
                Runtime management and backend processes may still be alive. Reload the page to recover the UI, then
                inspect the Local Runtime panel for backend state and recent events.
              </p>
            </div>
          </div>

          <div className="mt-5 rounded-2xl border border-[var(--border-subtle)] bg-[var(--surface-ink)] p-4">
            <pre className="overflow-auto whitespace-pre-wrap font-mono text-xs leading-5 text-white/75">
              {this.state.error.stack ?? this.state.error.message}
            </pre>
          </div>

          <div className="mt-5 flex flex-wrap gap-3">
            <button
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-4 py-2.5 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(42,96,220,0.26)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-hover)]"
              onClick={() => window.location.reload()}
              type="button"
            >
              <RefreshCcw className="size-4" />
              Reload Console
            </button>
          </div>
        </section>
      </main>
    )
  }
}
