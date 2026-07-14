import { AppErrorBoundary } from '@/app/error-boundary'
import { ConsoleShell } from '@/app/layout/console-shell'
import { AppProviders } from '@/app/providers'

export function App() {
  return (
    <AppProviders>
      <AppErrorBoundary>
        <ConsoleShell />
      </AppErrorBoundary>
    </AppProviders>
  )
}

export default App
