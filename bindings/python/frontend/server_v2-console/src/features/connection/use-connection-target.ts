import { useState } from 'react'

import {
  loadConnectionTarget,
  normalizeConnectionTarget,
  saveConnectionTarget,
} from '@/shared/model/connection-target'
import type { ConnectionTarget } from '@/shared/model/server'

export function useConnectionTarget() {
  const [target, setTargetState] = useState<ConnectionTarget>(() => loadConnectionTarget())

  function setTarget(nextTarget: ConnectionTarget) {
    const normalized = normalizeConnectionTarget(nextTarget)
    setTargetState(normalized)
    saveConnectionTarget(normalized)
  }

  return {
    setTarget,
    target,
  }
}
