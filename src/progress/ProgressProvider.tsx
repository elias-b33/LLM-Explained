import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  type ReactNode,
} from 'react'
import { useLocalStorage } from '@/hooks/useLocalStorage'
import type { ModuleStatus } from '@/types'

interface ProgressState {
  status: Record<string, ModuleStatus>
  /** best quiz score per module slug, as a fraction 0..1 */
  quizScores: Record<string, number>
}

interface ProgressContextValue {
  status: Record<string, ModuleStatus>
  quizScores: Record<string, number>
  getStatus: (slug: string) => ModuleStatus
  markInProgress: (slug: string) => void
  markDone: (slug: string) => void
  recordQuiz: (slug: string, fraction: number) => void
  resetAll: () => void
  completedCount: number
}

const ProgressContext = createContext<ProgressContextValue | null>(null)
const EMPTY: ProgressState = { status: {}, quizScores: {} }

export function ProgressProvider({ children }: { children: ReactNode }) {
  const [state, setState, reset] = useLocalStorage<ProgressState>('llm-progress', EMPTY)

  const getStatus = useCallback(
    (slug: string): ModuleStatus => state.status[slug] ?? 'unseen',
    [state.status],
  )

  const markInProgress = useCallback(
    (slug: string) => {
      setState((prev) => {
        const cur = prev.status[slug]
        if (cur === 'done' || cur === 'in-progress') return prev
        return { ...prev, status: { ...prev.status, [slug]: 'in-progress' } }
      })
    },
    [setState],
  )

  const markDone = useCallback(
    (slug: string) => {
      setState((prev) => {
        if (prev.status[slug] === 'done') return prev
        return { ...prev, status: { ...prev.status, [slug]: 'done' } }
      })
    },
    [setState],
  )

  const recordQuiz = useCallback(
    (slug: string, fraction: number) => {
      setState((prev) => ({
        ...prev,
        quizScores: {
          ...prev.quizScores,
          [slug]: Math.max(prev.quizScores[slug] ?? 0, fraction),
        },
      }))
    },
    [setState],
  )

  const completedCount = useMemo(
    () => Object.values(state.status).filter((s) => s === 'done').length,
    [state.status],
  )

  const value: ProgressContextValue = {
    status: state.status,
    quizScores: state.quizScores,
    getStatus,
    markInProgress,
    markDone,
    recordQuiz,
    resetAll: reset,
    completedCount,
  }

  return <ProgressContext.Provider value={value}>{children}</ProgressContext.Provider>
}

// eslint-disable-next-line react-refresh/only-export-components
export function useProgress(): ProgressContextValue {
  const ctx = useContext(ProgressContext)
  if (!ctx) throw new Error('useProgress must be used within a ProgressProvider')
  return ctx
}
