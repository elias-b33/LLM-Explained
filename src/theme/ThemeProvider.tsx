import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from 'react'
import { useLocalStorage } from '@/hooks/useLocalStorage'

export type ThemeChoice = 'dark' | 'light' | 'system'

interface ThemeContextValue {
  choice: ThemeChoice
  isDark: boolean
  setChoice: (c: ThemeChoice) => void
  toggle: () => void
}

const ThemeContext = createContext<ThemeContextValue | null>(null)

function systemPrefersDark(): boolean {
  return typeof window !== 'undefined' && window.matchMedia('(prefers-color-scheme: dark)').matches
}

function resolveDark(choice: ThemeChoice): boolean {
  return choice === 'system' ? systemPrefersDark() : choice === 'dark'
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [choice, setChoice] = useLocalStorage<ThemeChoice>('llm-theme', 'dark')
  const [isDark, setIsDark] = useState(() => resolveDark(choice))

  useEffect(() => {
    const apply = () => {
      const dark = resolveDark(choice)
      document.documentElement.classList.toggle('dark', dark)
      setIsDark(dark)
    }
    apply()
    if (choice === 'system') {
      const mq = window.matchMedia('(prefers-color-scheme: dark)')
      mq.addEventListener('change', apply)
      return () => mq.removeEventListener('change', apply)
    }
  }, [choice])

  const toggle = useCallback(() => {
    setChoice(resolveDark(choice) ? 'light' : 'dark')
  }, [choice, setChoice])

  return (
    <ThemeContext.Provider value={{ choice, isDark, setChoice, toggle }}>
      {children}
    </ThemeContext.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useTheme(): ThemeContextValue {
  const ctx = useContext(ThemeContext)
  if (!ctx) throw new Error('useTheme must be used within a ThemeProvider')
  return ctx
}
