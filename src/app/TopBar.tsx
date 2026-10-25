import { useTheme } from '@/theme/ThemeProvider'
import { useProgress } from '@/progress/ProgressProvider'
import { MenuIcon, MoonIcon, SunIcon, ResetIcon } from './icons'

export function TopBar({ onMenu }: { onMenu: () => void }) {
  const { isDark, toggle } = useTheme()
  const { resetAll, completedCount } = useProgress()

  const onReset = () => {
    if (completedCount === 0 || window.confirm('Reset all your progress? This cannot be undone.')) {
      resetAll()
    }
  }

  return (
    <header className="sticky top-0 z-20 flex h-14 items-center gap-2 border-b border-border bg-bg/70 px-4 backdrop-blur-xl">
      <button
        onClick={onMenu}
        className="rounded-lg p-2 text-fg-muted hover:bg-surface-2 hover:text-fg md:hidden"
        aria-label="Open menu"
      >
        <MenuIcon />
      </button>

      <span className="text-sm font-medium text-fg-muted md:hidden">LLM Explainer</span>

      <div className="flex-1" />

      <button
        onClick={onReset}
        className="hidden items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-xs font-medium text-fg-muted hover:bg-surface-2 hover:text-fg sm:flex"
      >
        <ResetIcon size={15} />
        Reset progress
      </button>

      <button
        onClick={toggle}
        className="rounded-lg p-2 text-fg-muted hover:bg-surface-2 hover:text-fg"
        aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
        title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
      >
        {isDark ? <SunIcon /> : <MoonIcon />}
      </button>
    </header>
  )
}
