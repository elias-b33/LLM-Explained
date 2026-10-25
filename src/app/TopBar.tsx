import { Link, useLocation } from 'react-router-dom'
import { useTheme } from '@/theme/ThemeProvider'
import { useProgress } from '@/progress/ProgressProvider'
import { getModule, moduleLabel } from '@/modules/registry'
import { MenuIcon, MoonIcon, SunIcon, ResetIcon, ChevronRightIcon } from './icons'

/** Wayfinding sign: "Main hall › Exhibit 04 · Attention". */
function Wayfinding() {
  const { pathname } = useLocation()
  const slug = pathname.startsWith('/module/') ? pathname.slice('/module/'.length) : undefined
  const meta = getModule(slug)
  return (
    <nav aria-label="Breadcrumb" className="flex min-w-0 items-center gap-1.5 text-xs">
      <Link
        to="/"
        className="exhibit-label shrink-0 !tracking-[0.18em] transition-colors hover:text-fg"
        aria-current={meta ? undefined : 'page'}
      >
        Main hall
      </Link>
      {meta && (
        <>
          <ChevronRightIcon size={13} className="shrink-0 text-brass" />
          <span className="truncate text-fg-muted" aria-current="page">
            <span className="hidden font-semibold tracking-[0.16em] text-fg-subtle uppercase sm:inline">
              Exhibit {moduleLabel(meta.index)} ·{' '}
            </span>
            <span className="font-display text-[15px] text-fg">{meta.title}</span>
          </span>
        </>
      )}
    </nav>
  )
}

export function TopBar({ onMenu }: { onMenu: () => void }) {
  const { isDark, toggle } = useTheme()
  const { resetAll, completedCount } = useProgress()

  const onReset = () => {
    if (completedCount === 0 || window.confirm('Reset all your progress? This cannot be undone.')) {
      resetAll()
    }
  }

  return (
    <header className="sticky top-0 z-20 border-b border-border bg-bg/80 backdrop-blur-md">
      <div className="flex h-14 items-center gap-2 px-4 sm:px-6">
        <button
          onClick={onMenu}
          className="-ml-1 rounded-md p-2 text-fg-muted hover:bg-surface-2 hover:text-fg md:hidden"
          aria-label="Open gallery guide"
        >
          <MenuIcon />
        </button>

        <Wayfinding />

        <div className="flex-1" />

        <button
          onClick={onReset}
          className="hidden items-center gap-1.5 rounded-md px-2.5 py-1.5 text-[11px] font-semibold tracking-[0.12em] text-fg-muted uppercase hover:bg-surface-2 hover:text-fg sm:flex"
        >
          <ResetIcon size={14} />
          Reset progress
        </button>

        <button
          onClick={toggle}
          className="flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1.5 text-fg-muted transition-colors hover:border-brass hover:text-fg"
          aria-label={isDark ? 'Switch to daytime gallery (light mode)' : 'Switch to after hours (dark mode)'}
          title={isDark ? 'Daytime gallery' : 'After hours'}
        >
          {isDark ? <SunIcon size={16} /> : <MoonIcon size={16} />}
          <span className="hidden text-[11px] font-semibold tracking-[0.12em] uppercase lg:inline">
            {isDark ? 'Daytime' : 'After hours'}
          </span>
        </button>
      </div>
      <div className="brass-rule opacity-60" aria-hidden="true" />
    </header>
  )
}
