import { NavLink } from 'react-router-dom'
import { motion, AnimatePresence } from 'motion/react'
import { MODULES, moduleLabel } from '@/modules/registry'
import { useProgress } from '@/progress/ProgressProvider'
import { cn } from '@/lib/cn'
import { BookIcon, CloseIcon } from './icons'
import { StatusDot } from './StatusDot'

function ProgressSummary() {
  const { completedCount } = useProgress()
  const pct = Math.round((completedCount / MODULES.length) * 100)
  return (
    <div className="px-5 pb-4">
      <div className="mb-2 flex items-center justify-between text-xs text-fg-muted">
        <span>Exhibits explored</span>
        <span className="tabular-nums">
          {completedCount} / {MODULES.length}
        </span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-surface-2">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundImage: 'linear-gradient(to right, var(--accent), var(--accent-2))' }}
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 200, damping: 30 }}
        />
      </div>
    </div>
  )
}

function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  const { getStatus } = useProgress()
  return (
    <nav className="flex-1 overflow-y-auto px-3 pb-8">
      <ul className="space-y-0.5">
        {MODULES.map((m) => {
          const status = getStatus(m.slug)
          return (
            <li key={m.id}>
              <NavLink
                to={`/module/${m.slug}`}
                onClick={onNavigate}
                className={({ isActive }) =>
                  cn(
                    'group flex items-start gap-3 rounded-xl px-3 py-2.5 transition-colors',
                    isActive
                      ? 'bg-surface-2 text-fg'
                      : 'text-fg-muted hover:bg-surface-2/60 hover:text-fg',
                  )
                }
              >
                {({ isActive }) => (
                  <>
                    <span
                      className={cn(
                        'mt-0.5 font-mono text-[11px] tabular-nums',
                        isActive ? 'text-accent' : 'text-fg-subtle',
                      )}
                    >
                      {moduleLabel(m.index)}
                    </span>
                    <span className="flex-1 text-sm leading-snug font-medium">{m.title}</span>
                    <StatusDot status={status} className="mt-0.5 shrink-0" />
                  </>
                )}
              </NavLink>
            </li>
          )
        })}
      </ul>
    </nav>
  )
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <NavLink
        to="/"
        onClick={onNavigate}
        className="flex items-center gap-2.5 px-5 pt-5 pb-4 text-fg"
      >
        <span
          className="flex h-8 w-8 items-center justify-center rounded-lg text-white"
          style={{ backgroundImage: 'linear-gradient(135deg, var(--accent), var(--accent-2))' }}
        >
          <BookIcon size={17} />
        </span>
        <span className="text-[15px] font-semibold tracking-tight">LLM Explainer</span>
      </NavLink>
      <ProgressSummary />
      <SidebarNav onNavigate={onNavigate} />
    </div>
  )
}

export function Sidebar({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  return (
    <>
      {/* Desktop: fixed rail */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-border bg-surface/40 backdrop-blur-xl md:block">
        <SidebarContent />
      </aside>

      {/* Mobile: slide-over */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-40 bg-black/50 md:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
            />
            <motion.aside
              className="fixed inset-y-0 left-0 z-50 w-72 border-r border-border bg-surface md:hidden"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            >
              <button
                onClick={onClose}
                className="absolute top-4 right-4 z-10 rounded-lg p-1.5 text-fg-muted hover:bg-surface-2 hover:text-fg"
                aria-label="Close menu"
              >
                <CloseIcon />
              </button>
              <SidebarContent onNavigate={onClose} />
            </motion.aside>
          </>
        )}
      </AnimatePresence>
    </>
  )
}
