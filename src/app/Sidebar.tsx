import { NavLink, Link } from 'react-router-dom'
import { motion, AnimatePresence } from 'motion/react'
import { MODULES, moduleLabel } from '@/modules/registry'
import { useProgress } from '@/progress/ProgressProvider'
import { cn } from '@/lib/cn'
import { CloseIcon, MuseumIcon, PlanetIcon, SparklesIcon, WrenchIcon } from './icons'
import { StatusDot } from './StatusDot'
import type { ModuleMeta } from '@/types'

/** The floor plan: exhibits grouped into galleries, like a printed visitor map. */
const FLOORS: { name: string; range: [number, number] }[] = [
  { name: 'Ground floor · Foundations', range: [0, 2] },
  { name: 'First floor · The machine', range: [3, 6] },
  { name: 'Second floor · Learning & use', range: [7, 9] },
  { name: 'Rotunda', range: [10, 10] },
]

function ProgressSummary() {
  const { completedCount } = useProgress()
  const pct = Math.round((completedCount / MODULES.length) * 100)
  return (
    <div className="px-5 pb-5">
      <div className="mb-2 flex items-baseline justify-between">
        <span className="exhibit-label !text-[10px]">Exhibits visited</span>
        <span className="font-display text-base text-fg tabular-nums">
          {completedCount} <span className="text-fg-subtle">/ {MODULES.length}</span>
        </span>
      </div>
      <div className="h-[3px] w-full overflow-hidden bg-surface-2">
        <motion.div
          className="h-full"
          style={{ backgroundImage: 'linear-gradient(to right, var(--brass), var(--gold))' }}
          initial={false}
          animate={{ width: `${pct}%` }}
          transition={{ type: 'spring', stiffness: 200, damping: 30 }}
        />
      </div>
    </div>
  )
}

function NavItem({ m, onNavigate }: { m: ModuleMeta; onNavigate?: () => void }) {
  const { getStatus } = useProgress()
  return (
    <li>
      <NavLink
        to={`/module/${m.slug}`}
        onClick={onNavigate}
        className={({ isActive }) =>
          cn(
            'group relative flex items-start gap-3 py-2 pr-3 pl-4 transition-colors',
            isActive ? 'bg-surface-2 text-fg' : 'text-fg-muted hover:bg-surface-2/60 hover:text-fg',
          )
        }
      >
        {({ isActive }) => (
          <>
            <span
              aria-hidden="true"
              className={cn(
                'absolute inset-y-1 left-0 w-[2px] transition-colors',
                isActive ? 'bg-gold' : 'bg-transparent group-hover:bg-border-strong',
              )}
            />
            <span
              className={cn(
                'w-6 shrink-0 font-display text-[15px] leading-5 font-semibold tabular-nums',
                isActive ? 'text-brass-ink' : 'text-fg-subtle',
              )}
            >
              {moduleLabel(m.index)}
            </span>
            <span className="flex-1 text-[13.5px] leading-5 font-medium">{m.title}</span>
            <StatusDot status={getStatus(m.slug)} className="mt-0.5 shrink-0" />
          </>
        )}
      </NavLink>
    </li>
  )
}

function SidebarNav({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav aria-label="Exhibits" className="flex-1 overflow-y-auto px-2 pb-4">
      {FLOORS.map((f) => (
        <div key={f.name} className="mb-4">
          <p className="px-4 pb-1.5 text-[10px] font-semibold tracking-[0.2em] text-fg-subtle uppercase">{f.name}</p>
          <ul>
            {MODULES.filter((m) => m.index >= f.range[0] && m.index <= f.range[1]).map((m) => (
              <NavItem key={m.id} m={m} onNavigate={onNavigate} />
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}

function Annexes({ onNavigate }: { onNavigate?: () => void }) {
  const link =
    'flex items-center gap-2.5 px-3 py-2 text-[13px] font-medium text-fg-muted transition-colors hover:bg-surface-2/70 hover:text-fg'
  return (
    <div className="border-t border-border px-2 py-3">
      <p className="px-3 pb-1 text-[10px] font-semibold tracking-[0.2em] text-fg-subtle uppercase">Annexes</p>
      <Link to="/museum/orientation" onClick={onNavigate} className={link}>
        <SparklesIcon size={15} className="text-accent" />
        Walk the 3D museum
      </Link>
      <Link to="/galaxy" onClick={onNavigate} className={link}>
        <PlanetIcon size={15} className="text-accent-2" />
        Planetarium · word galaxy
      </Link>
      <Link to="/workshop" onClick={onNavigate} className={link}>
        <WrenchIcon size={15} className="text-brass" />
        Workshop · train a model
      </Link>
    </div>
  )
}

function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="flex h-full flex-col">
      <NavLink to="/" onClick={onNavigate} className="group flex items-center gap-3 px-5 pt-6 pb-5 text-fg">
        <span className="flex h-10 w-10 shrink-0 items-center justify-center border border-brass bg-surface text-brass-ink shadow-[inset_0_0_0_3px_var(--surface),inset_0_0_0_4px_color-mix(in_srgb,var(--brass)_40%,transparent)]">
          <MuseumIcon size={20} />
        </span>
        <span className="leading-tight">
          <span className="block font-display text-[21px] text-fg">LLM Explainer</span>
          <span className="block text-[9.5px] font-semibold tracking-[0.22em] text-fg-subtle uppercase">
            Museum of language models
          </span>
        </span>
      </NavLink>
      <div className="brass-rule-left mx-5 mb-5 opacity-70" aria-hidden="true" />
      <ProgressSummary />
      <SidebarNav onNavigate={onNavigate} />
      <Annexes onNavigate={onNavigate} />
    </div>
  )
}

export function Sidebar({ mobileOpen, onClose }: { mobileOpen: boolean; onClose: () => void }) {
  return (
    <>
      {/* Desktop: fixed rail — the printed gallery guide */}
      <aside className="plaster fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-border-strong/70 bg-surface md:block">
        <SidebarContent />
      </aside>

      {/* Mobile: slide-over */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              className="fixed inset-0 z-40 bg-[#1a140e]/55 md:hidden"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={onClose}
            />
            <motion.aside
              className="plaster fixed inset-y-0 left-0 z-50 w-72 max-w-[85vw] border-r border-border-strong bg-surface md:hidden"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 320, damping: 34 }}
            >
              <button
                onClick={onClose}
                className="absolute top-4 right-3 z-10 rounded-md p-1.5 text-fg-muted hover:bg-surface-2 hover:text-fg"
                aria-label="Close gallery guide"
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
