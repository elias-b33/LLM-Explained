import { useState, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ChevronDownIcon } from '@/app/icons'
import { cn } from '@/lib/cn'

/** Collapsible "show the math" drawer — like the pull-out archive drawer under a display case. */
export function DeepDive({
  title = 'Show the math',
  children,
  defaultOpen = false,
}: {
  title?: string
  children: ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)

  return (
    <div className="overflow-hidden border border-border bg-surface/80 shadow-[0_10px_24px_-20px_rgb(var(--shadow-color)/0.5)]">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-3 px-4 py-3.5 text-left transition-colors hover:bg-surface-2/60"
        aria-expanded={open}
      >
        <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full border border-brass text-brass-ink">
          <ChevronDownIcon size={14} className={cn('transition-transform duration-200', open && 'rotate-180')} />
        </span>
        <span className="min-w-0">
          <span className="block text-[10px] font-semibold tracking-[0.2em] text-fg-subtle uppercase">
            From the archive
          </span>
          <span className="block font-display text-lg leading-tight text-fg">{title}</span>
        </span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.28, ease: [0.22, 1, 0.36, 1] }}
          >
            <div className="prose-content border-t border-dashed border-border-strong bg-bg/40 px-4 py-4 text-[0.95rem] sm:px-5">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
