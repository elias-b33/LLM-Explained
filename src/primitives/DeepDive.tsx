import { useState, type ReactNode } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { ChevronDownIcon } from '@/app/icons'
import { cn } from '@/lib/cn'

/** Collapsible "show the math" panel for optional depth. */
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
    <div className="overflow-hidden rounded-xl border border-border bg-surface/50">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 px-4 py-3 text-left text-sm font-medium text-fg transition-colors hover:bg-surface-2/60"
        aria-expanded={open}
      >
        <ChevronDownIcon
          size={16}
          className={cn('transition-transform duration-200', open && 'rotate-180')}
          style={{ color: 'var(--accent)' }}
        />
        {title}
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
            <div className="prose-content border-t border-border px-4 py-4 text-[0.95rem]">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
