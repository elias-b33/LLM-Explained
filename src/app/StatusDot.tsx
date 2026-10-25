import { cn } from '@/lib/cn'
import type { ModuleStatus } from '@/types'
import { CheckIcon } from './icons'

/** A small indicator of a module's completion status. */
export function StatusDot({ status, className }: { status: ModuleStatus; className?: string }) {
  if (status === 'done') {
    return (
      <span
        className={cn(
          'flex h-5 w-5 items-center justify-center rounded-full text-white',
          className,
        )}
        style={{ backgroundColor: 'var(--accent)' }}
      >
        <CheckIcon size={13} strokeWidth={2.5} />
      </span>
    )
  }
  return (
    <span
      className={cn(
        'flex h-5 w-5 items-center justify-center rounded-full border',
        status === 'in-progress' ? 'border-accent' : 'border-border-strong',
        className,
      )}
    >
      {status === 'in-progress' && (
        <span className="h-2 w-2 rounded-full" style={{ backgroundColor: 'var(--accent)' }} />
      )}
    </span>
  )
}
