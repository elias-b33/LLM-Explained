import { cn } from '@/lib/cn'
import type { ModuleStatus } from '@/types'
import { CheckIcon } from './icons'

const LABEL: Record<ModuleStatus, string> = {
  done: 'Visited',
  'in-progress': 'In progress',
  unseen: 'Not yet visited',
}

/** A small brass "visited" seal showing a module's completion status. */
export function StatusDot({ status, className }: { status: ModuleStatus; className?: string }) {
  if (status === 'done') {
    return (
      <span
        role="img"
        aria-label={LABEL[status]}
        title={LABEL[status]}
        className={cn(
          'flex h-5 w-5 items-center justify-center rounded-full border border-brass',
          className,
        )}
        style={{
          backgroundImage: 'linear-gradient(140deg, var(--gold), var(--brass))',
          boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.35)',
          color: '#2a1f12',
        }}
      >
        <CheckIcon size={12} strokeWidth={2.6} />
      </span>
    )
  }
  return (
    <span
      role="img"
      aria-label={LABEL[status]}
      title={LABEL[status]}
      className={cn(
        'flex h-5 w-5 items-center justify-center rounded-full border',
        status === 'in-progress' ? 'border-accent' : 'border-border-strong',
        className,
      )}
    >
      {status === 'in-progress' && (
        <span
          className="h-2 w-2 rounded-full"
          style={{ backgroundColor: 'var(--accent)', boxShadow: '0 0 6px var(--accent)' }}
        />
      )}
    </span>
  )
}
