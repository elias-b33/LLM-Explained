import type { ReactNode } from 'react'
import type { DemoStatus } from '@/types'

function StatusBadge({ status }: { status: DemoStatus }) {
  const real = status === 'real'
  return (
    <span
      className="tag shrink-0"
      style={{ color: real ? 'var(--success)' : 'var(--warning)' }}
      title={
        real
          ? 'This demo computes genuine output in your browser.'
          : 'This demo is a curated illustration, not a real model’s internals.'
      }
    >
      <span className="h-1.5 w-1.5 rounded-full bg-current" aria-hidden="true" />
      {real ? 'Real' : 'Illustrative'}
    </span>
  )
}

/**
 * A museum glass vitrine around an interactive demo: an oak base, a pane
 * reflection, and the exhibit's violet light leaking along the top edge.
 * The `status` badge is required — a demo cannot be mounted without honestly
 * declaring whether it is real or illustrative.
 */
export function DemoFrame({
  title,
  status,
  caption,
  children,
}: {
  title: string
  status: DemoStatus
  caption?: string
  children: ReactNode
}) {
  return (
    <figure className="vitrine my-4 rounded-sm">
      <figcaption className="relative z-[3] flex items-center justify-between gap-3 border-b border-border px-4 py-3 sm:px-5">
        <span className="min-w-0">
          <span className="flex items-center gap-2 text-[10px] font-semibold tracking-[0.2em] text-accent uppercase">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: 'var(--accent-2)', boxShadow: '0 0 8px var(--accent-2)' }}
              aria-hidden="true"
            />
            Interactive exhibit
          </span>
          <span className="mt-0.5 block font-display text-xl leading-tight text-fg">{title}</span>
        </span>
        <StatusBadge status={status} />
      </figcaption>
      {/* z above the pane reflection so controls stay crisp and clickable */}
      <div className="relative z-[3] p-4 sm:p-5">{children}</div>
      {caption && (
        <p className="relative z-[3] border-t border-border bg-surface-2/40 px-4 py-2.5 text-xs leading-relaxed text-fg-muted italic sm:px-5">
          {caption}
        </p>
      )}
    </figure>
  )
}
