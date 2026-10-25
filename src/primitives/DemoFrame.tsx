import type { ReactNode } from 'react'
import type { DemoStatus } from '@/types'
import { mix } from '@/lib/color'

function StatusBadge({ status }: { status: DemoStatus }) {
  const real = status === 'real'
  const color = real ? 'var(--success)' : 'var(--warning)'
  return (
    <span
      className="inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-1 text-[10.5px] font-semibold tracking-wide uppercase"
      style={{ color, backgroundColor: mix(color, 13) }}
      title={
        real
          ? 'This demo computes genuine output in your browser.'
          : 'This demo is a curated illustration, not a real model’s internals.'
      }
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ backgroundColor: color }} />
      {real ? 'Real' : 'Illustrative'}
    </span>
  )
}

/**
 * Card wrapper around an interactive demo. The `status` badge is required —
 * a demo cannot be mounted without honestly declaring whether it is real or
 * illustrative.
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
    <figure className="glow-card my-2 overflow-hidden rounded-2xl border border-border/60 bg-surface/70 shadow-[0_24px_60px_-34px_rgba(0,0,0,0.7)] backdrop-blur-md">
      <figcaption
        className="flex items-center justify-between gap-3 border-b border-border/70 px-4 py-3 sm:px-5"
        style={{
          backgroundImage:
            'linear-gradient(90deg, color-mix(in srgb, var(--accent) 12%, transparent), transparent 55%, color-mix(in srgb, var(--accent-2) 8%, transparent))',
        }}
      >
        <span className="flex items-center gap-2 text-sm font-semibold text-fg">
          <span
            className="h-2 w-2 rounded-full"
            style={{ background: 'var(--accent-2)', boxShadow: '0 0 10px var(--accent-2)' }}
            aria-hidden="true"
          />
          {title}
        </span>
        <StatusBadge status={status} />
      </figcaption>
      <div className="p-4 sm:p-5">{children}</div>
      {caption && (
        <p className="border-t border-border bg-surface-2/40 px-4 py-2.5 text-xs leading-relaxed text-fg-subtle sm:px-5">
          {caption}
        </p>
      )}
    </figure>
  )
}
