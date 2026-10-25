import type { ReactNode } from 'react'
import { AlertIcon, EyeIcon, InfoIcon, LightbulbIcon } from '@/app/icons'
import { mix } from '@/lib/color'

type CalloutVariant = 'info' | 'insight' | 'warning' | 'honesty'

const CONFIG: Record<
  CalloutVariant,
  { color: string; Icon: typeof InfoIcon; label: string; defaultTitle?: string }
> = {
  info: { color: 'var(--accent-2)', Icon: InfoIcon, label: 'Note' },
  insight: { color: 'var(--accent)', Icon: LightbulbIcon, label: 'Insight' },
  warning: { color: 'var(--warning)', Icon: AlertIcon, label: 'Caution' },
  honesty: { color: 'var(--brass-ink)', Icon: EyeIcon, label: 'Curator’s note', defaultTitle: 'What’s real here' },
}

/** A wall-text aside: parchment card, brass edge, a small round seal for the icon. */
export function Callout({
  variant = 'info',
  title,
  children,
}: {
  variant?: CalloutVariant
  title?: string
  children: ReactNode
}) {
  const { color, Icon, label, defaultTitle } = CONFIG[variant]
  const heading = title ?? defaultTitle
  return (
    <aside
      className="relative flex gap-3.5 border border-border p-4 pl-5 shadow-[0_12px_28px_-22px_rgb(var(--shadow-color)/0.5)]"
      style={{ backgroundColor: `color-mix(in srgb, ${color} 5%, var(--surface))` }}
    >
      <span
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ backgroundImage: `linear-gradient(${color}, ${mix(color, 40)})` }}
        aria-hidden="true"
      />
      <span
        className="mt-px flex h-8 w-8 shrink-0 items-center justify-center rounded-full border"
        style={{ color, borderColor: mix(color, 45), backgroundColor: mix(color, 8) }}
      >
        <Icon size={16} />
      </span>
      <div className="min-w-0 text-sm leading-relaxed text-fg-muted">
        <p className="mb-0.5 text-[10px] font-semibold tracking-[0.2em] uppercase" style={{ color }}>
          {label}
        </p>
        {heading && <p className="mb-1 font-display text-lg leading-snug text-fg">{heading}</p>}
        {children}
      </div>
    </aside>
  )
}
