import type { ReactNode } from 'react'
import { AlertIcon, EyeIcon, InfoIcon, LightbulbIcon } from '@/app/icons'
import { mix } from '@/lib/color'

type CalloutVariant = 'info' | 'insight' | 'warning' | 'honesty'

const CONFIG: Record<CalloutVariant, { color: string; Icon: typeof InfoIcon; defaultTitle?: string }> = {
  info: { color: 'var(--accent-2)', Icon: InfoIcon },
  insight: { color: 'var(--accent)', Icon: LightbulbIcon },
  warning: { color: 'var(--warning)', Icon: AlertIcon },
  honesty: { color: 'var(--accent-2)', Icon: EyeIcon, defaultTitle: 'What’s real here' },
}

export function Callout({
  variant = 'info',
  title,
  children,
}: {
  variant?: CalloutVariant
  title?: string
  children: ReactNode
}) {
  const { color, Icon, defaultTitle } = CONFIG[variant]
  const heading = title ?? defaultTitle
  return (
    <div
      className="relative flex gap-3.5 overflow-hidden rounded-xl border p-4 pl-5"
      style={{
        borderColor: mix(color, 30),
        backgroundImage: `linear-gradient(100deg, ${mix(color, 14)}, ${mix(color, 4)} 70%)`,
      }}
    >
      <span
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: color, boxShadow: `0 0 14px ${color}` }}
        aria-hidden="true"
      />
      <span
        className="mt-px flex h-8 w-8 shrink-0 items-center justify-center rounded-lg"
        style={{ color, backgroundColor: mix(color, 16), boxShadow: `0 0 18px -4px ${color}` }}
      >
        <Icon size={17} />
      </span>
      <div className="text-sm leading-relaxed" style={{ color: 'var(--fg-muted)' }}>
        {heading && (
          <p className="mb-1 font-semibold" style={{ color: 'var(--fg)' }}>
            {heading}
          </p>
        )}
        {children}
      </div>
    </div>
  )
}
