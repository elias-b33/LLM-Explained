import { motion } from 'motion/react'
import { springSoft } from '@/theme/tokens'
import { cn } from '@/lib/cn'

export interface BarDatum {
  label: string
  value: number
  color?: string
  dimmed?: boolean
}

/** Animated horizontal bar chart for probability/score distributions. */
export function BarChart({
  data,
  max,
  formatValue,
  labelWidth = '6rem',
  className,
}: {
  data: BarDatum[]
  max?: number
  formatValue?: (v: number) => string
  labelWidth?: string
  className?: string
}) {
  const m = max ?? Math.max(...data.map((d) => d.value), 1e-9)
  return (
    <div className={cn('space-y-1.5', className)}>
      {data.map((d, i) => (
        <div
          key={`${d.label}-${i}`}
          className={cn('flex items-center gap-2.5 transition-opacity', d.dimmed && 'opacity-35')}
        >
          <span
            className="shrink-0 truncate text-right font-mono text-xs text-fg-muted"
            style={{ width: labelWidth }}
          >
            {d.label}
          </span>
          <div className="relative h-6 flex-1 overflow-hidden rounded-md bg-surface-2">
            <motion.div
              className="absolute inset-y-0 left-0 rounded-md"
              style={{ backgroundColor: d.color ?? 'var(--accent)' }}
              initial={false}
              animate={{ width: `${Math.max(0, Math.min(1, d.value / m)) * 100}%` }}
              transition={springSoft}
            />
          </div>
          <span className="w-12 shrink-0 text-right font-mono text-xs tabular-nums text-fg">
            {formatValue ? formatValue(d.value) : d.value.toFixed(2)}
          </span>
        </div>
      ))}
    </div>
  )
}
