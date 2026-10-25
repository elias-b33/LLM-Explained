import { useId, type CSSProperties, type ReactNode } from 'react'

/** Labeled range input with a live value readout — a brass rail with a glowing exhibit knob. */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  onChange,
  format,
  accent = 'var(--accent)',
}: {
  label: ReactNode
  value: number
  min: number
  max: number
  step?: number
  onChange: (v: number) => void
  format?: (v: number) => string
  accent?: string
}) {
  const id = useId()
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0
  return (
    <div>
      <div className="mb-2 flex items-baseline justify-between gap-2 text-sm">
        <label htmlFor={id} className="text-fg-muted">
          {label}
        </label>
        <span className="border border-border bg-surface px-1.5 py-px font-mono text-[13px] tabular-nums text-fg">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="range-museum w-full cursor-pointer"
        style={{ '--pct': `${pct}%`, '--knob': accent } as CSSProperties}
      />
    </div>
  )
}
