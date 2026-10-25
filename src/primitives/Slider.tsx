import type { ReactNode } from 'react'

/** Labeled range input with a live value readout. */
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
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2 text-sm">
        <label className="text-fg-muted">{label}</label>
        <span className="font-mono tabular-nums text-fg">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="h-2 w-full cursor-pointer"
        style={{ accentColor: accent }}
      />
    </div>
  )
}
