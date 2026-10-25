import type { ReactNode } from 'react'
import { ArrowLeftIcon, ArrowRightIcon } from '@/app/icons'

const BTN =
  'flex h-9 w-9 items-center justify-center rounded-full border border-border-strong bg-surface text-fg shadow-[inset_0_1px_0_var(--glass-edge)] transition-colors enabled:hover:border-brass enabled:hover:text-brass-ink disabled:cursor-not-allowed disabled:opacity-30'

/** Prev / next control for step-through demos, styled like an audio-guide keypad. */
export function Stepper({
  step,
  count,
  onPrev,
  onNext,
  label,
}: {
  step: number
  count: number
  onPrev: () => void
  onNext: () => void
  label?: ReactNode
}) {
  return (
    <div className="flex items-center justify-between gap-3">
      <button className={BTN} onClick={onPrev} disabled={step <= 0} aria-label="Previous step">
        <ArrowLeftIcon size={16} />
      </button>
      <div className="text-center text-xs text-fg-muted" aria-live="polite">
        {label ?? (
          <span className="inline-flex items-center gap-2">
            <span className="text-[10px] font-semibold tracking-[0.2em] uppercase">Stop</span>
            <span className="font-display text-lg leading-none text-fg">{step + 1}</span>
            <span className="text-fg-subtle">of {count}</span>
          </span>
        )}
      </div>
      <button className={BTN} onClick={onNext} disabled={step >= count - 1} aria-label="Next step">
        <ArrowRightIcon size={16} />
      </button>
    </div>
  )
}
