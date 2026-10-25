import type { ReactNode } from 'react'
import { ArrowLeftIcon, ArrowRightIcon } from '@/app/icons'

const BTN =
  'flex h-9 w-9 items-center justify-center rounded-lg border border-border bg-surface-2/60 text-fg transition-colors enabled:hover:border-border-strong enabled:hover:bg-surface-2 disabled:cursor-not-allowed disabled:opacity-30'

/** Prev / next control for step-through demos. */
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
        <ArrowLeftIcon size={17} />
      </button>
      <div className="text-center text-xs text-fg-muted">
        {label ?? (
          <>
            Step <span className="font-mono text-fg">{step + 1}</span> of {count}
          </>
        )}
      </div>
      <button
        className={BTN}
        onClick={onNext}
        disabled={step >= count - 1}
        aria-label="Next step"
      >
        <ArrowRightIcon size={17} />
      </button>
    </div>
  )
}
