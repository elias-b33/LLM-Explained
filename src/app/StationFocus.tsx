import { lazy, Suspense, useEffect } from 'react'
import { CloseIcon } from './icons'
import type { FocusTarget } from '@/three/museum/halls'

const DemoBody = lazy(() => import('./museumDemos').then((m) => ({ default: m.DemoBody })))

/**
 * The focused 2D overlay that opens when you reach a station and press E. Renders
 * the real interactive demo / math in a comfortable reading surface over a dimmed
 * backdrop. Closing returns you to walking.
 */
export function StationFocus({ target, onClose }: { target: FocusTarget; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation()
        onClose()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div
      className="dark fixed inset-0 z-50 grid place-items-center bg-black/70 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative max-h-[88vh] w-full max-w-2xl overflow-auto rounded-2xl border border-border bg-bg p-5 shadow-2xl sm:p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-4 flex items-start justify-between gap-3">
          <div>
            <p className="font-mono text-[11px] tracking-wider text-fg-subtle uppercase">
              Interactive exhibit
            </p>
            <h2 className="text-lg font-semibold text-fg">{target.title}</h2>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-lg border border-border p-2 text-fg-muted transition-colors hover:text-fg"
          >
            <CloseIcon size={16} />
          </button>
        </div>
        <Suspense fallback={<div className="py-12 text-center text-sm text-fg-muted">Loading…</div>}>
          <DemoBody target={target} />
        </Suspense>
        <p className="mt-5 text-center text-xs text-fg-subtle">
          Press Esc or click outside to return to the exhibit
        </p>
      </div>
    </div>
  )
}
