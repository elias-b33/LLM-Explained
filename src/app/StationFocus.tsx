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
      className="dark fixed inset-0 z-50 grid place-items-center bg-[#140f0a]/75 p-4 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="relative max-h-[88vh] w-full max-w-2xl overflow-auto rounded-[3px] border border-[#c9a227]/35 bg-bg p-5 shadow-[0_24px_60px_-20px_rgba(0,0,0,0.85),inset_0_1px_0_rgba(255,236,200,0.06)] sm:p-7"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-5 flex items-start justify-between gap-3 border-b border-[#c9a227]/25 pb-4">
          <div>
            <p className="exhibit-label">Interactive exhibit</p>
            <h2 className="font-display mt-0.5 text-2xl leading-tight text-fg">{target.title}</h2>
          </div>
          <button
            onClick={onClose}
            aria-label="Close"
            className="rounded-[3px] border border-brass/60 p-2 text-fg-muted transition-colors hover:border-gold hover:text-fg"
          >
            <CloseIcon size={16} />
          </button>
        </div>
        <Suspense fallback={<p className="exhibit-label py-12 text-center">Loading…</p>}>
          <DemoBody target={target} />
        </Suspense>
        <p className="mt-6 text-center text-[11px] tracking-wide text-fg-subtle">
          Press Esc or click outside to return to the exhibit
        </p>
      </div>
    </div>
  )
}
