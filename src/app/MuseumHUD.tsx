import { CloseIcon } from './icons'

/** DOM heads-up display over the museum canvas (sibling of the Canvas). */
export function MuseumHUD({
  locked,
  hasEntered,
  focusOpen,
  title,
  exhibitLabel,
  onEnter,
  onExit,
}: {
  locked: boolean
  hasEntered: boolean
  focusOpen: boolean
  title: string
  exhibitLabel: string
  onEnter: () => void
  onExit: () => void
}) {
  const walking = locked && !focusOpen
  const paused = !locked && !focusOpen

  return (
    <div className="pointer-events-none fixed inset-0 z-40 select-none text-white">
      {/* exhibit label */}
      <div className="absolute top-5 left-5 z-10">
        <p className="font-mono text-[11px] tracking-wider text-white/45">EXHIBIT {exhibitLabel}</p>
        <p className="text-sm font-semibold">{title}</p>
      </div>

      {/* exit */}
      <button
        onClick={onExit}
        className="pointer-events-auto absolute top-5 right-5 z-10 flex items-center gap-1.5 rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-xs text-white/80 backdrop-blur transition-colors hover:bg-white/10"
      >
        <CloseIcon size={14} /> Exit to reading mode
      </button>

      {/* crosshair */}
      {walking && (
        <div className="absolute top-1/2 left-1/2 h-1.5 w-1.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-white/70 shadow-[0_0_6px_rgba(255,255,255,0.6)]" />
      )}

      {/* controls hint */}
      {walking && (
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 font-mono text-[11px] text-white/40">
          WASD move · mouse look · E to inspect · Esc to pause
        </div>
      )}

      {/* First entry: a prominent prompt. */}
      {paused && !hasEntered && (
        <button
          onClick={onEnter}
          className="pointer-events-auto absolute inset-0 grid place-items-center bg-black/45 backdrop-blur-sm"
        >
          <div className="rounded-2xl border border-white/15 bg-white/[0.07] px-8 py-7 text-center">
            <p className="text-lg font-semibold">Click to enter the exhibit</p>
            <p className="mt-2 text-sm text-white/60">
              <b className="font-mono text-white/80">WASD</b> move ·{' '}
              <b className="font-mono text-white/80">mouse</b> look ·{' '}
              <b className="font-mono text-white/80">E</b> inspect ·{' '}
              <b className="font-mono text-white/80">Esc</b> pause
            </p>
          </div>
        </button>
      )}

      {/* After entering: an unobtrusive resume — the live scene stays fully visible
          (no dimming/pause) so you can show it off; click anywhere to resume walking. */}
      {paused && hasEntered && (
        <button
          onClick={onEnter}
          aria-label="Click to resume walking"
          className="pointer-events-auto absolute inset-0 cursor-pointer bg-transparent"
        >
          <span className="absolute bottom-16 left-1/2 -translate-x-1/2 rounded-full border border-white/15 bg-black/40 px-4 py-1.5 text-xs text-white/80 backdrop-blur">
            click to resume walking
          </span>
        </button>
      )}
    </div>
  )
}
