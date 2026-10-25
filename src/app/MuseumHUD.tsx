import { CloseIcon } from './icons'
import type { Quality } from '@/three/museum/quality'

function SpeakerIcon({ muted }: { muted: boolean }) {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M11 5 6 9H2v6h4l5 4V5z" />
      {muted ? (
        <path d="m23 9-6 6M17 9l6 6" />
      ) : (
        <path d="M15.5 8.5a5 5 0 0 1 0 7M19 5a10 10 0 0 1 0 14" />
      )}
    </svg>
  )
}

/** DOM heads-up display over the museum canvas (sibling of the Canvas). */
export function MuseumHUD({
  locked,
  hasEntered,
  focusOpen,
  title,
  exhibitLabel,
  onEnter,
  onExit,
  prompt,
  explored,
  tourTotal,
  muted,
  onToggleMute,
  quality,
  onQuality,
  softwareRendering,
}: {
  locked: boolean
  hasEntered: boolean
  focusOpen: boolean
  title: string
  exhibitLabel: string
  onEnter: () => void
  onExit: () => void
  prompt: string | null
  explored: number
  tourTotal: number
  muted: boolean
  onToggleMute: () => void
  quality: Quality
  onQuality: (q: Quality) => void
  softwareRendering: boolean
}) {
  const walking = locked && !focusOpen
  const paused = !locked && !focusOpen
  // Pointer lock is unavailable in automated dev previews; show the prompt there too.
  const showPrompt = !!prompt && !focusOpen && (walking || import.meta.env.DEV)
  // Dark warm glass over the 3D view: walnut tint, brass hairline, brass-ink labels.
  const glass = 'border border-[#c9a227]/30 bg-[#1a140e]/72 backdrop-blur-md shadow-[0_8px_24px_-12px_rgba(0,0,0,0.7)]'
  const chip = `pointer-events-auto flex items-center gap-1.5 rounded-[3px] px-3 py-1.5 text-[11px] font-semibold tracking-[0.14em] uppercase text-[#f1e9da]/85 transition-colors hover:border-[#c9a227]/70 hover:text-[#f6efe0] ${glass}`

  return (
    <div className="dark pointer-events-none fixed inset-0 z-40 select-none text-[#f1e9da]">
      {/* exhibit label + explored tracker */}
      <div className={`absolute top-5 left-5 z-10 rounded-[3px] px-3.5 py-2.5 ${glass}`}>
        <p className="exhibit-label">Exhibit {exhibitLabel}</p>
        <p className="font-display text-lg leading-tight text-[#f6efe0]">{title}</p>
        {tourTotal > 0 && (
          <div className="mt-1.5 flex items-center gap-2" aria-label={`Exhibits explored ${explored} of ${tourTotal}`}>
            <span className="flex gap-1">
              {Array.from({ length: tourTotal }, (_, i) => (
                <span
                  key={i}
                  className="h-1.5 w-1.5 rotate-45 transition-colors duration-500"
                  style={{
                    background: i < explored ? '#e0b847' : 'transparent',
                    border: `1px solid ${i < explored ? '#e0b847' : 'rgba(212,176,98,0.45)'}`,
                    boxShadow: i < explored ? '0 0 6px rgba(224,184,71,0.7)' : 'none',
                  }}
                />
              ))}
            </span>
            <span className="text-[11px] tracking-wide text-[#c9b9a0]">
              Exhibits explored {explored}/{tourTotal}
            </span>
          </div>
        )}
      </div>

      {/* top-right controls */}
      <div className="absolute top-5 right-5 z-10 flex items-center gap-2">
        <div
          className={`pointer-events-auto flex overflow-hidden rounded-[3px] text-[11px] font-semibold tracking-[0.14em] uppercase ${glass}`}
          role="group"
          aria-label="Graphics quality"
        >
          {(['standard', 'high'] as const).map((q) => (
            <button
              key={q}
              onClick={(e) => {
                e.stopPropagation() // don't let drei's document click grab pointer lock
                onQuality(q)
              }}
              aria-pressed={quality === q}
              className={`px-2.5 py-1.5 transition-colors ${
                quality === q ? 'bg-[#d4b062] text-[#1a140e]' : 'text-[#c9b9a0] hover:text-[#f6efe0]'
              }`}
            >
              {q === 'standard' ? 'Standard' : 'High'}
            </button>
          ))}
        </div>
        <button
          onClick={(e) => {
            e.stopPropagation()
            onToggleMute()
          }}
          className={chip}
          aria-label={muted ? 'Unmute sound' : 'Mute sound'}
          aria-pressed={muted}
        >
          <SpeakerIcon muted={muted} />
        </button>
        <button onClick={onExit} className={chip}>
          <CloseIcon size={13} /> Exit to reading mode
        </button>
      </div>

      {softwareRendering && (
        <p className={`absolute bottom-5 left-5 z-10 max-w-[46ch] rounded-[3px] px-3 py-1.5 text-[11px] text-[#e8b04a] ${glass}`}>
          Hardware acceleration is off in your browser, so the museum will be slow.
        </p>
      )}

      {/* crosshair: a small warm dot that opens into a brass ring on a target */}
      {walking && (
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-150"
          style={
            prompt
              ? { width: 16, height: 16, border: '1.5px solid #e0b847', boxShadow: '0 0 8px rgba(224,184,71,0.55)' }
              : { width: 5, height: 5, background: 'rgba(246,239,224,0.8)', boxShadow: '0 0 5px rgba(0,0,0,0.6)' }
          }
        />
      )}

      {/* interaction prompt */}
      {showPrompt && (
        <div
          className={`absolute top-1/2 left-1/2 mt-8 -translate-x-1/2 rounded-[3px] px-3 py-1 text-[13px] whitespace-nowrap text-[#f1e9da] ${glass}`}
        >
          <b className="mr-2 rounded-[2px] bg-[#d4b062] px-1.5 py-px font-mono text-[11px] text-[#1a140e]">E</b>
          {prompt}
        </div>
      )}

      {/* controls hint */}
      {walking && (
        <div className="absolute bottom-5 left-1/2 -translate-x-1/2 text-[11px] tracking-wide text-[#f1e9da]/45 [text-shadow:0_1px_3px_#000]">
          WASD move · mouse look · E or click to interact · Esc to pause
        </div>
      )}

      {/* First entry: a prominent, museum-ticket style prompt. */}
      {paused && !hasEntered && (
        <button
          onClick={onEnter}
          className="pointer-events-auto absolute inset-0 grid place-items-center bg-[#140f0a]/55 backdrop-blur-sm"
        >
          <div className={`max-w-sm rounded-[3px] px-9 py-8 text-center ${glass}`}>
            <p className="exhibit-label">Exhibit {exhibitLabel}</p>
            <p className="font-display mt-1 text-3xl leading-tight text-[#f6efe0]">{title}</p>
            <div className="mx-auto my-4 h-px w-24 bg-gradient-to-r from-transparent via-[#c9a227] to-transparent" />
            <span className="btn-museum">Click to enter</span>
            <p className="mt-4 text-[12px] text-[#c9b9a0]">
              <b className="font-mono text-[#f1e9da]">WASD</b> move · <b className="font-mono text-[#f1e9da]">mouse</b> look ·{' '}
              <b className="font-mono text-[#f1e9da]">E</b> interact · <b className="font-mono text-[#f1e9da]">Esc</b> pause
            </p>
          </div>
        </button>
      )}

      {/* After entering: an unobtrusive resume. The live scene stays fully visible
          (no dimming/pause) so you can show it off; click anywhere to resume walking. */}
      {paused && hasEntered && (
        <button
          onClick={onEnter}
          aria-label="Click to resume walking"
          className="pointer-events-auto absolute inset-0 cursor-pointer bg-transparent"
        >
          <span
            className={`absolute bottom-16 left-1/2 -translate-x-1/2 rounded-[3px] px-4 py-1.5 text-[11px] font-semibold tracking-[0.14em] uppercase text-[#f1e9da]/85 ${glass}`}
          >
            Click to resume walking
          </span>
        </button>
      )}
    </div>
  )
}
