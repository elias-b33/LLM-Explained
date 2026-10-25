import { useState } from 'react'
import { cn } from '@/lib/cn'
import { CloseIcon } from '@/app/icons'
import type { GalaxyData } from './types'

/*
 * The planetarium's console. The 3-D sky is always dark, so this chrome uses a
 * fixed "after hours" palette (walnut glass, brass edges, cream type) in both themes.
 */
const PANEL =
  'border border-[#b8924f]/45 bg-[#1a140e]/72 shadow-[inset_0_1px_0_rgba(255,236,200,0.12),0_12px_30px_-12px_rgba(0,0,0,0.8)] backdrop-blur-md'
const CHIP = 'rounded-full border px-3 py-1 text-xs transition-colors'
const CHIP_ON = 'border-[#e0b847]/80 bg-[#e0b847]/18 text-[#f6efe0]'
const CHIP_OFF = 'border-[#b8924f]/30 bg-[#1a140e]/55 text-[#d9c9ae] hover:border-[#e0b847]/60 hover:text-[#f6efe0]'

export function GalaxyOverlay({
  data,
  selected,
  onSelect,
  selectByName,
  activeUniverse,
  selectUniverse,
  onExit,
}: {
  data: GalaxyData
  selected: number | null
  onSelect: (i: number) => void
  selectByName: (name: string) => boolean
  activeUniverse: number | null
  selectUniverse: (i: number | null) => void
  onExit: () => void
}) {
  const [q, setQ] = useState('')
  const [notFound, setNotFound] = useState(false)
  const word = selected != null ? data.words[selected] : null

  const submitSearch = (e: React.FormEvent) => {
    e.preventDefault()
    if (!q.trim()) return
    setNotFound(!selectByName(q))
  }

  return (
    <div className="pointer-events-none absolute inset-0 text-[#f1e9da] select-none">
      {/* Title plaque */}
      <div className="absolute top-4 left-4 max-w-[calc(100%-7rem)] [text-shadow:0_1px_10px_rgba(0,0,0,0.85)] sm:top-5 sm:left-5">
        <p className="text-[10px] font-semibold tracking-[0.24em] text-[#d4b062] uppercase">The Planetarium</p>
        <p className="font-display text-2xl leading-tight text-[#f6efe0]">Word Universes</p>
        <p className="mt-0.5 hidden text-xs text-[#d9c9ae]/80 sm:block">
          {data.words.length.toLocaleString()} words · {data.clusters.map((c) => c.name).join(' · ')}
        </p>
      </div>

      {/* Universe switcher */}
      <div className="pointer-events-auto absolute top-[5.25rem] right-4 left-4 flex flex-wrap gap-1.5 sm:top-[6.25rem] sm:right-auto sm:left-5">
        <button
          onClick={() => selectUniverse(null)}
          className={cn(CHIP, 'backdrop-blur-sm', activeUniverse === null ? CHIP_ON : CHIP_OFF)}
        >
          All
        </button>
        {data.clusters.map((c, i) => (
          <button
            key={i}
            onClick={() => selectUniverse(i)}
            className={cn(CHIP, 'flex items-center gap-1.5 backdrop-blur-sm', activeUniverse === i ? CHIP_ON : CHIP_OFF)}
          >
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: c.color, boxShadow: `0 0 6px ${c.color}` }} />
            {c.name}
          </button>
        ))}
      </div>

      {/* Search */}
      <form
        onSubmit={submitSearch}
        className="pointer-events-auto absolute bottom-4 left-4 flex items-center gap-2 sm:top-5 sm:bottom-auto sm:left-1/2 sm:-translate-x-1/2"
      >
        <label htmlFor="galaxy-search" className="sr-only">
          Search a word
        </label>
        <input
          id="galaxy-search"
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setNotFound(false)
          }}
          placeholder="Search a word…"
          spellCheck={false}
          className={cn(
            PANEL,
            'w-56 rounded-sm px-3 py-1.5 text-sm text-[#f6efe0] placeholder-[#d9c9ae]/55 outline-none focus:border-[#e0b847]',
          )}
        />
        {notFound && <span className="text-xs text-[#d9c9ae]/80">not found</span>}
      </form>

      {/* Exit */}
      <button
        onClick={onExit}
        className={cn(
          PANEL,
          'pointer-events-auto absolute top-4 right-4 flex items-center gap-1.5 rounded-sm px-3 py-1.5 text-[11px] font-semibold tracking-[0.16em] text-[#f1e9da] uppercase transition-colors hover:border-[#e0b847] sm:top-5 sm:right-5',
        )}
      >
        <CloseIcon size={14} /> Exit
      </button>

      {/* Hint */}
      <div className="absolute bottom-5 left-5 hidden text-xs text-[#d9c9ae]/65 sm:block">
        WASD + Q/E to fly · drag to look · scroll to zoom · hover to preview · click to pick
      </div>

      {/* Info panel — the specimen label for the selected word */}
      {word && (
        <div className={cn(PANEL, 'pointer-events-auto absolute right-4 bottom-16 w-60 rounded-sm p-4 sm:right-5 sm:bottom-5')}>
          <p className="text-[10px] font-semibold tracking-[0.22em] uppercase" style={{ color: data.clusters[word.c]?.color }}>
            {data.clusters[word.c]?.name}
          </p>
          <p className="mt-0.5 font-display text-2xl leading-tight text-[#f6efe0]" dir="auto">
            {word.w}
          </p>
          <div className="my-3 h-px bg-gradient-to-r from-[#b8924f]/70 to-transparent" />
          <p className="text-[10px] font-semibold tracking-[0.2em] text-[#d4b062] uppercase">Nearest words</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {word.nn.map((n) => (
              <button
                key={n}
                onClick={() => onSelect(n)}
                dir="auto"
                className="rounded-sm border border-[#b8924f]/30 bg-white/5 px-2 py-1 text-xs text-[#f1e9da] transition-colors hover:border-[#e0b847]/70 hover:bg-white/10"
              >
                {data.words[n].w}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
