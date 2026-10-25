import { useState } from 'react'
import { cn } from '@/lib/cn'
import { CloseIcon } from '@/app/icons'
import type { GalaxyData } from './types'

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
    <div className="pointer-events-none absolute inset-0 select-none">
      {/* Title */}
      <div className="absolute top-5 left-5">
        <p className="text-sm font-semibold text-white">Word Universes</p>
        <p className="mt-0.5 text-xs text-white/45">
          {data.words.length.toLocaleString()} words · {data.clusters.map((c) => c.name).join(' · ')}
        </p>
      </div>

      {/* Universe switcher */}
      <div className="pointer-events-auto absolute top-[3.75rem] left-5 flex flex-wrap gap-1.5">
        <button
          onClick={() => selectUniverse(null)}
          className={cn(
            'rounded-full border px-3 py-1 text-xs backdrop-blur transition-colors',
            activeUniverse === null ? 'border-white/30 bg-white/15 text-white' : 'border-white/10 bg-white/5 text-white/60 hover:text-white',
          )}
        >
          All
        </button>
        {data.clusters.map((c, i) => (
          <button
            key={i}
            onClick={() => selectUniverse(i)}
            className={cn(
              'flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs backdrop-blur transition-colors',
              activeUniverse === i ? 'border-white/30 bg-white/15 text-white' : 'border-white/10 bg-white/5 text-white/60 hover:text-white',
            )}
          >
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: c.color }} />
            {c.name}
          </button>
        ))}
      </div>

      {/* Search */}
      <form onSubmit={submitSearch} className="pointer-events-auto absolute top-5 left-1/2 -translate-x-1/2">
        <input
          value={q}
          onChange={(e) => {
            setQ(e.target.value)
            setNotFound(false)
          }}
          placeholder="Search a word…"
          spellCheck={false}
          className="w-56 rounded-lg border border-white/15 bg-black/40 px-3 py-1.5 text-sm text-white placeholder-white/35 backdrop-blur outline-none focus:border-white/35"
        />
        {notFound && <span className="ml-2 align-middle text-xs text-white/50">not found</span>}
      </form>

      {/* Exit */}
      <button
        onClick={onExit}
        className="pointer-events-auto absolute top-5 right-5 flex items-center gap-1.5 rounded-lg border border-white/15 bg-white/5 px-3 py-1.5 text-xs text-white/80 backdrop-blur transition-colors hover:bg-white/10"
      >
        <CloseIcon size={14} /> Exit
      </button>

      {/* Hint */}
      <div className="absolute bottom-5 left-5 text-xs text-white/40">
        WASD + Q/E to fly · drag to look · scroll to zoom · hover to preview · click to pick
      </div>

      {/* Info panel */}
      {word && (
        <div className="pointer-events-auto absolute right-5 bottom-5 w-60 rounded-xl border border-white/12 bg-black/50 p-4 backdrop-blur">
          <p className="text-lg font-semibold text-white" dir="auto">
            {word.w}
          </p>
          <p className="mt-0.5 text-xs" style={{ color: data.clusters[word.c]?.color }}>
            {data.clusters[word.c]?.name}
          </p>
          <p className="mt-3 text-[11px] tracking-wider text-white/40 uppercase">Nearest words</p>
          <div className="mt-1.5 flex flex-wrap gap-1.5">
            {word.nn.map((n) => (
              <button
                key={n}
                onClick={() => onSelect(n)}
                dir="auto"
                className="rounded-md bg-white/10 px-2 py-1 text-xs text-white/80 transition-colors hover:bg-white/20"
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
