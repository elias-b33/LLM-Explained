import { useState } from 'react'
import { scaleLinear } from 'd3-scale'
import {
  EMBEDDING_WORDS,
  ANALOGIES,
  GROUP_COLORS,
  type WordPoint,
} from '@/data/embeddings'
import { Button } from '@/primitives/Button'

const W = 560
const H = 360
const PAD = 28

function nearest(word: WordPoint, all: WordPoint[], k = 3): WordPoint[] {
  return all
    .filter((w) => w.word !== word.word)
    .map((w) => ({ w, d: Math.hypot(w.x - word.x, w.y - word.y) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k)
    .map((o) => o.w)
}

export function EmbeddingMap() {
  const [mode, setMode] = useState<'explore' | number>('explore')
  const [hover, setHover] = useState<string | null>(null)

  const sx = scaleLinear().domain([0, 1]).range([PAD, W - PAD])
  const sy = scaleLinear().domain([0, 1]).range([PAD, H - PAD])
  const byWord = Object.fromEntries(EMBEDDING_WORDS.map((w) => [w.word, w] as const))

  const hoverWord = hover ? byWord[hover] : null
  const neighbors = hoverWord && mode === 'explore' ? nearest(hoverWord, EMBEDDING_WORDS) : []
  const analogy = typeof mode === 'number' ? ANALOGIES[mode] : null

  const highlight = new Set<string>()
  if (analogy) [analogy.a, analogy.b, analogy.c, analogy.d].forEach((w) => highlight.add(w))
  if (hoverWord) {
    highlight.add(hoverWord.word)
    neighbors.forEach((n) => highlight.add(n.word))
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <Button variant={mode === 'explore' ? 'secondary' : 'ghost'} className="!px-2.5 !py-1 text-xs" onClick={() => setMode('explore')}>
          Explore
        </Button>
        {ANALOGIES.map((a, i) => (
          <Button key={i} variant={mode === i ? 'secondary' : 'ghost'} className="!px-2.5 !py-1 font-mono text-xs" onClick={() => setMode(i)}>
            {a.a} − {a.b} + {a.c}
          </Button>
        ))}
      </div>

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-sm border border-border bg-surface">
        <defs>
          {/* faint surveyor's grid, like a printed chart in a gallery guide */}
          <pattern id="emb-grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M40 0H0V40" fill="none" stroke="var(--brass)" strokeOpacity={0.14} strokeWidth={1} />
          </pattern>
          <marker id="emb-arrow" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent)" />
          </marker>
          <marker id="emb-arrow2" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="var(--accent-2)" />
          </marker>
        </defs>
        <rect width={W} height={H} fill="url(#emb-grid)" />

        {hoverWord &&
          neighbors.map((n) => (
            <line key={n.word} x1={sx(hoverWord.x)} y1={sy(hoverWord.y)} x2={sx(n.x)} y2={sy(n.y)} stroke="var(--accent)" strokeWidth={1} strokeOpacity={0.4} />
          ))}

        {analogy && (
          <g>
            <line x1={sx(byWord[analogy.b].x)} y1={sy(byWord[analogy.b].y)} x2={sx(byWord[analogy.a].x)} y2={sy(byWord[analogy.a].y)} stroke="var(--accent)" strokeWidth={2} markerEnd="url(#emb-arrow)" />
            <line x1={sx(byWord[analogy.c].x)} y1={sy(byWord[analogy.c].y)} x2={sx(byWord[analogy.d].x)} y2={sy(byWord[analogy.d].y)} stroke="var(--accent-2)" strokeWidth={2} strokeDasharray="4 3" markerEnd="url(#emb-arrow2)" />
          </g>
        )}

        {EMBEDDING_WORDS.map((w) => {
          const active = highlight.size === 0 || highlight.has(w.word)
          return (
            <g
              key={w.word}
              onMouseEnter={() => setHover(w.word)}
              onMouseLeave={() => setHover(null)}
              style={{ cursor: 'pointer', opacity: active ? 1 : 0.2 }}
            >
              <circle cx={sx(w.x)} cy={sy(w.y)} r={4.5} fill={GROUP_COLORS[w.group] ?? 'var(--accent)'} />
              <text x={sx(w.x) + 7} y={sy(w.y) + 3.5} fontSize={11} fill="var(--fg-muted)" fontFamily="var(--font-mono)">
                {w.word}
              </text>
            </g>
          )
        })}
      </svg>

      {analogy ? (
        <p className="text-sm leading-relaxed text-fg">
          <span className="font-mono">
            {analogy.a} − {analogy.b} + {analogy.c}
          </span>{' '}
          lands right on top of{' '}
          <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
            {analogy.d}
          </span>
          . The relationship is a consistent <em>direction</em> in the space.
        </p>
      ) : (
        <p className="text-xs text-fg-subtle">
          Hover a word to see its nearest neighbors — notice how related words cluster together.
        </p>
      )}
    </div>
  )
}
