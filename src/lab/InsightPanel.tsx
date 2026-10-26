import { useState } from 'react'
import { scaleLinear } from 'd3-scale'
import type { InsightMsg } from './protocol'

const VOWELS = new Set('aeiouyåäöAEIOUYÅÄÖéèàæ')
const kind = (ch: string): 'vowel' | 'consonant' | 'other' =>
  VOWELS.has(ch) ? 'vowel' : /\p{L}/u.test(ch) ? 'consonant' : 'other'
const COLORS = { vowel: 'var(--accent)', consonant: 'var(--fg-muted)', other: 'var(--brass-ink)' }
const show = (ch: string) => (ch === '\n' ? '↵' : ch === ' ' ? '␣' : ch)

function EmbeddingScatter({ points }: { points: InsightMsg['points'] }) {
  const W = 360
  const H = 300
  const P = 18
  const xs = points.map((p) => p.x)
  const ys = points.map((p) => p.y)
  const sx = scaleLinear().domain([Math.min(...xs), Math.max(...xs)]).range([P, W - P])
  const sy = scaleLinear().domain([Math.min(...ys), Math.max(...ys)]).range([H - P, P])
  return (
    <figure>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full border border-border bg-surface" role="img" aria-label="Character embeddings projected to 2D">
        {points.map((p) => (
          <text
            key={p.ch}
            x={sx(p.x)}
            y={sy(p.y)}
            textAnchor="middle"
            dy="0.35em"
            fontSize={13}
            fontFamily="var(--font-mono)"
            fontWeight={kind(p.ch) === 'vowel' ? 700 : 400}
            fill={COLORS[kind(p.ch)]}
          >
            {show(p.ch)}
          </text>
        ))}
      </svg>
      <figcaption className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-fg-muted">
        <span>
          <b style={{ color: COLORS.vowel }}>a e i</b> vowels
        </span>
        <span>
          <span style={{ color: COLORS.consonant }}>b c d</span> consonants
        </span>
        <span>
          <span style={{ color: COLORS.other }}>. , ␣</span> punctuation, spaces, digits
        </span>
      </figcaption>
    </figure>
  )
}

function AttentionHeatmap({ attn }: { attn: NonNullable<InsightMsg['attn']> }) {
  const [head, setHead] = useState(0)
  const h = Math.min(head, attn.heads.length - 1)
  const T = attn.tokens.length
  const cell = Math.max(10, Math.min(16, Math.floor(300 / T)))
  const L = 22
  const size = L + T * cell
  const m = attn.heads[h]
  return (
    <figure>
      <div className="mb-2 flex flex-wrap gap-1.5" role="tablist" aria-label="Attention head">
        {attn.heads.map((_, i) => (
          <button
            key={i}
            role="tab"
            aria-selected={i === h}
            onClick={() => setHead(i)}
            className={`border px-2 py-0.5 text-xs ${i === h ? 'holo-active border-transparent' : 'border-border bg-surface text-fg-muted hover:border-brass'}`}
          >
            Head {i + 1}
          </button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} role="img" aria-label="Attention heatmap: rows attend to columns">
          {attn.tokens.map((t, i) => (
            <g key={i}>
              <text x={L - 4} y={L + i * cell + cell / 2} dy="0.35em" textAnchor="end" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg-muted)">
                {show(t)}
              </text>
              <text x={L + i * cell + cell / 2} y={L - 6} textAnchor="middle" fontSize={11} fontFamily="var(--font-mono)" fill="var(--fg-muted)">
                {show(t)}
              </text>
            </g>
          ))}
          {attn.tokens.map((_, r) =>
            attn.tokens.map((__, c) => {
              const v = m[r * T + c]
              return (
                <rect
                  key={`${r}-${c}`}
                  x={L + c * cell}
                  y={L + r * cell}
                  width={cell - 1}
                  height={cell - 1}
                  fill={c > r ? 'var(--surface-2)' : 'var(--accent)'}
                  fillOpacity={c > r ? 0.5 : Math.max(0.04, Math.sqrt(v))}
                />
              )
            }),
          )}
        </svg>
      </div>
      <figcaption className="mt-2 text-xs text-fg-muted">
        Each row is a character looking back at earlier ones (it can never peek ahead). Stronger colour = more attention.
      </figcaption>
    </figure>
  )
}

export function InsightPanel({ insight }: { insight: InsightMsg | null }) {
  if (!insight) return <p className="text-sm text-fg-subtle italic">Train for a bit, then look inside.</p>
  return (
    <div className="grid gap-6 md:grid-cols-2">
      <div className="min-w-0">
        <p className="mb-1 font-display text-lg text-fg">Where it put each character</p>
        <p className="mb-3 text-xs text-fg-muted">Each character is a list of numbers. Squashed to 2D, similar ones drift together.</p>
        <EmbeddingScatter points={insight.points} />
      </div>
      <div className="min-w-0">
        <p className="mb-1 font-display text-lg text-fg">What it looks at</p>
        <p className="mb-3 text-xs text-fg-muted">Last layer’s attention for the end of your prompt.</p>
        {insight.attn ? (
          <AttentionHeatmap attn={insight.attn} />
        ) : (
          <p className="text-sm text-fg-subtle italic">Type a prompt of 2+ characters above.</p>
        )}
      </div>
    </div>
  )
}
