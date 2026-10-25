import { useState } from 'react'
import { dot } from '@/lib/vectorMath'
import { softmax } from '@/lib/softmax'
import { BarChart } from '@/primitives/BarChart'
import { colorForIndex } from '@/theme/tokens'
import { pct } from '@/lib/format'
import { cn } from '@/lib/cn'

const TOK = ['cat', 'sat', 'mat']
const Q = [
  [1, 0],
  [0, 1],
  [1, 1],
]
const K = [
  [1, 0],
  [0, 1],
  [1, 1],
]
const V = [
  [2, 0],
  [0, 3],
  [1, 1],
]

export function AttentionMath() {
  const [qi, setQi] = useState(0)
  const q = Q[qi]
  const scores = K.map((k) => dot(q, k))
  const weights = softmax(scores)
  const out = [0, 1].map((d) => weights.reduce((s, w, j) => s + w * V[j][d], 0))

  return (
    <div className="space-y-4">
      <p className="text-sm text-fg-muted">
        Three toy tokens, each with a 2-D key and value. Pick the{' '}
        <strong className="text-fg">query</strong> token and watch its output computed live.
      </p>

      <div className="flex gap-2">
        {TOK.map((t, i) => (
          <button
            key={t}
            onClick={() => setQi(i)}
            className={cn(
              'rounded-md px-3 py-1.5 font-mono text-sm transition-colors',
              i === qi ? 'text-white' : 'border border-border text-fg-muted hover:text-fg',
            )}
            style={i === qi ? { backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))' } : undefined}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="space-y-1.5 rounded-lg border border-border bg-surface-2/30 p-4 font-mono text-sm">
        <div className="text-fg">
          query <span style={{ color: 'var(--accent)' }}>Q = [{q.join(', ')}]</span>
        </div>
        <div className="text-fg-muted">
          1 · scores = Q·Kⱼ = [{scores.map((s) => s.toFixed(0)).join(', ')}]
        </div>
        <div className="text-fg-muted">2 · weights = softmax(scores) ↓</div>
      </div>

      <BarChart
        data={TOK.map((t, j) => ({ label: t, value: weights[j], color: colorForIndex(j) }))}
        max={1}
        formatValue={(v) => pct(v, 0)}
        labelWidth="3rem"
      />

      <div className="rounded-lg border border-border bg-surface-2/30 p-4 font-mono text-sm text-fg-muted">
        3 · output = Σ wⱼ·Vⱼ ={' '}
        <span style={{ color: 'var(--accent)' }}>[{out.map((o) => o.toFixed(2)).join(', ')}]</span>
      </div>

      <p className="text-xs leading-relaxed text-fg-subtle">
        Every number above is computed exactly in your browser. (For clarity we omit the 1/√d scaling
        that real models apply to the scores.)
      </p>
    </div>
  )
}
