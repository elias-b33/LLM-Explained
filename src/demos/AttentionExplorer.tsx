import { useState } from 'react'
import { ATTENTION_SENTENCE as S, ATTENTION_HEADS } from '@/data/attentionPatterns'
import { Button } from '@/primitives/Button'
import { cn } from '@/lib/cn'
import { mix } from '@/lib/color'

export function AttentionExplorer() {
  const [head, setHead] = useState(0)
  const [q, setQ] = useState(5) // default: "it"
  const W = ATTENTION_HEADS[head].weights

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {ATTENTION_HEADS.map((h, i) => (
          <Button
            key={h.name}
            variant={i === head ? 'secondary' : 'ghost'}
            className="!px-2.5 !py-1 text-xs"
            onClick={() => setHead(i)}
          >
            {h.name}
          </Button>
        ))}
      </div>

      <p className="text-xs text-fg-muted">{ATTENTION_HEADS[head].desc}</p>

      <div>
        <p className="mb-1.5 text-xs text-fg-subtle">Pick the token doing the looking (the query):</p>
        <div className="flex flex-wrap gap-1.5">
          {S.map((w, i) => (
            <button
              key={i}
              onClick={() => setQ(i)}
              className={cn(
                'rounded-md px-2 py-1 font-mono text-sm transition-colors',
                i === q ? 'text-white' : 'border border-border text-fg-muted hover:text-fg',
              )}
              style={i === q ? { backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))' } : undefined}
            >
              {w}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-lg border border-border bg-surface-2/30 p-4">
        <p className="mb-2.5 text-sm text-fg">
          <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
            {S[q]}
          </span>{' '}
          attends to:
        </p>
        <div className="flex flex-wrap gap-1.5">
          {S.map((w, j) => (
            <span
              key={j}
              className="rounded-md px-2 py-1 font-mono text-sm text-fg"
              style={{ backgroundColor: mix('var(--accent)', Math.round(W[q][j] * 100)), border: `1px solid ${mix('var(--accent)', 28)}` }}
              title={`${(W[q][j] * 100).toFixed(0)}%`}
            >
              {w}
              <span className="ml-1 text-[10px] text-fg-subtle">{(W[q][j] * 100).toFixed(0)}</span>
            </span>
          ))}
        </div>
      </div>

      <p className="text-xs leading-relaxed text-fg-subtle">
        Brighter = more attention. Try the <span className="font-mono">it</span> token with the
        Coreference head — it reaches back to <span className="font-mono">cat</span>.
      </p>
    </div>
  )
}
