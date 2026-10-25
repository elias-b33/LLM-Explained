import { useState } from 'react'
import { NEXT_TOKEN_EXAMPLES } from '@/data/nextTokenExamples'
import { BarChart, type BarDatum } from '@/primitives/BarChart'
import { Button } from '@/primitives/Button'
import { pct } from '@/lib/format'

export function NextTokenPredictor() {
  const [idx, setIdx] = useState(0)
  const ex = NEXT_TOKEN_EXAMPLES[idx]

  const bars: BarDatum[] = ex.tokens.map((t, i) => ({
    label: t.token.replace(/ /g, '·'),
    value: t.prob,
    color: i === 0 ? 'var(--accent)' : 'var(--fg-subtle)',
  }))

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {NEXT_TOKEN_EXAMPLES.map((e, i) => (
          <Button
            key={i}
            variant={i === idx ? 'secondary' : 'ghost'}
            className="!px-2.5 !py-1 font-mono text-xs"
            onClick={() => setIdx(i)}
          >
            {e.prompt}
          </Button>
        ))}
      </div>

      <div className="rounded-lg border border-border bg-surface-2/30 p-4">
        <p className="mb-3 font-mono text-sm text-fg">
          {ex.prompt}
          <span
            className="ml-0.5 inline-block h-4 w-[2px] animate-pulse align-middle"
            style={{ background: 'var(--accent)' }}
          />
        </p>
        <BarChart data={bars} max={1} formatValue={(v) => pct(v, 0)} labelWidth="4.5rem" />
      </div>

      <p className="text-xs leading-relaxed text-fg-subtle">
        The model assigns a probability to <em>every</em> token in its vocabulary — these are just its
        top guesses, with the most likely highlighted.
      </p>
    </div>
  )
}
