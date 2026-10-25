import { useState } from 'react'
import { softmax, filterTopKTopP, sampleIndex } from '@/lib/softmax'
import { Slider } from '@/primitives/Slider'
import { BarChart, type BarDatum } from '@/primitives/BarChart'
import { MathBlock } from '@/primitives/MathBlock'
import { Button } from '@/primitives/Button'
import { colorForIndex } from '@/theme/tokens'
import { pct } from '@/lib/format'

const TOKENS = ['sunny', 'warm', 'nice', 'cloudy', 'beautiful', 'cold', 'perfect', 'rainy', 'going', 'grey']
const LOGITS = [3.1, 2.6, 2.2, 1.9, 1.7, 1.4, 1.1, 0.8, 0.3, -0.2]

export function SamplingPlayground() {
  const [temp, setTemp] = useState(0.9)
  const [topK, setTopK] = useState(10)
  const [topP, setTopP] = useState(1)
  const [picked, setPicked] = useState<number | null>(null)

  const base = softmax(LOGITS, temp)
  const { probs, excluded } = filterTopKTopP(base, { topK, topP })

  const bars: BarDatum[] = TOKENS.map((t, i) => ({
    label: t,
    value: probs[i],
    color: colorForIndex(i),
    dimmed: excluded[i],
  }))

  const onControl = (setter: (v: number) => void) => (v: number) => {
    setter(v)
    setPicked(null)
  }

  return (
    <div className="space-y-5">
      <p className="text-sm text-fg-muted">
        Next token after <span className="font-mono text-fg">“The weather today is”</span>
      </p>

      <div className="grid gap-4 sm:grid-cols-3">
        <Slider label="Temperature τ" value={temp} min={0.1} max={2} step={0.05} onChange={onControl(setTemp)} format={(v) => v.toFixed(2)} />
        <Slider label={`Top-k${topK >= TOKENS.length ? ' (off)' : ''}`} value={topK} min={1} max={TOKENS.length} step={1} onChange={onControl(setTopK)} />
        <Slider label={`Top-p${topP >= 1 ? ' (off)' : ''}`} value={topP} min={0.3} max={1} step={0.05} onChange={onControl(setTopP)} format={(v) => v.toFixed(2)} />
      </div>

      <BarChart data={bars} max={1} formatValue={(v) => pct(v, 0)} labelWidth="5rem" />

      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={() => setPicked(sampleIndex(probs, Math.random()))}>
          🎲 Sample a token
        </Button>
        {picked !== null && (
          <span className="text-sm text-fg">
            → picked{' '}
            <span className="font-mono font-semibold" style={{ color: 'var(--accent)' }}>
              {TOKENS[picked]}
            </span>
          </span>
        )}
      </div>

      <div className="overflow-x-auto">
        <MathBlock display className="!my-0">
          {String.raw`p_i = \frac{e^{\,z_i / \tau}}{\sum_j e^{\,z_j / \tau}} \qquad \tau = ` + temp.toFixed(2)}
        </MathBlock>
      </div>
    </div>
  )
}
