import { useState } from 'react'
import { motion } from 'motion/react'
import { scaleLinear } from 'd3-scale'
import { Slider } from '@/primitives/Slider'
import { Button } from '@/primitives/Button'

const W = 520
const H = 230
const PAD = 24
const X_MIN = -3
const X_MAX = 3
const Y_MAX = 9
const START = 2.6
const f = (x: number) => x * x
const fp = (x: number) => 2 * x
const clamp = (x: number) => Math.max(X_MIN, Math.min(X_MAX, x))

export function GradientDescent() {
  const [lr, setLr] = useState(0.1)
  const [x, setX] = useState(START)
  const [steps, setSteps] = useState(0)

  const sx = scaleLinear().domain([X_MIN, X_MAX]).range([PAD, W - PAD])
  const sy = scaleLinear().domain([0, Y_MAX]).range([H - PAD, PAD])

  const pts: string[] = []
  for (let i = 0; i <= 80; i++) {
    const xx = X_MIN + ((X_MAX - X_MIN) * i) / 80
    pts.push(`${sx(xx).toFixed(1)},${sy(f(xx)).toFixed(1)}`)
  }
  const pathD = 'M' + pts.join(' L')

  const step = () => {
    setX(clamp(x - lr * fp(x)))
    setSteps((s) => s + 1)
  }
  const reset = () => {
    setX(START)
    setSteps(0)
  }

  return (
    <div className="space-y-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="A loss curve with a ball descending toward the minimum">
        <line x1={PAD} y1={sy(0)} x2={W - PAD} y2={sy(0)} stroke="var(--border)" strokeWidth={1} />
        <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={H - PAD} stroke="var(--border)" strokeWidth={1} strokeDasharray="3 4" />
        <path d={pathD} fill="none" stroke="var(--accent-2)" strokeWidth={2.5} />
        <motion.circle
          animate={{ cx: sx(clamp(x)), cy: sy(Math.min(Y_MAX, f(x))) }}
          transition={{ type: 'spring', stiffness: 180, damping: 16 }}
          r={8}
          fill="var(--accent)"
          stroke="var(--surface)"
          strokeWidth={2}
        />
      </svg>

      <div className="flex flex-wrap items-center gap-x-5 gap-y-1 font-mono text-xs text-fg-muted">
        <span>
          step <span className="text-fg">{steps}</span>
        </span>
        <span>
          x = <span className="text-fg">{x.toFixed(3)}</span>
        </span>
        <span>
          loss = <span className="text-fg">{f(x).toFixed(3)}</span>
        </span>
      </div>

      <Slider label="Learning rate η" value={lr} min={0.01} max={1.1} step={0.01} onChange={setLr} format={(v) => v.toFixed(2)} />

      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={step}>
          Take a step
        </Button>
        <Button variant="secondary" onClick={reset}>
          Reset
        </Button>
        {lr > 0.5 && lr <= 1 && (
          <span className="text-xs" style={{ color: 'var(--warning)' }}>
            η &gt; 0.5 — watch it overshoot and bounce across the valley.
          </span>
        )}
        {lr > 1 && (
          <span className="text-xs" style={{ color: 'var(--danger)' }}>
            η &gt; 1 — the step is too big; it diverges instead of settling.
          </span>
        )}
      </div>

      <p className="text-xs leading-relaxed text-fg-subtle">
        Each step nudges <span className="font-mono">x</span> downhill by{' '}
        <span className="font-mono">η · f′(x)</span>. Tiny steps glide smoothly to the bottom;
        too-large steps overshoot the minimum entirely.
      </p>
    </div>
  )
}
