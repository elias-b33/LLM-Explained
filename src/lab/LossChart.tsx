import { useMemo } from 'react'
import { scaleLinear, scaleLog } from 'd3-scale'
import { line } from 'd3-shape'

const W = 640
const H = 280
const M = { top: 16, right: 16, bottom: 30, left: 40 }
const MAX_POINTS = 320

/** Exponential moving average, debiased for the first few points. */
function smooth(values: number[], alpha: number): number[] {
  const out = new Array<number>(values.length)
  let ema = 0
  let w = 0
  for (let i = 0; i < values.length; i++) {
    ema = alpha * ema + (1 - alpha) * values[i]
    w = alpha * w + (1 - alpha)
    out[i] = ema / w
  }
  return out
}

/** A handful of readable ticks for a narrow log range (d3's log ticks get crowded or sparse here). */
function logTicks(lo: number, hi: number): number[] {
  const cands = [0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]
  const inRange = cands.filter((c) => c >= lo && c <= hi)
  if (inRange.length <= 6) return inRange
  const step = Math.ceil(inRange.length / 6)
  return inRange.filter((_, i) => i % step === 0)
}

function downsample(values: number[], max: number): [number, number][] {
  const n = values.length
  if (n <= max) return values.map((v, i) => [i + 1, v])
  const out: [number, number][] = []
  const stride = n / max
  for (let k = 0; k < max; k++) {
    const i = Math.min(n - 1, Math.round(k * stride))
    out.push([i + 1, values[i]])
  }
  out.push([n, values[n - 1]])
  return out
}

export function LossChart({
  losses,
  val,
  vocabSize,
  logScale,
}: {
  losses: number[]
  val: [number, number][]
  vocabSize: number
  logScale: boolean
}) {
  const random = Math.log(Math.max(2, vocabSize))
  const { raw, smoothed, xMax, yMin, yMax } = useMemo(() => {
    const alpha = losses.length > 400 ? 0.97 : 0.9
    const sm = smooth(losses, alpha)
    const raw = downsample(losses, MAX_POINTS)
    const smoothed = downsample(sm, MAX_POINTS)
    const xMax = Math.max(10, losses.length)
    let lo = Infinity
    for (const [, v] of smoothed) lo = Math.min(lo, v)
    for (const [, v] of val) lo = Math.min(lo, v)
    if (!Number.isFinite(lo)) lo = random * 0.5
    return { raw, smoothed, xMax, yMin: Math.max(0.05, lo * 0.85), yMax: random * 1.12 }
  }, [losses, val, random])

  const sx = scaleLinear().domain([0, xMax]).range([M.left, W - M.right])
  const yScale = logScale
    ? scaleLog().domain([yMin, yMax]).range([H - M.bottom, M.top]).clamp(true)
    : scaleLinear().domain([yMin, yMax]).range([H - M.bottom, M.top]).clamp(true)
  const sy = (v: number) => yScale(v)
  const mk = line<[number, number]>()
    .x((d) => sx(d[0]))
    .y((d) => sy(d[1]))
  const yTicks = (logScale ? logTicks(yMin, yMax) : yScale.ticks(5)).filter((t) => t >= yMin && t <= yMax)
  const xTicks = sx.ticks(6)
  const last = smoothed[smoothed.length - 1]
  const lastVal = val[val.length - 1]

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="w-full"
      role="img"
      aria-label={`Loss chart. Training loss ${last ? last[1].toFixed(2) : 'not started'}${lastVal ? `, validation loss ${lastVal[1].toFixed(2)}` : ''}. Random guessing is ${random.toFixed(2)}.`}
    >
      {/* grid */}
      {yTicks.map((t) => (
        <g key={`y${t}`}>
          <line x1={M.left} x2={W - M.right} y1={sy(t)} y2={sy(t)} stroke="var(--border)" strokeWidth={1} />
          <text x={M.left - 6} y={sy(t)} dy="0.32em" textAnchor="end" fontSize={11} fill="var(--fg-subtle)" fontFamily="var(--font-mono)">
            {t.toFixed(t < 1 ? 2 : 1)}
          </text>
        </g>
      ))}
      {xTicks.map((t) => (
        <text key={`x${t}`} x={sx(t)} y={H - M.bottom + 18} textAnchor="middle" fontSize={11} fill="var(--fg-subtle)" fontFamily="var(--font-mono)">
          {t}
        </text>
      ))}
      <text x={W - M.right} y={H - 2} textAnchor="end" fontSize={10} fill="var(--fg-subtle)" letterSpacing="0.12em">
        TRAINING STEPS
      </text>
      <line x1={M.left} x2={W - M.right} y1={H - M.bottom} y2={H - M.bottom} stroke="var(--border-strong)" />

      {/* random-guessing reference */}
      <line
        x1={M.left}
        x2={W - M.right}
        y1={sy(random)}
        y2={sy(random)}
        stroke="var(--brass)"
        strokeWidth={1.5}
        strokeDasharray="6 5"
      />
      <text x={W - M.right - 4} y={sy(random) - 6} textAnchor="end" fontSize={11} fill="var(--brass-ink)">
        random guessing · ln {vocabSize} = {random.toFixed(2)}
      </text>

      {/* raw per-step loss, faint */}
      {raw.length > 1 && <path d={mk(raw) ?? ''} fill="none" stroke="var(--accent)" strokeOpacity={0.18} strokeWidth={1} />}
      {/* smoothed training loss */}
      {smoothed.length > 1 && (
        <path d={mk(smoothed) ?? ''} fill="none" stroke="var(--accent)" strokeWidth={2.4} strokeLinejoin="round" />
      )}
      {/* validation loss */}
      {val.length > 1 && (
        <path d={mk(val) ?? ''} fill="none" stroke="var(--accent-2)" strokeWidth={2} strokeDasharray="1 0" strokeLinejoin="round" />
      )}
      {val.map(([s, v], i) =>
        i === val.length - 1 || val.length < 40 ? (
          <circle key={i} cx={sx(s)} cy={sy(v)} r={i === val.length - 1 ? 3.5 : 2} fill="var(--accent-2)" />
        ) : null,
      )}
      {last && <circle cx={sx(last[0])} cy={sy(last[1])} r={3.5} fill="var(--accent)" />}

      {/* legend */}
      <g transform={`translate(${M.left + 10}, ${M.top + 6})`} fontSize={11}>
        <rect x={-6} y={-10} width={222} height={20} fill="var(--surface)" fillOpacity={0.8} />
        <line x1={0} x2={18} y1={0} y2={0} stroke="var(--accent)" strokeWidth={2.4} />
        <text x={24} y={0} dy="0.32em" fill="var(--fg-muted)">
          training
        </text>
        <line x1={92} x2={110} y1={0} y2={0} stroke="var(--accent-2)" strokeWidth={2} />
        <text x={116} y={0} dy="0.32em" fill="var(--fg-muted)">
          validation (unseen 10%)
        </text>
      </g>
    </svg>
  )
}
