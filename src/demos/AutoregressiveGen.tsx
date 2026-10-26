import { useEffect, useMemo, useState } from 'react'
import { encode, decode } from 'gpt-tokenizer/encoding/o200k_base'
import { motion } from 'motion/react'
import { TokenChip } from '@/primitives/TokenChip'
import { Button } from '@/primitives/Button'
import { PlayIcon, PauseIcon } from '@/app/icons'
import { cn } from '@/lib/cn'

const PRESETS = [
  { seed: 'The sun dipped below', cont: ' the horizon, painting the sky in streaks of orange and pink.' },
  { seed: 'In the beginning', cont: ' there was nothing but a vast, silent darkness waiting to unfold.' },
  { seed: 'def fibonacci(n):', cont: '\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)' },
]

export function AutoregressiveGen() {
  const [pi, setPi] = useState(0)
  const preset = PRESETS[pi]

  const { tokens, seedCount } = useMemo(() => {
    const seedTokens = encode(preset.seed)
    const all = encode(preset.seed + preset.cont)
    return { tokens: all.map((id) => ({ id, text: decode([id]) })), seedCount: seedTokens.length }
  }, [preset])

  const total = tokens.length
  const [r, setR] = useState(0)
  const [playing, setPlaying] = useState(false)
  const revealed = Math.min(seedCount + r, total)
  const done = revealed >= total
  // auto-play stops by itself at the end (Reset / a new preset clear `playing`)
  const running = playing && !done

  const choosePreset = (i: number) => {
    if (i === pi) return
    setPi(i)
    setR(0)
    setPlaying(false)
  }

  useEffect(() => {
    if (!running) return
    const id = setTimeout(() => setR((v) => v + 1), 300)
    return () => clearTimeout(id)
  }, [running, r])

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p, i) => (
          <Button
            key={i}
            variant={i === pi ? 'secondary' : 'ghost'}
            className="!px-2.5 !py-1 font-mono text-xs"
            onClick={() => choosePreset(i)}
          >
            {p.seed.slice(0, 16)}…
          </Button>
        ))}
      </div>

      <div className="min-h-20 rounded-lg border border-border bg-surface-2/30 p-3 leading-loose">
        {tokens.slice(0, revealed).map((t, i) => (
          <motion.span
            key={i}
            initial={i >= seedCount ? { opacity: 0, scale: 0.7 } : false}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.18 }}
            className={cn('mr-1 inline-block', i < seedCount && 'opacity-55')}
          >
            <TokenChip text={t.text} id={t.id} showId={false} />
          </motion.span>
        ))}
        {!done && (
          <span className="inline-block h-4 w-[2px] animate-pulse align-middle" style={{ background: 'var(--accent)' }} />
        )}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" onClick={() => !done && setPlaying((p) => !p)} disabled={done}>
          {running ? (
            <>
              <PauseIcon size={15} /> Pause
            </>
          ) : (
            <>
              <PlayIcon size={15} /> {done ? 'Finished' : 'Auto-generate'}
            </>
          )}
        </Button>
        <Button variant="secondary" onClick={() => !done && setR((v) => v + 1)} disabled={done}>
          Next token
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            setR(0)
            setPlaying(false)
          }}
        >
          Reset
        </Button>
      </div>

      <div>
        <div className="mb-1 flex justify-between text-xs text-fg-muted">
          <span>Context window (grows each step)</span>
          <span className="font-mono text-fg">{revealed} tokens</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-surface-2">
          <motion.div
            className="h-full rounded-full"
            style={{ backgroundImage: 'linear-gradient(to right, var(--accent), var(--accent-2))' }}
            animate={{ width: `${(revealed / total) * 100}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>
      </div>

      <p className="text-xs leading-relaxed text-fg-subtle">
        Faded chips are your prompt; bright ones are “generated.” Each step the model re-reads the
        whole context (past keys &amp; values are cached) and appends one token. Tokenization is real;
        the next-token choices are scripted for this demo.
      </p>
    </div>
  )
}
