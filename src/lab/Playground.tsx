import { useEffect, useRef, useState } from 'react'
import { Button } from '@/primitives/Button'
import { Slider } from '@/primitives/Slider'
import type { GenMsg } from './protocol'

const LENGTH = 320

const show = (ch: string) => (ch === '\n' ? '↵' : ch === ' ' ? '␣' : ch === '\t' ? '⇥' : ch)

export function Playground({
  defaultPrompt,
  disabled,
  generate,
  stop,
  onPromptChange,
}: {
  defaultPrompt: string
  disabled: boolean
  generate: (prompt: string, temp: number, length: number, h: { onChar: (m: GenMsg) => void; onDone: () => void }) => void
  stop: () => void
  onPromptChange?: (p: string) => void
}) {
  const [prompt, setPrompt] = useState(defaultPrompt)
  const [temp, setTemp] = useState(0.8)
  const [out, setOut] = useState('')
  const [shownPrompt, setShownPrompt] = useState('')
  const [top, setTop] = useState<[string, number][]>([])
  const [busy, setBusy] = useState(false)
  const outRef = useRef<HTMLDivElement>(null)
  // characters arrive in bursts; buffer them and paint once per frame
  const buf = useRef<{ text: string; top: [string, number][] | null }>({ text: '', top: null })
  const raf = useRef(0)

  useEffect(() => {
    setPrompt(defaultPrompt)
    onPromptChange?.(defaultPrompt)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaultPrompt])

  useEffect(() => () => clearTimeout(raf.current), [])

  useEffect(() => {
    const el = outRef.current
    if (el) el.scrollTop = el.scrollHeight
  }, [out])

  const paint = () => {
    if (raf.current) return
    // a short timer rather than rAF, so output still appears if the tab is in the background
    raf.current = window.setTimeout(() => {
      raf.current = 0
      const b = buf.current
      if (b.text) setOut((o) => o + b.text)
      if (b.top) setTop(b.top)
      buf.current = { text: '', top: null }
    }, 30)
  }

  const run = () => {
    setOut('')
    setTop([])
    setShownPrompt(prompt)
    setBusy(true)
    buf.current = { text: '', top: null }
    generate(prompt, temp, LENGTH, {
      onChar: (m) => {
        buf.current.text += m.ch
        buf.current.top = m.top
        paint()
      },
      onDone: () => {
        paint()
        setBusy(false)
      },
    })
  }

  return (
    <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_15rem]">
      <div className="min-w-0 space-y-4">
        <div>
          <label htmlFor="ws-prompt" className="mb-1.5 block text-sm text-fg-muted">
            Your prompt
          </label>
          <input
            id="ws-prompt"
            value={prompt}
            maxLength={200}
            onChange={(e) => {
              setPrompt(e.target.value)
              onPromptChange?.(e.target.value)
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !disabled) run()
            }}
            className="w-full border border-border-strong bg-surface px-3 py-2 font-mono text-sm text-fg outline-none focus:border-accent"
          />
        </div>
        <Slider
          label="Temperature (low = safe and repetitive, high = wild)"
          value={temp}
          min={0.2}
          max={1.5}
          step={0.05}
          onChange={setTemp}
          format={(v) => v.toFixed(2)}
        />
        <div className="flex flex-wrap gap-3">
          <Button variant="primary" onClick={run} disabled={disabled}>
            Generate
          </Button>
          {busy && (
            <Button variant="secondary" onClick={stop}>
              Stop
            </Button>
          )}
        </div>
        <div
          ref={outRef}
          className="h-48 overflow-y-auto border border-border bg-surface p-3 font-mono text-[13px] leading-relaxed break-words whitespace-pre-wrap text-fg"
          aria-live="off"
        >
          {out || shownPrompt ? (
            <>
              <span className="text-brass-ink">{shownPrompt}</span>
              {out}
              {busy && <span className="ml-px inline-block h-[1em] w-[0.5em] translate-y-[2px] animate-pulse bg-accent" />}
            </>
          ) : (
            <span className="text-fg-subtle italic">The model’s writing appears here, one character at a time.</span>
          )}
        </div>
      </div>
      <div>
        <p className="mb-1 text-sm text-fg-muted">What it was choosing between</p>
        <p className="mb-3 text-xs text-fg-subtle">Top 5 guesses for the latest character.</p>
        <ul className="space-y-1.5" aria-label="Top five next-character probabilities">
          {(top.length ? top : Array.from({ length: 5 }, () => ['', 0] as [string, number])).map(([ch, p], i) => (
            <li key={i} className="flex items-center gap-2">
              <span className="grid h-7 w-7 shrink-0 place-items-center border border-border bg-surface font-mono text-sm text-fg">
                {ch ? show(ch) : ''}
              </span>
              <span className="relative h-3 flex-1 bg-surface-2">
                <span
                  className="absolute inset-y-0 left-0 transition-[width] duration-150"
                  style={{
                    width: `${Math.round(p * 100)}%`,
                    background: i === 0 ? 'var(--accent)' : 'color-mix(in srgb, var(--accent) 55%, var(--accent-2))',
                  }}
                />
              </span>
              <span className="w-10 text-right font-mono text-xs tabular-nums text-fg-muted">{ch ? `${Math.round(p * 100)}%` : ''}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}
