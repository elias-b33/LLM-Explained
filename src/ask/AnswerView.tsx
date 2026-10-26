import { useEffect, useRef, useState, type ReactNode } from 'react'
import { ChevronDownIcon } from '@/app/icons'
import { cn } from '@/lib/cn'
import type { Turn } from './useAsk'

/**
 * Citations: the requested "[2]" / "[1, 3]" form, plus two forms small models produce instead:
 * "Excerpt 2" / "utdrag 2", and a bare digit right after a sentence ends ("… Scotland. 2").
 */
const CITE_RE = /\[(\d+(?:\s*[,–-]\s*\d+)*)\]|\b((?:[Ee]xcerpt|[Uu]tdrag)\s+)(\d)\b|(?<=[.!?"”’)])(\s+)(\d)(?=\.?(?:\s|$))/g

/** the answer text with every citation turned into a button */
function withCitations(text: string, max: number, onCite: (n: number) => void, active: number | null): ReactNode[] {
  const out: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  CITE_RE.lastIndex = 0
  while ((m = CITE_RE.exec(text))) {
    if (m.index > last) out.push(text.slice(last, m.index))
    const prefix = m[2] ?? m[4] ?? ''
    const nums = (m[1] ?? m[3] ?? m[5]).split(/\s*[,–-]\s*/).map(Number)
    if (m[1] === undefined && nums[0] > max) {
      // a bare number that isn't a passage: leave it as plain text
      out.push(m[0])
      last = m.index + m[0].length
      continue
    }
    if (prefix) out.push(prefix)
    out.push(
      <span key={m.index} className="whitespace-nowrap">
        {nums.map((n, k) =>
          n >= 1 && n <= max ? (
            <button
              key={k}
              type="button"
              onClick={() => onCite(n)}
              aria-label={`Show passage ${n}`}
              className={cn(
                'mx-px inline-flex min-w-[1.5em] cursor-pointer items-center justify-center rounded-sm border px-1 align-baseline font-mono text-[0.75em] leading-[1.35] transition-colors',
                active === n
                  ? 'border-accent bg-accent text-white'
                  : 'border-brass text-brass-ink hover:border-accent hover:text-accent',
              )}
            >
              {n}
            </button>
          ) : (
            <span key={k} className="font-mono text-[0.75em] text-fg-subtle" title="The model cited a passage that doesn’t exist">
              [{n}?]
            </span>
          ),
        )}
      </span>,
    )
    last = m.index + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

export function TurnView({
  turn,
  answering,
  retrievalOnly = false,
  onStop,
}: {
  turn: Turn
  answering: boolean
  retrievalOnly?: boolean
  onStop: () => void
}) {
  const [active, setActive] = useState<number | null>(null)
  const [open, setOpen] = useState(false)
  const refs = useRef(new Map<number, HTMLLIElement>())

  useEffect(() => {
    if (active === null || !open) return
    refs.current.get(active)?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }, [active, open])

  const cite = (n: number) => {
    setOpen(true)
    setActive(n)
  }
  const tps = turn.decodeTps
  // Qwen3 runs with thinking off, but strip a stray <think> block just in case
  const answer = turn.answer.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').trim()
  // retrieval-only mode: show the passages right away
  const passagesOpen = open || retrievalOnly

  return (
    <div className="min-w-0 border border-border bg-surface/70">
      <div className="border-b border-border bg-surface-2/40 px-4 py-2.5">
        <p className="text-[10px] font-semibold tracking-[0.16em] text-fg-subtle uppercase">You asked</p>
        <p className="font-display text-lg leading-snug break-words text-fg">{turn.question}</p>
      </div>
      <div className="px-4 py-3">
        <p className="text-[10px] font-semibold tracking-[0.16em] text-fg-subtle uppercase">
          The model answers {turn.passages.length > 0 && <span className="normal-case tracking-normal">· from {turn.passages.length} passages</span>}
        </p>
        <div className="mt-1 text-[0.97rem] leading-relaxed break-words whitespace-pre-wrap text-fg" aria-live="polite" data-raw-answer={turn.done ? turn.answer : undefined}>
          {retrievalOnly ? (
            <span className="text-fg-subtle italic">
              {turn.passages.length ? 'No chat model without WebGPU: here are the closest passages.' : 'Searching the text…'}
            </span>
          ) : answer ? (
            withCitations(answer, turn.passages.length, cite, active)
          ) : turn.done ? (
            <span className="text-fg-subtle italic">(no answer)</span>
          ) : (
            <span className="text-fg-subtle italic">{turn.passages.length ? 'Reading the passages…' : 'Searching the text…'}</span>
          )}
          {!turn.done && answer && <span className="ml-0.5 inline-block h-[1em] w-[0.45em] animate-pulse bg-accent align-[-0.15em]" />}
        </div>
        {turn.error && (
          <p className="mt-2 text-sm" style={{ color: 'var(--danger)' }}>
            Generation failed: {turn.error}
          </p>
        )}
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 font-mono text-[11px] text-fg-subtle">
          {!turn.done && answering && (
            <button type="button" onClick={onStop} className="btn-outline !px-3 !py-1.5 !text-[11px]">
              ■ Stop
            </button>
          )}
          {turn.promptTokens > 0 && <span>prompt {turn.promptTokens} tokens</span>}
          {turn.done && turn.tokens > 0 && (
            <>
              <span>answer {turn.tokens} tokens</span>
              <span>{tps.toFixed(1)} tokens/s</span>
              <span>first token after {(turn.firstTokenMs / 1000).toFixed(2)} s</span>
            </>
          )}
          {turn.stopped && <span style={{ color: 'var(--warning)' }}>stopped</span>}
          {turn.done && turn.workerFetches !== null && (
            <span title="Network requests made by the model worker since the model finished loading">
              network requests since load: {turn.workerFetches}
            </span>
          )}
        </div>
      </div>
      {turn.passages.length > 0 && (
        <div className="border-t border-border">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            aria-expanded={passagesOpen}
            className="flex w-full cursor-pointer items-center gap-2 px-4 py-2.5 text-left text-sm text-fg-muted hover:bg-surface-2/60 hover:text-fg"
          >
            <ChevronDownIcon size={14} className={cn('shrink-0 transition-transform', passagesOpen && 'rotate-180')} />
            Retrieved passages and similarity scores
          </button>
          {passagesOpen && turn.hyde && (
            <p className="px-4 pb-2 text-xs text-fg-muted">
              <span className="font-semibold text-fg-subtle">Also searched with the model’s guess (HyDE):</span>{' '}
              <i>{turn.hyde}</i>
            </p>
          )}
          {passagesOpen && (
            <ol className="space-y-2 px-4 pb-4">
              {turn.passages.map((p) => (
                <li
                  key={p.n}
                  ref={(el) => {
                    if (el) refs.current.set(p.n, el)
                    else refs.current.delete(p.n)
                  }}
                  className={cn(
                    'min-w-0 border p-3 transition-colors',
                    active === p.n
                      ? 'border-accent bg-[color-mix(in_srgb,var(--accent)_9%,var(--surface))] shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_14%,transparent)]'
                      : 'border-border bg-bg/40',
                  )}
                >
                  <div className="mb-1.5 flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setActive(p.n)}
                      className="cursor-pointer rounded-sm border border-brass px-1.5 font-mono text-xs text-brass-ink"
                    >
                      [{p.n}]
                    </button>
                    <span className="font-mono text-[11px] text-fg-subtle">
                      cosine similarity {p.score.toFixed(3)} · keyword score {p.keyword.toFixed(1)}
                      {p.rerank !== null && ` · reranker ${p.rerank.toFixed(2)}`} · passage #{p.chunkId + 1}
                    </span>
                    <span className="h-1.5 min-w-12 flex-1 overflow-hidden rounded-full bg-surface-2">
                      <span
                        className="block h-full bg-brass"
                        style={{ width: `${Math.max(4, Math.min(100, ((p.score - 0.6) / 0.35) * 100))}%` }}
                      />
                    </span>
                  </div>
                  {p.header && <p className="mb-1 text-[11px] font-semibold tracking-wide text-brass-ink">{p.header}</p>}
                  <p className="font-mono text-[12px] leading-relaxed break-words whitespace-pre-wrap text-fg-muted">{p.text}</p>
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </div>
  )
}
