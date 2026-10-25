import { Fragment } from 'react'
import { Link } from 'react-router-dom'

const PIPELINE = [
  { slug: 'tokenization', label: 'Tokens', note: 'text → pieces' },
  { slug: 'embeddings', label: 'Embeddings', note: 'pieces → vectors' },
  { slug: 'attention', label: 'Attention', note: 'mix across tokens' },
  { slug: 'layers', label: 'Feed-forward', note: 'process each token' },
  { slug: 'sampling', label: 'Logits → token', note: 'softmax & sample' },
]

const SECONDARY = [
  { slug: 'pretraining', label: 'Pretraining', note: 'how the weights were learned' },
  { slug: 'post-training', label: 'Alignment', note: 'how it became helpful' },
  { slug: 'inference', label: 'Generation', note: 'looping, one token at a time' },
]

export function CapstoneDiagram() {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-stretch gap-2">
        {PIPELINE.map((n, i) => (
          <Fragment key={n.slug}>
            <Link
              to={`/module/${n.slug}`}
              className="group flex min-w-[7rem] flex-1 flex-col rounded-xl border border-border bg-surface-2/40 p-3 transition-all hover:-translate-y-0.5 hover:border-border-strong"
            >
              <span className="text-sm font-semibold text-fg transition-colors group-hover:text-accent">
                {n.label}
              </span>
              <span className="mt-0.5 text-[11px] text-fg-subtle">{n.note}</span>
            </Link>
            {i < PIPELINE.length - 1 && <span className="self-center text-fg-subtle">→</span>}
          </Fragment>
        ))}
      </div>

      <p className="text-center text-xs text-fg-subtle">
        …then append the new token and run the whole row again.
      </p>

      <div className="grid gap-2 sm:grid-cols-3">
        {SECONDARY.map((n) => (
          <Link
            key={n.slug}
            to={`/module/${n.slug}`}
            className="group rounded-xl border border-dashed border-border-strong bg-surface/40 p-3 transition-colors hover:bg-surface"
          >
            <span className="text-sm font-medium text-fg transition-colors group-hover:text-accent">
              {n.label}
            </span>
            <span className="mt-0.5 block text-[11px] text-fg-subtle">{n.note}</span>
          </Link>
        ))}
      </div>
    </div>
  )
}
