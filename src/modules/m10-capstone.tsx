import { Prose } from '@/primitives/Prose'
import { SectionHeading } from '@/primitives/SectionHeading'
import { Callout } from '@/primitives/Callout'
import { DemoFrame } from '@/primitives/DemoFrame'
import { CapstoneDiagram } from '@/demos/CapstoneDiagram'
import { Placard } from '@/primitives/Placard'
import { TAKEAWAYS } from '@/data/takeaways'

const GLOSSARY: { term: string; def: string }[] = [
  { term: 'Token', def: 'A sub-word chunk of text — the basic unit a model reads and predicts.' },
  { term: 'Embedding', def: 'A learned vector for a token; similar meanings sit nearby in space.' },
  { term: 'Attention', def: 'The mechanism letting each token gather information from others via Query·Key matching.' },
  { term: 'Query / Key / Value', def: 'What a token is looking for, what it offers, and what it passes on when attended to.' },
  { term: 'Residual stream', def: 'The running representation each layer reads from and adds a small edit back to.' },
  { term: 'Logits', def: 'Raw, unnormalized scores — one per vocabulary token — produced before softmax.' },
  { term: 'Softmax', def: 'Turns logits into a probability distribution that sums to 1.' },
  { term: 'Temperature', def: 'A knob that sharpens (low) or flattens (high) the sampling distribution.' },
  { term: 'Pretraining', def: 'Self-supervised next-token prediction over huge text corpora.' },
  { term: 'Cross-entropy loss', def: '“Surprise” at the true next token; training minimizes it.' },
  { term: 'RLHF', def: 'Tuning a model toward responses humans prefer, after supervised fine-tuning.' },
  { term: 'KV cache', def: 'Stored keys/values of past tokens, reused so generation stays efficient.' },
  { term: 'Context window', def: 'The maximum number of tokens a model can attend to at once.' },
  { term: 'Hallucination', def: 'A confident, fluent output that happens not to be true.' },
]

const READING: { title: string; by: string; href: string }[] = [
  { title: 'Attention Is All You Need', by: 'Vaswani et al., 2017 — the paper that introduced the transformer', href: 'https://arxiv.org/abs/1706.03762' },
  { title: 'The Illustrated Transformer', by: 'Jay Alammar — a beloved visual walkthrough', href: 'https://jalammar.github.io/illustrated-transformer/' },
  { title: 'Neural Networks: Zero to Hero', by: 'Andrej Karpathy — build a GPT from scratch, in code', href: 'https://karpathy.ai/zero-to-hero.html' },
  { title: 'But what is a GPT? (visual intro)', by: '3Blue1Brown — geometry-first intuition for transformers', href: 'https://www.youtube.com/watch?v=wjZofJX0v4M' },
]

export default function Capstone() {
  return (
    <>
      <Prose>
        <p>
          You made it. You can now trace a thought all the way through a language model — and the goal
          of this last module is to hold the whole thing in your head at once.
        </p>
      </Prose>

      <section className="space-y-3">
        <SectionHeading>The whole pipeline, in one breath</SectionHeading>
        <Prose>
          <p>
            Your text is split into <strong>tokens</strong>. Each token becomes an{' '}
            <strong>embedding</strong> vector, with position mixed in. That sequence flows up a stack
            of transformer <strong>blocks</strong>, where <strong>attention</strong> lets tokens share
            information and <strong>feed-forward</strong> layers process each one — every layer adding
            a small edit to the <strong>residual stream</strong>. The final vector is projected to{' '}
            <strong>logits</strong>, <strong>softmax</strong> turns them into probabilities, and a{' '}
            <strong>sampling</strong> step picks the next token. Append it, and do it all again. The
            weights that make every step work were forged by <strong>pretraining</strong> and then
            shaped into a helpful assistant by <strong>post-training</strong>.
          </p>
        </Prose>
        <DemoFrame
          title="The complete map"
          status="illustrative"
          caption="A single diagram of everything you’ve learned — hover any stage to revisit it."
        >
          <CapstoneDiagram />
        </DemoFrame>
      </section>

      <section className="space-y-3">
        <SectionHeading>Glossary</SectionHeading>
        <div className="grid gap-2 sm:grid-cols-2">
          {GLOSSARY.map((g) => (
            <div key={g.term} className="rounded-xl border border-border bg-surface/50 p-3.5">
              <p className="text-sm font-semibold text-fg">{g.term}</p>
              <p className="mt-1 text-[13px] leading-relaxed text-fg-muted">{g.def}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="space-y-3">
        <SectionHeading>Where to go next</SectionHeading>
        <Prose>
          <p>If this lit a spark, these are the classics worth your time:</p>
        </Prose>
        <ul className="space-y-2">
          {READING.map((r) => (
            <li key={r.href}>
              <a
                href={r.href}
                target="_blank"
                rel="noreferrer noopener"
                className="block rounded-xl border border-border bg-surface/50 p-4 transition-colors hover:border-border-strong hover:bg-surface"
              >
                <span className="text-sm font-medium text-accent">{r.title} ↗</span>
                <span className="mt-0.5 block text-[13px] text-fg-muted">{r.by}</span>
              </a>
            </li>
          ))}
        </ul>
      </section>

      <Placard takeaways={TAKEAWAYS['capstone']} />

      <Callout variant="insight" title="One last thought">
        Everything you saw — attention, training, sampling — is just arithmetic repeated at
        staggering scale. The mystery isn’t any single step; it’s that doing something so simple,
        so many times, on so much of what humanity has written, adds up to this.
      </Callout>
    </>
  )
}
