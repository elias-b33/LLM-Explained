import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Stepper } from '@/primitives/Stepper'
import { stepSlide } from '@/theme/tokens'
import { cn } from '@/lib/cn'

const STAGES = [
  { name: 'Tokens', detail: 'Your text is split into tokens — the discrete units the model reads.' },
  {
    name: 'Embeddings + position',
    detail: 'Each token id becomes a vector, with positional information added so the model knows the order.',
  },
  {
    name: 'Attention',
    detail: 'Inside each block, every token looks at the others and pulls in whatever context it needs.',
  },
  {
    name: 'Feed-forward',
    detail: 'Each token is then processed on its own. Attention + feed-forward is one block — repeated ×N.',
  },
  {
    name: 'Final vector',
    detail: 'After the last block, the vector at the final position is a rich summary of everything so far.',
  },
  {
    name: 'Logits → probabilities',
    detail: 'An unembedding turns that vector into a score per token; softmax makes it a probability distribution.',
  },
]

export function TransformerFlow() {
  const [step, setStep] = useState(0)

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-2">
        {STAGES.map((s, i) => (
          <div key={s.name} className="flex items-center gap-1.5">
            <button
              onClick={() => setStep(i)}
              className={cn(
                'rounded-sm border px-2.5 py-1.5 text-xs font-medium transition-colors',
                i === step
                  ? 'holo-active border-transparent'
                  : i < step
                    ? 'border-border bg-surface-2 text-fg'
                    : 'border-transparent text-fg-subtle hover:text-fg',
              )}
            >
              {s.name}
            </button>
            {i < STAGES.length - 1 && <span className="text-fg-subtle">→</span>}
          </div>
        ))}
      </div>

      <div className="relative min-h-28 overflow-hidden rounded-xl border border-border bg-surface-2/30 p-5">
        <AnimatePresence mode="wait" custom={1}>
          <motion.div key={step} custom={1} variants={stepSlide} initial="enter" animate="center" exit="exit">
            <p className="mb-1.5 text-sm font-semibold text-fg">
              {step + 1}. {STAGES[step].name}
            </p>
            <p className="text-sm leading-relaxed text-fg-muted">{STAGES[step].detail}</p>
          </motion.div>
        </AnimatePresence>
      </div>

      <Stepper
        step={step}
        count={STAGES.length}
        onPrev={() => setStep((s) => Math.max(0, s - 1))}
        onNext={() => setStep((s) => Math.min(STAGES.length - 1, s + 1))}
      />
    </div>
  )
}
