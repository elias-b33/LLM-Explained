import { motion } from 'motion/react'
import type { SampleMsg } from './protocol'

/** Every few seconds the model writes from the same prompt with the same dice: newest on top. */
export function SampleTimeline({ samples }: { samples: SampleMsg[] }) {
  const items = [...samples].reverse()
  return (
    <ol className="max-h-[30rem] space-y-3 overflow-y-auto pr-1" aria-live="polite" aria-relevant="additions">
      {items.map((s, i) => {
        const newest = i === 0
        return (
          <motion.li
            key={`${s.step}-${samples.length - i}`}
            initial={newest ? { opacity: 0, y: -8 } : false}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.35 }}
            className={
              newest
                ? 'border border-accent/50 bg-surface p-3 shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_12%,transparent),0_8px_24px_-14px_var(--accent)]'
                : 'border border-border bg-surface/60 p-3'
            }
          >
            <div className="mb-1.5 flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5">
              <span className="exhibit-label !tracking-[0.14em]">
                Step {s.step.toLocaleString()}
                {s.step === 0 && (s.before ? ' · pretrained, before fine-tuning' : ' · untrained')}
              </span>
              <span className="font-mono text-[11px] text-fg-subtle">
                loss {s.loss.toFixed(2)}
                {newest && <span className="ml-2 text-accent">newest</span>}
              </span>
            </div>
            <p className="font-mono text-[12.5px] leading-relaxed break-words whitespace-pre-wrap text-fg">
              <span className="text-brass-ink">{s.prompt}</span>
              {s.text}
            </p>
          </motion.li>
        )
      })}
    </ol>
  )
}
