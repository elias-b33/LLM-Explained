import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Stepper } from '@/primitives/Stepper'
import { colorForIndex } from '@/theme/tokens'

const CONTRIBS = [
  'Token embedding',
  'Attention · Layer 1',
  'Feed-forward · Layer 1',
  'Attention · Layer 2',
  'Feed-forward · Layer 2',
  'Attention · Layer 3',
  'Feed-forward · Layer 3',
]

export function ResidualStream() {
  const [n, setN] = useState(1)
  const shown = CONTRIBS.slice(0, n)

  return (
    <div className="space-y-4">
      <p className="text-sm leading-relaxed text-fg-muted">
        The residual stream is the <strong className="text-fg">running sum</strong> of every layer’s
        contribution. Step forward to stack each one on.
      </p>

      <div className="flex flex-col gap-5 sm:flex-row sm:items-center">
        <div className="flex w-full flex-col-reverse gap-1 sm:w-52">
          <AnimatePresence initial={false}>
            {shown.map((label, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, height: 0, marginTop: 0 }}
                animate={{ opacity: 1, height: 'auto', marginTop: i === 0 ? 0 : 4 }}
                exit={{ opacity: 0, height: 0, marginTop: 0 }}
                transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                className="flex items-center rounded-sm border-l-[3px] px-2.5 py-2 text-xs font-medium text-fg"
                style={{
                  borderLeftColor: colorForIndex(i),
                  backgroundColor: `color-mix(in srgb, ${colorForIndex(i)} 20%, var(--surface))`,
                }}
              >
                + {label}
              </motion.div>
            ))}
          </AnimatePresence>
        </div>

        <div className="flex-1">
          <p className="text-sm text-fg">
            After {n} step{n > 1 ? 's' : ''}, the stream holds the sum of{' '}
            <span className="font-mono text-accent">{n}</span> contribution{n > 1 ? 's' : ''}.
          </p>
          <p className="mt-1.5 text-xs leading-relaxed text-fg-subtle">
            Because every layer <em>adds</em> rather than replaces, the original signal is never lost —
            and during training, gradients can flow straight back down the stack.
          </p>
        </div>
      </div>

      <Stepper
        step={n - 1}
        count={CONTRIBS.length}
        onPrev={() => setN((v) => Math.max(1, v - 1))}
        onNext={() => setN((v) => Math.min(CONTRIBS.length, v + 1))}
        label={
          <>
            Contribution <span className="font-mono text-fg">{n}</span> of {CONTRIBS.length}
          </>
        }
      />
    </div>
  )
}
