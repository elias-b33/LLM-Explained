import { useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { BASE_VS_INSTRUCT as D } from '@/data/baseVsInstruct'
import { cn } from '@/lib/cn'
import { mix } from '@/lib/color'

type Mode = 'base' | 'instruct'

export function BaseVsInstruct() {
  const [mode, setMode] = useState<Mode>('base')
  const isBase = mode === 'base'
  const accent = isBase ? 'var(--warning)' : 'var(--success)'

  return (
    <div className="space-y-4">
      <div className="rounded-lg border border-border bg-surface-2/30 p-3">
        <span className="text-xs text-fg-subtle">Prompt</span>
        <p className="mt-0.5 font-mono text-sm text-fg">{D.prompt}</p>
      </div>

      <div className="inline-flex rounded-lg border border-border bg-surface-2/40 p-1">
        {(['base', 'instruct'] as const).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={cn(
              'rounded-md px-3 py-1.5 text-xs font-medium transition-colors',
              mode === m ? 'holo-active' : 'text-fg-muted hover:text-fg',
            )}
          >
            {m === 'base' ? 'Base model' : 'Aligned assistant'}
          </button>
        ))}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.22 }}
        >
          <div className="rounded-lg border p-4" style={{ borderColor: mix(accent, 35), backgroundColor: mix(accent, 6) }}>
            <p className="mb-2 text-xs font-semibold" style={{ color: accent }}>
              {isBase ? 'Base model — raw text completion' : 'Aligned assistant — after SFT + RLHF'}
            </p>
            <p className="text-sm leading-relaxed whitespace-pre-line text-fg-muted">
              {isBase ? D.base : D.instruct}
            </p>
          </div>
          <p className="mt-2 text-xs leading-relaxed text-fg-subtle">
            {isBase
              ? 'A base model just continues the text — here it spirals into more questions, mimicking the web pages it was trained on.'
              : 'After alignment, the very same prompt produces a direct, helpful answer.'}
          </p>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
