import { useMemo, useState } from 'react'
import { encode, decode } from 'gpt-tokenizer/encoding/o200k_base'
import { TokenChip } from '@/primitives/TokenChip'
import { Button } from '@/primitives/Button'

const PRESETS: { label: string; text: string }[] = [
  { label: 'Spaces & case', text: 'The THE the  the' },
  { label: 'A rare word', text: 'antidisestablishmentarianism' },
  { label: 'Numbers', text: '3.14159 and 1234567' },
  { label: 'Code', text: 'const x = [1, 2, 3];' },
  { label: 'Emoji', text: 'I love pizza 🍕' },
]

export function TokenizerPlayground() {
  const [text, setText] = useState('Tokenization is surprisingly weird!')

  const tokens = useMemo(() => {
    try {
      return encode(text).map((id) => ({ id, text: decode([id]) }))
    } catch {
      return []
    }
  }, [text])

  const charCount = [...text].length
  const ratio = tokens.length ? (charCount / tokens.length).toFixed(1) : '0'
  const hasFragments = tokens.some((t) => t.text.includes('�'))

  return (
    <div className="space-y-4">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={3}
        spellCheck={false}
        className="w-full resize-y rounded-lg border border-border bg-surface-2/40 px-3 py-2.5 font-mono text-sm text-fg outline-none transition-colors focus:border-accent"
        placeholder="Type anything…"
      />

      <div className="flex flex-wrap gap-2">
        {PRESETS.map((p) => (
          <Button key={p.label} variant="ghost" className="!px-2.5 !py-1 text-xs" onClick={() => setText(p.text)}>
            {p.label}
          </Button>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-fg-muted">
        <span>
          <span className="font-mono text-fg">{tokens.length}</span> tokens
        </span>
        <span>
          <span className="font-mono text-fg">{charCount}</span> characters
        </span>
        <span className="text-fg-subtle">≈ {ratio} chars / token</span>
      </div>

      <div className="flex min-h-16 flex-wrap gap-1.5 rounded-lg border border-border bg-surface-2/30 p-3">
        {tokens.length === 0 ? (
          <span className="text-sm text-fg-subtle">Type above to see the tokens…</span>
        ) : (
          tokens.map((t, i) => <TokenChip key={i} text={t.text} id={t.id} />)
        )}
      </div>

      {hasFragments && (
        <p className="text-xs leading-relaxed text-fg-subtle">
          Chips showing “�” are single tokens holding only part of a multi-byte character (common for
          emoji and non-Latin scripts). The model still stitches them back together correctly.
        </p>
      )}
    </div>
  )
}
