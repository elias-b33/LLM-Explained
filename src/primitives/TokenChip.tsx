import { colorForIndex } from '@/theme/tokens'
import { mix } from '@/lib/color'

/**
 * A colored chip representing a single token. Whitespace is made visible
 * (· for space, ↵ for newline) so token boundaries are legible.
 */
export function TokenChip({
  text,
  id,
  showId = true,
}: {
  text: string
  id: number
  showId?: boolean
}) {
  const color = colorForIndex(id)
  const display = text.replace(/ /g, '·').replace(/\n/g, '↵')
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 align-middle font-mono text-[13px]"
      style={{ backgroundColor: mix(color, 16), borderColor: mix(color, 32), color: 'var(--fg)' }}
      title={`token id: ${id}`}
    >
      <span className="whitespace-pre">{display}</span>
      {showId && <span className="text-[9px] tabular-nums text-fg-subtle">{id}</span>}
    </span>
  )
}
