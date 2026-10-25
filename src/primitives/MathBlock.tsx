import { useMemo } from 'react'
import katex from 'katex'
import { cn } from '@/lib/cn'

/**
 * Renders a LaTeX string with KaTeX. `display` for block (centered) math,
 * otherwise inline. Malformed input degrades gracefully instead of throwing.
 */
export function MathBlock({
  children,
  display = false,
  className,
}: {
  children: string
  display?: boolean
  className?: string
}) {
  const html = useMemo(
    () =>
      katex.renderToString(children, {
        displayMode: display,
        throwOnError: false,
        trust: false,
      }),
    [children, display],
  )

  if (display) {
    // An engraved formula plate: parchment strip between two brass hairlines.
    return (
      <div
        className={cn(
          'my-5 overflow-x-auto overflow-y-hidden border-y border-brass/50 bg-surface/70 px-3 py-3 text-center',
          className,
        )}
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }
  return <span className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
