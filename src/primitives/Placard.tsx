const ROMAN = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X', 'XI', 'XII']

/** A framed museum placard summarizing an exhibit — replaces the old end-of-module quiz. */
export function Placard({ takeaways }: { takeaways: string[] }) {
  return (
    <section aria-label="Key takeaways" className="placard-frame plaster px-6 py-7 sm:px-9 sm:py-8">
      <div className="mb-5 text-center">
        <p className="exhibit-label">On the placard</p>
        <p className="mt-1 font-display text-2xl text-fg sm:text-3xl">Key takeaways</p>
        <div className="ornament mx-auto mt-3 w-36" aria-hidden="true">
          <span />
        </div>
      </div>
      <ol className="space-y-4">
        {takeaways.map((t, i) => (
          <li key={i} className="flex gap-4 text-[0.95rem] leading-relaxed text-fg-muted">
            <span
              className="w-8 shrink-0 pt-0.5 text-right font-display text-lg leading-6 font-semibold text-brass-ink"
              aria-hidden="true"
            >
              {ROMAN[i] ?? i + 1}.
            </span>
            <span>{t}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
