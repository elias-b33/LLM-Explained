/** A museum placard summarizing an exhibit — replaces the old end-of-module quiz. */
export function Placard({ takeaways }: { takeaways: string[] }) {
  return (
    <section
      className="glow-card relative overflow-hidden rounded-2xl border border-border/60 p-5 sm:p-6"
      style={{
        backgroundImage:
          'radial-gradient(120% 90% at 0% 0%, color-mix(in srgb, var(--accent) 14%, transparent), transparent 60%), radial-gradient(90% 80% at 100% 100%, color-mix(in srgb, var(--accent-2) 10%, transparent), transparent 60%)',
        backgroundColor: 'color-mix(in srgb, var(--surface) 65%, transparent)',
      }}
    >
      <div className="mb-4 flex items-center gap-2.5">
        <span
          className="h-4 w-1 rounded-full"
          style={{
            backgroundImage: 'linear-gradient(var(--accent), var(--accent-2))',
            boxShadow: '0 0 12px var(--accent)',
          }}
          aria-hidden="true"
        />
        <span className="text-xs font-semibold tracking-[0.2em] text-fg-subtle uppercase">Key takeaways</span>
      </div>
      <ol className="space-y-3">
        {takeaways.map((t, i) => (
          <li key={i} className="flex gap-3 text-sm leading-relaxed text-fg-muted">
            <span
              className="flex h-5 w-5 shrink-0 items-center justify-center rounded-md font-mono text-[10px] font-semibold text-white"
              style={{
                backgroundImage: 'linear-gradient(140deg, var(--accent), var(--accent-2))',
                boxShadow: '0 0 12px -2px var(--accent)',
              }}
              aria-hidden="true"
            >
              {i + 1}
            </span>
            <span>{t}</span>
          </li>
        ))}
      </ol>
    </section>
  )
}
