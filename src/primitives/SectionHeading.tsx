import type { ReactNode } from 'react'

/** A lesson section heading with a small accent marker. */
export function SectionHeading({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <h2
      id={id}
      className="mt-2 flex scroll-mt-20 items-center gap-2.5 text-xl font-semibold tracking-tight text-fg sm:text-2xl"
    >
      <span
        className="h-5 w-1 shrink-0 rounded-full"
        style={{
          backgroundImage: 'linear-gradient(var(--accent), var(--accent-2))',
          boxShadow: '0 0 14px color-mix(in srgb, var(--accent) 70%, transparent)',
        }}
        aria-hidden="true"
      />
      {children}
    </h2>
  )
}
