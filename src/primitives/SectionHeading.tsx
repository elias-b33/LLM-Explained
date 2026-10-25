import type { ReactNode } from 'react'

/** A lesson section heading: serif title over a short brass hairline. */
export function SectionHeading({ children, id }: { children: ReactNode; id?: string }) {
  return (
    <h2 id={id} className="mt-6 scroll-mt-20 font-display text-[1.75rem] leading-tight text-fg sm:text-[2rem]">
      <span className="mb-2 flex items-center gap-2" aria-hidden="true">
        <span className="h-[5px] w-[5px] rotate-45 bg-gold" />
        <span className="brass-rule-left h-px w-16" />
      </span>
      {children}
    </h2>
  )
}
