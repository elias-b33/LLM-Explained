import { SectionHeading } from '@/primitives/SectionHeading'

/** A short wall placard: big headline + one line. */
export function Headline({ title, line }: { title: string; line: string }) {
  return (
    <>
      <SectionHeading>{title}</SectionHeading>
      <p style={{ marginTop: 10, fontSize: 17, lineHeight: 1.45, color: 'var(--fg-muted)' }}>{line}</p>
    </>
  )
}
