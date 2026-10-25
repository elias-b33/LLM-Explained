import type { ReactNode } from 'react'
import { cn } from '@/lib/cn'

/** Consistent typographic wrapper for lesson body text. */
export function Prose({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cn('prose-content', className)}>{children}</div>
}
