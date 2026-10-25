import type { ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost'

const VARIANTS: Record<Variant, string> = {
  // Primary lives inside demos, so it glows with the exhibit's violet light.
  primary: 'holo-active hover:-translate-y-px hover:brightness-110',
  secondary:
    'border border-border-strong bg-surface text-fg shadow-[inset_0_1px_0_var(--glass-edge)] hover:border-brass',
  ghost: 'border border-transparent text-fg-muted hover:border-border hover:bg-surface-2/70 hover:text-fg',
}

export function Button({
  variant = 'secondary',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-sm px-3.5 py-2 text-sm font-medium transition-all enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-40',
        VARIANTS[variant],
        className,
      )}
      {...props}
    />
  )
}
