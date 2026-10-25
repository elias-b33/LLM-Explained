import type { ButtonHTMLAttributes } from 'react'
import { cn } from '@/lib/cn'

type Variant = 'primary' | 'secondary' | 'ghost'

const VARIANTS: Record<Variant, string> = {
  primary: 'text-white hover:-translate-y-px shadow-sm',
  secondary:
    'border border-border bg-surface-2/60 text-fg hover:bg-surface-2 hover:border-border-strong',
  ghost: 'text-fg-muted hover:bg-surface-2 hover:text-fg',
}

export function Button({
  variant = 'secondary',
  className,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant }) {
  return (
    <button
      className={cn(
        'inline-flex items-center justify-center gap-2 rounded-lg px-3.5 py-2 text-sm font-medium transition-all enabled:cursor-pointer disabled:cursor-not-allowed disabled:opacity-40',
        VARIANTS[variant],
        className,
      )}
      style={
        variant === 'primary'
          ? { backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))' }
          : undefined
      }
      {...props}
    />
  )
}
