import type { Variants, Transition } from 'motion/react'

/**
 * 8-color palette for token chips and chart bars, indexable by `id % 8`.
 * Chosen to read clearly on both the dark and light themes.
 */
export const CHART_COLORS = [
  '#8b7bff', // violet (brand)
  '#2bd3e0', // teal
  '#35d07f', // green
  '#f0b429', // amber
  '#ff5d73', // rose
  '#4d9bff', // blue
  '#c084fc', // orchid
  '#ff9f5a', // orange
] as const

export const colorForIndex = (i: number): string =>
  CHART_COLORS[((i % CHART_COLORS.length) + CHART_COLORS.length) % CHART_COLORS.length]

const EASE_OUT: [number, number, number, number] = [0.22, 1, 0.36, 1]

/** Pattern 1 — entrance: fade up into place. */
export const fadeRise: Variants = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.45, ease: EASE_OUT } },
}

/** Container that staggers `fadeRise` children. */
export const staggerParent: Variants = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
}

/** Pattern 3 — step-through: directional slide for AnimatePresence (custom = direction). */
export const stepSlide: Variants = {
  enter: (dir: number) => ({ opacity: 0, x: dir >= 0 ? 28 : -28 }),
  center: { opacity: 1, x: 0, transition: { duration: 0.34, ease: EASE_OUT } },
  exit: (dir: number) => ({
    opacity: 0,
    x: dir >= 0 ? -28 : 28,
    transition: { duration: 0.22, ease: EASE_OUT },
  }),
}

/** Pattern 4 — value-change spring (bar widths, moving numbers). */
export const springSoft: Transition = { type: 'spring', stiffness: 240, damping: 30 }
export const springSnappy: Transition = { type: 'spring', stiffness: 340, damping: 26 }
