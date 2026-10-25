import type { Variants, Transition } from 'motion/react'

/**
 * 8-color palette for token chips and chart bars, indexable by `id % 8`.
 * Museum pigments: the two AI exhibit lights (violet, cyan) followed by
 * mid-tone mineral colours that sit well on cream limestone and dark walnut alike.
 */
export const CHART_COLORS = [
  '#7b6cf6', // exhibit violet (brand)
  '#1fb1c0', // exhibit cyan
  '#4f9a6a', // verdigris
  '#d4a032', // ochre / gilt
  '#d25f62', // terracotta
  '#4f82c9', // lapis
  '#a56ac8', // amethyst
  '#d9824a', // copper
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
