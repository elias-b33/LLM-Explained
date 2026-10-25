/** Format a 0..1 number as a percentage string. */
export const pct = (x: number, digits = 1): string => `${(x * 100).toFixed(digits)}%`

/** Fixed-decimal number string. */
export const fixed = (x: number, digits = 2): string => x.toFixed(digits)
