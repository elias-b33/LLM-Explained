/** Build a CSS `color-mix` string — handy for tinted backgrounds/borders from a token. */
export const mix = (color: string, pct: number, base = 'transparent'): string =>
  `color-mix(in srgb, ${color} ${pct}%, ${base})`
