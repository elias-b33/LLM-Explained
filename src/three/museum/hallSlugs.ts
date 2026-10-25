/**
 * Slugs that have an authored walkable 3D hall. Keep in sync with HALLS in
 * halls.tsx. This is intentionally lightweight (no content imports) so the 2D
 * pages can cheaply decide whether to show an "Explore in 3D" link without
 * pulling the museum/three bundle into the main chunk.
 */
export const HALL_SLUGS = new Set<string>(['orientation'])
