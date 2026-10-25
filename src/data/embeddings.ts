export interface WordPoint {
  word: string
  x: number
  y: number
  group: string
}

export interface Analogy {
  a: string
  b: string
  c: string
  d: string
}

/**
 * A curated 2-D word map. Positions are hand-placed to show clustering and to
 * make the analogies land cleanly — an illustration of real embedding structure,
 * not an actual projection of a model's vectors.
 */
export const EMBEDDING_WORDS: WordPoint[] = [
  { word: 'man', x: 0.18, y: 0.42, group: 'people' },
  { word: 'woman', x: 0.18, y: 0.62, group: 'people' },
  { word: 'boy', x: 0.1, y: 0.4, group: 'people' },
  { word: 'girl', x: 0.1, y: 0.64, group: 'people' },
  { word: 'king', x: 0.34, y: 0.42, group: 'royalty' },
  { word: 'queen', x: 0.34, y: 0.62, group: 'royalty' },
  { word: 'prince', x: 0.3, y: 0.34, group: 'royalty' },
  { word: 'princess', x: 0.3, y: 0.7, group: 'royalty' },
  { word: 'france', x: 0.6, y: 0.18, group: 'geography' },
  { word: 'paris', x: 0.76, y: 0.18, group: 'geography' },
  { word: 'japan', x: 0.6, y: 0.3, group: 'geography' },
  { word: 'tokyo', x: 0.76, y: 0.3, group: 'geography' },
  { word: 'italy', x: 0.6, y: 0.08, group: 'geography' },
  { word: 'rome', x: 0.76, y: 0.08, group: 'geography' },
  { word: 'cat', x: 0.74, y: 0.66, group: 'animals' },
  { word: 'dog', x: 0.82, y: 0.7, group: 'animals' },
  { word: 'horse', x: 0.91, y: 0.6, group: 'animals' },
  { word: 'cow', x: 0.8, y: 0.62, group: 'animals' },
  { word: 'lion', x: 0.91, y: 0.71, group: 'animals' },
  { word: 'mouse', x: 0.72, y: 0.75, group: 'animals' },
  { word: 'apple', x: 0.2, y: 0.86, group: 'food' },
  { word: 'banana', x: 0.29, y: 0.89, group: 'food' },
  { word: 'bread', x: 0.12, y: 0.82, group: 'food' },
  { word: 'pizza', x: 0.25, y: 0.8, group: 'food' },
  { word: 'one', x: 0.46, y: 0.86, group: 'number' },
  { word: 'two', x: 0.53, y: 0.89, group: 'number' },
  { word: 'three', x: 0.5, y: 0.82, group: 'number' },
]

export const ANALOGIES: Analogy[] = [
  { a: 'king', b: 'man', c: 'woman', d: 'queen' },
  { a: 'paris', b: 'france', c: 'japan', d: 'tokyo' },
]

export const GROUP_COLORS: Record<string, string> = {
  // Mid-tone pigments: legible as dots on cream paper and as glows in the dark 3-D case.
  people: '#4f82c9',
  royalty: '#7b6cf6',
  geography: '#d4a032',
  animals: '#3f9a62',
  food: '#d25f62',
  number: '#1fb1c0',
}

/**
 * Depth (z) per group for the 3-D view. Constant within a group, so the linear
 * analogies (king − man + woman ≈ queen) still hold exactly in three dimensions.
 */
export const GROUP_Z: Record<string, number> = {
  people: -0.35,
  royalty: 0.35,
  geography: 0.65,
  animals: -0.65,
  food: 0.0,
  number: 0.45,
}
