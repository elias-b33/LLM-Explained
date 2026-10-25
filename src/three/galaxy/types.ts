export interface GalaxyWord {
  /** the word */
  w: string
  /** 3D position */
  p: [number, number, number]
  /** cluster index */
  c: number
  /** indices of nearest neighbors (in the original high-dim space) */
  nn: number[]
}

export interface GalaxyCluster {
  name: string
  color: string
  /** center of this universe in world space (for fly-to) */
  center?: [number, number, number]
}

export interface GalaxyAnalogy {
  a: string
  b: string
  c: string
  d: string
}

export interface GalaxyData {
  words: GalaxyWord[]
  clusters: GalaxyCluster[]
  analogies: GalaxyAnalogy[]
}
