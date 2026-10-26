/**
 * The galaxy's 45k words, fetched once as a compact binary asset (see galaxyBin.ts).
 * The /galaxy route awaits loadGalaxy() in its lazy loader, so by the time GalaxyPage
 * (and the lazily imported GalaxyScene) render, galaxyData() is synchronous.
 */
import galaxyUrl from '@/data/galaxy.bin?url'
import { decodeGalaxyFile } from './galaxyBin'
import type { GalaxyData } from './types'

let data: GalaxyData | null = null
let pending: Promise<GalaxyData> | null = null

export function loadGalaxy(): Promise<GalaxyData> {
  if (data) return Promise.resolve(data)
  pending ??= (async () => {
    const res = await fetch(galaxyUrl)
    if (!res.ok) throw new Error(`could not load the galaxy data (HTTP ${res.status})`)
    data = await decodeGalaxyFile(new Uint8Array(await res.arrayBuffer()))
    return data
  })().finally(() => {
    pending = null // a failed load can be retried by navigating again
  })
  return pending
}

/** The loaded data; only valid after loadGalaxy() has resolved (the route guarantees it). */
export function galaxyData(): GalaxyData {
  if (!data) throw new Error('galaxy data used before loadGalaxy() resolved')
  return data
}
