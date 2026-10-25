import type { Hall } from './halls'

/**
 * Shared architectural constants for the classical gallery shell, plus the
 * walk colliders derived from a hall's columns, vitrines and decor. Kept in
 * one place so Hall (geometry), HallScene (placement) and the first-person
 * controller (collision) always agree.
 */
export const BAY = 6 // column spacing along the hall (m)
export const NAVE_X = 3.2 // the two column rows sit at x = ±NAVE_X
export const RISE = 1.8 // vault rise above the springing line
export const COL_R = 0.42 // collision radius of a column (its square plinth)
export const VITRINE = 0.62 // half-size of a vitrine footprint

export interface Colliders {
  circles: [x: number, z: number, r: number][]
  boxes: [minX: number, minZ: number, maxX: number, maxZ: number][]
}

/** z of every interior column line (bay boundaries, excluding the end walls). */
export function columnZs(depth: number): number[] {
  const out: number[] = []
  for (let z = -BAY; z > -depth + 0.5; z -= BAY) out.push(z)
  return out
}

/** z of every bay centre. */
export function bayCentres(depth: number): number[] {
  const out: number[] = []
  for (let z = -BAY / 2; z > -depth; z -= BAY) out.push(z)
  return out
}

export function buildColliders(hall: Hall): Colliders {
  const W = hall.room.width
  const circles: Colliders['circles'] = []
  const boxes: Colliders['boxes'] = []
  for (const z of columnZs(hall.room.depth)) {
    circles.push([-NAVE_X, z, COL_R], [NAVE_X, z, COL_R])
  }
  for (const seg of hall.segments) {
    if (seg.kind === 'station') {
      const x = seg.x ?? 0
      boxes.push([x - VITRINE, seg.z - VITRINE, x + VITRINE, seg.z + VITRINE])
    }
  }
  for (const d of hall.decor ?? []) {
    if (d.kind === 'vitrine') boxes.push([d.x - 0.5, d.z - 0.5, d.x + 0.5, d.z + 0.5])
    else if (d.kind === 'table') boxes.push([-0.8, Math.min(d.z0, d.z1), 0.8, Math.max(d.z0, d.z1)])
    // in-world exhibits: tight boxes around what stands on the floor
    else if (d.kind === 'slicer') boxes.push([d.x - 0.32, d.z - 0.52, d.x + 0.32, d.z + 0.52])
    else if (d.kind === 'embedding') boxes.push([d.x - 1.1, d.z - 1.15, d.x + 1.15, d.z + 1.15])
    else if (d.kind === 'machine') boxes.push([-0.9, d.consoleZ - 0.33, 0.9, d.consoleZ + 0.33])
    // (the attention walkway is floor tiles you walk over: no collider)
    else if (d.kind === 'curtain' && d.mode === 'partition') {
      const s = d.side === 'left' ? -1 : 1
      const inner = s * (W / 2 - d.len)
      const outer = s * (W / 2)
      boxes.push([Math.min(inner, outer), d.z - 0.12, Math.max(inner, outer), d.z + 0.12])
    }
  }
  return { circles, boxes }
}
