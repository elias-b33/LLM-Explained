import { useLayoutEffect, useMemo, useRef } from 'react'
import { Environment } from '@react-three/drei'
import {
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  Curve,
  DoubleSide,
  LatheGeometry,
  Object3D,
  PlaneGeometry,
  SpotLight,
  TubeGeometry,
  Vector2,
  Vector3,
} from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { Hall as HallData } from './halls'
import { BAY, NAVE_X, RISE, columnZs } from './layout'
import { tile, useMuseumTextures, type MuseumTextures } from './materials'

/* ------------------------------------------------------------------------ */
/* Colours                                                                  */
/* ------------------------------------------------------------------------ */
const PLASTER = '#f3ece0' // warm cream tint over the neutral grey scan
const PLASTER_DARK = '#ded4c2'
const CARPET = '#3d55a6' // navy tint over the grey wool scan
const LAMP = '#ffe1bd'
const RAIL = '#3a332b'

/* ------------------------------------------------------------------------ */
/* Procedural geometry                                                      */
/* ------------------------------------------------------------------------ */

/**
 * A groin vault over one rectangular bay: the union of two elliptical barrel
 * vaults, built as four quadrant patches so the groins stay crisp. Normals
 * are analytic and face down; UVs follow arc length so plaster doesn't stretch.
 */
function pushGroinVault(
  out: { pos: number[]; nor: number[]; uv: number[]; idx: number[] },
  cx: number,
  cz: number,
  a: number,
  b: number,
  rise: number,
  y0: number,
  n = 18,
) {
  const sq = (q: number) => Math.sqrt(Math.max(1e-4, q))
  const quads: Array<{ axis: 'x' | 'z'; sign: 1 | -1 }> = [
    { axis: 'z', sign: 1 },
    { axis: 'z', sign: -1 },
    { axis: 'x', sign: 1 },
    { axis: 'x', sign: -1 },
  ]
  for (const q of quads) {
    const base = out.pos.length / 3
    for (let i = 0; i <= n; i++) {
      const u = i / n
      for (let j = 0; j <= n; j++) {
        const v = (j / n) * 2 - 1
        let lx: number, lz: number, h: number, nx: number, ny: number, nz: number, uu: number, vv: number
        if (q.axis === 'z') {
          // barrel running along z (cross-section in x)
          lz = q.sign * u * b
          lx = v * a * u
          const s = sq(1 - (lx / a) ** 2)
          h = rise * s
          const d = (-rise * lx) / (a * a * s)
          nx = d
          ny = -1
          nz = 0
          uu = (Math.asin(Math.max(-1, Math.min(1, lx / a))) * (a + rise)) / 2
          vv = cz + lz
        } else {
          lx = q.sign * u * a
          lz = v * b * u
          const s = sq(1 - (lz / b) ** 2)
          h = rise * s
          const d = (-rise * lz) / (b * b * s)
          nx = 0
          ny = -1
          nz = d
          uu = cx + lx
          vv = (Math.asin(Math.max(-1, Math.min(1, lz / b))) * (b + rise)) / 2
        }
        const len = Math.hypot(nx, ny, nz)
        out.pos.push(cx + lx, y0 + h, cz + lz)
        out.nor.push(nx / len, ny / len, nz / len)
        out.uv.push(uu / 3, vv / 3)
      }
    }
    // two triangles per cell, wound so the visible face points down
    const P = out.pos
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        const A = base + i * (n + 1) + j
        const B = A + 1
        const C = A + (n + 1)
        const D = C + 1
        for (const [p, q2, r] of [
          [A, B, C],
          [B, D, C],
        ]) {
          const e1x = P[q2 * 3] - P[p * 3]
          const e1z = P[q2 * 3 + 2] - P[p * 3 + 2]
          const e2x = P[r * 3] - P[p * 3]
          const e2z = P[r * 3 + 2] - P[p * 3 + 2]
          const cy = e1z * e2x - e1x * e2z // y of cross(e1, e2)
          if (cy < 0) out.idx.push(p, q2, r)
          else out.idx.push(p, r, q2)
        }
      }
    }
  }
}

function buildVault(W: number, D: number, y0: number): BufferGeometry {
  const out = {
    pos: [] as number[],
    nor: [] as number[],
    uv: [] as number[],
    idx: [] as number[],
  }
  const aisleA = (W / 2 - NAVE_X) / 2
  const aisleX = NAVE_X + aisleA
  for (let cz = -BAY / 2; cz > -D; cz -= BAY) {
    pushGroinVault(out, 0, cz, NAVE_X, BAY / 2, RISE, y0)
    pushGroinVault(out, -aisleX, cz, aisleA, BAY / 2, RISE, y0, 14)
    pushGroinVault(out, aisleX, cz, aisleA, BAY / 2, RISE, y0, 14)
  }
  const g = new BufferGeometry()
  g.setAttribute('position', new BufferAttribute(new Float32Array(out.pos), 3))
  g.setAttribute('normal', new BufferAttribute(new Float32Array(out.nor), 3))
  g.setAttribute('uv', new BufferAttribute(new Float32Array(out.uv), 2))
  g.setIndex(out.idx)
  return g
}

/** A half-elliptical arch in a vertical plane — used for the plaster ribs. */
class ArchCurve extends Curve<Vector3> {
  private c: Vector3
  private half: number
  private rise: number
  private along: 'x' | 'z'
  constructor(c: Vector3, half: number, rise: number, along: 'x' | 'z') {
    super()
    this.c = c
    this.half = half
    this.rise = rise
    this.along = along
  }
  getPoint(t: number, target = new Vector3()) {
    const th = -Math.PI / 2 + t * Math.PI
    const off = this.half * Math.sin(th)
    const y = this.c.y + this.rise * Math.cos(th)
    return this.along === 'x' ? target.set(this.c.x + off, y, this.c.z) : target.set(this.c.x, y, this.c.z + off)
  }
}

function buildRibs(W: number, D: number, y0: number): BufferGeometry {
  const parts: BufferGeometry[] = []
  const aisleA = (W / 2 - NAVE_X) / 2
  const aisleX = NAVE_X + aisleA
  const rib = (c: Vector3, half: number, along: 'x' | 'z', r = 0.11) =>
    parts.push(new TubeGeometry(new ArchCurve(c, half, RISE, along), 40, r, 8, false))
  const lines = [0, ...columnZs(D), -D]
  for (const z of lines) {
    const zz = z === 0 ? -0.05 : z === -D ? -D + 0.05 : z
    rib(new Vector3(0, y0, zz), NAVE_X, 'x', 0.14)
    rib(new Vector3(-aisleX, y0, zz), aisleA, 'x')
    rib(new Vector3(aisleX, y0, zz), aisleA, 'x')
  }
  for (let cz = -BAY / 2; cz > -D; cz -= BAY) {
    for (const x of [-NAVE_X, NAVE_X]) rib(new Vector3(x, y0, cz), BAY / 2, 'z', 0.15)
    for (const x of [-W / 2 + 0.05, W / 2 - 0.05]) rib(new Vector3(x, y0, cz), BAY / 2, 'z', 0.1)
  }
  const g = mergeGeometries(parts)
  parts.forEach((p) => p.dispose())
  return g
}

/** Classical column profile (base mouldings, tapering shaft, echinus). */
function buildColumn(): LatheGeometry {
  const pts: [number, number][] = [
    [0.43, 0.3],
    [0.43, 0.36],
    [0.4, 0.4],
    [0.38, 0.43],
    [0.39, 0.47],
    [0.36, 0.51],
    [0.33, 0.55],
    [0.33, 1.4],
    [0.322, 2.3],
    [0.305, 3.2],
    [0.29, 3.66],
    [0.315, 3.69],
    [0.315, 3.73],
    [0.29, 3.76],
    [0.3, 3.82],
    [0.36, 3.9],
    [0.41, 3.95],
    [0.43, 3.98],
  ]
  const g = new LatheGeometry(
    pts.map(([r, y]) => new Vector2(r, y)),
    40,
  )
  // UVs by height so the emissive uplight gradient can ramp in metres.
  const p = g.attributes.position
  const uv = g.attributes.uv
  for (let i = 0; i < p.count; i++) uv.setY(i, p.getY(i) / 4)
  uv.needsUpdate = true
  return g
}

/** Warm uplight wash: bright at the base, fading by mid-shaft. */
function makeUplightTexture(): CanvasTexture {
  const c = document.createElement('canvas')
  c.width = 4
  c.height = 256
  const ctx = c.getContext('2d')!
  const g = ctx.createLinearGradient(0, 256, 0, 0) // canvas bottom = uv.y 0
  g.addColorStop(0, 'rgb(255,214,168)')
  g.addColorStop(0.08, 'rgb(205,165,120)')
  g.addColorStop(0.35, 'rgb(62,48,34)')
  g.addColorStop(0.6, 'rgb(8,5,3)')
  g.addColorStop(1, 'rgb(0,0,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, 4, 256)
  const t = new CanvasTexture(c)
  t.flipY = true
  return t
}

/** Sheer curtain: a plane rippled into soft folds. */
function buildCurtain(len: number, h: number): PlaneGeometry {
  const g = new PlaneGeometry(len, h, Math.max(8, Math.round(len * 18)), 4)
  const p = g.attributes.position
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i)
    const y = p.getY(i)
    const fold = Math.sin(x * 21) * 0.045 + Math.sin(x * 7.3 + 1.2) * 0.03
    p.setZ(i, fold * (0.85 + 0.15 * (y / h + 0.5)))
  }
  g.computeVertexNormals()
  return g
}

/* ------------------------------------------------------------------------ */
/* Track lighting                                                           */
/* ------------------------------------------------------------------------ */

interface Spot {
  pos: [number, number, number]
  target: [number, number, number]
  angle: number
  intensity: number
  shadow?: boolean
}

function buildSpots(hall: HallData): Spot[] {
  const W = hall.room.width
  const out: Spot[] = []
  const AY = 4.38 // aisle rail height
  const NY = 4.5 // nave rail height
  for (const seg of hall.segments) {
    if (seg.kind === 'panel') {
      const s = seg.side === 'left' ? -1 : seg.side === 'right' ? 1 : 0
      if (s === 0) continue
      out.push({
        pos: [s * 5.0, AY, seg.z + 1.9],
        target: [s * (W / 2 - 0.1), 1.7, seg.z],
        angle: 0.52,
        intensity: 70,
      })
    } else if (seg.kind === 'station') {
      const x = seg.x ?? 0
      out.push({
        pos: x === 0 ? [1.6, NY, seg.z + 2.2] : [Math.sign(x) * 5.0, AY, seg.z + 2.2],
        target: [x, 1.2, seg.z],
        angle: 0.34,
        intensity: 95,
        shadow: true,
      })
    } else if (seg.kind === 'doorway') {
      out.push({
        pos: [-1.6, NY, seg.z + 4.2],
        target: [0, 2.0, seg.z],
        angle: 0.5,
        intensity: 70,
      })
    }
  }
  for (const d of hall.decor ?? []) {
    if (d.kind === 'vitrine') {
      const s = Math.sign(d.x)
      out.push({
        pos: [s * 5.0, AY, d.z + 1.9],
        target: [d.x, 1.0, d.z],
        angle: 0.36,
        intensity: 60,
      })
    } else if (d.kind === 'table') {
      const mid = (d.z0 + d.z1) / 2
      out.push({
        pos: [-1.6, NY, mid + 1.4],
        target: [0, 0.9, mid],
        angle: 0.62,
        intensity: 110,
        shadow: true,
      })
    } else if (d.kind === 'slicer') {
      out.push({ pos: [Math.sign(d.x) * 5.0, AY, d.z + 2.0], target: [d.x, 1.1, d.z], angle: 0.42, intensity: 55 })
    } else if (d.kind === 'machine') {
      // one wide spot over the console and the board above it
      out.push({ pos: [1.6, NY, d.consoleZ + 2.6], target: [0, 1.3, d.consoleZ - 0.6], angle: 0.5, intensity: 70 })
    } else if (d.kind === 'screen') {
      out.push({
        pos: [1.6, NY, d.z + 3.4],
        target: [0, 2.6, d.z],
        angle: 0.5,
        intensity: 45,
      })
    }
  }
  return out
}

const DUMMY = new Object3D()

function TrackSpot({ spot }: { spot: Spot }) {
  const light = useRef<SpotLight>(null)
  const quat = useMemo(() => {
    DUMMY.position.set(...spot.pos)
    DUMMY.lookAt(...spot.target)
    return DUMMY.quaternion.clone()
  }, [spot])
  useLayoutEffect(() => {
    const l = light.current
    if (!l) return
    l.target.position.set(...spot.target)
    l.target.updateMatrixWorld()
  }, [spot])
  return (
    <group>
      {/* stem down from the rail + the can, aimed at the exhibit */}
      <mesh position={[spot.pos[0], spot.pos[1] + 0.07, spot.pos[2]]}>
        <cylinderGeometry args={[0.012, 0.012, 0.14, 6]} />
        <meshStandardMaterial color={RAIL} metalness={0.6} roughness={0.4} />
      </mesh>
      <group position={spot.pos} quaternion={quat}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.07, 0.06, 0.2, 16]} />
          <meshStandardMaterial color={RAIL} metalness={0.7} roughness={0.35} />
        </mesh>
        <mesh position={[0, 0, 0.101]}>
          <circleGeometry args={[0.052, 20]} />
          <meshBasicMaterial color={LAMP} toneMapped={false} />
        </mesh>
      </group>
      <spotLight
        ref={light}
        position={spot.pos}
        angle={spot.angle}
        penumbra={0.75}
        decay={2}
        distance={0}
        intensity={spot.intensity}
        color={LAMP}
        castShadow={!!spot.shadow}
        shadow-mapSize={[1024, 1024]}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
      />
    </group>
  )
}

/* ------------------------------------------------------------------------ */
/* The room                                                                 */
/* ------------------------------------------------------------------------ */

function Walls({ tex, W, D, H }: { tex: MuseumTextures; W: number; D: number; H: number }) {
  const top = H + RISE + 0.1
  const side = useMemo(() => tile(tex.plaster, D / 3, top / 3), [tex, D, top])
  const end = useMemo(() => tile(tex.plaster, W / 3, top / 3), [tex, W, top])
  const trim = useMemo(() => tile(tex.plaster, D / 3, 0.2), [tex, D])
  const pil = useMemo(() => tile(tex.plaster, 0.3, H / 3), [tex, H])
  const cols = useMemo(() => columnZs(D), [D])
  return (
    <group>
      {/* side + end walls */}
      <mesh position={[-W / 2, top / 2, -D / 2]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[D, top]} />
        <meshStandardMaterial {...side} color={PLASTER} roughness={1} />
      </mesh>
      <mesh position={[W / 2, top / 2, -D / 2]} rotation={[0, -Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[D, top]} />
        <meshStandardMaterial {...side} color={PLASTER} roughness={1} />
      </mesh>
      <mesh position={[0, top / 2, -D]} receiveShadow>
        <planeGeometry args={[W, top]} />
        <meshStandardMaterial {...end} color={PLASTER} roughness={1} />
      </mesh>
      <mesh position={[0, top / 2, 0]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[W, top]} />
        <meshStandardMaterial {...end} color={PLASTER} roughness={1} />
      </mesh>

      {/* cornice at the springing line + dark skirting */}
      {[-1, 1].map((s) => (
        <group key={s}>
          <mesh position={[s * (W / 2 - 0.1), H - 0.06, -D / 2]}>
            <boxGeometry args={[0.2, 0.2, D]} />
            <meshStandardMaterial {...trim} color={PLASTER_DARK} />
          </mesh>
          <mesh position={[s * (W / 2 - 0.16), H - 0.2, -D / 2]}>
            <boxGeometry args={[0.1, 0.08, D]} />
            <meshStandardMaterial {...trim} color={PLASTER_DARK} />
          </mesh>
          <mesh position={[s * (W / 2 - 0.02), 0.07, -D / 2]}>
            <boxGeometry args={[0.04, 0.14, D]} />
            <meshStandardMaterial color="#2b2119" roughness={0.6} />
          </mesh>
        </group>
      ))}

      {/* pilasters answering each column on the side walls */}
      {cols.map((z) =>
        [-1, 1].map((s) => (
          <group key={`${z}${s}`}>
            <mesh position={[s * (W / 2 - 0.12), H / 2, z]} castShadow receiveShadow>
              <boxGeometry args={[0.24, H, 0.62]} />
              <meshStandardMaterial {...pil} color={PLASTER} />
            </mesh>
            <mesh position={[s * (W / 2 - 0.16), 0.16, z]}>
              <boxGeometry args={[0.32, 0.32, 0.74]} />
              <meshStandardMaterial {...pil} color={PLASTER_DARK} />
            </mesh>
            <mesh position={[s * (W / 2 - 0.17), H - 0.32, z]}>
              <boxGeometry args={[0.34, 0.16, 0.76]} />
              <meshStandardMaterial {...pil} color={PLASTER_DARK} />
            </mesh>
          </group>
        )),
      )}
      {/* end-wall pilasters where the column rows meet the walls */}
      {[0, -D].map((z) =>
        [-NAVE_X, NAVE_X].map((x) => (
          <mesh key={`${z}${x}`} position={[x, H / 2, z + (z === 0 ? -0.12 : 0.12)]} receiveShadow>
            <boxGeometry args={[0.66, H, 0.24]} />
            <meshStandardMaterial {...pil} color={PLASTER} />
          </mesh>
        )),
      )}
    </group>
  )
}

function Columns({ tex, D, H }: { tex: MuseumTextures; D: number; H: number }) {
  const lathe = useMemo(() => buildColumn(), [])
  const glow = useMemo(() => makeUplightTexture(), [])
  const shaft = useMemo(() => tile(tex.plaster, 1, 1.4), [tex])
  const block = useMemo(() => tile(tex.plaster, 0.35, 0.35), [tex])
  const zs = useMemo(() => columnZs(D), [D])
  return (
    <group>
      {zs.map((z) =>
        [-NAVE_X, NAVE_X].map((x) => (
          <group key={`${x}${z}`} position={[x, 0, z]}>
            {/* square plinth */}
            <mesh position={[0, 0.15, 0]} castShadow receiveShadow>
              <boxGeometry args={[0.92, 0.3, 0.92]} />
              <meshStandardMaterial {...block} color={PLASTER_DARK} emissive="#ffd2a0" emissiveIntensity={0.1} />
            </mesh>
            {/* turned base, shaft and capital, warm-uplit from the floor */}
            <mesh geometry={lathe} castShadow receiveShadow>
              <meshStandardMaterial
                {...shaft}
                color={PLASTER}
                emissiveMap={glow}
                emissive="#ffffff"
                emissiveIntensity={0.42}
              />
            </mesh>
            {/* abacus + entablature block the arches spring from */}
            <mesh position={[0, 4.03, 0]}>
              <boxGeometry args={[0.92, 0.1, 0.92]} />
              <meshStandardMaterial {...block} color={PLASTER} />
            </mesh>
            <mesh position={[0, (4.08 + H) / 2, 0]}>
              <boxGeometry args={[0.8, H - 4.08, 0.8]} />
              <meshStandardMaterial {...block} color={PLASTER} />
            </mesh>
            <mesh position={[0, H - 0.04, 0]}>
              <boxGeometry args={[0.9, 0.09, 0.9]} />
              <meshStandardMaterial {...block} color={PLASTER_DARK} />
            </mesh>
            {/* the uplight puck at the foot */}
            <mesh position={[0, 0.31, 0.47]} rotation={[-Math.PI / 2.6, 0, 0]}>
              <circleGeometry args={[0.035, 12]} />
              <meshBasicMaterial color={LAMP} toneMapped={false} />
            </mesh>
          </group>
        )),
      )}
    </group>
  )
}

function Curtains({ hall, tex }: { hall: HallData; tex: MuseumTextures }) {
  const W = hall.room.width
  const H = hall.room.height
  const ch = H - 0.25
  const list = (hall.decor ?? []).filter((d) => d.kind === 'curtain')
  const geos = useMemo(() => {
    const m = new Map<number, PlaneGeometry>()
    for (const d of list) if (!m.has(d.len)) m.set(d.len, buildCurtain(d.len, ch))
    return m
  }, [list, ch])
  const linen = useMemo(() => tile(tex.linen, 3, 4), [tex])
  return (
    <group>
      {list.map((d, i) => {
        const s = d.side === 'left' ? -1 : 1
        const geo = geos.get(d.len)!
        const wall = d.mode === 'wall'
        const pos: [number, number, number] = wall
          ? [s * (W / 2 - 0.32), ch / 2 + 0.03, d.z]
          : [s * (W / 2 - d.len / 2), ch / 2 + 0.03, d.z]
        const rotY = wall ? -s * (Math.PI / 2) : 0
        return (
          <group key={i} position={pos} rotation={[0, rotY, 0]}>
            <mesh geometry={geo}>
              <meshStandardMaterial
                {...linen}
                color="#d8c39c"
                emissive="#5a3c1c"
                emissiveIntensity={0.35}
                roughness={0.95}
                transparent
                opacity={0.8}
                side={DoubleSide}
                depthWrite={false}
              />
            </mesh>
            <mesh position={[0, ch / 2 + 0.04, 0]} rotation={[0, 0, Math.PI / 2]}>
              <cylinderGeometry args={[0.015, 0.015, d.len + 0.1, 6]} />
              <meshStandardMaterial color={RAIL} metalness={0.6} roughness={0.4} />
            </mesh>
          </group>
        )
      })}
    </group>
  )
}

function Rails({ D }: { D: number }) {
  const rods = useMemo(() => {
    const out: number[] = []
    for (let z = -BAY / 2; z > -D; z -= BAY) out.push(z)
    return out
  }, [D])
  const len = D - 1
  return (
    <group>
      {[
        [-5.0, 4.45, 6.35],
        [5.0, 4.45, 6.35],
        [-1.6, 4.57, 6.05],
        [1.6, 4.57, 6.05],
      ].map(([x, y, top], k) => (
        <group key={k}>
          <mesh position={[x, y, -D / 2]}>
            <boxGeometry args={[0.05, 0.035, len]} />
            <meshStandardMaterial color={RAIL} metalness={0.6} roughness={0.4} />
          </mesh>
          {rods.map((z) => (
            <mesh key={z} position={[x, (y + top) / 2, z]}>
              <cylinderGeometry args={[0.006, 0.006, top - y, 4]} />
              <meshStandardMaterial color={RAIL} />
            </mesh>
          ))}
        </group>
      ))}
    </group>
  )
}

/**
 * The classical gallery shell: cream plaster walls with pilasters, two rows of
 * turned columns carrying a groin-vaulted ceiling, navy wool carpet, sheer
 * linen curtains and ceiling track rails whose spotlights pool warm light on
 * each exhibit. Image-based fill comes from a warm interior HDRI.
 */
export function Hall({ hall }: { hall: HallData }) {
  const { width: W, depth: D, height: H } = hall.room
  const tex = useMuseumTextures()
  const vault = useMemo(() => buildVault(W, D, H), [W, D, H])
  const ribs = useMemo(() => buildRibs(W, D, H), [W, D, H])
  const vaultTex = useMemo(() => tile(tex.plaster, 1, 1), [tex])
  const carpet = useMemo(() => tile(tex.carpet, W / 1.4, D / 1.4), [tex, W, D])
  const spots = useMemo(() => buildSpots(hall), [hall])

  return (
    <group>
      <Environment files="/hdri/ballroom_1k.hdr" environmentIntensity={0.42} />
      <hemisphereLight args={['#ffdcb0', '#1b2340', 0.18]} />

      {/* navy wool carpet */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, -D / 2]} receiveShadow>
        <planeGeometry args={[W, D]} />
        <meshStandardMaterial {...carpet} color={CARPET} roughness={1} normalScale={[0.7, 0.7]} />
      </mesh>

      <Walls tex={tex} W={W} D={D} H={H} />
      <Columns tex={tex} D={D} H={H} />

      {/* groin vaults + plaster ribs */}
      <mesh geometry={vault} receiveShadow>
        <meshStandardMaterial {...vaultTex} color={PLASTER} roughness={1} />
      </mesh>
      <mesh geometry={ribs}>
        <meshStandardMaterial {...vaultTex} color={PLASTER_DARK} />
      </mesh>

      <Curtains hall={hall} tex={tex} />
      <Rails D={D} />
      {spots.map((s, i) => (
        <TrackSpot key={i} spot={s} />
      ))}

      {/* soft warm wash on the vaults so the arches read when you look up */}
      {[-6, -18, -30, -42].map((z) => (
        <pointLight key={z} position={[0, H - 0.3, z]} intensity={5} distance={13} decay={1.6} color="#ffc98f" />
      ))}
    </group>
  )
}
