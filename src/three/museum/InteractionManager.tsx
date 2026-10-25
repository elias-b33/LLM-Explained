import { useEffect, useMemo, useRef } from 'react'
import { useFrame, useThree } from '@react-three/fiber'
import {
  AdditiveBlending,
  Raycaster,
  Vector2,
  Vector3,
  type Camera,
  type Group,
  type Mesh,
  type Object3D,
  type ShaderMaterial,
} from 'three'
import { consumeClick, keys } from './input'
import { isExplored, museumUi, registry, setPrompt, type Interactable } from './interaction'
import { sfx } from './audio'
import type { TourStop } from './halls'

const CENTER = new Vector2(0, 0)
const WP = new Vector3()

type Target = { it: { current: Interactable }; hit: Object3D }

/** Nearest registered interactable under the screen centre, within its reach. */
function pick(ray: Raycaster, cam: Camera): Target | null {
  ray.setFromCamera(CENTER, cam)
  let best = Infinity
  let next: Target | null = null
  for (const entry of registry) {
    const it = entry.current
    const reach = it.reach ?? 2.5
    for (const t of it.targets) {
      const obj = t.current
      if (!obj) continue
      obj.getWorldPosition(WP)
      if (WP.distanceTo(cam.position) > reach + 4) continue
      const h = ray.intersectObject(obj, true)[0]
      if (h && h.distance <= reach && h.distance < best) {
        best = h.distance
        next = { it: entry, hit: h.object }
      }
    }
  }
  return next
}

/**
 * The one place that decides what the player is looking at. Each frame it
 * raycasts from the screen centre against every registered exhibit near the
 * camera, highlights the nearest hit within that exhibit's reach, publishes the
 * "E · verb" prompt, and fires the exhibit on an E press / locked click
 * (edge-triggered). Also plays footstep ticks while walking.
 */
export function InteractionManager({ enabled }: { enabled: boolean }) {
  const ray = useMemo(() => {
    const r = new Raycaster()
    r.far = 8
    return r
  }, [])
  const current = useRef<Target | null>(null)
  const camera = useThree((st) => st.camera)
  const wasDown = useRef(false)
  const lastPos = useRef(new Vector3(NaN, 0, 0))
  const walked = useRef(0)

  // Dev-only hook: trigger whatever is under the crosshair (pointer lock is
  // unavailable in automated previews).
  useEffect(() => {
    if (!import.meta.env.DEV) return
    const w = window as unknown as { __museumInteract?: () => string | null }
    w.__museumInteract = () => {
      // re-pick now: background preview tabs may not have rendered a frame since the camera moved
      const c = pick(ray, camera)
      if (!c) return null
      const v = c.it.current.verb(c.hit)
      c.it.current.onInteract(c.hit)
      return v
    }
    return () => {
      delete w.__museumInteract
    }
  }, [ray, camera])

  useEffect(
    () => () => {
      current.current?.it.current.setHighlight?.(null)
      setPrompt(null)
    },
    [],
  )

  useFrame((state) => {
    const cam = state.camera

    // footsteps: one soft tick every ~0.8 m walked (ignore teleports)
    const lp = lastPos.current
    if (!Number.isNaN(lp.x)) {
      const d = Math.hypot(cam.position.x - lp.x, cam.position.z - lp.z)
      if (d < 0.5) {
        walked.current += d
        if (walked.current > 0.8) {
          walked.current = 0
          sfx('step', { v: 0.8 + Math.random() * 0.4 })
        }
      }
    }
    lp.copy(cam.position)

    const next = enabled ? pick(ray, cam) : null

    const prev = current.current
    if (prev?.it !== next?.it || prev?.hit !== next?.hit) {
      if (prev && prev.it !== next?.it) prev.it.current.setHighlight?.(null)
      if (next) next.it.current.setHighlight?.(next.hit)
      current.current = next
    }
    setPrompt(next ? next.it.current.verb(next.hit) : null)

    const down = keys.interact
    const clicked = consumeClick()
    if (next && enabled && ((down && !wasDown.current) || clicked)) next.it.current.onInteract(next.hit)
    wasDown.current = down
  })

  return null
}

const GUIDE_VERT = /* glsl */ `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }
`
const GUIDE_FRAG = /* glsl */ `
uniform float uTime;
uniform float uLen;
uniform float uFade;
varying vec2 vUv;
void main() {
  float d = vUv.x * uLen;
  float dash = smoothstep(0.0, 0.08, fract(d * 1.4 - uTime * 0.9)) * smoothstep(0.55, 0.45, fract(d * 1.4 - uTime * 0.9));
  float ends = smoothstep(0.0, 0.25, vUv.x) * smoothstep(1.0, 0.8, vUv.x);
  float edge = smoothstep(0.0, 0.35, vUv.y) * smoothstep(1.0, 0.65, vUv.y);
  vec3 col = mix(vec3(0.545, 0.482, 1.0), vec3(0.169, 0.827, 0.878), vUv.x);
  gl_FragColor = vec4(col * 0.7, dash * ends * edge * 0.3 * uFade);
}
`

/**
 * A faint animated dash line on the floor from the player toward the next
 * exhibit they haven't tried yet. One mesh, updated in place; hidden when
 * close or once everything is explored.
 */
export function GuideStrip({ tour }: { tour: TourStop[] }) {
  const g = useRef<Group>(null)
  const mesh = useRef<Mesh>(null)
  const uniforms = useMemo(() => ({ uTime: { value: 0 }, uLen: { value: 1 }, uFade: { value: 0 } }), [])
  const nextStop = useRef<TourStop | null>(tour[0] ?? null)
  useEffect(() => {
    const update = () => {
      nextStop.current = tour.find((s) => !isExplored(s.id)) ?? null
    }
    update()
    return museumUi.subscribe(update)
  }, [tour])

  useFrame((state, dt) => {
    const grp = g.current
    if (!grp || !mesh.current) return
    // R3F may hand the material its own uniforms object: always write through the material.
    const uniforms = (mesh.current.material as ShaderMaterial).uniforms
    const stop = nextStop.current
    const cam = state.camera.position
    let want = 0
    if (stop) {
      const dx = stop.x - cam.x
      const dz = stop.z - cam.z
      const dist = Math.hypot(dx, dz)
      if (dist > 2.6) {
        want = 1
        const ux = dx / dist
        const uz = dz / dist
        const start = 0.9
        const len = dist - start - 0.6
        grp.position.set(cam.x + ux * (start + len / 2), 0.012, cam.z + uz * (start + len / 2))
        grp.rotation.y = Math.atan2(-uz, ux)
        grp.scale.set(len, 1, 0.14)
        uniforms.uLen.value = len
      }
    }
    const f = uniforms.uFade.value
    uniforms.uFade.value = Math.min(1, Math.max(0, f + (want - f) * Math.min(1, Math.max(0, dt) * 3)))
    grp.visible = uniforms.uFade.value > 0.01
    uniforms.uTime.value = state.clock.elapsedTime
  })

  return (
    <group ref={g} visible={false}>
      <mesh ref={mesh} rotation={[-Math.PI / 2, 0, 0]} renderOrder={1}>
        <planeGeometry args={[1, 1]} />
        <shaderMaterial
          vertexShader={GUIDE_VERT}
          fragmentShader={GUIDE_FRAG}
          uniforms={uniforms}
          transparent
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </mesh>
    </group>
  )
}
