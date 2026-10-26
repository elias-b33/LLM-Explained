import { useEffect, useMemo, useRef } from 'react'
import { useThree, type ThreeEvent } from '@react-three/fiber'
import { CanvasTexture, Color, ShaderMaterial, UniformsLib, UniformsUtils, type Points } from 'three'
import type { GalaxyData } from './types'

/**
 * Among every point the ray passes near, pick the one whose centre is closest to
 * the cursor line (distanceToRay), with the nearer-to-camera point breaking ties.
 * This selects "the dot you're pointing at" instead of three's default (whichever
 * is nearest the camera), which made dense interiors almost impossible to target.
 */
function pickBest(e: ThreeEvent<PointerEvent | MouseEvent>): number | null {
  const hits = e.intersections
  if (!hits || hits.length === 0) return e.index ?? null
  let bestIdx: number | null = null
  let bestScore = Infinity
  for (const h of hits) {
    if (h.index == null) continue
    const ray = h.distanceToRay ?? 0
    const score = ray * 2000 + h.distance
    if (score < bestScore) {
      bestScore = score
      bestIdx = h.index
    }
  }
  return bestIdx ?? e.index ?? null
}

/** A crisper star sprite (tight bright core) so dense clusters read as distinct dots. */
function makeStarTexture(): CanvasTexture {
  const size = 64
  const canvas = document.createElement('canvas')
  canvas.width = canvas.height = size
  const ctx = canvas.getContext('2d')!
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2)
  g.addColorStop(0, 'rgba(255,255,255,1)')
  g.addColorStop(0.22, 'rgba(255,255,255,0.92)')
  g.addColorStop(0.5, 'rgba(255,255,255,0.22)')
  g.addColorStop(1, 'rgba(255,255,255,0)')
  ctx.fillStyle = g
  ctx.fillRect(0, 0, size, size)
  return new CanvasTexture(canvas)
}

export function GalaxyPoints({
  data,
  activeUniverse,
  onHover,
  onSelect,
  size = 0.62,
}: {
  data: GalaxyData
  activeUniverse: number | null
  onHover: (i: number | null) => void
  onSelect: (i: number) => void
  size?: number
}) {
  const sprite = useMemo(() => makeStarTexture(), [])
  const height = useThree((s) => s.size.height)
  const dpr = useThree((s) => s.viewport.dpr)
  const material = useMemo(() => makeStarMaterial(sprite), [sprite])
  const points = useRef<Points>(null)
  // Match three's own sizeAttenuation scale (half the drawing-buffer height).
  // Written through the mounted object (the attached material), not the memo value.
  useEffect(() => {
    const m = points.current?.material as ShaderMaterial | undefined
    if (m) m.uniforms.uScale.value = (height * dpr) / 2
  }, [material, height, dpr])
  useEffect(() => () => material.dispose(), [material])

  const { positions, colors, sizes } = useMemo(() => {
    const n = data.words.length
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    const sizes = new Float32Array(n)
    const clusterColors = data.clusters.map((c) => new Color(c.color))
    const dimmed = new Color('#141a24')
    const fallback = new Color('#ffffff')
    const white = new Color('#ffffff')
    const tmp = new Color()
    data.words.forEach((w, i) => {
      positions[i * 3] = w.p[0]
      positions[i * 3 + 1] = w.p[1]
      positions[i * 3 + 2] = w.p[2]
      const isDim = activeUniverse != null && w.c !== activeUniverse
      // Per-star hash → brightness + size, so clusters read as a real starfield:
      // mostly small dim stars, some bright ones, and the odd white-hot giant.
      const h = Math.sin((i + 1) * 12.9898) * 43758.5453
      const r = h - Math.floor(h)
      const giant = !isDim && r > 0.985
      tmp.copy(isDim ? dimmed : (clusterColors[w.c] ?? fallback))
      if (giant) tmp.lerp(white, 0.55)
      const b = isDim ? 1 : 0.4 + 0.6 * r
      colors[i * 3] = tmp.r * b
      colors[i * 3 + 1] = tmp.g * b
      colors[i * 3 + 2] = tmp.b * b
      sizes[i] = size * (isDim ? 0.7 : (0.55 + 0.95 * r * r) * (giant ? 1.8 : 1))
    })
    return { positions, colors, sizes }
  }, [data, activeUniverse, size])

  return (
    <points
      ref={points}
      onPointerMove={(e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation()
        const i = pickBest(e)
        if (i != null) {
          onHover(i)
          document.body.style.cursor = 'pointer'
        }
      }}
      onPointerOut={() => {
        onHover(null)
        document.body.style.cursor = 'auto'
      }}
      onClick={(e: ThreeEvent<MouseEvent>) => {
        e.stopPropagation()
        const i = pickBest(e)
        if (i != null) onSelect(i)
      }}
    >
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        <bufferAttribute attach="attributes-size" args={[sizes, 1]} />
      </bufferGeometry>
      <primitive object={material} attach="material" />
    </points>
  )
}

/**
 * pointsMaterial can't vary size per point, so this is a tiny equivalent shader:
 * a per-vertex `size` attribute with three's size attenuation, the star sprite,
 * vertex colours and scene fog. Raycasting is unaffected (it uses geometry).
 */
function makeStarMaterial(map: CanvasTexture): ShaderMaterial {
  return new ShaderMaterial({
    uniforms: {
      ...UniformsUtils.clone(UniformsLib.fog),
      uMap: { value: map },
      uScale: { value: 400 },
    },
    vertexColors: true,
    transparent: true,
    depthWrite: false,
    fog: true,
    vertexShader: /* glsl */ `
      attribute float size;
      uniform float uScale;
      varying vec3 vColor;
      #include <fog_pars_vertex>
      void main() {
        vColor = color;
        vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
        gl_PointSize = size * (uScale / -mvPosition.z);
        gl_Position = projectionMatrix * mvPosition;
        #include <fog_vertex>
      }
    `,
    fragmentShader: /* glsl */ `
      uniform sampler2D uMap;
      varying vec3 vColor;
      #include <fog_pars_fragment>
      void main() {
        vec4 t = texture2D(uMap, gl_PointCoord);
        if (t.a < 0.02) discard;
        gl_FragColor = vec4(vColor, t.a * 0.95);
        #include <tonemapping_fragment>
        #include <colorspace_fragment>
        #include <fog_fragment>
      }
    `,
  })
}
