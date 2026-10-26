import { Suspense, useEffect, useMemo, useState } from 'react'
import { Canvas, useThree } from '@react-three/fiber'
import { EffectComposer, Bloom, ToneMapping, Vignette } from '@react-three/postprocessing'
import { ToneMappingMode } from 'postprocessing'
import { makeDotTexture } from '@/three/galaxy/textures'
import { Hall } from './Hall'
import { WallPanel } from './WallPanel'
import { Station } from './Station'
import { Doorway } from './Doorway'
import { FirstPersonControls, type ControlsRef } from './FirstPersonControls'
import { DecorVitrine, DisplayTable } from './Exhibits'
import { useMuseumTextures } from './materials'
import { buildColliders } from './layout'
import { InteractionManager, GuideStrip } from './InteractionManager'
import { TokenSlicer } from './exhibits/TokenSlicer'
import { AttentionWalk } from './exhibits/AttentionWalk'
import { EmbeddingAlcove } from './exhibits/EmbeddingAlcove'
import { AutoregressMachine } from './exhibits/AutoregressMachine'
import { QUALITY_PROFILES, setActiveQuality, type Quality } from './quality'
import type { Hall as HallData, FocusTarget } from './halls'

/** Set dressing + in-world exhibits from the hall data. */
function Decor({ hall, sprite }: { hall: HallData; sprite: ReturnType<typeof makeDotTexture> }) {
  const tex = useMuseumTextures()
  return (
    <>
      {(hall.decor ?? []).map((d, i) => {
        if (d.kind === 'vitrine')
          return (
            <DecorVitrine
              key={i}
              tex={tex}
              sprite={sprite}
              position={[d.x, 0, d.z]}
              exhibit={d.exhibit}
              caption={d.caption}
            />
          )
        if (d.kind === 'table') return <DisplayTable key={i} tex={tex} z0={d.z0} z1={d.z1} />
        if (d.kind === 'slicer') return <TokenSlicer key={i} tex={tex} position={[d.x, 0, d.z]} />
        if (d.kind === 'embedding') return <EmbeddingAlcove key={i} tex={tex} position={[d.x, 0, d.z]} />
        if (d.kind === 'attention') return <AttentionWalk key={i} z={d.z} />
        if (d.kind === 'machine') return <AutoregressMachine key={i} tex={tex} z={d.z} consoleZ={d.consoleZ} />
        return null
      })}
    </>
  )
}

/**
 * In on-demand mode the scene renders once on mount — before the lazy panels'
 * refs attach — so the distance-fade never runs. Nudge a few frames after mount
 * so the parked entrance view settles to the correct per-panel opacity.
 */
/**
 * Rendered as the in-canvas Suspense fallback while textures / the HDRI load.
 * Suspending must stay inside the R3F tree: letting it bubble to a DOM
 * Suspense would hide the Canvas, tear down its root and lose the GL context.
 */
function LoadingFlag({ onChange }: { onChange: (loading: boolean) => void }) {
  useEffect(() => {
    onChange(true)
    return () => onChange(false)
  }, [onChange])
  return null
}

function SettleRenders() {
  const invalidate = useThree((s) => s.invalidate)
  const get = useThree((s) => s.get)
  useEffect(() => {
    // Lights and shadow casters never move, so shadows are baked: the shadow map
    // auto-update is off (see onCreated) and re-rendered only here, once the
    // suspended hall has mounted, instead of on every frame.
    const settle = () => {
      get().gl.shadowMap.needsUpdate = true
      invalidate()
    }
    settle()
    const t1 = setTimeout(settle, 160)
    const t2 = setTimeout(settle, 480)
    return () => {
      clearTimeout(t1)
      clearTimeout(t2)
    }
  }, [invalidate, get])
  return null
}

/**
 * The fullscreen museum canvas. All cross-canvas state is owned by MuseumPage
 * and passed in as props (React context does not cross the R3F boundary).
 */
export default function HallScene({
  hall,
  controlsRef,
  playing,
  focusOpen,
  onLock,
  onUnlock,
  onOpenStation,
  onTravel,
  nextLabel,
  prevLabel,
  quality,
}: {
  hall: HallData
  controlsRef: ControlsRef
  playing: boolean
  focusOpen: boolean
  onLock: () => void
  onUnlock: () => void
  onOpenStation: (target: FocusTarget) => void
  onTravel: (dir: 'next' | 'prev') => void
  nextLabel?: string
  prevLabel?: string
  quality: Quality
}) {
  // Module-level profile read by textures/boards while the (keyed) Canvas mounts.
  setActiveQuality(quality)
  const profile = QUALITY_PROFILES[quality]
  const sprite = useMemo(() => makeDotTexture(), [])
  const colliders = useMemo(() => buildColliders(hall), [hall])
  const [loading, setLoading] = useState(false)
  // The lazy canvas can mount before layout settles — nudge a resize after paint.
  useEffect(() => {
    const raf = requestAnimationFrame(() => window.dispatchEvent(new Event('resize')))
    const t = setTimeout(() => window.dispatchEvent(new Event('resize')), 150)
    return () => {
      cancelAnimationFrame(raf)
      clearTimeout(t)
    }
  }, [])

  return (
    <>
      <Canvas
        key={quality}
        frameloop={playing ? 'always' : 'demand'}
        camera={{ position: [0, 1.6, -1], fov: 70 }}
        gl={{
          alpha: false,
          // The EffectComposer renders into its own MSAA target, so the default
          // framebuffer's antialiasing would be wasted memory/bandwidth.
          antialias: false,
          powerPreference: 'high-performance',
          preserveDrawingBuffer: import.meta.env.DEV,
        }}
        dpr={profile.dpr}
        shadows="percentage"
        onCreated={(state) => {
          state.gl.shadowMap.autoUpdate = false
          // R3F aims the default camera at the scene origin; face it down the hall instead.
          state.camera.rotation.set(0, 0, 0)
          state.camera.updateMatrixWorld()
          // Dev-only handle for framing verification screenshots.
          if (import.meta.env.DEV) (window as unknown as { __museum?: unknown }).__museum = state
          state.invalidate()
        }}
      >
        <color attach="background" args={['#1a120b']} />
        <fog attach="fog" args={['#22170e', 16, 70]} />
        <Suspense fallback={<LoadingFlag onChange={setLoading} />}>
          <Hall hall={hall} />
          <Decor hall={hall} sprite={sprite} />
          <SettleRenders />
          {hall.segments.map((seg, i, all) => {
            if (seg.kind === 'panel') {
              const W = hall.room.width
              const D = hall.room.depth
              const y = Math.max(1.75, (seg.h ?? 2.5) / 2 + 0.45)
              let pos: [number, number, number]
              let rotY: number
              if (seg.side === 'left') {
                pos = [-W / 2 + 0.14, y, seg.z]
                rotY = Math.PI / 2
              } else if (seg.side === 'right') {
                pos = [W / 2 - 0.14, y, seg.z]
                rotY = -Math.PI / 2
              } else {
                pos = [0, y, -D + 0.14]
                rotY = 0
              }
              return (
                <WallPanel
                  key={i}
                  position={pos}
                  rotationY={rotY}
                  width={seg.w ?? 3.2}
                  height={seg.h}
                  accent={hall.accent}
                >
                  {seg.content}
                </WallPanel>
              )
            }
            if (seg.kind === 'station') {
              return (
                <Station
                  key={i}
                  position={[seg.x ?? 0, 0, seg.z]}
                  title={seg.title}
                  accent={hall.accent}
                  accent2={hall.accent2}
                  sprite={sprite}
                  variant={all.slice(0, i).filter((s) => s.kind === 'station').length % 2 ? 'network' : 'tokens'}
                  onOpen={() =>
                    onOpenStation({
                      demoKey: seg.demoKey,
                      title: seg.title,
                      status: seg.status,
                    })
                  }
                />
              )
            }
            if (seg.kind === 'doorway') {
              const label = seg.dir === 'next' ? nextLabel : prevLabel
              if (!label) return null
              return (
                <Doorway
                  key={i}
                  position={[0, 0, seg.z]}
                  label={label}
                  accent={hall.accent}
                  accent2={hall.accent2}
                  sprite={sprite}
                  onTravel={() => onTravel(seg.dir)}
                />
              )
            }
            return null
          })}
          <GuideStrip tour={hall.tour ?? []} />
        </Suspense>
        <InteractionManager enabled={!focusOpen} />
        <FirstPersonControls
          controlsRef={controlsRef}
          room={hall.room}
          colliders={colliders}
          focusOpen={focusOpen}
          onLock={onLock}
          onUnlock={onUnlock}
        />
        <EffectComposer multisampling={profile.msaa}>
          <Bloom intensity={0.55} luminanceThreshold={0.92} luminanceSmoothing={0.25} mipmapBlur />
          <ToneMapping mode={ToneMappingMode.ACES_FILMIC} />
          <Vignette eskil={false} offset={0.3} darkness={0.55} />
        </EffectComposer>
      </Canvas>
      {loading && (
        <div className="dark pointer-events-none absolute inset-0 grid place-items-center" style={{ background: '#140e09' }}>
          <p className="exhibit-label">Entering the exhibit…</p>
        </div>
      )}
    </>
  )
}
