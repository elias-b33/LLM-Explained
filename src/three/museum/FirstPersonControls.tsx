import { useEffect, type RefObject } from 'react'
import { useFrame } from '@react-three/fiber'
import { PointerLockControls } from '@react-three/drei'
import { Vector3 } from 'three'
import type { PointerLockControls as PointerLockControlsImpl } from 'three-stdlib'
import { installInput, keys } from './input'
import type { Colliders } from './layout'

export type ControlsRef = RefObject<PointerLockControlsImpl | null>

const UP = new Vector3(0, 1, 0)
const FWD = new Vector3()
const RIGHT = new Vector3()
const MOVE = new Vector3()
const EYE = 1.6 // eye height (m)
const SPEED = 3.6 // m/s
const MARGIN = 0.7 // keep this far off the walls
const BODY = 0.3 // player radius against columns / vitrines

/**
 * First-person walk: drei PointerLockControls for mouse-look + a shared keyboard
 * state for WASD, with horizontal movement (looking up/down doesn't fly), a fixed
 * eye height, an AABB clamp to the room interior and a simple push-out from
 * columns (circles) and vitrines/partitions (boxes) — no physics engine.
 */
export function FirstPersonControls({
  controlsRef,
  room,
  colliders,
  focusOpen,
  onLock,
  onUnlock,
}: {
  controlsRef: ControlsRef
  room: { width: number; depth: number; height: number }
  colliders?: Colliders
  focusOpen: boolean
  onLock: () => void
  onUnlock: () => void
}) {
  useEffect(() => installInput(), [])

  useFrame((state, delta) => {
    const c = controlsRef.current
    if (!c || !c.isLocked || focusOpen) return
    const cam = state.camera

    cam.getWorldDirection(FWD)
    FWD.y = 0
    if (FWD.lengthSq() < 1e-6) return
    FWD.normalize()
    RIGHT.crossVectors(FWD, UP).normalize()

    MOVE.set(0, 0, 0)
    if (keys.forward) MOVE.add(FWD)
    if (keys.back) MOVE.sub(FWD)
    if (keys.right) MOVE.add(RIGHT)
    if (keys.left) MOVE.sub(RIGHT)
    if (MOVE.lengthSq() > 0) {
      MOVE.normalize().multiplyScalar(SPEED * Math.min(delta, 0.05))
      cam.position.add(MOVE)
    }

    if (colliders) {
      const p = cam.position
      for (const [cx, cz, r] of colliders.circles) {
        const dx = p.x - cx
        const dz = p.z - cz
        const R = r + BODY
        const d2 = dx * dx + dz * dz
        if (d2 < R * R && d2 > 1e-8) {
          const d = Math.sqrt(d2)
          p.x = cx + (dx / d) * R
          p.z = cz + (dz / d) * R
        }
      }
      for (const [x0, z0, x1, z1] of colliders.boxes) {
        const ax = x0 - BODY
        const bx = x1 + BODY
        const az = z0 - BODY
        const bz = z1 + BODY
        if (p.x > ax && p.x < bx && p.z > az && p.z < bz) {
          // push out along the shallowest axis
          const px = Math.min(p.x - ax, bx - p.x)
          const pz = Math.min(p.z - az, bz - p.z)
          if (px < pz) p.x = p.x - ax < bx - p.x ? ax : bx
          else p.z = p.z - az < bz - p.z ? az : bz
        }
      }
    }

    const hw = room.width / 2 - MARGIN
    cam.position.x = Math.max(-hw, Math.min(hw, cam.position.x))
    cam.position.z = Math.max(-room.depth + MARGIN, Math.min(-MARGIN, cam.position.z))
    cam.position.y = EYE
  })

  return <PointerLockControls ref={controlsRef} makeDefault onLock={onLock} onUnlock={onUnlock} />
}
