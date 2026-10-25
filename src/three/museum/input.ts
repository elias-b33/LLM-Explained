/**
 * A tiny module-level keyboard state shared by the first-person controller and
 * the stations (there's only ever one hall mounted). Avoids drei's
 * KeyboardControls, which is a DOM/context helper and misbehaves inside a Canvas.
 */
export const keys = {
  forward: false,
  back: false,
  left: false,
  right: false,
  interact: false,
}

const MAP: Record<string, keyof typeof keys> = {
  KeyW: 'forward',
  ArrowUp: 'forward',
  KeyS: 'back',
  ArrowDown: 'back',
  KeyA: 'left',
  ArrowLeft: 'left',
  KeyD: 'right',
  ArrowRight: 'right',
  KeyE: 'interact',
}

let count = 0

/** Install window key listeners (ref-counted). Returns a cleanup. */
export function installInput(): () => void {
  count += 1
  if (count === 1) {
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
  }
  return () => {
    count -= 1
    if (count === 0) {
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
      for (const k in keys) keys[k as keyof typeof keys] = false
    }
  }
}

function onDown(e: KeyboardEvent) {
  const k = MAP[e.code]
  if (k) keys[k] = true
}
function onUp(e: KeyboardEvent) {
  const k = MAP[e.code]
  if (k) keys[k] = false
}
