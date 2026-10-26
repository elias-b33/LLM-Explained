import { useEffect, useLayoutEffect, useRef, type RefObject } from 'react'
import type { Object3D } from 'three'

/**
 * Look-at-and-press-E interaction, shared by every in-world exhibit.
 *
 * Exhibits register an Interactable (hit meshes + callbacks) with
 * `useInteractable`; the single InteractionManager inside the Canvas raycasts
 * from the screen centre each frame, picks the nearest hit within that
 * object's reach, toggles its highlight and fires it on E / click.
 *
 * The DOM HUD lives outside the Canvas, so the bits it shows (the "E · verb"
 * prompt and which exhibits have been explored) are published through a tiny
 * external store that changes only when its values change, never per frame.
 */
export interface Interactable {
  /** Hit meshes (raycast recursively; invisible materials are fine). */
  targets: RefObject<Object3D | null>[]
  /** Max distance from the eye to the hit point (m). Default 2.5. */
  reach?: number
  /** Short action shown as "E · verb"; receives the mesh under the crosshair. */
  verb: (hit: Object3D) => string
  onInteract: (hit: Object3D) => void
  /** Called when targeting starts/stops or moves to another of its meshes. */
  setHighlight?: (hit: Object3D | null) => void
}

export const registry = new Set<{ current: Interactable }>()

/** Register an interactable for the lifetime of the component. The definition
 *  is re-read every frame, so it may close over fresh props/state. */
export function useInteractable(def: Interactable) {
  const ref = useRef(def)
  // Keep the latest definition after each commit (before the next frame reads it).
  useLayoutEffect(() => {
    ref.current = def
  })
  useEffect(() => {
    const entry = ref
    registry.add(entry)
    return () => {
      registry.delete(entry)
    }
  }, [])
}

/* ------------------------------------------------------------------------ */
/* DOM-facing store                                                          */
/* ------------------------------------------------------------------------ */

export interface MuseumUiState {
  prompt: string | null
  explored: readonly string[]
}

let state: MuseumUiState = { prompt: null, explored: [] }
const listeners = new Set<() => void>()

function set(next: Partial<MuseumUiState>) {
  state = { ...state, ...next }
  listeners.forEach((l) => l())
}

export const museumUi = {
  subscribe(l: () => void) {
    listeners.add(l)
    return () => {
      listeners.delete(l)
    }
  },
  get: () => state,
}

export function setPrompt(prompt: string | null) {
  if (prompt !== state.prompt) set({ prompt })
}

/** Tick an exhibit off the "explored" tracker (first interaction only). */
export function markExplored(id: string) {
  if (!state.explored.includes(id)) set({ explored: [...state.explored, id] })
}

export function resetExplored() {
  set({ explored: [], prompt: null })
}

export function isExplored(id: string) {
  return state.explored.includes(id)
}
