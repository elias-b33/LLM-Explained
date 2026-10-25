import { useEffect, useState, type RefObject } from 'react'

/**
 * Returns an R3F `frameloop` value that is 'always' only while the element (a Canvas,
 * observed via its parent container) intersects the viewport AND the tab is visible.
 * Off-screen / hidden-tab canvases switch to 'never' so they stop drawing entirely.
 */
export function useOnScreenFrameloop(ref: RefObject<HTMLElement | null>): 'always' | 'never' {
  const [onScreen, setOnScreen] = useState(true)
  const [tabVisible, setTabVisible] = useState(() => typeof document === 'undefined' || !document.hidden)

  useEffect(() => {
    const el = ref.current?.parentElement ?? ref.current
    if (!el || typeof IntersectionObserver === 'undefined') return
    const obs = new IntersectionObserver(
      (entries) => {
        const e = entries[entries.length - 1]
        if (e) setOnScreen(e.isIntersecting)
      },
      { rootMargin: '64px' },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [ref])

  useEffect(() => {
    const onVis = () => setTabVisible(!document.hidden)
    document.addEventListener('visibilitychange', onVis)
    return () => document.removeEventListener('visibilitychange', onVis)
  }, [])

  return onScreen && tabVisible ? 'always' : 'never'
}
