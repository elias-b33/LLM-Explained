import { useEffect, useRef, useState } from 'react'

/**
 * Returns [ref, inView]. `inView` flips to true once when the element first
 * enters the viewport, then stops observing. Used for entrance animations and
 * marking lessons as read.
 */
export function useInViewOnce<T extends Element = HTMLDivElement>() {
  const ref = useRef<T | null>(null)
  const [inView, setInView] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el) return
    const obs = new IntersectionObserver(
      (entries) => {
        if (entries[0]?.isIntersecting) {
          setInView(true)
          obs.disconnect()
        }
      },
      { threshold: 0.12, rootMargin: '0px 0px -8% 0px' },
    )
    obs.observe(el)
    return () => obs.disconnect()
  }, [])

  return [ref, inView] as const
}
