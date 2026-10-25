import { motion, useScroll, useSpring } from 'motion/react'

/** Thin gradient bar pinned to the top that fills as you scroll the page. */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.3 })

  return (
    <motion.div
      style={{
        scaleX,
        backgroundImage: 'linear-gradient(to right, var(--accent), var(--accent-2))',
      }}
      className="fixed inset-x-0 top-0 z-50 h-[3px] origin-left"
      aria-hidden="true"
    />
  )
}
