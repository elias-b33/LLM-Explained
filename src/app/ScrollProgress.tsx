import { motion, useScroll, useSpring } from 'motion/react'

/** Thin brass rail pinned to the top that fills as you walk down the page. */
export function ScrollProgress() {
  const { scrollYProgress } = useScroll()
  const scaleX = useSpring(scrollYProgress, { stiffness: 120, damping: 30, mass: 0.3 })

  return (
    <motion.div
      style={{
        scaleX,
        backgroundImage: 'linear-gradient(to right, var(--brass), var(--gold) 70%, var(--accent))',
      }}
      className="fixed inset-x-0 top-0 z-50 h-[2px] origin-left"
      aria-hidden="true"
    />
  )
}
