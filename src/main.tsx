import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { RouterProvider } from 'react-router-dom'
import { MotionConfig } from 'motion/react'

// Self-hosted fonts (offline-friendly).
import '@fontsource/inter/400.css'
import '@fontsource/inter/500.css'
import '@fontsource/inter/600.css'
import '@fontsource/inter/700.css'
import '@fontsource/jetbrains-mono/400.css'
import '@fontsource/jetbrains-mono/500.css'

// Math typesetting + design system.
import 'katex/dist/katex.min.css'
import './index.css'

import { ThemeProvider } from '@/theme/ThemeProvider'
import { ProgressProvider } from '@/progress/ProgressProvider'
import { router } from '@/router'

// After a redeploy, a tab still running the old bundle asks for chunk hashes that
// no longer exist. Reload once to pick up the new index.html instead of crashing.
window.addEventListener('vite:preloadError', (e) => {
  try {
    const last = Number(sessionStorage.getItem('chunk-reload') ?? 0)
    if (Date.now() - last < 10_000) return
    sessionStorage.setItem('chunk-reload', String(Date.now()))
  } catch {
    return // no storage means no loop guard, so let the error boundary handle it
  }
  e.preventDefault()
  window.location.reload()
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <ProgressProvider>
        <MotionConfig reducedMotion="user">
          <RouterProvider router={router} />
        </MotionConfig>
      </ProgressProvider>
    </ThemeProvider>
  </StrictMode>,
)
