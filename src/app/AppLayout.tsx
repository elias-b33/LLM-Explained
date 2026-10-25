import { useEffect, useState } from 'react'
import { Outlet, useLocation } from 'react-router-dom'
import { Sidebar } from './Sidebar'
import { TopBar } from './TopBar'
import { ScrollProgress } from './ScrollProgress'

/** Scrolls to the top whenever the route changes. */
function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' })
  }, [pathname])
  return null
}

export function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen">
      <div className="site-backdrop" aria-hidden="true">
        <div className="blob a" />
        <div className="blob b" />
        <div className="blob c" />
      </div>
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-[60] focus:rounded-lg focus:border focus:border-border focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:text-fg"
      >
        Skip to content
      </a>
      <ScrollProgress />
      <Sidebar mobileOpen={mobileOpen} onClose={() => setMobileOpen(false)} />
      <div className="md:pl-72">
        <TopBar onMenu={() => setMobileOpen(true)} />
        <ScrollToTop />
        <main id="main-content" tabIndex={-1} className="outline-none">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
