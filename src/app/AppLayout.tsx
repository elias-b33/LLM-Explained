import { useEffect, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
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

/** Visitor-information footer, like the bottom of a museum website. */
function SiteFooter() {
  return (
    <footer className="mx-auto max-w-6xl px-5 pt-6 pb-10 sm:px-8">
      <div className="ornament mb-6" aria-hidden="true">
        <span />
      </div>
      <div className="flex flex-col items-center gap-3 text-center sm:flex-row sm:justify-between sm:text-left">
        <div>
          <p className="font-display text-lg text-fg">LLM Explainer</p>
          <p className="text-xs text-fg-subtle">A museum of how language models work · Admission free · Open at all hours</p>
        </div>
        <nav aria-label="Visitor links" className="flex flex-wrap justify-center gap-x-5 gap-y-1 text-xs">
          <Link to="/" className="exhibit-label hover:text-fg">Main hall</Link>
          <Link to="/museum/orientation" className="exhibit-label hover:text-fg">3D museum</Link>
          <Link to="/galaxy" className="exhibit-label hover:text-fg">Planetarium</Link>
        </nav>
      </div>
    </footer>
  )
}

export function AppLayout() {
  const [mobileOpen, setMobileOpen] = useState(false)

  return (
    <div className="min-h-screen">
      {/* Calm gallery wall: plaster grain + one warm overhead light. Static, no animation. */}
      <div className="gallery-backdrop" aria-hidden="true" />
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-2 focus:z-[60] focus:border focus:border-brass focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:text-fg"
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
        <SiteFooter />
      </div>
    </div>
  )
}
