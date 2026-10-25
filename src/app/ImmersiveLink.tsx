import { Link } from 'react-router-dom'
import { cn } from '@/lib/cn'
import { ArrowRightIcon, PlanetIcon } from './icons'

/** Entry point to the fullscreen embedding galaxy — the museum's planetarium. */
export function ImmersiveLink({ className, label = 'Explore the word universes' }: { className?: string; label?: string }) {
  return (
    <Link to="/galaxy" className={cn('btn-museum group', className)}>
      <PlanetIcon size={17} className="shrink-0" />
      {label}
      <span className="tag hidden opacity-70 sm:inline-flex">Planetarium</span>
      <ArrowRightIcon size={16} className="transition-transform group-hover:translate-x-0.5" />
    </Link>
  )
}
