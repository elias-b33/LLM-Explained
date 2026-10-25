import { Link } from 'react-router-dom'
import { cn } from '@/lib/cn'
import { SparklesIcon } from './icons'

/** Entry point to the fullscreen embedding galaxy. */
export function ImmersiveLink({ className, label = 'Explore the word universes' }: { className?: string; label?: string }) {
  return (
    <Link
      to="/galaxy"
      className={cn(
        'group inline-flex items-center gap-2 rounded-xl px-5 py-3 font-medium text-white shadow-lg transition-transform hover:-translate-y-0.5',
        className,
      )}
      style={{
        backgroundImage: 'linear-gradient(120deg, var(--accent), var(--accent-2))',
        boxShadow: '0 10px 30px -10px var(--accent)',
      }}
    >
      <SparklesIcon size={18} />
      {label}
      <span className="transition-transform group-hover:translate-x-0.5">→</span>
    </Link>
  )
}
