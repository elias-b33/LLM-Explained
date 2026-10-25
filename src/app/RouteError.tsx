import { useRouteError } from 'react-router-dom'

/** Friendly fallback for render/loader errors instead of React Router's dev screen. */
export function RouteError() {
  const error = useRouteError()
  const message = error instanceof Error ? error.message : String(error)
  const stale = /dynamically imported module|Importing a module script failed|Failed to fetch/i.test(message)
  return (
    <div className="plaster grid min-h-screen place-items-center bg-bg px-4 text-fg">
      <div className="placard-frame max-w-md px-8 py-9 text-center">
        <p className="exhibit-label">{stale ? 'Gallery rehung' : 'Temporarily closed'}</p>
        <h1 className="mt-2 font-display text-3xl">{stale ? 'The site was just updated' : 'Something went wrong'}</h1>
        <div className="ornament mx-auto mt-4 w-32" aria-hidden="true">
          <span />
        </div>
        <p className="mt-4 text-sm text-fg-muted">
          {stale ? 'Reload to get the latest version.' : 'Try reloading the page. If it keeps happening, go back to the start.'}
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-3">
          <button className="btn-museum" onClick={() => window.location.reload()}>
            Reload
          </button>
          <a className="btn-outline" href="/">
            Main hall
          </a>
        </div>
        {!stale && <p className="mt-5 font-mono text-xs break-words text-fg-subtle">{message}</p>}
      </div>
    </div>
  )
}
