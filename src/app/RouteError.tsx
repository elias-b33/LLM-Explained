import { useRouteError } from 'react-router-dom'

/** Friendly fallback for render/loader errors instead of React Router's dev screen. */
export function RouteError() {
  const error = useRouteError()
  const message = error instanceof Error ? error.message : String(error)
  const stale = /dynamically imported module|Importing a module script failed|Failed to fetch/i.test(message)
  return (
    <div className="grid min-h-screen place-items-center bg-bg px-4 text-fg">
      <div className="max-w-md text-center">
        <h1 className="text-xl font-semibold">{stale ? 'The site was just updated' : 'Something went wrong'}</h1>
        <p className="mt-2 text-sm text-fg-muted">
          {stale ? 'Reload to get the latest version.' : 'Try reloading the page. If it keeps happening, go back to the start.'}
        </p>
        <div className="mt-5 flex justify-center gap-3">
          <button
            className="rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white"
            onClick={() => window.location.reload()}
          >
            Reload
          </button>
          <a className="rounded-lg border border-border px-4 py-2 text-sm" href="/">
            Home
          </a>
        </div>
        {!stale && <p className="mt-4 font-mono text-xs text-fg-subtle">{message}</p>}
      </div>
    </div>
  )
}
