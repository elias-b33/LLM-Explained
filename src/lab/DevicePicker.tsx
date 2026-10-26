import { useState } from 'react'
import { cn } from '@/lib/cn'
import type { GpuInfo, SelfTestResult } from './protocol'
import type { Device } from './sizes'

function gpuName(i: GpuInfo): string {
  const parts = [i.description, i.vendor, i.architecture].filter(Boolean)
  const uniq = parts.filter((p, k) => parts.findIndex((q) => q.toLowerCase().includes(p.toLowerCase())) === k)
  return uniq.join(' · ') || 'unnamed GPU'
}

/** CPU / GPU segmented switch plus what the browser told us about its GPU. */
export function DevicePicker({
  device,
  onChange,
  gpu,
  disabled,
}: {
  device: Device
  onChange: (d: Device) => void
  gpu: GpuInfo | null
  disabled?: boolean
}) {
  const gpuOk = gpu?.status === 'ok'
  const opts: { id: Device; label: string; sub: string; enabled: boolean }[] = [
    { id: 'cpu', label: 'CPU', sub: 'TypeScript engine, always works', enabled: true },
    { id: 'gpu', label: 'GPU', sub: gpuOk ? 'WebGPU compute shaders' : gpu ? 'Not available here' : 'Checking…', enabled: gpuOk },
  ]
  return (
    <div className="space-y-2">
      <div role="radiogroup" aria-label="Run training on" className="grid grid-cols-2 gap-2.5">
        {opts.map((o) => (
          <button
            key={o.id}
            type="button"
            role="radio"
            aria-checked={device === o.id}
            disabled={disabled || !o.enabled}
            onClick={() => onChange(o.id)}
            className={cn(
              'flex min-w-0 flex-col items-start border px-3.5 py-2.5 text-left transition-colors enabled:cursor-pointer disabled:opacity-50',
              device === o.id
                ? 'border-accent bg-surface shadow-[0_0_0_3px_color-mix(in_srgb,var(--accent)_14%,transparent)]'
                : 'border-border bg-surface/60 enabled:hover:border-brass enabled:hover:bg-surface',
            )}
          >
            <span className="font-display text-lg leading-tight text-fg">Run on {o.label}</span>
            <span className="text-[11px] text-fg-muted">{o.sub}</span>
          </button>
        ))}
      </div>
      <GpuStatus gpu={gpu} />
    </div>
  )
}

function GpuStatus({ gpu }: { gpu: GpuInfo | null }) {
  if (!gpu) return <p className="text-xs text-fg-subtle">Looking for a WebGPU graphics adapter…</p>
  if (gpu.status === 'ok') {
    return (
      <p className="text-xs break-words text-fg-subtle">
        <span className="text-success">●</span> WebGPU adapter found: <b className="font-mono text-fg-muted">{gpuName(gpu)}</b> (hardware,
        not a fallback). Largest buffer it allows: {Math.round(gpu.maxStorageBufferBindingSize / 2 ** 20).toLocaleString('en-US')} MB.
      </p>
    )
  }
  return (
    <div className="border border-border bg-surface/70 p-2.5 text-xs leading-relaxed text-fg-muted">
      <p>
        <b className="text-fg">GPU training is off on this browser.</b> {gpu.reason}
        {gpu.status === 'software' && gpu.description && <> (Adapter: {gpuName(gpu)}.)</>}
      </p>
      <p className="mt-1">
        If you do have a graphics card, hardware acceleration may be switched off. In Chrome, open{' '}
        <code className="font-mono text-fg">chrome://gpu</code> and check that WebGPU says “Hardware accelerated”; if not, turn on
        Settings → System → “Use graphics acceleration when available” and restart the browser. The CPU engine works either way.
      </p>
    </div>
  )
}

/** Button + result table for the GPU-vs-CPU correctness test. */
export function SelfTestPanel({
  result,
  running,
  onRun,
}: {
  result: SelfTestResult | null
  running: boolean
  onRun: () => void
}) {
  const [showAll, setShowAll] = useState(false)
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onRun}
          disabled={running}
          className="border border-border-strong bg-surface px-3 py-1.5 text-xs text-fg transition-colors enabled:cursor-pointer enabled:hover:border-brass disabled:opacity-50"
        >
          {running ? 'Testing…' : result ? 'Run GPU self-test again' : 'Run GPU self-test'}
        </button>
        <span className="text-[11px] text-fg-subtle">Checks the GPU engine against the CPU engine, number by number.</span>
      </div>
      {result && (
        <div className="border border-border bg-surface/70 p-2.5 text-xs">
          <p className={cn('font-mono break-words', result.ok ? 'text-success' : '')} style={result.ok ? undefined : { color: 'var(--danger)' }}>
            {result.error ? `Could not run: ${result.error}` : result.summary}
          </p>
          {result.cases.length > 0 && (
            <>
              <p className="mt-1 text-fg-subtle">
                Same seed, same batch on both engines (CPU in 64-bit as the reference, GPU in 32-bit). {Math.round(result.ms)} ms.{' '}
                <button type="button" className="cursor-pointer underline decoration-brass underline-offset-2" onClick={() => setShowAll((v) => !v)}>
                  {showAll ? 'Hide' : 'Show'} per-tensor table
                </button>
              </p>
              {showAll && (
                <div className="mt-2 max-h-80 overflow-auto">
                  {result.cases.map((c) => (
                    <table key={c.label} className="mb-3 w-full font-mono text-[10.5px] tabular-nums">
                      <caption className="py-1 text-left text-fg-muted">
                        {c.ok ? '✓' : '✗'} {c.label}: loss CPU {c.lossCpu.toFixed(6)} / GPU {c.lossGpu.toFixed(6)}
                      </caption>
                      <thead>
                        <tr className="text-left text-fg-subtle">
                          <th className="pr-2 font-normal">tensor</th>
                          <th className="pr-2 font-normal">grad rel err</th>
                          <th className="font-normal">param Δ / lr</th>
                        </tr>
                      </thead>
                      <tbody>
                        {c.grads.map((g, i) => (
                          <tr key={g.tensor} className={g.ok && c.params[i].ok ? 'text-fg-muted' : ''}>
                            <td className="pr-2">{g.tensor}</td>
                            <td className="pr-2">{g.rel.toExponential(1)}</td>
                            <td>{c.params[i].rel.toExponential(1)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  ))}
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  )
}
