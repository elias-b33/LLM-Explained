/**
 * Delete every model file this page can leave in browser storage. Works in the page and in
 * the worker. WebLLM always writes downloads to storage (with the default "cache" backend:
 * Cache Storage buckets "webllm/model", "webllm/config", "webllm/wasm"); Transformers.js only
 * writes when the user opted in (our OPFS folder). IndexedDB is checked too, in case WebLLM's
 * "indexeddb" backend was ever used.
 */
import { OPFS_DIR } from './opfsCache'

const NAME_RE = /webllm|tvmjs|transformers|onnx|huggingface|mlc/i

export async function clearModelStorage(): Promise<number> {
  let removed = 0
  if (typeof caches !== 'undefined') {
    try {
      for (const k of await caches.keys()) {
        if (!NAME_RE.test(k)) continue
        const c = await caches.open(k)
        removed += (await c.keys()).length
        await caches.delete(k)
      }
    } catch {
      /* storage blocked */
    }
  }
  try {
    const root = await navigator.storage.getDirectory()
    const d = await root.getDirectoryHandle(OPFS_DIR)
    for await (const name of (d as unknown as { keys(): AsyncIterable<string> }).keys()) if (name) removed++
    await root.removeEntry(OPFS_DIR, { recursive: true })
  } catch {
    /* no folder */
  }
  try {
    const dbs = (await indexedDB.databases?.()) ?? []
    for (const db of dbs) {
      if (db.name && NAME_RE.test(db.name)) {
        removed++
        await new Promise<void>((res) => {
          const r = indexedDB.deleteDatabase(db.name!)
          r.onsuccess = r.onerror = r.onblocked = () => res()
        })
      }
    }
  } catch {
    /* no IndexedDB */
  }
  return removed
}

/** number of stored model entries (Cache Storage entries + OPFS files) */
export async function storedModelEntries(): Promise<number> {
  let n = 0
  if (typeof caches !== 'undefined') {
    try {
      for (const k of await caches.keys()) if (NAME_RE.test(k)) n += (await (await caches.open(k)).keys()).length
    } catch {
      /* blocked */
    }
  }
  try {
    const d = await (await navigator.storage.getDirectory()).getDirectoryHandle(OPFS_DIR)
    for await (const k of (d as unknown as { keys(): AsyncIterable<string> }).keys()) if (!k.endsWith('.part')) n++
  } catch {
    /* none */
  }
  return n
}
