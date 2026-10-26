/**
 * Opt-in model cache ("Keep the model for next time") in the Origin Private File System.
 *
 * Transformers.js' default browser cache (Cache Storage) failed in testing to store the
 * 786 MB model file ("UnknownError: Failed to execute 'put' on 'Cache'"), so with the
 * opt-in on, files go here instead, streamed straight to disk. Plugged in through
 * `env.customCache`, which only needs match() and put().
 */

export const OPFS_DIR = 'ask-the-text-models'

function fileName(key: string): string {
  return key.replace(/^https?:\/\//, '').replace(/[^a-zA-Z0-9._-]/g, '_')
}

async function dir(create: boolean): Promise<FileSystemDirectoryHandle> {
  const root = await navigator.storage.getDirectory()
  return root.getDirectoryHandle(OPFS_DIR, { create })
}

export const opfsCache = {
  async match(key: string): Promise<Response | undefined> {
    try {
      const d = await dir(false)
      const file = await (await d.getFileHandle(fileName(key))).getFile()
      return new Response(file, { headers: { 'content-length': String(file.size) } })
    } catch {
      return undefined
    }
  },

  async put(key: string, response: Response): Promise<void> {
    const name = fileName(key)
    const d = await dir(true)
    const part = await d.getFileHandle(`${name}.part`, { create: true })
    try {
      const w = await part.createWritable()
      if (response.body) await response.body.pipeTo(w)
      else {
        await w.write(await response.arrayBuffer())
        await w.close()
      }
      // only a complete file gets the real name, so a half-written one is never "found"
      await (part as FileSystemFileHandle & { move(name: string): Promise<void> }).move(name)
    } catch (err) {
      await d.removeEntry(`${name}.part`).catch(() => {})
      throw err
    }
  },
}
