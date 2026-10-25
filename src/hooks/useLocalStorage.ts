import { useCallback, useEffect, useRef, useState } from 'react'

type SetValue<T> = T | ((prev: T) => T)

/**
 * State synced to localStorage under `key`. Returns [value, setValue, reset].
 * Tolerant of private-mode / quota errors (falls back to in-memory state).
 */
export function useLocalStorage<T>(key: string, initialValue: T) {
  const initialRef = useRef(initialValue)

  const [value, setValue] = useState<T>(() => {
    try {
      const raw = localStorage.getItem(key)
      return raw === null ? initialRef.current : (JSON.parse(raw) as T)
    } catch {
      return initialRef.current
    }
  })

  useEffect(() => {
    try {
      localStorage.setItem(key, JSON.stringify(value))
    } catch {
      // ignore write failures
    }
  }, [key, value])

  const set = useCallback((next: SetValue<T>) => {
    setValue((prev) => (typeof next === 'function' ? (next as (p: T) => T)(prev) : next))
  }, [])

  const reset = useCallback(() => setValue(initialRef.current), [])

  return [value, set, reset] as const
}
