import { createStore, useStore } from 'zustand'
export interface ReferenceState { file: File | null; duration_seconds: number | null; object_url: string | null }
interface SelectionResult { ok: boolean; duration_seconds: number | null; too_long: boolean; cancelled?: boolean }
const referenceStore = createStore<ReferenceState>(() => ({ file: null, duration_seconds: null, object_url: null }))
let generation = 0
let cancelProbe: (() => void) | null = null
export function useReference() { return useStore(referenceStore) }

export async function setReferenceFile(file: File | null): Promise<SelectionResult> {
  generation++
  cancelProbe?.()
  cancelProbe = null
  const previous = referenceStore.getState().object_url
  if (previous) URL.revokeObjectURL(previous)
  referenceStore.setState({ file: null, duration_seconds: null, object_url: null })
  if (!file) return { ok: true, duration_seconds: null, too_long: false }
  const current = generation
  const objectUrl = URL.createObjectURL(file)
  const audio = new Audio(objectUrl)
  return new Promise(resolve => {
    let settled = false
    let timer: ReturnType<typeof setTimeout>
    function finish(ok: boolean, duration: number | null = null, cancelled = false) {
      if (settled) return
      settled = true
      clearTimeout(timer)
      audio.removeEventListener('loadedmetadata', loaded)
      audio.removeEventListener('error', failed)
      audio.removeAttribute('src')
      audio.load()
      if (current === generation) cancelProbe = null
      if (ok && current === generation) {
        referenceStore.setState({ file, duration_seconds: duration, object_url: objectUrl })
      } else URL.revokeObjectURL(objectUrl)
      resolve({ ok: ok && current === generation, duration_seconds: duration, too_long: (duration || 0) > 75, cancelled })
    }
    function loaded() {
      const duration = audio.duration
      finish(Number.isFinite(duration) && duration > 0, Number.isFinite(duration) ? duration : null)
    }
    function failed() { finish(false) }
    cancelProbe = () => finish(false, null, true)
    audio.addEventListener('loadedmetadata', loaded)
    audio.addEventListener('error', failed)
    timer = setTimeout(failed, 20000)
    audio.load()
  })
}
export function clearReference() { void setReferenceFile(null) }
export { referenceStore }
