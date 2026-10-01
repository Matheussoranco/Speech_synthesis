import { createStore } from '@tanstack/store'

export interface ReferenceState {
  file: File | null
  duration_seconds: number | null
  object_url: string | null
}

const referenceStore = createStore<ReferenceState>()({
  file: null,
  duration_seconds: null,
  object_url: null,
})

export function useReference() {
  return referenceStore.useStore()
}

export async function setReferenceFile(file: File | null): Promise<{ ok: boolean; duration_seconds: number | null; too_long: boolean }> {
  // Clean up previous object URL
  const prev = referenceStore.getState()
  if (prev.object_url) {
    URL.revokeObjectURL(prev.object_url)
  }
  
  if (!file) {
    referenceStore.setState({ file: null, duration_seconds: null, object_url: null })
    return { ok: true, duration_seconds: null, too_long: false }
  }
  
  // Probe duration
  const object_url = URL.createObjectURL(file)
  const audio = new Audio(object_url)
  
  return new Promise((resolve) => {
    audio.addEventListener('loadedmetadata', () => {
      const duration = audio.duration
      const too_long = duration > 75 // REF_HARD_MAX_SECONDS
      
      referenceStore.setState({
        file,
        duration_seconds: duration,
        object_url,
      })
      
      resolve({ ok: true, duration_seconds: duration, too_long })
    })
    
    audio.addEventListener('error', () => {
      URL.revokeObjectURL(object_url)
      referenceStore.setState({ file: null, duration_seconds: null, object_url: null })
      resolve({ ok: false, duration_seconds: null, too_long: false })
    })
    
    // Force metadata load
    audio.load()
  })
}

export function clearReference() {
  const prev = referenceStore.getState()
  if (prev.object_url) {
    URL.revokeObjectURL(prev.object_url)
  }
  referenceStore.setState({ file: null, duration_seconds: null, object_url: null })
}

export { referenceStore }
