import { useCallback, useRef, useSyncExternalStore } from 'react'

let activeSource: string | null = null
let activeStop: (() => void) | null = null
const subscribers = new Set<() => void>()

function notify() {
  subscribers.forEach(cb => cb())
}

function subscribe(callback: () => void) {
  subscribers.add(callback)
  return () => subscribers.delete(callback)
}

function getSnapshot() {
  return activeSource
}

function getServerSnapshot() {
  return activeSource
}

export function usePlaybackSource(): string | null {
  return useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}

export function claimPlayback(stop: () => void, source: string): () => void {
  // Stop any currently playing audio
  if (activeStop) {
    activeStop()
  }
  
  activeSource = source
  activeStop = stop
  notify()
  
  // Return cleanup function
  return () => {
    if (activeSource === source) {
      activeSource = null
      activeStop = null
      notify()
    }
  }
}

export function stopActivePlayback() {
  if (activeStop) {
    activeStop()
    activeStop = null
    activeSource = null
    notify()
  }
}

export function playBlob(blob: Blob, source: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    
    const cleanup = claimPlayback(() => {
      audio.pause()
      audio.src = ''
      URL.revokeObjectURL(url)
    }, source)
    
    audio.onended = () => {
      cleanup()
      URL.revokeObjectURL(url)
      resolve()
    }
    
    audio.onerror = (err) => {
      cleanup()
      URL.revokeObjectURL(url)
      reject(err)
    }
    
    audio.play().catch((err) => {
      cleanup()
      URL.revokeObjectURL(url)
      reject(err)
    })
  })
}
