import { useSyncExternalStore } from 'react'
import { apiJson } from '../lib/api'

export interface BackendStatus {
  running: boolean
  url: string
  version: string
}

let currentStatus: BackendStatus = { running: false, url: '', version: '' }
let lastError: Error | null = null
let checked = false
let subscribers = new Set<() => void>()

function notify() {
  subscribers.forEach(cb => cb())
}

async function pollStatus() {
  try {
    const status = await apiJson<{ app_version: string }>('/system/info')
    currentStatus = { running: true, url: '', version: status.app_version || '' }
    lastError = null
  } catch (error) {
    currentStatus = { running: false, url: '', version: '' }
    lastError = error instanceof Error ? error : new Error('Backend unavailable')
  }
  checked = true
  notify()
}

// Start polling
let pollInterval: ReturnType<typeof setInterval> | null = null
function startPolling() {
  if (pollInterval) return
  pollStatus()
  pollInterval = setInterval(pollStatus, 5000)
}

function stopPolling() {
  if (pollInterval) {
    clearInterval(pollInterval)
    pollInterval = null
  }
}

function subscribe(callback: () => void) {
  subscribers.add(callback)
  if (subscribers.size === 1) startPolling()
  return () => {
    subscribers.delete(callback)
    if (subscribers.size === 0) stopPolling()
  }
}

function getSnapshot() {
  return currentStatus
}

function getServerSnapshot() {
  return currentStatus
}

export function useBackendStatus(): { data: BackendStatus | undefined; error: Error | null; isLoading: boolean } {
  const status = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  return {
    data: status.running ? status : undefined,
    error: lastError,
    isLoading: !checked,
  }
}

// Re-export for components that need direct access
export { pollStatus }
