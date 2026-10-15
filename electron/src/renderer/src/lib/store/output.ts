import { createStore, useStore } from 'zustand'
import type { GenerateResult } from '../api/generate'

export interface OutputState {
  result: GenerateResult | null
  object_url: string | null
  text: string
}

const outputStore = createStore<OutputState>(() => ({
  result: null,
  object_url: null,
  text: '',
}))

export function useLatestOutput() {
  return useStore(outputStore)
}

export function setLatestOutput(result: GenerateResult, text: string) {
  // Clean up previous object URL
  const prev = outputStore.getState()
  if (prev.object_url) {
    URL.revokeObjectURL(prev.object_url)
  }
  
  outputStore.setState({ result, object_url: null, text })
}

export function clearOutput() {
  const prev = outputStore.getState()
  if (prev.object_url) {
    URL.revokeObjectURL(prev.object_url)
  }
  outputStore.setState({ result: null, object_url: null, text: '' })
}

export { outputStore }
