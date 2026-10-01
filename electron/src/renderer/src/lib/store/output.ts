import { createStore } from '@tanstack/store'
import type { GenerateResult } from '../api/generate'

export interface OutputState {
  result: GenerateResult | null
  object_url: string | null
  text: string
}

const outputStore = createStore<OutputState>()({
  result: null,
  object_url: null,
  text: '',
})

export function useLatestOutput() {
  return outputStore.useStore()
}

export function setLatestOutput(result: GenerateResult, text: string) {
  // Clean up previous object URL
  const prev = outputStore.getState()
  if (prev.object_url) {
    URL.revokeObjectURL(prev.object_url)
  }
  
  const object_url = URL.createObjectURL(new Blob([], { type: 'audio/wav' }))
  // We'll update the object_url when the actual audio blob is available
  // For now, the audio component will fetch from the path
  
  outputStore.setState({ result, object_url, text })
}

export function clearOutput() {
  const prev = outputStore.getState()
  if (prev.object_url) {
    URL.revokeObjectURL(prev.object_url)
  }
  outputStore.setState({ result: null, object_url: null, text: '' })
}

export { outputStore }
