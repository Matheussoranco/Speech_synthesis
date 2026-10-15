// @vitest-environment jsdom
import { act, createElement } from 'react'
import { createRoot, type Root } from 'react-dom/client'
import { beforeEach, afterEach, it, expect, vi } from 'vitest'
import { useRecording } from '../src/renderer/src/hooks/use-recording'
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: (key: string) => key }) }))
vi.mock('sonner', () => ({ toast: { error: vi.fn() } }))
let root: Root
let hook: ReturnType<typeof useRecording>
let track: { stop: ReturnType<typeof vi.fn> }
let getUserMedia: ReturnType<typeof vi.fn>
let recorded: ReturnType<typeof vi.fn>
let recorder: MockRecorder
class MockRecorder {
  static isTypeSupported() { return true }
  state = 'inactive'
  mimeType = 'audio/webm'
  onstop: (() => void) | null = null
  ondataavailable: ((event: { data: Blob }) => void) | null = null
  onerror: (() => void) | null = null
  // eslint-disable-next-line no-this-alias -- Expose the controlled device instance to lifecycle assertions.
  constructor() { recorder = this }
  start() { this.state = 'recording' }
  stop() {
    if (this.state === 'inactive') throw new Error('Already stopped')
    this.state = 'inactive'
    this.ondataavailable?.({ data: new Blob(['recorded-control']) })
    this.onstop?.()
  }
}
class MockContext {
  state = 'running'
  createAnalyser() { return { frequencyBinCount: 4, getByteFrequencyData: () => {}, fftSize: 256 } }
  createMediaStreamSource() { return { connect: () => {} } }
  close() { this.state = 'closed'; return Promise.resolve() }
}
beforeEach(async () => {
  vi.useFakeTimers()
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true)
  track = { stop: vi.fn() }
  getUserMedia = vi.fn(async () => ({ getTracks: () => [track] }))
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    enumerateDevices: async () => [], getUserMedia,
  } })
  vi.stubGlobal('MediaRecorder', MockRecorder); vi.stubGlobal('AudioContext', MockContext)
  vi.stubGlobal('requestAnimationFrame', (callback: () => void) => setTimeout(callback, 50))
  vi.stubGlobal('cancelAnimationFrame', clearTimeout)
  recorded = vi.fn()
  function Harness() { hook = useRecording({ onRecorded: recorded, maxDuration: 2 }); return null }
  root = createRoot(document.createElement('div'))
  await act(async () => root.render(createElement(Harness)))
})
afterEach(async () => { await act(async () => root.unmount()); vi.useRealTimers(); vi.unstubAllGlobals() })
it('the duration cap stops the real hook recorder rather than a stale closure', async () => {
  await act(async () => hook.start())
  expect(hook.isRecording).toBe(true)
  expect(track.stop).not.toHaveBeenCalled()
  await act(async () => vi.advanceTimersByTimeAsync(2000))
  expect(hook.isRecording).toBe(false)
  expect(track.stop).toHaveBeenCalledOnce()
  expect(recorded).toHaveBeenCalledOnce()
  expect((recorded.mock.calls[0][0] as File).type).toBe('audio/webm')
  await act(async () => hook.stop())
  expect(recorded).toHaveBeenCalledOnce()
})
it('unmount closes an active microphone without publishing a new file', async () => {
  await act(async () => hook.start())
  await act(async () => root.unmount())
  expect(track.stop).toHaveBeenCalledOnce()
  expect(recorded).not.toHaveBeenCalled()
  expect(recorder.state).toBe('inactive')
})
it('a late permission result after unmount releases its stream', async () => {
  let resolve!: (value: unknown) => void
  getUserMedia.mockImplementation(() => new Promise(done => { resolve = done }))
  let pending!: Promise<void>
  await act(async () => { pending = hook.start() })
  await act(async () => root.unmount())
  resolve({ getTracks: () => [track] }); await pending
  expect(track.stop).toHaveBeenCalledOnce()
  expect(recorded).not.toHaveBeenCalled()
})
it('a failure after permission acquisition releases its microphone', async () => {
  vi.stubGlobal('AudioContext', class { constructor() { throw new Error('Device unavailable') } })
  await act(async () => hook.start())
  expect(track.stop).toHaveBeenCalledOnce()
  expect(hook.isStarting).toBe(false)
  expect(hook.isRecording).toBe(false)
})
