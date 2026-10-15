// @vitest-environment jsdom
import { beforeEach, afterEach, it, expect, vi } from 'vitest'
import { setReferenceFile, clearReference, referenceStore } from '../src/renderer/src/lib/store/reference'
const audio: MockAudio[] = []
class MockAudio extends EventTarget {
  duration = 3
  constructor() { super(); audio.push(this) }
  load() {}
  removeAttribute() {}
}
beforeEach(() => {
  vi.useFakeTimers(); audio.length = 0
  vi.stubGlobal('Audio', MockAudio)
  Object.defineProperty(URL, 'createObjectURL', { configurable: true, value: vi.fn(() => `blob:control-${audio.length}`) })
  Object.defineProperty(URL, 'revokeObjectURL', { configurable: true, value: vi.fn() })
  clearReference()
})
afterEach(() => { clearReference(); vi.useRealTimers(); vi.unstubAllGlobals() })
it('late metadata from an older selection cannot replace the newest file', async () => {
  const first = setReferenceFile(new File(['a'], 'a.wav'))
  const file = new File(['b'], 'b.wav')
  const second = setReferenceFile(file)
  expect((await first).cancelled).toBe(true)
  audio[1].dispatchEvent(new Event('loadedmetadata'))
  expect((await second).ok).toBe(true)
  audio[0].dispatchEvent(new Event('loadedmetadata'))
  expect(referenceStore.getState().file).toBe(file)
})
it('clear cancels pending metadata and releases its URL', async () => {
  const pending = setReferenceFile(new File(['a'], 'a.wav'))
  clearReference()
  expect((await pending).cancelled).toBe(true)
  audio[0].dispatchEvent(new Event('loadedmetadata'))
  expect(referenceStore.getState().file).toBeNull()
  expect(URL.revokeObjectURL).toHaveBeenCalled()
})
it('non-finite metadata and timeout never leave an unusable file selected', async () => {
  const invalid = setReferenceFile(new File(['a'], 'a.wav'))
  audio[0].duration = Infinity; audio[0].dispatchEvent(new Event('loadedmetadata'))
  expect((await invalid).ok).toBe(false)
  const timeout = setReferenceFile(new File(['b'], 'b.wav'))
  await vi.advanceTimersByTimeAsync(20000)
  expect((await timeout).ok).toBe(false)
  expect(referenceStore.getState().file).toBeNull()
})
