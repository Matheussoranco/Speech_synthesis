import { it, expect, vi, afterEach } from 'vitest'
import { apiFetch, apiFormData, ApiError } from '../src/renderer/src/lib/api/client'
import { generateClone, generateTTS } from '../src/renderer/src/lib/api/generate'
import { createProfile, updateProfile } from '../src/renderer/src/lib/api/profiles'
afterEach(() => vi.unstubAllGlobals())
it('profile creation and edits preserve the backend multipart contract', async () => {
  const fetch = vi.fn(async () => new Response('{}', { headers: { 'content-type': 'application/json' } }))
  vi.stubGlobal('fetch', fetch)
  await createProfile({ name: 'Minha voz', ref_audio: new File(['audio'], 'voice.webm'), ref_audio_name: 'Referência.webm', ref_text: 'Olá', instruct: '', language: 'pt' })
  await updateProfile('control/id', { name: 'Nova voz', ref_text: 'Novo texto' })
  const [url, init] = fetch.mock.calls[1] as unknown as [string, RequestInit]
  expect(url).toBe('/api/profiles/control%2Fid')
  expect(init.method).toBe('PATCH')
  expect((init.body as FormData).get('name')).toBe('Nova voz')
  for (const call of fetch.mock.calls) expect(new Headers((call as unknown as [string, RequestInit])[1].headers).has('content-type')).toBe(false)
})
it('clone sends actual audio form fields without overriding browser multipart headers', async () => {
  const fetch = vi.fn(async () => new Response('{}', { headers: { 'content-type': 'application/json' } }))
  vi.stubGlobal('fetch', fetch)
  const audio = new File(['wave-control'], 'voice.wav', { type: 'audio/wav' })
  await generateClone({ text: 'Teste de voz', language: 'pt', ref_audio: audio, cfg: 0, denoise: false })
  const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit]
  expect(url).toBe('/api/generate/clone')
  expect(new Headers(init.headers).has('content-type')).toBe(false)
  expect(init.body).toBeInstanceOf(FormData)
  const body = init.body as FormData
  expect(body.get('text')).toBe('Teste de voz'); expect(body.get('cfg')).toBe('0')
  expect(body.get('denoise')).toBe('false'); expect((body.get('ref_audio') as File).name).toBe('voice.wav')
})
it('Headers inputs survive JSON and multipart helpers and multipart removes conflicting content-type', async () => {
  const fetch = vi.fn(async () => new Response('{}', { headers: { 'content-type': 'application/json' } }))
  vi.stubGlobal('fetch', fetch)
  await apiFetch('/control', { headers: new Headers({ authorization: 'test-control' }) })
  expect(new Headers((fetch.mock.calls[0] as unknown as [string, RequestInit])[1].headers).get('authorization')).toBe('test-control')
  await apiFormData('/control', new FormData(), { headers: new Headers({ 'content-type': 'application/json', 'x-control': 'yes' }) })
  const headers = new Headers((fetch.mock.calls[1] as unknown as [string, RequestInit])[1].headers)
  expect(headers.has('content-type')).toBe(false); expect(headers.get('x-control')).toBe('yes')
})
it('plain backend failures retain their body after failed JSON parsing', async () => {
  const error = await ApiError.fromResponse(new Response('Model unavailable', { status: 503 }))
  expect(error.status).toBe(503); expect(error.detail).toBe('Model unavailable')
})
it('standard TTS matches the actual backend form contract and preserves cancellation', async () => {
  const fetch = vi.fn(async () => new Response('{}', { headers: { 'content-type': 'application/json' } }))
  vi.stubGlobal('fetch', fetch)
  const controller = new AbortController()
  await generateTTS({ text: 'Teste', language: 'pt', voice: 'default', speed: 1 }, { signal: controller.signal })
  const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit]
  expect(url).toBe('/api/generate/tts')
  expect((init.body as FormData).get('language')).toBe('pt')
  expect(init.signal).toBe(controller.signal)
  expect(new Headers(init.headers).has('content-type')).toBe(false)
})
