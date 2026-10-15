import { useEffect, useRef, useState, type FormEvent } from 'react'
import { useTranslation } from 'react-i18next'
import { useEngines } from '../../hooks/use-engines'
import { generateTTS, type GenerateResult } from '../../lib/api/generate'
import { audioUrl } from '../../lib/api/client'
import { queryClient, queryKeys } from '../../lib/api/query'

export default function TtsWorkspace() {
  const { t } = useTranslation()
  const { activeTts, isLoading } = useEngines()
  const [text, setText] = useState('')
  const [language, setLanguage] = useState('auto')
  const [voice, setVoice] = useState('default')
  const [speed, setSpeed] = useState(1)
  const [result, setResult] = useState<GenerateResult | null>(null)
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)
  const controller = useRef<AbortController | null>(null)
  useEffect(() => () => controller.current?.abort(), [])

  async function submit(event: FormEvent) {
    event.preventDefault()
    if (controller.current || !activeTts || !text.trim()) return
    const current = new AbortController()
    controller.current = current
    setPending(true); setError(''); setResult(null)
    try {
      const generated = await generateTTS({ text, language, voice, speed }, { signal: current.signal })
      if (!current.signal.aborted) {
        setResult(generated)
        void queryClient.invalidateQueries({ queryKey: queryKeys.history })
      }
    } catch (failure) {
      if (!current.signal.aborted) setError(failure instanceof Error ? failure.message : t('common.error'))
    } finally {
      if (controller.current === current) { controller.current = null; setPending(false) }
    }
  }

  return <section className="max-w-3xl mx-auto space-y-4">
    <h1 className="text-2xl font-semibold">{t('tts.title')}</h1>
    <p>{t('tts.description')}</p>
    {!isLoading && !activeTts && <p role="status">{t('tts.unavailable')}</p>}
    <form onSubmit={submit} className="space-y-4">
      <label className="block">{t('clone.scriptLabel')}
        <textarea value={text} onChange={event => setText(event.target.value)} maxLength={5000} rows={6}
          className="block w-full border rounded-lg p-3" required disabled={pending} />
      </label>
      <label className="block">{t('clone.language')}
        <select value={language} onChange={event => setLanguage(event.target.value)} disabled={pending} className="border p-2 ml-2">
          <option value="auto">{t('clone.auto')}</option>
          {activeTts?.languages.map(value => <option key={value} value={value}>{value}</option>)}
        </select>
      </label>
      <label className="block">{t('tts.voice')}
        <select value={voice} onChange={event => setVoice(event.target.value)} disabled={pending} className="border p-2 ml-2">
          <option value="default">{t('tts.defaultVoice')}</option>
          {activeTts?.models.map(value => <option key={value} value={value}>{value}</option>)}
        </select>
      </label>
      <label className="block">{t('clone.speed')} ({speed.toFixed(1)})
        <input type="range" min="0.5" max="2" step="0.1" value={speed} disabled={pending}
          onChange={event => setSpeed(Number(event.target.value))} className="block w-full" />
      </label>
      <button type="submit" disabled={pending || !activeTts || !text.trim()} className="rounded-lg bg-primary text-white px-4 py-2 disabled:opacity-50">
        {pending ? t('common.loading') : t('clone.generate')}
      </button>
    </form>
    {error && <p role="alert">{error}</p>}
    {result && <div className="space-y-2">
      <audio controls src={audioUrl(result.audio_path.split('/').pop() || '')} className="w-full" />
      <p>{t('clone.generated', { duration: `${result.duration.toFixed(1)}s`, engine: result.engine })}</p>
    </div>}
  </section>
}
