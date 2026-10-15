import { useTranslation } from 'react-i18next'
import { Mic, Loader2 } from 'lucide-react'
import { useState } from 'react'
import { toast } from 'sonner'

export default function LoadingScreen({ error }: { error?: Error | null }) {
  const { t } = useTranslation()
  const [busy, setBusy] = useState(false)
  async function retry() {
    if (!window.voicestudio) { location.reload(); return }
    setBusy(true)
    try {
      const result = await window.voicestudio.backend.restart()
      if (!result.success) throw new Error(t('settings.restartFailed'))
    } catch (failure) { toast.error(failure instanceof Error ? failure.message : t('settings.restartFailed')) }
    finally { setBusy(false) }
  }
  
  return (
    <div className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-surface dark:bg-dark-surface">
      <div className="relative">
        <div className="w-24 h-24 rounded-2xl bg-primary/10 dark:bg-primary/20 flex items-center justify-center">
          <Mic size={40} className="text-primary" aria-hidden="true" />
        </div>
        {!error && <Loader2 size={32} className="absolute -top-2 -right-2 text-primary animate-spin" aria-hidden="true" />}
      </div>
      <div className="mt-6 text-center">
        <h1 className="text-2xl font-semibold text-text-primary dark:text-dark-text-primary">
          {t('loading.title')}
        </h1>
        <p className="mt-2 text-text-secondary dark:text-dark-text-secondary">
          {error ? t('loading.backendUnavailable') : t('loading.subtitle')}
        </p>
      </div>
      {error && <div className="mt-6 space-y-3 text-center">
        <p role="alert">{error.message}</p>
        <button className="px-4 py-2 rounded-lg border border-border" disabled={busy} onClick={() => void retry()}>{t('loading.retry')}</button>
      </div>}
      {!error && <div className="mt-8 w-64">
        <div className="h-2 bg-border dark:bg-dark-border rounded-full overflow-hidden">
          <div 
            className="h-full bg-primary rounded-full animate-pulse"
            style={{ width: '33%' }}
            aria-hidden="true"
          />
        </div>
      </div>}
    </div>
  )
}
