import { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useQuery } from '@tanstack/react-query'
import { toast } from 'sonner'
import { useTheme, type Theme } from '../hooks/use-theme'
import { useSystemInfo } from '../hooks/use-engines'
import { apiJson } from '../lib/api/client'
import { listProfiles } from '../lib/api/profiles'
import { listHistory } from '../lib/api/history'
import { clsx } from 'clsx'

interface StorageInfo { audio_bytes: number; cache_bytes: number; disk_free_bytes: number; profile_count: number; history_count: number }
const sections = ['appearance', 'storage', 'advanced', 'about'] as const
type Section = typeof sections[number]
const buttonClass = 'px-4 py-2 rounded-lg border border-border dark:border-dark-border disabled:opacity-50 hover:bg-surface-hover dark:hover:bg-dark-surface-hover'
function megabytes(bytes: number) { return (bytes / 1024 / 1024).toFixed(1) }

export default function Settings() {
  const { t } = useTranslation()
  const [section, setSection] = useState<Section>('appearance')
  const [theme, setTheme] = useTheme()
  const system = useSystemInfo()
  const storage = useQuery({ queryKey: ['system', 'storage'], queryFn: () => apiJson<StorageInfo>('/system/storage'), enabled: section === 'storage' })
  const [busy, setBusy] = useState(false)
  async function exportMetadata() {
    setBusy(true)
    try {
      const profiles = await listProfiles()
      const history = []
      for (let offset = 0; ; offset += 200) {
        const page = await listHistory(200, offset)
        history.push(...page)
        if (page.length < 200) break
      }
      const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), app_version: system.data?.app_version, profiles, history }, null, 2)], { type: 'application/json' })
      const url = URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = url; link.download = 'voicestudio-session.json'; link.click()
      setTimeout(() => URL.revokeObjectURL(url), 10000)
    } catch (error) { toast.error(error instanceof Error ? error.message : t('settings.exportFailed')) }
    finally { setBusy(false) }
  }
  async function restart() {
    if (!window.voicestudio) return
    setBusy(true)
    try {
      const result = await window.voicestudio.backend.restart()
      if (!result.success) throw new Error(t('settings.restartFailed'))
      await system.refetch()
    } catch (error) { toast.error(error instanceof Error ? error.message : t('settings.restartFailed')) }
    finally { setBusy(false) }
  }
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h1 className="text-2xl font-semibold">{t('settings.title')}</h1>
      <nav className="flex flex-wrap gap-2" aria-label={t('settings.title')}>
        {sections.map(id => <button key={id} onClick={() => setSection(id)} className={clsx(buttonClass, section === id && 'bg-primary/10')} aria-current={section === id ? 'page' : undefined}>{t('settings.' + id)}</button>)}
      </nav>
      {section === 'appearance' && <div className="space-y-4">
        <label className="flex gap-4 items-center">{t('settings.theme')}
          <select value={theme} onChange={event => setTheme(event.target.value as Theme)} className={buttonClass}>
            {(['system', 'light', 'dark'] as const).map(value => <option key={value} value={value}>{t('settings.' + value)}</option>)}
          </select>
        </label>
        <p>{t('settings.availableControls')}</p>
      </div>}
      {section === 'storage' && <div className="space-y-4">
        <p>{t('settings.sessionStorageNotice')}</p>
        {storage.isPending && <p>{t('common.loading')}</p>}
        {storage.error && <p role="alert">{storage.error.message}</p>}
        {storage.data && <dl className="grid grid-cols-2 gap-3">
          <dt>{t('settings.audioFiles')}</dt><dd>{megabytes(storage.data.audio_bytes)} MB</dd>
          <dt>{t('settings.cache')}</dt><dd>{megabytes(storage.data.cache_bytes)} MB</dd>
          <dt>{t('settings.available')}</dt><dd>{megabytes(storage.data.disk_free_bytes)} MB</dd>
          <dt>{t('settings.profiles')}</dt><dd>{storage.data.profile_count}</dd>
          <dt>{t('settings.history')}</dt><dd>{storage.data.history_count}</dd>
        </dl>}
        <button className={buttonClass} disabled={busy} onClick={() => void exportMetadata()}>{t('settings.exportData')}</button>
        <button className={buttonClass} disabled={storage.isFetching} onClick={() => void storage.refetch()}>{t('common.refresh')}</button>
      </div>}
      {(section === 'advanced' || section === 'about') && <div className="space-y-4">
        {system.isPending && <p>{t('common.loading')}</p>}
        {system.error && <p role="alert">{system.error.message}</p>}
        {system.data && <dl className="grid grid-cols-2 gap-3">
          <dt>VoiceStudio</dt><dd>{system.data.app_version}</dd>
          <dt>{t('settings.pythonVersion')}</dt><dd>{system.data.python_version}</dd>
          <dt>{t('settings.torchVersion')}</dt><dd>{system.data.torch_version}</dd>
          <dt>{t('settings.computeDevice')}</dt><dd>{system.data.device}</dd>
          <dt>{t('settings.platform')}</dt><dd>{system.data.platform}</dd>
        </dl>}
        {section === 'advanced' && <button className={buttonClass} disabled={busy || !window.voicestudio} onClick={() => void restart()}>{t('settings.restartBackend')}</button>}
        <p>{t('settings.capabilityNotice')}</p>
        {section === 'about' && <p>MIT</p>}
      </div>}
    </div>
  )
}
