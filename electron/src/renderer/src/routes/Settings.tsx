import { useTranslation } from 'react-i18next'
import { useState } from 'react'
import { 
  Monitor, 
  Globe, 
  Database, 
  Key, 
  Shield, 
  Bell, 
  HelpCircle,
  Cpu,
  HardDrive,
  Download,
  Trash2,
  RefreshCw,
} from 'lucide-react'
import { clsx } from 'clsx'

const settingsSections = [
  { id: 'appearance', label: 'Appearance', icon: Monitor },
  { id: 'general', label: 'General', icon: Globe },
  { id: 'storage', label: 'Storage', icon: Database },
  { id: 'api', label: 'API Keys', icon: Key },
  { id: 'privacy', label: 'Privacy', icon: Shield },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'advanced', label: 'Advanced', icon: Cpu },
  { id: 'about', label: 'About', icon: HelpCircle },
]

function SettingsNav({ activeSection, onSelect }: { activeSection: string; onSelect: (id: string) => void }) {
  return (
    <nav className="w-56 lg:w-64 flex-shrink-0 border-r border-border dark:border-dark-border" aria-label="Settings navigation">
      <div className="p-4 border-b border-border dark:border-dark-border">
        <h2 className="text-sm font-semibold uppercase tracking-wider text-text-muted dark:text-dark-text-muted">
          Settings
        </h2>
      </div>
      <ul className="flex-1 overflow-y-auto py-2" role="listbox">
        {settingsSections.map((section) => (
          <li key={section.id}>
            <button
              role="option"
              aria-selected={activeSection === section.id}
              onClick={() => onSelect(section.id)}
              className={clsx(
                'w-full flex items-center gap-3 px-3 py-2.5 text-sm font-medium transition-colors',
                'rounded-none border-l-2',
                activeSection === section.id
                  ? 'bg-primary/5 dark:bg-primary/10 border-primary text-primary'
                  : 'border-transparent text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover hover:text-text-primary dark:hover:text-dark-text-primary'
              )}
            >
              <section.icon size={18} aria-hidden="true" />
              <span>{section.label}</span>
            </button>
          </li>
        ))}
      </ul>
    </nav>
  )
}

function AppearanceSettings() {
  const { t } = useTranslation()
  const [theme, setTheme] = useState<'system' | 'light' | 'dark'>('system')
  const [font, setFont] = useState<'system' | 'inter'>('inter')
  const [scale, setScale] = useState(100)
  const [glass, setGlass] = useState(false)
  const [palette, setPalette] = useState('default')
  
  return (
    <div className="max-w-2xl space-y-6">
      <div>
        <h3 className="text-lg font-semibold mb-4">{t('settings.appearance')}</h3>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">{t('settings.theme')}</label>
            <div className="grid grid-cols-3 gap-2">
              {['system', 'light', 'dark'].map(t => (
                <button
                  key={t}
                  onClick={() => setTheme(t as any)}
                  className={clsx(
                    'p-3 rounded-lg border-2 transition-colors flex flex-col items-center gap-1',
                    theme === t
                      ? 'border-primary bg-primary/5 dark:bg-primary/10'
                      : 'border-border dark:border-dark-border hover:border-primary/50'
                  )}
                >
                  <span className="capitalize">{t}</span>
                </button>
              ))}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">{t('settings.font')}</label>
            <select
              value={font}
              onChange={e => setFont(e.target.value as any)}
              className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface"
            >
              <option value="system">{t('settings.systemFont')}</option>
              <option value="inter">Inter Variable</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">
              {t('settings.uiScale')} ({scale}%)
            </label>
            <input
              type="range"
              min="80"
              max="150"
              step="5"
              value={scale}
              onChange={e => setScale(parseInt(e.target.value))}
              className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">{t('settings.colorPalette')}</label>
            <div className="flex flex-wrap gap-2">
              {['default', 'canopy', 'current', 'hearth', 'orchid', 'classic'].map(p => (
                <button
                  key={p}
                  onClick={() => setPalette(p)}
                  className={clsx(
                    'px-3 py-1.5 rounded-lg text-sm font-medium border transition-colors',
                    palette === p
                      ? 'border-primary bg-primary/10 dark:bg-primary/20 text-primary'
                      : 'border-border dark:border-dark-border text-text-secondary dark:text-dark-text-secondary'
                  )}
                >
                  {p.charAt(0).toUpperCase() + p.slice(1)}
                </button>
              ))}
            </div>
          </div>
          
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={glass}
              onChange={e => setGlass(e.target.checked)}
              className="w-4 h-4 accent-primary"
            />
            <div>
              <p className="font-medium">{t('settings.glassMode')}</p>
              <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.glassModeDesc')}</p>
            </div>
          </label>
          
          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={false}
              onChange={() => {}}
              className="w-4 h-4 accent-primary"
            />
            <div>
              <p className="font-medium">{t('settings.autoPlayback')}</p>
              <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.autoPlaybackDesc')}</p>
            </div>
          </label>
        </div>
      </div>
    </div>
  )
}

function GeneralSettings() {
  const { t } = useTranslation()
  const [language, setLanguage] = useState('en')
  const [autoUpdate, setAutoUpdate] = useState(true)
  const [startMinimized, setStartMinimized] = useState(false)
  const [closeToTray, setCloseToTray] = useState(true)
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.general')}</h3>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">{t('settings.language')}</label>
          <select
            value={language}
            onChange={e => setLanguage(e.target.value)}
            className="w-full max-w-xs px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface"
          >
            <option value="en">English</option>
            <option value="zh">中文</option>
            <option value="es">Español</option>
            <option value="fr">Français</option>
            <option value="de">Deutsch</option>
            <option value="ja">日本語</option>
            <option value="ko">한국어</option>
          </select>
        </div>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.autoUpdate')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.autoUpdateDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={autoUpdate}
            onChange={e => setAutoUpdate(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.startMinimized')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.startMinimizedDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={startMinimized}
            onChange={e => setStartMinimized(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.closeToTray')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.closeToTrayDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={closeToTray}
            onChange={e => setCloseToTray(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
      </div>
    </div>
  )
}

function StorageSettings() {
  const { t } = useTranslation()
  const [used, setUsed] = useState({ profiles: 45, history: 230, cache: 1200, total: 1475 })
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.storage')}</h3>
      
      <div className="bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border p-4">
        <div className="flex items-center justify-between mb-4">
          <span className="font-medium">{t('settings.totalUsage')}</span>
          <span className="text-lg font-semibold">{used.total} MB</span>
        </div>
        <div className="h-2 bg-border dark:bg-dark-border rounded-full overflow-hidden mb-4">
          <div 
            className="h-full bg-primary rounded-full transition-all"
            style={{ width: `${Math.min(100, (used.total / 5000) * 100)}%` }}
          />
        </div>
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div className="flex justify-between">
            <span className="text-text-muted dark:text-dark-text-muted">{t('settings.profiles')}</span>
            <span>{used.profiles} MB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted dark:text-dark-text-muted">{t('settings.history')}</span>
            <span>{used.history} MB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted dark:text-dark-text-muted">{t('settings.cache')}</span>
            <span>{used.cache} MB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-text-muted dark:text-dark-text-muted">{t('settings.available')}</span>
            <span>{5000 - used.total} MB</span>
          </div>
        </div>
      </div>
      
      <div className="space-y-2">
        <button className="w-full px-4 py-2 border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover flex items-center justify-center gap-2">
          <Trash2 size={18} />
          {t('settings.clearCache')}
        </button>
        <button className="w-full px-4 py-2 border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover flex items-center justify-center gap-2">
          <Download size={18} />
          {t('settings.exportData')}
        </button>
        <button className="w-full px-4 py-2 border border-error text-error rounded-lg hover:bg-error/10 flex items-center justify-center gap-2">
          <Trash2 size={18} />
          {t('settings.resetAllData')}
        </button>
      </div>
    </div>
  )
}

function APISettings() {
  const { t } = useTranslation()
  const [remoteKey, setRemoteKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.apiKeys')}</h3>
      
      <div className="bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border p-4">
        <p className="text-sm text-text-muted dark:text-dark-text-muted mb-4">
          {t('settings.remoteApiDesc')}
        </p>
        
        <div className="relative">
          <input
            type={showKey ? 'text' : 'password'}
            value={remoteKey}
            onChange={e => setRemoteKey(e.target.value)}
            placeholder={t('settings.apiKeyPlaceholder')}
            className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface pr-10"
          />
          <button
            onClick={() => setShowKey(!showKey)}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary"
          >
            {showKey ? <EyeOff size={18} /> : <Eye size={18} />}
          </button>
        </div>
        
        <button className="mt-3 px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary-hover">
          {t('settings.saveApiKey')}
        </button>
      </div>
    </div>
  )
}

import { Eye, EyeOff } from 'lucide-react'

function PrivacySettings() {
  const { t } = useTranslation()
  const [analytics, setAnalytics] = useState(false)
  const [crashReports, setCrashReports] = useState(true)
  const [checkUpdates, setCheckUpdates] = useState(true)
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.privacy')}</h3>
      
      <div className="space-y-4">
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.analytics')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.analyticsDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={analytics}
            onChange={e => setAnalytics(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.crashReports')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.crashReportsDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={crashReports}
            onChange={e => setCrashReports(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.checkUpdates')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.checkUpdatesDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={checkUpdates}
            onChange={e => setCheckUpdates(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
      </div>
    </div>
  )
}

function NotificationsSettings() {
  const { t } = useTranslation()
  const [enabled, setEnabled] = useState(true)
  const [generationComplete, setGenerationComplete] = useState(true)
  const [errors, setErrors] = useState(true)
  const [updates, setUpdates] = useState(false)
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.notifications')}</h3>
      
      <div className="space-y-4">
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.notificationsEnabled')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.notificationsEnabledDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={enabled}
            onChange={e => setEnabled(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover opacity-50" disabled={!enabled}>
          <div>
            <p className="font-medium">{t('settings.generationComplete')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.generationCompleteDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={generationComplete}
            onChange={e => setGenerationComplete(e.target.checked)}
            className="w-5 h-5 accent-primary"
            disabled={!enabled}
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover opacity-50" disabled={!enabled}>
          <div>
            <p className="font-medium">{t('settings.errors')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.errorsDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={errors}
            onChange={e => setErrors(e.target.checked)}
            className="w-5 h-5 accent-primary"
            disabled={!enabled}
          />
        </label>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover opacity-50" disabled={!enabled}>
          <div>
            <p className="font-medium">{t('settings.updates')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.updatesDesc')}</p>
          </div>
          <input
            type="checkbox"
            checked={updates}
            onChange={e => setUpdates(e.target.checked)}
            className="w-5 h-5 accent-primary"
            disabled={!enabled}
          />
        </label>
      </div>
    </div>
  )
}

function AdvancedSettings() {
  const { t } = useTranslation()
  const [device, setDevice] = useState('auto')
  const [fp16, setFp16] = useState(true)
  const [gpuWorkers, setGpuWorkers] = useState(1)
  const [generateTimeout, setGenerateTimeout] = useState(300)
  const [cpuTimeout, setCpuTimeout] = useState(600)
  const [logLevel, setLogLevel] = useState('INFO')
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.advanced')}</h3>
      
      <div className="bg-error/5 border border-error/20 rounded-lg p-4 mb-4">
        <p className="text-sm text-error font-medium flex items-center gap-2">
          <AlertTriangle size={16} />
          {t('settings.advancedWarning')}
        </p>
      </div>
      
      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">{t('settings.computeDevice')}</label>
          <select
            value={device}
            onChange={e => setDevice(e.target.value)}
            className="w-full max-w-xs px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface"
          >
            <option value="auto">{t('settings.auto')}</option>
            <option value="cuda">CUDA (NVIDIA GPU)</option>
            <option value="mps">MPS (Apple Silicon)</option>
            <option value="cpu">CPU</option>
          </select>
        </div>
        
        <label className="flex items-center justify-between cursor-pointer p-3 rounded-lg hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
          <div>
            <p className="font-medium">{t('settings.fp16')}</p>
            <p className="text-sm text-text-muted dark:text-dark-text-muted">{t('settings.fp16Desc')}</p>
          </div>
          <input
            type="checkbox"
            checked={fp16}
            onChange={e => setFp16(e.target.checked)}
            className="w-5 h-5 accent-primary"
          />
        </label>
        
        <div>
          <label className="block text-sm font-medium mb-2">{t('settings.gpuWorkers')} ({gpuWorkers})</label>
          <input
            type="range"
            min="1"
            max="4"
            step="1"
            value={gpuWorkers}
            onChange={e => setGpuWorkers(parseInt(e.target.value))}
            className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary"
          />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">{t('settings.gpuTimeout')} (s)</label>
            <input
              type="number"
              value={generateTimeout}
              onChange={e => setGenerateTimeout(parseInt(e.target.value) || 300)}
              min="60"
              max="1800"
              className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">{t('settings.cpuTimeout')} (s)</label>
            <input
              type="number"
              value={cpuTimeout}
              onChange={e => setCpuTimeout(parseInt(e.target.value) || 600)}
              min="60"
              max="3600"
              className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface"
            />
          </div>
        </div>
        
        <div>
          <label className="block text-sm font-medium mb-2">{t('settings.logLevel')}</label>
          <select
            value={logLevel}
            onChange={e => setLogLevel(e.target.value)}
            className="w-full max-w-xs px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface"
          >
            <option value="DEBUG">Debug</option>
            <option value="INFO">Info</option>
            <option value="WARNING">Warning</option>
            <option value="ERROR">Error</option>
          </select>
        </div>
      </div>
    </div>
  )
}

import { AlertTriangle } from 'lucide-react'

function AboutSettings() {
  const { t } = useTranslation()
  
  return (
    <div className="max-w-2xl space-y-6">
      <h3 className="text-lg font-semibold mb-4">{t('settings.about')}</h3>
      
      <div className="bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border p-6 text-center">
        <div className="w-20 h-20 mx-auto mb-4 rounded-2xl bg-primary/10 dark:bg-primary/20 flex items-center justify-center">
          <Mic size={40} className="text-primary" aria-hidden="true />
        </div>
        <h4 className="text-xl font-semibold">VoiceStudio</h4>
        <p className="text-text-secondary dark:text-dark-text-secondary mt-1">Version 0.1.0</p>
        <p className="text-sm text-text-muted dark:text-dark-text-muted mt-2">
          {t('settings.aboutDesc')}
        </p>
        <div className="mt-6 flex flex-wrap justify-center gap-2">
          <a href="https://github.com/debpalash/VoiceStudio" target="_blank" rel="noopener noreferrer" className="px-3 py-1.5 text-sm border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
            GitHub
          </a>
          <a href="https://discord.gg/bzQavDfVV9" target="_blank" rel="noopener noreferrer" className="px-3 py-1.5 text-sm border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
            Discord
          </a>
          <a href="#" className="px-3 py-1.5 text-sm border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover">
            {t('settings.license')}
          </a>
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        <div className="p-3 bg-surface-hover dark:bg-dark-surface-hover rounded-lg">
          <p className="font-medium">{t('settings.pythonVersion')}</p>
          <p className="text-text-secondary dark:text-dark-text-secondary font-mono">3.11+</p>
        </div>
        <div className="p-3 bg-surface-hover dark:bg-dark-surface-hover rounded-lg">
          <p className="font-medium">{t('settings.torchVersion')}</p>
          <p className="text-text-secondary dark:text-dark-text-secondary font-mono">2.5+</p>
        </div>
        <div className="p-3 bg-surface-hover dark:bg-dark-surface-hover rounded-lg">
          <p className="font-medium">{t('settings.electronVersion')}</p>
          <p className="text-text-secondary dark:text-dark-text-secondary font-mono">44+</p>
        </div>
        <div className="p-3 bg-surface-hover dark:bg-dark-surface-hover rounded-lg">
          <p className="font-medium">{t('settings.reactVersion')}</p>
          <p className="text-text-secondary dark:text-dark-text-secondary font-mono">19+</p>
        </div>
      </div>
    </div>
  )
}

import { Mic } from 'lucide-react'

const sectionComponents = {
  appearance: AppearanceSettings,
  general: GeneralSettings,
  storage: StorageSettings,
  api: APISettings,
  privacy: PrivacySettings,
  notifications: NotificationsSettings,
  advanced: AdvancedSettings,
  about: AboutSettings,
}

export default function Settings() {
  const { t } = useTranslation()
  const [activeSection, setActiveSection] = useState('appearance')
  
  const SectionComponent = sectionComponents[activeSection as keyof typeof sectionComponents] || AppearanceSettings
  
  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-6">
        <h1 className="text-2xl font-semibold text-text-primary dark:text-dark-text-primary">
          {t('settings.title')}
        </h1>
        <p className="text-text-secondary dark:text-dark-text-secondary mt-1">
          {t('settings.description')}
        </p>
      </div>
      
      <div className="flex flex-col lg:flex-row gap-6">
        <SettingsNav activeSection={activeSection} onSelect={setActiveSection} />
        <div className="flex-1 p-6 lg:p-8 overflow-y-auto">
          <SectionComponent />
        </div>
      </div>
    </div>
  )
}
