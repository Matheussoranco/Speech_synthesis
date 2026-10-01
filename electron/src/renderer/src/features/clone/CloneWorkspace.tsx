import { useState, useCallback, useEffect, useRef } from 'react'
import { useTranslation } from 'react-i18next'
import { 
  Mic, 
  Upload, 
  X, 
  Play, 
  Pause, 
  Download, 
  Copy, 
  Check, 
  AlertCircle,
  Info,
  SlidersHorizontal,
  FolderOpen,
  RotateCcw,
  Trash2,
  Star,
  Volume2,
  VolumeX,
} from 'lucide-react'
import { toast } from 'sonner'
import { useGenerateClone } from '../../hooks/use-generate'
import { useRecording } from '../../hooks/use-recording'
import { referenceStore, setReferenceFile, clearReference } from '../../lib/store/reference'
import { cloneSettingsStore, useCloneSettings, patchCloneSettings, resetCloneOverrides } from '../../lib/store/clone-settings'
import { useProfiles, useCreateProfile } from '../../hooks/use-profiles'
import { apiJson } from '../../lib/api'
import { clsx } from 'clsx'

// Language list (bundled)
const LANGUAGES = [
  'auto', 'en', 'zh', 'es', 'fr', 'de', 'ja', 'ko', 'pt', 'it', 'ru', 'ar', 'hi', 'nl', 'pl', 'tr', 'vi', 'th', 'id', 'ms'
]

const POPULAR_LANGUAGES = ['en', 'zh', 'es', 'fr', 'de', 'ja', 'ko', 'pt', 'it']

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const mins = Math.floor(seconds / 60)
  const secs = seconds % 60
  return `${mins}:${secs.toFixed(1).padStart(4, '0')}`
}

function ReferenceAudio({ onChange, onClear, onRecord }: { 
  onChange: (file: File) => void
  onClear: () => void
  onRecord: () => void
}) {
  const { t } = useTranslation()
  const { file, duration_seconds, object_url } = referenceStore.useStore()
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile && droppedFile.type.startsWith('audio/')) {
      onChange(droppedFile)
    }
  }, [onChange])
  
  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(true)
  }, [])
  
  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragActive(false)
  }, [])
  
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile) onChange(selectedFile)
    if (inputRef.current) inputRef.current.value = ''
  }, [onChange])
  
  if (file && object_url) {
    return (
      <div className="relative group">
        <audio 
          ref={audioRef => { if (audioRef) audioRef.src = object_url }} 
          controls 
          className="w-full rounded-lg bg-surface-hover dark:bg-dark-surface-hover"
        />
        <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <button
            onClick={onClear}
            className="p-1.5 rounded-lg bg-error/10 text-error hover:bg-error/20 transition-colors"
            aria-label={t('clone.clearReference')}
          >
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <div className="mt-2 flex items-center justify-between text-sm text-text-muted dark:text-dark-text-muted">
          <span>{file.name}</span>
          <span>{duration_seconds ? formatDuration(duration_seconds) : t('clone.unknownDuration')}</span>
        </div>
      </div>
    )
  }
  
  return (
    <div 
      className={clsx(
        'border-2 border-dashed rounded-xl p-8 text-center transition-colors',
        dragActive 
          ? 'border-primary bg-primary/5 dark:border-primary dark:bg-primary/10' 
          : 'border-border dark:border-dark-border hover:border-primary/50'
      )}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      role="button"
      tabIndex={0}
      onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputRef.current?.click() }}}
    >
      <input 
        ref={inputRef} 
        type="file" 
        accept="audio/*" 
        onChange={handleFileSelect} 
        className="hidden" 
        id="ref-audio-upload"
      />
      <label htmlFor="ref-audio-upload" className="cursor-pointer">
        <Upload size={48} className="mx-auto mb-4 text-text-muted dark:text-dark-text-muted" aria-hidden="true" />
        <p className="text-lg font-medium text-text-primary dark:text-dark-text-primary mb-1">
          {t('clone.dropReference')}
        </p>
        <p className="text-text-secondary dark:text-dark-text-secondary mb-4">
          {t('clone.dropReferenceDesc')}
        </p>
        <div className="flex items-center justify-center gap-4">
          <button
            type="button"
            onClick={() => inputRef.current?.click()}
            className="px-4 py-2 border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover transition-colors flex items-center gap-2"
          >
            <FolderOpen size={18} aria-hidden="true" />
            {t('clone.browse')}
          </button>
          <button
            type="button"
            onClick={onRecord}
            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary-hover transition-colors flex items-center gap-2"
          >
            <Mic size={18} aria-hidden="true" />
            {t('clone.record')}
          </button>
        </div>
      </label>
    </div>
  )
}

function SavedVoicesPanel({ onSelect }: { onSelect: (profile: any) => void }) {
  const { t } = useTranslation()
  const { data: profiles, isLoading } = useProfiles()
  const { selected_profile_id } = useCloneSettings()
  const createProfile = useCreateProfile()
  const [showCreate, setShowCreate] = useState(false)
  const [newName, setNewName] = useState('')
  const [newRefText, setNewRefText] = useState('')
  
  if (isLoading) {
    return <div className="p-4 text-center text-text-muted dark:text-dark-text-muted">{t('common.loading')}</div>
  }
  
  if (showCreate) {
    return (
      <div className="space-y-3 p-3 bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border">
        <h4 className="font-medium">{t('profiles.createNew')}</h4>
        <input
          type="text"
          value={newName}
          onChange={e => setNewName(e.target.value)}
          placeholder={t('profiles.namePlaceholder')}
          className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface text-text-primary dark:text-dark-text-primary"
        />
        <textarea
          value={newRefText}
          onChange={e => setNewRefText(e.target.value)}
          placeholder={t('profiles.refTextPlaceholder')}
          rows={2}
          className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface text-text-primary dark:text-dark-text-primary"
        />
        <div className="flex gap-2">
          <button 
            onClick={() => { setShowCreate(false); setNewName(''); setNewRefText('') }}
            className="flex-1 px-3 py-2 border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover"
          >
            {t('common.cancel')}
          </button>
          <button
            onClick={async () => {
              // Note: This would need the reference audio file from the reference store
              // For now just close
              setShowCreate(false)
              setNewName('')
              setNewRefText('')
            }}
            className="flex-1 px-3 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary-hover"
            disabled={!newName.trim()}
          >
            {t('common.create')}
          </button>
        </div>
      </div>
    )
  }
  
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="font-medium">{t('profiles.savedVoices')}</h4>
        <button
          onClick={() => setShowCreate(true)}
          className="px-3 py-1.5 text-sm bg-primary text-primary-foreground rounded-lg hover:bg-primary-hover transition-colors"
        >
          <Plus size={14} aria-hidden="true" />
          {t('profiles.add')}
        </button>
      </div>
      {profiles.length === 0 ? (
        <p className="text-sm text-text-muted dark:text-dark-text-muted text-center py-4">
          {t('profiles.empty')}
        </p>
      ) : (
        <div className="space-y-2 max-h-60 overflow-y-auto thin-scrollbar">
          {profiles.map(profile => (
            <button
              key={profile.id}
              onClick={() => onSelect(profile)}
              className={clsx(
                'w-full p-3 rounded-lg border transition-colors text-left flex items-center gap-3',
                selected_profile_id === profile.id
                  ? 'border-primary bg-primary/10 dark:bg-primary/20'
                  : 'border-border dark:border-dark-border hover:bg-surface-hover dark:hover:bg-dark-surface-hover'
              )}
            >
              <div className="w-10 h-10 rounded-lg bg-primary/10 dark:bg-primary/20 flex items-center justify-center flex-shrink-0">
                <Mic size={18} className="text-primary" aria-hidden="true" />
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{profile.name}</p>
                <p className="text-xs text-text-muted dark:text-dark-text-muted truncate">{profile.language}</p>
              </div>
              <button
                onClick={e => { e.stopPropagation(); /* delete */ }}
                className="p-1 text-text-muted hover:text-error transition-colors"
                aria-label={t('profiles.delete')}
              >
                <Trash2 size={16} aria-hidden="true" />
              </button>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

import { Plus } from 'lucide-react'
import { useState } from 'react'
import { useCloneSettings } from '../../lib/store/clone-settings'

function Composer() {
  const { t } = useTranslation()
  const { isGenerating, progress, elapsedSeconds, generate, cancel } = useGenerateClone()
  const { 
    speed, cfg, steps, t_shift, pos_temp, class_temp, layer_penalty, 
    denoise, postprocess, duration, show_overrides, setCloneSetting 
  } = useCloneSettings()
  const { auto_play } = useCloneSettings()
  const [showAdvanced, setShowAdvanced] = useState(false)
  
  return (
    <div className="border-t border-border dark:border-dark-border pt-4 space-y-4">
      {/* Language & Speed row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
            {t('clone.language')}
          </label>
          <select
            value={useCloneSettings().language}
            onChange={e => setCloneSetting('language', e.target.value)}
            className="w-full px-3 py-2 border border-border dark:border-dark-border rounded-lg bg-surface dark:bg-dark-surface text-text-primary dark:text-dark-text-primary"
          >
            {POPULAR_LANGUAGES.map(lang => (
              <option key={lang} value={lang}>{lang.toUpperCase()}</option>
            ))}
            <option value="" disabled>────────</option>
            {LANGUAGES.filter(l => !POPULAR_LANGUAGES.includes(l)).map(lang => (
              <option key={lang} value={lang}>{lang.toUpperCase()}</option>
            ))}
          </select>
        </div>
        
        <div>
          <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
            {t('clone.speed')} ({speed.toFixed(1)}x)
          </label>
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.1"
            value={speed}
            onChange={e => setCloneSetting('speed', parseFloat(e.target.value))}
            className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary"
          />
        </div>
      </div>
      
      {/* Advanced toggles */}
      <button
        type="button"
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="flex items-center gap-2 text-sm text-text-secondary dark:text-dark-text-secondary hover:text-text-primary dark:hover:text-dark-text-primary"
      >
        <SlidersHorizontal size={18} aria-hidden="true" />
        <span>{showAdvanced ? t('clone.hideAdvanced') : t('clone.showAdvanced')}</span>
        <Info size={14} className="text-text-muted dark:text-dark-text-muted" aria-hidden="true" />
      </button>
      
      {showAdvanced && (
        <div className="space-y-4 p-4 bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.steps')} ({steps})
              </label>
              <input type="range" min="10" max="100" step="1" value={steps} onChange={e => setCloneSetting('steps', parseInt(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.cfg')} ({cfg.toFixed(1)})
              </label>
              <input type="range" min="0" max="2" step="0.1" value={cfg} onChange={e => setCloneSetting('cfg', parseFloat(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.tShift')} ({t_shift.toFixed(1)})
              </label>
              <input type="range" min="-1" max="1" step="0.1" value={t_shift} onChange={e => setCloneSetting('t_shift', parseFloat(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.duration')} ({duration || t('clone.auto')})
              </label>
              <input type="range" min="0" max="30" step="1" value={duration} onChange={e => setCloneSetting('duration', parseInt(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.posTemp')} ({pos_temp.toFixed(1)})
              </label>
              <input type="range" min="0" max="1.5" step="0.1" value={pos_temp} onChange={e => setCloneSetting('pos_temp', parseFloat(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.classTemp')} ({class_temp.toFixed(1)})
              </label>
              <input type="range" min="0" max="1.5" step="0.1" value={class_temp} onChange={e => setCloneSetting('class_temp', parseFloat(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
            <div>
              <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-1">
                {t('clone.layerPenalty')} ({layer_penalty.toFixed(1)})
              </label>
              <input type="range" min="0" max="2" step="0.1" value={layer_penalty} onChange={e => setCloneSetting('layer_penalty', parseFloat(e.target.value))} className="w-full h-2 bg-border dark:bg-dark-border rounded-lg appearance-none accent-primary" />
            </div>
          </div>
          
          <div className="flex flex-wrap gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={denoise} onChange={e => setCloneSetting('denoise', e.target.checked)} className="w-4 h-4 accent-primary" />
              <span className="text-sm text-text-secondary dark:text-dark-text-secondary">{t('clone.denoise')}</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={postprocess} onChange={e => setCloneSetting('postprocess', e.target.checked)} className="w-4 h-4 accent-primary" />
              <span className="text-sm text-text-secondary dark:text-dark-text-secondary">{t('clone.postprocess')}</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={auto_play} onChange={e => setCloneSetting('auto_play', e.target.checked)} className="w-4 h-4 accent-primary" />
              <span className="text-sm text-text-secondary dark:text-dark-text-secondary">{t('clone.autoPlay')}</span>
            </label>
          </div>
          
          <button
            type="button"
            onClick={resetCloneOverrides}
            className="px-3 py-1.5 text-sm border border-border dark:border-dark-border rounded-lg text-text-secondary dark:text-dark-text-secondary hover:bg-surface-hover dark:hover:bg-dark-surface-hover"
          >
            <RotateCcw size={14} aria-hidden="true" />
            {t('clone.resetDefaults')}
          </button>
        </div>
      )}
      
      {/* Generate button */}
      <button
        type="button"
        onClick={isGenerating ? cancel : generate}
        disabled={isGenerating && progress === null}
        className={clsx(
          'w-full py-3 rounded-lg font-semibold text-lg flex items-center justify-center gap-2 transition-colors',
          isGenerating
            ? 'bg-warning text-warning-foreground'
            : 'bg-primary text-primary-foreground hover:bg-primary-hover'
        )}
      >
        {isGenerating ? (
          <>
            <Loader2 size={20} className="animate-spin" aria-hidden="true" />
            {progress !== null ? `${Math.round(progress)}%` : t('clone.preparing')}
          </>
        ) : (
          <>
            <Mic size={20} aria-hidden="true" />
            {t('clone.generate')}
          </>
        )}
      </button>
      
      {isGenerating && progress !== null && (
        <div className="h-1 bg-border dark:bg-dark-border rounded-full overflow-hidden">
          <div 
            className="h-full bg-primary rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
            aria-hidden="true"
          />
        </div>
      )}
      
      {isGenerating && (
        <p className="text-xs text-text-muted dark:text-dark-text-muted text-center">
          {t('clone.elapsed', { time: formatDuration(elapsedSeconds) })}
        </p>
      )}
    </div>
  )
}

import { Loader2 } from 'lucide-react'
import { useCloneSettings } from '../../lib/store/clone-settings'

function OutputPlayer() {
  const { t } = useTranslation()
  const { result, object_url, text } = useLatestOutput()
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement>(null)
  
  if (!result) return null
  
  const play = async () => {
    if (!audioRef.current) {
      audioRef.current = new Audio(`/api/audio/${encodeURIComponent(result.audio_path.split('/').pop() || '')}`)
      audioRef.current.onended = () => setPlaying(false)
      audioRef.current.onerror = () => { setPlaying(false); toast.error(t('clone.playbackError')) }
    }
    try {
      await audioRef.current.play()
      setPlaying(true)
    } catch {
      toast.error(t('clone.playbackError'))
    }
  }
  
  const pause = () => {
    audioRef.current?.pause()
    setPlaying(false)
  }
  
  return (
    <div className="mt-6 space-y-3 animate-in">
      <div className="flex items-center justify-between">
        <h3 className="font-medium">{t('clone.output')}</h3>
        <div className="flex items-center gap-2">
          <button onClick={playing ? pause : play} className="p-2 rounded-lg bg-surface-hover dark:bg-dark-surface-hover hover:bg-surface-active dark:hover:bg-dark-surface-active transition-colors">
            {playing ? <Pause size={20} /> : <Play size={20} />}
          </button>
          <a 
            href={`/api/audio/${encodeURIComponent(result.audio_path.split('/').pop() || '')}`}
            download={`${text.slice(0, 30)}.wav`}
            className="p-2 rounded-lg bg-surface-hover dark:bg-dark-surface-hover hover:bg-surface-active dark:hover:bg-dark-surface-active transition-colors"
          >
            <Download size={20} />
          </a>
        </div>
      </div>
      
      <audio ref={audioRef} src={`/api/audio/${encodeURIComponent(result.audio_path.split('/').pop() || '')}`} onEnded={() => setPlaying(false)} />
      
      <div className="w-full h-32 bg-surface-hover dark:bg-dark-surface-hover rounded-lg border border-border dark:border-dark-border relative overflow-hidden">
        {/* Waveform placeholder - would use wavesurfer.js in production */}
        <canvas className="w-full h-full" />
      </div>
      
      <div className="text-sm text-text-muted dark:text-dark-text-muted">
        {t('clone.generated', { duration: formatDuration(result.duration), engine: result.engine })}
      </div>
    </div>
  )
}

import { useLatestOutput } from '../../lib/store/output'

export default function CloneWorkspace() {
  const { t } = useTranslation()
  const { text, setCloneSetting: setText, selected_profile_id, setCloneSetting: setProfile } = useCloneSettings()
  const reference = referenceStore.useStore()
  
  const handleReferenceChange = useCallback(async (file: File) => {
    const result = await setReferenceFile(file)
    if (!result.ok) {
      toast.error(t('clone.invalidAudio'))
      return
    }
    if (result.too_long) {
      toast.warning(t('clone.referenceTooLongWarning'))
    }
    // Clear selected profile when new reference is set
    setProfile('selected_profile_id', null)
  }, [setProfile])
  
  const handleReferenceClear = useCallback(() => {
    clearReference()
  }, [])
  
  const handleProfileSelect = useCallback((profile: any) => {
    setProfile('selected_profile_id', profile.id)
    // Would also load ref_text, instruct, language from profile
  }, [setProfile])
  
  const handleRecord = useCallback(() => {
    // Recording is handled by the recording hook in the reference component
  }, [])
  
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-semibold text-text-primary dark:text-dark-text-primary">
          {t('clone.title')}
        </h1>
        <p className="text-text-secondary dark:text-dark-text-secondary mt-1">
          {t('clone.description')}
        </p>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Script Editor */}
        <div className="lg:col-span-2 space-y-4">
          <div>
            <label className="block text-sm font-medium text-text-secondary dark:text-dark-text-secondary mb-2">
              {t('clone.scriptLabel')}
            </label>
            <textarea
              value={text}
              onChange={e => setText('text', e.target.value)}
              placeholder={t('clone.scriptPlaceholder')}
              rows={8}
              className="w-full px-4 py-3 border border-border dark:border-dark-border rounded-xl bg-surface dark:bg-dark-surface text-text-primary dark:text-dark-text-primary placeholder-text-muted dark:placeholder-dark-text-muted resize-y focus:outline-none focus:ring-2 focus:ring-primary/20 focus:border-primary transition-all"
              spellCheck={false}
            />
            <p className="mt-1 text-xs text-text-muted dark:text-dark-text-muted text-right">
              {text.length} / 5000 {t('common.characters')}
            </p>
          </div>
          
          <Composer />
        </div>
        
        {/* Right: Reference & Saved Voices */}
        <div className="space-y-4">
          <div className="sticky top-24 space-y-4">
            <ReferenceAudio 
              onChange={handleReferenceChange}
              onClear={handleReferenceClear}
              onRecord={handleRecord}
            />
            
            <SavedVoicesPanel onSelect={handleProfileSelect} />
          </div>
        </div>
      </div>
      
      {/* Output */}
      <OutputPlayer />
    </div>
  )
}
