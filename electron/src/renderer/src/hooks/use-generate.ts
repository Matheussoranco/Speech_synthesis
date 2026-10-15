import { useCallback, useEffect, useRef, useState } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'
import { ApiError } from '../lib/api/client'
import { generateClone, type CloneGenerateInput } from '../lib/api/generate'
import { setLatestOutput } from '../lib/store/output'
import { useCloneSettings } from '../lib/store/clone-settings'
import { useReference } from '../lib/store/reference'
import { queryClient } from '../lib/api/query'
import { queryKeys } from '../lib/api/query'

export function useGenerateClone() {
  const { t } = useTranslation()
  const [isGenerating, setIsGenerating] = useState(false)
  const [progress, setProgress] = useState<number | null>(null)
  const [elapsedSeconds, setElapsedSeconds] = useState(0)
  const abortRef = useRef<AbortController | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const cloneSettings = useCloneSettings()
  const reference = useReference()
  useEffect(() => () => {
    abortRef.current?.abort()
    abortRef.current = null
    if (intervalRef.current) clearInterval(intervalRef.current)
    intervalRef.current = null
  }, [])

  const cancel = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort()
      abortRef.current = null
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    setIsGenerating(false)
    setProgress(null)
    setElapsedSeconds(0)
  }, [])

  const generate = useCallback(async () => {
    if (abortRef.current) return
    const { text, language, selected_profile_id } = cloneSettings
    const { file: ref_file, duration_seconds: ref_duration } = reference
    
    // Validation
    if (!text.trim()) {
      toast.error(t('generate.errors.noText'))
      return
    }
    
    const hasReference = ref_file || selected_profile_id
    if (!hasReference) {
      toast.error(t('generate.errors.noReference'))
      return
    }
    
    if (ref_file && ref_duration && ref_duration > 75) {
      toast.error(t('generate.errors.referenceTooLong'))
      return
    }

    // Build input from settings + reference
    const input: CloneGenerateInput = {
      text: text.trim(),
      language,
      profile_id: selected_profile_id || undefined,
      instruct: cloneSettings.instruct || undefined,
      ref_text: cloneSettings.ref_text || undefined,
      steps: cloneSettings.steps,
      cfg: cloneSettings.cfg,
      speed: cloneSettings.speed,
      t_shift: cloneSettings.t_shift,
      pos_temp: cloneSettings.pos_temp,
      class_temp: cloneSettings.class_temp,
      layer_penalty: cloneSettings.layer_penalty,
      denoise: cloneSettings.denoise,
      postprocess: cloneSettings.postprocess,
      duration: cloneSettings.duration || undefined,
      ref_audio: ref_file || undefined,
    }

    // Start generation
    setIsGenerating(true)
    setProgress(null)
    setElapsedSeconds(0)
    
    const controller = new AbortController()
    abortRef.current = controller
    const startTime = Date.now()
    
    // Progress timer
    intervalRef.current = setInterval(() => {
      setElapsedSeconds((Date.now() - startTime) / 1000)
      // The elapsed timer is factual; progress stays indeterminate until reported.
    }, 500)

    try {
      const result = await generateClone(input, {
        signal: controller.signal,
        onProgress: (pct) => {
          if (pct !== null && abortRef.current === controller) setProgress(pct)
        },
      })

      if (controller.signal.aborted || abortRef.current !== controller) return
      // Success
      setProgress(100)
      setLatestOutput(result, text)
      
      // Auto-play
      if (cloneSettings.auto_play && result.audio_path) {
        const audio = new Audio(`/api/audio/${encodeURIComponent(result.audio_path.split('/').pop() || '')}`)
        audio.play().catch(() => {})
      }
      
      toast.success(t('generate.success'), {
        description: t('generate.successDesc', { duration: result.duration.toFixed(1) }),
      })
      
      // Invalidate history
      queryClient.invalidateQueries({ queryKey: queryKeys.history })
      
      return result
    } catch (error: unknown) {
      if (abortRef.current !== controller) return
      if (error instanceof Error && error.name === 'AbortError') {
        toast.info(t('generate.cancelled'))
      } else if (error instanceof ApiError) {
        const apiError = error
        if (apiError.status === 422 && apiError.detail.includes('similarity')) {
          toast.error(t('generate.errors.lowSimilarity'))
        } else {
          toast.error(t('generate.errors.failed'), { description: apiError.detail })
        }
      } else {
        toast.error(t('generate.errors.failed'))
      }
      throw error
    } finally {
      if (abortRef.current === controller) {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
      setIsGenerating(false)
      setProgress(null)
      setElapsedSeconds(0)
      abortRef.current = null
      }
    }
  }, [cloneSettings, reference, setLatestOutput, t])

  return {
    generate,
    cancel,
    isGenerating,
    progress,
    elapsedSeconds,
  }
}
