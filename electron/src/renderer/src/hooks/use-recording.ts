import { useCallback, useRef, useState, useEffect } from 'react'
import { useTranslation } from 'react-i18next'
import { toast } from 'sonner'

export interface UseRecordingOptions {
  onRecorded: (file: File) => void
  maxDuration?: number
}

export function useRecording({ onRecorded, maxDuration = 60 }: UseRecordingOptions) {
  const { t } = useTranslation()
  const [isRecording, setIsRecording] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [level, setLevel] = useState(0)
  const [inputs, setInputs] = useState<MediaDeviceInfo[]>([])
  const [selectedInputId, setSelectedInputId] = useState('default')
  const [channelMode, setChannelMode] = useState<'auto' | 'mono' | 'stereo'>('auto')
  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const contextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number | null>(null)
  const startingRef = useRef(false)
  const mountedRef = useRef(true)
  const generationRef = useRef(0)
  const callbackRef = useRef(onRecorded)
  callbackRef.current = onRecorded

  const stop = useCallback(() => {
    if (startingRef.current && !recorderRef.current) generationRef.current++
    startingRef.current = false
    const recorder = recorderRef.current
    recorderRef.current = null
    if (recorder && recorder.state !== 'inactive') recorder.stop()
    streamRef.current?.getTracks().forEach(track => track.stop())
    streamRef.current = null
    if (animationRef.current !== null) cancelAnimationFrame(animationRef.current)
    animationRef.current = null
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    if (contextRef.current && contextRef.current.state !== 'closed') void contextRef.current.close()
    contextRef.current = null
    analyserRef.current = null
    if (mountedRef.current) { setIsRecording(false); setIsStarting(false); setLevel(0) }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    navigator.mediaDevices?.enumerateDevices().then(devices => {
      if (mountedRef.current) setInputs(devices.filter(device => device.kind === 'audioinput'))
    }).catch(() => {})
    return () => { mountedRef.current = false; generationRef.current++; stop() }
  }, [stop])

  const updateLevel = useCallback(function readLevel() {
    const analyser = analyserRef.current
    if (!analyser || !mountedRef.current) return
    const data = new Uint8Array(analyser.frequencyBinCount)
    analyser.getByteFrequencyData(data)
    setLevel(data.reduce((sum, value) => sum + value, 0) / Math.max(1, data.length) / 255)
    animationRef.current = requestAnimationFrame(readLevel)
  }, [])

  const start = useCallback(async () => {
    if (startingRef.current || recorderRef.current) return
    startingRef.current = true
    const generation = ++generationRef.current
    setIsStarting(true); setSeconds(0)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: {
        deviceId: selectedInputId === 'default' ? undefined : { exact: selectedInputId },
        channelCount: channelMode === 'mono' ? 1 : channelMode === 'stereo' ? 2 : undefined,
        echoCancellation: true, noiseSuppression: true, autoGainControl: true,
      } })
      if (!mountedRef.current || generationRef.current !== generation) {
        stream.getTracks().forEach(track => track.stop())
        return
      }
      streamRef.current = stream
      const context = new AudioContext()
      contextRef.current = context
      const analyser = context.createAnalyser()
      analyser.fftSize = 256
      analyserRef.current = analyser
      context.createMediaStreamSource(stream).connect(analyser)
      const mimeType = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'].find(type => MediaRecorder.isTypeSupported(type))
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined)
      recorderRef.current = recorder
      const chunks: Blob[] = []
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data) }
      recorder.onstop = () => {
        if (!mountedRef.current || generationRef.current !== generation || !chunks.length) return
        const type = recorder.mimeType || mimeType || 'audio/webm'
        const extension = type.includes('mp4') ? 'm4a' : type.includes('ogg') ? 'ogg' : 'webm'
        callbackRef.current(new File(chunks, `recording-${Date.now()}.${extension}`, { type }))
      }
      recorder.onerror = () => { stop(); toast.error(t('recording.failed')) }
      recorder.start(100)
      startingRef.current = false
      setIsRecording(true); setIsStarting(false)
      updateLevel()
      let elapsed = 0
      timerRef.current = setInterval(() => {
        elapsed++
        setSeconds(elapsed)
        if (maxDuration > 0 && elapsed >= maxDuration) stop()
      }, 1000)
    } catch {
      const report = mountedRef.current && generationRef.current === generation
      stop()
      if (report) toast.error(t('recording.failed'))
    }
  }, [selectedInputId, channelMode, maxDuration, stop, updateLevel, t])

  return { isRecording, isStarting, isCleaning: false, seconds, level, inputs, selectedInputId,
    setSelectedInputId, channelMode, setChannelMode, start, stop, cleanup: stop }
}
