import { useCallback, useRef, useState, useEffect } from 'react'
import { toast } from 'sonner'

export interface UseRecordingOptions {
  onRecorded: (file: File) => void
  maxDuration?: number
}

export function useRecording({ onRecorded, maxDuration = 60 }: UseRecordingOptions) {
  const [isRecording, setIsRecording] = useState(false)
  const [isStarting, setIsStarting] = useState(false)
  const [isCleaning, setIsCleaning] = useState(false)
  const [seconds, setSeconds] = useState(0)
  const [level, setLevel] = useState(0)
  const [inputs, setInputs] = useState<MediaDeviceInfo[]>([])
  const [selectedInputId, setSelectedInputId] = useState<string>('default')
  const [channelMode, setChannelMode] = useState<'auto' | 'mono' | 'stereo'>('auto')
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number>()

  // Enumerate inputs
  useEffect(() => {
    const enumerate = async () => {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        const audioInputs = devices.filter(d => d.kind === 'audioinput')
        setInputs(audioInputs)
        if (audioInputs.length > 0 && selectedInputId === 'default') {
          setSelectedInputId(audioInputs[0].deviceId)
        }
      } catch (e) {
        console.error('Failed to enumerate devices:', e)
      }
    }
    enumerate()
  }, [selectedInputId])

  // Level monitoring
  const updateLevel = useCallback(() => {
    if (!analyserRef.current) return
    const data = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(data)
    const avg = data.reduce((a, b) => a + b, 0) / data.length
    setLevel(avg / 255)
    animationRef.current = requestAnimationFrame(updateLevel)
  }, [])

  const start = useCallback(async () => {
    if (isRecording || isStarting) return
    
    setIsStarting(true)
    setIsCleaning(false)
    
    try {
      const constraints: MediaStreamConstraints = {
        audio: {
          deviceId: selectedInputId === 'default' ? undefined : { exact: selectedInputId },
          channelCount: channelMode === 'mono' ? 1 : channelMode === 'stereo' ? 2 : undefined,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      }
      
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      
      // Set up level monitoring
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      analyserRef.current = analyser
      source.connect(analyser)
      updateLevel()
      
      // Determine MIME type
      let mimeType = 'audio/webm;codecs=opus'
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/webm'
      }
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/mp4'
      }
      
      const recorder = new MediaRecorder(stream, { mimeType })
      mediaRecorderRef.current = recorder
      chunksRef.current = []
      
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data)
      }
      
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: mimeType })
        const file = new File([blob], `recording-${Date.now()}.webm`, { type: mimeType })
        onRecorded(file)
      }
      
      recorder.start(100) // Collect data every 100ms
      setIsRecording(true)
      setIsStarting(false)
      setSeconds(0)
      
      // Duration timer
      intervalRef.current = setInterval(() => {
        setSeconds(s => {
          const next = s + 1
          if (maxDuration && next >= maxDuration) {
            stop()
          }
          return next
        })
      }, 1000)
      
    } catch (error) {
      console.error('Recording failed:', error)
      toast.error('Failed to start recording')
      setIsStarting(false)
    }
  }, [selectedInputId, channelMode, maxDuration, onRecorded, updateLevel])

  const stop = useCallback(() => {
    if (!isRecording) return
    
    mediaRecorderRef.current?.stop()
    streamRef.current?.getTracks().forEach(t => t.stop())
    
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
    }
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    audioContextRef.current?.close()
    
    setIsRecording(false)
    setSeconds(0)
    setLevel(0)
  }, [isRecording])

  const cleanup = useCallback(() => {
    stop()
    setIsCleaning(true)
    // Additional cleanup if needed
    setTimeout(() => setIsCleaning(false), 500)
  }, [stop])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stop()
      if (animationRef.current) cancelAnimationFrame(animationRef.current)
      if (intervalRef.current) clearInterval(intervalRef.current)
      audioContextRef.current?.close()
    }
  }, [stop])

  return {
    isRecording,
    isStarting,
    isCleaning,
    seconds,
    level,
    inputs,
    selectedInputId,
    setSelectedInputId,
    channelMode,
    setChannelMode,
    start,
    stop,
    cleanup,
  }
}
