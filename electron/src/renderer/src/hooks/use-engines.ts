import { useQuery } from '@tanstack/react-query'
import { queryKeys } from '../lib/api/query'
import { getEngines, getSystemInfo } from '../lib/api/engines'

export function useEngines() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: queryKeys.engines,
    queryFn: getEngines,
    refetchInterval: 10000, // Refetch every 10 seconds
  })
  
  const engines = data || { tts: [], asr: [], diarization: [], translation: [], vocoder: [] }
  const anyTtsReady = engines.tts.some(e => e.status === 'available')
  const activeTts = engines.tts.find(e => e.status === 'available') || null
  
  return {
    data: engines,
    anyTtsReady,
    activeTts,
    isLoading,
    error,
    refetch,
  }
}

export function useSystemInfo() {
  return useQuery({
    queryKey: queryKeys.systemInfo,
    queryFn: getSystemInfo,
    refetchInterval: 30000, // Refetch every 30 seconds
  })
}
