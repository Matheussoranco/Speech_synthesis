import { apiJson } from './client'

export interface EngineInfo {
  id: string
  name: string
  type: 'tts' | 'asr' | 'diarization' | 'translation' | 'vocoder'
  status: 'available' | 'loading' | 'error' | 'unavailable'
  version: string
  languages: string[]
  models: string[]
  device: string
  memory_usage?: number
  capabilities: string[]
}

export interface EnginesResponse {
  tts: EngineInfo[]
  asr: EngineInfo[]
  diarization: EngineInfo[]
  translation: EngineInfo[]
  vocoder: EngineInfo[]
}

export interface SystemInfo {
  app_version: string
  python_version: string
  torch_version: string
  cuda_version?: string
  device: string
  gpu_name?: string
  gpu_memory_total?: number
  gpu_memory_free?: number
  cpu_count: number
  memory_total: number
  memory_free: number
  platform: string
  uptime: number
}

export async function getEngines(): Promise<EnginesResponse> {
  return apiJson<EnginesResponse>('/engines')
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return apiJson<SystemInfo>('/system/info')
}

export async function loadEngine(engineId: string): Promise<EngineInfo> {
  return apiJson<EngineInfo>(`/engines/${encodeURIComponent(engineId)}/load`, { method: 'POST' })
}

export async function unloadEngine(engineId: string): Promise<void> {
  await apiJson(`/engines/${encodeURIComponent(engineId)}/unload`, { method: 'POST' })
}
