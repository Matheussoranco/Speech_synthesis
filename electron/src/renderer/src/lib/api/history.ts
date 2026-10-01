import { apiJson, apiFetch } from './client'

export interface HistoryItem {
  id: string
  type: 'clone' | 'tts' | 'dubbing' | 'design' | 'transcribe'
  text: string
  audio_path: string
  duration: number
  language: string
  engine: string
  metadata: Record<string, unknown>
  created_at: string
  starred: boolean
}

export async function listHistory(limit?: number, offset?: number): Promise<HistoryItem[]> {
  const params = new URLSearchParams()
  if (limit) params.append('limit', String(limit))
  if (offset) params.append('offset', String(offset))
  const query = params.toString() ? `?${params.toString()}` : ''
  return apiJson<HistoryItem[]>(`/history${query}`)
}

export async function clearHistory(): Promise<void> {
  await apiFetch('/history', { method: 'DELETE' })
}

export async function deleteHistoryItem(id: string): Promise<void> {
  await apiFetch(`/history/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function setHistoryStarred(id: string, starred: boolean): Promise<HistoryItem> {
  return apiJson<HistoryItem>(`/history/${encodeURIComponent(id)}/star`, {
    method: 'PATCH',
    body: JSON.stringify({ starred }),
  })
}
