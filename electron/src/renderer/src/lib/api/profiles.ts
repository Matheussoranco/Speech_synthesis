import { apiJson, apiFormData, apiFetch } from './client'

export interface Profile {
  id: string
  name: string
  ref_audio_path: string
  ref_audio_name: string
  ref_text: string
  instruct: string
  language: string
  created_at: string
  updated_at: string
  starred: boolean
}

export interface CreateProfileInput {
  name: string
  ref_audio: File
  ref_audio_name: string
  ref_text: string
  instruct: string
  language: string
}

export async function listProfiles(): Promise<Profile[]> {
  return apiJson<Profile[]>('/profiles')
}

export async function createProfile(input: CreateProfileInput): Promise<Profile> {
  const formData = new FormData()
  formData.append('name', input.name)
  formData.append('ref_audio', input.ref_audio)
  formData.append('ref_audio_name', input.ref_audio_name)
  formData.append('ref_text', input.ref_text)
  formData.append('instruct', input.instruct)
  formData.append('language', input.language)
  
  return apiJson<Profile>('/profiles', {
    method: 'POST',
    body: formData,
  })
}

export async function deleteProfile(id: string): Promise<void> {
  await apiFetch(`/profiles/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export async function getProfile(id: string): Promise<Profile> {
  return apiJson<Profile>(`/profiles/${encodeURIComponent(id)}`)
}

export async function updateProfile(id: string, updates: Partial<CreateProfileInput>): Promise<Profile> {
  return apiJson<Profile>(`/profiles/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  })
}
