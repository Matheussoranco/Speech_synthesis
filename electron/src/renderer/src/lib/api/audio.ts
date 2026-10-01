import { apiFormData } from './client'

export interface CleanAudioResult {
  audio_path: string
  original_duration: number
  cleaned_duration: number
  noise_reduction_db: number
}

export async function cleanAudio(
  blob: Blob,
  filename: string
): Promise<CleanAudioResult> {
  const formData = new FormData()
  formData.append('audio', blob, filename)
  
  return apiFormData<CleanAudioResult>('/clean-audio', formData, {
    headers: {
      'X-Clean-Filename': filename,
    },
  })
}
