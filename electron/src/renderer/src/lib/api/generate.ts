import { apiJson, apiFormData } from './client'

export interface CloneGenerateInput {
  text: string
  language: string
  ref_audio?: File
  ref_audio_path?: string
  profile_id?: string
  instruct?: string
  ref_text?: string
  steps?: number
  cfg?: number
  speed?: number
  t_shift?: number
  pos_temp?: number
  class_temp?: number
  layer_penalty?: number
  denoise?: boolean
  postprocess?: boolean
  duration?: number
}

export interface GenerateResult {
  audio_path: string
  duration: number
  sample_rate: number
  text: string
  language: string
  engine: string
  seed?: number
}

export const CLONE_MAX_SECONDS = 15
export const REF_HARD_MAX_SECONDS = 75

export async function generateClone(
  input: CloneGenerateInput,
  opts?: { signal?: AbortSignal; onProgress?: (pct: number | null) => void }
): Promise<GenerateResult> {
  const formData = new FormData()
  
  // Add text fields
  formData.append('text', input.text)
  formData.append('language', input.language)
  
  if (input.instruct) formData.append('instruct', input.instruct)
  if (input.ref_text) formData.append('ref_text', input.ref_text)
  if (input.profile_id) formData.append('profile_id', input.profile_id)
  if (input.steps !== undefined) formData.append('steps', String(input.steps))
  if (input.cfg !== undefined) formData.append('cfg', String(input.cfg))
  if (input.speed !== undefined) formData.append('speed', String(input.speed))
  if (input.t_shift !== undefined) formData.append('t_shift', String(input.t_shift))
  if (input.pos_temp !== undefined) formData.append('pos_temp', String(input.pos_temp))
  if (input.class_temp !== undefined) formData.append('class_temp', String(input.class_temp))
  if (input.layer_penalty !== undefined) formData.append('layer_penalty', String(input.layer_penalty))
  if (input.denoise !== undefined) formData.append('denoise', String(input.denoise))
  if (input.postprocess !== undefined) formData.append('postprocess', String(input.postprocess))
  if (input.duration !== undefined) formData.append('duration', String(input.duration))
  
  // Add reference audio
  if (input.ref_audio) {
    formData.append('ref_audio', input.ref_audio)
  } else if (input.ref_audio_path) {
    formData.append('ref_audio_path', input.ref_audio_path)
  }
  
  return apiFormData<GenerateResult>('/generate/clone', formData, {
    method: 'POST',
    signal: opts?.signal,
  })
}

// Sanitize voice design instruct (port of buildDesignInstruct)
export function sanitizeInstruct(free: string): {
  instruct: string
  unsupported: string[]
  duplicates: string[]
  conflicts: string[]
} {
  // This is a simplified version - the real one would parse the instruct string
  // and validate against known voice design parameters
  return {
    instruct: free,
    unsupported: [],
    duplicates: [],
    conflicts: [],
  }
}

export async function generateTTS(input: {
  text: string
  voice: string
  language: string
  speed?: number
  pitch?: number
}, options?: { signal?: AbortSignal }): Promise<GenerateResult> {
  const form = new FormData()
  for (const [key, value] of Object.entries(input)) {
    if (value !== undefined) form.append(key, String(value))
  }
  return apiFormData<GenerateResult>('/generate/tts', form, { signal: options?.signal })
}

export async function generateDubbing(input: {
  video_path: string
  target_language: string
  voice?: string
  background_preserve?: boolean
}): Promise<GenerateResult> {
  return apiJson<GenerateResult>('/generate/dubbing', {
    method: 'POST',
    body: JSON.stringify(input),
  })
}
