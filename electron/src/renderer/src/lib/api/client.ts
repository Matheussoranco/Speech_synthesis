const API_BASE = '/api'

export interface ApiErrorPayload {
  code?: string
  detail?: string
  field_errors?: Record<string, string[]>
  [key: string]: unknown
}

export class ApiError extends Error {
  status: number
  detail: string
  payload: ApiErrorPayload | null
  
  constructor(status: number, detail: string, payload: ApiErrorPayload | null = null) {
    super(detail)
    this.name = 'ApiError'
    this.status = status
    this.detail = detail
    this.payload = payload
  }
  
  static async fromResponse(response: Response): Promise<ApiError> {
    let payload: ApiErrorPayload | null = null
    let detail = response.statusText
    
    try {
      const data = await response.json()
      payload = data
      detail = data.detail || data.message || response.statusText
    } catch {
      detail = await response.text().catch(() => response.statusText)
    }
    
    return new ApiError(response.status, detail, payload)
  }
}

async function apiFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const url = `${API_BASE}${path}`
  
  const response = await fetch(url, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init.headers,
    },
    credentials: 'include',
  })
  
  if (!response.ok) {
    throw await ApiError.fromResponse(response)
  }
  
  return response
}

export async function apiJson<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await apiFetch(path, init)
  const contentType = response.headers.get('content-type')
  
  if (contentType?.includes('application/json')) {
    return response.json()
  }
  
  return response.text() as unknown as T
}

export function audioUrl(filename: string): string {
  return `${API_BASE}/audio/${encodeURIComponent(filename)}`
}

export function profileAudioUrl(profileId: string): string {
  return `${API_BASE}/profiles/${encodeURIComponent(profileId)}/audio`
}

// Helper for multipart/form-data
export async function apiFormData<T>(path: string, formData: FormData, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    method: init.method || 'POST',
    body: formData,
    headers: {
      ...init.headers,
      // Don't set Content-Type for FormData - browser sets it with boundary
    },
    credentials: 'include',
  })
  
  if (!response.ok) {
    throw await ApiError.fromResponse(response)
  }
  
  const contentType = response.headers.get('content-type')
  if (contentType?.includes('application/json')) {
    return response.json()
  }
  return response.text() as unknown as T
}
