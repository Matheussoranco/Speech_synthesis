import path from 'node:path'
import fs from 'node:fs'

export const APP_SCHEME = 'voicestudio:'
export const APP_HOST = 'voicestudio'

export function appUrl(value: string): URL | null {
  // eslint-disable-next-line no-control-regex -- Reject controls before URL normalization.
  if (typeof value !== 'string' || /[\u0000-\u0020\u007f]/.test(value)) return null
  try {
    const url = new URL(value)
    return url.protocol === APP_SCHEME && url.hostname === APP_HOST &&
      !url.port && !url.username && !url.password ? url : null
  } catch { return null }
}

export function within(root: string, target: string, paths = path): boolean {
  const relative = paths.relative(root, target)
  return !paths.isAbsolute(relative) && relative !== '..' &&
    !relative.startsWith(`..${paths.sep}`)
}

export function rendererPath(root: string, value: string, paths = path): string | null {
  const url = appUrl(value)
  if (!url) return null
  let pathname: string
  try { pathname = decodeURIComponent(url.pathname) } catch { return null }
  if (!pathname.startsWith('/') || pathname.startsWith('//') ||
      // eslint-disable-next-line no-control-regex -- Reject decoded controls and Windows path aliases.
      /[\\:\u0000-\u001f\u007f]/.test(pathname) || pathname.split('/').some(part => part === '..')) return null
  const target = paths.resolve(root, `.${pathname}`)
  return within(root, target, paths) ? target : null
}

export function realRendererFile(root: string, target: string): string | null {
  try {
    const realRoot = fs.realpathSync(root)
    const realTarget = fs.realpathSync(target)
    return within(realRoot, realTarget) && fs.statSync(realTarget).isFile() ? realTarget : null
  } catch { return null }
}

export function externalUrl(value: unknown): string | null {
  // eslint-disable-next-line no-control-regex -- Reject controls before URL normalization.
  if (typeof value !== 'string' || value.length > 4096 || /[\u0000-\u001f\u007f]/.test(value)) return null
  try {
    const url = new URL(value)
    return ['https:', 'http:'].includes(url.protocol) && url.hostname &&
      !url.username && !url.password ? url.href : null
  } catch { return null }
}

export function trustedRendererUrl(value: string, developmentUrl?: string): boolean {
  if (appUrl(value)) return true
  if (!developmentUrl) return false
  try {
    const expected = new URL(developmentUrl)
    const actual = new URL(value)
    return expected.protocol === 'http:' &&
      ['localhost', '127.0.0.1', '[::1]'].includes(expected.hostname) &&
      actual.origin === expected.origin && !actual.username && !actual.password
  } catch { return false }
}

export function backendHeaders(source: Headers): Headers {
  const headers = new Headers(source)
  for (const name of ['host', 'connection', 'content-length']) headers.delete(name)
  return headers
}

export function nativePath(value: unknown): string {
  if (typeof value !== 'string' || !value || value.length > 32767 || value.includes('\u0000')) {
    throw new Error('Invalid native path')
  }
  return value
}

export function dialogOptions<T>(value: T): T {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Invalid dialog options')
  const options = value as Record<string, unknown>
  for (const key of ['title', 'defaultPath', 'buttonLabel', 'message', 'nameFieldLabel']) {
    const field = options[key]
    if (field !== undefined && (typeof field !== 'string' || field.length > 32767 || field.includes('\u0000'))) {
      throw new Error('Invalid dialog text')
    }
  }
  if (options.properties !== undefined && (!Array.isArray(options.properties) ||
      !options.properties.every(item => typeof item === 'string'))) throw new Error('Invalid dialog properties')
  if (options.filters !== undefined && (!Array.isArray(options.filters) || !options.filters.every(filter =>
      filter && typeof filter === 'object' && typeof filter.name === 'string' &&
      Array.isArray(filter.extensions) && filter.extensions.every((ext: unknown) => typeof ext === 'string')))) {
    throw new Error('Invalid dialog filters')
  }
  return value
}
