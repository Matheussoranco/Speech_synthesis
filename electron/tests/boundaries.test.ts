import { describe, it, expect } from 'vitest'
import path from 'node:path'
import fs from 'node:fs'
import os from 'node:os'
import { appUrl, rendererPath, realRendererFile, externalUrl, trustedRendererUrl, backendHeaders } from '../src/main/boundaries'

describe('desktop shared boundaries', () => {
  it('rejects prefix siblings, raw and encoded Windows traversal, ADS and malformed input', () => {
    const root = 'C:\\app\\dist\\renderer'
    for (const suffix of ['..\\renderer-private\\secret.txt', '%2e%2e%5crenderer-private%5csecret.txt',
      '%2f%2fserver/share', 'file.txt:secret', '%00file', '%', '%2e%2e%2fsecret']) {
      expect(rendererPath(root, `voicestudio://voicestudio/${suffix}`, path.win32), suffix).toBeNull()
    }
    expect(rendererPath(root, 'voicestudio://voicestudio/assets/voice%20%C3%A1.wav', path.win32))
      .toBe('C:\\app\\dist\\renderer\\assets\\voice á.wav')
    expect(rendererPath(root, 'voicestudio://voicestudio/%252e%252e%255csecret', path.win32))
      .toBe('C:\\app\\dist\\renderer\\%2e%2e%5csecret')
  })
  it('does not admit other authorities, credentials, ports, schemes or opaque origins', () => {
    for (const value of ['voicestudio://evil/', 'voicestudio://user@voicestudio/',
      'voicestudio://voicestudio:42/', 'file://voicestudio/', 'null', '\nvoicestudio://voicestudio/']) {
      expect(appUrl(value)).toBeNull()
      expect(trustedRendererUrl(value, 'http://127.0.0.1:3902')).toBe(false)
    }
    expect(trustedRendererUrl('http://127.0.0.1:3902/clone', 'http://127.0.0.1:3902')).toBe(true)
    expect(trustedRendererUrl('http://127.0.0.1:3903', 'http://127.0.0.1:3902')).toBe(false)
    expect(trustedRendererUrl('http://127.0.0.1.evil:3902', 'http://127.0.0.1:3902')).toBe(false)
  })
  it('requires physical containment and a regular file', () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'speech-boundary-'))
    try {
      const root = path.join(temp, 'renderer')
      fs.mkdirSync(root)
      const outside = path.join(temp, 'renderer-private')
      fs.mkdirSync(outside)
      fs.writeFileSync(path.join(root, 'index.html'), 'control')
      fs.writeFileSync(path.join(outside, 'secret.txt'), 'secret')
      fs.symlinkSync(outside, path.join(root, 'escape'), process.platform === 'win32' ? 'junction' : 'dir')
      expect(realRendererFile(root, path.join(root, 'index.html'))).toBe(path.join(root, 'index.html'))
      expect(realRendererFile(root, path.join(root, 'escape', 'secret.txt'))).toBeNull()
      expect(realRendererFile(root, outside)).toBeNull()
      expect(realRendererFile(root, root)).toBeNull()
    } finally { fs.rmSync(temp, { recursive: true, force: true }) }
  })
  it('preserves multipart, cookies and audio ranges while removing transport headers', () => {
    const headers = backendHeaders(new Headers({ 'content-type': 'multipart/form-data; boundary=real',
      cookie: 'session=control', range: 'bytes=0-4', host: 'untrusted', 'content-length': '12' }))
    expect(headers.get('content-type')).toBe('multipart/form-data; boundary=real')
    expect(headers.get('cookie')).toBe('session=control')
    expect(headers.get('range')).toBe('bytes=0-4')
    expect(headers.has('host')).toBe(false)
    expect(headers.has('content-length')).toBe(false)
  })
  it('permits ordinary web links and rejects native schemes and non-string values', () => {
    expect(externalUrl('https://example.org/help?q=voice')).toBe('https://example.org/help?q=voice')
    for (const value of [null, {}, 1, 'file:///C:/secret', 'ms-settings:privacy', 'javascript:alert(1)',
      'https://user:pass@example.org', 'https://example.org/\n']) expect(externalUrl(value)).toBeNull()
  })
})
