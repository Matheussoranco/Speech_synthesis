// Launch the actual unpacked application and require a rendered, translated UI.
import { _electron } from 'playwright'
import assert from 'node:assert/strict'
import fs from 'node:fs'
import http from 'node:http'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..', 'electron', 'release')
const executablePath = process.platform === 'win32' ? resolve(root, 'win-unpacked/VoiceStudio.exe')
  : process.platform === 'darwin' ? resolve(root, 'mac/VoiceStudio.app/Contents/MacOS/VoiceStudio')
  : resolve(root, 'linux-unpacked/voicestudio')
assert.ok(fs.existsSync(executablePath), 'Build the unpacked desktop application first')
let application
const version = JSON.parse(fs.readFileSync(resolve(root, '../package.json'), 'utf8')).version
const backend = http.createServer((request, response) => {
  response.setHeader('Content-Type', 'application/json')
  response.end(JSON.stringify(request.url === '/api/system/info' ? { app_version: version }
    : request.url === '/api/engines' ? { tts: [], asr: [], diarization: [], translation: [], vocoder: [] } : []))
})
await new Promise(resolve => backend.listen(0, '127.0.0.1', resolve))
try {
  application = await _electron.launch({ executablePath, env: { ...process.env, VOICESTUDIO_SKIP_BACKEND: '1', OMNIVOICE_PORT: String(backend.address().port) }, timeout: 30000 })
  const page = await application.firstWindow()
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.locator('textarea').waitFor()
  for (const route of ['/tools', '/profiles', '/history', '/settings', '/clone']) {
    await page.goto('voicestudio://voicestudio' + route)
    await page.locator('nav a[href="/tools"]').waitFor()
    await page.locator('h1').first().waitFor()
  }
  assert.deepEqual(errors, [])
  console.log('Packaged desktop rendered all smoke routes without JavaScript errors against a controlled local smoke service; real backend deliberately skipped.')
} finally {
  if (application) await application.close()
  backend.closeAllConnections()
  await new Promise(resolve => backend.close(resolve))
}
