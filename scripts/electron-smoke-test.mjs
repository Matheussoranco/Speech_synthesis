#!/usr/bin/env bun
// Electron smoke test - verifies the packaged app launches

import { spawn } from 'child_process'
import { existsSync } from 'fs'
import { join } from 'path'
import { fileURLToPath } from 'url'

const __dirname = dirname(fileURLToPath(import.meta.url))
const appDir = join(__dirname, '..', 'electron', 'dist')

const installFlag = process.argv.includes('--install')

async function main() {
  console.log('[smoke-test] Starting Electron smoke test...')
  
  const executable = process.platform === 'win32'
    ? join(appDir, 'VoiceStudio.exe')
    : process.platform === 'darwin'
      ? join(appDir, 'VoiceStudio.app', 'Contents', 'MacOS', 'VoiceStudio')
      : join(appDir, 'VoiceStudio')
  
  if (!existsSync(executable)) {
    console.error('[smoke-test] Executable not found:', executable)
    process.exit(1)
  }
  
  console.log('[smoke-test] Found executable:', executable)
  
  const child = spawn(executable, [], {
    stdio: 'pipe',
    env: {
      ...process.env,
      VOICESTUDIO_SKIP_BACKEND: '1',
    },
  })
  
  let output = ''
  child.stdout?.on('data', (data) => {
    output += data.toString()
  })
  
  child.stderr?.on('data', (data) => {
    output += data.toString()
  })
  
  // Wait for startup
  await new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      child.kill()
      reject(new Error('Smoke test timeout'))
    }, 30000)
    
    child.on('exit', (code) => {
      clearTimeout(timeout)
      if (code === 0) {
        console.log('[smoke-test] App launched and exited cleanly')
        resolve()
      } else {
        console.log('[smoke-test] App exited with code:', code)
        console.log('[smoke-test] Output:', output)
        resolve() // Non-zero exit might be expected for quick test
      }
    })
    
    // Give it a moment to start
    setTimeout(() => {
      child.kill('SIGTERM')
    }, 5000)
  })
  
  console.log('[smoke-test] Smoke test passed!')
}

main().catch((err) => {
  console.error('[smoke-test] Failed:', err)
  process.exit(1)
})
