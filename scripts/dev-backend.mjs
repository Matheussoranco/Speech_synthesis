#!/usr/bin/env bun
// Development backend runner - spawns the FastAPI backend with uv

import { spawn } from 'child_process'
import { fileURLToPath } from 'url'
import { dirname, join } from 'path'

const __dirname = dirname(fileURLToPath(import.meta.url))
const backendDir = join(__dirname, '..', 'backend')

console.log('[dev-backend] Starting FastAPI backend...')

const child = spawn('uv', ['run', '--project', backendDir, 'uvicorn', 'main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', '3900', '--reload'], {
  cwd: backendDir,
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    PYTHONUTF8: '1',
    OMNIVOICE_DESKTOP_CONTAINED: '1',
    OMNIVOICE_PORT: '3900',
  },
})

child.on('exit', (code) => {
  console.log(`[dev-backend] Backend exited with code ${code}`)
  process.exit(code ?? 0)
})

child.on('error', (err) => {
  console.error('[dev-backend] Failed to start:', err)
  process.exit(1)
})

// Handle signals
process.on('SIGINT', () => child.kill('SIGINT'))
process.on('SIGTERM', () => child.kill('SIGTERM'))
