#!/usr/bin/env bun
// Clear development ports

import { execSync } from 'child_process'

const ports = process.argv.slice(2).map(Number).filter(n => !isNaN(n))

for (const port of ports) {
  try {
    if (process.platform === 'win32') {
      execSync(`netstat -ano | findstr :${port}`, { stdio: 'ignore' })
      // On Windows, we'd need to parse PID and taskkill
      console.log(`[clear-ports] Port ${port} - manual cleanup needed on Windows`)
    } else {
      execSync(`lsof -ti:${port} | xargs kill -9 2>/dev/null || true`, { stdio: 'ignore' })
      console.log(`[clear-ports] Cleared port ${port}`)
    }
  } catch {
    // Port not in use
  }
}
