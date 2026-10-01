import { app, BrowserWindow, ipcMain, protocol, net, shell, dialog } from 'electron'
import { join, dirname } from 'path'
import { fileURLToPath } from 'url'
import { spawn, ChildProcess } from 'child_process'
import { createRequire } from 'module'
import fs from 'fs'

const require = createRequire(import.meta.url)
const __dirname = dirname(fileURLToPath(import.meta.url))

// App version from package.json
const APP_VERSION = require('../../../package.json').version

// Configuration
const BACKEND_PORT = process.env.OMNIVOICE_PORT || 3900
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`
const STARTUP_BUDGET_S = parseInt(process.env.OMNIVOICE_STARTUP_BUDGET_S || '300', 10)
const SKIP_BACKEND = process.env.VOICESTUDIO_SKIP_BACKEND === '1'

let backendProcess: ChildProcess | null = null
let mainWindow: BrowserWindow | null = null
let isQuitting = false

// Secure the backend drain file descriptor early
try {
  const { secure_backend_drain_fd } = await import('../core/contained_subprocess.js')
  secure_backend_drain_fd()
} catch {}

// Windows: suppress console windows for child processes
if (process.platform === 'win32') {
  try {
    const { install } = await import('../core/win_subprocess.js')
    install()
  } catch {}
}

// Disable Triton on Windows
if (process.platform === 'win32') {
  process.env.TORCH_COMPILE_DISABLE = '1'
  process.env.TORCHDYNAMO_DISABLE = '1'
  process.env.TORCHINDUCTOR_DISABLE = '1'
}

// Ensure omnivoice is importable
try {
  const { ensure_omnivoice_importable } = await import('../core/omnivoice_path.js')
  ensure_omnivoice_importable(join(__dirname, '..', '..', 'backend'))
} catch {}

// ── App protocol for renderer ──────────────────────────────────────────────
const APP_PROTOCOL = 'voicestudio'

function setupAppProtocol() {
  protocol.handle(APP_PROTOCOL, async (request) => {
    const url = new URL(request.url)
    const pathname = url.pathname
    
    // Proxy /api/* to backend
    if (pathname.startsWith('/api/')) {
      const backendPath = pathname.slice(4) // Remove '/api'
      const backendReqUrl = `${BACKEND_URL}/api${backendPath}${url.search}`
      
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 30000)
      
      try {
        const response = await fetch(backendReqUrl, {
          method: request.method,
          headers: {
            'Content-Type': 'application/json',
            ...Object.fromEntries(
              Object.entries(request.headers).filter(([k]) => 
                !['host', 'connection', 'content-length'].includes(k.toLowerCase())
              )
            ),
          },
          body: request.method !== 'GET' && request.method !== 'HEAD' 
            ? await request.arrayBuffer() 
            : undefined,
          signal: controller.signal,
        })
        
        clearTimeout(timeout)
        
        return new Response(response.body, {
          status: response.status,
          statusText: response.statusText,
          headers: response.headers,
        })
      } catch (e) {
        clearTimeout(timeout)
        return new Response(JSON.stringify({ error: 'Backend unavailable' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        })
      }
    }
    
    // Serve static renderer files
    let filePath = join(__dirname, '..', 'renderer', pathname === '/' ? 'index.html' : pathname.slice(1))
    
    // Security: prevent directory traversal
    filePath = filePath.replace(/\.\./g, '')
    if (!filePath.startsWith(join(__dirname, '..', 'renderer'))) {
      return new Response('Not found', { status: 404 })
    }
    
    try {
      const stat = fs.statSync(filePath)
      if (stat.isDirectory()) {
        filePath = join(filePath, 'index.html')
      }
      const content = fs.readFileSync(filePath)
      const mime = getMimeType(filePath)
      return new Response(content, { headers: { 'Content-Type': mime } })
    } catch {
      // Fallback to index.html for SPA routing
      try {
        const content = fs.readFileSync(join(__dirname, '..', 'renderer', 'index.html'))
        return new Response(content, { headers: { 'Content-Type': 'text/html' } })
      } catch {
        return new Response('Not found', { status: 404 })
      }
    }
  })
}

function getMimeType(filePath: string): string {
  const ext = filePath.split('.').pop()?.toLowerCase()
  const types: Record<string, string> = {
    html: 'text/html',
    js: 'application/javascript',
    mjs: 'application/javascript',
    css: 'text/css',
    json: 'application/json',
    png: 'image/png',
    jpg: 'image/jpeg',
    jpeg: 'image/jpeg',
    gif: 'image/gif',
    svg: 'image/svg+xml',
    ico: 'image/x-icon',
    wav: 'audio/wav',
    mp3: 'audio/mpeg',
    ogg: 'audio/ogg',
    webp: 'image/webp',
    woff: 'font/woff',
    woff2: 'font/woff2',
    ttf: 'font/ttf',
  }
  return types[ext || ''] || 'application/octet-stream'
}

// ── Backend management ─────────────────────────────────────────────────────
async function probeBackend(): Promise<boolean> {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 2000)
    const response = await fetch(`${BACKEND_URL}/system/info`, { signal: controller.signal })
    clearTimeout(timeout)
    if (response.ok) {
      const data = await response.json()
      return data.app_version === APP_VERSION
    }
  } catch {}
  return false
}

function buildBackendArgv(): string[] {
  const backendDir = join(__dirname, '..', '..', 'backend')
  const pythonCmd = process.env.OMNIVOICE_BACKEND_CMD 
    ? JSON.parse(process.env.OMNIVOICE_BACKEND_CMD)
    : ['uv', 'run', '--project', backendDir, 'uvicorn', 'main:app', '--app-dir', 'backend', '--host', '127.0.0.1', '--port', String(BACKEND_PORT)]
  return pythonCmd
}

function spawnBackend(): ChildProcess {
  const argv = buildBackendArgv()
  const backendDir = join(__dirname, '..', '..', 'backend')
  
  const env = {
    ...process.env,
    PYTHONUNBUFFERED: '1',
    PYTHONUTF8: '1',
    OMNIVOICE_DESKTOP_CONTAINED: '1',
    OMNIVOICE_PORT: String(BACKEND_PORT),
    FOR_DISABLE_CONSOLE_CTRL_HANDLER: '1',
    // Remove problematic env vars
    PYTHONHOME: undefined,
    PYTHONPATH: undefined,
  }
  
  // Windows-specific
  if (process.platform === 'win32') {
    env.TORCHDYNAMO_DISABLE = '1'
    env.HF_HUB_DISABLE_SYMLINKS = '1'
    env.HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
  }
  
  // Filter out undefined values
  const cleanEnv: Record<string, string> = {}
  for (const [k, v] of Object.entries(env)) {
    if (v !== undefined) cleanEnv[k] = v
  }
  
  const child = spawn(argv[0], argv.slice(1), {
    cwd: backendDir,
    env: cleanEnv,
    stdio: ['pipe', 'pipe', 'pipe'],
    windowsHide: true,
    detached: process.platform !== 'win32',
  })
  
  child.stdout?.on('data', (data) => {
    console.log(`[backend] ${data.toString().trim()}`)
  })
  
  child.stderr?.on('data', (data) => {
    console.error(`[backend:err] ${data.toString().trim()}`)
  })
  
  child.on('exit', (code, signal) => {
    console.log(`[backend] exited with code ${code}, signal ${signal}`)
    if (code === 78) {
      console.log('[backend] Port in use - backend already running?')
    }
    backendProcess = null
  })
  
  child.on('error', (err) => {
    console.error('[backend] spawn error:', err)
    backendProcess = null
  })
  
  // Never write to stdin, never end it - closing it is the liveness signal
  // We just keep it open
  
  return child
}

async function ensureBackend(): Promise<void> {
  if (SKIP_BACKEND) {
    console.log('[main] VOICESTUDIO_SKIP_BACKEND=1, attaching only')
    const attached = await probeBackend()
    if (!attached) {
      throw new Error('Backend not running and VOICESTUDIO_SKIP_BACKEND=1')
    }
    return
  }
  
  const attached = await probeBackend()
  if (attached) {
    console.log('[main] Backend already running, attaching')
    return
  }
  
  console.log('[main] Spawning backend...')
  backendProcess = spawnBackend()
  
  // Wait for readiness
  const startTime = Date.now()
  const budgetMs = STARTUP_BUDGET_S * 1000
  
  while (Date.now() - startTime < budgetMs) {
    await new Promise(r => setTimeout(r, 500))
    if (await probeBackend()) {
      console.log('[main] Backend ready')
      return
    }
    if (!backendProcess || backendProcess.killed) {
      throw new Error('Backend process died during startup')
    }
  }
  
  throw new Error(`Backend startup timeout after ${STARTUP_BUDGET_S}s`)
}

async function killBackend(): Promise<void> {
  if (!backendProcess) return
  
  console.log('[main] Stopping backend...')
  
  if (process.platform === 'win32') {
    // Windows: taskkill /T /F
    try {
      const { spawn } = await import('child_process')
      await new Promise<void>((resolve) => {
        const killer = spawn('taskkill', ['/pid', String(backendProcess!.pid), '/T', '/F'])
        killer.on('exit', () => resolve())
      })
    } catch {}
  } else {
    // POSIX: kill process group
    try {
      if (backendProcess.pid) {
        process.kill(-backendProcess.pid, 'SIGTERM')
        await new Promise(r => setTimeout(r, 2000))
        try { process.kill(-backendProcess.pid!, 'SIGKILL') } catch {}
      }
    } catch {}
  }
  
  // Close stdin as liveness signal
  try { backendProcess.stdin?.end() } catch {}
  backendProcess = null
}

// ── Window creation ────────────────────────────────────────────────────────
function createWindow(): BrowserWindow {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    titleBarStyle: 'hidden',
    titleBarOverlay: {
      color: '#1a1a2e',
      symbolColor: '#ffffff',
      height: 36,
    },
    webPreferences: {
      preload: join(__dirname, '..', 'preload', 'index.js'),
      sandbox: false,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
      allowRunningInsecureContent: false,
    },
    show: false,
    icon: join(__dirname, '..', '..', '..', 'assets', 'icon.png'), // TODO: add icon
  })
  
  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })
  
  mainWindow.on('closed', () => {
    mainWindow = null
  })
  
  // Load the app
  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:3902')
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadURL(`${APP_PROTOCOL}://voicestudio/`)
  }
  
  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url)
    return { action: 'deny' }
  })
  
  return mainWindow
}

// ── IPC Handlers ───────────────────────────────────────────────────────────
function setupIpc() {
  // Backend status
  ipcMain.handle('backend:status', async () => {
    const running = await probeBackend()
    return { running, url: BACKEND_URL, version: APP_VERSION }
  })
  
  // Restart backend
  ipcMain.handle('backend:restart', async () => {
    await killBackend()
    await ensureBackend()
    return { success: true }
  })
  
  // Open external URL
  ipcMain.handle('shell:openExternal', async (_, url: string) => {
    await shell.openExternal(url)
  })
  
  // Show item in folder
  ipcMain.handle('shell:showItemInFolder', async (_, path: string) => {
    shell.showItemInFolder(path)
  })
  
  // Open file dialog
  ipcMain.handle('dialog:openFile', async (_, options: Electron.OpenDialogOptions) => {
    const result = await dialog.showOpenDialog(mainWindow!, options)
    return result
  })
  
  // Save file dialog
  ipcMain.handle('dialog:saveFile', async (_, options: Electron.SaveDialogOptions) => {
    const result = await dialog.showSaveDialog(mainWindow!, options)
    return result
  })
  
  // Get app version
  ipcMain.handle('app:getVersion', () => APP_VERSION)
  
  // Get platform info
  ipcMain.handle('app:getPlatform', () => ({
    platform: process.platform,
    arch: process.arch,
    versions: process.versions,
  }))
}

// ── App lifecycle ──────────────────────────────────────────────────────────
app.on('window-all-closed', async () => {
  if (process.platform !== 'darwin') {
    isQuitting = true
    await killBackend()
    app.quit()
  }
})

app.on('before-quit', async () => {
  isQuitting = true
  await killBackend()
})

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.whenReady().then(async () => {
  setupAppProtocol()
  setupIpc()
  
  try {
    await ensureBackend()
    createWindow()
  } catch (err) {
    console.error('[main] Failed to start:', err)
    dialog.showErrorBox('VoiceStudio Startup Error', 
      `Failed to start the backend:
${err instanceof Error ? err.message : String(err)}`)
    app.quit()
  }
})

// Handle uncaught errors
process.on('uncaughtException', (err) => {
  console.error('[main] Uncaught exception:', err)
})

process.on('unhandledRejection', (reason) => {
  console.error('[main] Unhandled rejection:', reason)
})
