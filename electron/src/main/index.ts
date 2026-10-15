import { app, BrowserWindow, ipcMain, protocol, shell, dialog } from 'electron'
import { join } from 'path'
import { spawn, ChildProcess } from 'child_process'
import fs from 'fs'

import metadata from '../../package.json'
import { appUrl, rendererPath, realRendererFile, externalUrl, trustedRendererUrl, backendHeaders, nativePath, dialogOptions } from './boundaries'

// App version from package.json
const APP_VERSION = metadata.version

// Configuration
const BACKEND_PORT = Number(process.env.OMNIVOICE_PORT || 3900)
if (!Number.isInteger(BACKEND_PORT) || BACKEND_PORT < 1 || BACKEND_PORT > 65535) throw new Error('Invalid backend port')
const BACKEND_URL = `http://127.0.0.1:${BACKEND_PORT}`
const STARTUP_BUDGET_S = parseInt(process.env.OMNIVOICE_STARTUP_BUDGET_S || '300', 10)
if (!Number.isFinite(STARTUP_BUDGET_S) || STARTUP_BUDGET_S <= 0) throw new Error('Invalid startup budget')
const SKIP_BACKEND = process.env.VOICESTUDIO_SKIP_BACKEND === '1'

let backendProcess: ChildProcess | null = null
let mainWindow: BrowserWindow | null = null
let isQuitting = false

// Python backend/main.py owns its descriptor and Windows subprocess setup.
// The desktop process owns the child lifetime below.
const DEVELOPMENT_URL = process.env.ELECTRON_RENDERER_URL
protocol.registerSchemesAsPrivileged([{ scheme: 'voicestudio', privileges: {
  standard: true, secure: true, supportFetchAPI: true, stream: true,
} }])

// ── App protocol for renderer ──────────────────────────────────────────────
const APP_PROTOCOL = 'voicestudio'

function setupAppProtocol() {
  protocol.handle(APP_PROTOCOL, async (request) => {
    const url = appUrl(request.url)
    if (!url) return new Response('Not found', { status: 404 })
    const initiatorOrigin = (request as Request & { initiatorOrigin?: string }).initiatorOrigin
    if (initiatorOrigin !== undefined && !trustedRendererUrl(initiatorOrigin, DEVELOPMENT_URL)) {
      return new Response('Forbidden', { status: 403 })
    }
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
          headers: backendHeaders(request.headers),
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
      } catch {
        clearTimeout(timeout)
        return new Response(JSON.stringify({ error: 'Backend unavailable' }), {
          status: 503,
          headers: { 'Content-Type': 'application/json' },
        })
      }
    }
    
    const rendererRoot = join(__dirname, '..', 'renderer')
    let target = rendererPath(rendererRoot, request.url)
    if (!target) return new Response('Not found', { status: 404 })
    try {
      if (fs.statSync(target).isDirectory()) target = join(target, 'index.html')
    } catch {
      // Existing SPA routes load the entry point; unsafe paths never reach this.
      target = join(rendererRoot, 'index.html')
    }
    const file = realRendererFile(rendererRoot, target)
    if (!file) return new Response('Not found', { status: 404 })
    try {
      return new Response(fs.readFileSync(file), { headers: { 'Content-Type': getMimeType(file) } })
    } catch {
      return new Response('Not found', { status: 404 })
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
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 2000)
  try {
    const response = await fetch(`${BACKEND_URL}/api/system/info`, { signal: controller.signal })
    if (response.ok) {
      const data = await response.json()
      return data.app_version === APP_VERSION
    }
  } catch {} finally { clearTimeout(timeout) }
  return false
}

function buildBackendArgv(): string[] {
  const backendDir = app.isPackaged ? join(process.resourcesPath, 'backend') : join(__dirname, '..', '..', '..', 'backend')
  const pythonCmd = process.env.OMNIVOICE_BACKEND_CMD 
    ? JSON.parse(process.env.OMNIVOICE_BACKEND_CMD)
    : ['uv', 'run', '--project', backendDir, 'uvicorn', 'main:app', '--app-dir', backendDir, '--host', '127.0.0.1', '--port', String(BACKEND_PORT)]
  if (!Array.isArray(pythonCmd) || !pythonCmd.length || pythonCmd.some(value =>
      typeof value !== 'string' || !value || value.includes('\u0000'))) throw new Error('Invalid backend command')
  return pythonCmd
}

function spawnBackend(): ChildProcess {
  const argv = buildBackendArgv()
  const backendDir = app.isPackaged ? join(process.resourcesPath, 'backend') : join(__dirname, '..', '..', '..', 'backend')
  
  const env: NodeJS.ProcessEnv = {
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
    if (backendProcess === child) backendProcess = null
  })
  
  child.on('error', (err) => {
    console.error('[backend] spawn error:', err)
    if (backendProcess === child) backendProcess = null
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
  
  while (!isQuitting && Date.now() - startTime < budgetMs) {
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
  const child = backendProcess
  if (!child) return
  backendProcess = null
  const pid = child.pid
  try {
    if (pid && process.platform === 'win32') {
      await new Promise<void>((resolve) => {
        const killer = spawn('taskkill', ['/pid', String(pid), '/T', '/F'], { windowsHide: true })
        killer.once('exit', () => resolve())
        killer.once('error', () => resolve())
      })
    } else if (pid) {
      try { process.kill(-pid, 'SIGTERM') } catch {}
      await new Promise(resolve => setTimeout(resolve, 2000))
      if (child.exitCode === null && child.signalCode === null) {
        try { process.kill(-pid, 'SIGKILL') } catch {}
      }
    }
  } finally {
    child.stdin?.end()
  }
}

let backendOperation: Promise<unknown> = Promise.resolve()
function backendSerial<T>(operation: () => Promise<T>): Promise<T> {
  const next = backendOperation.then(operation, operation)
  backendOperation = next.catch(() => {})
  return next
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
      preload: join(__dirname, '..', 'preload', 'index.cjs'),
      sandbox: true,
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
  if (DEVELOPMENT_URL && trustedRendererUrl(DEVELOPMENT_URL, DEVELOPMENT_URL)) {
    mainWindow.loadURL(DEVELOPMENT_URL)
    mainWindow.webContents.openDevTools({ mode: 'detach' })
  } else {
    mainWindow.loadURL(`${APP_PROTOCOL}://voicestudio/`)
  }
  
  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    const safe = externalUrl(url)
    if (safe) void shell.openExternal(safe).catch(console.error)
    return { action: 'deny' }
  })
  
  const guardNavigation = (event: Electron.Event & { url: string }) => {
    if (!trustedRendererUrl(event.url, DEVELOPMENT_URL)) event.preventDefault()
  }
  mainWindow.webContents.on('will-frame-navigate', guardNavigation)
  mainWindow.webContents.on('will-redirect', guardNavigation)
  return mainWindow
}

// ── IPC Handlers ───────────────────────────────────────────────────────────
function setupIpc() {
  const handle: typeof ipcMain.handle = (channel, listener) => {
    ipcMain.handle(channel, (event, ...args) => {
      if (!mainWindow || event.sender !== mainWindow.webContents ||
          !event.senderFrame || event.senderFrame !== mainWindow.webContents.mainFrame ||
          !trustedRendererUrl(event.senderFrame.url, DEVELOPMENT_URL)) {
        throw new Error('Untrusted desktop request')
      }
      return listener(event, ...args)
    })
  }
  // Backend status
  handle('backend:status', async () => {
    const running = await probeBackend()
    return { running, url: BACKEND_URL, version: APP_VERSION }
  })
  
  // Restart backend
  handle('backend:restart', async () => {
    await backendSerial(async () => {
      if (isQuitting) throw new Error('Application is quitting')
      await killBackend()
      await ensureBackend()
    })
    return { success: true }
  })
  
  // Open external URL
  handle('shell:openExternal', async (_, url: string) => {
    const safe = externalUrl(url)
    if (!safe) throw new Error('Unsupported external URL')
    await shell.openExternal(safe)
  })
  
  // Show item in folder
  handle('shell:showItemInFolder', async (_, path: string) => {
    shell.showItemInFolder(nativePath(path))
  })
  
  // Open file dialog
  handle('dialog:openFile', async (_, options: Electron.OpenDialogOptions) => {
    const result = await dialog.showOpenDialog(mainWindow!, dialogOptions(options))
    return result
  })
  
  // Save file dialog
  handle('dialog:saveFile', async (_, options: Electron.SaveDialogOptions) => {
    const result = await dialog.showSaveDialog(mainWindow!, dialogOptions(options))
    return result
  })
  
  // Get app version
  handle('app:getVersion', () => APP_VERSION)
  
  // Get platform info
  handle('app:getPlatform', () => ({
    platform: process.platform,
    arch: process.arch,
    versions: process.versions,
  }))
}

// ── App lifecycle ──────────────────────────────────────────────────────────
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', event => {
  if (isQuitting) return
  event.preventDefault()
  isQuitting = true
  void backendSerial(killBackend).finally(() => app.quit())
})

app.on('activate', () => {
  if (!isQuitting && BrowserWindow.getAllWindows().length === 0) {
    createWindow()
  }
})

app.whenReady().then(async () => {
  setupAppProtocol()
  setupIpc()
  
  try {
    await backendSerial(ensureBackend)
    if (!isQuitting) createWindow()
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
