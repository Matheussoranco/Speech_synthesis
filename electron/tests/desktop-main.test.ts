import { it, expect, vi, afterEach } from 'vitest'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'
import vm from 'node:vm'
import { EventEmitter } from 'node:events'
import { createRequire } from 'node:module'
import ts from 'typescript'

const nodeRequire = createRequire(import.meta.url)
const temporary: string[] = []
afterEach(() => { vi.useRealTimers(); for (const dir of temporary.splice(0)) fs.rmSync(dir, { recursive: true, force: true }) })
function harness() {
  const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'speech-main-'))
  temporary.push(temp)
  const mainDir = path.join(temp, 'dist/main')
  fs.mkdirSync(mainDir, { recursive: true })
  fs.mkdirSync(path.join(temp, 'dist/renderer'))
  fs.writeFileSync(path.join(temp, 'dist/renderer/index.html'), '<p>control</p>')
  const app = Object.assign(new EventEmitter(), { isPackaged: false, whenReady: () => new Promise(() => {}), quit: vi.fn() })
  const handlers = new Map<string, Function>()
  const protocolHandlers = new Map<string, Function>()
  const native = { openExternal: vi.fn(async () => {}), showItemInFolder: vi.fn() }
  const dialogs = { showOpenDialog: vi.fn(async () => ({ canceled: true, filePaths: [] })),
    showSaveDialog: vi.fn(async () => ({ canceled: true, filePath: undefined })), showErrorBox: vi.fn() }
  class Window extends EventEmitter {
    webContents = Object.assign(new EventEmitter(), { mainFrame: { url: 'voicestudio://voicestudio/' },
      setWindowOpenHandler: vi.fn(), openDevTools: vi.fn() })
    loadURL = vi.fn(); show = vi.fn()
    constructor(public options: unknown) { super() }
    static getAllWindows() { return [] }
  }
  const processMock = Object.assign(new EventEmitter(), { env: {}, platform: 'linux', arch: 'x64', versions: {}, kill: vi.fn() })
  const fetchMock = vi.fn(async () => new Response(JSON.stringify({ app_version: '0.1.0' }), { headers: { 'content-type': 'application/json' } }))
  const electron = { app, BrowserWindow: Window, ipcMain: { handle: (key: string, callback: Function) => handlers.set(key, callback) },
    protocol: { registerSchemesAsPrivileged: vi.fn(), handle: (key: string, callback: Function) => protocolHandlers.set(key, callback) },
    shell: native, dialog: dialogs }
  const compile = (source: string) => ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022, esModuleInterop: true } }).outputText
  const helpers = { exports: {} }
  vm.runInNewContext(compile(fs.readFileSync(path.resolve('src/main/boundaries.ts'), 'utf8')),
    { exports: helpers.exports, module: helpers, require: nodeRequire, URL, Headers })
  const context = vm.createContext({ exports: {}, __dirname: mainDir, console, URL, Response, Headers, AbortController,
    setTimeout, clearTimeout, process: processMock, fetch: fetchMock,
    require: (id: string) => id === 'electron' ? electron : id === './boundaries' ? helpers.exports :
      id === '../../package.json' ? { version: '0.1.0' } : nodeRequire(id) })
  vm.runInContext(compile(fs.readFileSync(path.resolve('src/main/index.ts'), 'utf8')), context)
  context.setupAppProtocol(); context.setupIpc()
  const window = context.createWindow()
  const trusted = { sender: window.webContents, senderFrame: window.webContents.mainFrame }
  return { context, app, window, trusted, handlers, protocol: protocolHandlers.get('voicestudio')!, native, dialogs, fetchMock, processMock }
}
it('actual protocol rejects traversal and foreign hosts/origins and serves legitimate SPA requests', async () => {
  const h = harness()
  for (const url of ['voicestudio://evil/api/system/info', 'voicestudio://voicestudio/..\\renderer-private\\secret',
    'voicestudio://voicestudio/file:stream', 'voicestudio://voicestudio/%00']) {
    expect((await h.protocol(new Request(url))).status).toBe(404)
  }
  expect(h.fetchMock).not.toHaveBeenCalled()
  const response = await h.protocol(new Request('voicestudio://voicestudio/clone'))
  expect(response.status).toBe(200); expect(await response.text()).toContain('control')
  const hostile = Object.assign(new Request('voicestudio://voicestudio/api/system/info'), { initiatorOrigin: 'https://evil.example' })
  expect((await h.protocol(hostile)).status).toBe(403)
})
it('actual protocol preserves multipart uploads and empty/ranged backend responses', async () => {
  const h = harness()
  h.fetchMock.mockResolvedValueOnce(new Response(null, { status: 204 }))
  const response = await h.protocol(new Request('voicestudio://voicestudio/api/generate/clone', { method: 'POST',
    headers: { 'content-type': 'multipart/form-data; boundary=test', range: 'bytes=0-4' }, body: '--test--' }))
  expect(response.status).toBe(204)
  const [url, init] = h.fetchMock.mock.calls[0] as unknown as [string, RequestInit]
  expect(url).toBe('http://127.0.0.1:3900/api/generate/clone')
  expect(new Headers(init.headers).get('content-type')).toBe('multipart/form-data; boundary=test')
  expect(new Headers(init.headers).get('range')).toBe('bytes=0-4')
  expect(new TextDecoder().decode(init.body as ArrayBuffer)).toBe('--test--')
})
it('every privileged handler rejects foreign, absent and child frames; trusted dialogs keep cancellation', async () => {
  const h = harness()
  for (const handler of h.handlers.values()) {
    for (const event of [{ sender: {}, senderFrame: h.trusted.senderFrame }, { sender: h.trusted.sender, senderFrame: null },
      { sender: h.trusted.sender, senderFrame: { url: 'voicestudio://voicestudio/' } }]) {
      expect(() => handler(event)).toThrow('Untrusted desktop request')
    }
  }
  const open = h.handlers.get('shell:openExternal')!
  await expect(open(h.trusted, 'file:///C:/secret')).rejects.toThrow('Unsupported')
  await open(h.trusted, 'https://example.org/help')
  expect(h.native.openExternal).toHaveBeenCalledExactlyOnceWith('https://example.org/help')
  await expect(h.handlers.get('dialog:openFile')!(h.trusted, { filters: [{ name: 'Audio', extensions: ['wav'] }] }))
    .resolves.toEqual({ canceled: true, filePaths: [] })
  await expect(h.handlers.get('dialog:openFile')!(h.trusted, { title: '', message: '', buttonLabel: '', defaultPath: '' }))
    .resolves.toEqual({ canceled: true, filePaths: [] })
  await expect(h.handlers.get('shell:showItemInFolder')!(h.trusted, {})).rejects.toThrow()
  h.window.webContents.mainFrame.url = 'https://evil.example/'
  expect(() => h.handlers.get('app:getVersion')!(h.trusted)).toThrow('Untrusted')
})
it('navigation/redirect and popup routes enforce the same trusted boundary', () => {
  const h = harness()
  for (const name of ['will-frame-navigate', 'will-redirect']) {
    const deny = vi.fn(); h.window.webContents.emit(name, { url: 'https://evil.example', preventDefault: deny })
    expect(deny).toHaveBeenCalledOnce()
    const allow = vi.fn(); h.window.webContents.emit(name, { url: 'voicestudio://voicestudio/#clone', preventDefault: allow })
    expect(allow).not.toHaveBeenCalled()
  }
  const popup = h.window.webContents.setWindowOpenHandler.mock.calls[0][0]
  expect(popup({ url: 'file:///C:/secret' })).toEqual({ action: 'deny' })
  expect(h.native.openExternal).not.toHaveBeenCalled()
  expect(h.window.options.webPreferences.sandbox).toBe(true)
})
it('cleanup retains child identity across async replacement and serializes operations', async () => {
  vi.useFakeTimers()
  const h = harness()
  const a = { pid: 101, exitCode: null, signalCode: null, stdin: { end: vi.fn() } }
  const b = { pid: 202, exitCode: null, signalCode: null, stdin: { end: vi.fn() } }
  h.context.childA = a; h.context.childB = b
  vm.runInContext('backendProcess = childA', h.context)
  const killed = h.context.killBackend()
  vm.runInContext('backendProcess = childB', h.context)
  await vi.advanceTimersByTimeAsync(2000); await killed
  expect(h.processMock.kill.mock.calls).toEqual([[-101, 'SIGTERM'], [-101, 'SIGKILL']])
  expect(a.stdin.end).toHaveBeenCalledOnce(); expect(b.stdin.end).not.toHaveBeenCalled()
  expect(vm.runInContext('backendProcess === childB', h.context)).toBe(true)
  const order: string[] = []; let finish!: () => void
  const first = h.context.backendSerial(async () => { order.push('a'); await new Promise<void>(resolve => { finish = resolve }); order.push('b') })
  const second = h.context.backendSerial(async () => { order.push('c') })
  await Promise.resolve(); expect(order).toEqual(['a']); finish(); await first; await second
  expect(order).toEqual(['a', 'b', 'c'])
})
