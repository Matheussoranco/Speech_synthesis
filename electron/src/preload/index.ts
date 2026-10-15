import { contextBridge, ipcRenderer } from 'electron'

// Type definitions for the exposed API
interface BackendStatus {
  running: boolean
  url: string
  version: string
}

interface PlatformInfo {
  platform: string
  arch: string
  versions: NodeJS.ProcessVersions
}

interface DialogResult {
  canceled: boolean
  filePaths: string[]
}

interface SaveDialogResult {
  canceled: boolean
  filePath: string | undefined
}

// Expose secure APIs to renderer
if (process.isMainFrame) contextBridge.exposeInMainWorld('voicestudio', {
  // Backend management
  backend: {
    getStatus: (): Promise<BackendStatus> => 
      ipcRenderer.invoke('backend:status'),
    restart: (): Promise<{ success: boolean }> => 
      ipcRenderer.invoke('backend:restart'),
  },
  
  // Shell operations
  shell: {
    openExternal: (url: string): Promise<void> => 
      ipcRenderer.invoke('shell:openExternal', url),
    showItemInFolder: (path: string): Promise<void> => 
      ipcRenderer.invoke('shell:showItemInFolder', path),
  },
  
  // Dialogs
  dialog: {
    openFile: (options: Electron.OpenDialogOptions): Promise<DialogResult> => 
      ipcRenderer.invoke('dialog:openFile', options),
    saveFile: (options: Electron.SaveDialogOptions): Promise<SaveDialogResult> => 
      ipcRenderer.invoke('dialog:saveFile', options),
  },
  
  // App info
  app: {
    getVersion: (): Promise<string> => 
      ipcRenderer.invoke('app:getVersion'),
    getPlatform: (): Promise<PlatformInfo> => 
      ipcRenderer.invoke('app:getPlatform'),
  },
  
  // Backend URL for direct WebSocket connections (live dictation)
  getBackendUrl: (): string => {
    const port = process.env.OMNIVOICE_PORT || '3900'
    return `http://127.0.0.1:${port}`
  },
  
  // Event listeners for backend status changes
  onBackendStatusChange: (callback: (status: BackendStatus) => void): (() => void) => {
    const handler = (_event: Electron.IpcRendererEvent, status: BackendStatus) => callback(status)
    ipcRenderer.on('backend:status-changed', handler)
    return () => ipcRenderer.off('backend:status-changed', handler)
  },
})

// Also expose a minimal node-like API for things that need it
if (process.isMainFrame) contextBridge.exposeInMainWorld('node', {
  platform: process.platform,
  arch: process.arch,
  versions: process.versions,
})

// Declare global types
declare global {
  interface Window {
    voicestudio: {
      backend: {
        getStatus: () => Promise<BackendStatus>
        restart: () => Promise<{ success: boolean }>
      }
      shell: {
        openExternal: (url: string) => Promise<void>
        showItemInFolder: (path: string) => Promise<void>
      }
      dialog: {
        openFile: (options: Electron.OpenDialogOptions) => Promise<DialogResult>
        saveFile: (options: Electron.SaveDialogOptions) => Promise<SaveDialogResult>
      }
      app: {
        getVersion: () => Promise<string>
        getPlatform: () => Promise<PlatformInfo>
      }
      getBackendUrl: () => string
      onBackendStatusChange: (callback: (status: BackendStatus) => void) => () => void
    }
    node: {
      platform: string
      arch: string
      versions: NodeJS.ProcessVersions
    }
  }
}


