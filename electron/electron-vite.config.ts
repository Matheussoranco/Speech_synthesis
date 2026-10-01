import { defineConfig } from 'electron-vite'
import { resolve } from 'path'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  main: {
    plugins: [tailwindcss()],
    build: {
      lib: {
        entry: 'src/main/index.ts',
        formats: ['cjs'],
        fileName: 'index',
      },
      rollupOptions: {
        external: ['electron', 'electron-store', 'ws', 'child_process', 'fs', 'path', 'url', 'os', 'crypto'],
      },
    },
  },
  preload: {
    plugins: [tailwindcss()],
    build: {
      lib: {
        entry: 'src/preload/index.ts',
        formats: ['cjs'],
        fileName: 'index',
      },
      rollupOptions: {
        external: ['electron'],
      },
    },
  },
  renderer: {
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        '@': resolve(__dirname, 'src/renderer/src'),
        '@shared': resolve(__dirname, 'src/shared'),
      },
    },
    build: {
      outDir: 'dist/renderer',
    },
    server: {
      port: 3902,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:3900',
          changeOrigin: true,
        },
      },
    },
  },
})
