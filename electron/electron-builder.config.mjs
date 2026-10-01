import { defineConfig } from 'electron-builder'

export default defineConfig({
  appId: 'com.voicestudio.app',
  productName: 'VoiceStudio',
  copyright: 'Copyright © 2024 VoiceStudio',
  directories: {
    output: 'release',
    buildResources: 'build',
  },
  files: [
    'dist/**/*',
    'package.json',
  ],
  extraResources: [
    {
      from: '../backend',
      to: 'backend',
      filter: ['**/*', '!**/__pycache__/**', '!**/*.pyc', '!**/.venv/**', '!**/outputs/**'],
    },
    {
      from: '../omnivoice',
      to: 'omnivoice',
      filter: ['**/*', '!**/__pycache__/**', '!**/*.pyc'],
    },
    {
      from: '../pyproject.toml',
      to: 'pyproject.toml',
    },
    {
      from: '../uv.lock',
      to: 'uv.lock',
    },
    {
      from: '../README.md',
      to: 'README.md',
    },
    {
      from: '../LICENSE',
      to: 'LICENSE',
    },
  ],
  asar: true,
  mac: {
    category: 'public.app-category.productivity',
    target: ['dmg', 'zip'],
    icon: 'build/icon.icns',
  },
  win: {
    target: ['nsis', 'portable'],
    icon: 'build/icon.ico',
  },
  linux: {
    target: ['AppImage', 'deb'],
    category: 'AudioVideo',
    icon: 'build/icon.png',
  },
  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
  },
  publish: {
    provider: 'github',
    owner: 'your-org',
    repo: 'VoiceStudio',
  },
})
