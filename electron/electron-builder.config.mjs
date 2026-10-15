export default {
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
      filter: ['**/*.py', '**/*.toml', '**/*.md'],
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
  },
  win: {
    target: ['nsis', 'portable'],
  },
  linux: {
    target: ['AppImage', 'deb'],
    category: 'AudioVideo',
  },
  nsis: {
    oneClick: false,
    allowToChangeInstallationDirectory: true,
    createDesktopShortcut: true,
    createStartMenuShortcut: true,
  },
}
