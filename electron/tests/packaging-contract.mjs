import assert from 'node:assert/strict'
import fs from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import config from '../electron-builder.config.mjs'

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const metadata = JSON.parse(fs.readFileSync(resolve(root, 'package.json'), 'utf8'))
assert.equal(metadata.main, 'dist/main/index.cjs')
for (const file of [metadata.main, 'dist/preload/index.cjs']) {
  new Function('require', 'exports', 'module', '__dirname', fs.readFileSync(resolve(root, file), 'utf8'))
}
const html = fs.readFileSync(resolve(root, 'dist/renderer/index.html'), 'utf8')
assert.match(html, /<base href="\/"\s*\/>/)
for (const match of html.matchAll(/(?:src|href)="([^"?#]+\.\w+)"/g)) {
  if (match[1].includes('assets/')) assert.ok(fs.existsSync(resolve(root, 'dist/renderer', match[1].replace(/^(\.\/|\/)/, ''))), match[1])
}
const backend = config.extraResources.find(resource => resource.to === 'backend')
assert.ok(backend)
for (const file of ['main.py', 'pyproject.toml', 'core/config.py', 'api/routers/system.py']) {
  assert.ok(fs.existsSync(resolve(root, backend.from, file)), file)
}
for (const resource of config.extraResources) assert.ok(fs.existsSync(resolve(root, resource.from)), resource.from)
console.log('Desktop entry, sandbox preload, nested-route assets and Python resources passed.')
