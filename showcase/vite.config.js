import { copyFileSync, mkdirSync, mkdtempSync, readdirSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { basename, dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

const root = dirname(fileURLToPath(import.meta.url))
const publicDir = mkdtempSync(join(tmpdir(), 'mgts-showcase-static-'))
process.once('exit', () => rmSync(publicDir, { recursive: true, force: true }))

function includeFiles(source, destination, files) {
  const target = join(publicDir, destination)
  mkdirSync(target, { recursive: true })
  for (const file of files) copyFileSync(join(source, file), join(target, basename(file)))
}

const predict = join(root, '../browser-ml')
includeFiles(predict, 'predict', [
  ...readdirSync(predict).filter(file => /\.(?:js|css)$/.test(file)),
  'index.html', 'heads-v2.json', 'oof-summary.json', 'input-quality-report.json',
])
includeFiles(join(predict, 'runtime'), 'predict/runtime', [
  'dino-int8.onnx', 'ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm',
  'DINOV2-LICENSE.txt', 'ONNXRUNTIME-LICENSE.txt', 'ONNXRUNTIME-NOTICES.txt', 'onnxruntime-readme.txt',
])
includeFiles(join(root, '../registry'), 'registry', ['index.html', 'registry.js', 'registry.css', 'results.json', 'argon-example.json', 'public-protocol.json'])
includeFiles(join(root, 'assets'), 'assets', ['mascot.svg', 'mascot-retake.svg', 'mascot-accept.svg'])

export default defineConfig({
  base: '/mgts-route-inspector/',
  publicDir,
  plugins: [{
    name: 'public-app-indexes',
    configureServer(server) {
      const base = server.config.base
      const pages = new Set([base + 'predict/', base + 'registry/'])
      server.middlewares.use((request, _response, next) => {
        const url = new URL(request.url, 'http://localhost')
        if (pages.has(url.pathname)) request.url = url.pathname + 'index.html' + url.search
        next()
      })
    },
  }],
})
