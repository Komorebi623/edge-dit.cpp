import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vitest/config'

const localRuntimeProxy = {
  changeOrigin: true,
  xfwd: true,
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    tsconfigPaths: true,
  },
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/runtime/v1': {
        ...localRuntimeProxy,
        target: 'http://127.0.0.1:8090',
      },
      '/ed/v2': {
        ...localRuntimeProxy,
        target: 'http://127.0.0.1:8080',
      },
      '/edgedit/v2': {
        ...localRuntimeProxy,
        target: 'http://127.0.0.1:8080',
      },
      '/edge-dit/v2': {
        ...localRuntimeProxy,
        target: 'http://127.0.0.1:8080',
      },
    },
  },
  test: {
    exclude: ['tests/e2e/**', 'dist/**', 'node_modules/**'],
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/shared/test/setup.ts',
    css: true,
  },
})
