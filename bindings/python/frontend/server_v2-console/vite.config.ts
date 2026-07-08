import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { defineConfig } from 'vitest/config'

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
        target: 'http://127.0.0.1:8090',
        changeOrigin: true,
      },
      '/ed/v2': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
      '/edgedit/v2': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
      '/edge-dit/v2': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
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
