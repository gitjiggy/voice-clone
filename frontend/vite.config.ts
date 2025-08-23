import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    host: '127.0.0.1',
    port: 4001,
    strictPort: true,
  },
  define: {
    'process.env.VITE_API_BASE': JSON.stringify('http://127.0.0.1:8000')
  }
})
