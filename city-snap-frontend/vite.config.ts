import {defineConfig} from 'vite';
import react from '@vitejs/plugin-react';

/**
 * When the React app requests /api/v1/building/info during npm run dev, Vite catches the path, opens a matching request to http://localhost:8081/api/v1/building/info, and
 * streams the response back, eliminating CORS issues and keeping cookies/headers consistent.
 * The changeOrigin flag makes the proxied request present localhost:8081 as its origin host, which some backends require when validating host headers.
 */
export default defineConfig({
    plugins: [react()],
    server: {
        port: 5174,
        open: true,
        proxy: {
            '/api': {
                target: 'http://127.0.0.1:8081',
                changeOrigin: true,
                secure: false,
                ws: true,
                configure: (proxy, _options) => {
                    proxy.on('error', (err, _req, _res) => {
                        console.log('proxy error', err, _req);
                    });
                    proxy.on('proxyReq', (proxyReq, req, _res) => {
                        console.log('Sending Request to the Target:', req.method, req.url);
                    });
                    proxy.on('proxyRes', (proxyRes, req, _res) => {
                        console.log('Received Response from the Target:', proxyRes.statusCode, req.url);
                    });
                },
            },
            '/uploads': {
                target: 'http://127.0.0.1:8081',
                changeOrigin: true,
                secure: false,
            },
        },
    }
});
