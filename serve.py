#!/usr/bin/env python3
"""
Simple HTTP server to view the generated light novel.
Run: python serve.py
Then open: http://localhost:8000
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

def serve():
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    Handler = MyHTTPRequestHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        url = f"http://localhost:{PORT}/viewer.html"
        print(f"✨ Light Novel Viewer Server")
        print(f"📖 Open in browser: {url}")
        print(f"⏹️  Press Ctrl+C to stop")
        
        # Try to open browser automatically
        try:
            webbrowser.open(url)
        except:
            pass
        
        httpd.serve_forever()

if __name__ == "__main__":
    serve()
