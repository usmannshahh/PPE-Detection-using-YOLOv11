"""
Combined Server - Backend + Frontend
Run from full_backend folder: python runs.py
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
import threading
import os

def run_backend():
    """Run FastAPI backend"""
    print("🚀 Starting Backend Server...")
    backend_dir = Path(__file__).parent
    os.chdir(backend_dir)
    subprocess.run([sys.executable, "main.py"])

def run_frontend():
    """Run frontend HTTP server"""
    print("🌐 Starting Frontend Server...")
    backend_dir = Path(__file__).parent
    frontend_path = backend_dir / "frontend"  # FIXED: Same directory level
    
    if not frontend_path.exists():
        print(f"❌ Frontend folder not found: {frontend_path}")
        return
    
    subprocess.run([
        sys.executable, 
        "-m", 
        "http.server", 
        "8080",
        "--directory",
        str(frontend_path)
    ])

def open_browser():
    """Open browser after delay"""
    time.sleep(3)
    print("\n🌐 Opening browser...")
    webbrowser.open("http://localhost:8080")
    print("📡 Backend API: http://localhost:8000")
    print("🎨 Frontend UI: http://localhost:8080")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════╗
    ║   🦺 PPE Detection - Combined Server        ║
    ║   📡 Backend: http://localhost:8000          ║
    ║   🎨 Frontend: http://localhost:8080         ║
    ╚═══════════════════════════════════════════════╝
    """)
    
    # Check if files exist
    backend_dir = Path(__file__).parent
    frontend_dir = backend_dir / "frontend"  # FIXED
    model_path = backend_dir / "best.pt"
    
    print(f"📂 Backend directory: {backend_dir}")
    print(f"📂 Frontend directory: {frontend_dir}")
    print(f"🤖 Model path: {model_path}")
    
    if not model_path.exists():
        print(f"❌ ERROR: best.pt not found at {model_path}")
        sys.exit(1)
    
    if not frontend_dir.exists():
        print(f"⚠️ WARNING: Frontend folder not found at {frontend_dir}")
    
    # Start browser opener in thread
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Start backend in thread
    backend_thread = threading.Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Start frontend in main thread
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n\n👋 Servers stopped!")
        sys.exit(0)