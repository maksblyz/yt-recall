#!/usr/bin/env python3
"""
Simple script to run the yt-recall app in one line.
Usage: python run_app.py [backend|frontend|both]
"""

import os
import sys
import subprocess
import threading
import time
import signal
import socket
from pathlib import Path

def find_available_port(start_port=8000, max_attempts=100):
    """Find an available port starting from start_port"""
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

def run_backend_thread(port):
    """Start the FastAPI backend server on a specific port (for threading)"""
    print(f"🚀 Starting backend server on port {port}...")
    
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    backend_dir = project_root / "backend"
    
    # Change to backend directory
    original_dir = os.getcwd()
    os.chdir(backend_dir)
    
    try:
        # Install dependencies if needed
        if not Path("venv").exists():
            print("Creating virtual environment and installing dependencies...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Activate venv and install requirements
        if os.name == 'nt':  # Windows
            python_path = Path("venv") / "Scripts" / "python.exe"
            pip_path = Path("venv") / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            python_path = Path("venv") / "bin" / "python"
            pip_path = Path("venv") / "bin" / "pip"
        
        if python_path.exists():
            print("Installing backend dependencies...")
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            
            print(f"🌐 Starting backend on http://localhost:{port}")
            subprocess.run([str(python_path), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"])
        else:
            print("Virtual environment not found. Please run: python -m venv backend/venv")
    finally:
        # Return to original directory
        os.chdir(original_dir)

def run_backend():
    """Start the FastAPI backend server"""
    print("Starting backend server...")
    
    # Find available port
    try:
        port = find_available_port(8000)
        print(f"🔍 Found available port: {port}")
    except RuntimeError as e:
        print(f"{e}")
        return None
    
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    backend_dir = project_root / "backend"
    
    # Change to backend directory
    original_dir = os.getcwd()
    os.chdir(backend_dir)
    
    try:
        # Install dependencies if needed
        if not Path("venv").exists():
            print("Creating virtual environment and installing dependencies...")
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        
        # Activate venv and install requirements
        if os.name == 'nt':  # Windows
            python_path = Path("venv") / "Scripts" / "python.exe"
            pip_path = Path("venv") / "Scripts" / "pip.exe"
        else:  # Unix/Linux/macOS
            python_path = Path("venv") / "bin" / "python"
            pip_path = Path("venv") / "bin" / "pip"
        
        if python_path.exists():
            print("Installing backend dependencies...")
            subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
            
            print(f"Starting backend on http://localhost:{port}")
            subprocess.run([str(python_path), "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", str(port), "--reload"])
        else:
            print("Virtual environment not found. Please run: python -m venv backend/venv")
            return None
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    return port

def run_frontend(backend_port=8000):
    """Start the React frontend server"""
    print("🚀 Starting frontend server...")
    
    # Get the absolute path to the project root
    project_root = Path(__file__).parent.absolute()
    frontend_dir = project_root / "frontend"
    
    # Change to frontend directory
    original_dir = os.getcwd()
    os.chdir(frontend_dir)
    
    try:
        # Check if node_modules exists, if not install dependencies
        if not Path("node_modules").exists():
            print("Installing frontend dependencies...")
            subprocess.run(["npm", "install"], check=True)
        
        # Set environment variable for backend port
        env = os.environ.copy()
        env['REACT_APP_API_URL'] = f"http://localhost:{backend_port}"
        
        print(f"🌐 Starting frontend on http://localhost:3000 (connecting to backend on port {backend_port})")
        subprocess.run(["npm", "start"], env=env)
    finally:
        # Return to original directory
        os.chdir(original_dir)

def run_docker():
    """Start the app using Docker Compose"""
    print("Starting app with Docker Compose...")
    subprocess.run(["docker-compose", "up", "--build"])

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_app.py [backend|frontend|both|docker]")
        print("  backend  - Start only the FastAPI backend")
        print("  frontend - Start only the React frontend")
        print("  both     - Start both backend and frontend (default)")
        print("  docker   - Start using Docker Compose")
        return
    
    mode = sys.argv[1].lower()
    
    if mode == "docker":
        run_docker()
        return
    
    try:
        if mode == "backend":
            run_backend()
        elif mode == "frontend":
            run_frontend()
        elif mode == "both":
            # Find available port first
            try:
                backend_port = find_available_port(8000)
                print(f"🔍 Found available port for backend: {backend_port}")
            except RuntimeError as e:
                print(f"Error: {e}")
                return
            
            # Start backend in a separate thread
            def run_backend_with_port():
                run_backend_thread(backend_port)
            
            backend_thread = threading.Thread(target=run_backend_with_port, daemon=True)
            backend_thread.start()
            
            # Wait a bit for backend to start
            time.sleep(3)
            
            # Start frontend with the correct backend port
            run_frontend(backend_port)
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: backend, frontend, both, docker")
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 