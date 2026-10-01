#!/usr/bin/env python3
"""
Setup script for VoiceStudio backend dependencies
"""

import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, cwd=None):
    print(f"[setup] Running: {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        print(f"[setup] Command failed with exit code {result.returncode}")
        return False
    return True

def main():
    backend_dir = Path(__file__).parent.parent / "backend"
    
    print("[setup] Setting up VoiceStudio backend...")
    
    # Install Python dependencies with uv
    if not run_cmd("uv sync", cwd=backend_dir):
        print("[setup] uv sync failed, trying pip...")
        if not run_cmd("pip install -e .", cwd=backend_dir):
            return False
    
    # Download models if needed
    print("[setup] Setup complete!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
