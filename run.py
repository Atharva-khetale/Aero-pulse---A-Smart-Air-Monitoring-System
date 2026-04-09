#!/usr/bin/env python3
"""
SmartAir — One-shot launcher
==============================
Run:   python run.py
Does:
  1. Trains / retrains ML model (overwrites pkl each time)
  2. Starts Flask server on port 5000
  3. Opens browser automatically
"""

import os, sys, webbrowser, threading, time

# Ensure we run from the project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def open_browser():
    time.sleep(2.5)
    webbrowser.open("http://localhost:5000")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║   🌬  SmartAir — Atmospheric Intelligence        ║
║      Aura Edition  |  Flask + ML Backend         ║
╚══════════════════════════════════════════════════╝
""")

    # Step 1: Train / retrain model
    from train_model import run as train_run
    train_run()

    # Step 2: Open browser in background
    threading.Thread(target=open_browser, daemon=True).start()

    # Step 3: Start Flask
    from app import app
    print("\n🚀 Server running at http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
