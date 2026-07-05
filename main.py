"""Entry point.

Run:                 python main.py
Gaming mode (CPU):   JOI_MODE=gaming python main.py
Plain fallback:      JOI_BACKEND=plain python main.py
"""
from joi.assistant import run

if __name__ == "__main__":
    run()