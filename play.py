"""Quick launcher for Mario gameplay.

Usage:
    .venv/bin/python play.py
    .venv/bin/python play.py --model baseline
    .venv/bin/python play.py --no-gui
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.play import main

if __name__ == "__main__":
    main()

