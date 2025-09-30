#!/usr/bin/env python3
"""Legacy entry point for the bsy-clippy CLI."""
from __future__ import annotations

import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

from bsy_clippy.cli import main

if __name__ == "__main__":
    main()
