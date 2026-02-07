#!/usr/bin/env python3
"""Wrapper to run the advanced analysis implementation."""
from pathlib import Path
import runpy
import sys


def main():
    script = Path(__file__).resolve().parent / "unused" / "analyze_advanced3.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing analysis script: {script}")
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
