"""
Main entry point for running guixi as a module.

Usage:
    python -m guixi           # Launch GUI
    python -m guixi gui       # Launch GUI explicitly
    python -m guixi web       # Start web server
    python -m guixi infer "?" # Run inference
"""

from .cli import main

if __name__ == "__main__":
    main()
