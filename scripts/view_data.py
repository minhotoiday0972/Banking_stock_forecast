#!/usr/bin/env python3
"""
Data viewing utility script
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from src.utils.data_viewer import main

if __name__ == "__main__":
    main()
