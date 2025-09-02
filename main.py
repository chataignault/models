#!/usr/bin/env python3
"""
Refactored main entry point for drone detection system.
Uses modular components for clean, maintainable code.

This file provides backward compatibility with the original main.py
while redirecting to the new modular video_processor.py implementation.
"""

from video_processor import VideoProcessor, main

# Legacy import compatibility - redirect to refactored VideoProcessor
__all__ = ['VideoProcessor', 'main']

if __name__ == "__main__":
    main()