@echo off
REM ============================================
REM Run Road Surface Layer Analyzer GUI
REM ============================================

call .venv\Scripts\activate.bat
python -m gui.main_window
