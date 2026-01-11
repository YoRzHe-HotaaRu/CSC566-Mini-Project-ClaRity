@echo off
REM ============================================
REM Road Surface Layer Analyzer - Setup Script
REM CSC566 Image Processing Mini Project
REM ============================================

echo.
echo ========================================
echo   Road Surface Layer Analyzer Setup
echo   ClaRity Group - CSC566
echo ========================================
echo.

REM Check Python version
python --version
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    exit /b 1
)

echo.
echo [1/4] Creating virtual environment...
python -m venv .venv
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to create virtual environment
    exit /b 1
)

echo.
echo [2/4] Activating virtual environment...
call .venv\Scripts\activate.bat

echo.
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo [4/4] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies
    exit /b 1
)

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo To activate the virtual environment, run:
echo   .venv\Scripts\activate
echo.
echo To run the application:
echo   python -m gui.main_window
echo.
echo To run tests:
echo   pytest tests/ -v
echo.
pause
