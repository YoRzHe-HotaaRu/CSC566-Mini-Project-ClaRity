@echo off
REM ============================================
REM Road Surface Layer Analyzer - GUI Launcher
REM ============================================

echo.
echo ========================================
echo Road Surface Layer Analyzer
echo ========================================
echo.
echo Activating virtual environment...
call .venv\Scripts\activate.bat
echo.
echo Starting GUI Application...
echo.
echo (The GUI window will open separately)
echo.

python -m gui.main_window

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo ERROR: GUI exited with code %ERRORLEVEL%
    echo ========================================
    echo.
    echo Troubleshooting:
    echo 1. Make sure you're in the project directory
    echo 2. Check that .venv folder exists
    echo 3. Try: python -m gui.main_window directly
    echo.
    pause
)
