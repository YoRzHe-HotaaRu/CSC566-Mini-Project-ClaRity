@echo off
REM ============================================
REM Launch GUI with Python Direct
REM ============================================

echo.
echo ========================================
echo Road Surface Layer Analyzer
echo ========================================
echo.
echo Starting GUI Application...
echo.

REM Use the Python from virtual environment directly
.venv\Scripts\python.exe -m gui.main_window

REM If GUI closes unexpectedly, pause to show error
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo GUI Exited with Error!
    echo ========================================
    echo Error Code: %ERRORLEVEL%
    echo.
    echo Possible causes:
    echo - Missing dependencies (run setup.bat)
    echo - Python version incompatibility
    echo - Missing PyQt5
    echo.
    pause
)
