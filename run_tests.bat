@echo off
REM ============================================
REM Run Tests with Coverage Report
REM ============================================

call .venv\Scripts\activate.bat
echo.
echo Running tests with coverage...
echo.
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
echo.
echo Coverage report saved to: htmlcov/index.html
pause
