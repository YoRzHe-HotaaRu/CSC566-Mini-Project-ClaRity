@echo off
REM ============================================
REM Install PyTorch with CUDA 12.x for RTX 4050
REM ============================================

call .venv\Scripts\activate.bat

echo.
echo ========================================
echo   Installing PyTorch with CUDA 12.x
echo   Optimized for RTX 4050 GPU
echo ========================================
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install PyTorch with CUDA
    echo Try installing manually:
    echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    exit /b 1
)

echo.
echo Verifying CUDA availability...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ========================================
echo   PyTorch with CUDA installed!
echo ========================================
echo.
pause
