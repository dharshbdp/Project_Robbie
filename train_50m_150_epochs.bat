@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================================
echo Project Robbie: Robbie-50M Model Training (150 Epochs SFT)
echo Strict 4.0 GB VRAM Safety Guard Active
echo ==============================================================================

if not exist ".venv\Scripts\python.exe" (
    echo Error: Python virtual environment not found in .venv.
    pause
    exit /b 1
)

echo.
echo Launching Stage 2 Supervised Fine-Tuning for 150 epochs on existing 50M Base...
.venv\Scripts\python.exe scripts\train_50m_chat.py --skip-stage1 --stage2-epochs 150 --batch-size 4 --accumulation-steps 4 --stage2-lr 1.5e-4 --vram-limit-gb 4.0

if %errorlevel% neq 0 (
    echo.
    echo Training encountered an error. Check the console output above.
    pause
    exit /b %errorlevel%
)

echo.
echo ==============================================================================
echo Robbie-50M 150-Epoch Training Complete!
echo Checkpoints updated in: checkpoints/
echo ==============================================================================
pause
