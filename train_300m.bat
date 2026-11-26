@echo off
setlocal
cd /d "%~dp0"

echo ======================================================================
echo           PROJECT ROBBIE - 300M PARAMETER TRAINING PIPELINE
echo ======================================================================
echo Hardware: NVIDIA Quadro RTX 3000 / RTX 4060
echo Safety Guard: Strict Max VRAM Ceiling = 4.8 GB (Leaves Headroom for OS)
echo Thermal Pacing: Automatic cooling intervals enabled
echo ======================================================================

if not exist ".venv\Scripts\python.exe" (
    echo Error: Python virtual environment not found in .venv.
    pause
    exit /b 1
)

.venv\Scripts\python.exe -u scripts\train_300m_chat.py --epochs 6 --batch-size 2 --accumulation-steps 16 --vram-limit-gb 4.8 --eval-every 200 %*

echo ======================================================================
echo           ROBBIE 300M TRAINING RUN FINISHED
echo ======================================================================
pause
