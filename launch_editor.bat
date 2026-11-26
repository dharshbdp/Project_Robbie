@echo off
title Robbie Studio Launcher
echo ===================================================
echo Starting Robbie Studio Desktop App (Tauri v2)...
echo ===================================================
cd /d "%~dp0editor"
if not exist "node_modules\@tauri-apps\cli" (
    echo Installing editor dependencies...
    call npm.cmd ci --ignore-scripts
    if errorlevel 1 exit /b 1
)
call npm.cmd run tauri dev
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Notice] Check that Node.js, Rust, and the Project Robbie Python environment are installed.
    echo.
    pause
)
