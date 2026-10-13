@echo off
title Robbie Studio Launcher
echo ===================================================
echo Starting Robbie Studio Desktop App (Tauri v2)...
echo ===================================================
cd /d "%~dp0editor"
call npm.cmd run tauri dev
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [Notice] If you see a PowerShell execution policy error, run:
    echo Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
    echo.
    pause
)
