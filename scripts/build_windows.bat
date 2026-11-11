@echo off
rem Convenience script to build the portable Windows executable for USB deployment
chcp 65001 >nul
set SCRIPT_DIR=%~dp0
cd /d "%SCRIPT_DIR%.."

echo Building Windows standalone player executable...
python scripts\build_usb_dist.py
if errorlevel 1 (
    echo [ERROR] Build failed. Verify pyinstaller is installed.
    pause
    exit /b 1
)
echo [SUCCESS] Build complete.
pause
