@echo off
chcp 65001 >nul
echo =======================================================
echo   LLM Tile Visualizer (tilevis) - Windows Demo Launcher
echo =======================================================

:: 1. Activate local virtual environment if present
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

:: 2. Run preflight environment & sample trace verification
echo [1/2] Running preflight diagnostics...
python -m tilevis check traces\samples\tiny_demo.trace.npz
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Preflight diagnostics failed. Please fix the issue above.
    pause
    exit /b %errorlevel%
)

:: 3. Launch interactive playback
echo.
echo [2/2] Starting interactive presentation playback...
python -m tilevis play traces\samples\tiny_demo.trace.npz %*
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Playback exited with code %errorlevel%.
    pause
    exit /b %errorlevel%
)
