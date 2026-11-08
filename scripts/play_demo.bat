@echo off
chcp 65001 >nul
echo Starting LLM Tile Visualizer Demo...

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python -m tilevis check traces\samples\tiny_demo.trace.npz 2>nul
if %errorlevel% neq 0 (
    echo Note: preflight check skipped or sample trace not yet generated.
)

python -m tilevis demo
if %errorlevel% neq 0 (
    echo Error executing tilevis demo.
    pause
)
