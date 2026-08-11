@echo off
REM ============================================================
REM setup_env.bat — Tạo virtual environment và cài dependencies
REM Dùng trên Windows (local, CPU/CUDA)
REM ============================================================
REM Cách dùng:
REM   setup_env.bat           <- cài torch CPU (cho smoke test)
REM   setup_env.bat cuda      <- cài torch CUDA 12.8 (nếu có GPU driver)
REM ============================================================

setlocal

set VENV_DIR=venv
set PYTHON=py

echo [1/4] Kiem tra Python...
%PYTHON% --version
if errorlevel 1 (
    echo [ERROR] Khong tim thay Python. Cai dat Python tu python.org
    exit /b 1
)

echo [2/4] Tao virtual environment: %VENV_DIR%\
%PYTHON% -m venv %VENV_DIR%
if errorlevel 1 (
    echo [ERROR] Khong the tao venv.
    exit /b 1
)

echo [3/4] Kich hoat venv va cai PyTorch...
call %VENV_DIR%\Scripts\activate.bat

if "%1"=="cuda" (
    echo Cai PyTorch CUDA 12.8...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
) else (
    echo Cai PyTorch CPU ^(smoke test^)...
    pip install torch
)

echo [4/4] Installing other dependencies...
pip install --quiet -r requirements.txt

echo [5/5] Applying compatibility patches...
python patch_venv.py

echo.
echo ============================================================
echo Setup complete! Activate the venv with:
echo   source venv/bin/activate
echo.
echo Run smoke test (quick sanity check, any machine):
echo   python train.py --config configs/train_smoke.yaml --smoke
echo.
echo Run full training (cloud GPU):
echo   python train.py --config configs/train_v1.yaml
echo.
echo With accelerate (multi-GPU):
echo   accelerate launch train.py --config configs/train_v1.yaml
echo ============================================================
endlocal
