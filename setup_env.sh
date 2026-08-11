#!/usr/bin/env bash
# ============================================================
# setup_env.sh — Tạo virtual environment và cài dependencies
# Dùng trên Linux (cloud: Colab, Kaggle, RunPod, Lambda, AWS…)
# ============================================================
# Cách dùng:
#   bash setup_env.sh           <- auto-detect CUDA
#   bash setup_env.sh cpu       <- ép dùng torch CPU
#   bash setup_env.sh cu118     <- torch CUDA 11.8
#   bash setup_env.sh cu121     <- torch CUDA 12.1
#   bash setup_env.sh cu128     <- torch CUDA 12.8 (mới nhất)
# ============================================================

set -e

VENV_DIR="venv"
PYTHON="${PYTHON:-python3}"
CUDA_ARG="${1:-auto}"

echo "=== Fine-tuneViT5 Environment Setup ==="

# --- Phát hiện CUDA nếu auto ---
if [ "$CUDA_ARG" = "auto" ]; then
    if command -v nvcc &>/dev/null; then
        CUDA_VER=$(nvcc --version | grep -oP "release \K[0-9]+\.[0-9]+" | head -1)
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        if   [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then CUDA_ARG="cu128"
        elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then CUDA_ARG="cu121"
        elif [ "$CUDA_MAJOR" -ge 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then CUDA_ARG="cu118"
        else CUDA_ARG="cpu"
        fi
        echo "[auto] CUDA $CUDA_VER detected -> using torch build: $CUDA_ARG"
    elif command -v nvidia-smi &>/dev/null; then
        CUDA_ARG="cu128"
        echo "[auto] nvidia-smi found, defaulting to cu128"
    else
        CUDA_ARG="cpu"
        echo "[auto] No GPU found -> CPU build"
    fi
fi

echo "[1/4] Python: $($PYTHON --version)"
echo "[2/4] Creating venv: $VENV_DIR/"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "[3/4] Installing PyTorch ($CUDA_ARG)..."
case "$CUDA_ARG" in
    cu128)
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 ;;
    cu121)
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 ;;
    cu118)
        pip install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 ;;
    cpu)
        pip install --quiet torch ;;
    *)
        echo "[WARN] Unknown CUDA arg '$CUDA_ARG', falling back to CPU"
        pip install --quiet torch ;;
esac

echo "[4/4] Installing other dependencies..."
pip install --quiet -r requirements.txt

echo "[5/5] Applying compatibility patches (vit5-base tokenizer fix)..."
python patch_venv.py

echo ""
echo "============================================================"
echo "Setup complete! Activate the venv with:"
echo "  source venv/bin/activate"
echo ""
echo "Run smoke test (quick sanity check, any machine):"
echo "  python train.py --config configs/train_smoke.yaml --smoke"
echo ""
echo "Run full training (cloud GPU):"
echo "  python train.py --config configs/train_v1.yaml"
echo ""
echo "With accelerate (multi-GPU):"
echo "  accelerate launch train.py --config configs/train_v1.yaml"
echo "============================================================"
