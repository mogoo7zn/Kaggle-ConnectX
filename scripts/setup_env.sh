#!/bin/bash
# ============================================
# ConnectX Environment Setup Script (Linux/Mac)
# ============================================
# This script automatically creates a virtual environment
# and installs all required dependencies for the ConnectX project.

set -e  # Exit on error

echo "============================================"
echo "ConnectX Environment Setup (Linux/Mac)"
echo "============================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed or not in PATH!"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/4] Checking Python version..."
python3 --version

# Get the project root directory (parent of scripts folder)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Check if virtual environment already exists
if [ -d "venv" ]; then
    echo ""
    echo "[WARNING] Virtual environment 'venv' already exists!"
    read -p "Do you want to recreate it? (y/N): " OVERWRITE
    if [[ ! "$OVERWRITE" =~ ^[Yy]$ ]]; then
        echo "Skipping virtual environment creation..."
    else
        echo "Removing existing virtual environment..."
        rm -rf venv
        echo "[2/4] Creating virtual environment..."
        python3 -m venv venv
        echo "[SUCCESS] Virtual environment created!"
    fi
else
    echo ""
    echo "[2/4] Creating virtual environment..."
    python3 -m venv venv
    echo "[SUCCESS] Virtual environment created!"
fi

echo ""
echo "[3/4] Activating virtual environment..."
source venv/bin/activate

echo ""
echo "[4/4] Installing dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    echo "[ERROR] requirements.txt not found!"
    echo "Please make sure you're running this script from the project root."
    exit 1
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "[ERROR] Failed to install some dependencies!"
    echo "Please check the error messages above."
    exit 1
fi

echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, simply run:"
echo "  deactivate"
echo ""
echo "You can now start training agents or playing the game:"
echo "  python playground/play.py"
echo ""

