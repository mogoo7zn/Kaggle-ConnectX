#!/bin/bash

# Change directory to the script's directory (deploy/)
cd "$(dirname "$0")"

echo "Cleaning up previous builds..."
rm -rf build dist

# Create and activate a virtual environment to avoid PEP 668 errors
# This is necessary on modern Linux distributions (Ubuntu 24.04+, Debian 12+)
if [ ! -d "build_venv" ]; then
    echo "Creating virtual environment (build_venv)..."
    python3 -m venv build_venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment."
        echo "Please ensure python3-venv is installed (e.g., sudo apt install python3-venv)"
        exit 1
    fi
fi

echo "Activating virtual environment..."
source build_venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r ../requirements.txt
pip install pyinstaller

echo "Building ConnectX Ultimate..."
pyinstaller build.spec --noconfirm

echo ""
if [ -f "dist/ConnectX_Ultimate/ConnectX_Ultimate" ]; then
    echo "========================================================"
    echo " BUILD SUCCESSFUL!"
    echo "========================================================"
    echo ""
    echo "Executable location:"
    echo "$(pwd)/dist/ConnectX_Ultimate/ConnectX_Ultimate"
    echo ""
else
    echo "========================================================"
    echo " BUILD FAILED"
    echo "========================================================"
    echo "Please check the error messages above."
fi

deactivate
