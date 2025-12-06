@echo off
REM ============================================
REM ConnectX Environment Setup Script (Windows)
REM ============================================
REM This script automatically creates a virtual environment
REM and installs all required dependencies for the ConnectX project.

echo ============================================
echo ConnectX Environment Setup (Windows)
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH!
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version

REM Get the project root directory (parent of scripts folder)
set "PROJECT_ROOT=%~dp0.."
cd /d "%PROJECT_ROOT%"

REM Check if virtual environment already exists
if exist "venv" (
    echo.
    echo [WARNING] Virtual environment 'venv' already exists!
    set /p OVERWRITE="Do you want to recreate it? (y/N): "
    if /i not "%OVERWRITE%"=="y" (
        echo Skipping virtual environment creation...
        goto :install_deps
    )
    echo Removing existing virtual environment...
    rmdir /s /q venv
)

echo.
echo [2/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    echo Make sure you have 'venv' module installed.
    pause
    exit /b 1
)

echo [SUCCESS] Virtual environment created!

:install_deps
echo.
echo [3/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo [4/4] Installing dependencies from requirements.txt...
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found!
    echo Please make sure you're running this script from the project root.
    pause
    exit /b 1
)

python -m pip install --upgrade pip
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install some dependencies!
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ============================================
echo Setup Complete!
echo ============================================
echo.
echo To activate the virtual environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, simply run:
echo   deactivate
echo.
echo You can now start training agents or playing the game:
echo   python playground\play.py
echo.
pause

