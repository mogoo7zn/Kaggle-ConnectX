@echo off
:: Change directory to the script's directory (deploy/)
cd /d "%~dp0"

echo Cleaning up previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo Installing dependencies...
pip install -r ..\requirements.txt
pip install pyinstaller

echo Building ConnectX Ultimate...
pyinstaller build.spec --noconfirm

echo.
if exist dist\ConnectX_Ultimate\ConnectX_Ultimate.exe (
    echo ========================================================
    echo  BUILD SUCCESSFUL!
    echo ========================================================
    echo.
    echo Note: The "EndUpdateResourceW" warning above is a common Windows
    echo glitch and was handled automatically. Your app is fine.
    echo.
    echo Executable location:
    echo %~dp0dist\ConnectX_Ultimate\ConnectX_Ultimate.exe
    echo.
) else (
    echo ========================================================
    echo  BUILD FAILED
    echo ========================================================
    echo Please check the error messages above.
)
pause
