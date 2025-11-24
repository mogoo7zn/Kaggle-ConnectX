@echo off
echo ========================================
echo Embedding Model into main.py
echo ========================================
echo.

echo Step 1: Converting model to Base64...
python embed_model.py ..\submission\best_model.pth model_weights_embedded.txt
if %errorlevel% neq 0 (
    echo Error in step 1!
    pause
    exit /b 1
)
echo.

echo Step 2: Creating embedded main.py...
python create_embedded_main.py --original ..\submission\main_backup.py --model model_weights_embedded.txt --output ..\submission\main.py
if %errorlevel% neq 0 (
    echo Error in step 2!
    pause
    exit /b 1
)
echo.

echo Step 3: Cleaning up...
del model_weights_embedded.txt
echo.

echo ========================================
echo Completed! main.py is ready!
echo ========================================
pause

