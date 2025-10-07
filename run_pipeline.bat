@echo off
echo ========================================
echo Banking Stock Prediction Pipeline
echo ========================================
echo.

:menu
echo Choose an option:
echo 1. Run Full Pipeline (All Models)
echo 2. Run Quick Pipeline (Transformer Only)
echo 3. Check Results
echo 4. Update Data Only
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto full_pipeline
if "%choice%"=="2" goto quick_pipeline
if "%choice%"=="3" goto check_results
if "%choice%"=="4" goto update_data
if "%choice%"=="5" goto exit
echo Invalid choice. Please try again.
goto menu

:full_pipeline
echo.
echo Running Full Pipeline...
python run_full_pipeline.py full
if %errorlevel% equ 0 (
    echo.
    echo Pipeline completed successfully!
    python check_results.py
) else (
    echo.
    echo Pipeline failed. Check logs for details.
)
pause
goto menu

:quick_pipeline
echo.
echo Running Quick Pipeline...
python run_full_pipeline.py quick
if %errorlevel% equ 0 (
    echo.
    echo Quick pipeline completed successfully!
    python check_results.py
) else (
    echo.
    echo Quick pipeline failed. Check logs for details.
)
pause
goto menu

:check_results
echo.
echo Checking Results...
python check_results.py
pause
goto menu

:update_data
echo.
echo Updating Data...
python main.py collect
echo Data update completed.
pause
goto menu

:exit
echo.
echo Goodbye!
exit /b 0