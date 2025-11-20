@echo off
REM Quick start script for training with improvements V2
echo ========================================
echo STOCK PREDICTION - TRAINING V2
echo ========================================
echo.
echo Improvements:
echo - Ticker-specific class weights
echo - Dynamic focal loss with horizon adjustment
echo - Optimized regularization
echo - Patient training parameters
echo.
echo ========================================
echo.

REM Check if backup is needed
set /p backup="Do you want to backup old models first? (y/n): "
if /i "%backup%"=="y" (
    echo.
    echo Running backup...
    call backup_old_models.bat
)

echo.
echo ========================================
echo STARTING TRAINING
echo ========================================
echo.
echo This will train 132 models:
echo - 11 tickers
echo - 2 architectures (CNN-BiLSTM, Transformer)
echo - 6 horizons (t+1, t+3, t+5, t+30, t+60, t+90)
echo.
echo Expected duration: 2-3 hours
echo.
echo ========================================
echo.

REM Start training
python run_full_pipeline.py

echo.
echo ========================================
echo TRAINING COMPLETED!
echo ========================================
echo.
echo Next steps:
echo 1. Run: python analyze_results.py
echo 2. Check: TRAINING_IMPROVEMENTS_V2.md for expected improvements
echo 3. Compare with previous results
echo.
pause
