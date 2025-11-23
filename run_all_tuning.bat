@echo OFF
setlocal ENABLEDELAYEDEXPANSION

REM =================================================================
REM SCRIPT TO AUTOMATE OPTUNA HYPERPARAMETER TUNING
REM =================================================================
REM This script will loop through all specified models, tickers, and
REM horizons and run the hyperparameter tuning for each combination.
REM 
REM Please be aware that this process can be very time-consuming, 
REM potentially taking many hours or even days to complete depending
REM on the number of combinations and trials.
REM =================================================================

REM --- CONFIGURATION ---
REM Set the number of trials for each Optuna study.
SET TRIALS=50

REM Define the models, tickers, and horizons to tune.
REM These should match your config.yaml file.
SET MODELS_TO_TUNE=cnn_bilstm transformer
SET TICKERS_TO_TUNE=VCB BID CTG TCB MBB VPB ACB STB HDB TPB
SET HORIZONS_TO_TUNE=1 3 5 30 60 90

echo.
echo Starting batch hyperparameter tuning...
echo.
echo Configuration:
echo   - Trials per study: %TRIALS%
echo   - Models: %MODELS_TO_TUNE%
echo   - Tickers: %TICKERS_TO_TUNE%
echo   - Horizons: %HORIZONS_TO_TUNE%
echo.

FOR %%m IN (%MODELS_TO_TUNE%) DO (
    FOR %%t IN (%TICKERS_TO_TUNE%) DO (
        FOR %%h IN (%HORIZONS_TO_TUNE%) DO (
            echo =================================================================
            echo [INFO] Starting Study: Model=%%m, Ticker=%%t, Horizon=%%h
            echo =================================================================
            
            REM Construct the study name
            SET "STUDY_NAME=study-%%m-%%t-t%%h"
            
            REM Run the tuning script
            python hyperparameter_tuning.py --model %%m --ticker %%t --horizon %%h --trials %TRIALS% --study-name !STUDY_NAME!
            
            IF ERRORLEVEL 1 (
                echo.
                echo [ERROR] An error occurred during the study for %%m, %%t, t+%%h.
                echo Stopping the batch process. Please check the logs.
                echo.
                goto:eof
            )
            
            echo.
            echo [SUCCESS] Finished Study: Model=%%m, Ticker=%%t, Horizon=%%h
            echo.
        )
    )
)

echo =================================================================
echo ALL HYPERPARAMETER TUNING JOBS ARE COMPLETE.
echo =================================================================

endlocal
