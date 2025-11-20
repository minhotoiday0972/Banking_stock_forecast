@echo off
REM Backup old models before retraining
echo ========================================
echo BACKUP OLD MODELS
echo ========================================

REM Create backup directory with timestamp
set timestamp=%date:~-4%%date:~3,2%%date:~0,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set timestamp=%timestamp: =0%
set backup_dir=models_backup_%timestamp%

echo Creating backup directory: %backup_dir%
mkdir %backup_dir%

REM Backup models
echo Backing up models...
xcopy /E /I /Y models %backup_dir%\models

REM Backup outputs
echo Backing up outputs...
xcopy /E /I /Y outputs %backup_dir%\outputs

REM Backup logs
echo Backing up logs...
xcopy /E /I /Y logs %backup_dir%\logs

echo.
echo ========================================
echo BACKUP COMPLETED!
echo ========================================
echo Backup location: %backup_dir%
echo.
echo You can now run: python run_full_pipeline.py
echo.
pause
