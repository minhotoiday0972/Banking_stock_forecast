#!/usr/bin/env python3
"""
Schedule Pipeline Runner
Automatically runs pipeline at specified times
"""
import schedule
import time
import subprocess
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/scheduler.log'),
        logging.StreamHandler()
    ]
)

def run_daily_pipeline():
    """Run daily pipeline update"""
    logging.info("🚀 Starting scheduled pipeline run")
    
    try:
        # Run quick pipeline (transformer only for daily updates)
        result = subprocess.run(
            ["python", "run_full_pipeline.py", "quick"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logging.info("✅ Scheduled pipeline completed successfully")
            
            # Run results check
            subprocess.run(["python", "check_results.py"], capture_output=True)
            logging.info("📊 Results check completed")
            
        else:
            logging.error(f"❌ Scheduled pipeline failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logging.error("⏰ Pipeline timed out after 1 hour")
    except Exception as e:
        logging.error(f"❌ Scheduler error: {e}")

def run_weekly_full_pipeline():
    """Run full pipeline weekly"""
    logging.info("🚀 Starting scheduled FULL pipeline run")
    
    try:
        # Run full pipeline (all models)
        result = subprocess.run(
            ["python", "run_full_pipeline.py", "full"],
            capture_output=True,
            text=True,
            timeout=7200  # 2 hours timeout
        )
        
        if result.returncode == 0:
            logging.info("✅ Scheduled FULL pipeline completed successfully")
        else:
            logging.error(f"❌ Scheduled FULL pipeline failed: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        logging.error("⏰ Full pipeline timed out after 2 hours")
    except Exception as e:
        logging.error(f"❌ Full scheduler error: {e}")

def setup_schedule():
    """Setup the schedule"""
    # Daily quick update at 7 PM (after market close in Vietnam - 3:30 PM + buffer)
    schedule.every().day.at("19:00").do(run_daily_pipeline)
    
    # Weekly full retrain on Sunday at 2 AM
    schedule.every().sunday.at("02:00").do(run_weekly_full_pipeline)
    
    logging.info("📅 Schedule setup complete:")
    logging.info("   - Daily quick pipeline: 19:00 (7 PM)")
    logging.info("   - Weekly full pipeline: Sunday 02:00")

def main():
    """Main scheduler function"""
    print("🕐 Banking Stock Prediction Scheduler")
    print("=====================================")
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Setup schedule
    setup_schedule()
    
    print("⏰ Scheduler started. Press Ctrl+C to stop.")
    print("📋 Check logs/scheduler.log for activity")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n🛑 Scheduler stopped by user")
        logging.info("🛑 Scheduler stopped by user")

if __name__ == "__main__":
    main()