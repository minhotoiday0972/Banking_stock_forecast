# src/utils/logger.py
import logging
import os
from datetime import datetime

# Global logger instances
_loggers = {}

def get_logger(name: str, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standard Python logger.

    Args:
        name (str): The name of the logger.
        log_dir (str, optional): The directory to save log files. Defaults to "logs".
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent duplicate handlers if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )

    # File handler
    log_file = os.path.join(
        log_dir,
        f"{name.replace('__', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Keep console output concise
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _loggers[name] = logger
    return logger
