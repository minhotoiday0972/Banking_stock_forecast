# src/utils/logger.py
import logging
import os
from datetime import datetime

# Global logger instances
_loggers = {}

def get_logger(name: str, log_filename: str = None, log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a standard Python logger.

    Args:
        name (str): The name for the logger, which will appear in log messages.
        log_filename (str, optional): A specific filename for the log file. 
                                     If None, a name is generated from the logger name and date.
                                     Defaults to None.
        log_dir (str, optional): The directory to save log files. Defaults to "logs".
        level (int, optional): The logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: A configured logger instance.
    """
    # Use a unique key for the logger instance to allow different file handlers for the same name
    logger_key = f"{name}-{log_filename}" if log_filename else name
    if logger_key in _loggers:
        return _loggers[logger_key]

    # Create logger instance with the unique key
    logger = logging.getLogger(logger_key)
    logger.setLevel(level)
    logger.propagate = False # Prevent log messages from propagating to the root logger
    
    # Prevent duplicate handlers if this function is ever re-called for the same key
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)

    # Use the original 'name' in the log format for clarity
    file_formatter = logging.Formatter(
        f'%(asctime)s - {name} - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        f'%(levelname)s - {name} - %(message)s'
    )

    # Determine the log file name
    effective_log_filename = log_filename
    if effective_log_filename is None:
        effective_log_filename = f"{name.replace('__', '_')}_{datetime.now().strftime('%Y%m%d')}.log"
    
    log_file_path = os.path.join(log_dir, effective_log_filename)

    # File handler
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Keep console output concise
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    _loggers[logger_key] = logger
    return logger
