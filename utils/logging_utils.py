"""
Logging utilities for the Automated Trading System.
Configures logging format, handlers, and convenience functions.
"""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys

from config import settings

# Ensure log directory exists
os.makedirs(settings.LOG_DIR, exist_ok=True)

# Define log filename with date
def get_log_filename():
    """Get the log filename with current date"""
    today = datetime.now().strftime('%Y-%m-%d')
    return os.path.join(settings.LOG_DIR, f"trading_system_{today}.log")

def setup_logger(name, level=None):
    """
    Set up a logger with file and console handlers
    
    Args:
        name (str): Name of the logger
        level (str, optional): Log level, defaults to settings.LOG_LEVEL
        
    Returns:
        logger: Configured logger
    """
    if level is None:
        level = settings.LOG_LEVEL
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Create logger and set level
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler (rotating to limit size)
    file_handler = RotatingFileHandler(
        get_log_filename(),
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def log_execution_time(logger):
    """
    Decorator to log execution time of functions
    
    Args:
        logger: Logger to use for logging
        
    Returns:
        decorator: Execution time logging decorator
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.now()
            logger.debug(f"Starting {func.__name__}")
            
            result = func(*args, **kwargs)
            
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            logger.debug(f"Finished {func.__name__} in {execution_time:.2f} seconds")
            
            return result
        return wrapper
    return decorator

# Create a system logger
system_logger = setup_logger('system')

def log_error(exc, context=None):
    """
    Log an exception with optional context
    
    Args:
        exc (Exception): The exception to log
        context (dict, optional): Additional context information
    """
    error_msg = f"ERROR: {type(exc).__name__}: {str(exc)}"
    if context:
        error_msg += f" | Context: {context}"
    
    system_logger.error(error_msg, exc_info=True)
    
    # In debug mode, also log to console
    if settings.DEBUG:
        print(error_msg, file=sys.stderr)

def log_trade(trade_info):
    """
    Log trading activity to a dedicated trade log
    
    Args:
        trade_info (dict): Information about the trade
    """
    trade_logger = setup_logger('trades')
    
    # Format trade information
    timestamp = datetime.now().isoformat()
    symbol = trade_info.get('symbol', 'Unknown')
    action = trade_info.get('action', 'Unknown')
    quantity = trade_info.get('quantity', 0)
    price = trade_info.get('price', 0.0)
    
    trade_msg = f"TRADE: {timestamp} | {symbol} | {action} | {quantity} | {price}"
    
    # Add additional information if available
    for key in ['stop_loss', 'target', 'strategy', 'timeframe']:
        if key in trade_info:
            trade_msg += f" | {key}: {trade_info[key]}"
    
    trade_logger.info(trade_msg)