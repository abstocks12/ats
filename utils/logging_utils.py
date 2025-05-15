"""
Logging utilities for the Automated Trading System.
Configures logging format, handlers, and convenience functions.
"""

import os
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler
import sys
import functools
import time

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
        @functools.wraps(func)
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

def setup_subprocess_logger(name):
    """
    Set up a logger for subprocess output
    
    Args:
        name (str): Name of the subprocess
        
    Returns:
        logger: Configured logger
    """
    logger = setup_logger(f"subprocess.{name}")
    return logger

def log_data_collection(logger, data_type, symbol, exchange, status, count=None, duration=None):
    """
    Log a data collection event with standardized format
    
    Args:
        logger: Logger to use
        data_type (str): Type of data being collected
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        status (str): Collection status ('started', 'completed', 'failed')
        count (int, optional): Number of items collected
        duration (float, optional): Duration of collection in seconds
    """
    msg = f"Data collection {status} - {data_type} for {symbol}:{exchange}"
    
    if count is not None:
        msg += f" | Items: {count}"
    
    if duration is not None:
        msg += f" | Duration: {duration:.2f}s"
    
    if status == 'started':
        logger.info(msg)
    elif status == 'completed':
        logger.info(msg)
    elif status == 'failed':
        logger.error(msg)
    else:
        logger.info(msg)

def log_performance(operation_name, start_time=None):
    """
    Context manager to log performance of a block of code
    
    Args:
        operation_name (str): Name of the operation to log
        start_time (datetime, optional): Start time if already recorded
        
    Example:
        with log_performance('data_processing'):
            process_data()
    """
    class PerformanceLogger:
        def __init__(self, operation_name, start_time):
            self.operation_name = operation_name
            self.start_time = start_time
            self.logger = setup_logger("performance")
        
        def __enter__(self):
            if self.start_time is None:
                self.start_time = datetime.now()
            self.logger.debug(f"Starting {self.operation_name}")
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            if exc_type is None:
                self.logger.debug(f"Completed {self.operation_name} in {duration:.2f} seconds")
            else:
                self.logger.error(f"Failed {self.operation_name} after {duration:.2f} seconds: {exc_val}")
    
    return PerformanceLogger(operation_name, start_time)