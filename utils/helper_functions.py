"""
Helper functions for the Automated Trading System.
Contains various utility functions used across the system.
"""

import os
import json
import uuid
import hashlib
import time
import random
from datetime import datetime, timedelta
import re
import threading
import functools
from typing import List, Dict, Any, Optional, Union, Callable, TypeVar

T = TypeVar('T')  # For generic functions

def generate_unique_id():
    """Generate a unique ID for transactions or records"""
    return str(uuid.uuid4())

def hash_string(input_string):
    """Create a hash of an input string"""
    return hashlib.sha256(input_string.encode()).hexdigest()

def retry_function(func, max_retries=3, delay=1, backoff=2, exceptions=(Exception,)):
    """
    Retry a function with exponential backoff
    
    Args:
        func: The function to retry
        max_retries (int): Maximum number of retries
        delay (float): Initial delay between retries in seconds
        backoff (float): Backoff multiplier
        exceptions (tuple): Exceptions to catch and retry
        
    Returns:
        The result of the function call
    """
    retries = 0
    while True:
        try:
            return func()
        except exceptions as e:
            retries += 1
            if retries >= max_retries:
                raise e
            
            sleep_time = delay * (backoff ** (retries - 1))
            # Add some randomness to avoid thundering herd problem
            sleep_time = sleep_time * (0.9 + 0.2 * random.random())
            time.sleep(sleep_time)

def parse_timeframe(timeframe):
    """
    Parse a timeframe string into minutes
    
    Args:
        timeframe (str): Timeframe string (e.g., '1m', '5m', '1h', '1d')
        
    Returns:
        int: Timeframe in minutes, or None for daily/weekly timeframes
    """
    if not timeframe:
        return None
    
    # Try simple numeric conversion first
    try:
        return int(timeframe)
    except ValueError:
        pass
    
    # Parse with regex
    pattern = r'^(\d+)([mhdw])$'
    match = re.match(pattern, timeframe.lower())
    
    if match:
        value, unit = match.groups()
        value = int(value)
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
        elif unit == 'w':
            return value * 60 * 24 * 7
    
    # Handle special cases
    if timeframe.lower() == 'day':
        return 60 * 24
    elif timeframe.lower() == 'week':
        return 60 * 24 * 7
    
    return None

def format_currency(amount, currency='₹'):
    """Format a number as currency"""
    if amount is None:
        return 'N/A'
    
    try:
        amount = float(amount)
        if amount >= 10000000:  # 1 crore
            return f"{currency}{amount/10000000:.2f} Cr"
        elif amount >= 100000:  # 1 lakh
            return f"{currency}{amount/100000:.2f} L"
        elif amount >= 1000:
            return f"{currency}{amount/1000:.2f} K"
        else:
            return f"{currency}{amount:.2f}"
    except (ValueError, TypeError):
        return str(amount)

def format_percentage(value):
    """Format a number as a percentage"""
    if value is None:
        return 'N/A'
    
    try:
        value = float(value)
        return f"{value:.2f}%"
    except (ValueError, TypeError):
        return str(value)

def get_date_range(days=30, end_date=None):
    """
    Get a date range for data collection
    
    Args:
        days (int): Number of days to include
        end_date (datetime, optional): End date, defaults to today
        
    Returns:
        tuple: (start_date, end_date) as datetime objects
    """
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.fromisoformat(end_date)
    
    start_date = end_date - timedelta(days=days)
    
    return start_date, end_date

def is_market_open(current_time=None):
    """
    Check if the market is currently open
    
    Args:
        current_time (datetime, optional): Time to check, defaults to now
        
    Returns:
        bool: True if market is open, False otherwise
    """
    from config import settings
    
    if current_time is None:
        current_time = datetime.now()
    
    # Check if it's a weekday (0 = Monday, 6 = Sunday)
    if current_time.weekday() >= 5:  # Saturday or Sunday
        return False
    
    # Create datetime objects for market open and close times
    market_open = current_time.replace(
        hour=settings.MARKET_OPEN_HOUR,
        minute=settings.MARKET_OPEN_MINUTE,
        second=0,
        microsecond=0
    )
    
    market_close = current_time.replace(
        hour=settings.MARKET_CLOSE_HOUR,
        minute=settings.MARKET_CLOSE_MINUTE,
        second=0,
        microsecond=0
    )
    
    # Check if current time is between market open and close
    return market_open <= current_time <= market_close

def sanitize_filename(filename):
    """
    Sanitize a string to be used as a filename
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename
    """
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    
    # Remove special characters
    filename = re.sub(r'[^\w\-\.]', '', filename)
    
    # Ensure it's not too long
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:255-len(ext)] + ext
    
    return filename

def load_json_file(file_path, default=None):
    """
    Load a JSON file with error handling
    
    Args:
        file_path (str): Path to the JSON file
        default (any, optional): Default value if file can't be loaded
        
    Returns:
        dict: Loaded JSON data or default value
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError):
        return default if default is not None else {}

def save_json_file(file_path, data, indent=2):
    """
    Save data to a JSON file
    
    Args:
        file_path (str): Path to the JSON file
        data (dict): Data to save
        indent (int, optional): JSON indentation
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        return True
    except (IOError, TypeError):
        return False

def extract_stock_symbol(text):
    """
    Extract stock symbol from text
    
    Args:
        text (str): Text containing a stock symbol
        
    Returns:
        str: Extracted stock symbol or None
    """
    # Common Indian stock exchange suffixes
    exchange_suffixes = ['.NS', '.BO', '.BSE', '.NSE']
    
    # Try to find symbols like "RELIANCE.NS" or "HDFCBANK.BO"
    pattern = r'\b[A-Z]+(?:&[A-Z]+)?(?:\.NS|\.BO|\.BSE|\.NSE)?\b'
    matches = re.findall(pattern, text.upper())
    
    if matches:
        # Return the first match without the exchange suffix
        for match in matches:
            for suffix in exchange_suffixes:
                if match.endswith(suffix):
                    return match[:-len(suffix)]
            return match
    
    return None

def calculate_returns(initial_value, final_value):
    """
    Calculate percentage return
    
    Args:
        initial_value (float): Initial value
        final_value (float): Final value
        
    Returns:
        float: Percentage return
    """
    if initial_value <= 0:
        return 0
    
    return ((final_value - initial_value) / initial_value) * 100

def calculate_cagr(initial_value, final_value, years):
    """
    Calculate Compound Annual Growth Rate
    
    Args:
        initial_value (float): Initial value
        final_value (float): Final value
        years (float): Number of years
        
    Returns:
        float: CAGR as a percentage
    """
    if initial_value <= 0 or years <= 0:
        return 0
    
    return (((final_value / initial_value) ** (1 / years)) - 1) * 100

def calculate_drawdown(values):
    """
    Calculate maximum drawdown from a series of values
    
    Args:
        values (list): List of values
        
    Returns:
        float: Maximum drawdown as a percentage
    """
    if not values or len(values) < 2:
        return 0
    
    max_drawdown = 0
    peak_value = values[0]
    
    for value in values:
        if value > peak_value:
            peak_value = value
        
        drawdown = (peak_value - value) / peak_value * 100
        max_drawdown = max(max_drawdown, drawdown)
    
    return max_drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate Sharpe ratio
    
    Args:
        returns (list): List of percentage returns
        risk_free_rate (float): Risk-free rate as a percentage
        
    Returns:
        float: Sharpe ratio
    """
    import numpy as np
    
    if not returns:
        return 0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    if np.std(excess_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(excess_returns)

def calculate_sortino_ratio(returns, risk_free_rate=0):
    """
    Calculate Sortino ratio
    
    Args:
        returns (list): List of percentage returns
        risk_free_rate (float): Risk-free rate as a percentage
        
    Returns:
        float: Sortino ratio
    """
    import numpy as np
    
    if not returns:
        return 0
    
    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate
    
    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0 or np.std(negative_returns) == 0:
        return 0
    
    return np.mean(excess_returns) / np.std(negative_returns)

def calculate_win_rate(trades):
    """
    Calculate win rate from a list of trades
    
    Args:
        trades (list): List of trade results (positive for profit, negative for loss)
        
    Returns:
        float: Win rate as a percentage
    """
    if not trades:
        return 0
    
    winning_trades = sum(1 for trade in trades if trade > 0)
    return (winning_trades / len(trades)) * 100

def calculate_profit_factor(trades):
    """
    Calculate profit factor from a list of trades
    
    Args:
        trades (list): List of trade results (positive for profit, negative for loss)
        
    Returns:
        float: Profit factor
    """
    if not trades:
        return 0
    
    gross_profit = sum(trade for trade in trades if trade > 0)
    gross_loss = abs(sum(trade for trade in trades if trade < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0
    
    return gross_profit / gross_loss

def calculate_expectancy(trades):
    """
    Calculate expectancy from a list of trades
    
    Args:
        trades (list): List of trade results (positive for profit, negative for loss)
        
    Returns:
        float: Expectancy
    """
    if not trades:
        return 0
    
    win_rate = calculate_win_rate(trades) / 100
    
    winning_trades = [trade for trade in trades if trade > 0]
    losing_trades = [abs(trade) for trade in trades if trade < 0]
    
    avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(losing_trades) / len(losing_trades) if losing_trades else 0
    
    if avg_loss == 0:
        return avg_win * win_rate
    
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    return (win_rate * win_loss_ratio) - (1 - win_rate)

def normalize_symbol(symbol, exchange=None):
    """
    Normalize a stock symbol to a standard format
    
    Args:
        symbol (str): Stock symbol
        exchange (str, optional): Exchange code (NSE, BSE)
        
    Returns:
        str: Normalized symbol
    """
    if not symbol:
        return None
    
    # Convert to uppercase and remove whitespace
    symbol = symbol.upper().strip()
    
    # Remove exchange suffixes if present
    symbol = re.sub(r'\.(NS|NSE|BO|BSE)$', '', symbol)
    
    # Add exchange suffix if provided
    if exchange:
        exchange = exchange.upper().strip()
        if exchange in ['NSE', 'NS']:
            return f"{symbol}.NS"
        elif exchange in ['BSE', 'BO']:
            return f"{symbol}.BO"
    
    return symbol

def extract_date_from_text(text):
    """
    Extract date from text
    
    Args:
        text (str): Text containing a date
        
    Returns:
        datetime: Extracted date or None
    """
    # Various date patterns
    patterns = [
        # DD-MM-YYYY or DD/MM/YYYY
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{4})',
        # YYYY-MM-DD
        r'(\d{4})[-/](\d{1,2})[-/](\d{1,2})',
        # Month name formats
        r'(\d{1,2})\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{4})',
        r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})[,\s]+(\d{4})'
    ]
    
    for pattern in patterns:
        matches = re.search(pattern, text, re.IGNORECASE)
        if matches:
            groups = matches.groups()
            
            # Handle different formats
            if len(groups) == 3:
                if groups[0] in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] or \
                   groups[0].capitalize() in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                    # Month name first (e.g., "Jan 15, 2023")
                    month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                    month = month_dict[groups[0].capitalize()[:3]]
                    day = int(groups[1])
                    year = int(groups[2])
                elif len(groups[0]) == 4 and groups[0].isdigit():
                    # YYYY-MM-DD
                    year = int(groups[0])
                    month = int(groups[1])
                    day = int(groups[2])
                else:
                    # DD-MM-YYYY
                    day = int(groups[0])
                    if groups[1] in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                        # DD Month YYYY
                        month_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                                     'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
                        month = month_dict[groups[1].capitalize()[:3]]
                    else:
                        # DD-MM-YYYY
                        month = int(groups[1])
                    year = int(groups[2])
                
                try:
                    return datetime(year, month, day)
                except (ValueError, TypeError):
                    continue
    
    return None

def get_instrument_type(symbol):
    """
    Guess instrument type from symbol
    
    Args:
        symbol (str): Instrument symbol
        
    Returns:
        str: Instrument type ('equity', 'futures', 'options', or 'unknown')
    """
    if not symbol:
        return 'unknown'
    
    symbol = symbol.upper()
    
    # Futures typically have month/year codes
    if re.search(r'[A-Z]+\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}FUT', symbol):
        return 'futures'
    
    # Options have strike prices and CE/PE indicators
    if re.search(r'[A-Z]+\d{2}(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\d{2}\d+(?:CE|PE)', symbol):
        return 'options'
    
    # Default to equity
    return 'equity'

def parse_time_string(time_str):
    """
    Parse a time string into datetime object
    
    Args:
        time_str (str): Time string (e.g. '09:15', '15:30')
        
    Returns:
        datetime: datetime object with today's date and parsed time
    """
    try:
        # Try HH:MM format
        hours, minutes = map(int, time_str.split(':'))
        today = datetime.now().replace(hour=hours, minute=minutes, second=0, microsecond=0)
        return today
    except (ValueError, AttributeError):
        try:
            # Try full datetime format
            return datetime.fromisoformat(time_str)
        except (ValueError, TypeError):
            return None

def get_trading_sessions():
    """
    Get pre-market, market, and post-market sessions for today
    
    Returns:
        dict: Dict with start and end times for each session
    """
    from config import settings
    
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    # Market session
    market_open = today.replace(hour=settings.MARKET_OPEN_HOUR, minute=settings.MARKET_OPEN_MINUTE)
    market_close = today.replace(hour=settings.MARKET_CLOSE_HOUR, minute=settings.MARKET_CLOSE_MINUTE)
    
    # Pre-market session (2 hours before market open)
    pre_market_start = market_open - timedelta(hours=2)
    pre_market_end = market_open
    
    # Post-market session (2 hours after market close)
    post_market_start = market_close
    post_market_end = market_close + timedelta(hours=2)
    
    return {
        'pre_market': {
            'start': pre_market_start,
            'end': pre_market_end
        },
        'market': {
            'start': market_open,
            'end': market_close
        },
        'post_market': {
            'start': post_market_start,
            'end': post_market_end
        }
    }

# New functions to support Data Orchestrator

def run_in_thread(func):
    """
    Decorator to run a function in a separate thread
    
    Args:
        func: Function to run in thread
        
    Returns:
        threading.Thread: The thread object
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread
    return wrapper

def chunk_list(items, chunk_size):
    """
    Split a list into chunks of specified size
    
    Args:
        items (list): List to split
        chunk_size (int): Size of each chunk
        
    Returns:
        list: List of chunks
    """
    if not items:
        return []
    
    return [items[i:i+chunk_size] for i in range(0, len(items), chunk_size)]

def rate_limit(max_calls=10, time_period=1):
    """
    Decorator to rate limit function calls
    
    Args:
        max_calls (int): Maximum number of calls in time period
        time_period (float): Time period in seconds
        
    Returns:
        Decorated function
    """
    calls = []
    lock = threading.Lock()
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with lock:
                # Remove old calls
                now = time.time()
                while calls and calls[0] < now - time_period:
                    calls.pop(0)
                
                # Check if we've hit the limit
                if len(calls) >= max_calls:
                    # We need to wait
                    sleep_time = calls[0] + time_period - now
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                
                # Add this call
                calls.append(time.time())
            
            # Call the function
            return func(*args, **kwargs)
        return wrapper
    return decorator

def merge_dictionaries(*dicts, mode='update'):
    """
    Merge multiple dictionaries with different merging modes
    
    Args:
        *dicts: Dictionaries to merge
        mode (str): Merge mode ('update', 'override', 'keep_first')
        
    Returns:
        dict: Merged dictionary
    """
    result = {}
    
    if mode == 'update':
        # Update keys, recursively for nested dicts
        for d in dicts:
            for key, value in d.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dictionaries(result[key], value, mode=mode)
                else:
                    result[key] = value
    
    elif mode == 'override':
       # Later dictionaries override earlier ones
       for d in dicts:
           result.update(d)
   
    elif mode == 'keep_first':
        # First dictionary keys take precedence
        for d in dicts:
            for key, value in d.items():
                if key not in result:
                    result[key] = value
    
    return result

def safe_get(obj, *keys, default=None):
    """
    Safely get a value from a nested dictionary/object

    Args:
        obj: Object or dictionary to extract from
        *keys: Keys or attributes to navigate
        default: Default value if not found
        
    Returns:
        Value or default
    """
    current = obj

    for key in keys:
        if current is None:
            return default
            
        if isinstance(current, dict):
            current = current.get(key)
        else:
            try:
                current = getattr(current, key)
            except (AttributeError, TypeError):
                return default

    return current if current is not None else default

def execute_with_timeout(func, args=(), kwargs={}, timeout=60):
    """
    Execute a function with a timeout

    Args:
        func: Function to execute
        args: Function arguments
        kwargs: Function keyword arguments
        timeout: Timeout in seconds
        
    Returns:
        Function result or None if timeout
    """
    result_container = []
    exception_container = []

    def worker():
        try:
            result = func(*args, **kwargs)
            result_container.append(result)
        except Exception as e:
            exception_container.append(e)

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            # Timeout occurred
            return None

        if exception_container:
            # Re-raise the exception
            raise exception_container[0]

        if result_container:
            return result_container[0]

        return None

def synchronize(lock_name=None):
    """
    Decorator to synchronize a function with a lock

    Args:
        lock_name: Name of the lock to use (uses function name if None)
        
    Returns:
        Decorated function
    """
    # Dictionary to store locks
    locks = {}

    def decorator(func):
        # Use function name as lock name if not provided
        nonlocal lock_name
        if lock_name is None:
            lock_name = func.__name__
        
        # Create lock if it doesn't exist
        if lock_name not in locks:
            locks[lock_name] = threading.Lock()
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with locks[lock_name]:
                return func(*args, **kwargs)
        return wrapper
    return decorator

def convert_timeframe_to_minutes(timeframe):
    """
    Convert various timeframe formats to minutes

    Args:
        timeframe (str): Timeframe string ('1min', '1h', 'day', etc.)
        
    Returns:
        int: Minutes or None if not convertible
    """
    # Map common timeframe strings
    timeframe_map = {
        "1min": 1,
        "5min": 5,
        "15min": 15,
        "30min": 30,
        "60min": 60,
        "1h": 60,
        "4h": 240,
        "day": 1440,   # 24 * 60
        "week": 10080  # 7 * 24 * 60
    }

    # Check direct mapping
    if timeframe in timeframe_map:
        return timeframe_map[timeframe]

    # Try to parse with regex
    return parse_timeframe(timeframe)