"""
Stop Management Module - Manages stop loss placement and updates
"""

import logging
import numpy as np
from datetime import datetime

def update_stop_loss(position, prediction, db):
    """
    Update stop loss for an existing position
    
    Args:
        position (dict): Position document
        prediction (dict): New prediction document
        db (MongoDBConnector): Database connection
        
    Returns:
        float: New stop loss price or None if no update needed
    """
    logger = logging.getLogger(__name__)
    
    # Get basic information
    symbol = position["symbol"]
    exchange = position["exchange"]
    position_type = position["position_type"]
    entry_price = position["entry_price"]
    current_stop = position.get("stop_loss")
    
    # Get current market data
    market_data = _get_latest_market_data(symbol, exchange, db)
    if not market_data:
        logger.error(f"No market data available for {symbol}")
        return None
    
    current_price = market_data["close"]
    
    # Check if we need to trail the stop loss
    if "trailing_stop_enabled" in position and position["trailing_stop_enabled"]:
        new_stop = _calculate_trailing_stop(
            position_type, 
            current_price, 
            current_stop,
            position.get("trailing_stop_percent", 1.0),
            position.get("trailing_stop_amount", 0)
        )
        
        if new_stop and ((position_type == "long" and new_stop > current_stop) or 
                         (position_type == "short" and new_stop < current_stop)):
            logger.info(f"Trailing stop for {symbol} updated: {current_stop} -> {new_stop}")
            return new_stop
    
    # If no trailing stop or no update needed from trailing, check if we should use prediction-based stop
    if "stop_loss" in prediction:
        prediction_stop = prediction["stop_loss"]
        
        # Only use prediction stop if it's better than current stop
        if current_stop is None or (
            (position_type == "long" and prediction_stop > current_stop) or
            (position_type == "short" and prediction_stop < current_stop)
        ):
            logger.info(f"Using prediction-based stop for {symbol}: {prediction_stop}")
            return prediction_stop
    
    # Check if we need to move to breakeven
    breakeven_stop = _check_breakeven_stop(position, current_price)
    if breakeven_stop:
        logger.info(f"Moving stop to breakeven for {symbol}: {breakeven_stop}")
        return breakeven_stop
    
    # No update needed
    return None

def calculate_initial_stop(symbol, exchange, entry_price, direction, db, atr_multiple=2.0):
    """
    Calculate initial stop loss for a new position
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        entry_price (float): Entry price
        direction (str): Trade direction ('long' or 'short')
        db (MongoDBConnector): Database connection
        atr_multiple (float): Multiple of ATR to use for stop distance
        
    Returns:
        float: Stop loss price
    """
    logger = logging.getLogger(__name__)
    
    # Get recent price data for ATR calculation
    price_data = _get_recent_price_data(symbol, exchange, "15min", 20, db)
    
    if not price_data or len(price_data) < 14:  # Need at least 14 periods for ATR
        logger.warning(f"Insufficient price data for {symbol}, using percentage-based stop")
        # Fallback to percentage-based stop
        stop_percent = 0.02  # 2% default stop
        if direction == "long":
            return entry_price * (1 - stop_percent)
        else:  # short
            return entry_price * (1 + stop_percent)
    
    # Calculate ATR
    atr = _calculate_atr(price_data, 14)
    
    # Calculate stop based on ATR
    if direction == "long":
        stop_loss = entry_price - (atr * atr_multiple)
    else:  # short
        stop_loss = entry_price + (atr * atr_multiple)
    
    logger.info(f"Calculated initial stop for {symbol} {direction}: {stop_loss} (ATR: {atr})")
    
    return round(stop_loss, 2)  # Round to 2 decimal places

def _calculate_trailing_stop(position_type, current_price, current_stop, trail_percent=1.0, trail_amount=0):
    """
    Calculate trailing stop loss
    
    Args:
        position_type (str): Position type ('long' or 'short')
        current_price (float): Current market price
        current_stop (float): Current stop loss price
        trail_percent (float): Trailing stop percentage
        trail_amount (float): Trailing stop fixed amount
        
    Returns:
        float: New stop loss price or None if no update needed
    """
    if not current_stop:
        return None
    
    # Calculate new stop based on position type
    if position_type == "long":
        # For long positions, we move the stop up as price increases
        if trail_percent > 0:
            new_stop = current_price * (1 - trail_percent / 100)
        elif trail_amount > 0:
            new_stop = current_price - trail_amount
        else:
            return None
        
        # Only update if new stop is higher than current stop
        if new_stop > current_stop:
            return round(new_stop, 2)
    else:  # short
        # For short positions, we move the stop down as price decreases
        if trail_percent > 0:
            new_stop = current_price * (1 + trail_percent / 100)
        elif trail_amount > 0:
            new_stop = current_price + trail_amount
        else:
            return None
        
        # Only update if new stop is lower than current stop
        if new_stop < current_stop:
            return round(new_stop, 2)
    
    return None

def _check_breakeven_stop(position, current_price):
    """
    Check if we should move the stop to breakeven
    
    Args:
        position (dict): Position document
        current_price (float): Current market price
        
    Returns:
        float: Breakeven stop price or None if not applicable
    """
    # Get position details
    position_type = position["position_type"]
    entry_price = position["entry_price"]
    current_stop = position.get("stop_loss")
    
    # Only move to breakeven if we don't have a stop or it's worse than breakeven
    if not current_stop or (
        (position_type == "long" and current_stop < entry_price) or
        (position_type == "short" and current_stop > entry_price)
    ):
        # Check if we have enough profit to move to breakeven
        if position_type == "long":
            profit_percent = (current_price - entry_price) / entry_price * 100
            # If we have at least 1% profit, consider moving to breakeven
            if profit_percent >= 1.0:
                # Add a small buffer (0.1%) above entry price
                return round(entry_price * 1.001, 2)
        else:  # short
            profit_percent = (entry_price - current_price) / entry_price * 100
            # If we have at least 1% profit, consider moving to breakeven
            if profit_percent >= 1.0:
                # Add a small buffer (0.1%) below entry price
                return round(entry_price * 0.999, 2)
    
    return None

def _get_latest_market_data(symbol, exchange, db):
    """
    Get latest market data for an instrument
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        db (MongoDBConnector): Database connection
        
    Returns:
        dict: Market data document or None if not available
    """
    # Try 1-minute timeframe first
    data = db.market_data_collection.find_one(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": "1min"
        },
        sort=[("timestamp", -1)]
    )
    
    if not data:
        # Try other timeframes
        for timeframe in ["5min", "15min", "60min", "day"]:
            data = db.market_data_collection.find_one(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                },
                sort=[("timestamp", -1)]
            )
            
            if data:
                break
    
    return data

def _get_recent_price_data(symbol, exchange, timeframe, limit, db):
    """
    Get recent price data for an instrument
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        timeframe (str): Data timeframe
        limit (int): Number of data points to retrieve
        db (MongoDBConnector): Database connection
        
    Returns:
        list: List of price data documents
    """
    data = list(db.market_data_collection.find(
        {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe
        },
        sort=[("timestamp", -1)],
        limit=limit
    ))
    
    # Reverse to get chronological order
    data.reverse()
    
    return data

def _calculate_atr(price_data, period=14):
    """
    Calculate Average True Range
    
    Args:
        price_data (list): List of price data dictionaries
        period (int): ATR period
        
    Returns:
        float: ATR value
    """
    if len(price_data) < period + 1:
        return 0
    
    # Extract high, low, close
    highs = np.array([candle["high"] for candle in price_data])
    lows = np.array([candle["low"] for candle in price_data])
    closes = np.array([candle["close"] for candle in price_data])
    
    # Calculate true ranges
    tr = np.zeros(len(price_data))
    for i in range(1, len(price_data)):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i-1])
        low_close = abs(lows[i] - closes[i-1])
        tr[i] = max(high_low, high_close, low_close)
    
    # Calculate ATR
    atr = np.mean(tr[-period:])
    
    return atr