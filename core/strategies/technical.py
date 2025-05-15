"""
Technical Trading Strategies Module
"""

import logging
import numpy as np
from datetime import datetime, timedelta

def get_entry_parameters(prediction, db):
    """
    Get entry parameters for a trade based on prediction and technical analysis
    
    Args:
        prediction (dict): Prediction document
        db (MongoDBConnector): Database connection
        
    Returns:
        dict: Entry parameters
    """
    logger = logging.getLogger(__name__)
    
    symbol = prediction["symbol"]
    exchange = prediction["exchange"]
    direction = prediction["prediction"]  # "up" or "down"
    
    # Get latest market data
    market_data = _get_latest_market_data(symbol, exchange, db)
    
    if not market_data:
        logger.error(f"No market data available for {symbol}")
        return {}
    
    # Get the instrument configuration
    instrument = db.portfolio_collection.find_one({
        "symbol": symbol,
        "exchange": exchange
    })
    
    if not instrument:
        logger.error(f"Instrument {symbol} not found in portfolio")
        return {}
    
    # Get recent price data
    price_data = _get_recent_price_data(symbol, exchange, "15min", 20, db)
    
    if not price_data or len(price_data) < 5:
        logger.error(f"Insufficient price data for {symbol}")
        return {}
    
    # Current price
    current_price = market_data["close"]
    
    # Calculate ATR (Average True Range) for stop loss placement
    atr = _calculate_atr(price_data, 14)
    
    # Default risk percentage from instrument config
    risk_percent = instrument.get("trading_config", {}).get("max_risk_percent", 1.0)
    
    # Calculate stop loss and target based on direction and ATR
    if direction == "up":
        # For long positions
        stop_loss = current_price - (atr * 2)  # 2x ATR for stop loss
        target = current_price + (atr * 3)     # 3x ATR for target (1.5:1 reward-to-risk)
    else:
        # For short positions
        stop_loss = current_price + (atr * 2)  # 2x ATR for stop loss
        target = current_price - (atr * 3)     # 3x ATR for target (1.5:1 reward-to-risk)
    
    # Round to 2 decimal places
    stop_loss = round(stop_loss, 2)
    target = round(target, 2)
    
    # Get additional entry signals
    signals = _get_entry_signals(symbol, exchange, direction, price_data, db)
    
    # Entry method based on confidence
    confidence = prediction.get("confidence", 0.5)
    if confidence > 0.8:
        # High confidence - market order
        limit_price = None
        entry_notes = "Market order due to high confidence prediction"
    elif confidence > 0.65:
        # Medium confidence - limit order at current price
        limit_price = current_price
        entry_notes = "Limit order at current price due to medium confidence"
    else:
        # Lower confidence - limit order with better price
        if direction == "up":
            # For long positions, try to get a slightly lower entry
            limit_price = current_price * 0.995  # 0.5% below current price
        else:
            # For short positions, try to get a slightly higher entry
            limit_price = current_price * 1.005  # 0.5% above current price
        entry_notes = "Limit order with price improvement due to lower confidence"
    
    # Round limit price to 2 decimal places if it exists
    if limit_price:
        limit_price = round(limit_price, 2)
    
    # Put together the entry parameters
    entry_params = {
        "limit_price": limit_price,
        "stop_loss": stop_loss,
        "target": target,
        "risk_percent": risk_percent,
        "atr": atr,
        "signals": signals,
        "strategy": "technical_prediction",
        "notes": entry_notes,
        "trailing_stop_enabled": True,
        "trailing_stop_percent": 1.0  # 1% trailing stop
    }
    
    logger.info(f"Entry parameters for {symbol} {direction}: {entry_params}")
    
    return entry_params

def analyze_technical_signals(symbol, exchange, db, timeframe="15min"):
    """
    Analyze technical signals for an instrument
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        db (MongoDBConnector): Database connection
        timeframe (str, optional): Data timeframe (default: 15min)
        
    Returns:
        dict: Technical analysis signals
    """
    logger = logging.getLogger(__name__)
    
    # Get recent price data
    price_data = _get_recent_price_data(symbol, exchange, timeframe, 100, db)
    
    if not price_data or len(price_data) < 50:
        logger.error(f"Insufficient price data for {symbol}")
        return {"error": "Insufficient price data"}
    
    # Extract price arrays
    closes = np.array([candle["close"] for candle in price_data])
    highs = np.array([candle["high"] for candle in price_data])
    lows = np.array([candle["low"] for candle in price_data])
    volumes = np.array([candle["volume"] for candle in price_data])
    
    # Calculate indicators
    sma20 = _calculate_sma(closes, 20)
    sma50 = _calculate_sma(closes, 50)
    rsi14 = _calculate_rsi(closes, 14)
    macd, macd_signal = _calculate_macd(closes)
    
    # Current values
    current_close = closes[-1]
    current_sma20 = sma20[-1]
    current_sma50 = sma50[-1]
    current_rsi = rsi14[-1]
    current_macd = macd[-1]
    current_macd_signal = macd_signal[-1]
    
    # Determine trend
    trend = "neutral"
    if current_close > current_sma20 and current_sma20 > current_sma50:
        trend = "uptrend"
    elif current_close < current_sma20 and current_sma20 < current_sma50:
        trend = "downtrend"
    
    # Check for SMA crossover
    sma_crossover = "none"
    if sma20[-2] < sma50[-2] and sma20[-1] > sma50[-1]:
        sma_crossover = "bullish"
    elif sma20[-2] > sma50[-2] and sma20[-1] < sma50[-1]:
        sma_crossover = "bearish"
    
    # Check for MACD crossover
    macd_crossover = "none"
    if macd[-2] < macd_signal[-2] and macd[-1] > macd_signal[-1]:
        macd_crossover = "bullish"
    elif macd[-2] > macd_signal[-2] and macd[-1] < macd_signal[-1]:
        macd_crossover = "bearish"
    
    # Check for RSI conditions
    rsi_condition = "neutral"
    if current_rsi < 30:
        rsi_condition = "oversold"
    elif current_rsi > 70:
        rsi_condition = "overbought"
    
    # Check for volume surge
    avg_volume = np.mean(volumes[-10:-1])  # Average of last 9 volumes (excluding current)
    volume_surge = volumes[-1] > (avg_volume * 1.5)  # 50% above average
    
    # Calculate ATR
    atr = _calculate_atr(price_data, 14)
    
    # Check for support/resistance levels
    support, resistance = _find_support_resistance(price_data, 10)
    
    # Determine signal strength
    signal_strength = 0
    
    # Uptrend signals
    if trend == "uptrend":
        signal_strength += 1
    if sma_crossover == "bullish":
        signal_strength += 2
    if macd_crossover == "bullish":
        signal_strength += 2
    if rsi_condition == "oversold":
        signal_strength += 1
    if volume_surge and current_close > closes[-2]:
        signal_strength += 1
    if current_close > resistance:
        signal_strength += 2  # Breakout
    
    # Downtrend signals
    if trend == "downtrend":
        signal_strength -= 1
    if sma_crossover == "bearish":
        signal_strength -= 2
    if macd_crossover == "bearish":
        signal_strength -= 2
    if rsi_condition == "overbought":
        signal_strength -= 1
    if volume_surge and current_close < closes[-2]:
        signal_strength -= 1
    if current_close < support:
        signal_strength -= 2  # Breakdown
    
    # Normalize signal strength to range [-1, 1]
    signal_strength = max(min(signal_strength / 5, 1), -1)
    
    # Determine direction based on signal strength
    direction = "neutral"
    if signal_strength > 0.2:
        direction = "up"
    elif signal_strength < -0.2:
        direction = "down"
    
    # Determine confidence based on signal strength
    confidence = abs(signal_strength)
    
    # Compile results
    results = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe,
        "timestamp": datetime.now(),
        "indicators": {
            "sma20": current_sma20,
            "sma50": current_sma50,
            "rsi14": current_rsi,
            "macd": current_macd,
            "macd_signal": current_macd_signal,
            "atr": atr
        },
        "signals": {
            "trend": trend,
            "sma_crossover": sma_crossover,
            "macd_crossover": macd_crossover,
            "rsi_condition": rsi_condition,
            "volume_surge": volume_surge
        },
        "levels": {
            "support": support,
            "resistance": resistance
        },
        "analysis": {
            "direction": direction,
            "signal_strength": signal_strength,
            "confidence": confidence
        }
    }
    
    logger.info(f"Technical analysis for {symbol}: {direction} (confidence: {confidence:.2f})")
    
    return results

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

def _calculate_sma(prices, period):
    """
    Calculate Simple Moving Average
    
    Args:
        prices (ndarray): Array of price data
        period (int): SMA period
        
    Returns:
        ndarray: SMA values
    """
    sma = np.zeros_like(prices)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    return sma

def _calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index
    
    Args:
        prices (ndarray): Array of price data
        period (int): RSI period
        
    Returns:
        ndarray: RSI values
    """
    # Calculate price changes
    deltas = np.diff(prices)
    deltas = np.append(0, deltas)  # Add 0 for the first element
    
    # Get gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # First value
    avg_gain[period] = np.mean(gains[1:period+1])
    avg_loss[period] = np.mean(losses[1:period+1])
    
    # Rest of the values
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + losses[i]) / period
    
    # Calculate RS and RSI
    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def _calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD and Signal Line
    
    Args:
        prices (ndarray): Array of price data
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period
        
    Returns:
        tuple: (MACD, Signal Line)
    """
    # Calculate EMAs
    ema_fast = _calculate_ema(prices, fast)
    ema_slow = _calculate_ema(prices, slow)
    
    # Calculate MACD
    macd = ema_fast - ema_slow
    
    # Calculate Signal Line
    signal_line = _calculate_ema(macd, signal)
    
    return macd, signal_line

def _calculate_ema(prices, period):
    """
    Calculate Exponential Moving Average
    
    Args:
        prices (ndarray): Array of price data
        period (int): EMA period
        
    Returns:
        ndarray: EMA values
    """
    ema = np.zeros_like(prices)
    ema[period - 1] = np.mean(prices[:period])
    
    multiplier = 2 / (period + 1)
    
    for i in range(period, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema

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

def _find_support_resistance(price_data, lookback=10):
    """
    Find support and resistance levels
    
    Args:
        price_data (list): List of price data dictionaries
        lookback (int): Number of periods to look back
        
    Returns:
        tuple: (support, resistance)
    """
    if len(price_data) < lookback:
        return None, None
    
    # Extract highs and lows for the period
    recent_data = price_data[-lookback:]
    highs = [candle["high"] for candle in recent_data]
    lows = [candle["low"] for candle in recent_data]
    
    # Simple approach: use min/max as support/resistance
    support = min(lows)
    resistance = max(highs)
    
    return support, resistance

def _get_entry_signals(symbol, exchange, direction, price_data, db):
    """
    Get entry signals for a trade
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        direction (str): Trade direction ('up' or 'down')
        price_data (list): List of price data dictionaries
        db (MongoDBConnector): Database connection
        
    Returns:
        list: Entry signals
    """
    signals = []
    
    # Get the latest data
    latest = price_data[-1]
    prev = price_data[-2] if len(price_data) > 1 else None
    
    # Extract close prices
    closes = np.array([candle["close"] for candle in price_data])
    
    # Calculate some indicators
    sma20 = _calculate_sma(closes, 20)[-1] if len(closes) >= 20 else None
    rsi14 = _calculate_rsi(closes, 14)[-1] if len(closes) >= 14 else None
    
    # Check for price action signals
    if direction == "up" and prev:
        # Bullish signals
        if latest["close"] > prev["close"] and latest["volume"] > prev["volume"]:
            signals.append({"name": "volume_confirmation", "time": datetime.now()})
        
        if latest["close"] > latest["open"] and prev["close"] < prev["open"]:
            signals.append({"name": "bullish_reversal", "time": datetime.now()})
        
        if sma20 and latest["close"] > sma20:
            signals.append({"name": "above_sma20", "time": datetime.now()})
        
        if rsi14 and rsi14 < 30:
            signals.append({"name": "oversold_rsi", "time": datetime.now()})
    
    elif direction == "down" and prev:
        # Bearish signals
        if latest["close"] < prev["close"] and latest["volume"] > prev["volume"]:
            signals.append({"name": "volume_confirmation", "time": datetime.now()})
        
        if latest["close"] < latest["open"] and prev["close"] > prev["open"]:
            signals.append({"name": "bearish_reversal", "time": datetime.now()})
        
        if sma20 and latest["close"] < sma20:
            signals.append({"name": "below_sma20", "time": datetime.now()})
        
        if rsi14 and rsi14 > 70:
            signals.append({"name": "overbought_rsi", "time": datetime.now()})
    
    return signals