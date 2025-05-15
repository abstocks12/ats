"""
Position Sizing Module - Calculates optimal position sizes for trades
"""

import logging
from datetime import datetime

def calculate_position_size(prediction, trading_config, db):
    """
    Calculate optimal position size based on risk parameters
    
    Args:
        prediction (dict): Prediction document with entry/exit points
        trading_config (dict): Instrument trading configuration
        db (MongoDBConnector): Database connection for portfolio context
        
    Returns:
        int: Position size in number of shares/contracts
    """
    logger = logging.getLogger(__name__)
    
    # Get basic information
    symbol = prediction["symbol"]
    exchange = prediction["exchange"]
    direction = prediction["prediction"]  # "up" or "down"
    confidence = prediction.get("confidence", 0.5)
    
    # Get risk parameters from trading config
    position_size_percent = trading_config.get("position_size_percent", 5.0)
    max_risk_percent = trading_config.get("max_risk_percent", 1.0)
    
    # Additional trade-specific parameters
    stop_loss = prediction.get("stop_loss")
    target_price = prediction.get("target_price")
    expected_change = prediction.get("expected_change_percent", 0)
    
    # Get current price
    current_data = _get_latest_market_data(symbol, exchange, db)
    if not current_data:
        logger.error(f"No market data available for {symbol}")
        return 0
    
    current_price = current_data["close"]
    
    # Determine entry price (could be current price or limit price)
    # For simplicity, we'll use current price as entry price
    entry_price = current_price
    
    # Get portfolio value and cash available
    portfolio_info = _get_portfolio_info(db)
    portfolio_value = portfolio_info["total_value"]
    cash_available = portfolio_info["cash_available"]
    
    # Calculate position value based on position size percentage
    max_position_value = portfolio_value * (position_size_percent / 100)
    
    # Scale position size based on confidence
    confidence_scale = min(confidence / 0.7, 1.0)  # Scale 0.7+ confidence to 1.0
    adjusted_position_value = max_position_value * confidence_scale
    
    # Calculate stop-loss distance
    if stop_loss:
        if direction == "up":
            stop_distance = entry_price - stop_loss
        else:  # "down"
            stop_distance = stop_loss - entry_price
    else:
        # If no stop-loss provided, use a default percentage
        stop_distance = entry_price * 0.02  # Default 2% stop
    
    # Make sure we have a positive stop distance
    stop_distance = max(stop_distance, 0.01 * entry_price)  # At least 1% of price
    
    # Calculate risk amount based on max_risk_percent
    max_risk_amount = portfolio_value * (max_risk_percent / 100)
    
    # Calculate risk-based position size
    if stop_distance > 0:
        risk_based_size = max_risk_amount / stop_distance
    else:
        risk_based_size = 0
        
    # Calculate quantity based on risk and position size percentages
    max_quantity_by_position_size = adjusted_position_value / entry_price
    max_quantity_by_risk = risk_based_size
    
    # Use the more conservative approach
    quantity = min(max_quantity_by_position_size, max_quantity_by_risk)
    
    # Check if we have enough cash
    max_quantity_by_cash = cash_available / entry_price
    quantity = min(quantity, max_quantity_by_cash)
    
    # Round down to nearest whole number for stocks
    quantity = int(quantity)
    
    # Adjust for minimum quantities and lot sizes (example for Indian markets)
    min_quantity = 1
    if quantity < min_quantity:
        logger.warning(f"Calculated quantity {quantity} is below minimum {min_quantity} for {symbol}")
        return 0
    
    # Log the position sizing details
    logger.info(
        f"Position sizing for {symbol}: calculated {quantity} shares "
        f"(portfolio value: {portfolio_value:.2f}, max risk: {max_risk_amount:.2f}, "
        f"entry: {entry_price:.2f}, stop: {stop_loss if stop_loss else 'N/A'}, confidence: {confidence:.2f})"
    )
    
    return quantity

def calculate_exit_quantity(position, reason, db):
    """
    Calculate how many shares/contracts to exit
    
    Args:
        position (dict): Current position document
        reason (str): Exit reason (e.g., 'stop_loss', 'target', 'signal_reversal', 'time_based')
        db (MongoDBConnector): Database connection for context
        
    Returns:
        int: Quantity to exit
    """
    logger = logging.getLogger(__name__)
    
    # Get basic position information
    symbol = position["symbol"]
    quantity = position["quantity"]
    
    # Different exit rules based on reason
    if reason == "stop_loss":
        # Exit entire position on stop loss
        exit_quantity = quantity
    elif reason == "target":
        # Partial profit-taking at target (e.g., exit half the position)
        exit_quantity = quantity // 2
    elif reason == "signal_reversal":
        # Exit entire position on signal reversal
        exit_quantity = quantity
    elif reason == "time_based":
        # Exit portion of position after certain time period
        exit_quantity = quantity // 3
    else:
        # Default to full exit for other reasons
        exit_quantity = quantity
    
    logger.info(f"Calculated exit quantity for {symbol}: {exit_quantity} shares (reason: {reason})")
    
    return exit_quantity

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

def _get_portfolio_info(db):
    """
    Get current portfolio information
    
    Args:
        db (MongoDBConnector): Database connection
        
    Returns:
        dict: Portfolio information
    """
    # This would typically be calculated based on current positions and cash
    # For simplicity, we're using hardcoded values or could use a system_status collection
    
    # If system_status collection exists, use it
    system_status = db.system_status_collection.find_one({"type": "portfolio_status"})
    
    if system_status and "portfolio_value" in system_status and "cash_available" in system_status:
        return {
            "total_value": system_status["portfolio_value"],
            "cash_available": system_status["cash_available"]
        }
    
    # Otherwise calculate from open positions
    positions = list(db.positions_collection.find({"is_open": True}))
    
    # Calculate position values
    position_values = []
    for position in positions:
        symbol = position["symbol"]
        exchange = position["exchange"]
        quantity = position["quantity"]
        
        # Get current price
        market_data = _get_latest_market_data(symbol, exchange, db)
        if market_data:
            current_price = market_data["close"]
            position_value = quantity * current_price
            position_values.append(position_value)
    
    positions_total = sum(position_values)
    
    # Assume default values if we don't have system status
    default_cash = 1000000  # Default 10 lakh initial capital
    default_portfolio_value = default_cash  # Initial value
    
    # Use defaults if no better information available
    if system_status:
        cash_available = system_status.get("cash_available", default_cash)
    else:
        cash_available = default_cash - positions_total
    
    total_value = positions_total + cash_available
    
    return {
        "total_value": total_value,
        "cash_available": cash_available
    }