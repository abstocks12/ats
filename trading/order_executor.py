"""
Order Executor Module - Executes and manages trading orders
"""

import logging
from datetime import datetime
import uuid
from bson import ObjectId

def execute_order(symbol, exchange, order_type, quantity, price=None, stop_loss=None, target=None, db=None):
    """
    Execute a trading order
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        order_type (str): Order type ('buy' or 'sell')
        quantity (int): Order quantity
        price (float, optional): Limit price (default: None for market order)
        stop_loss (float, optional): Stop loss price
        target (float, optional): Target price
        db (MongoDBConnector, optional): Database connection
        
    Returns:
        dict: Order result
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Executing {order_type} order for {symbol} on {exchange}: {quantity} units")
    
    try:
        # Connect to Zerodha API (this would be implemented in a real system)
        from data.market.zerodha_connector import ZerodhaConnector
        zerodha = ZerodhaConnector()
        
        # Determine order parameters
        is_buy = order_type.lower() == 'buy'
        transaction_type = 'BUY' if is_buy else 'SELL'
        
        # Get current market price if not provided
        if price is None:
            market_data = _get_latest_market_data(symbol, exchange, db)
            price = market_data["close"] if market_data else None
            
            if price is None:
                logger.error(f"Cannot execute order: Unable to determine price for {symbol}")
                return {"success": False, "error": "Unable to determine price"}
        
        # Create the order parameters
        order_params = {
            "symbol": symbol,
            "exchange": exchange,
            "transaction_type": transaction_type,
            "quantity": quantity,
            "order_type": "LIMIT" if price else "MARKET",
            "price": price,
            "product": "MIS",  # Intraday (modify as needed)
            "trigger_price": None,
            "stoploss": stop_loss,
            "target": target,
            "trailing_stoploss": None,
            "validity": "DAY",
            "variety": "regular",
            "tag": "AutoTrader"
        }
        
        # Instead of actual API call, we'll simulate the order for this implementation
        order_id = f"SIMULATED_{uuid.uuid4()}"
        
        # Log the order
        logger.info(f"Order {order_id} placed successfully: {order_params}")
        
        # In a real implementation, we would do:
        # order_id = zerodha.place_order(**order_params)
        
        # Store the order in the database
        if db:
            order_record = {
                "order_id": order_id,
                "symbol": symbol,
                "exchange": exchange,
                "transaction_type": transaction_type,
                "quantity": quantity,
                "order_type": order_params["order_type"],
                "price": price,
                "trigger_price": order_params.get("trigger_price"),
                "stop_loss": stop_loss,
                "target": target,
                "status": "COMPLETE",  # Simulated as complete
                "filled_quantity": quantity,
                "filled_price": price,
                "order_timestamp": datetime.now(),
                "exchange_timestamp": datetime.now(),
                "is_simulated": True
            }
            
            db.orders_collection.insert_one(order_record)
        
        # Create a position record for the new position
        position_type = "long" if is_buy else "short"
        position_id = _create_position_record(
            symbol, exchange, position_type, quantity, price, stop_loss, target, order_id, db
        )
        
        return {
            "success": True,
            "order_id": order_id,
            "position_id": position_id,
            "filled_price": price,
            "filled_quantity": quantity
        }
        
    except Exception as e:
        logger.error(f"Error executing order: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def close_position(position, db=None, price=None, reason=None):
    """
    Close an existing position
    
    Args:
        position (dict): Position document
        db (MongoDBConnector, optional): Database connection
        price (float, optional): Exit price (default: market price)
        reason (str, optional): Exit reason
        
    Returns:
        dict: Order result
    """
    logger = logging.getLogger(__name__)
    
    symbol = position["symbol"]
    exchange = position["exchange"]
    quantity = position["quantity"]
    position_type = position["position_type"]
    
    # Determine order type (opposite of position type)
    order_type = "sell" if position_type == "long" else "buy"
    
    logger.info(f"Closing position for {symbol}: {order_type} {quantity} units")
    
    try:
        # Execute the order
        order_result = execute_order(
            symbol=symbol,
            exchange=exchange,
            order_type=order_type,
            quantity=quantity,
            price=price,
            db=db
        )
        
        if not order_result["success"]:
            logger.error(f"Failed to close position: {order_result['error']}")
            return False
        
        # Update the position record as closed
        if db:
            # Calculate P&L
            entry_price = position["entry_price"]
            exit_price = order_result["filled_price"]
            
            if position_type == "long":
                profit_loss = (exit_price - entry_price) * quantity
                profit_loss_percent = (exit_price - entry_price) / entry_price * 100
            else:  # short
                profit_loss = (entry_price - exit_price) * quantity
                profit_loss_percent = (entry_price - exit_price) / entry_price * 100
            
            # Update position
            db.positions_collection.update_one(
                {"_id": position["_id"]},
                {
                    "$set": {
                        "is_open": False,
                        "exit_price": exit_price,
                        "exit_time": datetime.now(),
                        "profit_loss": profit_loss,
                        "profit_loss_percent": profit_loss_percent,
                        "exit_reason": reason or "Manual close",
                        "exit_order_id": order_result["order_id"]
                    }
                }
            )
            
            # Update trades collection
            db.trades_collection.update_one(
                {"trade_id": position.get("trade_id")},
                {
                    "$set": {
                        "exit_price": exit_price,
                        "exit_time": datetime.now(),
                        "profit_loss": profit_loss,
                        "profit_loss_percent": profit_loss_percent,
                        "exit_reason": reason or "Manual close",
                        "status": "closed"
                    }
                }
            )
            
            logger.info(
                f"Closed position for {symbol}: {position_type} {quantity} @ {exit_price}, "
                f"P&L: {profit_loss:.2f} ({profit_loss_percent:.2f}%)"
            )
        
        return True
        
    except Exception as e:
        logger.error(f"Error closing position: {e}", exc_info=True)
        return False

def update_order(position_id, updates, db):
    """
    Update an existing order/position
    
    Args:
        position_id (ObjectId): Position ID
        updates (dict): Fields to update
        db (MongoDBConnector): Database connection
        
    Returns:
        bool: True if updated successfully, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Get the position
        position = db.positions_collection.find_one({"_id": position_id})
        
        if not position:
            logger.error(f"Cannot update order: Position {position_id} not found")
            return False
        
        # Update the position
        result = db.positions_collection.update_one(
            {"_id": position_id},
            {"$set": updates}
        )
        
        # In a real implementation, we would need to modify the order on the exchange as well
        # This would involve canceling and replacing orders, or modifying them if supported
        
        logger.info(f"Updated position {position_id} for {position['symbol']}: {updates}")
        
        return result.modified_count > 0
        
    except Exception as e:
        logger.error(f"Error updating order: {e}", exc_info=True)
        return False

def _create_position_record(symbol, exchange, position_type, quantity, price, stop_loss, target, order_id, db):
    """
    Create a position record in the database
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        position_type (str): 'long' or 'short'
        quantity (int): Position size
        price (float): Entry price
        stop_loss (float): Stop loss price
        target (float): Target price
        order_id (str): Order ID
        db (MongoDBConnector): Database connection
        
    Returns:
        ObjectId: ID of the created position
    """
    if not db:
        return None
    
    # Create a unique trade ID
    trade_id = str(uuid.uuid4())
    
    # Calculate risk amount
    initial_risk = 0
    if stop_loss:
        if position_type == "long":
            initial_risk = (price - stop_loss) * quantity
        else:  # short
            initial_risk = (stop_loss - price) * quantity
    
    # Create position document
    position = {
        "symbol": symbol,
        "exchange": exchange,
        "position_type": position_type,
        "quantity": quantity,
        "entry_price": price,
        "entry_time": datetime.now(),
        "stop_loss": stop_loss,
        "target": target,
        "is_open": True,
        "is_paper": False,
        "trade_id": trade_id,
        "entry_order_id": order_id,
        "strategy": "default",  # This would come from the strategy that generated the signal
        "notes": "Live trading position",
        "signals": [],  # This would come from the strategy
        "initial_risk_amount": initial_risk
    }
    
    # Insert position into database
    result = db.positions_collection.insert_one(position)
    position_id = result.inserted_id
    
    # Create a trade record for historical tracking
    trade = {
        "trade_id": trade_id,
        "symbol": symbol,
        "exchange": exchange,
        "position_type": position_type,
        "quantity": quantity,
        "entry_price": price,
        "entry_time": datetime.now(),
        "stop_loss": stop_loss,
        "target": target,
        "strategy": "default",
        "is_paper": False,
        "initial_risk_amount": initial_risk,
        "signals": [],
        "notes": "Live trading position",
        "status": "open"
    }
    
    db.trades_collection.insert_one(trade)
    
    return position_id

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
    if not db:
        return None
    
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