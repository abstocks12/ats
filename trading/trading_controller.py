"""
Position Manager Module - Manages trading positions and tracking
"""

import logging
from datetime import datetime
import uuid
from bson import ObjectId

class PositionManager:
    """
    Manages trading positions, including opening, updating, and closing positions.
    Handles both live and paper trading modes.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the position manager
        
        Args:
            db_connector (MongoDBConnector): Database connection
        """
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        self.query_optimizer = db_connector.get_query_optimizer()
    
    def _get_latest_predictions(self, instruments):
        """Get latest predictions using optimized queries."""
        predictions = []
        
        for instrument in instruments:
            try:
                # Use optimized query instead of direct database access
                symbol = instrument["symbol"]
                
                result = self.query_optimizer.get_prediction_accuracy(
                    days=1,  # Just today's predictions
                    by_symbol=True
                )
                
                if result["status"] == "success":
                    # Extract this symbol's prediction
                    symbol_predictions = [
                        p for p in result["accuracy_stats"] 
                        if p.get("symbol") == symbol
                    ]
                    
                    if symbol_predictions:
                        predictions.extend(symbol_predictions)
                
            except Exception as e:
                self.logger.error(f"Error getting prediction for {instrument['symbol']}: {e}")
        
        return predictions
    
    def get_position(self, symbol, exchange):
        """
        Get current position for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            dict: Position document or None if no position exists
        """
        return self.db.positions_collection.find_one({
            "symbol": symbol,
            "exchange": exchange,
            "is_open": True
        })
    
    def get_all_positions(self, include_closed=False):
        """
        Get all positions
        
        Args:
            include_closed (bool): Whether to include closed positions
            
        Returns:
            list: List of position documents
        """
        query = {}
        if not include_closed:
            query["is_open"] = True
            
        return list(self.db.positions_collection.find(query).sort("entry_time", -1))
    
    def add_paper_position(self, symbol, exchange, position_type, quantity, entry_params):
        """
        Add a new paper trading position
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            position_type (str): 'buy' or 'sell'
            quantity (int): Position size
            entry_params (dict): Entry parameters
            
        Returns:
            ObjectId: ID of the created position
        """
        # Get current market data
        current_data = self._get_latest_market_data(symbol, exchange)
        
        if not current_data:
            self.logger.error(f"Cannot add paper position for {symbol}: No market data available")
            return None
        
        # Set entry price (use limit price if provided, otherwise use current price)
        entry_price = entry_params.get("limit_price", current_data["close"])
        
        # Create position document
        position = {
            "symbol": symbol,
            "exchange": exchange,
            "position_type": "long" if position_type == "buy" else "short",
            "quantity": quantity,
            "entry_price": entry_price,
            "entry_time": datetime.now(),
            "stop_loss": entry_params.get("stop_loss"),
            "target": entry_params.get("target"),
            "is_open": True,
            "is_paper": True,
            "trade_id": str(uuid.uuid4()),
            "strategy": entry_params.get("strategy", "default"),
            "notes": entry_params.get("notes", "Paper trading position"),
            "signals": entry_params.get("signals", []),
            "initial_risk_amount": self._calculate_risk_amount(
                entry_price, 
                entry_params.get("stop_loss"), 
                quantity, 
                position_type
            )
        }
        
        # Insert position into database
        result = self.db.positions_collection.insert_one(position)
        position_id = result.inserted_id
        
        self.logger.info(f"Added paper position for {symbol}: {position_type} {quantity} @ {entry_price}")
        
        # Log the trade in the trades collection for historical record
        self._log_trade_entry(position)
        
        return position_id
    
    def update_paper_position(self, symbol, exchange, updates):
        """
        Update a paper trading position
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            updates (dict): Fields to update
            
        Returns:
            bool: True if updated successfully, False otherwise
        """
        # Get the position
        position = self.get_position(symbol, exchange)
        
        if not position or not position.get("is_paper", False):
            self.logger.error(f"Cannot update paper position for {symbol}: No open paper position found")
            return False
        
        # Update the position
        result = self.db.positions_collection.update_one(
            {"_id": position["_id"]},
            {"$set": updates}
        )
        
        self.logger.info(f"Updated paper position for {symbol}: {updates}")
        
        return result.modified_count > 0
    
    def close_paper_position(self, symbol, exchange, exit_price=None, exit_reason=None):
        """
        Close a paper trading position
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            exit_price (float, optional): Exit price (default: current market price)
            exit_reason (str, optional): Reason for exiting the position
            
        Returns:
            dict: Closed position with P&L information or None if no position exists
        """
        # Get the position
        position = self.get_position(symbol, exchange)
        
        if not position or not position.get("is_paper", False):
            self.logger.error(f"Cannot close paper position for {symbol}: No open paper position found")
            return None
        
        # Get current market data if exit price not provided
        if exit_price is None:
            current_data = self._get_latest_market_data(symbol, exchange)
            if not current_data:
                self.logger.error(f"Cannot close paper position for {symbol}: No market data available")
                return None
            exit_price = current_data["close"]
        
        # Calculate P&L
        entry_price = position["entry_price"]
        quantity = position["quantity"]
        position_type = position["position_type"]
        
        if position_type == "long":
            profit_loss = (exit_price - entry_price) * quantity
            profit_loss_percent = (exit_price - entry_price) / entry_price * 100
        else:  # short
            profit_loss = (entry_price - exit_price) * quantity
            profit_loss_percent = (entry_price - exit_price) / entry_price * 100
        
        # Update position
        updates = {
            "is_open": False,
            "exit_price": exit_price,
            "exit_time": datetime.now(),
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "exit_reason": exit_reason or "Manual close"
        }
        
        result = self.db.positions_collection.update_one(
            {"_id": position["_id"]},
            {"$set": updates}
        )
        
        if result.modified_count > 0:
            # Get the updated position
            closed_position = self.db.positions_collection.find_one({"_id": position["_id"]})
            
            # Log the trade exit
            self._log_trade_exit(closed_position)
            
            self.logger.info(
                f"Closed paper position for {symbol}: {position_type} {quantity} @ {exit_price}, "
                f"P&L: {profit_loss:.2f} ({profit_loss_percent:.2f}%)"
            )
            
            return closed_position
        
        return None
    
    def close_all_positions(self):
        """
        Close all open positions
        
        Returns:
            int: Number of positions closed
        """
        # Get all open positions
        open_positions = self.get_all_positions(include_closed=False)
        
        closed_count = 0
        for position in open_positions:
            try:
                symbol = position["symbol"]
                exchange = position["exchange"]
                
                if position.get("is_paper", False):
                    # Paper trading position
                    if self.close_paper_position(symbol, exchange, exit_reason="System shutdown"):
                        closed_count += 1
                else:
                    # Live trading position
                    from trading.order_executor import close_position
                    if close_position(position, db=self.db, reason="System shutdown"):
                        closed_count += 1
            except Exception as e:
                self.logger.error(f"Error closing position for {position['symbol']}: {e}")
        
        return closed_count
    
    def update_all_positions(self):
        """
        Update all open positions (check stops, update trailing stops, etc.)
        
        Returns:
            int: Number of positions updated
        """
        # Get all open positions
        open_positions = self.get_all_positions(include_closed=False)
        
        updated_count = 0
        for position in open_positions:
            try:
                # Get current market data
                symbol = position["symbol"]
                exchange = position["exchange"]
                current_data = self._get_latest_market_data(symbol, exchange)
                
                if not current_data:
                    continue
                
                # Check if stop loss hit
                current_price = current_data["close"]
                stop_loss = position.get("stop_loss")
                target = position.get("target")
                
                # Check stop loss
                if stop_loss and self._is_stop_loss_hit(position, current_price):
                    if position.get("is_paper", False):
                        self.close_paper_position(symbol, exchange, exit_price=stop_loss, exit_reason="Stop loss")
                    else:
                        from trading.order_executor import close_position
                        close_position(position, db=self.db, price=stop_loss, reason="Stop loss")
                    updated_count += 1
                    continue
                
                # Check target
                if target and self._is_target_hit(position, current_price):
                    if position.get("is_paper", False):
                        self.close_paper_position(symbol, exchange, exit_price=target, exit_reason="Target reached")
                    else:
                        from trading.order_executor import close_position
                        close_position(position, db=self.db, price=target, reason="Target reached")
                    updated_count += 1
                    continue
                
                # Update trailing stop if applicable
                if position.get("trailing_stop_enabled", False):
                    new_stop = self._update_trailing_stop(position, current_price)
                    if new_stop and new_stop != position.get("stop_loss"):
                        updates = {"stop_loss": new_stop}
                        if position.get("is_paper", False):
                            self.update_paper_position(symbol, exchange, updates)
                        else:
                            from trading.order_executor import update_order
                            update_order(position["_id"], updates, self.db)
                        updated_count += 1
            
            except Exception as e:
                self.logger.error(f"Error updating position for {position['symbol']}: {e}")
        
        return updated_count
    
    def _get_latest_market_data(self, symbol, exchange):
        """
        Get latest market data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            dict: Market data document or None if not available
        """
        # Try 1-minute timeframe first
        data = self.db.market_data_collection.find_one(
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
                data = self.db.market_data_collection.find_one(
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
    
    def _calculate_risk_amount(self, entry_price, stop_loss, quantity, position_type):
        """
        Calculate the risk amount for a position
        
        Args:
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            quantity (int): Position size
            position_type (str): 'buy' or 'sell'
            
        Returns:
            float: Risk amount in currency
        """
        if not stop_loss:
            return 0
        
        if position_type == "buy":
            return (entry_price - stop_loss) * quantity
        else:  # sell/short
            return (stop_loss - entry_price) * quantity
    
    def _is_stop_loss_hit(self, position, current_price):
        """
        Check if stop loss is hit
        
        Args:
            position (dict): Position document
            current_price (float): Current market price
            
        Returns:
            bool: True if stop loss is hit, False otherwise
        """
        stop_loss = position.get("stop_loss")
        if not stop_loss:
            return False
        
        if position["position_type"] == "long":
            return current_price <= stop_loss
        else:  # short
            return current_price >= stop_loss
    
    def _is_target_hit(self, position, current_price):
        """
        Check if target is hit
        
        Args:
            position (dict): Position document
            current_price (float): Current market price
            
        Returns:
            bool: True if target is hit, False otherwise
        """
        target = position.get("target")
        if not target:
            return False
        
        if position["position_type"] == "long":
            return current_price >= target
        else:  # short
            return current_price <= target
    
    def _update_trailing_stop(self, position, current_price):
        """
        Update trailing stop based on current price
        
        Args:
            position (dict): Position document
            current_price (float): Current market price
            
        Returns:
            float: New stop loss price or None if no update needed
        """
        # Check if trailing stop is enabled
        if not position.get("trailing_stop_enabled", False):
            return None
        
        stop_loss = position.get("stop_loss")
        if not stop_loss:
            return None
        
        # Get trailing stop parameters
        trail_percent = position.get("trailing_stop_percent", 0)
        trail_amount = position.get("trailing_stop_amount", 0)
        
        if trail_percent <= 0 and trail_amount <= 0:
            return None
        
        # Calculate new stop based on position type
        if position["position_type"] == "long":
            # For long positions, we move the stop up as price increases
            if trail_percent > 0:
                new_stop = current_price * (1 - trail_percent / 100)
            else:
                new_stop = current_price - trail_amount
            
            # Only update if new stop is higher than current stop
            if new_stop > stop_loss:
                return new_stop
        else:  # short
            # For short positions, we move the stop down as price decreases
            if trail_percent > 0:
                new_stop = current_price * (1 + trail_percent / 100)
            else:
                new_stop = current_price + trail_amount
            
            # Only update if new stop is lower than current stop
            if new_stop < stop_loss:
                return new_stop
        
        return None
    
    def _log_trade_entry(self, position):
        """
        Log a trade entry to the trades collection
        
        Args:
            position (dict): Position document
        """
        # Create a trade entry record
        trade = {
            "trade_id": position["trade_id"],
            "symbol": position["symbol"],
            "exchange": position["exchange"],
            "position_type": position["position_type"],
            "quantity": position["quantity"],
            "entry_price": position["entry_price"],
            "entry_time": position["entry_time"],
            "stop_loss": position.get("stop_loss"),
            "target": position.get("target"),
            "strategy": position.get("strategy", "default"),
            "is_paper": position.get("is_paper", True),
            "initial_risk_amount": position.get("initial_risk_amount", 0),
            "signals": position.get("signals", []),
            "notes": position.get("notes", ""),
            "status": "open"
        }
        
        # Insert trade into trades collection
        self.db.trades_collection.insert_one(trade)
    
    def _log_trade_exit(self, position):
        """
        Log a trade exit to the trades collection
        
        Args:
            position (dict): Closed position document
        """
        # Update the trade record
        self.db.trades_collection.update_one(
            {"trade_id": position["trade_id"]},
            {
                "$set": {
                    "exit_price": position["exit_price"],
                    "exit_time": position["exit_time"],
                    "profit_loss": position["profit_loss"],
                    "profit_loss_percent": position["profit_loss_percent"],
                    "exit_reason": position.get("exit_reason", "Unknown"),
                    "status": "closed"
                }
            }
        )