# stop_management.py
import numpy as np
from datetime import datetime, timedelta
import logging

class StopManager:
    def __init__(self, db_connector):
        """Initialize the stop manager"""
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
    
    def calculate_initial_stop(self, symbol, exchange, entry_price, direction="long", 
                            method="atr", risk_percent=None, fixed_points=None, 
                            timeframe="day", atr_multiple=2.0):
        """
        Calculate initial stop loss level using various methods
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - entry_price: Entry price
        - direction: Trade direction ('long' or 'short')
        - method: Stop calculation method ('atr', 'percent', 'fixed', 'support_resistance', 'swing')
        - risk_percent: Risk percentage (for 'percent' method)
        - fixed_points: Fixed distance in points (for 'fixed' method)
        - timeframe: Timeframe for ATR calculation
        - atr_multiple: Multiple of ATR for stop distance
        
        Returns:
        - Dictionary with stop details
        """
        try:
            if method == "atr":
                return self._calculate_atr_stop(symbol, exchange, entry_price, direction, timeframe, atr_multiple)
            elif method == "percent":
                return self._calculate_percent_stop(entry_price, direction, risk_percent)
            elif method == "fixed":
                return self._calculate_fixed_stop(entry_price, direction, fixed_points)
            elif method == "support_resistance":
                return self._calculate_sr_stop(symbol, exchange, entry_price, direction)
            elif method == "swing":
                return self._calculate_swing_stop(symbol, exchange, entry_price, direction, timeframe)
            else:
                return {"status": "error", "message": f"Unknown stop method: {method}"}
                
        except Exception as e:
            self.logger.error(f"Error calculating initial stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_atr_stop(self, symbol, exchange, entry_price, direction, timeframe, atr_multiple):
        """Calculate stop based on Average True Range"""
        try:
            # Get ATR value
            atr = self._get_atr(symbol, exchange, timeframe)
            
            if not atr:
                return {"status": "error", "message": "Unable to calculate ATR"}
            
            # Calculate stop distance
            stop_distance = atr * atr_multiple
            
            # Calculate stop level
            stop_level = entry_price - stop_distance if direction == "long" else entry_price + stop_distance
            
            # Ensure stop is positive
            stop_level = max(0.01, stop_level)
            
            return {
                "status": "success",
                "stop_level": stop_level,
                "method": "atr",
                "atr": atr,
                "atr_multiple": atr_multiple,
                "stop_distance": stop_distance,
                "stop_percent": (stop_distance / entry_price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_atr(self, symbol, exchange, timeframe):
        """Get Average True Range for a symbol"""
        try:
            # Try to get ATR from technical analyzer
            try:
                from research.technical_analyzer import TechnicalAnalyzer
                tech_analyzer = TechnicalAnalyzer(self.db)
                indicators = tech_analyzer.get_indicators(symbol, exchange, timeframe)
                
                if indicators and "atr_14" in indicators:
                    return indicators["atr_14"]
            except Exception as tech_error:
                self.logger.warning(f"Technical analyzer not available: {str(tech_error)}")
            
            # Fallback to calculating ATR directly
            # Query for historical price data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            # Get last 15 days of price data
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1}
            ).sort("timestamp", -1).limit(15))
            
            if not data or len(data) < 5:
                return None
            
            # Sort data chronologically
            data.reverse()
            
            # Calculate True Range
            tr_values = []
            for i in range(1, len(data)):
                high = data[i]["high"]
                low = data[i]["low"]
                prev_close = data[i-1]["close"]
                
                tr1 = high - low
                tr2 = abs(high - prev_close)
                tr3 = abs(low - prev_close)
                
                tr_values.append(max(tr1, tr2, tr3))
            
            # Calculate ATR (14-period average)
            atr = sum(tr_values) / len(tr_values)
            
            return atr
            
        except Exception as e:
            self.logger.error(f"Error getting ATR: {str(e)}")
            return None
    
    def _calculate_percent_stop(self, entry_price, direction, risk_percent):
        """Calculate stop based on percentage risk"""
        try:
            if not risk_percent or risk_percent <= 0:
                risk_percent = 2.0  # Default 2% risk
            
            # Calculate stop distance
            stop_distance = entry_price * (risk_percent / 100.0)
            
            # Calculate stop level
            stop_level = entry_price - stop_distance if direction == "long" else entry_price + stop_distance
            
            # Ensure stop is positive
            stop_level = max(0.01, stop_level)
            
            return {
                "status": "success",
                "stop_level": stop_level,
                "method": "percent",
                "risk_percent": risk_percent,
                "stop_distance": stop_distance
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating percent stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_fixed_stop(self, entry_price, direction, fixed_points):
        """Calculate stop based on fixed distance in points"""
        try:
            if not fixed_points or fixed_points <= 0:
                fixed_points = 10.0  # Default 10 points
            
            # Calculate stop level
            stop_level = entry_price - fixed_points if direction == "long" else entry_price + fixed_points
            
            # Ensure stop is positive
            stop_level = max(0.01, stop_level)
            
            return {
                "status": "success",
                "stop_level": stop_level,
                "method": "fixed",
                "fixed_points": fixed_points,
                "stop_percent": (fixed_points / entry_price) * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating fixed stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_sr_stop(self, symbol, exchange, entry_price, direction):
        """Calculate stop based on support/resistance levels"""
        try:
            # Try to get support/resistance levels from technical analyzer
            try:
                from research.technical_analyzer import TechnicalAnalyzer
                tech_analyzer = TechnicalAnalyzer(self.db)
                levels = tech_analyzer.get_support_resistance_levels(symbol, exchange)
                
                if not levels or "support" not in levels or "resistance" not in levels:
                    return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                
                # For long positions, find nearest support below entry
                if direction == "long":
                    supports = [s for s in levels["support"] if s < entry_price]
                    if not supports:
                        return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                    
                    stop_level = max(supports)  # Nearest support
                    
                # For short positions, find nearest resistance above entry
                else:
                    resistances = [r for r in levels["resistance"] if r > entry_price]
                    if not resistances:
                        return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                    
                    stop_level = min(resistances)  # Nearest resistance
                
                # Calculate stop distance and percentage
                stop_distance = abs(entry_price - stop_level)
                stop_percent = (stop_distance / entry_price) * 100
                
                # Limit maximum risk to 5%
                if stop_percent > 5.0:
                    return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                
                return {
                    "status": "success",
                    "stop_level": stop_level,
                    "method": "support_resistance",
                    "stop_distance": stop_distance,
                    "stop_percent": stop_percent
                }
                
            except Exception as tech_error:
                self.logger.warning(f"Technical analyzer not available: {str(tech_error)}")
                return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                
        except Exception as e:
            self.logger.error(f"Error calculating support/resistance stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_swing_stop(self, symbol, exchange, entry_price, direction, timeframe):
        """Calculate stop based on recent swing points"""
        try:
            # Get historical data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            # Get last 20 periods of price data
            data = list(self.db.market_data_collection.find(
                query,
                {"timestamp": 1, "high": 1, "low": 1}
            ).sort("timestamp", -1).limit(20))
            
            if not data or len(data) < 10:
                return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
            
            # Sort data chronologically
            data.reverse()
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            # Simple swing detection (can be improved with more sophisticated algorithms)
            for i in range(2, len(data) - 2):
                # Swing high
                if (data[i]["high"] > data[i-1]["high"] and
                    data[i]["high"] > data[i-2]["high"] and
                    data[i]["high"] > data[i+1]["high"] and
                    data[i]["high"] > data[i+2]["high"]):
                    swing_highs.append(data[i]["high"])
                
                # Swing low
                if (data[i]["low"] < data[i-1]["low"] and
                    data[i]["low"] < data[i-2]["low"] and
                    data[i]["low"] < data[i+1]["low"] and
                    data[i]["low"] < data[i+2]["low"]):
                    swing_lows.append(data[i]["low"])
            
            # For long positions, find nearest swing low below entry
            if direction == "long":
                valid_lows = [l for l in swing_lows if l < entry_price]
                if not valid_lows:
                    return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                
                stop_level = max(valid_lows)  # Nearest swing low
                
            # For short positions, find nearest swing high above entry
            else:
                valid_highs = [h for h in swing_highs if h > entry_price]
                if not valid_highs:
                    return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
                
                stop_level = min(valid_highs)  # Nearest swing high
            
            # Calculate stop distance and percentage
            stop_distance = abs(entry_price - stop_level)
            stop_percent = (stop_distance / entry_price) * 100
            
            # Limit maximum risk to 5%
            if stop_percent > 5.0:
                return self._calculate_percent_stop(entry_price, direction, 2.0)  # Fallback
            
            return {
                "status": "success",
                "stop_level": stop_level,
                "method": "swing",
                "stop_distance": stop_distance,
                "stop_percent": stop_percent
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating swing stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def update_trailing_stop(self, position, current_price, method="atr", params=None):
        """
        Update trailing stop for an existing position
        
        Parameters:
        - position: Current position data
        - current_price: Current market price
        - method: Trailing method ('atr', 'percent', 'fixed', 'chandelier')
        - params: Additional parameters for the method
        
        Returns:
        - Dictionary with updated stop details
        """
        try:
            # Get position details
            entry_price = position.get("entry_price")
            direction = position.get("direction", "long")
            current_stop = position.get("stop_loss")
            
            if not entry_price or not current_stop or not current_price:
                return {"status": "error", "message": "Missing position data"}
            
            # Check if in profit yet
            in_profit = (direction == "long" and current_price > entry_price) or \
                       (direction == "short" and current_price < entry_price)
            
            # Don't trail stop if not in profit yet
            if not in_profit:
                return {
                    "status": "success",
                    "stop_level": current_stop,
                    "updated": False,
                    "reason": "Position not in profit yet"
                }
            
            # Calculate new stop based on method
            if method == "atr":
                return self._update_atr_trailing_stop(position, current_price, params)
            elif method == "percent":
                return self._update_percent_trailing_stop(position, current_price, params)
            elif method == "fixed":
                return self._update_fixed_trailing_stop(position, current_price, params)
            elif method == "chandelier":
                return self._update_chandelier_stop(position, current_price, params)
            else:
                return {"status": "error", "message": f"Unknown trailing method: {method}"}
                
        except Exception as e:
            self.logger.error(f"Error updating trailing stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _update_atr_trailing_stop(self, position, current_price, params=None):
        """Update stop based on ATR trailing method"""
        try:
            # Default parameters
            atr_multiple = 2.0
            if params and "atr_multiple" in params:
                atr_multiple = params["atr_multiple"]
            
            # Get ATR value
            symbol = position.get("symbol")
            exchange = position.get("exchange")
            timeframe = position.get("timeframe", "day")
            
            atr = self._get_atr(symbol, exchange, timeframe)
            if not atr:
                return {
                    "status": "success",
                    "stop_level": position["stop_loss"],
                    "updated": False,
                    "reason": "Unable to calculate ATR"
                }
            
            # Calculate new stop level
            stop_distance = atr * atr_multiple
            
            if position["direction"] == "long":
                new_stop = current_price - stop_distance
                # Only raise stop, never lower it
                if new_stop <= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be lower than current stop"
                    }
            else:  # Short position
                new_stop = current_price + stop_distance
                # Only lower stop, never raise it
                if new_stop >= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be higher than current stop"
                    }
            
            return {
                "status": "success",
                "stop_level": new_stop,
                "updated": True,
                "old_stop": position["stop_loss"],
                "method": "atr"
            }
            
        except Exception as e:
            self.logger.error(f"Error updating ATR trailing stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _update_percent_trailing_stop(self, position, current_price, params=None):
        """Update stop based on percentage trailing method"""
        try:
            # Default parameters
            trail_percent = 2.0
            if params and "trail_percent" in params:
                trail_percent = params["trail_percent"]
            
            # Calculate new stop level
            stop_distance = current_price * (trail_percent / 100.0)
            
            if position["direction"] == "long":
                new_stop = current_price - stop_distance
                # Only raise stop, never lower it
                if new_stop <= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be lower than current stop"
                    }
            else:  # Short position
                new_stop = current_price + stop_distance
                # Only lower stop, never raise it
                if new_stop >= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be higher than current stop"
                    }
            
            return {
                "status": "success",
                "stop_level": new_stop,
                "updated": True,
                "old_stop": position["stop_loss"],
                "method": "percent"
            }
            
        except Exception as e:
            self.logger.error(f"Error updating percent trailing stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _update_fixed_trailing_stop(self, position, current_price, params=None):
        """Update stop based on fixed points trailing method"""
        try:
            # Default parameters
            trail_points = 10.0
            if params and "trail_points" in params:
                trail_points = params["trail_points"]
            
            if position["direction"] == "long":
                new_stop = current_price - trail_points
                # Only raise stop, never lower it
                if new_stop <= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be lower than current stop"
                    }
            else:  # Short position
                new_stop = current_price + trail_points
                # Only lower stop, never raise it
                if new_stop >= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be higher than current stop"
                    }
            
            return {
                "status": "success",
                "stop_level": new_stop,
                "updated": True,
                "old_stop": position["stop_loss"],
                "method": "fixed"
            }
            
        except Exception as e:
            self.logger.error(f"Error updating fixed trailing stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _update_chandelier_stop(self, position, current_price, params=None):
        """
        Update stop using Chandelier Exit method
        Places stop at X ATRs below highest high (for longs) or above lowest low (for shorts)
        """
        try:
            # Default parameters
            atr_multiple = 3.0
            lookback = 10
            if params:
                if "atr_multiple" in params:
                    atr_multiple = params["atr_multiple"]
                if "lookback" in params:
                    lookback = params["lookback"]
            
            # Get ATR value
            symbol = position.get("symbol")
            exchange = position.get("exchange")
            timeframe = position.get("timeframe", "day")
            
            atr = self._get_atr(symbol, exchange, timeframe)
            if not atr:
                return {
                    "status": "success",
                    "stop_level": position["stop_loss"],
                    "updated": False,
                    "reason": "Unable to calculate ATR"
                }
            
            # Get highest high or lowest low
            extreme_price = self._get_extreme_price(
                symbol, exchange, timeframe, 
                lookback, "high" if position["direction"] == "long" else "low"
            )
            
            if not extreme_price:
                return {
                    "status": "success",
                    "stop_level": position["stop_loss"],
                    "updated": False,
                    "reason": "Unable to calculate extreme price"
                }
            
            # Calculate new stop level
            if position["direction"] == "long":
                new_stop = extreme_price - (atr * atr_multiple)
                # Only raise stop, never lower it
                if new_stop <= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be lower than current stop"
                    }
            else:  # Short position
                new_stop = extreme_price + (atr * atr_multiple)
                # Only lower stop, never raise it
                if new_stop >= position["stop_loss"]:
                    return {
                        "status": "success",
                        "stop_level": position["stop_loss"],
                        "updated": False,
                        "reason": "New stop would be higher than current stop"
                    }
            
            return {
                "status": "success",
                "stop_level": new_stop,
                "updated": True,
                "old_stop": position["stop_loss"],
                "method": "chandelier",
                "extreme_price": extreme_price
            }
            
        except Exception as e:
            self.logger.error(f"Error updating chandelier stop: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_extreme_price(self, symbol, exchange, timeframe, lookback, extreme_type):
        """Get highest high or lowest low over lookback period"""
        try:
            # Query for historical price data
            query = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            # Get price data
            data = list(self.db.market_data_collection.find(
                query,
                {"high": 1, "low": 1}
            ).sort("timestamp", -1).limit(lookback))
            
            if not data:
                return None
            
            # Find extreme price
            if extreme_type == "high":
                extreme_price = max(d["high"] for d in data)
            else:
                extreme_price = min(d["low"] for d in data)
            
            return extreme_price
            
        except Exception as e:
            self.logger.error(f"Error getting extreme price: {str(e)}")
            return None
    
    def calculate_take_profit(self, entry_price, stop_loss, direction="long", risk_reward_ratio=2.0):
        """
        Calculate take profit level based on risk-reward ratio
        
        Parameters:
        - entry_price: Entry price
        - stop_loss: Stop loss price
        - direction: Trade direction ('long' or 'short')
        - risk_reward_ratio: Target risk-reward ratio (default: 2.0)
        
        Returns:
        - Dictionary with take profit details
        """
        try:
            # Calculate risk (distance to stop)
            risk = abs(entry_price - stop_loss)
            
            # Calculate reward based on risk-reward ratio
            reward = risk * risk_reward_ratio
            
            # Calculate take profit level
            if direction == "long":
                take_profit = entry_price + reward
            else:
                take_profit = entry_price - reward
            
            return {
                "status": "success",
                "take_profit": take_profit,
                "risk": risk,
                "reward": reward,
                "risk_reward_ratio": risk_reward_ratio
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def adjust_position_for_target(self, position, current_price, reached_target):
        """
        Adjust position when a target is reached (partial exit, move stop)
        
        Parameters:
        - position: Current position data
        - current_price: Current market price
        - reached_target: Which target has been reached (e.g., "target1", "target2")
        
        Returns:
        - Dictionary with position adjustment details
        """
        try:
            # Default adjustments
            adjustments = {
                "target1": {"exit_percent": 50, "move_stop": "entry"},
                "target2": {"exit_percent": 30, "move_stop": "halfway"},
                "target3": {"exit_percent": 20, "move_stop": "trail"}
            }
            
            if reached_target not in adjustments:
                return {"status": "error", "message": f"Unknown target: {reached_target}"}
            
            # Get adjustment parameters
            exit_percent = adjustments[reached_target]["exit_percent"]
            move_stop = adjustments[reached_target]["move_stop"]
            
            # Calculate quantity to exit
            current_quantity = position.get("quantity", 0)
            exit_quantity = int(current_quantity * (exit_percent / 100.0))
            remaining_quantity = current_quantity - exit_quantity
            
            # Calculate new stop loss level
            new_stop = position["stop_loss"]
            
            if move_stop == "entry":
                new_stop = position["entry_price"]
            elif move_stop == "halfway":
                # Halfway between current price and entry
                if position["direction"] == "long":
                    new_stop = position["entry_price"] + (current_price - position["entry_price"]) / 2
                else:
                    new_stop = position["entry_price"] - (position["entry_price"] - current_price) / 2
            elif move_stop == "trail":
                # Calculate trailing stop (2% from current price)
                if position["direction"] == "long":
                    new_stop = current_price * 0.98
                else:
                    new_stop = current_price * 1.02
            
            # For long positions, only raise stop; for shorts, only lower it
            if position["direction"] == "long":
                new_stop = max(position["stop_loss"], new_stop)
            else:
                new_stop = min(position["stop_loss"], new_stop)
            
            return {
                "status": "success",
                "reached_target": reached_target,
                "exit_quantity": exit_quantity,
                "remaining_quantity": remaining_quantity,
                "new_stop": new_stop,
                "stop_adjustment": move_stop
            }
            
        except Exception as e:
            self.logger.error(f"Error adjusting position for target: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def get_stop_statistics(self, symbol, exchange, timeframe="day", lookback=100):
        """
        Get statistics about stop loss performance
        
        Parameters:
        - symbol: Trading symbol
        - exchange: Exchange (e.g., NSE)
        - timeframe: Timeframe to analyze
        - lookback: Number of past trades to analyze
        
        Returns:
        - Dictionary with stop loss statistics
        """
        try:
            # Query for past trades
            query = {
                "symbol": symbol,
                "exchange": exchange
            }
            
            trades = list(self.db.trade_collection.find(query).sort("exit_time", -1).limit(lookback))
            
            if not trades:
                return {"status": "error", "message": "No trade history found"}
            
            # Calculate statistics
            total_trades = len(trades)
            stopped_out = sum(1 for t in trades if t.get("exit_reason") == "stop_loss")
            stop_win = sum(1 for t in trades if t.get("exit_reason") == "stop_loss" and t.get("profit_loss", 0) > 0)
            
            stop_percent = (stopped_out / total_trades) * 100 if total_trades > 0 else 0
            win_percent = (stop_win / stopped_out) * 100 if stopped_out > 0 else 0
            
            # Calculate average drawdown before exit for winning trades
            winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
            avg_drawdown = None
            
            if winning_trades:
                drawdowns = []
                for trade in winning_trades:
                    if "max_drawdown" in trade:
                        drawdowns.append(trade["max_drawdown"])
                
                if drawdowns:
                    avg_drawdown = sum(drawdowns) / len(drawdowns)
            
            return {
                "status": "success",
                "total_trades": total_trades,
                "stopped_out_count": stopped_out,
                "stopped_out_percent": stop_percent,
                "stop_win_count": stop_win,
                "stop_win_percent": win_percent,
                "avg_drawdown_winning": avg_drawdown
            }
            
        except Exception as e:
            self.logger.error(f"Error getting stop statistics: {str(e)}")
            return {"status": "error", "message": str(e)}