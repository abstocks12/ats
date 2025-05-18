"""
Position Sizing Module for the Automated Trading System.

This module implements various position sizing strategies to determine 
the appropriate position size for trades based on risk parameters,
account size, and market conditions.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class PositionSizer:
    """
    Calculates appropriate position sizes based on various risk management techniques.
    
    This class provides methods for determining position sizes based on different
    approaches including fixed risk, volatility-adjusted risk, and portfolio-based
    allocation methods.
    """
    
    def __init__(self, db_connector, max_portfolio_risk=0.02, default_risk_per_trade=0.01):
        """
        Initialize the PositionSizer class.
        
        Args:
            db_connector: MongoDB database connector
            max_portfolio_risk (float): Maximum overall portfolio risk (default 2%)
            default_risk_per_trade (float): Default risk per trade (default 1%)
        """
        self.db = db_connector
        self.logger = logging.getLogger(__name__)
        self.max_portfolio_risk = max_portfolio_risk
        self.default_risk_per_trade = default_risk_per_trade
    
    def calculate_position_size(self, symbol, exchange, strategy_type, entry_price, 
                               stop_loss, instrument_type="equity", risk_per_trade=None,
                               confidence=None, custom_portfolio_allocation=None):
        """
        Calculate optimal position size based on risk parameters.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange identifier (NSE, BSE)
            strategy_type (str): Strategy type (technical, fundamental, etc.)
            entry_price (float): Entry price for the trade
            stop_loss (float): Stop loss price
            instrument_type (str): Instrument type (equity, futures, options)
            risk_per_trade (float): Risk per trade as percentage of account (optional)
            confidence (float): Confidence score from prediction model (optional)
            custom_portfolio_allocation (float): Custom allocation for this symbol (optional)
        
        Returns:
            dict: Position sizing information including quantity and risk metrics
        """
        try:
            # Get account equity from database
            account_info = self.db.system_collection.find_one({"type": "account_info"})
            if not account_info:
                self.logger.error("Account information not found in database")
                return None
            
            # Get current account equity
            account_equity = account_info.get("equity", 0)
            if account_equity <= 0:
                self.logger.error(f"Invalid account equity: {account_equity}")
                return None
            
            # Get instrument info (for options and futures)
            instrument_info = None
            if instrument_type in ["futures", "options"]:
                instrument_info = self.db.instrument_collection.find_one({
                    "symbol": symbol,
                    "exchange": exchange,
                    "instrument_type": instrument_type
                })
                
                if not instrument_info:
                    self.logger.error(f"Instrument info not found for {symbol} {instrument_type}")
                    return None
            
            # Determine risk per trade (if not provided)
            if risk_per_trade is None:
                # Get instrument settings from portfolio
                instrument_settings = self.db.portfolio_collection.find_one({
                    "symbol": symbol,
                    "exchange": exchange
                })
                
                if instrument_settings and "trading_config" in instrument_settings:
                    risk_per_trade = instrument_settings["trading_config"].get(
                        "max_risk_percent", self.default_risk_per_trade
                    ) / 100  # Convert percentage to decimal
                else:
                    risk_per_trade = self.default_risk_per_trade
            
            # Adjust risk based on confidence (if provided)
            if confidence is not None:
                # Scale risk by confidence level: higher confidence = higher risk
                confidence_factor = 0.5 + (confidence / 2)  # Maps 0-1 confidence to 0.5-1.0 factor
                risk_per_trade = risk_per_trade * confidence_factor
            
            # Adjust for overall portfolio risk limit
            total_open_risk = self._calculate_current_portfolio_risk(account_equity)
            available_risk = max(0, self.max_portfolio_risk - total_open_risk)
            
            if available_risk <= 0:
                self.logger.warning("Maximum portfolio risk reached. No new positions allowed.")
                return {
                    "quantity": 0,
                    "risk_amount": 0,
                    "risk_percentage": 0,
                    "message": "Maximum portfolio risk limit reached"
                }
            
            # Cap risk per trade to available risk
            risk_per_trade = min(risk_per_trade, available_risk)
            
            # Calculate risk amount in currency
            risk_amount = account_equity * risk_per_trade
            
            # Calculate position size based on instrument type
            if instrument_type == "equity":
                return self._calculate_equity_position(
                    symbol, exchange, entry_price, stop_loss, risk_amount, account_equity
                )
            elif instrument_type == "futures":
                return self._calculate_futures_position(
                    symbol, exchange, entry_price, stop_loss, risk_amount, 
                    account_equity, instrument_info
                )
            elif instrument_type == "options":
                return self._calculate_options_position(
                    symbol, exchange, entry_price, stop_loss, risk_amount, 
                    account_equity, instrument_info
                )
            else:
                self.logger.error(f"Unsupported instrument type: {instrument_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return None
    
    def _calculate_equity_position(self, symbol, exchange, entry_price, stop_loss, 
                                  risk_amount, account_equity):
        """
        Calculate position size for equity instruments.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange identifier
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            risk_amount (float): Amount to risk in currency
            account_equity (float): Total account equity
            
        Returns:
            dict: Position sizing information
        """
        # Calculate risk per share (price difference between entry and stop)
        if entry_price <= 0 or stop_loss <= 0:
            self.logger.error(f"Invalid price values: entry={entry_price}, stop={stop_loss}")
            return None
            
        # For long positions
        if entry_price > stop_loss:
            risk_per_share = entry_price - stop_loss
        # For short positions
        elif entry_price < stop_loss:
            risk_per_share = stop_loss - entry_price
        else:
            self.logger.error("Entry price and stop loss cannot be equal")
            return None
        
        # Calculate shares based on risk amount and risk per share
        shares = risk_amount / risk_per_share
        
        # Round down to whole number of shares
        shares = int(shares)
        
        # Check minimum lot size (if applicable)
        min_lot_size = self._get_min_lot_size(symbol, exchange)
        if min_lot_size > 1:
            # Round down to nearest lot size
            shares = (shares // min_lot_size) * min_lot_size
        
        # Calculate actual risk amount and percentage based on final shares
        actual_risk_amount = shares * risk_per_share
        actual_risk_percentage = (actual_risk_amount / account_equity) * 100
        
        # Check if position value exceeds maximum allocation
        position_value = shares * entry_price
        max_position_percentage = self._get_max_position_size_percentage(symbol, exchange)
        max_position_value = account_equity * max_position_percentage
        
        if position_value > max_position_value:
            # Cap shares to max position value
            shares = int(max_position_value / entry_price)
            # Adjust to lot size if needed
            if min_lot_size > 1:
                shares = (shares // min_lot_size) * min_lot_size
            
            # Recalculate risk metrics
            actual_risk_amount = shares * risk_per_share
            actual_risk_percentage = (actual_risk_amount / account_equity) * 100
            position_value = shares * entry_price
        
        return {
            "quantity": shares,
            "risk_amount": actual_risk_amount,
            "risk_percentage": actual_risk_percentage,
            "position_value": position_value,
            "position_percentage": (position_value / account_equity) * 100,
            "risk_reward_ratio": self._estimate_risk_reward_ratio(symbol, exchange, entry_price, stop_loss)
        }
    
    def _calculate_futures_position(self, symbol, exchange, entry_price, stop_loss, 
                                   risk_amount, account_equity, instrument_info):
        """
        Calculate position size for futures instruments.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange identifier
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            risk_amount (float): Amount to risk in currency
            account_equity (float): Total account equity
            instrument_info (dict): Futures instrument information
            
        Returns:
            dict: Position sizing information
        """
        # Get lot size and multiplier from instrument info
        lot_size = instrument_info.get("lot_size", 1)
        multiplier = instrument_info.get("multiplier", 1)
        
        # Calculate risk per contract
        if entry_price > stop_loss:  # Long position
            risk_per_point = entry_price - stop_loss
        else:  # Short position
            risk_per_point = stop_loss - entry_price
        
        risk_per_contract = risk_per_point * lot_size * multiplier
        
        # Calculate number of contracts
        contracts = risk_amount / risk_per_contract
        
        # Round down to whole number of contracts
        contracts = int(contracts)
        
        # Calculate margin requirement
        margin_per_contract = instrument_info.get("margin_required", entry_price * lot_size * 0.2)  # Default to 20% if not specified
        total_margin = contracts * margin_per_contract
        
        # Check if margin requirement exceeds available margin
        available_margin = account_equity * 0.8  # Use 80% of equity as available margin
        if total_margin > available_margin:
            contracts = int(available_margin / margin_per_contract)
        
        # Final calculations
        actual_risk_amount = contracts * risk_per_contract
        actual_risk_percentage = (actual_risk_amount / account_equity) * 100
        position_value = contracts * entry_price * lot_size * multiplier
        
        return {
            "quantity": contracts,
            "contract_size": lot_size,
            "multiplier": multiplier,
            "risk_amount": actual_risk_amount,
            "risk_percentage": actual_risk_percentage,
            "position_value": position_value,
            "position_percentage": (position_value / account_equity) * 100,
            "margin_required": contracts * margin_per_contract,
            "risk_reward_ratio": self._estimate_risk_reward_ratio(symbol, exchange, entry_price, stop_loss)
        }
    
    def _calculate_options_position(self, symbol, exchange, entry_price, stop_loss, 
                                   risk_amount, account_equity, instrument_info):
        """
        Calculate position size for options instruments.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange identifier
            entry_price (float): Entry price (premium)
            stop_loss (float): Stop loss price (premium)
            risk_amount (float): Amount to risk in currency
            account_equity (float): Total account equity
            instrument_info (dict): Options instrument information
            
        Returns:
            dict: Position sizing information
        """
        # Get lot size from instrument info
        lot_size = instrument_info.get("lot_size", 1)
        
        # For options, risk calculation differs between buying and writing options
        option_type = instrument_info.get("option_type", "call")
        is_buy = instrument_info.get("is_buy", True)
        
        if is_buy:  # Buying options
            # For buying options, max risk is the premium paid
            # Risk per contract for buying options is limited to premium
            max_risk_per_contract = entry_price * lot_size
            
            # For partial stop loss exits
            risk_per_contract = (entry_price - stop_loss) * lot_size if entry_price > stop_loss else entry_price * lot_size
            
            # Calculate number of contracts based on risk amount
            contracts = risk_amount / risk_per_contract
            
            # Round down to whole number of contracts
            contracts = int(contracts)
            
            # Calculate total premium
            total_premium = contracts * entry_price * lot_size
            
            # Check if premium exceeds max allocation
            max_option_allocation = account_equity * 0.05  # Limit option buying to 5% of account by default
            if total_premium > max_option_allocation:
                contracts = int(max_option_allocation / (entry_price * lot_size))
            
            # Final calculations
            actual_risk_amount = contracts * risk_per_contract
            actual_risk_percentage = (actual_risk_amount / account_equity) * 100
            
            return {
                "quantity": contracts,
                "lot_size": lot_size,
                "risk_amount": actual_risk_amount,
                "risk_percentage": actual_risk_percentage,
                "premium_paid": contracts * entry_price * lot_size,
                "max_loss": contracts * entry_price * lot_size,
                "risk_reward_ratio": self._estimate_option_risk_reward(
                    symbol, exchange, option_type, is_buy, entry_price, instrument_info
                )
            }
            
        else:  # Writing options
            # For writing options, theoretical risk can be much higher (unlimited for calls)
            # Use margin requirements as a constraint
            margin_per_contract = instrument_info.get("margin_required", 
                                                     entry_price * lot_size * 3)  # Default to 3x premium if not specified
            
            # For writing options, typically use a higher stop loss multiple
            if stop_loss <= 0:  # If no stop loss provided for written option
                stop_loss = entry_price * 2  # Default to 2x premium as stop loss
            
            risk_per_contract = (stop_loss - entry_price) * lot_size
            
            # Calculate contracts based on risk
            contracts = risk_amount / risk_per_contract
            
            # Round down to whole number
            contracts = int(contracts)
            
            # Check margin requirements
            total_margin = contracts * margin_per_contract
            available_margin = account_equity * 0.5  # Limit option writing to 50% of account
            
            if total_margin > available_margin:
                contracts = int(available_margin / margin_per_contract)
            
            # Final calculations
            actual_risk_amount = contracts * risk_per_contract
            actual_risk_percentage = (actual_risk_amount / account_equity) * 100
            
            return {
                "quantity": contracts,
                "lot_size": lot_size,
                "risk_amount": actual_risk_amount,
                "risk_percentage": actual_risk_percentage,
                "premium_received": contracts * entry_price * lot_size,
                "margin_required": contracts * margin_per_contract,
                "max_loss_estimate": "Potentially unlimited" if option_type == "call" else 
                                    f"{contracts * lot_size * instrument_info.get('strike_price', 0)}",
                "risk_reward_ratio": self._estimate_option_risk_reward(
                    symbol, exchange, option_type, is_buy, entry_price, instrument_info
                )
            }
    
    def _calculate_current_portfolio_risk(self, account_equity):
        """
        Calculate the current risk exposure of the portfolio.
        
        Args:
            account_equity (float): Current account equity
            
        Returns:
            float: Current portfolio risk as decimal (0.0-1.0)
        """
        try:
            # Get all open positions
            open_positions = list(self.db.positions_collection.find({"status": "open"}))
            
            if not open_positions:
                return 0
            
            total_risk = 0
            
            for position in open_positions:
                # Calculate risk for this position
                entry_price = position.get("entry_price", 0)
                stop_loss = position.get("stop_loss", 0)
                quantity = position.get("quantity", 0)
                instrument_type = position.get("instrument_type", "equity")
                
                if entry_price <= 0 or stop_loss <= 0 or quantity <= 0:
                    continue
                
                # Calculate risk based on instrument type
                if instrument_type == "equity":
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_risk = risk_per_unit * quantity
                
                elif instrument_type == "futures":
                    lot_size = position.get("lot_size", 1)
                    multiplier = position.get("multiplier", 1)
                    risk_per_unit = abs(entry_price - stop_loss)
                    position_risk = risk_per_unit * quantity * lot_size * multiplier
                
                elif instrument_type == "options":
                    lot_size = position.get("lot_size", 1)
                    is_buy = position.get("is_buy", True)
                    
                    if is_buy:
                        # For long options, max risk is premium paid
                        position_risk = position.get("premium_paid", entry_price * quantity * lot_size)
                    else:
                        # For short options, risk is based on stop loss
                        risk_per_unit = abs(entry_price - stop_loss)
                        position_risk = risk_per_unit * quantity * lot_size
                
                # Add to total risk
                total_risk += position_risk
            
            # Return as percentage of account
            return total_risk / account_equity
            
        except Exception as e:
            self.logger.error(f"Error calculating current portfolio risk: {e}")
            return 0
    
    def _get_min_lot_size(self, symbol, exchange):
        """Get minimum lot size for the instrument."""
        try:
            # Check if symbol has a minimum lot size
            instrument = self.db.instrument_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            if instrument and "lot_size" in instrument:
                return instrument["lot_size"]
            
            return 1  # Default to 1 if not specified
            
        except Exception as e:
            self.logger.error(f"Error getting lot size: {e}")
            return 1
    
    def _get_max_position_size_percentage(self, symbol, exchange):
        """Get maximum position size as percentage of account."""
        try:
            # Check if symbol has custom position size limit
            instrument_settings = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            if instrument_settings and "trading_config" in instrument_settings:
                return instrument_settings["trading_config"].get("position_size_percent", 20) / 100
            
            return 0.20  # Default to 20% if not specified
            
        except Exception as e:
            self.logger.error(f"Error getting max position size: {e}")
            return 0.20
    
    def _estimate_risk_reward_ratio(self, symbol, exchange, entry_price, stop_loss):
        """
        Estimate risk-reward ratio based on historical data and technical analysis.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            
        Returns:
            float: Estimated risk-reward ratio
        """
        try:
            # Get recent price data
            price_data = list(self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "day"
                },
                {"timestamp": 1, "high": 1, "low": 1, "close": 1}
            ).sort("timestamp", -1).limit(30))  # Last 30 days
            
            if not price_data:
                return 2.0  # Default R:R if no data
            
            # Calculate potential reward based on recent price action
            df = pd.DataFrame(price_data)
            
            # For long trades
            if entry_price > stop_loss:
                # Risk is entry - stop
                risk = entry_price - stop_loss
                
                # Potential targets based on recent swings
                recent_highs = df["high"].rolling(window=10).max().dropna()
                if len(recent_highs) > 0:
                    potential_target = recent_highs.max()
                    
                    # If target is below entry, use ATR to estimate target
                    if potential_target <= entry_price:
                        atr = self._calculate_atr(df)
                        potential_target = entry_price + (atr * 3)  # 3x ATR as target
                    
                    reward = potential_target - entry_price
                else:
                    # Fallback if no recent highs
                    reward = risk * 2  # Default 2:1 reward:risk
            
            # For short trades
            else:
                risk = stop_loss - entry_price
                
                # Potential targets based on recent swings
                recent_lows = df["low"].rolling(window=10).min().dropna()
                if len(recent_lows) > 0:
                    potential_target = recent_lows.min()
                    
                    # If target is above entry, use ATR to estimate target
                    if potential_target >= entry_price:
                        atr = self._calculate_atr(df)
                        potential_target = entry_price - (atr * 3)  # 3x ATR as target
                    
                    reward = entry_price - potential_target
                else:
                    # Fallback if no recent lows
                    reward = risk * 2  # Default 2:1 reward:risk
            
            # Calculate ratio
            if risk > 0:
                ratio = reward / risk
                return min(max(ratio, 1.0), 5.0)  # Cap between 1 and 5
            else:
                return 2.0  # Default if risk calculation failed
                
        except Exception as e:
            self.logger.error(f"Error estimating risk-reward ratio: {e}")
            return 2.0  # Default R:R if calculation failed
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range for volatility measurement."""
        try:
            if len(df) < period:
                return (df["high"].max() - df["low"].min()) / len(df)
                
            # Calculate True Range
            df = df.copy()
            df["prev_close"] = df["close"].shift(1)
            df["tr1"] = abs(df["high"] - df["low"])
            df["tr2"] = abs(df["high"] - df["prev_close"])
            df["tr3"] = abs(df["low"] - df["prev_close"])
            df["tr"] = df[["tr1", "tr2", "tr3"]].max(axis=1)
            
            # Calculate ATR
            atr = df["tr"].rolling(window=period).mean().iloc[-1]
            return atr
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0
    
    def _estimate_option_risk_reward(self, symbol, exchange, option_type, is_buy, 
                                    premium, instrument_info):
        """
        Estimate risk-reward ratio for options trades.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            option_type (str): "call" or "put"
            is_buy (bool): True if buying, False if writing
            premium (float): Option premium
            instrument_info (dict): Option instrument information
            
        Returns:
            float: Estimated risk-reward ratio
        """
        try:
            strike_price = instrument_info.get("strike_price", 0)
            if strike_price <= 0:
                return 2.0  # Default if missing strike price
            
            # Get underlying price
            underlying_symbol = instrument_info.get("underlying_symbol", symbol.split("_")[0])
            underlying_data = self.db.market_data_collection.find_one(
                {
                    "symbol": underlying_symbol,
                    "exchange": exchange,
                    "timeframe": "day"
                },
                sort=[("timestamp", -1)]
            )
            
            if not underlying_data:
                return 2.0  # Default if no underlying data
                
            underlying_price = underlying_data.get("close", 0)
            if underlying_price <= 0:
                return 2.0  # Default if invalid price
            
            # Calculate days to expiration
            expiry_date = instrument_info.get("expiry_date")
            if not expiry_date:
                days_to_expiry = 30  # Default if missing expiry
            else:
                if isinstance(expiry_date, str):
                    expiry_date = datetime.strptime(expiry_date, "%Y-%m-%d")
                days_to_expiry = (expiry_date - datetime.now()).days
                days_to_expiry = max(1, days_to_expiry)  # At least 1 day
            
            # Different calculations based on option type and buy/write
            if is_buy:
                if option_type == "call":
                    # For buying calls
                    if underlying_price > strike_price:  # In the money
                        intrinsic = underlying_price - strike_price
                        extrinsic = premium - intrinsic
                        
                        # Estimate potential upside based on recent volatility
                        volatility = self._estimate_volatility(underlying_symbol, exchange)
                        potential_move = underlying_price * volatility * (days_to_expiry / 365) ** 0.5
                        potential_price = underlying_price + potential_move
                        potential_value = max(0, potential_price - strike_price)
                        
                        # Risk is premium, reward is potential value - premium
                        risk = premium
                        reward = potential_value - premium
                    else:  # Out of the money
                        # Risk is premium, reward based on potential move
                        volatility = self._estimate_volatility(underlying_symbol, exchange)
                        potential_move = underlying_price * volatility * (days_to_expiry / 365) ** 0.5
                        potential_price = underlying_price + potential_move
                        potential_value = max(0, potential_price - strike_price)
                        
                        risk = premium
                        reward = potential_value - premium
                
                elif option_type == "put":
                    # For buying puts
                    if underlying_price < strike_price:  # In the money
                        intrinsic = strike_price - underlying_price
                        extrinsic = premium - intrinsic
                        
                        # Estimate potential downside
                        volatility = self._estimate_volatility(underlying_symbol, exchange)
                        potential_move = underlying_price * volatility * (days_to_expiry / 365) ** 0.5
                        potential_price = underlying_price - potential_move
                        potential_value = max(0, strike_price - potential_price)
                        
                        risk = premium
                        reward = potential_value - premium
                    else:  # Out of the money
                        # Estimate potential downside
                        volatility = self._estimate_volatility(underlying_symbol, exchange)
                        potential_move = underlying_price * volatility * (days_to_expiry / 365) ** 0.5
                        potential_price = underlying_price - potential_move
                        potential_value = max(0, strike_price - potential_price)
                        
                        risk = premium
                        reward = potential_value - premium
            else:
                # For writing options
                if option_type == "call":
                    # Risk is theoretically unlimited, use a high multiple of premium
                    risk = premium * 3  # Simplified estimation
                    reward = premium
                elif option_type == "put":
                    # Maximum risk is strike price - premium
                    risk = strike_price - premium
                    reward = premium
            
            # Calculate ratio
            if risk > 0 and reward > 0:
                ratio = reward / risk
                return min(max(ratio, 0.5), 5.0)  # Cap between 0.5 and 5
            else:
                return 1.0  # Default if calculation invalid
                
        except Exception as e:
            self.logger.error(f"Error estimating option risk-reward: {e}")
            return 1.5  # Default R:R if calculation failed
    
    def _estimate_volatility(self, symbol, exchange, period=30):
        """
        Estimate volatility of underlying instrument.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            period (int): Period in days
            
        Returns:
            float: Estimated annualized volatility
        """
        try:
            # Get historical data
            price_data = list(self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": "day"
                },
                {"timestamp": 1, "close": 1}
            ).sort("timestamp", -1).limit(period + 1))
            
            if len(price_data) < period:
                return 0.3  # Default 30% volatility if insufficient data
            
            # Calculate daily returns
            df = pd.DataFrame(price_data)
            df = df.sort_values("timestamp")
            df["return"] = df["close"].pct_change()
            
            # Calculate annualized volatility
            daily_volatility = df["return"].std()
            annualized_volatility = daily_volatility * (252 ** 0.5)  # 252 trading days per year
            
            return max(min(annualized_volatility, 1.0), 0.1)
            
        except Exception as e:
            self.logger.error(f"Error estimating volatility: {e}")
            return 0.3  # Default if calculation failed


def calculate_position_size(prediction, trading_config, db):
    """
    Standalone function for calculating position size based on prediction and config.
    
    This function is designed to be called from the trading engine.
    
    Args:
        prediction (dict): Prediction data containing symbol, confidence, etc.
        trading_config (dict): Trading configuration
        db: Database connector
    
    Returns:
        int: Position size (quantity)
    """
    try:
        # Extract prediction data
        symbol = prediction.get("symbol")
        exchange = prediction.get("exchange")
        confidence = prediction.get("confidence", 0.7)
        
        # Get entry and exit prices from prediction
        entry_price = prediction.get("target_price")
        stop_loss = prediction.get("stop_loss")
        
        if not entry_price or not stop_loss:
            # Try to get from supporting factors
            supporting_factors = prediction.get("supporting_factors", [])
            for factor in supporting_factors:
                if "target_price" in factor and not entry_price:
                    entry_price = factor["target_price"]
                if "stop_loss" in factor and not stop_loss:
                    stop_loss = factor["stop_loss"]
        
        # If still missing prices, return 0
        if not entry_price or not stop_loss:
            return 0
        
        # Get risk parameters from trading config
        risk_per_trade = trading_config.get("max_risk_percent", 1) / 100
        position_size_percent = trading_config.get("position_size_percent", 5) / 100
        
        # Get account information
        account_info = db.system_collection.find_one({"type": "account_info"})
        if not account_info:
            return 0
        
        account_equity = account_info.get("equity", 0)
        if account_equity <= 0:
            return 0
        
        # Calculate position size
        position_sizer = PositionSizer(db, default_risk_per_trade=risk_per_trade)
        sizing_result = position_sizer.calculate_position_size(
            symbol=symbol,
            exchange=exchange,
            strategy_type=trading_config.get("strategies", ["technical"])[0],
            entry_price=entry_price,
            stop_loss=stop_loss,
            instrument_type=trading_config.get("instrument_type", "equity"),
            risk_per_trade=risk_per_trade,
            confidence=confidence
        )
        
        if not sizing_result:
            return 0
        
        return sizing_result.get("quantity", 0)
        
    except Exception as e:
        logging.error(f"Error in calculate_position_size: {e}")
        return 0