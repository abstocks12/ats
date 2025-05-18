# realtime/execution/cost_optimizer.py
import logging
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

class CostOptimizer:
    """
    Optimizes trade execution to minimize transaction costs.
    """
    
    def __init__(self, db_connector, market_data_connector=None, logger=None):
        """
        Initialize the cost optimizer.
        
        Args:
            db_connector: MongoDB connector
            market_data_connector: Market data connector
            logger: Logger instance
        """
        self.db = db_connector
        self.market_data_connector = market_data_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Cache and state
        self.spread_cache = {}
        self.volume_profile_cache = {}
        self.vwap_cache = {}
        self.last_data_update = {}
        
        # Configuration
        self.config = {
            'limit_order_buffer': 0.05,  # % buffer added to limit orders
            'twap_interval': 60,  # seconds between TWAP slices
            'vwap_interval': 300,  # seconds between VWAP calculations
            'max_order_timeout': 300,  # seconds to wait for a limit order before adjusting
            'volume_profile_days': 20,  # days of data to use for volume profile
            'aggressive_threshold': 0.5,  # price improvement % to use aggressive orders
            'spread_timeout': 600,  # seconds between spread cache updates
            'min_tick_size': {
                'NSE': 0.05,  # Default tick size for NSE stocks
                'NFO': 0.05,  # Default tick size for NSE F&O
                'BSE': 0.01,  # Default tick size for BSE
                'default': 0.05  # Default tick size
            },
            'max_spread_factor': 5.0,  # Maximum spread factor for liquid instruments
            'max_price_impact': 0.2,  # Maximum allowed price impact (%)
            'max_slippage': {
                'equity': 0.1,  # Maximum slippage for equity (%)
                'futures': 0.05,  # Maximum slippage for futures (%)
                'options': 0.2  # Maximum slippage for options (%)
            }
        }
        
        self.logger.info("Cost optimizer initialized")
    
    def set_config(self, config):
        """
        Set optimizer configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated cost optimizer configuration")
    
    def optimize_limit_price(self, symbol, exchange, action, current_price=None, order_type='LIMIT', product_type=None):
        """
        Optimize limit price for a given symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            action (str): Action (BUY or SELL)
            current_price (float): Current price (optional)
            order_type (str): Order type
            product_type (str): Product type
            
        Returns:
            float: Optimized limit price
        """
        try:
            # If market order, return the current price
            if order_type == 'MARKET':
                return current_price
                
            # Get current price if not provided
            if not current_price and self.market_data_connector:
                data = self.market_data_connector.get_last_price(symbol, exchange)
                
                if data:
                    current_price = data.get('last_price')
                    
            if not current_price:
                self.logger.error(f"No price data for {symbol}")
                return None
                
            # Get spread
            spread = self._get_spread(symbol, exchange)
            
            # Get tick size
            tick_size = self._get_tick_size(symbol, exchange)
            
            # Add buffer for limit orders to increase chance of execution
            buffer_pct = self.config['limit_order_buffer']
            buffer_amount = current_price * buffer_pct / 100
            
            # Adjust based on action
            if action == 'BUY':
                # For buy orders, increase the price slightly to improve chance of execution
                limit_price = current_price + buffer_amount
            else:
                # For sell orders, decrease the price slightly
                limit_price = current_price - buffer_amount
                
            # Round to nearest tick size
            limit_price = round(limit_price / tick_size) * tick_size
            
            # For very illiquid instruments, adjust based on spread
            if spread and spread > 0:
                max_spread = current_price * self.config['max_spread_factor'] / 100
                
                if spread > max_spread:
                    # Very wide spread, adjust price more aggressively
                    if action == 'BUY':
                        # Offer a price halfway between current price and ask
                        limit_price = current_price + (spread / 2)
                    else:
                        # Offer a price halfway between current price and bid
                        limit_price = current_price - (spread / 2)
                        
                    # Round to nearest tick size
                    limit_price = round(limit_price / tick_size) * tick_size
            
            return limit_price
            
        except Exception as e:
            self.logger.error(f"Error optimizing limit price for {symbol}: {e}")
            return current_price
    
    def _get_spread(self, symbol, exchange):
        """
        Get bid-ask spread for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            float: Bid-ask spread
        """
        try:
            # Check cache first
            key = f"{symbol}:{exchange}"
            
            now = datetime.now()
            last_update = self.last_data_update.get(key)
            
            if last_update and (now - last_update).total_seconds() < self.config['spread_timeout'] and key in self.spread_cache:
                return self.spread_cache[key]
                
            # Get fresh data
            if self.market_data_connector:
                data = self.market_data_connector.get_last_price(symbol, exchange)
                
                if data:
                    bid = data.get('bid')
                    ask = data.get('ask')
                    
                    if bid is not None and ask is not None:
                        spread = ask - bid
                        
                        # Update cache
                        self.spread_cache[key] = spread
                        self.last_data_update[key] = now
                        
                        return spread
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting spread for {symbol}: {e}")
            return None
    
    def _get_tick_size(self, symbol, exchange):
        """
        Get tick size for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            float: Tick size
        """
        try:
            # Try to get symbol-specific tick size from database
            symbol_info = self.db.market_info_collection.find_one({
                'symbol': symbol,
                'exchange': exchange
            })
            
            if symbol_info and 'tick_size' in symbol_info:
                return symbol_info['tick_size']
                
            # Use default for exchange
            tick_size = self.config['min_tick_size'].get(exchange, self.config['min_tick_size']['default'])
            
            return tick_size
            
        except Exception as e:
            self.logger.error(f"Error getting tick size for {symbol}: {e}")
            return self.config['min_tick_size']['default']
    
    def calculate_optimal_execution_strategy(self, symbol, exchange, action, quantity, price, urgency='normal'):
        """
        Calculate optimal execution strategy for large orders.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            action (str): Action (BUY or SELL)
            quantity (int): Total quantity
            price (float): Current price
            urgency (str): Execution urgency ('high', 'normal', 'low')
            
        Returns:
            dict: Execution strategy
        """
        try:
            # Calculate average daily volume
            adv = self._get_adv(symbol, exchange)
            
            if not adv:
                # Default to single order if no volume data
                return {
                    'strategy': 'single',
                    'parameters': {'quantity': quantity, 'price': price}
                }
                
            # Calculate percentage of ADV
            pct_of_adv = (quantity * price) / adv * 100
            
            # Determine execution strategy based on order size and urgency
            if pct_of_adv < 1 or urgency == 'high':
                # Small order or high urgency, use single order
                return {
                    'strategy': 'single',
                    'parameters': {'quantity': quantity, 'price': price}
                }
            elif pct_of_adv < 5:
                # Medium order, use TWAP
                # Calculate number of slices based on urgency
                if urgency == 'normal':
                    slices = 3
                else:  # low urgency
                    slices = 5
                    
                slice_quantity = quantity // slices
                remainder = quantity % slices
                
                # Create schedule
                schedule = []
                for i in range(slices):
                    qty = slice_quantity + (1 if i < remainder else 0)
                    schedule.append({
                        'slice': i + 1,
                        'quantity': qty,
                        'time_offset': i * self.config['twap_interval']
                    })
                
                return {
                    'strategy': 'twap',
                    'parameters': {
                        'slices': slices,
                        'interval': self.config['twap_interval'],
                        'schedule': schedule
                    }
                }
            else:
                # Large order, use VWAP
                # Get volume profile
                volume_profile = self._get_volume_profile(symbol, exchange)
                
                if not volume_profile:
                    # Fall back to TWAP if no volume profile
                    return self.calculate_optimal_execution_strategy(
                        symbol, exchange, action, quantity, price, 'normal'
                    )
                
                # Calculate slices based on volume profile
                slices = []
                cumulative_volume = sum(volume_profile.values())
                
                remaining_quantity = quantity
                for period, volume_pct in sorted(volume_profile.items()):
                    # Calculate quantity for this period based on volume percentage
                    period_quantity = int(quantity * volume_pct / cumulative_volume)
                    
                    # Ensure we don't exceed the remaining quantity
                    period_quantity = min(period_quantity, remaining_quantity)
                    
                    if period_quantity > 0:
                        slices.append({
                            'period': period,
                            'quantity': period_quantity,
                            'volume_pct': volume_pct
                        })
                        
                        remaining_quantity -= period_quantity
                
                # Add any remaining quantity to the last slice
                if remaining_quantity > 0 and slices:
                    slices[-1]['quantity'] += remaining_quantity
                
                return {
                    'strategy': 'vwap',
                    'parameters': {
                        'volume_profile': volume_profile,
                        'slices': slices
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating execution strategy for {symbol}: {e}")
            return {
                'strategy': 'single',
                'parameters': {'quantity': quantity, 'price': price}
            }
    
    def _get_adv(self, symbol, exchange):
        """
        Get average daily volume for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            float: Average daily volume (in value terms)
        """
        try:
            # Try to get from database
            result = self.db.market_stats_collection.find_one({
                'symbol': symbol,
                'exchange': exchange,
                'stat_type': 'adv'
            })
            
            if result and 'value' in result:
                return result['value']
                
            # Calculate from historical data
            if not self.market_data_connector:
                return None
                
            # Get historical data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=30)
            
            data = self.market_data_connector.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                timeframe='day',
                from_date=from_date,
                to_date=to_date
            )
            
            if not data:
                return None
                
            # Calculate ADV
            volumes = []
            for bar in data:
                volume = bar.get('volume', 0)
                close = bar.get('close', 0)
                
                if volume and close:
                    value = volume * close
                    volumes.append(value)
            
            if not volumes:
                return None
                
            adv = sum(volumes) / len(volumes)
            
            # Store in database
            self.db.market_stats_collection.update_one(
                {
                    'symbol': symbol,
                    'exchange': exchange,
                    'stat_type': 'adv'
                },
                {
                    '$set': {
                        'value': adv,
                        'updated_at': datetime.now()
                    }
                },
                upsert=True
            )
            
            return adv
            
        except Exception as e:
            self.logger.error(f"Error getting ADV for {symbol}: {e}")
            return None
    
    def _get_volume_profile(self, symbol, exchange):
        """
        Get intraday volume profile for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            dict: Volume profile (period -> volume percentage)
        """
        try:
            # Check cache first
            key = f"{symbol}:{exchange}"
            
            now = datetime.now()
            last_update = self.last_data_update.get(f"{key}_volume_profile")
            
            if last_update and (now - last_update).total_seconds() < 86400 and key in self.volume_profile_cache:
                return self.volume_profile_cache[key]
                
            # Try to get from database
            result = self.db.market_stats_collection.find_one({
                'symbol': symbol,
                'exchange': exchange,
                'stat_type': 'volume_profile'
            })
            
            if result and 'profile' in result:
                # Update cache
                self.volume_profile_cache[key] = result['profile']
                self.last_data_update[f"{key}_volume_profile"] = now
                
                return result['profile']
                
            # Calculate from historical data
            if not self.market_data_connector:
                return None
                
            # Get historical data
            to_date = datetime.now()
            from_date = to_date - timedelta(days=self.config['volume_profile_days'])
            
            data = self.market_data_connector.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                timeframe='1hour',
                from_date=from_date,
                to_date=to_date
            )
            
            if not data:
                return None
                
            # Create hour-wise volume profile
            hourly_volumes = {}
            
            for bar in data:
                timestamp = bar.get('date')
                volume = bar.get('volume', 0)
                
                if not timestamp or not volume:
                    continue
                    
                if isinstance(timestamp, str):
                    timestamp = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
                
                hour = timestamp.hour
                
                if hour not in hourly_volumes:
                    hourly_volumes[hour] = []
                    
                hourly_volumes[hour].append(volume)
            
            # Calculate average volume for each hour
            volume_profile = {}
            
            for hour, volumes in hourly_volumes.items():
                if volumes:
                    avg_volume = sum(volumes) / len(volumes)
                    volume_profile[hour] = avg_volume
                    
            # Convert to percentages
            total_volume = sum(volume_profile.values())
            
            if total_volume > 0:
                for hour in volume_profile:
                    volume_profile[hour] = volume_profile[hour] / total_volume
                    
            # Store in database
            self.db.market_stats_collection.update_one(
                {
                    'symbol': symbol,
                    'exchange': exchange,
                    'stat_type': 'volume_profile'
                },
                {
                    '$set': {
                        'profile': volume_profile,
                        'updated_at': datetime.now()
                    }
                },
                upsert=True
            )
            
            # Update cache
            self.volume_profile_cache[key] = volume_profile
            self.last_data_update[f"{key}_volume_profile"] = now
            
            return volume_profile
            
        except Exception as e:
            self.logger.error(f"Error getting volume profile for {symbol}: {e}")
            return None
    
    def calculate_vwap(self, symbol, exchange, timeframe='day'):
        """
        Calculate VWAP (Volume-Weighted Average Price).
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            timeframe (str): Timeframe
            
        Returns:
            float: VWAP price
        """
        try:
            # Check cache first
            key = f"{symbol}:{exchange}:{timeframe}"
            
            now = datetime.now()
            last_update = self.last_data_update.get(f"{key}_vwap")
            
            if last_update and (now - last_update).total_seconds() < self.config['vwap_interval'] and key in self.vwap_cache:
                return self.vwap_cache[key]
                
            # Get historical data
            if not self.market_data_connector:
                return None
                
            # Get historical data
            to_date = datetime.now()
            
            if timeframe == 'day':
                from_date = datetime(to_date.year, to_date.month, to_date.day, 0, 0, 0)
            else:
                # Default to 1 day of data
                from_date = to_date - timedelta(days=1)
            
            data = self.market_data_connector.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                timeframe='5min',  # Use 5-minute data for accuracy
                from_date=from_date,
                to_date=to_date
            )
            
            if not data:
                return None
                
            # Calculate VWAP
            cumulative_pv = 0
            cumulative_volume = 0
            
            for bar in data:
                typical_price = (bar.get('high', 0) + bar.get('low', 0) + bar.get('close', 0)) / 3
                volume = bar.get('volume', 0)
                
                if typical_price and volume:
                    cumulative_pv += typical_price * volume
                    cumulative_volume += volume
            
            if cumulative_volume > 0:
                vwap = cumulative_pv / cumulative_volume
                
                # Update cache
                self.vwap_cache[key] = vwap
                self.last_data_update[f"{key}_vwap"] = now
                
                return vwap
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error calculating VWAP for {symbol}: {e}")
            return None
    
    def estimate_market_impact(self, symbol, exchange, quantity, price):
        """
        Estimate market impact of an order.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            quantity (int): Order quantity
            price (float): Current price
            
        Returns:
            float: Estimated market impact (percentage)
        """
        try:
            # Get ADV
            adv = self._get_adv(symbol, exchange)
            
            if not adv:
                return self.config['max_price_impact']
                
            # Calculate order value
            order_value = quantity * price
            
            # Calculate percentage of ADV
            pct_of_adv = order_value / adv * 100
            
            # Square root model for market impact
            impact = 0.1 * np.sqrt(pct_of_adv)
            
            # Cap at maximum impact
            impact = min(impact, self.config['max_price_impact'])
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Error estimating market impact for {symbol}: {e}")
            return self.config['max_price_impact']
    
    def calculate_optimal_order_size(self, symbol, exchange, action, current_price, max_risk_amount, urgency='normal'):
        """
        Calculate optimal order size based on market impact and risk tolerance.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            action (str): Action (BUY or SELL)
            current_price (float): Current price
            max_risk_amount (float): Maximum risk amount
            urgency (str): Execution urgency ('high', 'normal', 'low')
            
        Returns:
            int: Optimal order size
        """
        try:
            # Get market data
            adv = self._get_adv(symbol, exchange)
            spread = self._get_spread(symbol, exchange)
            
            if not adv:
                # Default to simple calculation if no volume data
                return int(max_risk_amount / current_price)
                
            # Determine maximum acceptable market impact based on urgency
            if urgency == 'high':
                max_impact = self.config['max_price_impact']
            elif urgency == 'normal':
                max_impact = self.config['max_price_impact'] * 0.7
            else:  # low urgency
                max_impact = self.config['max_price_impact'] * 0.5
                
            # Solve for quantity that would cause max_impact
            # impact = 0.1 * sqrt(quantity * price / adv * 100)
            # max_impact = 0.1 * sqrt(quantity * price / adv * 100)
            # (max_impact / 0.1)^2 = quantity * price / adv * 100
            # quantity = (max_impact / 0.1)^2 * adv / 100 / price
            
            max_quantity = ((max_impact / 0.1) ** 2) * adv / 100 / current_price
            
            # Calculate maximum quantity based on max risk amount
            risk_quantity = max_risk_amount / current_price
            
            # Use the smaller of the two
            optimal_quantity = min(max_quantity, risk_quantity)
            
            # Round down to integer
            optimal_quantity = int(optimal_quantity)
            
            return optimal_quantity
            
        except Exception as e:
            self.logger.error(f"Error calculating optimal order size for {symbol}: {e}")
            return int(max_risk_amount / current_price)
    
    def adjust_limit_order(self, order, elapsed_time, fill_percentage=0):
        """
        Adjust limit order parameters based on execution progress.
        
        Args:
            order (dict): Original order
            elapsed_time (float): Elapsed time since order placement (seconds)
            fill_percentage (float): Percentage of order filled
            
        Returns:
            dict: Adjusted order parameters
        """
        try:
            if 'price' not in order:
                return order
                
            # Check if timeout has been reached
            if elapsed_time >= self.config['max_order_timeout']:
                # Significant time has passed without full execution
                
                # Get current market data
                symbol = order.get('symbol')
                exchange = order.get('exchange')
                action = order.get('action')
                
                if not symbol or not exchange or not action:
                    return order
                    
                current_data = None
                
                if self.market_data_connector:
                    current_data = self.market_data_connector.get_last_price(symbol, exchange)
                
                if not current_data:
                    return order
                    
                current_price = current_data.get('last_price')
                
                if not current_price:
                    return order
                    
                # Determine how to adjust price based on action
                if action == 'BUY':
                    # For buy orders, increase price to improve chance of execution
                    adjustment_factor = 1 + (self.config['limit_order_buffer'] * 2 / 100)
                    new_price = order['price'] * adjustment_factor
                    
                    # Cap at current ask + buffer
                    ask = current_data.get('ask')
                    if ask:
                        ask_with_buffer = ask * (1 + self.config['limit_order_buffer'] / 100)
                        new_price = min(new_price, ask_with_buffer)
                else:
                    # For sell orders, decrease price
                    adjustment_factor = 1 - (self.config['limit_order_buffer'] * 2 / 100)
                    new_price = order['price'] * adjustment_factor
                    
                    # Cap at current bid - buffer
                    bid = current_data.get('bid')
                    if bid:
                        bid_with_buffer = bid * (1 - self.config['limit_order_buffer'] / 100)
                        new_price = max(new_price, bid_with_buffer)
                
                # Round to tick size
                tick_size = self._get_tick_size(symbol, exchange)
                new_price = round(new_price / tick_size) * tick_size
                
                # Create adjusted order
                adjusted_order = order.copy()
                adjusted_order['price'] = new_price
                
                # Consider switching to market order if multiple adjustments have failed
                if 'adjustment_count' in order and order['adjustment_count'] >= 2:
                    adjusted_order['order_type'] = 'MARKET'
                    
                # Update adjustment count
                adjusted_order['adjustment_count'] = order.get('adjustment_count', 0) + 1
                
                return adjusted_order
            
            # No adjustment needed
            return order
            
        except Exception as e:
            self.logger.error(f"Error adjusting limit order: {e}")
            return order