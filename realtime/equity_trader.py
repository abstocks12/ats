# realtime/equity_trader.py
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class EquityTrader:
    """
    Specialized trading module for equity (stock) trading.
    """
    
    def __init__(self, trading_engine, db_connector, logger=None):
        """
        Initialize the equity trader.
        
        Args:
            trading_engine: Trading engine instance
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.engine = trading_engine
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # State
        self.active_symbols = set()
        self.active_strategies = {}
        self.is_running = False
        
        # Configuration
        self.config = {
            'max_symbols': 50,  # Maximum number of symbols to trade
            'order_types': ['LIMIT', 'MARKET'],  # Allowed order types
            'default_order_type': 'LIMIT',
            'default_product': 'CNC',  # CNC for delivery, MIS for intraday
            'default_exchange': 'NSE',
            'max_slippage': 0.1,  # Maximum slippage in percentage
            'price_tick_size': 0.05,  # Price tick size
            'use_limit_orders': True,  # Use limit orders instead of market orders
            'limit_order_timeout': 60,  # Timeout for limit orders in seconds
            'volume_filter': 100000,  # Minimum volume filter
            'min_price': 10.0,  # Minimum price filter
            'max_price': 10000.0,  # Maximum price filter
            'trade_session_start': '09:15:00',  # Trading session start time
            'trade_session_end': '15:30:00',  # Trading session end time
            'max_positions': 10,  # Maximum number of positions
            'intraday_square_off_time': '15:15:00',  # Intraday square off time
            'strategy_timeframes': ['1min', '5min', '15min', '30min', '1hour', 'day']  # Supported timeframes
        }
        
        # Initialize
        self._initialize()
        
        self.logger.info("Equity trader initialized")
    
    def set_config(self, config):
        """
        Set configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated equity trader configuration: {self.config}")
    
    def _initialize(self):
        """
        Initialize the equity trader.
        """
        # Register with trading engine
        if self.engine:
            # If engine is already running, we need to make sure it has our callback
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                self.engine.market_data_connector.set_data_callback(self._on_market_data)
        else:
            self.logger.warning("No trading engine provided")
    
    def add_symbol(self, symbol, exchange=None):
        """
        Add a symbol for trading.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            bool: Success status
        """
        exchange = exchange or self.config['default_exchange']
        
        # Check if already added
        symbol_key = f"{symbol}:{exchange}"
        if symbol_key in self.active_symbols:
            self.logger.info(f"Symbol {symbol_key} already added")
            return True
            
        # Check maximum symbols
        if len(self.active_symbols) >= self.config['max_symbols']:
            self.logger.warning(f"Maximum symbols reached: {self.config['max_symbols']}")
            return False
            
        # Add to active symbols
        self.active_symbols.add(symbol_key)
        
        # Subscribe to market data
        if self.engine and self.is_running:
            if self.engine.market_data_connector:
                success = self.engine.market_data_connector.subscribe([symbol], [exchange])
                
                if not success:
                    self.logger.error(f"Failed to subscribe to market data for {symbol_key}")
                    self.active_symbols.remove(symbol_key)
                    return False
        
        self.logger.info(f"Added symbol for trading: {symbol_key}")
        return True
    
    def remove_symbol(self, symbol, exchange=None):
        """
        Remove a symbol from trading.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            bool: Success status
        """
        exchange = exchange or self.config['default_exchange']
        
        # Check if in active symbols
        symbol_key = f"{symbol}:{exchange}"
        if symbol_key not in self.active_symbols:
            self.logger.info(f"Symbol {symbol_key} not in active symbols")
            return True
            
        # Close any open positions
        self._close_positions_for_symbol(symbol, exchange)
        
        # Unsubscribe from market data
        if self.engine and self.is_running:
            if self.engine.market_data_connector:
                self.engine.market_data_connector.unsubscribe([symbol], [exchange])
        
        # Remove from active symbols
        self.active_symbols.remove(symbol_key)
        
        self.logger.info(f"Removed symbol from trading: {symbol_key}")
        return True
    
    def register_strategy(self, strategy, symbols=None, timeframe='day'):
        """
        Register a trading strategy.
        
        Args:
            strategy: Strategy instance
            symbols (list): List of symbols for this strategy
            timeframe (str): Timeframe for strategy execution
            
        Returns:
            str: Strategy ID
        """
        # Check if timeframe is supported
        if timeframe not in self.config['strategy_timeframes']:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return None
            
        # Register with trading engine
        if not self.engine:
            self.logger.error("No trading engine provided")
            return None
            
        # Validate symbols
        if symbols:
            valid_symbols = []
            valid_exchanges = []
            
            for symbol in symbols:
                parts = symbol.split(':')
                
                if len(parts) == 2:
                    symbol_name, exchange = parts
                else:
                    symbol_name = symbol
                    exchange = self.config['default_exchange']
                    
                # Add to valid lists
                valid_symbols.append(symbol_name)
                valid_exchanges.append(exchange)
                
                # Add to active symbols
                self.add_symbol(symbol_name, exchange)
        else:
            # Use all active symbols
            valid_symbols = []
            valid_exchanges = []
            
            for symbol_key in self.active_symbols:
                parts = symbol_key.split(':')
                
                if len(parts) == 2:
                    symbol_name, exchange = parts
                    valid_symbols.append(symbol_name)
                    valid_exchanges.append(exchange)
        
        # Register strategy with engine
        strategy_id = self.engine.register_strategy(
            strategy=strategy,
            symbols=valid_symbols,
            timeframe=timeframe
        )
        
        if not strategy_id:
            self.logger.error("Failed to register strategy with trading engine")
            return None
            
        # Store in active strategies
        self.active_strategies[strategy_id] = {
            'strategy': strategy,
            'symbols': valid_symbols,
            'exchanges': valid_exchanges,
            'timeframe': timeframe
        }
        
        self.logger.info(f"Registered strategy {strategy_id} for {len(valid_symbols)} symbols")
        return strategy_id
    
    def unregister_strategy(self, strategy_id):
        """
        Unregister a trading strategy.
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            bool: Success status
        """
        if strategy_id not in self.active_strategies:
            self.logger.warning(f"Strategy {strategy_id} not found")
            return False
            
        # Unregister with trading engine
        if self.engine:
            self.engine.unregister_strategy(strategy_id)
            
        # Remove from active strategies
        del self.active_strategies[strategy_id]
        
        self.logger.info(f"Unregistered strategy {strategy_id}")
        return True
    
    def start(self):
        """
        Start the equity trader.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Equity trader is already running")
            return False
            
        if not self.engine:
            self.logger.error("No trading engine provided")
            return False
            
        # Start trading engine if not running
        if not self.engine.is_running:
            if not self.engine.start():
                self.logger.error("Failed to start trading engine")
                return False
        
        # Subscribe to market data for all active symbols
        if self.active_symbols:
            symbols = []
            exchanges = []
            
            for symbol_key in self.active_symbols:
                parts = symbol_key.split(':')
                
                if len(parts) == 2:
                    symbol, exchange = parts
                    symbols.append(symbol)
                    exchanges.append(exchange)
            
            if symbols and self.engine.market_data_connector:
                self.engine.market_data_connector.subscribe(symbols, exchanges)
        
        # Set data callback
        if self.engine.market_data_connector:
            self.engine.market_data_connector.set_data_callback(self._on_market_data)
        
        self.is_running = True
        self.logger.info("Equity trader started")
        
        # Start monitoring thread
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
        return True
    
    def stop(self):
        """
        Stop the equity trader.
        
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Equity trader is not running")
            return False
            
        self.is_running = False
        self.logger.info("Equity trader stopped")
        
        return True
    
    def _monitoring_loop(self):
        """
        Monitoring loop for the equity trader.
        """
        while self.is_running:
            try:
                # Check trading session
                if not self._is_trading_session():
                    # Outside trading session, check if we need to square off intraday positions
                    self._check_intraday_square_off()
                
                # Sleep for 60 seconds
                time.sleep(60)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(10)  # Sleep longer on error
    
    def _is_trading_session(self):
        """
        Check if current time is within trading session.
        
        Returns:
            bool: True if within trading session
        """
        now = datetime.now().time()
        start_time = datetime.strptime(self.config['trade_session_start'], '%H:%M:%S').time()
        end_time = datetime.strptime(self.config['trade_session_end'], '%H:%M:%S').time()
        
        return start_time <= now <= end_time
    
    def _check_intraday_square_off(self):
        """
        Check and square off intraday positions if needed.
        """
        now = datetime.now().time()
        square_off_time = datetime.strptime(self.config['intraday_square_off_time'], '%H:%M:%S').time()
        
        # Check if it's time to square off
        if now >= square_off_time:
            # Get all open positions
            if self.engine:
                positions = self.engine.get_open_positions()
                
                for position in positions:
                    # Check if intraday position
                    if position.get('product_type') == 'MIS':
                        # Close position
                        symbol = position.get('symbol')
                        exchange = position.get('exchange')
                        
                        if symbol and exchange:
                            self.logger.info(f"Square off intraday position for {symbol}:{exchange}")
                            self.engine.close_position(symbol, exchange, 'intraday_square_off')
    
    def _close_positions_for_symbol(self, symbol, exchange):
        """
        Close all positions for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
        """
        if not self.engine:
            return
            
        # Close position
        self.engine.close_position(symbol, exchange, 'symbol_removal')
    
    def _on_market_data(self, data):
        """
        Callback for market data updates.
        
        Args:
            data (dict): Market data update
        """
        try:
            symbol = data.get('symbol')
            exchange = data.get('exchange')
            
            if not symbol or not exchange:
                return
                
            # Check if symbol is active
            symbol_key = f"{symbol}:{exchange}"
            if symbol_key not in self.active_symbols:
                return
                
            # Apply filters
            if not self._apply_filters(data):
                return
                
            # We don't need to do anything else here since the trading engine
            # will process the data and execute strategies
            
        except Exception as e:
            self.logger.error(f"Error processing market data: {e}")
    
    def _apply_filters(self, data):
        """
        Apply trading filters to market data.
        
        Args:
            data (dict): Market data
            
        Returns:
            bool: True if passes filters
        """
        # Check price range
        price = data.get('last_price')
        if price and (price < self.config['min_price'] or price > self.config['max_price']):
            return False
            
        # Check volume
        volume = data.get('volume')
        if volume and volume < self.config['volume_filter']:
            return False
            
        return True
    
    def place_equity_order(self, symbol, action, quantity, order_type=None, price=None, product_type=None, exchange=None):
        """
        Place an equity order.
        
        Args:
            symbol (str): Symbol
            action (str): Action (BUY or SELL)
            quantity (int): Quantity
            order_type (str): Order type (optional)
            price (float): Price (optional)
            product_type (str): Product type (optional)
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Order result
        """
        if not self.engine:
            self.logger.error("No trading engine provided")
            return None
            
        # Default values
        order_type = order_type or self.config['default_order_type']
        product_type = product_type or self.config['default_product']
        exchange = exchange or self.config['default_exchange']
        
        # Validate parameters
        if order_type not in self.config['order_types']:
            self.logger.error(f"Unsupported order type: {order_type}")
            return None
            
        if action not in ["BUY", "SELL"]:
            self.logger.error(f"Invalid action: {action}")
            return None
            
        if quantity <= 0:
            self.logger.error(f"Invalid quantity: {quantity}")
            return None
            
        # Check if symbol is active
        symbol_key = f"{symbol}:{exchange}"
        if symbol_key not in self.active_symbols:
            self.logger.warning(f"Symbol {symbol_key} not in active symbols")
            # Add symbol automatically
            self.add_symbol(symbol, exchange)
        
        # Get current price if not provided
        if not price and self.engine.market_data_connector:
            data = self.engine.market_data_connector.get_last_price(symbol, exchange)
            
            if data:
                price = data.get('last_price')
        
        # Create order
        order = {
            'symbol': symbol,
            'exchange': exchange,
            'action': action,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'product_type': product_type
        }
        
        # Add to order queue
        if hasattr(self.engine, 'order_queue'):
            self.engine.order_queue.put(order)
            self.logger.info(f"Placed {action} order for {symbol}: {quantity} @ {price}")
            
            # Return a placeholder result since we don't have the actual order ID yet
            return {
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': quantity,
                'price': price,
                'status': 'QUEUED'
            }
        else:
            self.logger.error("Trading engine does not have order queue")
            return None
    
    def get_equity_position(self, symbol, exchange=None):
        """
        Get equity position details.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Position details
        """
        if not self.engine:
            return None
            
        exchange = exchange or self.config['default_exchange']
        return self.engine.get_position_details(symbol, exchange)
    
    def get_all_equity_positions(self):
        """
        Get all equity positions.
        
        Returns:
            list: All positions
        """
        if not self.engine:
            return []
            
        return self.engine.get_open_positions()
    
    def close_equity_position(self, symbol, exchange=None, reason='manual_exit'):
        """
        Close an equity position.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            reason (str): Exit reason
            
        Returns:
            bool: Success status
        """
        if not self.engine:
            return False
            
        exchange = exchange or self.config['default_exchange']
        return self.engine.close_position(symbol, exchange, reason)
    
    def close_all_equity_positions(self, reason='manual_exit'):
        """
        Close all equity positions.
        
        Args:
            reason (str): Exit reason
            
        Returns:
            int: Number of positions closed
        """
        if not self.engine:
            return 0
            
        return self.engine.close_all_positions(reason)
    
    def get_equity_performance(self):
        """
        Get equity trading performance.
        
        Returns:
            dict: Performance metrics
        """
        if not self.engine:
            return {}
            
        return self.engine.get_performance_metrics()
    
    def get_historical_equity_data(self, symbol, timeframe='day', days=30, exchange=None):
        """
        Get historical equity data.
        
        Args:
            symbol (str): Symbol
            timeframe (str): Timeframe
            days (int): Number of days
            exchange (str): Exchange (optional)
            
        Returns:
            DataFrame: Historical data
        """
        try:
            if not self.engine or not self.engine.market_data_connector:
                return None
                
            exchange = exchange or self.config['default_exchange']
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Get historical data
            data = self.engine.market_data_connector.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                from_date=from_date,
                to_date=to_date
            )
            
            if not data:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Rename columns to standard format
            if 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
                df.drop('date', axis=1, inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None