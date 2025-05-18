# realtime/futures_trader.py
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class FuturesTrader:
    """
    Specialized trading module for futures trading.
    """
    
    def __init__(self, trading_engine, db_connector, logger=None):
        """
        Initialize the futures trader.
        
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
        self.futures_info = {}  # Store contract details
        
        # Configuration
        self.config = {
            'max_symbols': 30,  # Maximum number of futures to trade
            'order_types': ['LIMIT', 'MARKET', 'SL', 'SL-M'],  # Allowed order types
            'default_order_type': 'LIMIT',
            'default_product': 'NRML',  # NRML for futures
            'default_exchange': 'NFO',  # NFO for futures
            'max_slippage': 0.05,  # Maximum slippage in percentage
            'price_tick_size': 0.05,  # Price tick size
            'use_limit_orders': True,  # Use limit orders instead of market orders
            'limit_order_timeout': 60,  # Timeout for limit orders in seconds
            'volume_filter': 50000,  # Minimum volume filter
            'min_price': 50.0,  # Minimum price filter
            'max_price': 50000.0,  # Maximum price filter
            'trade_session_start': '09:15:00',  # Trading session start time
            'trade_session_end': '15:30:00',  # Trading session end time
            'max_positions': 5,  # Maximum number of positions
            'expiry_buffer_days': 5,  # Minimum days before expiry to trade
            'strategy_timeframes': ['1min', '5min', '15min', '30min', '1hour', 'day'],  # Supported timeframes
            'max_leverage': 5.0,  # Maximum leverage
            'margin_buffer': 1.5  # Margin buffer (multiplier for required margin)
        }
        
        # Initialize
        self._initialize()
        
        self.logger.info("Futures trader initialized")
    
    def set_config(self, config):
        """
        Set configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated futures trader configuration: {self.config}")
    
    def _initialize(self):
        """
        Initialize the futures trader.
        """
        # Register with trading engine
        if self.engine:
            # If engine is already running, we need to make sure it has our callback
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                self.engine.market_data_connector.set_data_callback(self._on_market_data)
                
            # Load futures contracts
            self._load_futures_contracts()
        else:
            self.logger.warning("No trading engine provided")
    
    def _load_futures_contracts(self):
        """
        Load futures contract details.
        """
        try:
            # Get all futures contracts from database
            futures = self.db.futures_contracts_collection.find({
                'expiry_date': {'$gte': datetime.now()}  # Only active contracts
            })
            
            for contract in futures:
                symbol = contract.get('symbol')
                exchange = contract.get('exchange')
                expiry = contract.get('expiry_date')
                
                if symbol and exchange and expiry:
                    # Store in futures info
                    key = f"{symbol}:{exchange}"
                    self.futures_info[key] = contract
                    
            self.logger.info(f"Loaded {len(self.futures_info)} futures contracts")
            
        except Exception as e:
            self.logger.error(f"Error loading futures contracts: {e}")
    
    def add_futures_contract(self, symbol, exchange=None, expiry=None):
        """
        Add a futures contract for trading.
        
        Args:
            symbol (str): Symbol (e.g., 'NIFTY', 'RELIANCE')
            exchange (str): Exchange (optional)
            expiry (str/datetime): Expiry date (optional)
            
        Returns:
            bool: Success status
        """
        exchange = exchange or self.config['default_exchange']
        
        # If expiry is a string, convert to datetime
        if isinstance(expiry, str):
            try:
                expiry = datetime.strptime(expiry, '%Y-%m-%d')
            except Exception as e:
                self.logger.error(f"Invalid expiry date format: {e}")
                return False
        
        # Get active futures contract
        contract = self._get_futures_contract(symbol, exchange, expiry)
        
        if not contract:
            self.logger.error(f"Futures contract not found for {symbol}")
            return False
            
        # Extract contract symbol
        contract_symbol = contract.get('tradingsymbol')
        
        if not contract_symbol:
            self.logger.error(f"No trading symbol for futures contract {symbol}")
            return False
            
        # Check if already added
        symbol_key = f"{contract_symbol}:{exchange}"
        if symbol_key in self.active_symbols:
            self.logger.info(f"Futures contract {symbol_key} already added")
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
                success = self.engine.market_data_connector.subscribe([contract_symbol], [exchange])
                
                if not success:
                    self.logger.error(f"Failed to subscribe to market data for {symbol_key}")
                    self.active_symbols.remove(symbol_key)
                    return False
        
        self.logger.info(f"Added futures contract for trading: {symbol_key}")
        return True
    
    def _get_futures_contract(self, symbol, exchange, expiry=None):
        """
        Get futures contract details.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            expiry (datetime): Expiry date (optional)
            
        Returns:
            dict: Contract details
        """
        try:
            # Query for active contracts
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'instrument_type': 'FUT',
                'expiry_date': {'$gte': datetime.now()}  # Only active contracts
            }
            
            # Add expiry if provided
            if expiry:
                query['expiry_date'] = expiry
                
            # Sort by expiry (closest first)
            cursor = self.db.futures_contracts_collection.find(query).sort('expiry_date', 1)
            
            # Get first contract (closest expiry)
            contract = next(cursor, None)
            
            if not contract:
                self.logger.error(f"No active futures contract found for {symbol}")
                return None
                
            # Check expiry buffer
            if contract.get('expiry_date'):
                days_to_expiry = (contract['expiry_date'] - datetime.now()).days
                
                if days_to_expiry < self.config['expiry_buffer_days']:
                    self.logger.warning(f"Futures contract {symbol} is close to expiry ({days_to_expiry} days)")
                    
                    # Try to get next contract
                    next_contract = next(cursor, None)
                    
                    if next_contract:
                        self.logger.info(f"Using next futures contract for {symbol}")
                        return next_contract
            
            return contract
            
        except Exception as e:
            self.logger.error(f"Error getting futures contract: {e}")
            return None
    
    def remove_futures_contract(self, symbol, exchange=None):
        """
        Remove a futures contract from trading.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            bool: Success status
        """
        exchange = exchange or self.config['default_exchange']
        
        # Find all active symbols for this underlying
        remove_keys = []
        
        for symbol_key in self.active_symbols:
            parts = symbol_key.split(':')
            
            if len(parts) == 2:
                trading_symbol, symbol_exchange = parts
                
                # Check if this is the symbol to remove
                if symbol_exchange == exchange and symbol in trading_symbol:
                    remove_keys.append(symbol_key)
        
        if not remove_keys:
            self.logger.info(f"No active futures contracts found for {symbol}")
            return True
            
        # Remove each contract
        for key in remove_keys:
            parts = key.split(':')
            
            if len(parts) == 2:
                trading_symbol, symbol_exchange = parts
                
                # Close any open positions
                self._close_positions_for_symbol(trading_symbol, symbol_exchange)
                
                # Unsubscribe from market data
                if self.engine and self.is_running:
                    if self.engine.market_data_connector:
                        self.engine.market_data_connector.unsubscribe([trading_symbol], [symbol_exchange])
                
                # Remove from active symbols
                self.active_symbols.remove(key)
                
                self.logger.info(f"Removed futures contract from trading: {key}")
        
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
                    
                # Check if this is a futures contract or underlying
                contract = self._get_futures_contract(symbol_name, exchange)
                
                if contract:
                    # This is an underlying, use the contract symbol
                    contract_symbol = contract.get('tradingsymbol')
                    
                    if contract_symbol:
                        # Add to valid lists
                        valid_symbols.append(contract_symbol)
                        valid_exchanges.append(exchange)
                        
                        # Add to active symbols
                        self.add_futures_contract(symbol_name, exchange)
                else:
                    # This might be a contract symbol directly
                    valid_symbols.append(symbol_name)
                    valid_exchanges.append(exchange)
                    
                    # Add to active symbols
                    symbol_key = f"{symbol_name}:{exchange}"
                    if symbol_key not in self.active_symbols:
                        self.active_symbols.add(symbol_key)
                        
                        # Subscribe to market data
                        if self.engine and self.is_running:
                            if self.engine.market_data_connector:
                                self.engine.market_data_connector.subscribe([symbol_name], [exchange])
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
        Start the futures trader.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Futures trader is already running")
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
        self.logger.info("Futures trader started")
        
        # Start monitoring thread
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
        return True
    
    def stop(self):
        """
        Stop the futures trader.
        
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Futures trader is not running")
            return False
            
        self.is_running = False
        self.logger.info("Futures trader stopped")
        
        return True
    
    def _monitoring_loop(self):
        """
        Monitoring loop for the futures trader.
        """
        while self.is_running:
            try:
                # Check trading session
                if not self._is_trading_session():
                    # Outside trading session, check if we need to close positions
                    pass
                
                # Check expiry of active contracts
                self._check_contract_expiry()
                
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
    
    def _check_contract_expiry(self):
        """
        Check for contracts nearing expiry.
        """
        # Refresh futures info
        self._load_futures_contracts()
        
        # Check each active symbol
        for symbol_key in list(self.active_symbols):
            parts = symbol_key.split(':')
            
            if len(parts) != 2:
                continue
                
            symbol, exchange = parts
            
            # Check contract details
            contract = self.futures_info.get(symbol_key)
            
            if not contract:
                continue
                
            expiry = contract.get('expiry_date')
            
            if not expiry:
                continue
                
            # Check days to expiry
            days_to_expiry = (expiry - datetime.now()).days
            
            if days_to_expiry < self.config['expiry_buffer_days']:
                self.logger.warning(f"Futures contract {symbol_key} nearing expiry ({days_to_expiry} days)")
                
                # Check if we have positions
                if self.engine:
                    position = self.engine.get_position_details(symbol, exchange)
                    
                    if position:
                        self.logger.warning(f"Closing position for expiring contract {symbol_key}")
                        self.engine.close_position(symbol, exchange, 'contract_expiry')
                
                # Roll over to next contract
                underlying = contract.get('symbol')
                
                if underlying:
                    # Try to add next contract
                    self.logger.info(f"Rolling over to next contract for {underlying}")
                    self.remove_futures_contract(symbol, exchange)
                    self.add_futures_contract(underlying, exchange)
    
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
    
    def place_futures_order(self, symbol, action, quantity, order_type=None, price=None, exchange=None):
        """
        Place a futures order.
        
        Args:
            symbol (str): Symbol (can be underlying or contract symbol)
            action (str): Action (BUY or SELL)
            quantity (int): Quantity
            order_type (str): Order type (optional)
            price (float): Price (optional)
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Order result
        """
        if not self.engine:
            self.logger.error("No trading engine provided")
            return None
            
        # Default values
        order_type = order_type or self.config['default_order_type']
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
        
        # Check if symbol is a contract or underlying
        contract_symbol = symbol
        
        # If it's an underlying, get the contract
        if not any(f"{symbol}:" in s for s in self.active_symbols):
            contract = self._get_futures_contract(symbol, exchange)
            
            if not contract:
                self.logger.error(f"Futures contract not found for {symbol}")
                return None
                
            contract_symbol = contract.get('tradingsymbol')
            
            if not contract_symbol:
                self.logger.error(f"No trading symbol for futures contract {symbol}")
                return None
        
        # Check if symbol is active
        symbol_key = f"{contract_symbol}:{exchange}"
        if symbol_key not in self.active_symbols:
            self.logger.warning(f"Symbol {symbol_key} not in active symbols")
            # Add symbol automatically
            self.add_futures_contract(symbol, exchange)
        
        # Get current price if not provided
        if not price and self.engine.market_data_connector:
            data = self.engine.market_data_connector.get_last_price(contract_symbol, exchange)
            
            if data:
                price = data.get('last_price')
        
        # Create order
        order = {
            'symbol': contract_symbol,
            'exchange': exchange,
            'action': action,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'product_type': self.config['default_product']
        }
        
        # Add to order queue
        if hasattr(self.engine, 'order_queue'):
            self.engine.order_queue.put(order)
            self.logger.info(f"Placed {action} futures order for {contract_symbol}: {quantity} @ {price}")
            
            # Return a placeholder result since we don't have the actual order ID yet
            return {
                'symbol': contract_symbol,
                'exchange': exchange,
                'action': action,
                'quantity': quantity,
                'price': price,
                'status': 'QUEUED'
            }
        else:
            self.logger.error("Trading engine does not have order queue")
            return None
    
    def get_futures_position(self, symbol, exchange=None):
        """
        Get futures position details.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Position details
        """
        if not self.engine:
            return None
            
        exchange = exchange or self.config['default_exchange']
        
        # If it's an underlying, get the contract symbol
        if not any(f"{symbol}:" in s for s in self.active_symbols):
            contract = self._get_futures_contract(symbol, exchange)
            
            if not contract:
                return None
                
            symbol = contract.get('tradingsymbol', symbol)
        
        return self.engine.get_position_details(symbol, exchange)
    
    def get_all_futures_positions(self):
        """
        Get all futures positions.
        
        Returns:
            list: All futures positions
        """
        if not self.engine:
            return []
            
        # Get all positions
        all_positions = self.engine.get_open_positions()
        
        # Filter for futures positions
        futures_positions = []
        
        for position in all_positions:
            symbol = position.get('symbol')
            exchange = position.get('exchange')
            
            if not symbol or not exchange:
                continue
                
            symbol_key = f"{symbol}:{exchange}"
            
            # Check if this is a futures position
            if symbol_key in self.active_symbols:
                futures_positions.append(position)
        
        return futures_positions
    
    def close_futures_position(self, symbol, exchange=None, reason='manual_exit'):
        """
        Close a futures position.
        
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
        
        # If it's an underlying, get the contract symbol
        if not any(f"{symbol}:" in s for s in self.active_symbols):
            contract = self._get_futures_contract(symbol, exchange)
            
            if not contract:
                return False
                
            symbol = contract.get('tradingsymbol', symbol)
        
        return self.engine.close_position(symbol, exchange, reason)
    
    def close_all_futures_positions(self, reason='manual_exit'):
        """
        Close all futures positions.
        
        Args:
            reason (str): Exit reason
            
        Returns:
            int: Number of positions closed
        """
        closed_count = 0
        
        # Get all futures positions
        positions = self.get_all_futures_positions()
        
        # Close each position
        for position in positions:
            symbol = position.get('symbol')
            exchange = position.get('exchange')
            
            if symbol and exchange:
                if self.close_futures_position(symbol, exchange, reason):
                    closed_count += 1
        
        return closed_count
    
    def get_futures_performance(self):
        """
        Get futures trading performance.
        
        Returns:
            dict: Performance metrics
        """
        if not self.engine:
            return {}
            
        # Get all performance metrics
        all_metrics = self.engine.get_performance_metrics()
        
        # Filter trade history for futures trades
        if hasattr(self.engine, 'trade_history'):
            futures_trades = []
            
            for trade in self.engine.trade_history:
                symbol = trade.get('symbol')
                exchange = trade.get('exchange')
                
                if not symbol or not exchange:
                    continue
                    
                symbol_key = f"{symbol}:{exchange}"
                
                # Check if this was a futures trade
                if any(symbol in s for s in self.active_symbols) or exchange == self.config['default_exchange']:
                    futures_trades.append(trade)
            
            # Calculate futures-specific metrics
            if futures_trades:
                total_trades = len(futures_trades)
                winning_trades = [t for t in futures_trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in futures_trades if t.get('profit_loss', 0) <= 0]
                
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
                
                total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Update metrics
                all_metrics.update({
                    'futures_trades': total_trades,
                    'futures_win_rate': win_rate,
                    'futures_profit_factor': profit_factor,
                    'futures_total_profit': total_profit,
                    'futures_total_loss': total_loss
                })
        
        return all_metrics
    
    def get_historical_futures_data(self, symbol, timeframe='day', days=30, exchange=None):
        """
        Get historical futures data.
        
        Args:
            symbol (str): Symbol (underlying or contract)
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
            
            # If it's an underlying, get the contract symbol
            if not any(f"{symbol}:" in s for s in self.active_symbols):
                contract = self._get_futures_contract(symbol, exchange)
                
                if not contract:
                    return None
                    
                symbol = contract.get('tradingsymbol', symbol)
            
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
                    