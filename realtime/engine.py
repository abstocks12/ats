# realtime/engine.py
import logging
import threading
import time
import queue
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

class TradingEngine:
    """
    Core trading engine for real-time strategy execution.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the trading engine.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Trading state
        self.is_running = False
        self.trading_thread = None
        self.mode = "paper"  # "paper" or "live"
        
        # Data and event handling
        self.data_queue = queue.Queue()
        self.order_queue = queue.Queue()
        self.strategy_registry = {}
        self.active_symbols = set()
        self.market_data_cache = {}
        self.position_cache = {}
        
        # Performance tracking
        self.equity_curve = []
        self.trade_history = []
        
        # Configuration
        self.config = {
            'update_interval': 1,  # seconds
            'max_positions': 10,
            'risk_per_trade': 0.02,  # 2% risk per trade
            'max_risk_multiplier': 3,  # Maximum risk multiplier for high-confidence trades
            'default_exchange': 'NSE',
            'default_product': 'CNC',  # CNC for delivery, MIS for intraday
            'default_order_type': 'LIMIT',
            'slippage_buffer': 0.1,  # % buffer for limit orders
            'use_stoploss': True,
            'use_target': True
        }
        
        # Initialize integrations
        self.market_data_connector = None
        self.order_executor = None
        
        self.logger.info("Trading engine initialized")
    
    def set_config(self, config):
        """
        Set engine configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated engine configuration: {self.config}")
    
    def set_mode(self, mode):
        """
        Set trading mode (paper or live).
        
        Args:
            mode (str): Trading mode ("paper" or "live")
        """
        if mode not in ["paper", "live"]:
            self.logger.error(f"Invalid trading mode: {mode}")
            return
            
        self.mode = mode
        self.logger.info(f"Trading mode set to: {mode}")
    
    def set_market_data_connector(self, connector):
        """
        Set market data connector.
        
        Args:
            connector: Market data connector instance
        """
        self.market_data_connector = connector
        self.logger.info(f"Market data connector set: {connector.__class__.__name__}")
    
    def set_order_executor(self, executor):
        """
        Set order executor.
        
        Args:
            executor: Order executor instance
        """
        self.order_executor = executor
        self.logger.info(f"Order executor set: {executor.__class__.__name__}")
    
    def register_strategy(self, strategy, symbols=None, timeframe="1min"):
        """
        Register a trading strategy.
        
        Args:
            strategy: Strategy instance
            symbols (list): List of symbols for this strategy
            timeframe (str): Timeframe for strategy execution
            
        Returns:
            str: Strategy ID
        """
        # Generate strategy ID
        strategy_id = f"{strategy.__class__.__name__}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Register strategy
        self.strategy_registry[strategy_id] = {
            'strategy': strategy,
            'symbols': symbols or [],
            'timeframe': timeframe,
            'last_run': None,
            'is_active': True
        }
        
        # Add symbols to active symbols set
        if symbols:
            self.active_symbols.update(symbols)
        
        self.logger.info(f"Registered strategy {strategy_id} for {len(symbols or [])} symbols")
        
        return strategy_id
    
    def unregister_strategy(self, strategy_id):
        """
        Unregister a trading strategy.
        
        Args:
            strategy_id (str): Strategy ID
            
        Returns:
            bool: Success status
        """
        if strategy_id not in self.strategy_registry:
            self.logger.warning(f"Strategy {strategy_id} not found")
            return False
            
        # Remove strategy
        self.strategy_registry.pop(strategy_id)
        
        # Recalculate active symbols
        self.active_symbols = set()
        for reg in self.strategy_registry.values():
            if reg['symbols']:
                self.active_symbols.update(reg['symbols'])
        
        self.logger.info(f"Unregistered strategy {strategy_id}")
        
        return True
    
    def update_strategy_symbols(self, strategy_id, symbols):
        """
        Update symbols for a registered strategy.
        
        Args:
            strategy_id (str): Strategy ID
            symbols (list): List of symbols
            
        Returns:
            bool: Success status
        """
        if strategy_id not in self.strategy_registry:
            self.logger.warning(f"Strategy {strategy_id} not found")
            return False
            
        # Update symbols
        self.strategy_registry[strategy_id]['symbols'] = symbols
        
        # Recalculate active symbols
        self.active_symbols = set()
        for reg in self.strategy_registry.values():
            if reg['symbols']:
                self.active_symbols.update(reg['symbols'])
        
        self.logger.info(f"Updated symbols for strategy {strategy_id}: {symbols}")
        
        return True
    
    def start(self):
        """
        Start the trading engine.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Trading engine is already running")
            return False
            
        if not self.market_data_connector:
            self.logger.error("No market data connector set")
            return False
            
        if not self.order_executor and self.mode == "live":
            self.logger.error("No order executor set for live trading")
            return False
            
        # Start trading thread
        self.is_running = True
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()
        
        self.logger.info(f"Trading engine started in {self.mode} mode")
        
        return True
    
    def stop(self, wait=True):
        """
        Stop the trading engine.
        
        Args:
            wait (bool): Wait for trading thread to complete
            
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Trading engine is not running")
            return False
            
        # Stop trading thread
        self.is_running = False
        
        if wait and self.trading_thread:
            self.trading_thread.join(timeout=30)
            
        self.logger.info("Trading engine stopped")
        
        return True
    
    def _trading_loop(self):
        """
        Main trading loop.
        """
        # Initialize market data connection
        if not self._initialize_market_data():
            self.logger.error("Failed to initialize market data")
            self.is_running = False
            return
        
        # Initialize order execution
        if self.mode == "live" and not self._initialize_order_execution():
            self.logger.error("Failed to initialize order execution")
            self.is_running = False
            return
        
        self.logger.info("Trading loop started")
        
        # Main loop
        while self.is_running:
            try:
                # Process data queue
                self._process_data_queue()
                
                # Execute strategy run cycle
                self._execute_strategy_cycle()
                
                # Process order queue
                self._process_order_queue()
                
                # Process position updates
                self._update_positions()
                
                # Sleep for update interval
                time.sleep(self.config['update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(5)  # Sleep longer on error
        
        # Clean up
        self._cleanup()
        
        self.logger.info("Trading loop stopped")
    
    def _initialize_market_data(self):
        """
        Initialize market data connection.
        
        Returns:
            bool: Success status
        """
        try:
            # Subscribe to market data for all active symbols
            if self.active_symbols:
                symbols = list(self.active_symbols)
                exchanges = [self.config['default_exchange']] * len(symbols)
                
                self.logger.info(f"Subscribing to market data for {len(symbols)} symbols")
                
                success = self.market_data_connector.subscribe(symbols, exchanges)
                
                if not success:
                    self.logger.error("Failed to subscribe to market data")
                    return False
            
            # Set data callback
            self.market_data_connector.set_data_callback(self._on_market_data)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing market data: {e}")
            return False
    
    def _initialize_order_execution(self):
        """
        Initialize order execution.
        
        Returns:
            bool: Success status
        """
        try:
            # Test order connection
            if not self.order_executor.is_connected():
                self.logger.error("Order executor is not connected")
                return False
                
            # Set order callback
            self.order_executor.set_order_callback(self._on_order_update)
            
            # Test account status
            account_info = self.order_executor.get_account_info()
            
            if not account_info:
                self.logger.error("Failed to get account information")
                return False
                
            self.logger.info(f"Account balance: {account_info.get('balance', 'Unknown')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing order execution: {e}")
            return False
    
    def _on_market_data(self, data):
        """
        Callback for market data updates.
        
        Args:
            data (dict): Market data update
        """
        try:
            # Queue data for processing
            self.data_queue.put(data)
        except Exception as e:
            self.logger.error(f"Error in market data callback: {e}")
    
    def _on_order_update(self, data):
        """
        Callback for order updates.
        
        Args:
            data (dict): Order update
        """
        try:
            # Process order update
            order_id = data.get('order_id')
            status = data.get('status')
            
            if not order_id:
                return
                
            self.logger.info(f"Order update: {order_id} - {status}")
            
            # Update order in database
            self._update_order_in_db(data)
            
            # Update positions if order is filled
            if status == "COMPLETE":
                self._update_position_on_fill(data)
                
        except Exception as e:
            self.logger.error(f"Error in order update callback: {e}")
    
    def _process_data_queue(self):
        """
        Process market data queue.
        """
        processed = 0
        
        # Process up to 100 items at once to avoid blocking
        while not self.data_queue.empty() and processed < 100:
            try:
                data = self.data_queue.get_nowait()
                
                if not data:
                    continue
                    
                # Extract data fields
                symbol = data.get('symbol')
                exchange = data.get('exchange')
                
                if not symbol:
                    continue
                
                # Update market data cache
                symbol_key = f"{symbol}:{exchange}" if exchange else symbol
                self.market_data_cache[symbol_key] = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'last_price': data.get('last_price'),
                    'bid': data.get('bid'),
                    'ask': data.get('ask'),
                    'volume': data.get('volume'),
                    'timestamp': data.get('timestamp') or datetime.now(),
                    'open': data.get('open'),
                    'high': data.get('high'),
                    'low': data.get('low'),
                    'close': data.get('close')
                }
                
                # Mark as processed
                processed += 1
                self.data_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing market data: {e}")
                self.data_queue.task_done()
        
        if processed > 0:
            self.logger.debug(f"Processed {processed} market data updates")
    
    def _execute_strategy_cycle(self):
        """
        Execute one cycle of strategy evaluation.
        """
        now = datetime.now()
        executed = 0
        
        # Process each registered strategy
        for strategy_id, reg in self.strategy_registry.items():
            try:
                if not reg['is_active']:
                    continue
                    
                strategy = reg['strategy']
                symbols = reg['symbols']
                timeframe = reg['timeframe']
                last_run = reg['last_run']
                
                # Check if it's time to run this strategy
                if last_run and self._should_skip_run(last_run, now, timeframe):
                    continue
                
                # Run strategy for each symbol
                for symbol in symbols:
                    try:
                        # Get market data for this symbol
                        market_data = self._get_symbol_data(symbol)
                        
                        if not market_data:
                            continue
                            
                        # Run strategy
                        signals = strategy.generate_signals(symbol, market_data)
                        
                        if signals:
                            # Process signals
                            self._process_signals(signals, strategy_id)
                            
                    except Exception as e:
                        self.logger.error(f"Error executing strategy for {symbol}: {e}")
                
                # Update last run time
                self.strategy_registry[strategy_id]['last_run'] = now
                executed += 1
                
            except Exception as e:
                self.logger.error(f"Error executing strategy {strategy_id}: {e}")
        
        if executed > 0:
            self.logger.debug(f"Executed {executed} strategies")
    
    def _should_skip_run(self, last_run, now, timeframe):
        """
        Check if strategy run should be skipped based on timeframe.
        
        Args:
            last_run (datetime): Last run time
            now (datetime): Current time
            timeframe (str): Strategy timeframe
            
        Returns:
            bool: True if should skip, False otherwise
        """
        if not last_run:
            return False
            
        # Calculate minimum time between runs
        if timeframe == "1min":
            min_interval = timedelta(seconds=60)
        elif timeframe == "5min":
            min_interval = timedelta(seconds=300)
        elif timeframe == "15min":
            min_interval = timedelta(seconds=900)
        elif timeframe == "30min":
            min_interval = timedelta(seconds=1800)
        elif timeframe == "1hour":
            min_interval = timedelta(seconds=3600)
        elif timeframe == "day":
            min_interval = timedelta(days=1)
        else:
            min_interval = timedelta(seconds=60)  # Default to 1 minute
            
        # Add a small buffer (10% of interval)
        buffer = min_interval.total_seconds() * 0.1
        min_interval += timedelta(seconds=buffer)
        
        return (now - last_run) < min_interval
    
    def _get_symbol_data(self, symbol, exchange=None):
        """
        Get market data for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Market data
        """
        exchange = exchange or self.config['default_exchange']
        symbol_key = f"{symbol}:{exchange}"
        
        # Check cache first
        if symbol_key in self.market_data_cache:
            return self.market_data_cache[symbol_key]
            
        # Not in cache, try to get from market data connector
        try:
            data = self.market_data_connector.get_last_price(symbol, exchange)
            
            if data:
                # Update cache
                self.market_data_cache[symbol_key] = data
                return data
                
        except Exception as e:
            self.logger.error(f"Error getting data for {symbol_key}: {e}")
            
        return None
    
    def _process_signals(self, signals, strategy_id):
        """
        Process trading signals.
        
        Args:
            signals (list): List of trading signals
            strategy_id (str): Strategy ID
        """
        if not signals:
            return
            
        for signal in signals:
            try:
                # Extract signal parameters
                action = signal.get('action')
                symbol = signal.get('symbol')
                exchange = signal.get('exchange') or self.config['default_exchange']
                
                if not action or not symbol:
                    continue
                    
                # Check if we have an existing position
                position_key = f"{symbol}:{exchange}"
                has_position = position_key in self.position_cache
                
                # Entry signal
                if action in ["BUY", "SELL"] and not has_position:
                    # Check if we can take a new position
                    if len(self.position_cache) >= self.config['max_positions']:
                        self.logger.warning(f"Maximum positions reached, skipping {action} signal for {symbol}")
                        continue
                        
                    # Create new entry order
                    self._create_entry_order(signal, strategy_id)
                    
                # Exit signal
                elif action == "EXIT" and has_position:
                    # Create exit order
                    self._create_exit_order(signal, strategy_id)
                    
                # Update signal for existing position
                elif has_position:
                    # Update position
                    self._update_position_params(signal, strategy_id)
                    
            except Exception as e:
                self.logger.error(f"Error processing signal: {e}")
    
    def _create_entry_order(self, signal, strategy_id):
        """
        Create entry order from signal.
        
        Args:
            signal (dict): Trading signal
            strategy_id (str): Strategy ID
        """
        try:
            # Extract signal parameters
            symbol = signal.get('symbol')
            exchange = signal.get('exchange') or self.config['default_exchange']
            action = signal.get('action')
            entry_price = signal.get('entry_price')
            stop_loss = signal.get('stop_loss')
            target = signal.get('target')
            confidence = signal.get('confidence', 0.5)
            timeframe = signal.get('timeframe', 'day')
            expiry = signal.get('expiry')
            
            # Get current market price if entry price not specified
            if not entry_price:
                market_data = self._get_symbol_data(symbol, exchange)
                
                if not market_data:
                    self.logger.error(f"No market data for {symbol}, cannot create entry order")
                    return
                    
                entry_price = market_data.get('last_price')
                
                if not entry_price:
                    self.logger.error(f"No price data for {symbol}, cannot create entry order")
                    return
            
            # Calculate position size
            position_size = self._calculate_position_size(
                symbol=symbol, 
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence,
                action=action
            )
            
            if position_size <= 0:
                self.logger.warning(f"Invalid position size for {symbol}, skipping order")
                return
                
            # Create order
            order_type = signal.get('order_type') or self.config['default_order_type']
            product_type = signal.get('product_type') or self.config['default_product']
            
            # For limit orders, add slippage buffer
            if order_type == "LIMIT":
                buffer = entry_price * self.config['slippage_buffer'] / 100
                
                if action == "BUY":
                    entry_price += buffer  # Buy at slightly higher price to ensure execution
                else:
                    entry_price -= buffer  # Sell at slightly lower price to ensure execution
            
            # Create order object
            order = {
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': position_size,
                'price': entry_price,
                'order_type': order_type,
                'product_type': product_type,
                'stop_loss': stop_loss,
                'target': target,
                'strategy_id': strategy_id,
                'confidence': confidence,
                'timeframe': timeframe,
                'expiry': expiry,
                'timestamp': datetime.now()
            }
            
            # Add to order queue
            self.order_queue.put(order)
            
            self.logger.info(f"Created {action} order for {symbol} - {position_size} @ {entry_price}")
            
        except Exception as e:
            self.logger.error(f"Error creating entry order: {e}")
    
    def _create_exit_order(self, signal, strategy_id):
        """
        Create exit order from signal.
        
        Args:
            signal (dict): Trading signal
            strategy_id (str): Strategy ID
        """
        try:
            # Extract signal parameters
            symbol = signal.get('symbol')
            exchange = signal.get('exchange') or self.config['default_exchange']
            exit_price = signal.get('exit_price')
            
            # Get position
            position_key = f"{symbol}:{exchange}"
            
            if position_key not in self.position_cache:
                self.logger.warning(f"No position found for {symbol}, cannot create exit order")
                return
                
            position = self.position_cache[position_key]
            
            # Get current market price if exit price not specified
            if not exit_price:
                market_data = self._get_symbol_data(symbol, exchange)
                
                if not market_data:
                    self.logger.error(f"No market data for {symbol}, cannot create exit order")
                    return
                    
                exit_price = market_data.get('last_price')
                
                if not exit_price:
                    self.logger.error(f"No price data for {symbol}, cannot create exit order")
                    return
            
            # Determine exit action (opposite of entry)
            exit_action = "SELL" if position['action'] == "BUY" else "BUY"
            
            # Create order
            order_type = signal.get('order_type') or self.config['default_order_type']
            product_type = position.get('product_type') or self.config['default_product']
            
            # Create order object
            order = {
                'symbol': symbol,
                'exchange': exchange,
                'action': exit_action,
                'quantity': position['quantity'],
                'price': exit_price,
                'order_type': order_type,
                'product_type': product_type,
                'strategy_id': strategy_id,
                'position_id': position.get('position_id'),
                'exit_reason': signal.get('reason', 'strategy_exit'),
                'timestamp': datetime.now()
            }
            
            # Add to order queue
            self.order_queue.put(order)
            
            self.logger.info(f"Created exit order for {symbol} - {position['quantity']} @ {exit_price}")
            
        except Exception as e:
            self.logger.error(f"Error creating exit order: {e}")
    
    def _update_position_params(self, signal, strategy_id):
        """
        Update position parameters based on signal.
        
        Args:
            signal (dict): Trading signal
            strategy_id (str): Strategy ID
        """
        try:
            # Extract signal parameters
            symbol = signal.get('symbol')
            exchange = signal.get('exchange') or self.config['default_exchange']
            stop_loss = signal.get('stop_loss')
            target = signal.get('target')
            
            # Get position
            position_key = f"{symbol}:{exchange}"
            
            if position_key not in self.position_cache:
                return
                
            position = self.position_cache[position_key]
            position_id = position.get('position_id')
            
            if not position_id:
                return
                
            # Update parameters
            updates = {}
            
            if stop_loss is not None and self.config['use_stoploss']:
                updates['stop_loss'] = stop_loss
                
            if target is not None and self.config['use_target']:
                updates['target'] = target
                
            if not updates:
                return
                
            # Update in database
            self._update_position_in_db(position_id, updates)
            
            # Update in cache
            for key, value in updates.items():
                position[key] = value
                
            self.logger.info(f"Updated position parameters for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error updating position parameters: {e}")
    
    def _calculate_position_size(self, symbol, entry_price, stop_loss, confidence, action):
        """
        Calculate position size based on risk parameters.
        
        Args:
            symbol (str): Symbol
            entry_price (float): Entry price
            stop_loss (float): Stop loss price
            confidence (float): Signal confidence (0-1)
            action (str): Action (BUY or SELL)
            
        Returns:
            int: Position size
        """
        # Default position size
        default_size = 1
        
        try:
            # Get account balance
            account_balance = self._get_account_balance()
            
            if not account_balance or account_balance <= 0:
                return default_size
                
            # Calculate risk amount
            risk_pct = self.config['risk_per_trade']
            
            # Adjust risk based on confidence
            if confidence > 0.5:
                # Increase risk for high-confidence signals
                confidence_factor = 1 + (confidence - 0.5) * 2 * (self.config['max_risk_multiplier'] - 1)
                risk_pct *= confidence_factor
            elif confidence < 0.5:
                # Decrease risk for low-confidence signals
                confidence_factor = confidence * 2
                risk_pct *= confidence_factor
            
            risk_amount = account_balance * risk_pct
            
            # Calculate stop loss distance
            if stop_loss and entry_price:
                if action == "BUY":
                    # For long positions
                    if stop_loss >= entry_price:
                        self.logger.warning(f"Invalid stop loss for long position: {stop_loss} >= {entry_price}")
                        return default_size
                        
                    stop_distance = entry_price - stop_loss
                else:
                    # For short positions
                    if stop_loss <= entry_price:
                        self.logger.warning(f"Invalid stop loss for short position: {stop_loss} <= {entry_price}")
                        return default_size
                        
                    stop_distance = stop_loss - entry_price
                
                # Calculate position size
                position_size = risk_amount / stop_distance
                
                # Convert to number of shares
                shares = int(position_size / entry_price)
                
                # Ensure minimum size
                return max(1, shares)
            else:
                # No stop loss specified, use fixed position size based on price
                return max(1, int(risk_amount / entry_price))
                
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return default_size
    
    def _get_account_balance(self):
        """
        Get account balance.
        
        Returns:
            float: Account balance
        """
        try:
            if self.mode == "live" and self.order_executor:
                account_info = self.order_executor.get_account_info()
                return account_info.get('balance', 0)
            else:
                # For paper trading, use a fixed balance or get from database
                return 100000  # Default paper trading balance
                
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 0
    
    def _process_order_queue(self):
        """
        Process order queue.
        """
        processed = 0
        
        # Process up to 10 orders at once
        while not self.order_queue.empty() and processed < 10:
            try:
                order = self.order_queue.get_nowait()
                
                if not order:
                    continue
                    
                # Process order
                if self.mode == "live":
                    self._execute_live_order(order)
                else:
                    self._execute_paper_order(order)
                    
                # Mark as processed
                processed += 1
                self.order_queue.task_done()
                
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Error processing order: {e}")
                self.order_queue.task_done()
        
        if processed > 0:
            self.logger.debug(f"Processed {processed} orders")
    
    def _execute_live_order(self, order):
        """
        Execute order in live trading mode.
        
        Args:
            order (dict): Order details
        """
        try:
            if not self.order_executor:
                self.logger.error("No order executor set, cannot execute live order")
                return
                
            # Store order in database
            order_id = self._store_order_in_db(order)
            
            if not order_id:
                return
                
            # Update order with ID
            order['order_id'] = order_id
            
            # Execute order
            result = self.order_executor.place_order(order)
            
            if not result:
                self.logger.error(f"Failed to place order for {order['symbol']}")
                
                # Update order status in database
                self._update_order_in_db({
                    'order_id': order_id,
                    'status': 'FAILED',
                    'error': 'Failed to place order'
                })
                
                return
                
            # Update order with broker ID
            broker_order_id = result.get('broker_order_id')
            
            if broker_order_id:
                self._update_order_in_db({
                    'order_id': order_id,
                    'broker_order_id': broker_order_id,
                    'status': 'PENDING'
                })
                
            # For entry orders, create position
            if order.get('action') in ["BUY", "SELL"] and not order.get('position_id'):
                self._create_position_from_order(order)
                
            self.logger.info(f"Executed live order for {order['symbol']}")
            
        except Exception as e:
            self.logger.error(f"Error executing live order: {e}")
    
    def _execute_paper_order(self, order):
        """
        Execute order in paper trading mode.
        
        Args:
            order (dict): Order details
        """
        try:
            # Store order in database
            order_id = self._store_order_in_db(order)
            
            if not order_id:
                return
                
            # Update order with ID
            order['order_id'] = order_id
            
            # Simulate order execution
            symbol = order.get('symbol')
            exchange = order.get('exchange')
            action = order.get('action')
            quantity = order.get('quantity', 0)
            price = order.get('price')
            
            # Get current market price if not specified
            if not price:
                market_data = self._get_symbol_data(symbol, exchange)
                
                if not market_data:
                    self.logger.error(f"No market data for {symbol}, cannot execute paper order")
                    return
                    
                price = market_data.get('last_price')
                
                if not price:
                    self.logger.error(f"No price data for {symbol}, cannot execute paper order")
                    return
            
            # Update order in database
            self._update_order_in_db({
                'order_id': order_id,
                'status': 'COMPLETE',
                'executed_price': price,
                'executed_quantity': quantity,
                'execution_time': datetime.now()
            })
            
            # For entry orders, create position
            if action in ["BUY", "SELL"] and not order.get('position_id'):
                position_id = self._create_position_from_order(order, price)
                
                if position_id:
                    self.logger.info(f"Created paper position for {symbol}: {quantity} @ {price}")
            
            # For exit orders, close position
            elif order.get('position_id'):
                position_id = order.get('position_id')
                self._close_position(position_id, price, order.get('exit_reason', 'strategy_exit'))
                self.logger.info(f"Closed paper position for {symbol}: {quantity} @ {price}")
            
            self.logger.info(f"Executed paper order for {symbol}: {action} {quantity} @ {price}")
            
        except Exception as e:
            self.logger.error(f"Error executing paper order: {e}")
    
    def _store_order_in_db(self, order):
        """
        Store order in database.
        
        Args:
            order (dict): Order details
            
        Returns:
            str: Order ID
        """
        try:
            # Create order document
            order_doc = {
                'symbol': order.get('symbol'),
                'exchange': order.get('exchange'),
                'action': order.get('action'),
                'quantity': order.get('quantity'),
                'price': order.get('price'),
                'order_type': order.get('order_type'),
                'product_type': order.get('product_type'),
                'strategy_id': order.get('strategy_id'),
                'position_id': order.get('position_id'),
                'stop_loss': order.get('stop_loss'),
                'target': order.get('target'),
                'confidence': order.get('confidence'),
                'timeframe': order.get('timeframe'),
                'expiry': order.get('expiry'),
                'status': 'CREATED',
                'created_at': datetime.now(),
                'trading_mode': self.mode
            }
            
            # Insert into database
            result = self.db.orders_collection.insert_one(order_doc)
            order_id = str(result.inserted_id)
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error storing order in database: {e}")
            return None
    
    def _update_order_in_db(self, order_update):
        """
        Update order in database.
        
        Args:
            order_update (dict): Order update details
            
        Returns:
            bool: Success status
        """
        try:
            order_id = order_update.get('order_id')
            
            if not order_id:
                return False
                
            # Convert to ObjectId if string
            from bson.objectid import ObjectId
            if isinstance(order_id, str):
                order_id = ObjectId(order_id)
            
            # Create update document
            update_doc = {
                'updated_at': datetime.now()
            }
            
            # Add update fields
            for key, value in order_update.items():
                if key != 'order_id':
                    update_doc[key] = value
            
            # Update in database
            result = self.db.orders_collection.update_one(
                {'_id': order_id},
                {'$set': update_doc}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            self.logger.error(f"Error updating order in database: {e}")
            return False
    
    def _create_position_from_order(self, order, executed_price=None):
        """
        Create position from order.
        
        Args:
            order (dict): Order details
            executed_price (float): Executed price (for paper trading)
            
        Returns:
            str: Position ID
        """
        try:
            # Extract order details
            symbol = order.get('symbol')
            exchange = order.get('exchange')
            action = order.get('action')
            quantity = order.get('quantity', 0)
            price = executed_price or order.get('price')
            order_id = order.get('order_id')
            
            if not symbol or not action or not price or quantity <= 0:
                self.logger.error(f"Invalid order details for position creation")
                return None
                
            # Create position document
            position_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': quantity,
                'entry_price': price,
                'entry_time': datetime.now(),
                'order_id': order_id,
                'stop_loss': order.get('stop_loss'),
                'target': order.get('target'),
                'strategy_id': order.get('strategy_id'),
                'timeframe': order.get('timeframe', 'day'),
                'confidence': order.get('confidence', 0.5),
                'status': 'OPEN',
                'trading_mode': self.mode
            }
            
            # Insert into database
            result = self.db.positions_collection.insert_one(position_doc)
            position_id = str(result.inserted_id)
            
            # Update order with position ID
            if order_id:
                self._update_order_in_db({
                    'order_id': order_id,
                    'position_id': position_id
                })
            
            # Add to position cache
            position_key = f"{symbol}:{exchange}"
            self.position_cache[position_key] = {
                'position_id': position_id,
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': quantity,
                'entry_price': price,
                'stop_loss': order.get('stop_loss'),
                'target': order.get('target'),
                'entry_time': datetime.now()
            }
            
            return position_id
            
        except Exception as e:
            self.logger.error(f"Error creating position from order: {e}")
            return None
    
    def _update_positions(self):
        """
        Update positions with current market data.
        """
        # Check for stop loss and target hits
        for key, position in list(self.position_cache.items()):
            try:
                position_id = position.get('position_id')
                symbol = position.get('symbol')
                exchange = position.get('exchange')
                action = position.get('action')
                stop_loss = position.get('stop_loss')
                target = position.get('target')
                
                if not position_id or not symbol:
                    continue
                    
                # Get current market price
                market_data = self._get_symbol_data(symbol, exchange)
                
                if not market_data:
                    continue
                    
                current_price = market_data.get('last_price')
                
                if not current_price:
                    continue
                
                # Check stop loss hit
                if stop_loss and self.config['use_stoploss']:
                    if (action == "BUY" and current_price <= stop_loss) or \
                       (action == "SELL" and current_price >= stop_loss):
                        # Create exit order at market price
                        exit_action = "SELL" if action == "BUY" else "BUY"
                        
                        order = {
                            'symbol': symbol,
                            'exchange': exchange,
                            'action': exit_action,
                            'quantity': position['quantity'],
                            'price': current_price,
                            'order_type': 'MARKET',
                            'product_type': self.config['default_product'],
                            'position_id': position_id,
                            'exit_reason': 'stop_loss',
                            'timestamp': datetime.now()
                        }
                        
                        # Add to order queue
                        self.order_queue.put(order)
                        
                        self.logger.info(f"Stop loss hit for {symbol} at {current_price}")
                        continue
                
                # Check target hit
                if target and self.config['use_target']:
                    if (action == "BUY" and current_price >= target) or \
                       (action == "SELL" and current_price <= target):
                        # Create exit order at market price
                        exit_action = "SELL" if action == "BUY" else "BUY"
                        
                        order = {
                            'symbol': symbol,
                            'exchange': exchange,
                            'action': exit_action,
                            'quantity': position['quantity'],
                            'price': current_price,
                            'order_type': 'MARKET',
                            'product_type': self.config['default_product'],
                            'position_id': position_id,
                            'exit_reason': 'target',
                            'timestamp': datetime.now()
                        }
                        
                        # Add to order queue
                        self.order_queue.put(order)
                        
                        self.logger.info(f"Target hit for {symbol} at {current_price}")
                        continue
                
                # Update unrealized PnL
                entry_price = position.get('entry_price', 0)
                quantity = position.get('quantity', 0)
                
                if action == "BUY":
                    unrealized_pnl = (current_price - entry_price) * quantity
                else:
                    unrealized_pnl = (entry_price - current_price) * quantity
                    
                # Update position in cache
                position['current_price'] = current_price
                position['unrealized_pnl'] = unrealized_pnl
                
                # Periodically update position in database
                if (datetime.now() - position.get('last_update', datetime.min)).total_seconds() > 60:
                    self._update_position_in_db(position_id, {
                        'current_price': current_price,
                        'unrealized_pnl': unrealized_pnl,
                        'updated_at': datetime.now()
                    })
                    position['last_update'] = datetime.now()
                
            except Exception as e:
                self.logger.error(f"Error updating position {key}: {e}")
    
    def _update_position_in_db(self, position_id, updates):
        """
        Update position in database.
        
        Args:
            position_id (str): Position ID
            updates (dict): Update fields
            
        Returns:
            bool: Success status
        """
        try:
            # Convert to ObjectId if string
            from bson.objectid import ObjectId
            if isinstance(position_id, str):
                position_id = ObjectId(position_id)
            
            # Create update document
            update_doc = {
                'updated_at': datetime.now()
            }
            
            # Add update fields
            for key, value in updates.items():
                update_doc[key] = value
            
            # Update in database
            result = self.db.positions_collection.update_one(
                {'_id': position_id},
                {'$set': update_doc}
            )
            
            return result.modified_count > 0
            
        except Exception as e:
            self.logger.error(f"Error updating position in database: {e}")
            return False
    
    def _update_position_on_fill(self, order_data):
        """
        Update position when order is filled.
        
        Args:
            order_data (dict): Order update data
        """
        try:
            position_id = order_data.get('position_id')
            
            if not position_id:
                return
                
            order_action = order_data.get('action')
            executed_price = order_data.get('executed_price')
            executed_quantity = order_data.get('executed_quantity')
            
            if not executed_price or not executed_quantity:
                return
                
            # Get position
            from bson.objectid import ObjectId
            position = self.db.positions_collection.find_one({
                '_id': ObjectId(position_id)
            })
            
            if not position:
                return
                
            symbol = position.get('symbol')
            exchange = position.get('exchange')
            position_action = position.get('action')
            
            # Check if this is an exit order
            if ((position_action == "BUY" and order_action == "SELL") or
                (position_action == "SELL" and order_action == "BUY")):
                # Close position
                self._close_position(position_id, executed_price, order_data.get('exit_reason', 'strategy_exit'))
                
                # Remove from cache
                position_key = f"{symbol}:{exchange}"
                if position_key in self.position_cache:
                    del self.position_cache[position_key]
                    
                self.logger.info(f"Closed position for {symbol} at {executed_price}")
                
            # Update entry price for partial fills
            elif executed_quantity < position.get('quantity', 0):
                # Update position with new average price
                self._update_position_in_db(position_id, {
                    'entry_price': executed_price,
                    'quantity': executed_quantity
                })
                
                # Update cache
                position_key = f"{symbol}:{exchange}"
                if position_key in self.position_cache:
                    self.position_cache[position_key]['entry_price'] = executed_price
                    self.position_cache[position_key]['quantity'] = executed_quantity
                
        except Exception as e:
            self.logger.error(f"Error updating position on fill: {e}")
    
    def _close_position(self, position_id, exit_price, exit_reason='strategy_exit'):
        """
        Close position and record profit/loss.
        
        Args:
            position_id (str): Position ID
            exit_price (float): Exit price
            exit_reason (str): Exit reason
            
        Returns:
            bool: Success status
        """
        try:
            # Get position
            from bson.objectid import ObjectId
            position = self.db.positions_collection.find_one({
                '_id': ObjectId(position_id)
            })
            
            if not position:
                return False
                
            # Calculate profit/loss
            entry_price = position.get('entry_price', 0)
            quantity = position.get('quantity', 0)
            action = position.get('action')
            
            if action == "BUY":
                profit_loss = (exit_price - entry_price) * quantity
            else:
                profit_loss = (entry_price - exit_price) * quantity
                
            # Calculate profit/loss percentage
            if entry_price > 0:
                profit_loss_pct = (profit_loss / (entry_price * quantity)) * 100
            else:
                profit_loss_pct = 0
                
            # Calculate holding period
            entry_time = position.get('entry_time')
            exit_time = datetime.now()
            
            if entry_time:
                holding_period = (exit_time - entry_time).total_seconds() / 3600  # in hours
            else:
                holding_period = 0
                
            # Update position in database
            self._update_position_in_db(position_id, {
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': exit_time,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'holding_period': holding_period,
                'exit_reason': exit_reason
            })
            
            # Add to trade history
            self.trade_history.append({
                'position_id': position_id,
                'symbol': position.get('symbol'),
                'exchange': position.get('exchange'),
                'action': action,
                'quantity': quantity,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'holding_period': holding_period,
                'exit_reason': exit_reason,
                'strategy_id': position.get('strategy_id')
            })
            
            # Update equity curve
            current_equity = self._get_current_equity()
            self.equity_curve.append({
                'timestamp': datetime.now(),
                'equity': current_equity,
                'trade_id': position_id
            })
            
            # Log trade result
            self.logger.info(f"Closed position {position.get('symbol')} with P&L: {profit_loss:.2f} ({profit_loss_pct:.2f}%)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return False
    
    def _get_current_equity(self):
        """
        Calculate current equity including open positions.
        
        Returns:
            float: Current equity
        """
        try:
            # Get account balance
            equity = self._get_account_balance()
            
            # Add unrealized P&L from open positions
            for position in self.position_cache.values():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                equity += unrealized_pnl
                
            return equity
            
        except Exception as e:
            self.logger.error(f"Error calculating current equity: {e}")
            return 0
    
    def get_open_positions(self):
        """
        Get open positions.
        
        Returns:
            list: Open positions
        """
        return list(self.position_cache.values())
    
    def get_position_details(self, symbol, exchange=None):
        """
        Get position details for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Position details
        """
        exchange = exchange or self.config['default_exchange']
        position_key = f"{symbol}:{exchange}"
        
        return self.position_cache.get(position_key)
    
    def close_position(self, symbol, exchange=None, reason='manual_exit'):
        """
        Close position for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange (optional)
            reason (str): Exit reason
            
        Returns:
            bool: Success status
        """
        exchange = exchange or self.config['default_exchange']
        position_key = f"{symbol}:{exchange}"
        
        if position_key not in self.position_cache:
            self.logger.warning(f"No position found for {symbol}")
            return False
            
        position = self.position_cache[position_key]
        position_id = position.get('position_id')
        
        if not position_id:
            return False
            
        # Get current market price
        market_data = self._get_symbol_data(symbol, exchange)
        
        if not market_data:
            self.logger.error(f"No market data for {symbol}, cannot close position")
            return False
            
        current_price = market_data.get('last_price')
        
        if not current_price:
            self.logger.error(f"No price data for {symbol}, cannot close position")
            return False
            
        # Create exit order
        action = "SELL" if position['action'] == "BUY" else "BUY"
        
        order = {
            'symbol': symbol,
            'exchange': exchange,
            'action': action,
            'quantity': position['quantity'],
            'price': current_price,
            'order_type': 'MARKET',
            'product_type': self.config['default_product'],
            'position_id': position_id,
            'exit_reason': reason,
            'timestamp': datetime.now()
        }
        
        # Add to order queue
        self.order_queue.put(order)
        
        self.logger.info(f"Closing position for {symbol} at {current_price}")
        
        return True
    
    def close_all_positions(self, reason='system_exit'):
        """
        Close all open positions.
        
        Args:
            reason (str): Exit reason
            
        Returns:
            int: Number of positions closed
        """
        closed_count = 0
        
        for position_key in list(self.position_cache.keys()):
            parts = position_key.split(':')
            
            if len(parts) == 2:
                symbol, exchange = parts
                if self.close_position(symbol, exchange, reason):
                    closed_count += 1
        
        self.logger.info(f"Closed {closed_count} positions")
        
        return closed_count
    
    def get_performance_metrics(self):
        """
        Get performance metrics.
        
        Returns:
            dict: Performance metrics
        """
        try:
            # Get all closed trades
            closed_trades = self.trade_history
            
            if not closed_trades:
                return {
                    'total_trades': 0,
                    'win_rate': 0,
                    'profit_factor': 0,
                    'average_profit': 0,
                    'average_loss': 0,
                    'max_profit': 0,
                    'max_loss': 0,
                    'total_profit_loss': 0,
                    'starting_equity': self._get_account_balance(),
                    'current_equity': self._get_current_equity(),
                    'open_positions': len(self.position_cache)
                }
                
            # Calculate metrics
            total_trades = len(closed_trades)
            winning_trades = [t for t in closed_trades if t['profit_loss'] > 0]
            losing_trades = [t for t in closed_trades if t['profit_loss'] <= 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
            
            total_profit = sum(t['profit_loss'] for t in winning_trades)
            total_loss = abs(sum(t['profit_loss'] for t in losing_trades))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
            
            average_profit = total_profit / win_count if win_count > 0 else 0
            average_loss = total_loss / loss_count if loss_count > 0 else 0
            
            max_profit = max([t['profit_loss'] for t in winning_trades]) if winning_trades else 0
            max_loss = min([t['profit_loss'] for t in losing_trades]) if losing_trades else 0
            
            total_profit_loss = total_profit - total_loss
            
            # Get current equity
            current_equity = self._get_current_equity()
            
            return {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'average_profit': average_profit,
                'average_loss': average_loss,
                'max_profit': max_profit,
                'max_loss': max_loss,
                'total_profit_loss': total_profit_loss,
                'starting_equity': self._get_account_balance(),
                'current_equity': current_equity,
                'open_positions': len(self.position_cache)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def save_performance_snapshot(self):
        """
        Save performance snapshot to database.
        
        Returns:
            str: Snapshot ID
        """
        try:
            # Get performance metrics
            metrics = self.get_performance_metrics()
            
            # Create snapshot document
            snapshot = {
                'timestamp': datetime.now(),
                'metrics': metrics,
                'equity_curve': self.equity_curve[-100:] if len(self.equity_curve) > 100 else self.equity_curve,
                'open_positions': list(self.position_cache.values()),
                'trading_mode': self.mode,
                'strategy_count': len(self.strategy_registry)
            }
            
            # Insert into database
            result = self.db.performance_snapshots_collection.insert_one(snapshot)
            snapshot_id = str(result.inserted_id)
            
            self.logger.info(f"Saved performance snapshot with ID: {snapshot_id}")
            
            return snapshot_id
            
        except Exception as e:
            self.logger.error(f"Error saving performance snapshot: {e}")
            return None
    
    def _cleanup(self):
        """
        Clean up resources on shutdown.
        """
        try:
            # Save performance snapshot
            self.save_performance_snapshot()
            
            # Disconnect market data
            if self.market_data_connector:
                self.market_data_connector.disconnect()
                
            # Disconnect order execution
            if self.order_executor:
                self.order_executor.disconnect()
                
            self.logger.info("Trading engine resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up resources: {e}")