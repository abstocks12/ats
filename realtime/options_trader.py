# realtime/options_trader.py
import logging
import threading
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class OptionsTrader:
    """
    Specialized trading module for options trading.
    """
    
    def __init__(self, trading_engine, db_connector, logger=None):
        """
        Initialize the options trader.
        
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
        self.options_info = {}  # Store contract details
        
        # Configuration
        self.config = {
            'max_symbols': 50,  # Maximum number of options to trade
            'order_types': ['LIMIT', 'MARKET', 'SL', 'SL-M'],  # Allowed order types
            'default_order_type': 'LIMIT',
            'default_product': 'NRML',  # NRML for options
            'default_exchange': 'NFO',  # NFO for options
            'max_slippage': 0.05,  # Maximum slippage in percentage
            'price_tick_size': 0.05,  # Price tick size
            'use_limit_orders': True,  # Use limit orders instead of market orders
            'limit_order_timeout': 60,  # Timeout for limit orders in seconds
            'min_premium': 0.5,  # Minimum premium filter
            'max_premium': 2000.0,  # Maximum premium filter
            'min_oi': 1000,  # Minimum open interest filter
            'min_volume': 100,  # Minimum volume filter
            'trade_session_start': '09:15:00',  # Trading session start time
            'trade_session_end': '15:30:00',  # Trading session end time
            'max_positions': 10,  # Maximum number of positions
            'expiry_buffer_days': 3,  # Minimum days before expiry to trade
            'strategy_timeframes': ['1min', '5min', '15min', '30min', '1hour', 'day'],  # Supported timeframes
            'max_theta_decay': 0.3,  # Maximum allowed theta decay (% of premium)
            'iv_rank_threshold': 50,  # IV rank threshold for option strategies
            'max_leverage': 5.0,  # Maximum leverage
            'max_loss_per_trade': 1.0,  # Maximum loss per trade (% of account)
            'greek_calculation': True,  # Calculate option Greeks
            'greek_update_interval': 300  # Greek update interval in seconds
        }
        
        # Initialize
        self._initialize()
        
        self.logger.info("Options trader initialized")
    
    def set_config(self, config):
        """
        Set configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated options trader configuration: {self.config}")
    
    def _initialize(self):
        """
        Initialize the options trader.
        """
        # Register with trading engine
        if self.engine:
            # If engine is already running, we need to make sure it has our callback
            if hasattr(self.engine, 'market_data_connector') and self.engine.market_data_connector:
                self.engine.market_data_connector.set_data_callback(self._on_market_data)
                
            # Load options contracts
            self._load_options_contracts()
            
            # Start Greeks calculation thread if enabled
            if self.config['greek_calculation']:
                threading.Thread(target=self._greek_calculation_loop, daemon=True).start()
        else:
            self.logger.warning("No trading engine provided")
    
    def _load_options_contracts(self):
        """
        Load options contract details.
        """
        try:
            # Get all options contracts from database
            options = self.db.options_contracts_collection.find({
                'expiry_date': {'$gte': datetime.now()}  # Only active contracts
            })
            
            for contract in options:
                symbol = contract.get('symbol')
                exchange = contract.get('exchange')
                expiry = contract.get('expiry_date')
                strike = contract.get('strike_price')
                option_type = contract.get('option_type')
                
                if symbol and exchange and expiry and strike and option_type:
                    # Store in options info
                    key = f"{symbol}:{exchange}:{strike}:{option_type}"
                    self.options_info[key] = contract
                    
                    # Also store by trading symbol if available
                    trading_symbol = contract.get('tradingsymbol')
                    if trading_symbol:
                        self.options_info[f"{trading_symbol}:{exchange}"] = contract
                    
            self.logger.info(f"Loaded {len(self.options_info)} options contracts")
            
        except Exception as e:
            self.logger.error(f"Error loading options contracts: {e}")
    
    def add_option_contract(self, symbol, strike_price, option_type, expiry=None, exchange=None):
        """
        Add an options contract for trading.
        
        Args:
            symbol (str): Symbol (e.g., 'NIFTY', 'RELIANCE')
            strike_price (float): Strike price
            option_type (str): Option type ('CE' or 'PE')
            expiry (str/datetime): Expiry date (optional)
            exchange (str): Exchange (optional)
            
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
        
        # Get option contract
        contract = self._get_option_contract(symbol, strike_price, option_type, exchange, expiry)
        
        if not contract:
            self.logger.error(f"Option contract not found for {symbol} {strike_price} {option_type}")
            return False
            
        # Extract contract symbol
        contract_symbol = contract.get('tradingsymbol')
        
        if not contract_symbol:
            self.logger.error(f"No trading symbol for option contract {symbol} {strike_price} {option_type}")
            return False
            
        # Check if already added
        symbol_key = f"{contract_symbol}:{exchange}"
        if symbol_key in self.active_symbols:
            self.logger.info(f"Option contract {symbol_key} already added")
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
        
        self.logger.info(f"Added option contract for trading: {symbol_key}")
        return True
    
    def _get_option_contract(self, symbol, strike_price, option_type, exchange, expiry=None):
        """
        Get option contract details.
        
        Args:
            symbol (str): Symbol
            strike_price (float): Strike price
            option_type (str): Option type ('CE' or 'PE')
            exchange (str): Exchange
            expiry (datetime): Expiry date (optional)
            
        Returns:
            dict: Contract details
        """
        try:
            # Query for active contracts
            query = {
                'symbol': symbol,
                'strike_price': strike_price,
                'option_type': option_type,
                'exchange': exchange,
                'instrument_type': 'OPT',
                'expiry_date': {'$gte': datetime.now()}  # Only active contracts
            }
            
            # Add expiry if provided
            if expiry:
                query['expiry_date'] = expiry
                
            # Sort by expiry (closest first)
            cursor = self.db.options_contracts_collection.find(query).sort('expiry_date', 1)
            
            # Get first contract (closest expiry)
            contract = next(cursor, None)
            
            if not contract:
                self.logger.error(f"No active option contract found for {symbol} {strike_price} {option_type}")
                return None
                
            # Check expiry buffer
            if contract.get('expiry_date'):
                days_to_expiry = (contract['expiry_date'] - datetime.now()).days
                
                if days_to_expiry < self.config['expiry_buffer_days']:
                    self.logger.warning(f"Option contract {symbol} {strike_price} {option_type} is close to expiry ({days_to_expiry} days)")
                    
                    # Try to get next contract
                    next_contract = next(cursor, None)
                    
                    if next_contract:
                        self.logger.info(f"Using next option contract for {symbol} {strike_price} {option_type}")
                        return next_contract
            
            return contract
            
        except Exception as e:
            self.logger.error(f"Error getting option contract: {e}")
            return None
    
    def remove_option_contract(self, symbol, exchange=None):
        """
        Remove an option contract from trading.
        
        Args:
            symbol (str): Symbol (can be trading symbol or underlying)
            exchange (str): Exchange (optional)
            
        Returns:
            bool: Success status
        """
        exchange = exchange or self.config['default_exchange']
        
        # Find all active symbols for this option
        remove_keys = []
        
        for symbol_key in self.active_symbols:
            parts = symbol_key.split(':')
            
            if len(parts) == 2:
                trading_symbol, symbol_exchange = parts
                
                # Check if this is the symbol to remove
                if symbol_exchange == exchange and (symbol in trading_symbol or trading_symbol == symbol):
                    remove_keys.append(symbol_key)
        
        if not remove_keys:
            self.logger.info(f"No active option contracts found for {symbol}")
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
                
                self.logger.info(f"Removed option contract from trading: {key}")
        
        return True
    
    def _greek_calculation_loop(self):
        """
        Background loop for calculating option Greeks.
        """
        while self.is_running and self.config['greek_calculation']:
            try:
                # Calculate Greeks for all active options
                self._calculate_all_greeks()
                
                # Sleep for the specified interval
                time.sleep(self.config['greek_update_interval'])
                
            except Exception as e:
                self.logger.error(f"Error in Greek calculation loop: {e}")
                time.sleep(10)  # Sleep longer on error
    
    def _calculate_all_greeks(self):
        """
        Calculate Greeks for all active options.
        """
        try:
            for symbol_key in self.active_symbols:
                parts = symbol_key.split(':')
                
                if len(parts) != 2:
                    continue
                    
                trading_symbol, exchange = parts
                
                # Get contract details
                contract = self.options_info.get(symbol_key)
                
                if not contract:
                    continue
                    
                # Get current price
                if self.engine and self.engine.market_data_connector:
                    data = self.engine.market_data_connector.get_last_price(trading_symbol, exchange)
                    
                    if not data:
                        continue
                        
                    current_price = data.get('last_price')
                    
                    if not current_price:
                        continue
                
                # Get underlying price
                underlying_symbol = contract.get('symbol')
                underlying_exchange = contract.get('underlying_exchange', 'NSE')
                
                if not underlying_symbol:
                    continue
                    
                underlying_data = None
                
                if self.engine and self.engine.market_data_connector:
                    underlying_data = self.engine.market_data_connector.get_last_price(underlying_symbol, underlying_exchange)
                
                if not underlying_data:
                    continue
                    
                underlying_price = underlying_data.get('last_price')
                
                if not underlying_price:
                    continue
                
                # Calculate Greeks
                strike_price = contract.get('strike_price')
                days_to_expiry = (contract.get('expiry_date') - datetime.now()).days + (datetime.now().hour / 24.0)
                option_type = contract.get('option_type')
                
                if not strike_price or not days_to_expiry or not option_type:
                    continue
                
                # Use implied volatility if available, otherwise use historical volatility
                iv = contract.get('implied_volatility')
                
                if not iv:
                    # Estimate IV from price
                    iv = self._estimate_implied_volatility(
                        option_price=current_price,
                        underlying_price=underlying_price,
                        strike_price=strike_price,
                        days_to_expiry=days_to_expiry,
                        option_type=option_type
                    )
                
                if not iv:
                    continue
                
                # Calculate Greeks
                greeks = self._calculate_option_greeks(
                    underlying_price=underlying_price,
                    strike_price=strike_price,
                    days_to_expiry=days_to_expiry,
                    volatility=iv,
                    option_type=option_type
                )
                
                if not greeks:
                    continue
                
                # Update contract with Greeks
                contract.update({
                    'current_price': current_price,
                    'underlying_price': underlying_price,
                    'greeks': greeks,
                    'implied_volatility': iv,
                    'updated_at': datetime.now()
                })
                
                # Store updated contract
                self.options_info[symbol_key] = contract
                
                # Update in database
                self._update_option_contract_in_db(contract)
                
        except Exception as e:
            self.logger.error(f"Error calculating Greeks: {e}")
    
    def _estimate_implied_volatility(self, option_price, underlying_price, strike_price, days_to_expiry, option_type):
        """
        Estimate implied volatility using the bisection method.
        
        Args:
            option_price (float): Current option price
            underlying_price (float): Current underlying price
            strike_price (float): Strike price
            days_to_expiry (float): Days to expiry
            option_type (str): Option type ('CE' or 'PE')
            
        Returns:
            float: Implied volatility
        """
        try:
            # Convert days to years
            t = days_to_expiry / 365.0
            
            if t <= 0:
                return None
                
            # Set initial values for bisection
            low_vol = 0.01
            high_vol = 5.0
            tolerance = 0.0001
            max_iterations = 100
            
            # Bisection method
            for i in range(max_iterations):
                mid_vol = (low_vol + high_vol) / 2.0
                
                # Calculate option price with mid volatility
                mid_price = self._black_scholes_price(
                    underlying_price=underlying_price,
                    strike_price=strike_price,
                    days_to_expiry=days_to_expiry,
                    volatility=mid_vol,
                    option_type=option_type
                )
                
                # Check if we're close enough
                if abs(mid_price - option_price) < tolerance:
                    return mid_vol
                    
                # Adjust bounds
                if mid_price > option_price:
                    high_vol = mid_vol
                else:
                    low_vol = mid_vol
                    
                # Check if bounds are too close
                if abs(high_vol - low_vol) < tolerance:
                    return mid_vol
            
            # Return the last mid volatility
            return (low_vol + high_vol) / 2.0
            
        except Exception as e:
            self.logger.error(f"Error estimating implied volatility: {e}")
            return None
    
    def _black_scholes_price(self, underlying_price, strike_price, days_to_expiry, volatility, option_type):
        """
        Calculate Black-Scholes option price.
        
        Args:
            underlying_price (float): Current underlying price
            strike_price (float): Strike price
            days_to_expiry (float): Days to expiry
            volatility (float): Volatility
            option_type (str): Option type ('CE' or 'PE')
            
        Returns:
            float: Option price
        """
        import math
        from scipy.stats import norm
        
        # Convert days to years
        t = days_to_expiry / 365.0
        
        if t <= 0:
            # For expired options, return intrinsic value
            if option_type == 'CE':
                return max(0, underlying_price - strike_price)
            else:
                return max(0, strike_price - underlying_price)
        
        # Risk-free rate (assumed)
        r = 0.05
        
        # Black-Scholes formula
        d1 = (math.log(underlying_price / strike_price) + (r + 0.5 * volatility**2) * t) / (volatility * math.sqrt(t))
        d2 = d1 - volatility * math.sqrt(t)
        
        if option_type == 'CE':
            # Call option
            return underlying_price * norm.cdf(d1) - strike_price * math.exp(-r * t) * norm.cdf(d2)
        else:
            # Put option
            return strike_price * math.exp(-r * t) * norm.cdf(-d2) - underlying_price * norm.cdf(-d1)
    
    def _calculate_option_greeks(self, underlying_price, strike_price, days_to_expiry, volatility, option_type):
        """
        Calculate option Greeks.
        
        Args:
            underlying_price (float): Current underlying price
            strike_price (float): Strike price
            days_to_expiry (float): Days to expiry
            volatility (float): Volatility
            option_type (str): Option type ('CE' or 'PE')
            
        Returns:
            dict: Option Greeks
        """
        try:
            import math
            from scipy.stats import norm
            
            # Convert days to years
            t = days_to_expiry / 365.0
            
            if t <= 0:
                return {
                    'delta': 1.0 if option_type == 'CE' and underlying_price > strike_price else 0.0,
                    'gamma': 0.0,
                    'theta': 0.0,
                    'vega': 0.0,
                    'rho': 0.0
                }
            
            # Risk-free rate (assumed)
            r = 0.05
            
            # Black-Scholes formula parameters
            d1 = (math.log(underlying_price / strike_price) + (r + 0.5 * volatility**2) * t) / (volatility * math.sqrt(t))
            d2 = d1 - volatility * math.sqrt(t)
            
            # Calculate Greeks
            if option_type == 'CE':
                # Delta for call
                delta = norm.cdf(d1)
                # Rho for call
                rho = strike_price * t * math.exp(-r * t) * norm.cdf(d2) / 100
            else:
                # Delta for put
                delta = norm.cdf(d1) - 1
                # Rho for put
                rho = -strike_price * t * math.exp(-r * t) * norm.cdf(-d2) / 100
            
            # Common Greeks
            gamma = norm.pdf(d1) / (underlying_price * volatility * math.sqrt(t))
            theta = (-underlying_price * norm.pdf(d1) * volatility / (2 * math.sqrt(t)) - 
                    r * strike_price * math.exp(-r * t) * norm.cdf(d2 if option_type == 'CE' else -d2)) / 365
            vega = underlying_price * math.sqrt(t) * norm.pdf(d1) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating option Greeks: {e}")
            return None
    
    def _update_option_contract_in_db(self, contract):
        """
        Update option contract in database.
        
        Args:
            contract (dict): Option contract details
        """
        try:
            # Get contract ID
            contract_id = contract.get('_id')
            
            if not contract_id:
                return
                
            # Create update document
            from bson.objectid import ObjectId
            
            if isinstance(contract_id, str):
                contract_id = ObjectId(contract_id)
                
            # Remove _id from update
            contract_copy = contract.copy()
            contract_copy.pop('_id', None)
            
            # Update in database
            self.db.options_contracts_collection.update_one(
                {'_id': contract_id},
                {'$set': contract_copy}
            )
            
        except Exception as e:
            self.logger.error(f"Error updating option contract in database: {e}")
    
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
                    
                # Check if this is an option contract or a specification
                if symbol_name in [s.split(':')[0] for s in self.active_symbols]:
                    # This is an existing contract
                    valid_symbols.append(symbol_name)
                    valid_exchanges.append(exchange)
                elif '|' in symbol_name:
                    # This is an option specification (e.g., NIFTY|14000|CE)
                    spec_parts = symbol_name.split('|')
                    
                    if len(spec_parts) == 3:
                        underlying, strike, option_type = spec_parts
                        
                        try:
                            strike = float(strike)
                            
                            # Add option contract
                            if self.add_option_contract(underlying, strike, option_type, exchange=exchange):
                                # Get contract symbol
                                for key, contract in self.options_info.items():
                                    if (contract.get('symbol') == underlying and 
                                        contract.get('strike_price') == strike and 
                                        contract.get('option_type') == option_type and 
                                        contract.get('exchange') == exchange):
                                        
                                        trading_symbol = contract.get('tradingsymbol')
                                        
                                        if trading_symbol:
                                            valid_symbols.append(trading_symbol)
                                            valid_exchanges.append(exchange)
                                            break
                        except ValueError:
                            self.logger.error(f"Invalid strike price in option specification: {symbol_name}")
                else:
                    self.logger.warning(f"Invalid option symbol: {symbol_name}")
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
        Start the options trader.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Options trader is already running")
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
        self.logger.info("Options trader started")
        
        # Start monitoring thread
        threading.Thread(target=self._monitoring_loop, daemon=True).start()
        
        return True
    
    def stop(self):
        """
        Stop the options trader.
        
        Returns:
            bool: Success status
        """
        if not self.is_running:
            self.logger.warning("Options trader is not running")
            return False
            
        self.is_running = False
        self.logger.info("Options trader stopped")
        
        return True
    
    def _monitoring_loop(self):
        """
        Monitoring loop for the options trader.
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
        # Refresh options info
        self._load_options_contracts()
        
        # Check each active symbol
        for symbol_key in list(self.active_symbols):
            parts = symbol_key.split(':')
            
            if len(parts) != 2:
                continue
                
            symbol, exchange = parts
            
            # Check contract details
            contract = self.options_info.get(symbol_key)
            
            if not contract:
                continue
                
            expiry = contract.get('expiry_date')
            
            if not expiry:
                continue
                
            # Check days to expiry
            days_to_expiry = (expiry - datetime.now()).days
            
            if days_to_expiry < self.config['expiry_buffer_days']:
                self.logger.warning(f"Option contract {symbol_key} nearing expiry ({days_to_expiry} days)")
                
                # Check if we have positions
                if self.engine:
                    position = self.engine.get_position_details(symbol, exchange)
                    
                    if position:
                        self.logger.warning(f"Closing position for expiring contract {symbol_key}")
                        self.engine.close_position(symbol, exchange, 'contract_expiry')
                
                # Remove from active symbols
                self.active_symbols.remove(symbol_key)
                
                # Unsubscribe from market data
                if self.engine and self.engine.market_data_connector:
                    self.engine.market_data_connector.unsubscribe([symbol], [exchange])
    
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
                
            # Update options info
            contract = self.options_info.get(symbol_key)
            
            if contract:
                # Update current price
                contract['current_price'] = data.get('last_price')
                contract['updated_at'] = datetime.now()
                
                # Store updated contract
                self.options_info[symbol_key] = contract
            
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
        # Check premium range
        price = data.get('last_price')
        if price and (price < self.config['min_premium'] or price > self.config['max_premium']):
            return False
            
        # Check volume
        volume = data.get('volume')
        if volume and volume < self.config['min_volume']:
            return False
            
        return True
    
    def place_option_order(self, symbol, strike_price, option_type, action, quantity,
                          order_type=None, price=None, expiry=None, exchange=None):
        """
        Place an options order.
        
        Args:
            symbol (str): Symbol (underlying)
            strike_price (float): Strike price
            option_type (str): Option type ('CE' or 'PE')
            action (str): Action (BUY or SELL)
            quantity (int): Quantity
            order_type (str): Order type (optional)
            price (float): Price (optional)
            expiry (datetime): Expiry date (optional)
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
            
        if option_type not in ["CE", "PE"]:
            self.logger.error(f"Invalid option type: {option_type}")
            return None
        
        # Get option contract
        contract = self._get_option_contract(symbol, strike_price, option_type, exchange, expiry)
        
        if not contract:
            self.logger.error(f"Option contract not found: {symbol} {strike_price} {option_type}")
            return None
            
        contract_symbol = contract.get('tradingsymbol')
        
        if not contract_symbol:
            self.logger.error(f"No trading symbol for option contract: {symbol} {strike_price} {option_type}")
            return None
        
        # Check if symbol is active
        symbol_key = f"{contract_symbol}:{exchange}"
        if symbol_key not in self.active_symbols:
            self.logger.warning(f"Symbol {symbol_key} not in active symbols")
            
            # Add option contract automatically
            if not self.add_option_contract(symbol, strike_price, option_type, expiry, exchange):
                self.logger.error(f"Failed to add option contract: {symbol} {strike_price} {option_type}")
                return None
        
        # Get current price if not provided
        if not price and self.engine.market_data_connector:
            data = self.engine.market_data_connector.get_last_price(contract_symbol, exchange)
            
            if data:
                price = data.get('last_price')
                
                if not price:
                    self.logger.error(f"No price data for {contract_symbol}")
                    return None
        
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
            self.logger.info(f"Placed {action} option order for {contract_symbol}: {quantity} @ {price}")
            
            # Return a placeholder result since we don't have the actual order ID yet
            return {
                'symbol': contract_symbol,
                'exchange': exchange,
                'action': action,
                'quantity': quantity,
                'price': price,
                'status': 'QUEUED',
                'underlying': symbol,
                'strike_price': strike_price,
                'option_type': option_type
            }
        else:
            self.logger.error("Trading engine does not have order queue")
            return None
    
    def get_option_position(self, symbol, strike_price=None, option_type=None, exchange=None):
        """
        Get option position details.
        
        Args:
            symbol (str): Symbol (can be contract symbol or underlying)
            strike_price (float): Strike price (optional)
            option_type (str): Option type (optional)
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Position details
        """
        if not self.engine:
            return None
            
        exchange = exchange or self.config['default_exchange']
        
        # Check if symbol is a contract symbol
        symbol_key = f"{symbol}:{exchange}"
        if symbol_key in self.active_symbols:
            return self.engine.get_position_details(symbol, exchange)
            
        # If strike_price and option_type are provided, get specific contract
        if strike_price is not None and option_type:
            contract = self._get_option_contract(symbol, strike_price, option_type, exchange)
            
            if not contract:
                return None
                
            contract_symbol = contract.get('tradingsymbol')
            
            if not contract_symbol:
                return None
                
            return self.engine.get_position_details(contract_symbol, exchange)
            
        # Otherwise, get all positions for the underlying
        all_positions = self.engine.get_open_positions()
        
        for position in all_positions:
            position_symbol = position.get('symbol')
            
            if not position_symbol:
                continue
                
            # Check if this position is for an option of the underlying
            contract_key = f"{position_symbol}:{exchange}"
            contract = self.options_info.get(contract_key)
            
            if contract and contract.get('symbol') == symbol:
                return position
                
        return None
    
    def get_all_option_positions(self):
        """
        Get all option positions.
        
        Returns:
            list: All option positions
        """
        if not self.engine:
            return []
            
        # Get all positions
        all_positions = self.engine.get_open_positions()
        
        # Filter for option positions
        option_positions = []
        
        for position in all_positions:
            symbol = position.get('symbol')
            exchange = position.get('exchange')
            
            if not symbol or not exchange:
                continue
                
            symbol_key = f"{symbol}:{exchange}"
            
            # Check if this is an option position
            if symbol_key in self.active_symbols:
                # Add option details
                contract = self.options_info.get(symbol_key)
                
                if contract:
                    position.update({
                        'underlying': contract.get('symbol'),
                        'strike_price': contract.get('strike_price'),
                        'option_type': contract.get('option_type'),
                        'expiry_date': contract.get('expiry_date'),
                        'greeks': contract.get('greeks', {})
                    })
                
                option_positions.append(position)
        
        return option_positions
    
    def close_option_position(self, symbol, strike_price=None, option_type=None, exchange=None, reason='manual_exit'):
        """
        Close an option position.
        
        Args:
            symbol (str): Symbol (can be contract symbol or underlying)
            strike_price (float): Strike price (optional)
            option_type (str): Option type (optional)
            exchange (str): Exchange (optional)
            reason (str): Exit reason
            
        Returns:
            bool: Success status
        """
        if not self.engine:
            return False
            
        exchange = exchange or self.config['default_exchange']
        
        # Check if symbol is a contract symbol
        symbol_key = f"{symbol}:{exchange}"
        if symbol_key in self.active_symbols:
            return self.engine.close_position(symbol, exchange, reason)
            
        # If strike_price and option_type are provided, close specific contract
        if strike_price is not None and option_type:
            contract = self._get_option_contract(symbol, strike_price, option_type, exchange)
            
            if not contract:
                return False
                
            contract_symbol = contract.get('tradingsymbol')
            
            if not contract_symbol:
                return False
                
            return self.engine.close_position(contract_symbol, exchange, reason)
            
        # Otherwise, close all positions for the underlying
        closed = False
        all_positions = self.engine.get_open_positions()
        
        for position in all_positions:
            position_symbol = position.get('symbol')
            position_exchange = position.get('exchange')
            
            if not position_symbol or not position_exchange:
                continue
                
            # Check if this position is for an option of the underlying
            contract_key = f"{position_symbol}:{position_exchange}"
            contract = self.options_info.get(contract_key)
            
            if contract and contract.get('symbol') == symbol:
                if self.engine.close_position(position_symbol, position_exchange, reason):
                    closed = True
                    
        return closed
    
    def close_all_option_positions(self, reason='manual_exit'):
        """
        Close all option positions.
        
        Args:
            reason (str): Exit reason
            
        Returns:
            int: Number of positions closed
        """
        closed_count = 0
        
        # Get all option positions
        positions = self.get_all_option_positions()
        
        # Close each position
        for position in positions:
            symbol = position.get('symbol')
            exchange = position.get('exchange')
            
            if symbol and exchange:
                if self.engine.close_position(symbol, exchange, reason):
                    closed_count += 1
        
        return closed_count
    
    def get_option_chain(self, symbol, expiry=None, exchange=None):
        """
        Get option chain for a symbol.
        
        Args:
            symbol (str): Symbol (underlying)
            expiry (datetime): Expiry date (optional)
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Option chain
        """
        try:
            exchange = exchange or self.config['default_exchange']
            
            # Get option contracts
            query = {
                'symbol': symbol,
                'exchange': exchange,
                'instrument_type': 'OPT',
                'expiry_date': {'$gte': datetime.now()}  # Only active contracts
            }
            
            # Add expiry if provided
            if expiry:
                if isinstance(expiry, str):
                    try:
                        expiry = datetime.strptime(expiry, '%Y-%m-%d')
                    except Exception as e:
                        self.logger.error(f"Invalid expiry date format: {e}")
                        return None
                        
                query['expiry_date'] = expiry
                
            # Get all contracts sorted by expiry, then strike
            contracts = list(self.db.options_contracts_collection.find(query).sort(
                [('expiry_date', 1), ('strike_price', 1)]
            ))
            
            if not contracts:
                self.logger.warning(f"No option contracts found for {symbol}")
                return None
                
            # Group by expiry
            expirations = {}
            
            for contract in contracts:
                expiry_date = contract.get('expiry_date')
                
                if not expiry_date:
                    continue
                    
                expiry_str = expiry_date.strftime('%Y-%m-%d')
                
                if expiry_str not in expirations:
                    expirations[expiry_str] = {
                        'calls': [],
                        'puts': []
                    }
                    
                # Add contract to appropriate list
                option_type = contract.get('option_type')
                
                if option_type == 'CE':
                    expirations[expiry_str]['calls'].append(contract)
                elif option_type == 'PE':
                    expirations[expiry_str]['puts'].append(contract)
            
            # Get underlying price
            underlying_price = None
            
            if self.engine and self.engine.market_data_connector:
                data = self.engine.market_data_connector.get_last_price(symbol, 'NSE')
                
                if data:
                    underlying_price = data.get('last_price')
            
            # Create option chain
            option_chain = {
                'underlying': symbol,
                'underlying_price': underlying_price,
                'expirations': expirations
            }
            
            return option_chain
            
        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            return None
    
    def get_option_chain_df(self, symbol, expiry=None, exchange=None):
        """
        Get option chain as a pandas DataFrame.
        
        Args:
            symbol (str): Symbol (underlying)
            expiry (datetime): Expiry date (optional)
            exchange (str): Exchange (optional)
            
        Returns:
            DataFrame: Option chain
        """
        try:
            # Get option chain
            chain = self.get_option_chain(symbol, expiry, exchange)
            
            if not chain:
                return None
                
            # Select expiry
            expirations = chain.get('expirations', {})
            
            if not expirations:
                return None
                
            # If expiry not specified, use first expiry
            if expiry is None:
                expiry_str = list(expirations.keys())[0]
            elif isinstance(expiry, datetime):
                expiry_str = expiry.strftime('%Y-%m-%d')
            else:
                expiry_str = expiry
                
            # Get contracts for this expiry
            expiry_data = expirations.get(expiry_str)
            
            if not expiry_data:
                return None
                
            calls = expiry_data.get('calls', [])
            puts = expiry_data.get('puts', [])
            
            # Create DataFrame
            call_data = []
            for contract in calls:
                call_data.append({
                    'strike': contract.get('strike_price'),
                    'symbol': contract.get('tradingsymbol'),
                    'bid': contract.get('bid', 0),
                    'ask': contract.get('ask', 0),
                    'last': contract.get('current_price', 0),
                    'volume': contract.get('volume', 0),
                    'oi': contract.get('open_interest', 0),
                    'iv': contract.get('implied_volatility', 0),
                    'delta': contract.get('greeks', {}).get('delta', 0),
                    'gamma': contract.get('greeks', {}).get('gamma', 0),
                    'theta': contract.get('greeks', {}).get('theta', 0),
                    'vega': contract.get('greeks', {}).get('vega', 0)
                })
                
            put_data = []
            for contract in puts:
                put_data.append({
                    'strike': contract.get('strike_price'),
                    'symbol': contract.get('tradingsymbol'),
                    'bid': contract.get('bid', 0),
                    'ask': contract.get('ask', 0),
                    'last': contract.get('current_price', 0),
                    'volume': contract.get('volume', 0),
                    'oi': contract.get('open_interest', 0),
                    'iv': contract.get('implied_volatility', 0),
                    'delta': contract.get('greeks', {}).get('delta', 0),
                    'gamma': contract.get('greeks', {}).get('gamma', 0),
                    'theta': contract.get('greeks', {}).get('theta', 0),
                    'vega': contract.get('greeks', {}).get('vega', 0)
                })
                
            # Create DataFrames
            calls_df = pd.DataFrame(call_data) if call_data else pd.DataFrame()
            puts_df = pd.DataFrame(put_data) if put_data else pd.DataFrame()
            
            # Sort by strike
            if not calls_df.empty:
                calls_df = calls_df.sort_values('strike')
                
            if not puts_df.empty:
                puts_df = puts_df.sort_values('strike')
                
            # Merge into a single DataFrame
            merged_df = pd.DataFrame()
            
            if not calls_df.empty and not puts_df.empty:
                # Rename columns
                calls_df = calls_df.add_prefix('call_')
                puts_df = puts_df.add_prefix('put_')
                
                # Reset index for merge
                calls_df = calls_df.reset_index(drop=True)
                puts_df = puts_df.reset_index(drop=True)
                
                # Move strike columns
                calls_df['strike'] = calls_df['call_strike']
                calls_df = calls_df.drop('call_strike', axis=1)
                
                puts_df['strike'] = puts_df['put_strike']
                puts_df = puts_df.drop('put_strike', axis=1)
                
                # Merge on strike
                merged_df = pd.merge(calls_df, puts_df, on='strike', how='outer')
                merged_df = merged_df.sort_values('strike')
            elif not calls_df.empty:
                merged_df = calls_df
            elif not puts_df.empty:
                merged_df = puts_df
                
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error getting option chain DataFrame: {e}")
            return None
    
    def get_option_greeks(self, symbol, strike_price, option_type, expiry=None, exchange=None):
        """
        Get option Greeks.
        
        Args:
            symbol (str): Symbol (underlying)
            strike_price (float): Strike price
            option_type (str): Option type ('CE' or 'PE')
            expiry (datetime): Expiry date (optional)
            exchange (str): Exchange (optional)
            
        Returns:
            dict: Option Greeks
        """
        exchange = exchange or self.config['default_exchange']
        
        # Get option contract
        contract = self._get_option_contract(symbol, strike_price, option_type, exchange, expiry)
        
        if not contract:
            return None
            
        # Get Greeks from contract
        greeks = contract.get('greeks')
        
        if not greeks:
            # Calculate Greeks
            underlying_price = None
            
            if self.engine and self.engine.market_data_connector:
                data = self.engine.market_data_connector.get_last_price(symbol, 'NSE')
                
                if data:
                    underlying_price = data.get('last_price')
                    
                    if underlying_price:
                        # Get option price
                        contract_symbol = contract.get('tradingsymbol')
                        
                        if contract_symbol:
                            option_data = self.engine.market_data_connector.get_last_price(contract_symbol, exchange)
                            
                            if option_data:
                                option_price = option_data.get('last_price')
                                
                                if option_price:
                                    # Calculate days to expiry
                                    days_to_expiry = (contract.get('expiry_date') - datetime.now()).days + (datetime.now().hour / 24.0)
                                    
                                    # Estimate IV
                                    iv = self._estimate_implied_volatility(
                                        option_price=option_price,
                                        underlying_price=underlying_price,
                                        strike_price=strike_price,
                                        days_to_expiry=days_to_expiry,
                                        option_type=option_type
                                    )
                                    
                                    if iv:
                                        # Calculate Greeks
                                        greeks = self._calculate_option_greeks(
                                            underlying_price=underlying_price,
                                            strike_price=strike_price,
                                            days_to_expiry=days_to_expiry,
                                            volatility=iv,
                                            option_type=option_type
                                        )
                                        
                                        # Update contract
                                        contract['greeks'] = greeks
                                        contract['implied_volatility'] = iv
                                        contract['current_price'] = option_price
                                        contract['underlying_price'] = underlying_price
                                        contract['updated_at'] = datetime.now()
                                        
                                        # Store updated contract
                                        contract_key = f"{contract_symbol}:{exchange}"
                                        self.options_info[contract_key] = contract
                                        
                                        # Update in database
                                        self._update_option_contract_in_db(contract)
        
        return greeks
    
    def get_option_performance(self):
        """
        Get options trading performance.
        
        Returns:
            dict: Performance metrics
        """
        if not self.engine:
            return {}
            
        # Get all performance metrics
        all_metrics = self.engine.get_performance_metrics()
        
        # Filter trade history for options trades
        if hasattr(self.engine, 'trade_history'):
            options_trades = []
            
            for trade in self.engine.trade_history:
                symbol = trade.get('symbol')
                exchange = trade.get('exchange')
                
                if not symbol or not exchange:
                    continue
                    
                symbol_key = f"{symbol}:{exchange}"
                
                # Check if this was an options trade
                if any(symbol in s for s in self.active_symbols) or exchange == self.config['default_exchange']:
                    options_trades.append(trade)
            
            # Calculate options-specific metrics
            if options_trades:
                total_trades = len(options_trades)
                winning_trades = [t for t in options_trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in options_trades if t.get('profit_loss', 0) <= 0]
                
                win_count = len(winning_trades)
                loss_count = len(losing_trades)
                
                win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
                
                total_profit = sum(t.get('profit_loss', 0) for t in winning_trades)
                total_loss = abs(sum(t.get('profit_loss', 0) for t in losing_trades))
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                # Calculate by option type
                call_trades = [t for t in options_trades if self._is_call_option(t.get('symbol'))]
                put_trades = [t for t in options_trades if self._is_put_option(t.get('symbol'))]
                
                call_win_rate = (len([t for t in call_trades if t.get('profit_loss', 0) > 0]) / len(call_trades)) * 100 if call_trades else 0
                put_win_rate = (len([t for t in put_trades if t.get('profit_loss', 0) > 0]) / len(put_trades)) * 100 if put_trades else 0
                
                # Update metrics
                all_metrics.update({
                    'options_trades': total_trades,
                    'options_win_rate': win_rate,
                    'options_profit_factor': profit_factor,
                    'options_total_profit': total_profit,
                    'options_total_loss': total_loss,
                    'call_trades': len(call_trades),
                    'call_win_rate': call_win_rate,
                    'put_trades': len(put_trades),
                    'put_win_rate': put_win_rate
                })
        
        return all_metrics
    
    def _is_call_option(self, symbol):
        """
        Check if a symbol is a call option.
        
        Args:
            symbol (str): Symbol
            
        Returns:
            bool: True if call option
        """
        # Try to find contract
        for contract in self.options_info.values():
            if contract.get('tradingsymbol') == symbol and contract.get('option_type') == 'CE':
                return True
                
        # Check if symbol contains CE
        return 'CE' in symbol
    
    def _is_put_option(self, symbol):
        """
        Check if a symbol is a put option.
        
        Args:
            symbol (str): Symbol
            
        Returns:
            bool: True if put option
        """
        # Try to find contract
        for contract in self.options_info.values():
            if contract.get('tradingsymbol') == symbol and contract.get('option_type') == 'PE':
                return True
                
        # Check if symbol contains PE
        return 'PE' in symbol
                