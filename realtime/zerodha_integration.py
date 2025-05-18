# realtime/zerodha_integration.py
import logging
import time
import json
import hashlib
import threading
import requests
from datetime import datetime, timedelta
from kiteconnect import KiteConnect, KiteTicker

class ZerodhaConnector:
    """
    Zerodha integration for market data and order execution.
    """
    
    def __init__(self, api_key, api_secret, access_token=None, logger=None):
        """
        Initialize Zerodha connector.
        
        Args:
            api_key (str): Zerodha API key
            api_secret (str): Zerodha API secret
            access_token (str): Zerodha access token (optional)
            logger: Logger instance
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.logger = logger or logging.getLogger(__name__)
        
        # Kite instances
        self.kite = None
        self.ticker = None
        
        # State
        self.is_connected = False
        self.connection_time = None
        self.subscribed_tokens = {}
        self.token_symbol_map = {}
        self.symbol_token_map = {}
        
        # Data callbacks
        self.data_callback = None
        self.order_callback = None
        
        # Caches
        self.instrument_cache = {}
        self.ltp_cache = {}
        self.order_cache = {}
        
        # Initialize
        self._initialize()
        
        self.logger.info("Zerodha connector initialized")
    
    def _initialize(self):
        """
        Initialize Kite instance.
        """
        try:
            # Create Kite instance
            self.kite = KiteConnect(api_key=self.api_key)
            
            # Set access token if provided
            if self.access_token:
                self.kite.set_access_token(self.access_token)
                self.is_connected = True
                self.connection_time = datetime.now()
                
                # Initialize instrument cache
                self._initialize_instruments()
            
        except Exception as e:
            self.logger.error(f"Error initializing Zerodha connector: {e}")
    
    def _initialize_instruments(self):
        """
        Initialize instrument cache.
        """
        try:
            # Get all instruments
            instruments = self.kite.instruments()
            
            # Build caches
            for instrument in instruments:
                tradingsymbol = instrument.get('tradingsymbol')
                exchange = instrument.get('exchange')
                instrument_token = instrument.get('instrument_token')
                
                if tradingsymbol and exchange and instrument_token:
                    # Store in cache
                    key = f"{tradingsymbol}:{exchange}"
                    self.instrument_cache[key] = instrument
                    
                    # Build token maps
                    self.token_symbol_map[instrument_token] = key
                    self.symbol_token_map[key] = instrument_token
            
            self.logger.info(f"Initialized {len(self.instrument_cache)} instruments")
            
        except Exception as e:
            self.logger.error(f"Error initializing instruments: {e}")
    
    def login(self):
        """
        Login to Zerodha (generate request token).
        
        Returns:
            str: Login URL
        """
        try:
            # Generate login URL
            login_url = self.kite.login_url()
            
            self.logger.info(f"Login URL generated: {login_url}")
            
            return login_url
            
        except Exception as e:
            self.logger.error(f"Error generating login URL: {e}")
            return None
    
    def generate_session(self, request_token):
        """
        Generate session from request token.
        
        Args:
            request_token (str): Request token
            
        Returns:
            bool: Success status
        """
        try:
            # Generate session
            data = self.kite.generate_session(
                request_token=request_token,
                api_secret=self.api_secret
            )
            
            # Set access token
            access_token = data.get('access_token')
            
            if not access_token:
                self.logger.error("No access token received")
                return False
                
            self.access_token = access_token
            self.kite.set_access_token(access_token)
            
            self.is_connected = True
            self.connection_time = datetime.now()
            
            # Initialize instrument cache
            self._initialize_instruments()
            
            self.logger.info("Session generated successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error generating session: {e}")
            return False
    
    def is_connected(self):
        """
        Check if connected to Zerodha.
        
        Returns:
            bool: Connection status
        """
        if not self.is_connected or not self.access_token:
            return False
            
        # Check if access token is expired (tokens are valid for a day)
        if self.connection_time:
            now = datetime.now()
            elapsed = now - self.connection_time
            
            if elapsed > timedelta(hours=6):
                # Token might be close to expiry, verify by making a simple API call
                try:
                    self.kite.margins()
                    return True
                except Exception:
                    self.is_connected = False
                    return False
        
        return self.is_connected
    
    def subscribe(self, symbols, exchanges=None):
        """
        Subscribe to market data for symbols.
        
        Args:
            symbols (list): List of symbols
            exchanges (list): List of exchanges (optional)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.is_connected():
                self.logger.error("Not connected to Zerodha")
                return False
                
            # Create ticker if not exists
            if not self.ticker:
                self.ticker = KiteTicker(self.api_key, self.access_token)
                self.ticker.on_ticks = self._on_ticks
                self.ticker.on_connect = self._on_connect
                self.ticker.on_close = self._on_close
                self.ticker.on_error = self._on_error
                
                # Start ticker in a separate thread
                threading.Thread(target=self.ticker.connect).start()
                
                # Wait for connection
                start_time = time.time()
                while not self.ticker.is_connected() and time.time() - start_time < 10:
                    time.sleep(0.1)
                    
                if not self.ticker.is_connected():
                    self.logger.error("Ticker connection timeout")
                    return False
            
            # Get instrument tokens for symbols
            tokens = []
            
            for i, symbol in enumerate(symbols):
                exchange = exchanges[i] if exchanges and i < len(exchanges) else "NSE"
                
                # Build key
                key = f"{symbol}:{exchange}"
                
                # Get token from cache
                token = self.symbol_token_map.get(key)
                
                if not token:
                    # Try to find instrument
                    instrument = self._find_instrument(symbol, exchange)
                    
                    if not instrument:
                        self.logger.warning(f"Instrument not found: {symbol}:{exchange}")
                        continue
                        
                    token = instrument.get('instrument_token')
                    
                    if not token:
                        self.logger.warning(f"No token for instrument: {symbol}:{exchange}")
                        continue
                        
                    # Update caches
                    self.instrument_cache[key] = instrument
                    self.token_symbol_map[token] = key
                    self.symbol_token_map[key] = token
                
                tokens.append(token)
                self.subscribed_tokens[token] = {
                    'symbol': symbol,
                    'exchange': exchange
                }
            
            # Subscribe to tokens
            if tokens:
                self.ticker.subscribe(tokens)
                self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
                
                self.logger.info(f"Subscribed to {len(tokens)} symbols")
                
                return True
            else:
                self.logger.warning("No valid symbols to subscribe")
                return False
                
        except Exception as e:
            self.logger.error(f"Error subscribing to market data: {e}")
            return False
    
    def unsubscribe(self, symbols, exchanges=None):
        """
        Unsubscribe from market data for symbols.
        
        Args:
            symbols (list): List of symbols
            exchanges (list): List of exchanges (optional)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.ticker or not self.ticker.is_connected():
                return False
                
            # Get instrument tokens for symbols
            tokens = []
            
            for i, symbol in enumerate(symbols):
                exchange = exchanges[i] if exchanges and i < len(exchanges) else "NSE"
                
                # Build key
                key = f"{symbol}:{exchange}"
                
                # Get token from cache
                token = self.symbol_token_map.get(key)
                
                if token:
                    tokens.append(token)
                    
                    # Remove from subscribed tokens
                    if token in self.subscribed_tokens:
                        del self.subscribed_tokens[token]
            
            # Unsubscribe from tokens
            if tokens:
                self.ticker.unsubscribe(tokens)
                
                self.logger.info(f"Unsubscribed from {len(tokens)} symbols")
                
                return True
            else:
                self.logger.warning("No valid symbols to unsubscribe")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unsubscribing from market data: {e}")
            return False
    
    def disconnect(self):
        """
        Disconnect from Zerodha.
        """
        try:
            if self.ticker:
                self.ticker.close(1000, "Manual disconnect")
                self.ticker = None
                
            self.is_connected = False
            self.logger.info("Disconnected from Zerodha")
            
        except Exception as e:
            self.logger.error(f"Error disconnecting from Zerodha: {e}")
    
    def _on_ticks(self, ws, ticks):
        """
        Callback for tick data.
        
        Args:
            ws: WebSocket instance
            ticks (list): List of tick data
        """
        try:
            for tick in ticks:
                token = tick.get('instrument_token')
                
                if not token:
                    continue
                    
                # Get symbol from token
                symbol_key = self.token_symbol_map.get(token)
                
                if not symbol_key:
                    continue
                    
                parts = symbol_key.split(':')
                
                if len(parts) != 2:
                    continue
                    
                symbol, exchange = parts
                
                # Create market data update
                data = {
                    'symbol': symbol,
                    'exchange': exchange,
                    'last_price': tick.get('last_price'),
                    'volume': tick.get('volume'),
                    'timestamp': datetime.fromtimestamp(tick.get('timestamp', time.time())),
                    'open': tick.get('ohlc', {}).get('open'),
                    'high': tick.get('ohlc', {}).get('high'),
                    'low': tick.get('ohlc', {}).get('low'),
                    'close': tick.get('ohlc', {}).get('close'),
                    'bid': tick.get('depth', {}).get('buy', [{}])[0].get('price'),
                    'ask': tick.get('depth', {}).get('sell', [{}])[0].get('price')
                }
                
                # Update LTP cache
                self.ltp_cache[symbol_key] = data
                
                # Call data callback
                if self.data_callback:
                    self.data_callback(data)
                    
        except Exception as e:
            self.logger.error(f"Error in tick callback: {e}")
    
    def _on_connect(self, ws, response):
        """
        Callback for WebSocket connection.
        
        Args:
            ws: WebSocket instance
            response: Connection response
        """
        self.logger.info("Connected to Zerodha WebSocket")
    
    def _on_close(self, ws, code, reason):
        """
        Callback for WebSocket close.
        
        Args:
            ws: WebSocket instance
            code: Close code
            reason: Close reason
        """
        self.logger.info(f"Disconnected from Zerodha WebSocket: {code} - {reason}")
        
        # Try to reconnect
        threading.Thread(target=self._reconnect_ticker).start()
    
    def _on_error(self, ws, error):
        """
        Callback for WebSocket error.
        
        Args:
            ws: WebSocket instance
            error: Error
        """
        self.logger.error(f"Zerodha WebSocket error: {error}")
    
    def _reconnect_ticker(self, max_retries=5):
        """
        Reconnect to ticker.
        
        Args:
            max_retries (int): Maximum number of retries
        """
        retries = 0
        
        while retries < max_retries:
            try:
                self.logger.info(f"Reconnecting to ticker (attempt {retries+1}/{max_retries})")
                
                # Wait before retrying
                time.sleep(5 * (retries + 1))
                
                # Create new ticker
                self.ticker = KiteTicker(self.api_key, self.access_token)
                self.ticker.on_ticks = self._on_ticks
                self.ticker.on_connect = self._on_connect
                self.ticker.on_close = self._on_close
                self.ticker.on_error = self._on_error
                
                # Connect
                self.ticker.connect()
                
                # Wait for connection
                start_time = time.time()
                while not self.ticker.is_connected() and time.time() - start_time < 10:
                    time.sleep(0.1)
                    
                if self.ticker.is_connected():
                    # Resubscribe to tokens
                    tokens = list(self.subscribed_tokens.keys())
                    
                    if tokens:
                        self.ticker.subscribe(tokens)
                        self.ticker.set_mode(self.ticker.MODE_FULL, tokens)
                        
                        self.logger.info(f"Resubscribed to {len(tokens)} symbols")
                    
                    return True
                    
            except Exception as e:
                self.logger.error(f"Error reconnecting to ticker: {e}")
                
            retries += 1
            
        self.logger.error("Failed to reconnect to ticker after maximum retries")
        return False
    
    def _find_instrument(self, symbol, exchange):
        """
        Find instrument details.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            dict: Instrument details
        """
        try:
            # Check cache first
            key = f"{symbol}:{exchange}"
            
            if key in self.instrument_cache:
                return self.instrument_cache[key]
                
            # Search through instruments
            search_results = self.kite.search_instruments(exchange, symbol)
            
            for instrument in search_results:
                if instrument['tradingsymbol'] == symbol and instrument['exchange'] == exchange:
                    return instrument
                    
            # Not found, try to get exchange instruments
            exchange_instruments = self.kite.instruments(exchange)
            
            for instrument in exchange_instruments:
                if instrument['tradingsymbol'] == symbol:
                    return instrument
                    
            return None
            
        except Exception as e:
            self.logger.error(f"Error finding instrument {symbol}:{exchange}: {e}")
            return None
    
    def set_data_callback(self, callback):
        """
        Set callback for market data updates.
        
        Args:
            callback: Callback function
        """
        self.data_callback = callback
    
    def set_order_callback(self, callback):
        """
        Set callback for order updates.
        
        Args:
            callback: Callback function
        """
        self.order_callback = callback
    
    def get_historical_data(self, symbol, exchange, timeframe, from_date, to_date):
        """
        Get historical data for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            timeframe (str): Timeframe
            from_date (datetime): From date
            to_date (datetime): To date
            
        Returns:
            list: Historical data
        """
        try:
            if not self.is_connected():
                self.logger.error("Not connected to Zerodha")
                return None
                
            # Find instrument token
            key = f"{symbol}:{exchange}"
            token = self.symbol_token_map.get(key)
            
            if not token:
                instrument = self._find_instrument(symbol, exchange)
                
                if not instrument:
                    self.logger.error(f"Instrument not found: {symbol}:{exchange}")
                    return None
                    
                token = instrument.get('instrument_token')
                
                if not token:
                    self.logger.error(f"No token for instrument: {symbol}:{exchange}")
                    return None
                    
                # Update caches
                self.instrument_cache[key] = instrument
                self.token_symbol_map[token] = key
                self.symbol_token_map[key] = token
            
            # Convert timeframe
            interval = self._convert_timeframe(timeframe)
            
            # Get historical data
            data = self.kite.historical_data(
                instrument_token=token,
                from_date=from_date,
                to_date=to_date,
                interval=interval
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return None
    
    def _convert_timeframe(self, timeframe):
        """
        Convert timeframe to Kite format.
        
        Args:
            timeframe (str): Timeframe
            
        Returns:
            str: Kite timeframe
        """
        # Convert to Kite format
        if timeframe == "1min":
            return "minute"
        elif timeframe == "5min":
            return "5minute"
        elif timeframe == "15min":
            return "15minute"
        elif timeframe == "30min":
            return "30minute"
        elif timeframe == "1hour":
            return "60minute"
        elif timeframe == "day":
            return "day"
        else:
            return "minute"  # Default to 1 minute
    
    def get_last_price(self, symbol, exchange):
        """
        Get last price for a symbol.
        
        Args:
            symbol (str): Symbol
            exchange (str): Exchange
            
        Returns:
            dict: Market data
        """
        try:
            key = f"{symbol}:{exchange}"
            
            # Check cache first
            if key in self.ltp_cache:
                return self.ltp_cache[key]
                
            # Not in cache, get from API
            if not self.is_connected():
                self.logger.error("Not connected to Zerodha")
                return None
                
            # Find instrument token
            token = self.symbol_token_map.get(key)
            
            if not token:
                instrument = self._find_instrument(symbol, exchange)
                
                if not instrument:
                    self.logger.error(f"Instrument not found: {symbol}:{exchange}")
                    return None
                    
                token = instrument.get('instrument_token')
                
                if not token:
                    self.logger.error(f"No token for instrument: {symbol}:{exchange}")
                    return None
                    
                # Update caches
                self.instrument_cache[key] = instrument
                self.token_symbol_map[token] = key
                self.symbol_token_map[key] = token
            
            # Get last price
            ltp = self.kite.ltp([f"{exchange}:{symbol}"])
            
            if not ltp:
                return None
                
            quote_key = f"{exchange}:{symbol}"
            quote = ltp.get(quote_key)
            
            if not quote:
                return None
                
            # Create market data
            data = {
                'symbol': symbol,
                'exchange': exchange,
                'last_price': quote.get('last_price'),
                'timestamp': datetime.now()
            }
            
            # Update cache
            self.ltp_cache[key] = data
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error getting last price for {symbol}:{exchange}: {e}")
            return None
    
    def place_order(self, order):
        """
        Place an order.
        
        Args:
            order (dict): Order details
            
        Returns:
            dict: Order result
        """
        try:
            if not self.is_connected():
                self.logger.error("Not connected to Zerodha")
                return None
                
            # Extract order details
            symbol = order.get('symbol')
            exchange = order.get('exchange', 'NSE')
            action = order.get('action')
            quantity = order.get('quantity', 0)
            price = order.get('price')
            order_type = order.get('order_type', 'LIMIT')
            product_type = order.get('product_type', 'CNC')
            
            if not symbol or not action or quantity <= 0:
                self.logger.error("Invalid order details")
                return None
                
            # Convert order type
            if order_type == "MARKET":
                kite_order_type = self.kite.ORDER_TYPE_MARKET
            elif order_type == "LIMIT":
                kite_order_type = self.kite.ORDER_TYPE_LIMIT
            elif order_type == "SL":
                kite_order_type = self.kite.ORDER_TYPE_SL
            elif order_type == "SL-M":
                kite_order_type = self.kite.ORDER_TYPE_SLM
            else:
                kite_order_type = self.kite.ORDER_TYPE_LIMIT
                
            # Convert action
            if action == "BUY":
                kite_transaction_type = self.kite.TRANSACTION_TYPE_BUY
            elif action == "SELL":
                kite_transaction_type = self.kite.TRANSACTION_TYPE_SELL
            else:
                self.logger.error(f"Invalid action: {action}")
                return None
                
            # Convert product type
            if product_type == "CNC":
                kite_product = self.kite.PRODUCT_CNC
            elif product_type == "MIS":
                kite_product = self.kite.PRODUCT_MIS
            elif product_type == "NRML":
                kite_product = self.kite.PRODUCT_NRML
            else:
                kite_product = self.kite.PRODUCT_CNC
            
            # Place order
            order_params = {
                "tradingsymbol": symbol,
                "exchange": exchange,
                "transaction_type": kite_transaction_type,
                "quantity": quantity,
                "order_type": kite_order_type,
                "product": kite_product
            }
            
            # Add price for limit orders
            if kite_order_type == self.kite.ORDER_TYPE_LIMIT:
                order_params["price"] = price
                
            # Add trigger price for SL orders
            if kite_order_type in [self.kite.ORDER_TYPE_SL, self.kite.ORDER_TYPE_SLM]:
                trigger_price = order.get('trigger_price')
                
                if trigger_price:
                    order_params["trigger_price"] = trigger_price
            
            # Place order
            order_id = self.kite.place_order(**order_params)
            
            if not order_id:
                self.logger.error("No order ID received")
                return None
                
            # Create order result
            result = {
                'order_id': order.get('order_id'),
                'broker_order_id': order_id,
                'status': 'PENDING'
            }
            
            # Store in cache
            self.order_cache[order_id] = {
                'order_id': order.get('order_id'),
                'broker_order_id': order_id,
                'symbol': symbol,
                'exchange': exchange,
                'action': action,
                'quantity': quantity,
                'price': price,
                'order_type': order_type,
                'product_type': product_type,
                'status': 'PENDING',
                'position_id': order.get('position_id')
            }
            
            # Start order tracking thread
            threading.Thread(target=self._track_order, args=(order_id,)).start()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def _track_order(self, broker_order_id, interval=2, max_retries=30):
        """
        Track order status.
        
        Args:
            broker_order_id (str): Broker order ID
            interval (int): Check interval in seconds
            max_retries (int): Maximum number of retries
        """
        retries = 0
        
        while retries < max_retries:
            try:
                # Sleep before checking
                time.sleep(interval)
                
                # Get order details
                order = self.kite.order_history(broker_order_id)
                
                if not order:
                    retries += 1
                    continue
                    
                # Get latest status
                latest = order[-1]
                status = latest.get('status')
                
                if not status:
                    retries += 1
                    continue
                    
                # Check if order is complete or rejected
                if status in ['COMPLETE', 'REJECTED', 'CANCELLED']:
                    # Get cached order
                    cached_order = self.order_cache.get(broker_order_id)
                    
                    if not cached_order:
                        return
                        
                    # Update order status
                    cached_order['status'] = status
                    cached_order['executed_price'] = latest.get('average_price')
                    cached_order['executed_quantity'] = latest.get('filled_quantity')
                    cached_order['execution_time'] = datetime.now()
                    
                    # Call order callback
                    if self.order_callback:
                        self.order_callback(cached_order)
                        
                    return
                    
            except Exception as e:
                self.logger.error(f"Error tracking order {broker_order_id}: {e}")
                
            retries += 1
    
    def get_order_status(self, broker_order_id):
        """
        Get order status.
        
        Args:
            broker_order_id (str): Broker order ID
            
        Returns:
            dict: Order status
        """
        try:
            if not self.is_connected():
                return None
                
            # Check cache first
            if broker_order_id in self.order_cache:
                return self.order_cache[broker_order_id]
                
            # Get from API
            order = self.kite.order_history(broker_order_id)
            
            if not order:
                return None
                
            # Get latest status
            latest = order[-1]
            
            # Create order status
            status = {
                'broker_order_id': broker_order_id,
                'status': latest.get('status'),
                'executed_price': latest.get('average_price'),
                'executed_quantity': latest.get('filled_quantity'),
                'pending_quantity': latest.get('pending_quantity')
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    def cancel_order(self, broker_order_id):
        """
        Cancel an order.
        
        Args:
            broker_order_id (str): Broker order ID
            
        Returns:
            bool: Success status
        """
        try:
            if not self.is_connected():
                return False
                
            # Cancel order
            self.kite.cancel_order(broker_order_id)
            
            # Update cache
            if broker_order_id in self.order_cache:
                self.order_cache[broker_order_id]['status'] = 'CANCELLED'
                
                # Call order callback
                if self.order_callback:
                    self.order_callback(self.order_cache[broker_order_id])
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def get_account_info(self):
        """
        Get account information.
        
        Returns:
            dict: Account information
        """
        try:
            if not self.is_connected():
                return None
                
            # Get profile
            profile = self.kite.profile()
            
            # Get margins
            margins = self.kite.margins()
            
            # Create account info
            account_info = {
                'user_id': profile.get('user_id'),
                'user_name': profile.get('user_name'),
                'email': profile.get('email'),
                'balance': margins.get('equity', {}).get('available', {}).get('cash', 0),
                'margins': margins
            }
            
            return account_info
            
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return None