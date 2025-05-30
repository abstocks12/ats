"""
Zerodha Connector Module - Provides connection to Zerodha APIs
"""

import os
import time
import logging
import json
from datetime import datetime, timedelta
from kiteconnect import KiteConnect
from kiteconnect import KiteTicker

class ZerodhaConnector:
    """
    Provides connection to Zerodha Kite APIs for market data and trading.
    Handles authentication, rate limiting, and API call management.
    """
    
    def __init__(self, api_key=None, api_secret=None, access_token=None, auto_reconnect=True):
        """
        Initialize the Zerodha connector
        
        Args:
            api_key (str, optional): Zerodha API key (default: from environment)
            api_secret (str, optional): Zerodha API secret (default: from environment)
            access_token (str, optional): Access token (default: from saved file or None)
            auto_reconnect (bool, optional): Whether to auto reconnect websocket (default: True)
        """
        self.logger = logging.getLogger(__name__)
        
        # Get API credentials
        self.api_key = api_key or os.environ.get('ZERODHA_API_KEY')
        self.api_secret = api_secret or os.environ.get('ZERODHA_API_SECRET')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Zerodha API credentials not provided. Set ZERODHA_API_KEY and ZERODHA_API_SECRET environment variables or pass them as parameters")
        
        # Create a directory for storing tokens
        self.token_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'auth_tokens'
        )
        os.makedirs(self.token_dir, exist_ok=True)
        
        self.token_file = os.path.join(self.token_dir, 'zerodha_token.json')
        self.alt_token_file = os.path.join('config', 'zerodha_token.txt')
        
        # Initialize KiteConnect instance
        self.kite = KiteConnect(api_key=self.api_key)
        
        # Initialize access token
        self.access_token = access_token
        if not self.access_token:
            self._load_access_token()
        
        # Set the access token
        if self.access_token:
            self.kite.set_access_token(self.access_token)
            self.logger.info("Access token set successfully")
        else:
            # Try to authenticate
            self.authenticate()

        # Initialize ticker (websocket connection)
        self.ticker = None
        self.ticker_connected = False
        self.auto_reconnect = auto_reconnect
        
        # Rate limiting
        self.last_api_call_time = 0
        self.min_time_between_calls = 0.5  # seconds
        
        # Initialize instrument mapping
        self.instrument_tokens = {}
        self.token_to_instrument = {}
        
        # Initialize callbacks
        self.on_tick_callbacks = []
        self.on_connection_callbacks = []
        
        # Load instruments if access token is available
        if self.access_token:
            self._load_instruments()
    
    def _load_access_token(self):
        """Load access token from file"""
        try:
            if os.path.exists(self.token_file):
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                
                # Check if token is expired (valid for one day)
                expiry_time = datetime.fromisoformat(token_data.get('expiry', '2000-01-01'))
                if expiry_time > datetime.now():
                    self.access_token = token_data.get('access_token')
                    self.logger.info("Loaded valid access token from file")
                else:
                    self.logger.warning("Access token expired, needs re-authentication")
        except Exception as e:
            self.logger.error(f"Error loading access token: {e}")
    
    def _save_access_token(self):
        """Save access token to file"""
        try:
            token_data = {
                'access_token': self.access_token,
                'api_key': self.api_key,
                'expiry': (datetime.now() + timedelta(days=1)).isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(token_data, f)
            
            self.logger.info("Saved access token to file")
        except Exception as e:
            self.logger.error(f"Error saving access token: {e}")
    
    def authenticate(self, request_token=None):
        """
        Authenticate with Zerodha
        
        Args:
            request_token (str, optional): Request token from manual login process
            
        Returns:
            bool: True if authenticated successfully
        """
        # If we already have a valid access token, we're good
        if self.access_token:
            try:
                # Test if token is valid by getting profile
                self.kite.profile()
                self.logger.info("Using existing access token")
                return True
            except Exception:
                self.logger.warning("Existing access token is invalid, will try to re-authenticate")
                self.access_token = None
        
        # If we have a request token, use it to generate a session
        if request_token:
            return self.generate_session(request_token)
        
        # No valid token or request token, prompt for manual login
        login_url = self.generate_login_url()
        self.logger.info(f"""
        ==== Zerodha Authentication Required ====
        Please complete these steps to authenticate:
        
        1. Open this URL in your browser: {login_url}
        2. Login with your Zerodha credentials
        3. You will be redirected to a URL containing the request token
        4. Run this command with the request token:
        python scripts/zerodha_login.py --request-token YOUR_TOKEN
        """)
        
        return False

    def _load_instruments(self):
        """Load instrument details from Zerodha"""
        try:
            self.logger.info("Loading instruments from Zerodha...")
            instruments = self.kite.instruments()
            
            if not instruments:
                self.logger.error("No instruments returned from Zerodha API")
                return
                
            self.logger.info(f"Received {len(instruments)} instruments from API")
            
            # Map instruments to tokens
            for instrument in instruments:
                key = f"{instrument['tradingsymbol']}:{instrument['exchange']}"
                self.instrument_tokens[key] = instrument['instrument_token']
                self.token_to_instrument[instrument['instrument_token']] = {
                    'symbol': instrument['tradingsymbol'],
                    'exchange': instrument['exchange'],
                    'lot_size': instrument.get('lot_size', 1),
                    'tick_size': instrument.get('tick_size', 0.05),
                    'instrument_type': instrument.get('instrument_type', 'EQ')
                }
            
            self.logger.info(f"Loaded {len(instruments)} instruments from Zerodha")
            
            # Check specifically for common symbols
            test_symbols = ["RELIANCE", "TCS", "INFY", "NBCC"]
            for symbol in test_symbols:
                test_key = f"{symbol}:NSE"
                if test_key in self.instrument_tokens:
                    self.logger.info(f"{test_key} found with token {self.instrument_tokens[test_key]}")
                else:
                    self.logger.warning(f"{test_key} not found in loaded instruments!")
        except Exception as e:
            self.logger.error(f"Error loading instruments: {e}")
    
    def _rate_limit(self):
        """Apply rate limiting for API calls"""
        current_time = time.time()
        elapsed = current_time - self.last_api_call_time
        
        if elapsed < self.min_time_between_calls:
            sleep_time = self.min_time_between_calls - elapsed
            time.sleep(sleep_time)
        
        self.last_api_call_time = time.time()
    
    def generate_login_url(self):
        """
        Generate login URL for Zerodha authentication
        
        Returns:
            str: Login URL
        """
        return self.kite.login_url()
    
    def generate_session(self, request_token):
        """
        Generate session from request token
        
        Args:
            request_token (str): Request token from Zerodha callback
            
        Returns:
            bool: True if successful
        """
        try:
            data = self.kite.generate_session(request_token, api_secret=self.api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            
            # Save the token for future use
            self._save_access_token()
            
            # Load instruments
            self._load_instruments()
            
            self.logger.info("Successfully generated session and set access token")
            return True
        except Exception as e:
            self.logger.error(f"Error generating session: {e}")
            return False
    
    def get_quote(self, symbols, exchanges=None):
        """
        Get quotes for instruments
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges corresponding to symbols
            
        Returns:
            dict: Quote data
        """
        try:
            self._rate_limit()
            
            # Format the instruments in exchange:symbol format
            instruments = []
            if exchanges:
                for symbol, exchange in zip(symbols, exchanges):
                    instruments.append(f"{exchange}:{symbol}")
            else:
                # Default to NSE
                for symbol in symbols:
                    instruments.append(f"NSE:{symbol}")
            
            quotes = self.kite.quote(instruments)
            return quotes
        except Exception as e:
            self.logger.error(f"Error getting quotes: {e}")
            return {}
    
    
    def get_historical_data(self, symbol, exchange, interval_input, from_date=None, to_date=None, continuous=0, oi=0):
        """
        Collect historical data for a specific instrument and timeframe
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            interval_input (str): Interval to collect. Must be passed as a complete string.
                            Valid values: 'minute', '3minute', '5minute', '10minute', '15minute', '30minute', '60minute', 'day'
                            Legacy values also supported: '1min', '5min', '15min', '30min', '60min', 'hour'
            from_date (str): From date in format 'yyyy-mm-dd HH:MM:SS'
            to_date (str): To date in format 'yyyy-mm-dd HH:MM:SS'
            continuous (int): Get continuous data (0 or 1). Pass 1 to get continuous data for futures
            oi (int): Get open interest data (0 or 1). Pass 1 to get OI data
                
        Returns:
            pandas.DataFrame: Historical data
            
        Warning:
            DO NOT iterate over the interval string! Pass the entire string (e.g., "day") as one parameter.
            Incorrect: for char in "day": get_historical_data(..., char, ...)
            Correct: get_historical_data(..., "day", ...)
        """
        import pandas as pd
        from datetime import datetime, timedelta
        import inspect
        import traceback
        
        # EMERGENCY DEBUG CODE
        # Print the complete stack trace to identify the source of the problem
        stack_trace = traceback.format_stack()
        
        self.logger.critical(f"DEBUG: get_historical_data called with interval_input='{interval_input}', type={type(interval_input)}")
        self.logger.critical(f"DEBUG: Call Stack:")
        for line in stack_trace:
            self.logger.critical(f"DEBUG: {line.strip()}")
            
        # Get caller info for debugging
        caller_info = []
        try:
            frame = inspect.currentframe().f_back
            while frame:
                info = inspect.getframeinfo(frame)
                caller_info.append(f"File: {info.filename}, Line: {info.lineno}, Function: {info.function}")
                if len(caller_info) >= 5:  # Limit to 5 frames to avoid excessive logging
                    break
                frame = frame.f_back
        except:
            caller_info.append("Could not get caller info")
        
        for info in caller_info:
            self.logger.critical(f"DEBUG: Caller: {info}")
        
        # Check if we're called from a loop that's iterating over a string
        try:
            frame = inspect.currentframe().f_back
            if frame:
                code_context = frame.f_code.co_name
                self.logger.critical(f"DEBUG: Called from function: {code_context}")
                
                # Try to get the source code of the caller
                try:
                    lines, starting_line_no = inspect.getsourcelines(frame)
                    for i, line in enumerate(lines[:10]):  # Only log the first 10 lines
                        self.logger.critical(f"DEBUG: Source line {starting_line_no + i}: {line.strip()}")
                except:
                    self.logger.critical("DEBUG: Could not get source lines")
        except Exception as e:
            self.logger.critical(f"DEBUG: Error getting frame info: {e}")
        
        try:
            # Type and value validation with detailed logging
            if not isinstance(interval_input, str):
                self.logger.error(f"Invalid interval type: {type(interval_input)}. Must be a string.")
                return None
                
            # Single character check - almost certainly an error
            if len(interval_input) == 1:
                self.logger.error(f"Single character interval '{interval_input}' is invalid.")
                self.logger.error(f"This error typically occurs when iterating over an interval string.")
                self.logger.error(f"Check if you're doing: for timeframe in 'day': get_historical_data(..., timeframe)")
                return None
                
            self.logger.info(f"Collecting historical data for {symbol}:{exchange}")
            
            # Get instrument token using existing method
            key = f"{symbol}:{exchange}"
            instrument_token = self.get_instrument_token(symbol, exchange)
            
            # If token not found, try to load instruments and get token again
            if not instrument_token:
                self.logger.warning(f"Token not found for {key}, attempting to load instruments")
                
                # Make sure we have a valid access token
                if not self.access_token:
                    self.logger.warning("No access token, attempting to authenticate")
                    login_url = self.generate_login_url()
                    self.logger.info(f"Please login using this URL: {login_url}")
                    return None
                
                # Try to load instruments
                self._load_instruments()
                
                # Try to get the token again
                instrument_token = self.get_instrument_token(symbol, exchange)
                
                if not instrument_token:
                    # Last resort - try to fetch instruments directly
                    self.logger.warning("Still couldn't find token, attempting direct instrument lookup")
                    try:
                        # Apply rate limiting
                        self._rate_limit()
                        
                        # Fetch all instruments
                        instruments = self.kite.instruments(exchange=exchange)
                        
                        # Look for our symbol
                        for instr in instruments:
                            if instr['tradingsymbol'] == symbol:
                                instrument_token = instr['instrument_token']
                                self.logger.info(f"Found token {instrument_token} for {symbol}:{exchange} via direct lookup")
                                
                                # Add it to our mappings
                                self.instrument_tokens[key] = instrument_token
                                self.token_to_instrument[instrument_token] = {
                                    'symbol': symbol,
                                    'exchange': exchange,
                                    'lot_size': instr.get('lot_size', 1),
                                    'tick_size': instr.get('tick_size', 0.05),
                                    'instrument_type': instr.get('instrument_type', 'EQ')
                                }
                                break
                        
                        # Log available symbols if not found
                        if not instrument_token:
                            symbols_in_exchange = [instr['tradingsymbol'] for instr in instruments[:10]]
                            self.logger.info(f"Sample symbols available in {exchange}: {symbols_in_exchange}")
                    except Exception as e:
                        self.logger.error(f"Error during direct instrument lookup: {e}")
            
            if not instrument_token:
                self.logger.error(f"Could not find instrument token for {exchange}:{symbol}")
                return None
                
            self.logger.info(f"Using instrument token: {instrument_token} for {symbol}:{exchange}")
            
            # Validate interval parameter - handle case where interval might be passed character by character
            # First, ensure interval_input is a proper string
            if not isinstance(interval_input, str):
                # Try to convert to string if possible
                try:
                    interval = str(interval_input)
                except:
                    self.logger.error(f"Invalid interval type: {type(interval_input)}. Must be a string.")
                    return None
            else:
                interval = interval_input
                    
            # Check if someone is trying to iterate over the interval string
            if len(interval) == 1:
                self.logger.warning(f"Received single-character interval: '{interval}'. This may indicate an iteration issue.")
                # Check if we're in a loop that's iterating over the interval string
                # We'll try to detect if the caller is in a loop by checking the stack
                import inspect
                caller_frame = inspect.currentframe().f_back
                if caller_frame:
                    caller_code = inspect.getframeinfo(caller_frame).code_context
                    self.logger.debug(f"Called from: {caller_code}")
                    if caller_code and any('for' in line and 'interval' in line for line in caller_code):
                        self.logger.error("Detected loop over interval string. Do not iterate over the interval parameter!")
                        return None
            
            # Validate the interval against known valid values
            valid_intervals = ['minute', '3minute', '5minute', '10minute', '15minute', '30minute', '60minute', 'day']
            legacy_intervals = ['1min', '3min', '5min', '15min', '30min', '60min', 'hour', '1hour']
            
            if interval not in valid_intervals and interval not in legacy_intervals:
                self.logger.error(f"Invalid interval: '{interval}'. Must be one of {valid_intervals} or {legacy_intervals}")
                return None
                
            # If using legacy interval format, convert to standard format
            legacy_map = {
                '1min': 'minute',
                '3min': '3minute',
                '5min': '5minute',
                '15min': '15minute',
                '30min': '30minute',
                '60min': '60minute',
                '1hour': '60minute',
                'hour': '60minute'
            }
            
            if interval in legacy_map:
                interval = legacy_map[interval]
                    
            self.logger.info(f"Using validated interval: '{interval}'")
            
            # Set default time range if not provided
            if not from_date or not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                from_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d %H:%M:%S')
            
            # For each interval, there are recommended date range limits to avoid API errors
            # Adjust the chunk size based on the interval
            chunk_size = 365  # Default chunk size in days
            
            # Define optimal chunk sizes based on interval to respect API limits
            if interval in ['minute', '3minute', '5minute']:
                chunk_size = 60  # 60 days for minute-level data
            elif interval in ['10minute', '15minute', '30minute', '60minute']:
                chunk_size = 100  # 100 days for higher frequency data
            else:  # 'day'
                chunk_size = 365  # 365 days for daily data
            
            # Convert date strings to datetime objects if they're strings
            if isinstance(from_date, str):
                from_date_obj = datetime.strptime(from_date, '%Y-%m-%d %H:%M:%S')
            else:
                from_date_obj = from_date
                
            if isinstance(to_date, str):
                to_date_obj = datetime.strptime(to_date, '%Y-%m-%d %H:%M:%S')
            else:
                to_date_obj = to_date
            
            # Calculate the date range in days
            date_range_days = (to_date_obj - from_date_obj).days + 1
            
            # Calculate the number of chunks needed
            num_chunks = (date_range_days + chunk_size - 1) // chunk_size  # Ceiling division
            
            # Prepare to collect data in chunks
            all_data = []
            current_to_date = to_date_obj
            
            for i in range(num_chunks):
                # Calculate current from_date
                current_from_date = max(
                    from_date_obj,  # Don't go earlier than requested
                    current_to_date - timedelta(days=chunk_size - 1)  # Chunk size
                )
                
                # Format dates as strings in the format Kite expects: YYYY-MM-DD HH:MM:SS
                current_from_date_str = current_from_date.strftime('%Y-%m-%d %H:%M:%S')
                current_to_date_str = current_to_date.strftime('%Y-%m-%d %H:%M:%S')
                
                self.logger.info(f"Collecting chunk {i+1}/{num_chunks} for {exchange}:{symbol} ({interval}) "
                            f"from {current_from_date_str} to {current_to_date_str}")
                
                # Apply rate limiting
                self._rate_limit()
                
                try:
                    # Call the Kite historical_data API with correct parameters
                    self.logger.debug(f"API call params: token={instrument_token}, from={current_from_date_str}, "
                                f"to={current_to_date_str}, interval={interval}, continuous={continuous}, oi={oi}")
                    
                    # Ensure parameters are passed correctly
                    chunk_data = self.kite.historical_data(
                        instrument_token=instrument_token,
                        from_date=current_from_date_str,
                        to_date=current_to_date_str,
                        interval=interval,
                        continuous=continuous,
                        oi=oi
                    )
                    
                    # Debug output for data received
                    if chunk_data:
                        self.logger.debug(f"Sample data received: {chunk_data[0]}")
                except Exception as e:
                    self.logger.error(f"Error collecting data in chunk {i+1}: {e}")
                    chunk_data = []
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    self.logger.info(f"Collected {len(chunk_data)} records in chunk {i+1}")
                else:
                    self.logger.warning(f"No data found in chunk {i+1} for {exchange}:{symbol}")
                
                # Update to_date for next chunk
                current_to_date = current_from_date - timedelta(days=1)
                
                # Respect API rate limits
                time.sleep(0.5)
                
                # Break if we've gone back far enough
                if current_to_date <= from_date_obj:
                    self.logger.info(f"Collected sufficient historical data (back to {from_date_obj.date()})")
                    break
            
            if not all_data:
                self.logger.warning(f"No historical data found for {exchange}:{symbol}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Add instrument details
            df['symbol'] = symbol
            df['exchange'] = exchange
            df['interval'] = interval
            
            self.logger.info(f"Collected {len(df)} total records for {exchange}:{symbol} ({interval})")
            
            # Debug DataFrame information
            self.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error collecting historical data for {symbol}: {e}")
            # Log the full exception for easier debugging
            import traceback
            self.logger.error(f"Exception traceback: {traceback.format_exc()}")
            return None
    
    def get_instrument_token(self, symbol, exchange):
        """
        Get instrument token for a symbol
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            int: Instrument token
        """
        key = f"{symbol}:{exchange}"
        
        # Log the lookup attempt
        self.logger.debug(f"Looking up token for {key}")
        
        token = self.instrument_tokens.get(key)
        
        if not token:
            # Try case-insensitive lookup
            for stored_key, stored_token in self.instrument_tokens.items():
                if key.upper() == stored_key.upper():
                    token = stored_token
                    self.logger.info(f"Found token {token} with case-insensitive match for {key}")
                    break
        
        if token:
            self.logger.debug(f"Found token {token} for {key}")
        else:
            self.logger.warning(f"Token not found for {key}")
            # Check if we have any instruments loaded
            if not self.instrument_tokens:
                self.logger.error("No instruments loaded. Make sure you're authenticated.")
        
        return token
    
    def connect_ticker(self, instrument_tokens=None):
        """
        Connect to Zerodha websocket for real-time data
        
        Args:
            instrument_tokens (list, optional): List of instrument tokens to subscribe
            
        Returns:
            bool: True if connected successfully
        """
        if not self.access_token:
            self.logger.error("Cannot connect ticker: No access token")
            return False
        
        try:
            # Initialize ticker
            self.ticker = KiteTicker(self.api_key, self.access_token)
            
            # Set callbacks
            self.ticker.on_ticks = self._on_ticks
            self.ticker.on_connect = self._on_connect
            self.ticker.on_close = self._on_close
            self.ticker.on_error = self._on_error
            self.ticker.on_reconnect = self._on_reconnect
            self.ticker.on_noreconnect = self._on_noreconnect
            
            # Connect
            self.ticker.connect(threaded=True)
            
            # Subscribe to tokens if provided
            if instrument_tokens:
                self.ticker.subscribe(instrument_tokens)
            
            self.logger.info("Ticker connection initiated")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting ticker: {e}")
            return False
    
    def subscribe_symbols(self, symbols, exchanges=None):
        """
        Subscribe to symbols for real-time data
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list, optional): List of exchanges corresponding to symbols
            
        Returns:
            bool: True if subscribed successfully
        """
        if not self.ticker_connected:
            connected = self.connect_ticker()
            if not connected:
                return False
        
        try:
            # Get instrument tokens
            tokens = []
            if exchanges:
                for symbol, exchange in zip(symbols, exchanges):
                    key = f"{symbol}:{exchange}"
                    token = self.instrument_tokens.get(key)
                    if token:
                        tokens.append(token)
                    else:
                        self.logger.warning(f"No token found for {key}")
            else:
                # Default to NSE
                for symbol in symbols:
                    key = f"{symbol}:NSE"
                    token = self.instrument_tokens.get(key)
                    if token:
                        tokens.append(token)
                    else:
                        self.logger.warning(f"No token found for {key}")
            
            if not tokens:
                self.logger.error("No valid instrument tokens found for subscription")
                return False
            
            # Subscribe
            self.ticker.subscribe(tokens)
            self.logger.info(f"Subscribed to {len(tokens)} instruments")
            return True
        except Exception as e:
            self.logger.error(f"Error subscribing: {e}")
            return False
    
    def add_tick_callback(self, callback):
        """
        Add callback for tick data
        
        Args:
            callback (function): Callback function
        """
        self.on_tick_callbacks.append(callback)
    
    def add_connection_callback(self, callback):
        """
        Add callback for connection events
        
        Args:
            callback (function): Callback function
        """
        self.on_connection_callbacks.append(callback)
    
    def place_order(self, **params):
        """
        Place an order on Zerodha
        
        Args:
            **params: Order parameters
            
        Returns:
            str: Order ID or None if failed
        """
        try:
            self._rate_limit()
            
            order_id = self.kite.place_order(
                variety=params.get("variety", "regular"),
                exchange=params.get("exchange", "NSE"),
                tradingsymbol=params.get("symbol"),
                transaction_type=params.get("transaction_type", "BUY"),
                quantity=params.get("quantity", 1),
                product=params.get("product", "MIS"),
                order_type=params.get("order_type", "MARKET"),
                price=params.get("price"),
                trigger_price=params.get("trigger_price"),
                tag=params.get("tag", "AutoTrader")
            )
            
            self.logger.info(f"Order placed successfully: {order_id}")
            return order_id
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def get_order_status(self, order_id):
        """
        Get status of an order
        
        Args:
            order_id (str): Order ID
            
        Returns:
            dict: Order details
        """
        try:
            self._rate_limit()
            orders = self.kite.orders()
            
            for order in orders:
                if order["order_id"] == order_id:
                    return order
            
            return None
        except Exception as e:
            self.logger.error(f"Error getting order status: {e}")
            return None
    
    def get_positions(self):
        """
        Get current positions
        
        Returns:
            dict: Positions data
        """
        try:
            self._rate_limit()
            positions = self.kite.positions()
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {"net": [], "day": []}
    
    def get_holdings(self):
        """
        Get holdings
        
        Returns:
            list: Holdings data
        """
        try:
            self._rate_limit()
            holdings = self.kite.holdings()
            return holdings
        except Exception as e:
            self.logger.error(f"Error getting holdings: {e}")
            return []
    
    def _on_ticks(self, ws, ticks):
        """Ticker callback for tick data"""
        for callback in self.on_tick_callbacks:
            try:
                callback(ticks)
            except Exception as e:
                self.logger.error(f"Error in tick callback: {e}")
    
    def _on_connect(self, ws, response):
        """Ticker callback for connection"""
        self.ticker_connected = True
        self.logger.info("Ticker connected")
        
        for callback in self.on_connection_callbacks:
            try:
                callback(True)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def _on_close(self, ws, code, reason):
        """Ticker callback for connection close"""
        self.ticker_connected = False
        self.logger.warning(f"Ticker connection closed: {code} - {reason}")
        
        for callback in self.on_connection_callbacks:
            try:
                callback(False)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def _on_error(self, ws, code, reason):
        """Ticker callback for errors"""
        self.logger.error(f"Ticker error: {code} - {reason}")
    
    def _on_reconnect(self, ws, attempt_count):
        """Ticker callback for reconnection"""
        self.logger.info(f"Ticker reconnecting: attempt {attempt_count}")
    
    def _on_noreconnect(self, ws):
        """Ticker callback for failed reconnection"""
        self.ticker_connected = False
        self.logger.error("Ticker failed to reconnect")
        
        # Notify callbacks
        for callback in self.on_connection_callbacks:
            try:
                callback(False)
            except Exception as e:
                self.logger.error(f"Error in connection callback: {e}")
    
    def _convert_interval(self, interval):
        """
        Convert interval to Zerodha format
        
        Args:
            interval (str): Internal interval format
            
        Returns:
            str: Zerodha interval format
        """
        mapping = {
            "1min": "minute",
            "5min": "5minute",
            "15min": "15minute",
            "30min": "30minute",
            "60min": "60minute",
            "day": "day",
            "week": "week"
        }
        
        return mapping.get(interval, interval)