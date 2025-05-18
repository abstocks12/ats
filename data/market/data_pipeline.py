"""
Market Data Pipeline

This module provides a pipeline for validating, cleaning, and transforming market data.
It ties together the validator and transformer into a coherent workflow.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from market_data_validator import MarketDataValidator
from data_transformer import MarketDataTransformer
from database.query_optimizer import QueryOptimizer 

class MarketDataPipeline:
    """
    Manages the full data processing pipeline for market data.
    Coordinates validation, cleaning, and transformation processes.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the market data pipeline with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Initialize validator and transformer
        self.validator = MarketDataValidator(db_connector)
        self.transformer = MarketDataTransformer(db_connector)
        self.query_optimizer = db_connector.get_query_optimizer()

    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this module."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def process_market_data(self, symbol: str, timeframe: str = "day", 
                          days: int = 30, auto_fix: bool = True) -> Dict[str, Any]:
        """
        Run the full data processing pipeline on market data.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe ('1min', '5min', '15min', 'hour', 'day', etc.)
            days: Number of days to process
            auto_fix: Whether to automatically fix issues
            
        Returns:
            Dictionary containing processing results
        """
        self.logger.info(f"Processing {timeframe} data for {symbol} over {days} days")
        
        # Step 1: Validate the data
        validation_results = self.validator.validate_market_data(symbol, timeframe, days)
        
        # If no data or critical errors and not auto-fixing, stop here
        if validation_results.get("status") == "error" and not auto_fix:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "error",
                "error": validation_results.get("error", "Validation failed"),
                "processed": False,
                "validation": validation_results
            }
        
        # Step 2: Transform/clean the data if needed and auto_fix is enabled
        transformation_results = None
        if auto_fix and validation_results.get("status") in ["error", "warning"]:
            transformation_results = self.transformer.transform_market_data(symbol, timeframe, days)
            
            # If transformation fails, return error
            if transformation_results.get("status") == "error":
                return {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": transformation_results.get("error", "Transformation failed"),
                    "processed": False,
                    "validation": validation_results,
                    "transformation": transformation_results
                }
        
        # Step 3: Calculate indicators
        indicator_results = self._calculate_indicators(symbol, timeframe, days)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "success",
            "processed": True,
            "validation": validation_results,
            "transformation": transformation_results,
            "indicators": indicator_results
        }
    
    def _calculate_indicators(self, symbol: str, timeframe: str, days: int) -> Dict[str, Any]:
        """
        Calculate technical indicators for the market data.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            days: Number of days
            
        Returns:
            Dictionary with indicator calculation results
        """
        try:
            # Get the market data
            start_date = datetime.now() - timedelta(days=days)
            data_result = self.query_optimizer.get_market_data(
            symbol=symbol,
            timeframe=timeframe,
            days=days,
            with_indicators=False
        )
        
            if data_result["status"] != "success":
                return {
                    "status": "error",
                    "error": data_result.get("error", "Failed to retrieve data")
                }
            
            # Convert to DataFrame and continue with existing logic
            import pandas as pd
            df = pd.DataFrame(data_result["data"])
            
            
            # Calculate indicators
            indicators_calculated = self._calculate_technical_indicators(df)
            
            # Save indicators back to database
            self._save_indicators(symbol, timeframe, df)
            
            return {
                "status": "success",
                "indicators_calculated": indicators_calculated
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_technical_indicators(self, df):
        """
        Calculate various technical indicators on the DataFrame.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Number of indicators calculated
        """
        # Ensure DataFrame is sorted by timestamp
        df = df.sort_values("timestamp")
        
        # Calculate Simple Moving Averages
        df['sma_5'] = df['close'].rolling(window=5).mean()
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Calculate Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
        # Calculate Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Calculate MACD
        # Calculate MACD
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate Bollinger Bands
        df['bollinger_middle'] = df['close'].rolling(window=20).mean()
        df['bollinger_std'] = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * 2)
        df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * 2)
        
        # Calculate Average True Range (ATR)
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['true_range'] = pd.DataFrame({'tr1': df['tr1'], 'tr2': df['tr2'], 'tr3': df['tr3']}).max(axis=1)
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        
        # Calculate Moving Average Convergence Divergence (MACD) Histogram
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate Rate of Change (ROC)
        df['roc_10'] = ((df['close'] / df['close'].shift(10)) - 1) * 100
        
        # Calculate Stochastic Oscillator
        df['lowest_14'] = df['low'].rolling(window=14).min()
        df['highest_14'] = df['high'].rolling(window=14).max()
        df['%K'] = ((df['close'] - df['lowest_14']) / (df['highest_14'] - df['lowest_14'])) * 100
        df['%D'] = df['%K'].rolling(window=3).mean()
        
        # Calculate On-Balance Volume (OBV)
        if 'volume' in df.columns:
            df['obv'] = 0
            df.loc[0, 'obv'] = df.loc[0, 'volume']
            for i in range(1, len(df)):
                if df.loc[i, 'close'] > df.loc[i-1, 'close']:
                    df.loc[i, 'obv'] = df.loc[i-1, 'obv'] + df.loc[i, 'volume']
                elif df.loc[i, 'close'] < df.loc[i-1, 'close']:
                    df.loc[i, 'obv'] = df.loc[i-1, 'obv'] - df.loc[i, 'volume']
                else:
                    df.loc[i, 'obv'] = df.loc[i-1, 'obv']
        
        # Calculate Average Directional Index (ADX)
        # +DM, -DM
        df['plus_dm'] = 0.0
        df['minus_dm'] = 0.0
        for i in range(1, len(df)):
            up_move = df.loc[i, 'high'] - df.loc[i-1, 'high']
            down_move = df.loc[i-1, 'low'] - df.loc[i, 'low']
            
            if up_move > down_move and up_move > 0:
                df.loc[i, 'plus_dm'] = up_move
            else:
                df.loc[i, 'plus_dm'] = 0.0
                
            if down_move > up_move and down_move > 0:
                df.loc[i, 'minus_dm'] = down_move
            else:
                df.loc[i, 'minus_dm'] = 0.0
        
        # Calculate smoothed averages
        df['tr14'] = df['true_range'].rolling(window=14).sum()
        df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=14).sum() / df['tr14'])
        df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=14).sum() / df['tr14'])
        
        # Calculate directional movement index
        df['dx'] = 100 * (abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14']))
        
        # Calculate ADX
        df['adx'] = df['dx'].rolling(window=14).mean()
        
        # Calculate Ichimoku Cloud
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        # Return DataFrame with indicators and count of indicators calculated
        indicator_count = len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']])
        return indicator_count

    def _save_indicators(self, symbol: str, timeframe: str, df) -> bool:
        """
        Save calculated indicators back to the database.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            df: DataFrame with calculated indicators
            
        Returns:
            True if successful, False otherwise
        """
        try:
            from pymongo import UpdateOne
            
            # Prepare bulk updates
            bulk_operations = []
            
            # List of indicator columns
            indicator_columns = [col for col in df.columns if col not in ['_id', 'symbol', 'timeframe', 'timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            for _, row in df.iterrows():
                # Skip rows with mostly NaN indicators
                nan_count = sum(pd.isna(row[col]) for col in indicator_columns)
                if nan_count > len(indicator_columns) * 0.5:  # If more than 50% are NaN
                    continue
                
                # Create indicator document
                indicators = {}
                for col in indicator_columns:
                    if not pd.isna(row[col]):
                        indicators[col] = float(row[col])
                
                # Only update if we have indicators
                if indicators:
                    # Add metadata
                    indicators["calculated_at"] = datetime.now()
                    
                    # Create update operation
                    bulk_operations.append(
                        UpdateOne(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "timestamp": row["timestamp"]
                            },
                            {
                                "$set": {"indicators": indicators}
                            }
                        )
                    )
            
            # Execute bulk update if there are operations
            if bulk_operations:
                result = self.db.market_data_collection.bulk_write(bulk_operations)
                self.logger.info(f"Updated indicators for {result.modified_count} documents")
                return True
            else:
                self.logger.info("No indicator updates needed")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving indicators: {e}")
            return False
    
    def process_all_active_instruments(self, timeframe="day", days=30, auto_fix=True):
        """
        Process data for all active instruments in the portfolio.
        
        Args:
            timeframe: Data timeframe
            days: Number of days to process
            auto_fix: Whether to automatically fix issues
            
        Returns:
            Dictionary with processing results for each instrument
        """
        try:
            # Get all active instruments
            instruments = list(self.db.portfolio_collection.find({
                "status": "active",
                "trading_config.enabled": True
            }))
            
            self.logger.info(f"Processing {timeframe} data for {len(instruments)} active instruments")
            
            results = {}
            
            # Process each instrument
            for instrument in instruments:
                symbol = instrument["symbol"]
                exchange = instrument["exchange"]
                
                # Process this instrument
                result = self.process_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    auto_fix=auto_fix
                )
                
                results[symbol] = result
                
                # Log the result
                if result.get("processed", False):
                    self.logger.info(f"Successfully processed {symbol} {timeframe} data")
                else:
                    self.logger.warning(f"Failed to process {symbol} {timeframe} data: {result.get('error')}")
            
            return {
                "status": "success",
                "instruments_processed": len(results),
                "results": results
            }
                
        except Exception as e:
            self.logger.error(f"Error processing instruments: {e}")
            return {
                "status": "error",
                "error": str(e)
            }