"""
Market Data Transformer

This module handles cleaning and transformation of market data.
It applies corrections to data issues identified by the validator.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union

class MarketDataTransformer:
    """
    Transforms and cleans market data based on validation results.
    Provides methods for filling gaps, removing outliers, and data normalization.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the market data transformer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
    
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
    
    def transform_market_data(self, symbol: str, timeframe: str = "day", 
                           days: int = 30) -> Dict[str, Any]:
        """
        Apply transformations to market data.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe ('1min', '5min', '15min', 'hour', 'day', etc.)
            days: Number of days to transform
            
        Returns:
            Dictionary containing transformation results
        """
        self.logger.info(f"Transforming {timeframe} data for {symbol} over {days} days")
        
        # Retrieve data from database
        data = self._get_market_data(symbol, timeframe, days)
        
        if not data:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "status": "error",
                "error": "No data found",
                "transformed": False
            }
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(data)
        original_count = len(df)
        
        # Store original DataFrame for comparison
        original_df = df.copy()
        
        # Apply transformations
        df = self._remove_duplicates(df)
        df = self._fill_missing_data(df, timeframe)
        df = self._handle_price_anomalies(df)
        df = self._handle_volume_anomalies(df)
        df = self._fix_ohlc_issues(df)
        
        # Calculate transformation statistics
        stats = {
            "original_rows": original_count,
            "final_rows": len(df),
            "duplicates_removed": original_count - len(df.drop_duplicates(subset=["timestamp"], keep="first")),
            "missing_filled": df["filled"].sum() if "filled" in df.columns else 0,
            "prices_adjusted": df["price_adjusted"].sum() if "price_adjusted" in df.columns else 0,
            "volumes_adjusted": df["volume_adjusted"].sum() if "volume_adjusted" in df.columns else 0,
            "ohlc_fixed": df["ohlc_fixed"].sum() if "ohlc_fixed" in df.columns else 0
        }
        
        # Save transformed data
        self._save_transformed_data(symbol, timeframe, df)
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "success",
            "transformed": True,
            "stats": stats
        }
    
    def _get_market_data(self, symbol: str, timeframe: str, days: int) -> List[Dict[str, Any]]:
        """
        Get market data from database.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            days: Number of days
            
        Returns:
            List of market data documents
        """
        try:
            # Calculate start date based on timeframe and days
            if timeframe == "day":
                # For daily data, use calendar days
                start_date = datetime.now() - timedelta(days=days)
            else:
                # For intraday data, use a buffer to account for non-trading days
                buffer_factor = 2 if days <= 30 else 1.5  # Larger buffer for shorter periods
                start_date = datetime.now() - timedelta(days=int(days * buffer_factor))
            
            # Query the database
            cursor = self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": {"$gte": start_date}
                }
            ).sort("timestamp", 1)
            
            data = list(cursor)
            
            if not data:
                self.logger.warning(f"No {timeframe} data found for {symbol} since {start_date}")
            else:
                self.logger.info(f"Retrieved {len(data)} {timeframe} data points for {symbol}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            return []
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate data points.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            DataFrame with duplicates removed
        """
        # Add tracking column for statistics
        df["duplicate"] = df.duplicated(subset=["timestamp"], keep="first")
        
        # Remove duplicates
        df_cleaned = df.drop_duplicates(subset=["timestamp"], keep="first")
        
        # Log the cleaning operation
        duplicate_count = len(df) - len(df_cleaned)
        if duplicate_count > 0:
            self.logger.info(f"Removed {duplicate_count} duplicate records")
        
        return df_cleaned
    
    def _fill_missing_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Fill missing data points using interpolation.
        
        Args:
            df: DataFrame containing market data
            timeframe: Data timeframe
            
        Returns:
            DataFrame with missing data filled
        """
        # Sort dataframe by timestamp
        df = df.sort_values("timestamp")
        
        # Add a column to track filled values
        df["filled"] = False
        
        # Determine expected frequency based on timeframe
        if timeframe == "day":
            # Business days only (skip weekends)
            freq = "B"
        elif timeframe == "hour":
            # Hourly during trading hours
            freq = "H"
        elif timeframe == "1min":
            freq = "1min"
        elif timeframe == "5min":
            freq = "5min"
        elif timeframe == "15min":
            freq = "15min"
        else:
            # Default to daily
            freq = "B"
        
        # Create a date range at the expected frequency
        if timeframe == "day":
            # For daily data, use business days
            expected_dates = pd.date_range(
                start=df["timestamp"].min(),
                end=df["timestamp"].max(),
                freq=freq
            )
            
            # Filter expected dates to trading days only
            expected_dates = expected_dates[expected_dates.dayofweek < 5]  # 0-4 are Monday-Friday
        else:
            # For intraday data, use trading hours (9:15 AM to 3:30 PM for Indian markets)
            expected_dates = []
            current_date = df["timestamp"].min().date()
            end_date = df["timestamp"].max().date()
            
            while current_date <= end_date:
                # Skip weekends
                if current_date.weekday() < 5:  # 0-4 are Monday-Friday
                    if timeframe == "hour":
                        # Trading hours: 9:15 AM to 3:30 PM (round to hours)
                        for hour in range(9, 16):  # 9 AM to 3 PM
                            expected_dates.append(datetime.combine(current_date, datetime.min.time()) + timedelta(hours=hour))
                    elif timeframe in ["1min", "5min", "15min"]:
                        # Trading minutes: 9:15 AM to 3:30 PM
                        start_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=9, minutes=15)
                        end_time = datetime.combine(current_date, datetime.min.time()) + timedelta(hours=15, minutes=30)
                        
                        # Generate expected times at the appropriate frequency
                        if timeframe == "1min":
                            delta = timedelta(minutes=1)
                        elif timeframe == "5min":
                            delta = timedelta(minutes=5)
                        else:  # 15min
                            delta = timedelta(minutes=15)
                        
                        current_time = start_time
                        while current_time <= end_time:
                            expected_dates.append(current_time)
                            current_time += delta
                
                current_date += timedelta(days=1)
            
            expected_dates = pd.DatetimeIndex(expected_dates)
        
        # Create a new DataFrame with the expected timestamps
        df_reindexed = df.set_index("timestamp").reindex(expected_dates)
        
        # Interpolate missing values
        for col in ["open", "high", "low", "close"]:
            if col in df_reindexed.columns:
                df_reindexed[col] = df_reindexed[col].interpolate(method="linear")
        
        # For volume, forward fill (or use previous day's volume as estimate)
        if "volume" in df_reindexed.columns:
            df_reindexed["volume"] = df_reindexed["volume"].fillna(method="ffill")
        
        # Mark filled rows
        df_reindexed["filled"] = df_reindexed["filled"].fillna(True)
        
        # Reset index to get timestamp as a column again
        df_reindexed = df_reindexed.reset_index().rename(columns={"index": "timestamp"})
        
        # Log the filling operation
        fill_count = df_reindexed["filled"].sum()
        if fill_count > 0:
            self.logger.info(f"Filled {fill_count} missing data points")
        
        return df_reindexed
    
    def _handle_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle price anomalies by smoothing extreme jumps or gaps.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            DataFrame with price anomalies handled
        """
        # Add a column to track price adjustments
        df["price_adjusted"] = False
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Calculate price changes
        df['price_change_pct'] = (df['close'].pct_change() * 100).fillna(0)
        
        # Calculate gap between previous close and current open
        df['gap_pct'] = ((df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100).fillna(0)
        
        # Threshold for extreme price movements (%)
        threshold = 15.0
        
        # Find extreme price jumps
        extreme_indices = df[abs(df['price_change_pct']) > threshold].index
        
        # Handle each extreme price point
        for idx in extreme_indices:
            # Get previous and next rows for context
            prev_idx = df.index[df.index.get_loc(idx) - 1] if df.index.get_loc(idx) > 0 else None
            next_idx = df.index[df.index.get_loc(idx) + 1] if df.index.get_loc(idx) < len(df) - 1 else None
            
            if prev_idx is not None and next_idx is not None:
                # Adjust using the average of previous and next points
                df.loc[idx, 'close'] = (df.loc[prev_idx, 'close'] + df.loc[next_idx, 'close']) / 2
                df.loc[idx, 'high'] = max(df.loc[prev_idx, 'high'], df.loc[next_idx, 'high'])
                df.loc[idx, 'low'] = min(df.loc[prev_idx, 'low'], df.loc[next_idx, 'low'])
                df.loc[idx, 'price_adjusted'] = True
        
        # Handle extreme gaps
        gap_indices = df[abs(df['gap_pct']) > threshold].index
        
        for idx in gap_indices:
            # Get previous close
            prev_idx = df.index[df.index.get_loc(idx) - 1] if df.index.get_loc(idx) > 0 else None
            
            if prev_idx is not None:
                # Adjust the open price to reduce the gap
                prev_close = df.loc[prev_idx, 'close']
                curr_open = df.loc[idx, 'open']
                
                # Limit the gap to the threshold percentage
                max_allowed_gap = prev_close * (1 + (threshold / 100) * (1 if curr_open > prev_close else -1))
                
                if abs((curr_open - prev_close) / prev_close * 100) > threshold:
                    df.loc[idx, 'open'] = max_allowed_gap
                    df.loc[idx, 'price_adjusted'] = True
        
        # Recalculate OHLC to ensure consistency
        df = self._ensure_ohlc_consistency(df)
        
        # Log the adjustment operation
        adjustment_count = df["price_adjusted"].sum()
        if adjustment_count > 0:
            self.logger.info(f"Adjusted {adjustment_count} price anomalies")
        
        return df
    
    def _handle_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle volume anomalies by adjusting extreme spikes and filling zero values.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            DataFrame with volume anomalies handled
        """
        # Add a column to track volume adjustments
        df["volume_adjusted"] = False
        
        if 'volume' not in df.columns:
            return df
        
        # Calculate average volume
        avg_volume = df['volume'].median()  # Median is more robust to outliers than mean
        
        # Handle zero volumes
        zero_indices = df[df['volume'] == 0].index
        
        for idx in zero_indices:
            # Use the average volume as a replacement
            df.loc[idx, 'volume'] = avg_volume
            df.loc[idx, 'volume_adjusted'] = True
        
        # Handle volume spikes (more than 5x the average)
        spike_threshold = 5.0
        spike_indices = df[df['volume'] > avg_volume * spike_threshold].index
        
        for idx in spike_indices:
            # Cap the volume at the threshold
            df.loc[idx, 'volume'] = avg_volume * spike_threshold
            df.loc[idx, 'volume_adjusted'] = True
        
        # Log the adjustment operation
        adjustment_count = df["volume_adjusted"].sum()
        if adjustment_count > 0:
            self.logger.info(f"Adjusted {adjustment_count} volume anomalies")
        
        return df
    
    def _fix_ohlc_issues(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix OHLC data consistency issues.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            DataFrame with OHLC issues fixed
        """
        # Add a column to track OHLC fixes
        df["ohlc_fixed"] = False
        
        # Fix high < low issues
        high_low_invalid = df[df['high'] < df['low']].index
        
        for idx in high_low_invalid:
            # Swap high and low
            temp = df.loc[idx, 'high']
            df.loc[idx, 'high'] = df.loc[idx, 'low']
            df.loc[idx, 'low'] = temp
            df.loc[idx, 'ohlc_fixed'] = True
        
        # Fix high < open or high < close issues
        high_invalid = df[(df['high'] < df['open']) | (df['high'] < df['close'])].index
        
        for idx in high_invalid:
            # Set high to the maximum of open and close
            df.loc[idx, 'high'] = max(df.loc[idx, 'open'], df.loc[idx, 'close'])
            df.loc[idx, 'ohlc_fixed'] = True
        
        # Fix low > open or low > close issues
        low_invalid = df[(df['low'] > df['open']) | (df['low'] > df['close'])].index
        
        for idx in low_invalid:
            # Set low to the minimum of open and close
            df.loc[idx, 'low'] = min(df.loc[idx, 'open'], df.loc[idx, 'close'])
            df.loc[idx, 'ohlc_fixed'] = True
        
        # Ensure high-low range is reasonable
        min_diff_pct = 0.01  # Minimum 0.01% difference
        min_diff_check = df[((df['high'] - df['low']) / df['low'] * 100) < min_diff_pct].index
        
        for idx in min_diff_check:
            # Widen the range to meet the minimum difference
            mid_price = (df.loc[idx, 'high'] + df.loc[idx, 'low']) / 2
            half_range = mid_price * (min_diff_pct / 100) / 2
            
            df.loc[idx, 'high'] = mid_price + half_range
            df.loc[idx, 'low'] = mid_price - half_range
            df.loc[idx, 'ohlc_fixed'] = True
        
        # Log the fixing operation
        fix_count = df["ohlc_fixed"].sum()
        if fix_count > 0:
            self.logger.info(f"Fixed {fix_count} OHLC consistency issues")
        
        return df
    
    def _ensure_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure OHLC data is consistent after adjustments.
        
        Args:
            df: DataFrame containing market data
            
        Returns:
            DataFrame with consistent OHLC data
        """
        # For each row, ensure high >= max(open, close) and low <= min(open, close)
        for idx in df.index:
            # Adjust high if needed
            if df.loc[idx, 'high'] < max(df.loc[idx, 'open'], df.loc[idx, 'close']):
                df.loc[idx, 'high'] = max(df.loc[idx, 'open'], df.loc[idx, 'close'])
                df.loc[idx, 'ohlc_fixed'] = True
            
            # Adjust low if needed
            if df.loc[idx, 'low'] > min(df.loc[idx, 'open'], df.loc[idx, 'close']):
                df.loc[idx, 'low'] = min(df.loc[idx, 'open'], df.loc[idx, 'close'])
                df.loc[idx, 'ohlc_fixed'] = True
        
        return df
    
    def _save_transformed_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> bool:
        """
        Save transformed data back to the database.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            df: DataFrame with transformed data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare bulk updates
            bulk_operations = []
            
            for _, row in df.iterrows():
                # Only update if row was modified
                if row.get("filled", False) or row.get("price_adjusted", False) or \
                   row.get("volume_adjusted", False) or row.get("ohlc_fixed", False):
                    
                    # Create update operation
                    update_data = {
                        "open": row["open"],
                        "high": row["high"],
                        "low": row["low"],
                        "close": row["close"]
                    }
                    
                    if "volume" in row:
                        update_data["volume"] = row["volume"]
                    
                    # Add metadata about transformations
                    update_data["transformed"] = True
                    update_data["transformation_date"] = datetime.now()
                    update_data["transformation_type"] = []
                    
                    if row.get("filled", False):
                        update_data["transformation_type"].append("filled")
                    if row.get("price_adjusted", False):
                        update_data["transformation_type"].append("price_adjusted")
                    if row.get("volume_adjusted", False):
                        update_data["transformation_type"].append("volume_adjusted")
                    if row.get("ohlc_fixed", False):
                        update_data["transformation_type"].append("ohlc_fixed")
                    
                    # Create update operation
                    bulk_operations.append(
                        UpdateOne(
                            {
                                "symbol": symbol,
                                "timeframe": timeframe,
                                "timestamp": row["timestamp"]
                            },
                            {
                                "$set": update_data
                            }
                        )
                    )
            
            # Execute bulk update if there are operations
            if bulk_operations:
                result = self.db.market_data_collection.bulk_write(bulk_operations)
                self.logger.info(f"Updated {result.modified_count} documents in the database")
                return True
            else:
                self.logger.info("No updates needed, all data was already consistent")
                return True
                
        except Exception as e:
            self.logger.error(f"Error saving transformed data: {e}")
            return False