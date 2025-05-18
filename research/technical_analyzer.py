"""
Technical Analysis System

This module provides comprehensive technical analysis capabilities for the automated trading system.
It implements advanced indicators, chart patterns, candlestick patterns, and market regime detection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
import statistics

class TechnicalAnalyzer:
    """
    Provides technical analysis capabilities for trading decisions.
    Implements advanced indicators, pattern recognition, and trading signals.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the technical analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get query optimizer if available
        self.query_optimizer = getattr(self.db, 'get_query_optimizer', lambda: None)()
        
        # Define pattern detection thresholds
        self.pattern_thresholds = {
            # Chart patterns
            "double_top_min_peak_distance": 10,      # Minimum bars between tops
            "double_top_max_price_diff": 0.03,       # Maximum price difference between tops (%)
            "double_bottom_min_trough_distance": 10, # Minimum bars between bottoms
            "double_bottom_max_price_diff": 0.03,    # Maximum price difference between bottoms (%)
            "head_shoulders_min_peak_distance": 5,   # Minimum bars between peaks
            "head_shoulders_min_height_ratio": 1.2,  # Head must be at least 1.2x shoulder height
            
            # Candlestick patterns
            "doji_body_max_ratio": 0.1,              # Maximum ratio of body to full range
            "hammer_body_max_ratio": 0.3,            # Maximum ratio of body to full range
            "hammer_tail_min_ratio": 0.6,            # Minimum ratio of lower wick to full range
            "engulfing_min_size_ratio": 1.5,         # Minimum size ratio for engulfing pattern
            
            # Support/Resistance
            "support_resistance_lookback": 30,       # Bars to look back for S/R levels
            "support_resistance_touch_count": 2,     # Minimum touches to confirm S/R level
            "support_resistance_significance": 0.02,  # Price distance for merging nearby levels (%)
            
            # Volatility
            "low_volatility_atr_percentile": 25,     # Percentile to define low volatility
            "high_volatility_atr_percentile": 75,    # Percentile to define high volatility
        }
        
        # Define indicator parameters
        self.indicator_params = {
            # Moving Averages
            "ma_short": 20,                          # Short-term MA period
            "ma_medium": 50,                         # Medium-term MA period
            "ma_long": 200,                          # Long-term MA period
            
            # Momentum
            "rsi_period": 14,                        # RSI period
            "rsi_overbought": 70,                    # RSI overbought level
            "rsi_oversold": 30,                      # RSI oversold level
            "macd_fast": 12,                         # MACD fast EMA period
            "macd_slow": 26,                         # MACD slow EMA period
            "macd_signal": 9,                        # MACD signal period
            "stoch_k_period": 14,                    # Stochastic %K period
            "stoch_d_period": 3,                     # Stochastic %D period
            
            # Volatility
            "atr_period": 14,                        # ATR period
            "bollinger_period": 20,                  # Bollinger Bands period
            "bollinger_std": 2,                      # Bollinger Bands standard deviation
            
            # Volume
            "volume_ma_period": 20,                  # Volume moving average period
            "obv_ma_period": 20,                     # OBV moving average period
        }
    
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

    def analyze(self, symbol: str, exchange: str = "NSE", timeframe: str = "day", days: int = 100) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe ('1min', '5min', '15min', 'hour', 'day', etc.)
            days: Number of days to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            self.logger.info(f"Analyzing {symbol} ({exchange}) on {timeframe} timeframe")
            
            # Get market data - use query optimizer if available or fallback to direct query
            if self.query_optimizer:
                data_result = self.query_optimizer.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    with_indicators=False
                )
                
                if data_result["status"] != "success":
                    return {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "status": "error",
                        "error": data_result.get("error", "Failed to retrieve data")
                    }
                
                data = data_result["data"]
            else:
                # Fallback to direct query
                start_date = datetime.now() - timedelta(days=days)
                cursor = self.db.market_data_collection.find({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": {"$gte": start_date}
                }).sort("timestamp", 1)
                
                data = list(cursor)
            
            if not data:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "No data found"
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Make sure required columns are present
            if not all(col in df.columns for col in ["timestamp", "open", "high", "low", "close", "volume"]):
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Incomplete data: missing required columns"
                }
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate technical indicators
            df = self._calculate_indicators(df)
            
            # Detect patterns
            chart_patterns = self._detect_chart_patterns(df)
            candlestick_patterns = self._detect_candlestick_patterns(df)
            
            # Identify support and resistance levels
            support_resistance = self._find_support_resistance(df)
            
            # Determine market regime
            market_regime = self._determine_market_regime(df)
            
            # Generate trading signals
            signals = self._generate_signals(df, market_regime)
            
            # Format timestamp as string for proper JSON serialization
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Compute market trend
            trend = self._compute_trend(df)
            
            # Compute volatility regime
            volatility = self._compute_volatility(df)
            
            # Only include the last 5 rows for the indicators (to reduce payload size)
            recent_indicators = df.iloc[-5:].to_dict('records')
            
            # Get recent signals (last 3)
            recent_signals = signals[-3:] if len(signals) > 0 else []
            
            # Get recent patterns (last 3)
            recent_chart_patterns = chart_patterns[-3:] if len(chart_patterns) > 0 else []
            recent_candlestick_patterns = candlestick_patterns[-3:] if len(candlestick_patterns) > 0 else []
            
            # Assemble the analysis result
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "status": "success",
                "data_points": len(df),
                "date_range": f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}",
                "trend": trend,
                "volatility": volatility,
                "market_regime": market_regime,
                "support_resistance": support_resistance,
                "recent_indicators": recent_indicators,
                "recent_signals": recent_signals,
                "recent_chart_patterns": recent_chart_patterns,
                "recent_candlestick_patterns": recent_candlestick_patterns,
                "summary": self._generate_analysis_summary(
                    df, trend, volatility, market_regime, signals, 
                    chart_patterns, candlestick_patterns, support_resistance
                )
            }
            
            # Save analysis result to database
            self._save_analysis(symbol, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the market data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with indicators added
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Ensure DataFrame is sorted by timestamp
        df = df.sort_values("timestamp")
        
        # Moving Averages
        df['sma_short'] = df['close'].rolling(window=self.indicator_params["ma_short"]).mean()
        df['sma_medium'] = df['close'].rolling(window=self.indicator_params["ma_medium"]).mean()
        df['sma_long'] = df['close'].rolling(window=self.indicator_params["ma_long"]).mean()
        
        df['ema_short'] = df['close'].ewm(span=self.indicator_params["ma_short"], adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=self.indicator_params["ma_medium"], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.indicator_params["ma_long"], adjust=False).mean()
        
        # Relative Strength Index (RSI)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.indicator_params["rsi_period"]).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.indicator_params["rsi_period"]).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_fast = df['close'].ewm(span=self.indicator_params["macd_fast"], adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.indicator_params["macd_slow"], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.indicator_params["macd_signal"], adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        df['bollinger_middle'] = df['close'].rolling(window=self.indicator_params["bollinger_period"]).mean()
        df['bollinger_std'] = df['close'].rolling(window=self.indicator_params["bollinger_period"]).std()
        df['bollinger_upper'] = df['bollinger_middle'] + (df['bollinger_std'] * self.indicator_params["bollinger_std"])
        df['bollinger_lower'] = df['bollinger_middle'] - (df['bollinger_std'] * self.indicator_params["bollinger_std"])
        
        # Bollinger Band Width
        df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['bollinger_middle']
        
        # Average True Range (ATR)
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        df['true_range'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = df['true_range'].rolling(window=self.indicator_params["atr_period"]).mean()
        
        # Normalize ATR as percentage of price
        df['atr_percent'] = df['atr'] / df['close'] * 100
        
        # Stochastic Oscillator
        df['stoch_lowest_low'] = df['low'].rolling(window=self.indicator_params["stoch_k_period"]).min()
        df['stoch_highest_high'] = df['high'].rolling(window=self.indicator_params["stoch_k_period"]).max()
        
        df['stoch_k'] = 100 * ((df['close'] - df['stoch_lowest_low']) / 
                            (df['stoch_highest_high'] - df['stoch_lowest_low']))
        df['stoch_d'] = df['stoch_k'].rolling(window=self.indicator_params["stoch_d_period"]).mean()
        
        # Volume Analysis
        df['volume_ma'] = df['volume'].rolling(window=self.indicator_params["volume_ma_period"]).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # On-Balance Volume (OBV)
        df['obv'] = 0
        df.loc[1:, 'obv'] = (
            (df.loc[1:, 'close'].values > df.loc[1:, 'close'].shift(1).values) * df.loc[1:, 'volume'].values -
            (df.loc[1:, 'close'].values < df.loc[1:, 'close'].shift(1).values) * df.loc[1:, 'volume'].values
        ).cumsum()
        
        # OBV Moving Average
        df['obv_ma'] = df['obv'].rolling(window=self.indicator_params["obv_ma_period"]).mean()
        
        # Price Rate of Change (ROC)
        df['price_roc'] = df['close'].pct_change(periods=self.indicator_params["ma_short"]) * 100
        
        # Average Directional Index (ADX)
        # Calculate +DM and -DM
        df['plus_dm'] = 0.0
        df['minus_dm'] = 0.0
        for i in range(1, len(df)):
            high_diff = df['high'].iloc[i] - df['high'].iloc[i-1]
            low_diff = df['low'].iloc[i-1] - df['low'].iloc[i]
            
            if high_diff > low_diff and high_diff > 0:
                df['plus_dm'].iloc[i] = high_diff
            else:
                df['plus_dm'].iloc[i] = 0.0
            
            if low_diff > high_diff and low_diff > 0:
                df['minus_dm'].iloc[i] = low_diff
            else:
                df['minus_dm'].iloc[i] = 0.0
        
        # Calculate true range and directional indicators
        df['tr14'] = df['true_range'].rolling(window=14).sum()
        df['plus_di14'] = 100 * (df['plus_dm'].rolling(window=14).sum() / df['tr14'])
        df['minus_di14'] = 100 * (df['minus_dm'].rolling(window=14).sum() / df['tr14'])
        
        # Calculate directional index and ADX
        df['dx'] = 100 * abs(df['plus_di14'] - df['minus_di14']) / (df['plus_di14'] + df['minus_di14'])
        df['adx'] = df['dx'].rolling(window=14).mean()
        
        # Calculate parabolic SAR
        df['sar'] = self._calculate_psar(df)
        
        # Compute Ichimoku Cloud components
        df['tenkan_sen'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        df['kijun_sen'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
        df['senkou_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        df['chikou_span'] = df['close'].shift(-26)
        
        # Filter out rows with too many NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _calculate_psar(self, df: pd.DataFrame, iaf: float = 0.02, maxaf: float = 0.2) -> pd.Series:
        """
        Calculate Parabolic SAR (Stop and Reverse).
        
        Args:
            df: DataFrame with market data
            iaf: Initial acceleration factor
            maxaf: Maximum acceleration factor
            
        Returns:
            Series with PSAR values
        """
        length = len(df)
        if length < 2:
            return pd.Series(np.nan, index=df.index)
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        # Initialize psar, dir (direction), af (acceleration factor), and ep (extreme point)
        psar = np.zeros(length)
        dir = np.ones(length)  # 1 for uptrend, -1 for downtrend
        af = np.ones(length) * iaf
        ep = np.zeros(length)
        
        # Initialize first values
        psar[0] = low[0]
        dir[0] = 1  # Start with uptrend
        ep[0] = high[0]
        
        # Calculate PSAR values
        for i in range(1, length):
            # Previous PSAR value
            psar[i] = psar[i-1] + af[i-1] * (ep[i-1] - psar[i-1])
            
            # Adjust PSAR value for current bar
            if dir[i-1] == 1:  # Uptrend
                # PSAR can't be higher than the low of the previous two bars
                psar[i] = min(psar[i], low[i-1], low[i-2] if i > 1 else low[i-1])
                
                # Check if reversal
                if low[i] < psar[i]:
                    dir[i] = -1  # Change to downtrend
                    psar[i] = ep[i-1]  # PSAR becomes previous extreme point
                    ep[i] = low[i]  # New extreme point is current low
                    af[i] = iaf  # Reset acceleration factor
                else:
                    dir[i] = 1  # Continue uptrend
                    if high[i] > ep[i-1]:  # New high
                        ep[i] = high[i]  # Update extreme point
                        af[i] = min(af[i-1] + iaf, maxaf)  # Increase acceleration factor
                    else:
                        ep[i] = ep[i-1]  # Keep previous extreme point
                        af[i] = af[i-1]  # Keep previous acceleration factor
            else:  # Downtrend
                # PSAR can't be lower than the high of the previous two bars
                psar[i] = max(psar[i], high[i-1], high[i-2] if i > 1 else high[i-1])
                
                # Check if reversal
                if high[i] > psar[i]:
                    dir[i] = 1  # Change to uptrend
                    psar[i] = ep[i-1]  # PSAR becomes previous extreme point
                    ep[i] = high[i]  # New extreme point is current high
                    af[i] = iaf  # Reset acceleration factor
                else:
                    dir[i] = -1  # Continue downtrend
                    if low[i] < ep[i-1]:  # New low
                        ep[i] = low[i]  # Update extreme point
                        af[i] = min(af[i-1] + iaf, maxaf)  # Increase acceleration factor
                    else:
                        ep[i] = ep[i-1]  # Keep previous extreme point
                        af[i] = af[i-1]  # Keep previous acceleration factor
        
        return pd.Series(psar, index=df.index)
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect chart patterns in price data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of detected patterns with details
        """
        patterns = []
        
        # Create a smaller DataFrame with just the columns we need
        data = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        
        # Double Top pattern
        double_tops = self._detect_double_top(data)
        patterns.extend(double_tops)
        
        # Double Bottom pattern
        double_bottoms = self._detect_double_bottom(data)
        patterns.extend(double_bottoms)
        
        # Head and Shoulders pattern
        head_shoulders = self._detect_head_shoulders(data)
        patterns.extend(head_shoulders)
        
        # Triangles
        triangles = self._detect_triangles(data)
        patterns.extend(triangles)
        
        # Sort patterns by timestamp
        patterns.sort(key=lambda x: x['timestamp'])
        
        return patterns
    
    def _detect_double_top(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect Double Top pattern.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected Double Top patterns
        """
        patterns = []
        min_peak_distance = self.pattern_thresholds["double_top_min_peak_distance"]
        max_price_diff = self.pattern_thresholds["double_top_max_price_diff"]
        
        # Find peaks
        # A peak is a point where the price is higher than the previous and next points
        peaks = []
        for i in range(1, len(data) - 1):
            if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                peaks.append((i, data['high'].iloc[i]))
        
        # Look for pairs of peaks (double tops)
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                idx1, price1 = peaks[i]
                idx2, price2 = peaks[j]
                
                # Check if peaks are far enough apart
                if idx2 - idx1 < min_peak_distance:
                    continue
                
                # Check if prices are similar
                price_diff = abs(price1 - price2) / price1
                if price_diff > max_price_diff:
                    continue
                
                # Check if there's a significant trough between the peaks
                trough_idx = data['low'].iloc[idx1:idx2].idxmin()
                trough_price = data['low'].iloc[trough_idx]
                
                # Calculate confirmations
                # A double top is confirmed when the price breaks below the trough
                confirmed = False
                confirmation_idx = None
                
                for k in range(idx2 + 1, len(data)):
                    if data['close'].iloc[k] < trough_price:
                        confirmed = True
                        confirmation_idx = k
                        break
                
                if confirmed:
                    patterns.append({
                        "type": "double_top",
                        "timestamp": data['timestamp'].iloc[confirmation_idx],
                        "direction": "bearish",
                        "first_peak": {
                            "timestamp": data['timestamp'].iloc[idx1],
                            "price": price1
                        },
                        "second_peak": {
                            "timestamp": data['timestamp'].iloc[idx2],
                            "price": price2
                        },
                        "trough": {
                            "timestamp": data['timestamp'].iloc[trough_idx],
                            "price": trough_price
                        },
                        "confirmation": {
                            "timestamp": data['timestamp'].iloc[confirmation_idx],
                            "price": data['close'].iloc[confirmation_idx]
                        },
                        "target_price": trough_price - (price1 - trough_price),  # Measured move
                        "stop_loss": max(price1, price2) * 1.02  # 2% above the highest peak
                    })
        
        return patterns
    
    def _detect_double_bottom(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect Double Bottom pattern.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected Double Bottom patterns
        """
        patterns = []
        min_trough_distance = self.pattern_thresholds["double_bottom_min_trough_distance"]
        max_price_diff = self.pattern_thresholds["double_bottom_max_price_diff"]
        
        # Find troughs
        # A trough is a point where the price is lower than the previous and next points
        troughs = []
        for i in range(1, len(data) - 1):
            if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                troughs.append((i, data['low'].iloc[i]))
        
        # Look for pairs of troughs (double bottoms)
        for i in range(len(troughs) - 1):
            for j in range(i + 1, len(troughs)):
                idx1, price1 = troughs[i]
                idx2, price2 = troughs[j]
                
                # Check if troughs are far enough apart
                if idx2 - idx1 < min_trough_distance:
                    continue
                
                # Check if prices are similar
                price_diff = abs(price1 - price2) / price1
                if price_diff > max_price_diff:
                    continue
                
                # Check if there's a significant peak between the troughs
                peak_idx = data['high'].iloc[idx1:idx2].idxmax()
                peak_price = data['high'].iloc[peak_idx]
                
                # Calculate confirmations
                # A double bottom is confirmed when the price breaks above the peak
                confirmed = False
                confirmation_idx = None
                
                for k in range(idx2 + 1, len(data)):
                    if data['close'].iloc[k] > peak_price:
                        confirmed = True
                        confirmation_idx = k
                        break
                
                if confirmed:
                    patterns.append({
                        "type": "double_bottom",
                        "timestamp": data['timestamp'].iloc[confirmation_idx],
                        "direction": "bullish",
                        "first_trough": {
                            "timestamp": data['timestamp'].iloc[idx1],
                            "price": price1
                        },
                        "second_trough": {
                            "timestamp": data['timestamp'].iloc[idx2],
                            "price": price2
                        },
                        "peak": {
                            "timestamp": data['timestamp'].iloc[peak_idx],
                            "price": peak_price
                        },
                        "confirmation": {
                            "timestamp": data['timestamp'].iloc[confirmation_idx],
                            "price": data['close'].iloc[confirmation_idx]
                        },
                        "target_price": peak_price + (peak_price - price1),  # Measured move
                        "stop_loss": min(price1, price2) * 0.98  # 2% below the lowest trough
                    })
        
        return patterns
    
    def _detect_head_shoulders(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect Head and Shoulders pattern.
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected Head and Shoulders patterns
        """
        patterns = []
        min_peak_distance = self.pattern_thresholds["head_shoulders_min_peak_distance"]
        min_height_ratio = self.pattern_thresholds["head_shoulders_min_height_ratio"]
        
        # Find peaks
        peaks = []
        for i in range(1, len(data) - 1):
            if data['high'].iloc[i] > data['high'].iloc[i-1] and data['high'].iloc[i] > data['high'].iloc[i+1]:
                peaks.append((i, data['high'].iloc[i]))
        
        # Need at least 3 peaks for head and shoulders
        if len(peaks) < 3:
            return patterns
        
        # Look for triplets of peaks (head and shoulders)
        for i in range(len(peaks) - 2):
            idx1, price1 = peaks[i]  # Left shoulder
            idx2, price2 = peaks[i+1]  # Head
            idx3, price3 = peaks[i+2]  # Right shoulder
            
            # Check if peaks are far enough apart
            if idx2 - idx1 < min_peak_distance or idx3 - idx2 < min_peak_distance:
                continue
            
            # Check if head is higher than shoulders
            if price2 <= price1 or price2 <= price3:
                continue
            
            # Check if shoulders are at similar levels
            shoulder_diff = abs(price1 - price3) / price1
            if shoulder_diff > 0.1:  # Allow 10% difference between shoulders
                continue
            
            # Check if head is significantly higher than shoulders
            if price2 < min_height_ratio * (price1 + price3) / 2:
                continue
            
            # Find troughs between peaks
            # Find troughs between peaks
            trough1_idx = data['low'].iloc[idx1:idx2].idxmin()
            trough2_idx = data['low'].iloc[idx2:idx3].idxmin()
            
            # Check if troughs are at similar levels (neckline)
            trough1_price = data['low'].iloc[trough1_idx]
            trough2_price = data['low'].iloc[trough2_idx]
            
            neckline_diff = abs(trough1_price - trough2_price) / trough1_price
            if neckline_diff > 0.05:  # Allow 5% difference in neckline
                continue
            
            # Calculate neckline
            neckline_price = (trough1_price + trough2_price) / 2
            
            # Calculate confirmations
            # A head and shoulders is confirmed when price breaks below the neckline
            confirmed = False
            confirmation_idx = None
            
            for k in range(idx3 + 1, len(data)):
                if data['close'].iloc[k] < neckline_price:
                    confirmed = True
                    confirmation_idx = k
                    break
            
            if confirmed:
                # Calculate target price (measured move)
                # Distance from head to neckline, projected downward
                target_price = neckline_price - (price2 - neckline_price)
                
                patterns.append({
                    "type": "head_and_shoulders",
                    "timestamp": data['timestamp'].iloc[confirmation_idx],
                    "direction": "bearish",
                    "left_shoulder": {
                        "timestamp": data['timestamp'].iloc[idx1],
                        "price": price1
                    },
                    "head": {
                        "timestamp": data['timestamp'].iloc[idx2],
                        "price": price2
                    },
                    "right_shoulder": {
                        "timestamp": data['timestamp'].iloc[idx3],
                        "price": price3
                    },
                    "left_trough": {
                        "timestamp": data['timestamp'].iloc[trough1_idx],
                        "price": trough1_price
                    },
                    "right_trough": {
                        "timestamp": data['timestamp'].iloc[trough2_idx],
                        "price": trough2_price
                    },
                    "neckline": neckline_price,
                    "confirmation": {
                        "timestamp": data['timestamp'].iloc[confirmation_idx],
                        "price": data['close'].iloc[confirmation_idx]
                    },
                    "target_price": target_price,
                    "stop_loss": price2 * 1.02  # 2% above the head
                })
        
        # Also check for inverse head and shoulders (bullish)
        troughs = []
        for i in range(1, len(data) - 1):
            if data['low'].iloc[i] < data['low'].iloc[i-1] and data['low'].iloc[i] < data['low'].iloc[i+1]:
                troughs.append((i, data['low'].iloc[i]))
        
        # Need at least 3 troughs for inverse head and shoulders
        if len(troughs) < 3:
            return patterns
        
        # Look for triplets of troughs (inverse head and shoulders)
        for i in range(len(troughs) - 2):
            idx1, price1 = troughs[i]  # Left shoulder
            idx2, price2 = troughs[i+1]  # Head
            idx3, price3 = troughs[i+2]  # Right shoulder
            
            # Check if troughs are far enough apart
            if idx2 - idx1 < min_peak_distance or idx3 - idx2 < min_peak_distance:
                continue
            
            # Check if head is lower than shoulders
            if price2 >= price1 or price2 >= price3:
                continue
            
            # Check if shoulders are at similar levels
            shoulder_diff = abs(price1 - price3) / price1
            if shoulder_diff > 0.1:  # Allow 10% difference between shoulders
                continue
            
            # Check if head is significantly lower than shoulders
            avg_shoulder = (price1 + price3) / 2
            if price2 > avg_shoulder / min_height_ratio:
                continue
            
            # Find peaks between troughs
            peak1_idx = data['high'].iloc[idx1:idx2].idxmax()
            peak2_idx = data['high'].iloc[idx2:idx3].idxmax()
            
            # Check if peaks are at similar levels (neckline)
            peak1_price = data['high'].iloc[peak1_idx]
            peak2_price = data['high'].iloc[peak2_idx]
            
            neckline_diff = abs(peak1_price - peak2_price) / peak1_price
            if neckline_diff > 0.05:  # Allow 5% difference in neckline
                continue
            
            # Calculate neckline
            neckline_price = (peak1_price + peak2_price) / 2
            
            # Calculate confirmations
            # An inverse head and shoulders is confirmed when price breaks above the neckline
            confirmed = False
            confirmation_idx = None
            
            for k in range(idx3 + 1, len(data)):
                if data['close'].iloc[k] > neckline_price:
                    confirmed = True
                    confirmation_idx = k
                    break
            
            if confirmed:
                # Calculate target price (measured move)
                # Distance from head to neckline, projected upward
                target_price = neckline_price + (neckline_price - price2)
                
                patterns.append({
                    "type": "inverse_head_and_shoulders",
                    "timestamp": data['timestamp'].iloc[confirmation_idx],
                    "direction": "bullish",
                    "left_shoulder": {
                        "timestamp": data['timestamp'].iloc[idx1],
                        "price": price1
                    },
                    "head": {
                        "timestamp": data['timestamp'].iloc[idx2],
                        "price": price2
                    },
                    "right_shoulder": {
                        "timestamp": data['timestamp'].iloc[idx3],
                        "price": price3
                    },
                    "left_peak": {
                        "timestamp": data['timestamp'].iloc[peak1_idx],
                        "price": peak1_price
                    },
                    "right_peak": {
                        "timestamp": data['timestamp'].iloc[peak2_idx],
                        "price": peak2_price
                    },
                    "neckline": neckline_price,
                    "confirmation": {
                        "timestamp": data['timestamp'].iloc[confirmation_idx],
                        "price": data['close'].iloc[confirmation_idx]
                    },
                    "target_price": target_price,
                    "stop_loss": price2 * 0.98  # 2% below the head
                })
        
        return patterns
    
    def _detect_triangles(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect triangle patterns (ascending, descending, symmetrical).
        
        Args:
            data: DataFrame with OHLC data
            
        Returns:
            List of detected triangle patterns
        """
        patterns = []
        
        # Need at least 20 bars for a triangle
        if len(data) < 20:
            return patterns
        
        # For multiple windows
        for window in [20, 30, 40]:
            if len(data) <= window:
                continue
                
            # Look at the last 'window' bars
            window_data = data.iloc[-window:].copy()
            
            # Find highs and lows
            highs = []
            lows = []
            
            for i in range(1, len(window_data) - 1):
                # Find swing highs
                if window_data['high'].iloc[i] > window_data['high'].iloc[i-1] and \
                   window_data['high'].iloc[i] > window_data['high'].iloc[i+1]:
                    highs.append((i, window_data['high'].iloc[i]))
                
                # Find swing lows
                if window_data['low'].iloc[i] < window_data['low'].iloc[i-1] and \
                   window_data['low'].iloc[i] < window_data['low'].iloc[i+1]:
                    lows.append((i, window_data['low'].iloc[i]))
            
            # Need at least 2 highs and 2 lows for a triangle
            if len(highs) < 2 or len(lows) < 2:
                continue
            
            # Fit linear regression to highs and lows
            high_indices = [h[0] for h in highs]
            high_prices = [h[1] for h in highs]
            
            low_indices = [l[0] for l in lows]
            low_prices = [l[1] for l in lows]
            
            # Calculate slope for highs
            high_slope, high_intercept = np.polyfit(high_indices, high_prices, 1)
            
            # Calculate slope for lows
            low_slope, low_intercept = np.polyfit(low_indices, low_prices, 1)
            
            # Check if we have a triangle pattern
            if abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
                # Both lines are flat, not a triangle
                continue
            
            # Calculate R-squared to assess fit quality
            high_mean = np.mean(high_prices)
            low_mean = np.mean(low_prices)
            
            high_ss_tot = sum([(y - high_mean) ** 2 for y in high_prices])
            high_ss_res = sum([(y - (high_slope * x + high_intercept)) ** 2 for x, y in zip(high_indices, high_prices)])
            
            low_ss_tot = sum([(y - low_mean) ** 2 for y in low_prices])
            low_ss_res = sum([(y - (low_slope * x + low_intercept)) ** 2 for x, y in zip(low_indices, low_prices)])
            
            if high_ss_tot > 0:
                high_r2 = 1 - (high_ss_res / high_ss_tot)
            else:
                high_r2 = 0
                
            if low_ss_tot > 0:
                low_r2 = 1 - (low_ss_res / low_ss_tot)
            else:
                low_r2 = 0
            
            # Check R-squared values
            if high_r2 < 0.6 or low_r2 < 0.6:
                # Poor fit, not a clear triangle
                continue
            
            # Determine triangle type
            triangle_type = None
            if high_slope < -0.01 and low_slope > 0.01:
                triangle_type = "symmetrical"
            elif high_slope < -0.01 and abs(low_slope) < 0.01:
                triangle_type = "descending"
            elif abs(high_slope) < 0.01 and low_slope > 0.01:
                triangle_type = "ascending"
            
            if triangle_type:
                # Check if price is near the apex (convergence)
                last_idx = len(window_data) - 1
                upper_line = high_slope * last_idx + high_intercept
                lower_line = low_slope * last_idx + low_intercept
                
                # Apex percentage - how close price is to converging point
                price_range = upper_line - lower_line
                full_range = high_prices[0] - low_prices[0] if high_prices and low_prices else 1
                
                if full_range == 0:
                    continue
                    
                apex_percentage = (1 - price_range / full_range) * 100
                
                # Only include triangles that are at least 60% formed
                if apex_percentage < 60:
                    continue
                
                # Get last price
                last_price = window_data['close'].iloc[-1]
                
                # Calculate direction based on triangle type and price position
                direction = "neutral"
                if triangle_type == "ascending":
                    direction = "bullish"
                elif triangle_type == "descending":
                    direction = "bearish"
                elif triangle_type == "symmetrical":
                    if last_price > (upper_line + lower_line) / 2:
                        direction = "bullish"
                    else:
                        direction = "bearish"
                
                # Calculate target price
                height = full_range
                if direction == "bullish":
                    target_price = upper_line + height
                else:
                    target_price = lower_line - height
                
                # Calculate stop loss
                if direction == "bullish":
                    stop_loss = lower_line * 0.98  # 2% below the lower line
                else:
                    stop_loss = upper_line * 1.02  # 2% above the upper line
                
                # Add triangle pattern
                patterns.append({
                    "type": f"{triangle_type}_triangle",
                    "timestamp": data['timestamp'].iloc[-1],
                    "direction": direction,
                    "upper_line": {
                        "start": high_prices[0],
                        "end": upper_line,
                        "slope": high_slope
                    },
                    "lower_line": {
                        "start": low_prices[0],
                        "end": lower_line,
                        "slope": low_slope
                    },
                    "apex_percentage": apex_percentage,
                    "target_price": target_price,
                    "stop_loss": stop_loss
                })
        
        return patterns
    
    def _detect_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect candlestick patterns in price data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            List of detected candlestick patterns with details
        """
        patterns = []
        
        # Create a smaller DataFrame with just the columns we need
        data = df[['timestamp', 'open', 'high', 'low', 'close']].copy()
        
        # Calculate some values needed for pattern detection
        data['body_high'] = data[['open', 'close']].max(axis=1)
        data['body_low'] = data[['open', 'close']].min(axis=1)
        data['body_size'] = data['body_high'] - data['body_low']
        data['range'] = data['high'] - data['low']
        data['is_bullish'] = data['close'] > data['open']
        
        # Calculate average body size for relative comparisons
        avg_body_size = data['body_size'].mean()
        avg_range = data['range'].mean()
        
        # Doji
        doji_max_ratio = self.pattern_thresholds["doji_body_max_ratio"]
        for i in range(len(data)):
            # A doji has a very small body compared to its range
            body_to_range_ratio = data['body_size'].iloc[i] / data['range'].iloc[i] if data['range'].iloc[i] > 0 else 0
            
            if body_to_range_ratio <= doji_max_ratio and data['range'].iloc[i] >= avg_range * 0.5:
                patterns.append({
                    "type": "doji",
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": "neutral",
                    "open": data['open'].iloc[i],
                    "high": data['high'].iloc[i],
                    "low": data['low'].iloc[i],
                    "close": data['close'].iloc[i],
                    "significance": "high" if i > 0 and (
                        (data['is_bullish'].iloc[i-1] and data['close'].iloc[i] > data['high'].iloc[i-1]) or
                        (not data['is_bullish'].iloc[i-1] and data['close'].iloc[i] < data['low'].iloc[i-1])
                    ) else "medium"
                })
        
        # Hammer and Hanging Man
        hammer_body_max_ratio = self.pattern_thresholds["hammer_body_max_ratio"]
        hammer_tail_min_ratio = self.pattern_thresholds["hammer_tail_min_ratio"]
        
        for i in range(1, len(data)):
            # A hammer has a small body at the top with a long lower wick
            body_to_range_ratio = data['body_size'].iloc[i] / data['range'].iloc[i] if data['range'].iloc[i] > 0 else 0
            lower_wick = data['body_low'].iloc[i] - data['low'].iloc[i]
            lower_wick_ratio = lower_wick / data['range'].iloc[i] if data['range'].iloc[i] > 0 else 0
            
            if body_to_range_ratio <= hammer_body_max_ratio and lower_wick_ratio >= hammer_tail_min_ratio:
                # Determine if hammer or hanging man based on prior trend
                prev_trend = "bearish" if data.iloc[max(0, i-5):i]['close'].mean() > data['close'].iloc[i] else "bullish"
                
                pattern_type = "hammer" if prev_trend == "bearish" else "hanging_man"
                direction = "bullish" if pattern_type == "hammer" else "bearish"
                
                patterns.append({
                    "type": pattern_type,
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": direction,
                    "open": data['open'].iloc[i],
                    "high": data['high'].iloc[i],
                    "low": data['low'].iloc[i],
                    "close": data['close'].iloc[i],
                    "significance": "high" if (
                        (direction == "bullish" and data['low'].iloc[i] < data.iloc[max(0, i-5):i]['low'].min()) or
                        (direction == "bearish" and data['high'].iloc[i] > data.iloc[max(0, i-5):i]['high'].max())
                    ) else "medium"
                })
        
        # Shooting Star and Inverted Hammer
        for i in range(1, len(data)):
            # A shooting star/inverted hammer has a small body at the bottom with a long upper wick
            body_to_range_ratio = data['body_size'].iloc[i] / data['range'].iloc[i] if data['range'].iloc[i] > 0 else 0
            upper_wick = data['high'].iloc[i] - data['body_high'].iloc[i]
            upper_wick_ratio = upper_wick / data['range'].iloc[i] if data['range'].iloc[i] > 0 else 0
            
            if body_to_range_ratio <= hammer_body_max_ratio and upper_wick_ratio >= hammer_tail_min_ratio:
                # Determine if shooting star or inverted hammer based on prior trend
                prev_trend = "bullish" if data.iloc[max(0, i-5):i]['close'].mean() < data['close'].iloc[i] else "bearish"
                
                pattern_type = "shooting_star" if prev_trend == "bullish" else "inverted_hammer"
                direction = "bearish" if pattern_type == "shooting_star" else "bullish"
                
                patterns.append({
                    "type": pattern_type,
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": direction,
                    "open": data['open'].iloc[i],
                    "high": data['high'].iloc[i],
                    "low": data['low'].iloc[i],
                    "close": data['close'].iloc[i],
                    "significance": "high" if (
                        (direction == "bearish" and data['high'].iloc[i] > data.iloc[max(0, i-5):i]['high'].max()) or
                        (direction == "bullish" and data['low'].iloc[i] < data.iloc[max(0, i-5):i]['low'].min())
                    ) else "medium"
                })
        
        # Engulfing patterns
        engulfing_min_size_ratio = self.pattern_thresholds["engulfing_min_size_ratio"]
        
        for i in range(1, len(data)):
            # Skip if current or previous candle is a doji
            curr_body_ratio = data['body_size'].iloc[i] / data['range'].iloc[i] if data['range'].iloc[i] > 0 else 0
            prev_body_ratio = data['body_size'].iloc[i-1] / data['range'].iloc[i-1] if data['range'].iloc[i-1] > 0 else 0
            
            if curr_body_ratio <= doji_max_ratio or prev_body_ratio <= doji_max_ratio:
                continue
            
            # Bullish engulfing
            if (data['is_bullish'].iloc[i] and not data['is_bullish'].iloc[i-1] and
                data['open'].iloc[i] <= data['close'].iloc[i-1] and
                data['close'].iloc[i] >= data['open'].iloc[i-1] and
                data['body_size'].iloc[i] >= data['body_size'].iloc[i-1] * engulfing_min_size_ratio):
                
                patterns.append({
                    "type": "bullish_engulfing",
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": "bullish",
                    "open": data['open'].iloc[i],
                    "high": data['high'].iloc[i],
                    "low": data['low'].iloc[i],
                    "close": data['close'].iloc[i],
                    "significance": "high" if data['low'].iloc[i] < data.iloc[max(0, i-5):i-1]['low'].min() else "medium"
                })
            
            # Bearish engulfing
            elif (not data['is_bullish'].iloc[i] and data['is_bullish'].iloc[i-1] and
                  data['open'].iloc[i] >= data['close'].iloc[i-1] and
                  data['close'].iloc[i] <= data['open'].iloc[i-1] and
                  data['body_size'].iloc[i] >= data['body_size'].iloc[i-1] * engulfing_min_size_ratio):
                
                patterns.append({
                    "type": "bearish_engulfing",
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": "bearish",
                    "open": data['open'].iloc[i],
                    "high": data['high'].iloc[i],
                    "low": data['low'].iloc[i],
                    "close": data['close'].iloc[i],
                    "significance": "high" if data['high'].iloc[i] > data.iloc[max(0, i-5):i-1]['high'].max() else "medium"
                })
        
        # Morning Star and Evening Star
        for i in range(2, len(data)):
            # Morning Star
            if (not data['is_bullish'].iloc[i-2] and                   # First candle is bearish
                data['body_size'].iloc[i-1] <= avg_body_size * 0.5 and # Middle candle has small body
                data['is_bullish'].iloc[i] and                         # Third candle is bullish
                data['close'].iloc[i] > (data['open'].iloc[i-2] + data['close'].iloc[i-2]) / 2 and  # Closed above midpoint of first candle
                data['body_size'].iloc[i] >= avg_body_size):           # Third candle has significant body
                
                patterns.append({
                    "type": "morning_star",
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": "bullish",
                    "candle1": {
                        "timestamp": data['timestamp'].iloc[i-2],
                        "open": data['open'].iloc[i-2],
                        "close": data['close'].iloc[i-2]
                    },
                    "candle2": {
                        "timestamp": data['timestamp'].iloc[i-1],
                        "open": data['open'].iloc[i-1],
                        "close": data['close'].iloc[i-1]
                    },
                    "candle3": {
                        "timestamp": data['timestamp'].iloc[i],
                        "open": data['open'].iloc[i],
                        "close": data['close'].iloc[i]
                    },
                    "significance": "high"
                })
            
            # Evening Star
            elif (data['is_bullish'].iloc[i-2] and                     # First candle is bullish
                  data['body_size'].iloc[i-1] <= avg_body_size * 0.5 and  # Middle candle has small body
                  not data['is_bullish'].iloc[i] and                   # Third candle is bearish
                  data['close'].iloc[i] < (data['open'].iloc[i-2] + data['close'].iloc[i-2]) / 2 and  # Closed below midpoint of first candle
                  data['body_size'].iloc[i] >= avg_body_size):         # Third candle has significant body
                
                patterns.append({
                    "type": "evening_star",
                    "timestamp": data['timestamp'].iloc[i],
                    "direction": "bearish",
                    "candle1": {
                        "timestamp": data['timestamp'].iloc[i-2],
                        "open": data['open'].iloc[i-2],
                        "close": data['close'].iloc[i-2]
                    },
                    "candle2": {
                        "timestamp": data['timestamp'].iloc[i-1],
                        "open": data['open'].iloc[i-1],
                        "close": data['close'].iloc[i-1]
                    },
                    "candle3": {
                        "timestamp": data['timestamp'].iloc[i],
                        "open": data['open'].iloc[i],
                        "close": data['close'].iloc[i]
                    },
                    "significance": "high"
                })
        
        # Sort patterns by timestamp
        patterns.sort(key=lambda x: x['timestamp'])
        
        return patterns
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify key support and resistance levels.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with support and resistance levels
        """
        lookback = self.pattern_thresholds["support_resistance_lookback"]
        touch_count = self.pattern_thresholds["support_resistance_touch_count"]
        significance = self.pattern_thresholds["support_resistance_significance"]
        
        # Use only recent data for identifying levels
        if len(df) > lookback:
            recent_df = df.iloc[-lookback:].copy()
        else:
            recent_df = df.copy()
        
        # Find local minima and maxima
        local_mins = []
        local_maxes = []
        
        for i in range(1, len(recent_df) - 1):
            # Local minimum (potential support)
            if recent_df['low'].iloc[i] < recent_df['low'].iloc[i-1] and recent_df['low'].iloc[i] < recent_df['low'].iloc[i+1]:
                local_mins.append(recent_df['low'].iloc[i])
            
            # Local maximum (potential resistance)
            if recent_df['high'].iloc[i] > recent_df['high'].iloc[i-1] and recent_df['high'].iloc[i] > recent_df['high'].iloc[i+1]:
                local_maxes.append(recent_df['high'].iloc[i])
        
        # Cluster similar price levels
        def cluster_levels(levels, significance_pct):
            if not levels:
                return []
            
            # Sort levels
            sorted_levels = sorted(levels)
            
            # Initialize clusters
            clusters = [[sorted_levels[0]]]
            
            # Group levels into clusters
            for level in sorted_levels[1:]:
                # Calculate percentage difference from first level in last cluster
                diff_pct = abs(level - clusters[-1][0]) / clusters[-1][0]
                
                if diff_pct <= significance_pct:
                    # Add to existing cluster
                    clusters[-1].append(level)
                else:
                    # Start new cluster
                    clusters.append([level])
            
            # Calculate average level for each cluster
            return [sum(cluster) / len(cluster) for cluster in clusters]
        
        # Get clustered support and resistance levels
        support_levels = cluster_levels(local_mins, significance)
        resistance_levels = cluster_levels(local_maxes, significance)
        
        # Calculate strength based on number of touches
        def count_touches(level, price_series, significance_pct):
            # Count how many times price has touched or come close to the level
            touches = 0
            for price in price_series:
                if abs(price - level) / level <= significance_pct:
                    touches += 1
            return touches
        
        # Calculate support and resistance strength
        support_strength = []
        for level in support_levels:
            touches = count_touches(level, recent_df['low'].values, significance)
            if touches >= touch_count:
                # Only include levels with sufficient touches
                support_strength.append({
                    "level": level,
                    "touches": touches,
                    "significance": "high" if touches >= touch_count * 2 else "medium"
                })
        
        resistance_strength = []
        for level in resistance_levels:
            touches = count_touches(level, recent_df['high'].values, significance)
            if touches >= touch_count:
                # Only include levels with sufficient touches
                resistance_strength.append({
                    "level": level,
                    "touches": touches,
                    "significance": "high" if touches >= touch_count * 2 else "medium"
                })
        
        # Sort by price level
        support_strength.sort(key=lambda x: x["level"])
        resistance_strength.sort(key=lambda x: x["level"])
        
        # Add most recent price for context
        current_price = df['close'].iloc[-1]
        
        # Find nearest support and resistance
        nearest_support = None
        nearest_support_distance = float('inf')
        
        for support in support_strength:
            if support["level"] < current_price:
                distance = current_price - support["level"]
                if distance < nearest_support_distance:
                    nearest_support_distance = distance
                    nearest_support = support["level"]
        
        nearest_resistance = None
        nearest_resistance_distance = float('inf')
        
        for resistance in resistance_strength:
            if resistance["level"] > current_price:
                distance = resistance["level"] - current_price
                if distance < nearest_resistance_distance:
                    nearest_resistance_distance = distance
                    nearest_resistance = resistance["level"]
        
        return {
            "support_levels": support_strength,
            "resistance_levels": resistance_strength,
            "current_price": current_price,
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance
        }
    
    def _determine_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Determine the current market regime.
        
        Args:
            df: DataFrame with market data and indicators
            
        Returns:
            Dictionary with market regime characteristics
        """
        # Use recent data for regime detection (last 30 bars)
        recent_df = df.iloc[-min(30, len(df)):].copy()
        
        # Check trend direction
        trend_direction = "neutral"
        
        # Use SMA relationship to determine trend
        if "sma_short" in recent_df.columns and "sma_medium" in recent_df.columns and "sma_long" in recent_df.columns:
            last_short = recent_df['sma_short'].iloc[-1]
            last_medium = recent_df['sma_medium'].iloc[-1]
            last_long = recent_df['sma_long'].iloc[-1]
            
            if last_short > last_medium > last_long:
                trend_direction = "bullish"
            elif last_short < last_medium < last_long:
                trend_direction = "bearish"
        
        # Determine trend strength using ADX
        trend_strength = "weak"
        if "adx" in recent_df.columns:
            last_adx = recent_df['adx'].iloc[-1]
            
            if last_adx < 20:
                trend_strength = "weak"
            elif 20 <= last_adx < 40:
                trend_strength = "moderate"
            else:
                trend_strength = "strong"
        
        # Determine volatility using ATR percentage and Bollinger Band width
        volatility = "moderate"
        if "atr_percent" in recent_df.columns:
            atr_percent = recent_df['atr_percent'].iloc[-1]
            atr_percentile = percentile_rank(recent_df['atr_percent'].dropna(), atr_percent)
            
            if atr_percentile < 25:
                volatility = "low"
            elif atr_percentile > 75:
                volatility = "high"
            else:
                volatility = "moderate"
        
        # Determine if market is ranging or trending
        market_type = "undefined"
        if "bollinger_width" in recent_df.columns and "adx" in recent_df.columns:
            bb_width = recent_df['bollinger_width'].iloc[-1]
            bb_width_percentile = percentile_rank(recent_df['bollinger_width'].dropna(), bb_width)
            
            last_adx = recent_df['adx'].iloc[-1]
            
            # Ranging market typically has narrow BB and low ADX
            if bb_width_percentile < 30 and last_adx < 20:
                market_type = "ranging"
            # Trending market typically has wide BB and high ADX
            elif bb_width_percentile > 70 and last_adx > 25:
                market_type = "trending"
            # Transitioning market has mixed characteristics
            else:
                market_type = "transitioning"
        
        # Determine momentum
        momentum = "neutral"
        if "rsi" in recent_df.columns and "macd" in recent_df.columns:
            last_rsi = recent_df['rsi'].iloc[-1]
            last_macd = recent_df['macd'].iloc[-1]
            last_macd_signal = recent_df['macd_signal'].iloc[-1]
            
            rsi_bullish = last_rsi > 50 and last_rsi < 70
            rsi_bearish = last_rsi < 50 and last_rsi > 30
            macd_bullish = last_macd > last_macd_signal and last_macd > 0
            macd_bearish = last_macd < last_macd_signal and last_macd < 0
            
            if rsi_bullish and macd_bullish:
                momentum = "bullish"
            elif rsi_bearish and macd_bearish:
                momentum = "bearish"
        
        # Determine if market is overbought or oversold
        market_condition = "neutral"
        if "rsi" in recent_df.columns and "stoch_k" in recent_df.columns:
            last_rsi = recent_df['rsi'].iloc[-1]
            last_stoch_k = recent_df['stoch_k'].iloc[-1]
            last_stoch_d = recent_df['stoch_d'].iloc[-1]
            
            if last_rsi > 70 and last_stoch_k > 80 and last_stoch_d > 80:
                market_condition = "overbought"
            elif last_rsi < 30 and last_stoch_k < 20 and last_stoch_d < 20:
                market_condition = "oversold"
        
        # Determine volume trend
        volume_trend = "neutral"
        if "volume_ratio" in recent_df.columns:
            recent_volume_ratios = recent_df['volume_ratio'].iloc[-5:].values
            avg_volume_ratio = np.mean(recent_volume_ratios)
            
            if avg_volume_ratio > 1.5:
                volume_trend = "increasing"
            elif avg_volume_ratio < 0.75:
                volume_trend = "decreasing"
        
        # Combine all factors to determine overall regime
        overall_regime = "neutral"
        if market_type == "trending":
            if trend_direction == "bullish" and trend_strength in ["moderate", "strong"]:
                overall_regime = "bullish_trend"
            elif trend_direction == "bearish" and trend_strength in ["moderate", "strong"]:
                overall_regime = "bearish_trend"
        elif market_type == "ranging":
            if market_condition == "oversold":
                overall_regime = "oversold_range"
            elif market_condition == "overbought":
                overall_regime = "overbought_range"
            else:
                overall_regime = "sideways"
        elif volatility == "high":
            overall_regime = "volatile"
        
        return {
            "overall_regime": overall_regime,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "volatility": volatility,
            "market_type": market_type,
            "momentum": momentum,
            "market_condition": market_condition,
            "volume_trend": volume_trend
        }
    
    def _generate_signals(self, df: pd.DataFrame, market_regime: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate trading signals based on technical analysis.
        
        Args:
            df: DataFrame with market data and indicators
            market_regime: Dictionary with market regime characteristics
            
        Returns:
            List of trading signals
        """
        signals = []
        
        # Use recent data for signal generation (last 10 bars)
        recent_df = df.iloc[-min(10, len(df)):].copy()
        
        # Moving Average Crossovers
        if "sma_short" in recent_df.columns and "sma_medium" in recent_df.columns:
            # Check if short MA crossed above medium MA
            if recent_df['sma_short'].iloc[-2] <= recent_df['sma_medium'].iloc[-2] and \
               recent_df['sma_short'].iloc[-1] > recent_df['sma_medium'].iloc[-1]:
                
                # Generate bullish signal
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "ma_crossover",
                    "direction": "bullish",
                    "description": "Short-term MA crossed above medium-term MA",
                    "strength": "medium" if market_regime["trend_direction"] == "bullish" else "weak",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.03,  # 3% target
                    "stop_loss": recent_df['close'].iloc[-1] * 0.98  # 2% stop loss
                })
            
            # Check if short MA crossed below medium MA
            elif recent_df['sma_short'].iloc[-2] >= recent_df['sma_medium'].iloc[-2] and \
                 recent_df['sma_short'].iloc[-1] < recent_df['sma_medium'].iloc[-1]:
                
                # Generate bearish signal
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "ma_crossover",
                    "direction": "bearish",
                    "description": "Short-term MA crossed below medium-term MA",
                    "strength": "medium" if market_regime["trend_direction"] == "bearish" else "weak",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.97,  # 3% target
                    "stop_loss": recent_df['close'].iloc[-1] * 1.02  # 2% stop loss
                })
        
        # MACD Crossovers
        if "macd" in recent_df.columns and "macd_signal" in recent_df.columns:
            # Check if MACD crossed above signal line
            if recent_df['macd'].iloc[-2] <= recent_df['macd_signal'].iloc[-2] and \
               recent_df['macd'].iloc[-1] > recent_df['macd_signal'].iloc[-1]:
                
                # Generate bullish signal
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "macd_crossover",
                    "direction": "bullish",
                    "description": "MACD crossed above signal line",
                    "strength": "high" if recent_df['macd'].iloc[-1] < 0 else "medium",  # Stronger if crossing at negative values
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.04,  # 4% target
                    "stop_loss": recent_df['close'].iloc[-1] * 0.98  # 2% stop loss
                })
            
            # Check if MACD crossed below signal line
            elif recent_df['macd'].iloc[-2] >= recent_df['macd_signal'].iloc[-2] and \
                 recent_df['macd'].iloc[-1] < recent_df['macd_signal'].iloc[-1]:
                
                # Generate bearish signal
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "macd_crossover",
                    "direction": "bearish",
                    "description": "MACD crossed below signal line",
                    "strength": "high" if recent_df['macd'].iloc[-1] > 0 else "medium",  # Stronger if crossing at positive values
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.96,  # 4% target
                    "stop_loss": recent_df['close'].iloc[-1] * 1.02  # 2% stop loss
                })
        
        # RSI Oversold/Overbought
        if "rsi" in recent_df.columns:
            rsi_overbought = self.indicator_params["rsi_overbought"]
            rsi_oversold = self.indicator_params["rsi_oversold"]
            
            # Check for RSI moving out of oversold territory
            if recent_df['rsi'].iloc[-2] <= rsi_oversold and recent_df['rsi'].iloc[-1] > rsi_oversold:
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "rsi_oversold_exit",
                    "direction": "bullish",
                    "description": "RSI exited oversold territory",
                    "strength": "high" if market_regime["market_condition"] == "oversold" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.05,  # 5% target
                    "stop_loss": recent_df['close'].iloc[-1] * 0.97  # 3% stop loss
                })
            
            # Check for RSI moving out of overbought territory
            elif recent_df['rsi'].iloc[-2] >= rsi_overbought and recent_df['rsi'].iloc[-1] < rsi_overbought:
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "rsi_overbought_exit",
                    "direction": "bearish",
                    "description": "RSI exited overbought territory",
                    "strength": "high" if market_regime["market_condition"] == "overbought" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.95,  # 5% target
                    "stop_loss": recent_df['close'].iloc[-1] * 1.03  # 3% stop loss
                })
        
        # Bollinger Band Bounces
        if "bollinger_lower" in recent_df.columns and "bollinger_upper" in recent_df.columns:
            # Check for bounce off lower band
            if recent_df['low'].iloc[-2] <= recent_df['bollinger_lower'].iloc[-2] and \
               recent_df['close'].iloc[-1] > recent_df['bollinger_lower'].iloc[-1] and \
               recent_df['close'].iloc[-1] > recent_df['close'].iloc[-2]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "bollinger_bounce_lower",
                    "direction": "bullish",
                    "description": "Price bounced off lower Bollinger Band",
                    "strength": "high" if market_regime["market_type"] == "ranging" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['bollinger_middle'].iloc[-1],  # Target middle band
                    "stop_loss": recent_df['low'].iloc[-1] * 0.99  # Stop just below the low
                })
            
            # Check for bounce off upper band
            elif recent_df['high'].iloc[-2] >= recent_df['bollinger_upper'].iloc[-2] and \
                 recent_df['close'].iloc[-1] < recent_df['bollinger_upper'].iloc[-1] and \
                 recent_df['close'].iloc[-1] < recent_df['close'].iloc[-2]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "bollinger_bounce_upper",
                    "direction": "bearish",
                    "description": "Price bounced off upper Bollinger Band",
                    "strength": "high" if market_regime["market_type"] == "ranging" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['bollinger_middle'].iloc[-1],  # Target middle band
                    "stop_loss": recent_df['high'].iloc[-1] * 1.01  # Stop just above the high
                })
        
        # Bollinger Band Breakouts
        if "bollinger_lower" in recent_df.columns and "bollinger_upper" in recent_df.columns:
            # Check for upper band breakout
            if recent_df['close'].iloc[-2] <= recent_df['bollinger_upper'].iloc[-2] and \
               recent_df['close'].iloc[-1] > recent_df['bollinger_upper'].iloc[-1]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "bollinger_breakout_upper",
                    "direction": "bullish",
                    "description": "Price broke above upper Bollinger Band",
                    "strength": "high" if market_regime["market_type"] == "trending" and \
                                          market_regime["trend_direction"] == "bullish" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.03,  # 3% target
                    "stop_loss": recent_df['bollinger_middle'].iloc[-1]  # Stop at middle band
                })
            
            # Check for lower band breakout
            elif recent_df['close'].iloc[-2] >= recent_df['bollinger_lower'].iloc[-2] and \
                 recent_df['close'].iloc[-1] < recent_df['bollinger_lower'].iloc[-1]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "bollinger_breakout_lower",
                    "direction": "bearish",
                    "description": "Price broke below lower Bollinger Band",
                    "strength": "high" if market_regime["market_type"] == "trending" and \
                                          market_regime["trend_direction"] == "bearish" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.97,  # 3% target
                    "stop_loss": recent_df['bollinger_middle'].iloc[-1]  # Stop at middle band
                })
        
        # Parabolic SAR Signals
        if "sar" in recent_df.columns:
            # Check for bullish signal (price crosses above SAR)
            if recent_df['close'].iloc[-2] <= recent_df['sar'].iloc[-2] and \
               recent_df['close'].iloc[-1] > recent_df['sar'].iloc[-1]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "psar_signal",
                    "direction": "bullish",
                    "description": "Price crossed above Parabolic SAR",
                    "strength": "high" if market_regime["trend_direction"] == "bullish" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.04,  # 4% target
                    "stop_loss": recent_df['sar'].iloc[-1]  # SAR value as stop loss
                })
            
            # Check for bearish signal (price crosses below SAR)
            elif recent_df['close'].iloc[-2] >= recent_df['sar'].iloc[-2] and \
                 recent_df['close'].iloc[-1] < recent_df['sar'].iloc[-1]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "psar_signal",
                    "direction": "bearish",
                    "description": "Price crossed below Parabolic SAR",
                    "strength": "high" if market_regime["trend_direction"] == "bearish" else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.96,  # 4% target
                    "stop_loss": recent_df['sar'].iloc[-1]  # SAR value as stop loss
                })
        
        # Check for volume confirmation of price moves
        if "volume_ratio" in recent_df.columns:
            # High volume on bullish candle
            if recent_df['close'].iloc[-1] > recent_df['open'].iloc[-1] and \
               recent_df['volume_ratio'].iloc[-1] > 1.5:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "volume_confirmation",
                    "direction": "bullish",
                    "description": "Bullish candle with high volume",
                    "strength": "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.03,  # 3% target
                    "stop_loss": recent_df['low'].iloc[-1] * 0.99  # Just below the low
                })
            
            # High volume on bearish candle
            elif recent_df['close'].iloc[-1] < recent_df['open'].iloc[-1] and \
                 recent_df['volume_ratio'].iloc[-1] > 1.5:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "volume_confirmation",
                    "direction": "bearish",
                    "description": "Bearish candle with high volume",
                    "strength": "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.97,  # 3% target
                    "stop_loss": recent_df['high'].iloc[-1] * 1.01  # Just above the high
                })
        
        # Check for Ichimoku signals
        if all(col in recent_df.columns for col in ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b"]):
            # Bullish TK Cross
            if recent_df['tenkan_sen'].iloc[-2] <= recent_df['kijun_sen'].iloc[-2] and \
               recent_df['tenkan_sen'].iloc[-1] > recent_df['kijun_sen'].iloc[-1]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "ichimoku_tk_cross",
                    "direction": "bullish",
                    "description": "Tenkan-sen crossed above Kijun-sen",
                    "strength": "high" if recent_df['close'].iloc[-1] > recent_df['senkou_span_a'].iloc[-1] and \
                                          recent_df['close'].iloc[-1] > recent_df['senkou_span_b'].iloc[-1] else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 1.05,  # 5% target
                    "stop_loss": recent_df['kijun_sen'].iloc[-1]  # Kijun-sen as stop loss
                })
            
            # Bearish TK Cross
            elif recent_df['tenkan_sen'].iloc[-2] >= recent_df['kijun_sen'].iloc[-2] and \
                 recent_df['tenkan_sen'].iloc[-1] < recent_df['kijun_sen'].iloc[-1]:
                
                signals.append({
                    "timestamp": recent_df['timestamp'].iloc[-1],
                    "type": "ichimoku_tk_cross",
                    "direction": "bearish",
                    "description": "Tenkan-sen crossed below Kijun-sen",
                    "strength": "high" if recent_df['close'].iloc[-1] < recent_df['senkou_span_a'].iloc[-1] and \
                                          recent_df['close'].iloc[-1] < recent_df['senkou_span_b'].iloc[-1] else "medium",
                    "price": recent_df['close'].iloc[-1],
                    "target_price": recent_df['close'].iloc[-1] * 0.95,  # 5% target
                    "stop_loss": recent_df['kijun_sen'].iloc[-1]  # Kijun-sen as stop loss
                })
        
        # Sort signals by timestamp
        signals.sort(key=lambda x: x['timestamp'])
        
        return signals
    
    def _compute_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute the current market trend.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with trend details
        """
        # Use only the last 50 bars (or less if not available)
        recent_df = df.iloc[-min(50, len(df)):].copy()
        
        # Simple trend determination based on closing prices
        close_prices = recent_df['close'].values
        first_price = close_prices[0]
        last_price = close_prices[-1]
        
        # Trend direction
        if last_price > first_price:
            direction = "bullish"
            change_percent = ((last_price / first_price) - 1) * 100
        else:
            direction = "bearish"
            change_percent = ((first_price / last_price) - 1) * 100
        
        # Linear regression for trend line
        x = np.arange(len(close_prices))
        slope, intercept = np.polyfit(x, close_prices, 1)
        
        # Calculate R-squared to assess strength of trend
        y_pred = slope * x + intercept
        ss_tot = np.sum((close_prices - np.mean(close_prices)) ** 2)
        ss_res = np.sum((close_prices - y_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Determine trend strength
        if r_squared > 0.7:
            strength = "strong"
        elif r_squared > 0.3:
            strength = "moderate"
        else:
            strength = "weak"
        
        # Check for trend acceleration/deceleration
        half_point = len(close_prices) // 2
        first_half_slope, _ = np.polyfit(np.arange(half_point), close_prices[:half_point], 1)
        second_half_slope, _ = np.polyfit(np.arange(half_point), close_prices[half_point:], 1)
        
        if abs(second_half_slope) > abs(first_half_slope):
            acceleration = "accelerating"
        elif abs(second_half_slope) < abs(first_half_slope) * 0.5:
            acceleration = "decelerating"
        else:
            acceleration = "steady"
        
        # Determine higher timeframe trend (if we have enough data)
        higher_tf_direction = "unknown"
        if len(df) >= 200:
            long_term_df = df.iloc[-200:].copy()
            long_term_sma = long_term_df['close'].rolling(window=50).mean()
            
            if long_term_sma.iloc[-1] > long_term_sma.iloc[-50]:
                higher_tf_direction = "bullish"
            else:
                higher_tf_direction = "bearish"
        
        return {
            "direction": direction,
            "strength": strength,
            "change_percent": round(change_percent, 2),
            "slope": slope,
            "r_squared": r_squared,
            "acceleration": acceleration,
            "higher_timeframe": higher_tf_direction
        }
    
    def _compute_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute the current market volatility.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with volatility details
        """
        # Use recent data for volatility calculation
        recent_df = df.iloc[-min(20, len(df)):].copy()
        
        # Calculate daily returns
        returns = recent_df['close'].pct_change().dropna().values
        
        # Calculate standard deviation of returns
        std_dev = np.std(returns) * 100  # Convert to percentage
        
        # Determine volatility level
        if "atr_percent" in recent_df.columns:
            atr_percent = recent_df['atr_percent'].mean()
            volatility_score = (atr_percent + std_dev) / 2
            
            if volatility_score < 1:
                level = "very_low"
            elif volatility_score < 2:
                level = "low"
            elif volatility_score < 4:
                level = "moderate"
            elif volatility_score < 7:
                level = "high"
            else:
                level = "very_high"
        else:
            # Fallback if ATR not available
            if std_dev < 1:
                level = "very_low"
            elif std_dev < 2:
                level = "low"
            elif std_dev < 3:
                level = "moderate"
            elif std_dev < 5:
                level = "high"
            else:
                level = "very_high"
        
        # Check for volatility expansion
        if "bollinger_width" in recent_df.columns:
            recent_bb_width = recent_df['bollinger_width'].iloc[-5:].values
            bb_width_change = (recent_bb_width[-1] / recent_bb_width[0] - 1) * 100
            
            if bb_width_change > 20:
                expansion = "expanding"
            elif bb_width_change < -20:
                expansion = "contracting"
            else:
                expansion = "stable"
        else:
            expansion = "unknown"
        
        return {
            "level": level,
            "std_dev_percent": round(std_dev, 2),
            "expansion": expansion,
            "note": "Volatility calculated based on recent price action"
        }
    
    def _generate_analysis_summary(self, df: pd.DataFrame, trend: Dict[str, Any], 
                                 volatility: Dict[str, Any], market_regime: Dict[str, Any],
                                 signals: List[Dict[str, Any]], chart_patterns: List[Dict[str, Any]],
                                 candlestick_patterns: List[Dict[str, Any]], 
                                 support_resistance: Dict[str, Any]) -> str:
        """
        Generate a comprehensive analysis summary.
        
        Args:
            df: DataFrame with market data
            trend: Trend dictionary
            volatility: Volatility dictionary
            market_regime: Market regime dictionary
            signals: List of trading signals
            chart_patterns: List of chart patterns
            candlestick_patterns: List of candlestick patterns
            support_resistance: Support and resistance levels
            
        Returns:
            Analysis summary string
        """
        summary = []
        
        # General market description
        summary.append(f"The market is currently in a {market_regime['overall_regime']} regime with "
                      f"{trend['strength']} {trend['direction']} trend and {volatility['level']} volatility.")
        
        # Add trend details
        summary.append(f"Price has moved {trend['change_percent']}% over the analysis period and is "
                      f"{trend['acceleration']} in momentum.")
        
        # Add support/resistance information
        if support_resistance['nearest_support'] is not None:
            summary.append(f"Nearest support is at {support_resistance['nearest_support']:.2f}, "
                          f"{((df['close'].iloc[-1] / support_resistance['nearest_support']) - 1) * 100:.2f}% below current price.")
        
        if support_resistance['nearest_resistance'] is not None:
            summary.append(f"Nearest resistance is at {support_resistance['nearest_resistance']:.2f}, "
                          f"{((support_resistance['nearest_resistance'] / df['close'].iloc[-1]) - 1) * 100:.2f}% above current price.")
        
        # Add recent signals
        recent_signals = signals[-3:] if signals else []
        if recent_signals:
            signal_summary = "Recent signals: " + ", ".join([
                f"{s['description']} ({s['direction']})" for s in recent_signals
            ])
            summary.append(signal_summary)
        
        # Add recent patterns
        # Add recent patterns
        recent_chart_patterns = chart_patterns[-2:] if chart_patterns else []
        if recent_chart_patterns:
            pattern_summary = "Recent chart patterns: " + ", ".join([
                f"{p['type']} ({p['direction']})" for p in recent_chart_patterns
            ])
            summary.append(pattern_summary)
        
        recent_candlestick_patterns = candlestick_patterns[-2:] if candlestick_patterns else []
        if recent_candlestick_patterns:
            cs_pattern_summary = "Recent candlestick patterns: " + ", ".join([
                f"{p['type']} ({p['direction']})" for p in recent_candlestick_patterns
            ])
            summary.append(cs_pattern_summary)
        
        # Add trading recommendation based on analysis
        if signals:
            # Count recent bullish vs bearish signals
            bullish_count = sum(1 for s in signals[-5:] if s['direction'] == 'bullish')
            bearish_count = sum(1 for s in signals[-5:] if s['direction'] == 'bearish')
            
            if bullish_count > bearish_count and trend['direction'] == 'bullish':
                trading_bias = "bullish"
                summary.append("Trading bias: Bullish. Multiple indicators suggest potential upward movement.")
            elif bearish_count > bullish_count and trend['direction'] == 'bearish':
                trading_bias = "bearish"
                summary.append("Trading bias: Bearish. Multiple indicators suggest potential downward movement.")
            else:
                trading_bias = "neutral"
                summary.append("Trading bias: Neutral. Mixed signals suggest caution and reduced position sizes.")
        else:
            trading_bias = "neutral"
            summary.append("Trading bias: Neutral. Insufficient clear signals at this time.")
        
        # Add specific trading suggestions based on market regime
        if market_regime['overall_regime'] == 'bullish_trend':
            summary.append("Strategy suggestion: Look for pullbacks to key support levels as potential entry points for long positions.")
        elif market_regime['overall_regime'] == 'bearish_trend':
            summary.append("Strategy suggestion: Consider short positions on rallies to key resistance levels.")
        elif market_regime['overall_regime'] == 'sideways':
            summary.append("Strategy suggestion: Range trading between identified support and resistance levels may be effective.")
        elif market_regime['overall_regime'] == 'volatile':
            summary.append("Strategy suggestion: Exercise caution with reduced position sizes due to high volatility.")
        elif market_regime['overall_regime'] == 'oversold_range':
            summary.append("Strategy suggestion: Look for reversal signals for potential long entries from oversold conditions.")
        elif market_regime['overall_regime'] == 'overbought_range':
            summary.append("Strategy suggestion: Look for reversal signals for potential short entries from overbought conditions.")
        
        # Return the full summary
        return "\n".join(summary)
    
    def _save_analysis(self, symbol: str, exchange: str, timeframe: str, result: Dict[str, Any]) -> bool:
        """
        Save analysis result to database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            result: Analysis result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document for storage
            document = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "trend": result.get("trend"),
                "volatility": result.get("volatility"),
                "market_regime": result.get("market_regime"),
                "support_resistance": result.get("support_resistance"),
                "recent_signals": result.get("recent_signals"),
                "recent_chart_patterns": result.get("recent_chart_patterns"),
                "recent_candlestick_patterns": result.get("recent_candlestick_patterns"),
                "summary": result.get("summary")
            }
            
            # Insert into database
            result = self.db.technical_analysis_collection.insert_one(document)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving analysis result: {e}")
            return False

def find_patterns_by_symbol(self, symbol: str, exchange: str = "NSE", timeframe: str = "day", 
                           days: int = 100, pattern_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Find specific chart patterns for a symbol.
    
    Args:
        symbol: Stock symbol
        exchange: Stock exchange
        timeframe: Data timeframe
        days: Number of days to analyze
        pattern_types: List of pattern types to find (None for all patterns)
        
    Returns:
        Dictionary with pattern detection results
    """
    try:
        self.logger.info(f"Finding patterns for {symbol} ({exchange}) on {timeframe} timeframe")
        
        # Get market data
        if self.query_optimizer:
            data_result = self.query_optimizer.get_market_data(
                symbol=symbol,
                timeframe=timeframe,
                days=days,
                with_indicators=False
            )
            
            if data_result["status"] != "success":
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": data_result.get("error", "Failed to retrieve data")
                }
            
            data = data_result["data"]
        else:
            # Fallback to direct query
            start_date = datetime.now() - timedelta(days=days)
            cursor = self.db.market_data_collection.find({
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": {"$gte": start_date}
            }).sort("timestamp", 1)
            
            data = list(cursor)
        
        if not data:
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": "No data found"
            }
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Make sure required columns are present
        if not all(col in df.columns for col in ["timestamp", "open", "high", "low", "close"]):
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": "Incomplete data: missing required columns"
            }
        
        # Sort by timestamp
        df = df.sort_values("timestamp")
        
        # Detect patterns
        chart_patterns = self._detect_chart_patterns(df)
        candlestick_patterns = self._detect_candlestick_patterns(df)
        
        # Filter by pattern types if specified
        if pattern_types:
            chart_patterns = [p for p in chart_patterns if p["type"] in pattern_types]
            candlestick_patterns = [p for p in candlestick_patterns if p["type"] in pattern_types]
        
        # Format timestamp as string for proper JSON serialization
        df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "status": "success",
            "data_points": len(df),
            "date_range": f"{df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}",
            "chart_patterns": chart_patterns,
            "candlestick_patterns": candlestick_patterns,
            "pattern_count": len(chart_patterns) + len(candlestick_patterns)
        }
        
    except Exception as e:
        self.logger.error(f"Error finding patterns for {symbol}: {e}")
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "status": "error",
            "error": str(e)
        }
    
def percentile_rank(series, value):
    """
    Calculate the percentile rank of a value in a series.
    
    Args:
        series: Series of values
        value: Value to find rank for
        
    Returns:
        Percentile rank (0-100)
    """
    if len(series) == 0:
        return 50  # default to median if no data
    
    # Count values less than the given value
    count_less = sum(1 for x in series if x < value)
    
    # Calculate percentile rank
    return (count_less / len(series)) * 100