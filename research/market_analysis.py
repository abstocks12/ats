"""
Market Regime Analysis System

This module provides market regime classification and analysis capabilities for the automated trading system.
It identifies different market regimes and their characteristics to inform trading strategy selection.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
import statistics
from scipy import stats

class MarketAnalyzer:
    """
    Provides market regime analysis capabilities for trading decisions.
    Identifies market regimes and their characteristics across different timeframes.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the market analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get query optimizer if available
        self.query_optimizer = getattr(self.db, 'get_query_optimizer', lambda: None)()
        
        # Define analysis parameters
        self.analysis_params = {
            # Regime classification
            "min_trend_adx": 20.0,           # Minimum ADX for trending regime
            "strong_trend_adx": 30.0,        # Strong trend ADX threshold
            "weak_range_adx": 15.0,          # ADX below this suggests ranging market
            "high_volatility_atr": 2.5,      # ATR% above this indicates high volatility
            "low_volatility_atr": 1.0,       # ATR% below this indicates low volatility
            
            # Market cycle thresholds
            "overbought_rsi": 70.0,          # RSI above this suggests overbought conditions
            "oversold_rsi": 30.0,            # RSI below this suggests oversold conditions
            
            # Market breadth thresholds
            "bullish_advance_decline": 2.0,  # A/D ratio above this is bullish
            "bearish_advance_decline": 0.5,  # A/D ratio below this is bearish
            
            # Correlation thresholds
            "high_correlation": 0.7,         # Correlation above this is considered high
            "inverse_correlation": -0.7,     # Correlation below this is considered inverse
            
            # Timeframes for analysis
            "short_term_days": 20,           # Short-term analysis period (days)
            "medium_term_days": 60,          # Medium-term analysis period (days)
            "long_term_days": 200            # Long-term analysis period (days)
        }
        
        # Market regime characteristics (for reference)
        self.regime_characteristics = {
            "bull_trending": {
                "description": "Strong uptrend with momentum",
                "indicators": {
                    "adx": "High (>25)",
                    "trend_direction": "Bullish",
                    "volatility": "Moderate to high",
                    "volume": "Above average on advances"
                },
                "strategy_types": ["trend_following", "breakout", "momentum"]
            },
            "bear_trending": {
                "description": "Strong downtrend with momentum",
                "indicators": {
                    "adx": "High (>25)",
                    "trend_direction": "Bearish",
                    "volatility": "Moderate to high",
                    "volume": "Above average on declines"
                },
                "strategy_types": ["trend_following", "breakdown", "short"]
            },
            "bull_ranging": {
                "description": "Sideways movement with upward bias",
                "indicators": {
                    "adx": "Low (<20)",
                    "trend_direction": "Neutral with bullish bias",
                    "volatility": "Low to moderate",
                    "volume": "Below average"
                },
                "strategy_types": ["mean_reversion", "range_trading", "support_resistance"]
            },
            "bear_ranging": {
                "description": "Sideways movement with downward bias",
                "indicators": {
                    "adx": "Low (<20)",
                    "trend_direction": "Neutral with bearish bias",
                    "volatility": "Low to moderate",
                    "volume": "Below average"
                },
                "strategy_types": ["mean_reversion", "range_trading", "resistance_support"]
            },
            "high_volatility": {
                "description": "Large price swings with unclear direction",
                "indicators": {
                    "adx": "Variable",
                    "trend_direction": "Unclear",
                    "volatility": "High",
                    "volume": "Usually high"
                },
                "strategy_types": ["volatility_based", "options_strategies", "reduced_position_size"]
            },
            "low_volatility": {
                "description": "Minimal price movement in narrow range",
                "indicators": {
                    "adx": "Very low (<15)",
                    "trend_direction": "Neutral",
                    "volatility": "Low",
                    "volume": "Below average"
                },
                "strategy_types": ["breakout_anticipation", "accumulation", "income_strategies"]
            },
            "early_bull": {
                "description": "Beginning of bullish trend, transitioning from bear",
                "indicators": {
                    "adx": "Rising",
                    "trend_direction": "Turning bullish",
                    "volatility": "Decreasing",
                    "volume": "Increasing on advances"
                },
                "strategy_types": ["early_trend", "accumulation", "bottom_fishing"]
            },
            "late_bull": {
                "description": "Mature bullish trend, potentially overextended",
                "indicators": {
                    "adx": "High but potentially plateauing",
                    "trend_direction": "Bullish but slowing",
                    "volatility": "Often decreasing",
                    "volume": "Potentially diverging"
                },
                "strategy_types": ["trend_following_with_caution", "profit_taking", "reduced_exposure"]
            },
            "early_bear": {
                "description": "Beginning of bearish trend, transitioning from bull",
                "indicators": {
                    "adx": "Rising",
                    "trend_direction": "Turning bearish",
                    "volatility": "Increasing",
                    "volume": "Increasing on declines"
                },
                "strategy_types": ["early_short", "hedging", "protective_positions"]
            },
            "late_bear": {
                "description": "Mature bearish trend, potentially oversold",
                "indicators": {
                    "adx": "High but potentially plateauing",
                    "trend_direction": "Bearish but slowing",
                    "volatility": "Often decreasing",
                    "volume": "Potentially diverging"
                },
                "strategy_types": ["selective_accumulation", "bottoming_patterns", "reduced_shorts"]
            }
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
    
    def analyze_market_regime(self, symbol: str, exchange: str = "NSE", 
                            timeframe: str = "day") -> Dict[str, Any]:
        """
        Analyze market regime for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe ('1min', '5min', '15min', 'hour', 'day', etc.)
            
        Returns:
            Dictionary containing market regime analysis
        """
        try:
            self.logger.info(f"Analyzing market regime for {symbol} ({exchange}) on {timeframe} timeframe")
            
            # Retrieve data for different timeframes
            short_term_data = self._get_market_data(symbol, exchange, timeframe, self.analysis_params["short_term_days"])
            medium_term_data = self._get_market_data(symbol, exchange, timeframe, self.analysis_params["medium_term_days"])
            long_term_data = self._get_market_data(symbol, exchange, timeframe, self.analysis_params["long_term_days"])
            
            if not short_term_data or not medium_term_data or not long_term_data:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Insufficient data for analysis"
                }
            
            # Convert to DataFrames
            short_term_df = pd.DataFrame(short_term_data)
            medium_term_df = pd.DataFrame(medium_term_data)
            long_term_df = pd.DataFrame(long_term_data)
            
            # Make sure required columns are present
            required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
            for df, term in [(short_term_df, "short-term"), (medium_term_df, "medium-term"), (long_term_df, "long-term")]:
                if not all(col in df.columns for col in required_columns):
                    return {
                        "symbol": symbol,
                        "exchange": exchange,
                        "timeframe": timeframe,
                        "status": "error",
                        "error": f"Incomplete {term} data: missing required columns"
                    }
            
            # Sort by timestamp
            short_term_df = short_term_df.sort_values("timestamp")
            medium_term_df = medium_term_df.sort_values("timestamp")
            long_term_df = long_term_df.sort_values("timestamp")
            
            # Calculate technical indicators for each timeframe
            short_term_df = self._calculate_regime_indicators(short_term_df)
            medium_term_df = self._calculate_regime_indicators(medium_term_df)
            long_term_df = self._calculate_regime_indicators(long_term_df)
            
            # Identify market regimes for each timeframe
            short_term_regime = self._identify_market_regime(short_term_df)
            medium_term_regime = self._identify_market_regime(medium_term_df)
            long_term_regime = self._identify_market_regime(long_term_df)
            
            # Analyze market cycle
            market_cycle = self._analyze_market_cycle(short_term_df, medium_term_df, long_term_df)
            
            # Analyze volatility regimes
            volatility_regime = self._analyze_volatility_regime(short_term_df, medium_term_df)
            
            # Analyze mean reversion vs momentum
            mean_reversion_momentum = self._analyze_mean_reversion_vs_momentum(short_term_df, medium_term_df)
            
            # Check for regime transitions
            regime_transitions = self._detect_regime_transitions(short_term_df)
            
            # Recommend trading approaches based on regimes
            trading_approaches = self._recommend_trading_approaches(
                short_term_regime, medium_term_regime, long_term_regime,
                volatility_regime, mean_reversion_momentum, market_cycle
            )
            
            # Analyze market breadth (using index data if symbol is an index)
            market_breadth = None
            if symbol.endswith("NIFTY") or symbol.endswith("SENSEX"):
                market_breadth = self._analyze_market_breadth(symbol, exchange, timeframe)
            
            # Format timestamp as string for proper JSON serialization
            for df in [short_term_df, medium_term_df, long_term_df]:
                if isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate regime summary
            regime_summary = self._generate_regime_summary(
                short_term_regime, medium_term_regime, long_term_regime,
                volatility_regime, market_cycle, regime_transitions
            )
            
            # Assemble the analysis result
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "status": "success",
                "regimes": {
                    "short_term": short_term_regime,
                    "medium_term": medium_term_regime,
                    "long_term": long_term_regime
                },
                "market_cycle": market_cycle,
                "volatility_regime": volatility_regime,
                "mean_reversion_momentum": mean_reversion_momentum,
                "regime_transitions": regime_transitions,
                "market_breadth": market_breadth,
                "trading_approaches": trading_approaches,
                "regime_summary": regime_summary,
                "recent_indicators": {
                    "short_term": short_term_df.iloc[-5:][['timestamp', 'close', 'adx', 'atr_percent', 'rsi', 'trend_direction']].to_dict('records'),
                    "medium_term": medium_term_df.iloc[-1][['adx', 'atr_percent', 'rsi', 'trend_direction']]
                }
            }
            
            # Save analysis result to database
            self._save_analysis(symbol, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing market regime for {symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e)
            }
    
    def _get_market_data(self, symbol: str, exchange: str, timeframe: str, days: int) -> List[Dict[str, Any]]:
        """
        Get market data for analysis.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days of data to retrieve
            
        Returns:
            List of market data points
        """
        try:
            # Use query optimizer if available
            if self.query_optimizer:
                data_result = self.query_optimizer.get_market_data(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                    with_indicators=True
                )
                
                if data_result.get("status") == "success":
                    return data_result.get("data", [])
            
            # Fallback to direct database query
            start_date = datetime.now() - timedelta(days=days)
            cursor = self.db.market_data_collection.find({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": {"$gte": start_date}
            }).sort("timestamp", 1)
            
            data = list(cursor)
            
            if not data:
                self.logger.warning(f"No data found for {symbol} ({exchange}) on {timeframe} timeframe")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            return []
    
    def _calculate_regime_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for regime analysis.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added indicators
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Check if indicators already exist (may be included from database)
        indicators_exist = all(ind in df.columns for ind in ["adx", "atr", "rsi"])
        
        if not indicators_exist:
            # Calculate Average Directional Index (ADX)
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # True Range
            tr1 = abs(high[1:] - low[1:])
            tr2 = abs(high[1:] - close[:-1])
            tr3 = abs(low[1:] - close[:-1])
            tr = np.maximum(np.maximum(tr1, tr2), tr3)
            
            # Smoothed True Range (ATR)
            atr = np.zeros_like(close)
            atr[14] = np.mean(tr[:14])
            for i in range(15, len(close)):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
            
            # Plus and Minus Directional Movement
            plus_dm = np.zeros_like(close)
            minus_dm = np.zeros_like(close)
            
            for i in range(1, len(close)):
                up_move = high[i] - high[i-1]
                down_move = low[i-1] - low[i]
                
                if up_move > down_move and up_move > 0:
                    plus_dm[i] = up_move
                else:
                    plus_dm[i] = 0
                
                if down_move > up_move and down_move > 0:
                    minus_dm[i] = down_move
                else:
                    minus_dm[i] = 0
            
            # Smoothed Plus and Minus Directional Movement
            smoothed_plus_dm = np.zeros_like(close)
            smoothed_minus_dm = np.zeros_like(close)
            
            smoothed_plus_dm[14] = np.sum(plus_dm[1:15])
            smoothed_minus_dm[14] = np.sum(minus_dm[1:15])
            
            for i in range(15, len(close)):
                smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / 14) + plus_dm[i]
                smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / 14) + minus_dm[i]
            
            # Plus and Minus Directional Index
            plus_di = np.zeros_like(close)
            minus_di = np.zeros_like(close)
            
            for i in range(14, len(close)):
                if atr[i] > 0:
                    plus_di[i] = 100 * smoothed_plus_dm[i] / atr[i]
                    minus_di[i] = 100 * smoothed_minus_dm[i] / atr[i]
            
            # Directional Index and Average Directional Index
            dx = np.zeros_like(close)
            for i in range(14, len(close)):
                if (plus_di[i] + minus_di[i]) > 0:
                    dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
            
            adx = np.zeros_like(close)
            adx[27] = np.mean(dx[14:28])
            
            for i in range(28, len(close)):
                adx[i] = (adx[i-1] * 13 + dx[i]) / 14
            
            df['adx'] = adx
            df['plus_di'] = plus_di
            df['minus_di'] = minus_di
            
            # Determine trend direction
            df['trend_direction'] = 'neutral'
            df.loc[(df['plus_di'] > df['minus_di']), 'trend_direction'] = 'bullish'
            df.loc[(df['minus_di'] > df['plus_di']), 'trend_direction'] = 'bearish'
            
            # Average True Range and ATR percent
            df['atr'] = atr
            df['atr_percent'] = df['atr'] / df['close'] * 100
            
            # Relative Strength Index (RSI)
            delta = df['close'].diff()
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = -loss
            
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['sma20'] = df['close'].rolling(window=20).mean()
            df['stddev'] = df['close'].rolling(window=20).std()
            df['bollinger_upper'] = df['sma20'] + (df['stddev'] * 2)
            df['bollinger_lower'] = df['sma20'] - (df['stddev'] * 2)
            df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma20']
            
            # Moving Average Convergence Divergence (MACD)
            df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = df['ema12'] - df['ema26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Calculate additional regime-specific indicators if not already present
        
        # Percent from 50-day moving average (for trend strength)
        if 'sma50' not in df.columns:
            df['sma50'] = df['close'].rolling(window=50).mean()
        
        df['percent_from_sma50'] = (df['close'] - df['sma50']) / df['sma50'] * 100
        
        # Choppiness Index (for range vs trend identification)
        if 'choppiness_index' not in df.columns:
            n = 14
            df['atr_sum'] = df['atr'].rolling(window=n).sum()
            df['high_low_range'] = df['high'].rolling(window=n).max() - df['low'].rolling(window=n).min()
            
            # Avoid division by zero
            df.loc[df['high_low_range'] != 0, 'choppiness_index'] = 100 * np.log10(df.loc[df['high_low_range'] != 0, 'atr_sum'] / df.loc[df['high_low_range'] != 0, 'high_low_range']) / np.log10(n)
        
        # Volume Trend
        if 'volume_sma' not in df.columns and 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price Momentum
        if 'momentum' not in df.columns:
            df['momentum'] = df['close'].pct_change(periods=10) * 100
        
        # Clean up any NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _identify_market_regime(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify market regime based on technical indicators.
        
        Args:
            df: DataFrame with market indicators
            
        Returns:
            Dictionary with market regime analysis
        """
        # Get the most recent indicator values
        recent_data = df.iloc[-20:]  # Use last 20 bars for more stable assessment
        
        # Extract key indicators
        adx = recent_data['adx'].mean()
        atr_percent = recent_data['atr_percent'].mean()
        rsi = recent_data['rsi'].iloc[-1]
        trend_direction = recent_data['trend_direction'].iloc[-1]
        
        # Additional indicators
        choppiness_index = recent_data['choppiness_index'].mean() if 'choppiness_index' in recent_data.columns else 50
        percent_from_sma50 = recent_data['percent_from_sma50'].iloc[-1] if 'percent_from_sma50' in recent_data.columns else 0
        
        # Calculate price action statistics
        price_change = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
        max_high = recent_data['high'].max()
        min_low = recent_data['low'].min()
        price_range = (max_high - min_low) / min_low * 100
        
        # Check for volume confirmation if volume data is available
        volume_confirms_trend = False
        if 'volume_ratio' in recent_data.columns:
            # Volume should be higher on trend-confirming days
            if trend_direction == 'bullish':
                # Check if up days have higher volume
                up_days = recent_data[recent_data['close'] > recent_data['open']]
                down_days = recent_data[recent_data['close'] <= recent_data['open']]
                
                if not up_days.empty and not down_days.empty:
                    avg_up_volume = up_days['volume_ratio'].mean()
                    avg_down_volume = down_days['volume_ratio'].mean()
                    volume_confirms_trend = avg_up_volume > avg_down_volume
            elif trend_direction == 'bearish':
                # Check if down days have higher volume
                up_days = recent_data[recent_data['close'] > recent_data['open']]
                down_days = recent_data[recent_data['close'] <= recent_data['open']]
                
                if not up_days.empty and not down_days.empty:
                    avg_up_volume = up_days['volume_ratio'].mean()
                    avg_down_volume = down_days['volume_ratio'].mean()
                    volume_confirms_trend = avg_down_volume > avg_up_volume
        
        # Identify the primary regime
        primary_regime = "undefined"
        
        # Trending regimes
        if adx >= self.analysis_params["min_trend_adx"]:
            if trend_direction == "bullish":
                primary_regime = "bull_trending"
            elif trend_direction == "bearish":
                primary_regime = "bear_trending"
        
        # Ranging regimes
        elif adx < self.analysis_params["weak_range_adx"]:
            if trend_direction == "bullish" or percent_from_sma50 > 0:
                primary_regime = "bull_ranging"
            elif trend_direction == "bearish" or percent_from_sma50 < 0:
                primary_regime = "bear_ranging"
            else:
                primary_regime = "low_volatility" if atr_percent < self.analysis_params["low_volatility_atr"] else "bull_ranging"
        
        # Transitional regimes
        else:
            if trend_direction == "bullish":
                primary_regime = "early_bull" if adx > adx.shift(10).mean() else "bull_ranging"
            elif trend_direction == "bearish":
                primary_regime = "early_bear" if adx > adx.shift(10).mean() else "bear_ranging"
        
        # Consider volatility override
        if atr_percent >= self.analysis_params["high_volatility_atr"]:
            if primary_regime in ["bull_trending", "bear_trending"]:
                # Keep the trend designation but note high volatility
                volatility_regime = "high_volatility"
            else:
                # Override with high volatility regime
                primary_regime = "high_volatility"
                volatility_regime = "high_volatility"
        elif atr_percent <= self.analysis_params["low_volatility_atr"]:
            if primary_regime in ["bull_ranging", "bear_ranging"]:
                # Override with low volatility regime
                primary_regime = "low_volatility"
                volatility_regime = "low_volatility"
            else:
                volatility_regime = "low_volatility"
        else:
            volatility_regime = "normal_volatility"
        
        # Consider cycle position (are we late in a trend?)
        if primary_regime == "bull_trending" and rsi >= self.analysis_params["overbought_rsi"]:
            primary_regime = "late_bull"
        elif primary_regime == "bear_trending" and rsi <= self.analysis_params["oversold_rsi"]:
            primary_regime = "late_bear"
        
        # Calculate confidence level in regime identification
        confidence = 0.0
        
        # Higher ADX values increase confidence in trending regimes
        if primary_regime in ["bull_trending", "bear_trending", "early_bull", "early_bear", "late_bull", "late_bear"]:
            confidence += min(0.5, (adx - self.analysis_params["min_trend_adx"]) / (self.analysis_params["strong_trend_adx"] - self.analysis_params["min_trend_adx"]) * 0.5)
        else:
            confidence += min(0.5, (self.analysis_params["weak_range_adx"] - adx) / self.analysis_params["weak_range_adx"] * 0.5)
        
        # Consistent trend direction increases confidence
        trend_consistency = (recent_data['trend_direction'] == trend_direction).mean()
        confidence += trend_consistency * 0.2
        
        # Volume confirmation increases confidence
        if volume_confirms_trend:
            confidence += 0.1
        
        # Clear price action increases confidence
        if price_range > 5.0:
            confidence += min(0.1, price_range / 50.0)
        
        # Strong RSI values increase confidence in cycle position
        if primary_regime == "late_bull" and rsi > self.analysis_params["overbought_rsi"] + 5:
            confidence += 0.1
        elif primary_regime == "late_bear" and rsi < self.analysis_params["oversold_rsi"] - 5:
            confidence += 0.1
        
        # Cap confidence at 1.0
        confidence = min(1.0, confidence)
        
        # Prepare additional contextual information about the regime
        regime_context = self.regime_characteristics.get(primary_regime, {})
        
        return {
            "primary_regime": primary_regime,
            "confidence": confidence,
            "indicators": {
                "adx": adx,
                "trend_direction": trend_direction,
                "atr_percent": atr_percent,
                "rsi": rsi,
                "choppiness_index": choppiness_index,
                "percent_from_sma50": percent_from_sma50
            },
            "price_action": {
                "price_change_percent": price_change,
                "price_range_percent": price_range,
                "volume_confirms_trend": volume_confirms_trend
            },
            "volatility_regime": volatility_regime,
            "description": regime_context.get("description", ""),
            "strategy_types": regime_context.get("strategy_types", [])
        }
    
    def _analyze_market_cycle(self, short_term_df: pd.DataFrame, medium_term_df: pd.DataFrame, 
                           long_term_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market cycle stage based on multiple timeframes.
        
        Args:
            short_term_df: Short-term market data
            medium_term_df: Medium-term market data
            long_term_df: Long-term market data
            
        Returns:
            Dictionary with market cycle analysis
        """
        # Extract key indicators
        long_term_trend = self._determine_trend(long_term_df)
        medium_term_trend = self._determine_trend(medium_term_df)
        short_term_trend = self._determine_trend(short_term_df)
        
        # Get recent indicator values
        short_rsi = short_term_df['rsi'].iloc[-1]
        medium_rsi = medium_term_df['rsi'].iloc[-1]
        
        # Price relative to moving averages
        short_term_ma = short_term_df['sma20'].iloc[-1]
        medium_term_ma = medium_term_df['sma50'].iloc[-1] if 'sma50' in medium_term_df.columns else medium_term_df['sma20'].iloc[-1]
        long_term_ma = long_term_df['sma50'].iloc[-1] if 'sma50' in long_term_df.columns else long_term_df['sma20'].iloc[-1]
        
        current_price = short_term_df['close'].iloc[-1]
        
        # Check momentum trends
        short_momentum = short_term_df['momentum'].iloc[-1] if 'momentum' in short_term_df.columns else 0
        medium_momentum = medium_term_df['momentum'].iloc[-1] if 'momentum' in medium_term_df.columns else 0
        
        # Identify market cycle stage
        cycle_stage = "undefined"
        
        # Accumulation phase: Sideways after bearish, oversold conditions improving
        if (long_term_trend == "bearish" and medium_term_trend in ["neutral", "bullish"] and 
            short_term_trend in ["neutral", "bullish"] and 
            short_rsi > 30 and short_rsi < 60 and 
            current_price > short_term_ma and
            short_momentum > 0):
            cycle_stage = "accumulation"
        
        # Markup phase: Bullish trend across timeframes, strong momentum
        elif (long_term_trend in ["neutral", "bullish"] and medium_term_trend == "bullish" and 
              short_term_trend == "bullish" and
              current_price > short_term_ma and current_price > medium_term_ma and
              short_momentum > 0 and medium_momentum > 0):
            
            # Early markup: Just starting the bullish trend
            if long_term_trend == "neutral" and medium_rsi < 70:
                cycle_stage = "early_markup"
            # Late markup: Extended bullish trend, possibly overbought
            elif short_rsi > 70 or medium_rsi > 70:
                cycle_stage = "late_markup"
            else:
                cycle_stage = "markup"
        
        # Distribution phase: Sideways after bullish, overbought conditions deteriorating
        elif (long_term_trend == "bullish" and medium_term_trend in ["neutral", "bearish"] and 
              short_rsi < 70 and short_rsi > 40 and
              current_price < short_term_ma and
              short_momentum < 0):
            cycle_stage = "distribution"
        
        # Markdown phase: Bearish trend across timeframes, weak momentum
        elif (long_term_trend in ["neutral", "bearish"] and medium_term_trend == "bearish" and 
              short_term_trend == "bearish" and
              current_price < short_term_ma and current_price < medium_term_ma and
              short_momentum < 0 and medium_momentum < 0):
            
            # Early markdown: Just starting the bearish trend
            if long_term_trend == "neutral" and medium_rsi > 30:
                cycle_stage = "early_markdown"
            # Late markdown: Extended bearish trend, possibly oversold
            elif short_rsi < 30 or medium_rsi < 30:
                cycle_stage = "late_markdown"
            else:
                cycle_stage = "markdown"
        
        # Calculate cycle position (0-100%, where 0 is start of accumulation, 100 is end of markdown)
        cycle_position = 0
        
        if cycle_stage == "accumulation":
            # 0-25% range, based on how far through accumulation we are
            cycle_position = 12.5 + (short_rsi - 30) / 30 * 12.5
        elif cycle_stage == "early_markup":
            cycle_position = 25 + 8.33  # First third of markup (25-50%)
        elif cycle_stage == "markup":
            cycle_position = 33.33 + 8.33  # Second third of markup
        elif cycle_stage == "late_markup":
            cycle_position = 41.66 + 8.33  # Last third of markup
        elif cycle_stage == "distribution":
            # 50-75% range, based on how far through distribution we are
            cycle_position = 50 + (70 - short_rsi) / 30 * 25
        elif cycle_stage == "early_markdown":
            cycle_position = 75 + 8.33  # First third of markdown (75-100%)
        elif cycle_stage == "markdown":
            cycle_position = 83.33 + 8.33  # Second third of markdown
        elif cycle_stage == "late_markdown":
            cycle_position = 91.66 + 8.33  # Last third of markdown
        
        # Describe the cycle implications
        implications = {}
        
        if cycle_stage in ["accumulation", "early_markup"]:
            implications = {
                "trend_expectation": "Potential emerging uptrend",
                "risk_level": "Moderate but improving risk/reward",
                "positioning": "Gradual accumulation and position building",
                "time_horizon": "Medium to long-term"
            }
        elif cycle_stage in ["markup"]:
            implications = {
                "trend_expectation": "Established uptrend likely to continue",
                "risk_level": "Favorable risk/reward",
                "positioning": "Maintain or add to long positions",
                "time_horizon": "Medium-term"
            }
        elif cycle_stage in ["late_markup", "distribution"]:
            implications = {
                "trend_expectation": "Uptrend potentially exhausting",
                "risk_level": "Deteriorating risk/reward",
                "positioning": "Begin reducing position sizes and tighten stops",
                "time_horizon": "Short to medium-term"
            }
        elif cycle_stage in ["early_markdown", "markdown"]:
            implications = {
                "trend_expectation": "Established downtrend likely to continue",
                "risk_level": "Adverse risk/reward for long positions",
                "positioning": "Defensive positioning, consider short exposure",
                "time_horizon": "Medium-term"
            }
        elif cycle_stage in ["late_markdown"]:
            implications = {
                "trend_expectation": "Downtrend potentially exhausting",
                "risk_level": "Improving risk/reward for contrarians",
                "positioning": "Begin watchlist preparation for accumulation",
                "time_horizon": "Medium to long-term"
            }
        
        return {
            "cycle_stage": cycle_stage,
            "cycle_position_percent": round(cycle_position, 1),
            "trends": {
                "long_term": long_term_trend,
                "medium_term": medium_term_trend,
                "short_term": short_term_trend
            },
            "price_structure": {
                "vs_short_term_ma": "above" if current_price > short_term_ma else "below",
                "vs_medium_term_ma": "above" if current_price > medium_term_ma else "below",
                "vs_long_term_ma": "above" if current_price > long_term_ma else "below"
            },
            "momentum": {
                "short_term": "positive" if short_momentum > 0 else "negative",
                "medium_term": "positive" if medium_momentum > 0 else "negative"
            },
            "implications": implications
        }
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """
        Determine the overall trend direction from price data.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Trend direction string
        """
        # Use recent data for trend determination
        recent_df = df.iloc[-20:]
        
        # Get average values
        avg_adx = recent_df['adx'].mean()
        avg_trend = recent_df['trend_direction'].mode()[0]
        
        # Price slope
        close_prices = recent_df['close'].values
        slope = np.polyfit(np.arange(len(close_prices)), close_prices, 1)[0]
        
        # Price vs moving averages
        price = close_prices[-1]
        ma_short = recent_df['sma20'].iloc[-1]
        ma_medium = recent_df['sma50'].iloc[-1] if 'sma50' in recent_df.columns else ma_short
        
        # Determine trend based on multiple factors
        if avg_adx >= self.analysis_params["min_trend_adx"]:
            # Strong trend present
            return avg_trend
        elif slope > 0 and price > ma_short and price > ma_medium:
            # Positive slope and above moving averages
            return "bullish"
        elif slope < 0 and price < ma_short and price < ma_medium:
            # Negative slope and below moving averages
            return "bearish"
        else:
            # No clear trend
            return "neutral"
    
    def _analyze_volatility_regime(self, short_term_df: pd.DataFrame, 
                               medium_term_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volatility regime characteristics.
        
        Args:
            short_term_df: Short-term market data
            medium_term_df: Medium-term market data
            
        Returns:
            Dictionary with volatility regime analysis
        """
        # Extract volatility metrics
        recent_short_df = short_term_df.iloc[-20:]
        recent_medium_df = medium_term_df.iloc[-30:]
        
        # Calculate current volatility
        current_atr_percent = recent_short_df['atr_percent'].mean()
        
        # Calculate historical volatility (standard deviation of returns)
        returns = recent_medium_df['close'].pct_change().dropna() * 100
        historical_volatility = returns.std()
        
        # Calculate volatility of volatility (how stable is the volatility)
        atr_volatility = recent_medium_df['atr_percent'].std() / recent_medium_df['atr_percent'].mean()
        
        # Determine Bollinger Band width trend
        if 'bollinger_width' in recent_short_df.columns:
            bb_width = recent_short_df['bollinger_width'].iloc[-1]
            bb_width_avg = recent_short_df['bollinger_width'].mean()
            bb_width_trend = "expanding" if bb_width > bb_width_avg * 1.1 else "contracting" if bb_width < bb_width_avg * 0.9 else "stable"
        else:
            bb_width = None
            bb_width_trend = "unknown"
        
        # Get 90-day historical percentile of current volatility
        if len(medium_term_df) >= 90:
            history_90d = medium_term_df.iloc[-90:]['atr_percent']
            current_percentile = sum(history_90d < current_atr_percent) / len(history_90d) * 100
        else:
            current_percentile = 50  # Default to middle if insufficient history
        
        # Determine primary volatility regime
        if current_atr_percent >= self.analysis_params["high_volatility_atr"]:
            primary_regime = "high_volatility"
        elif current_atr_percent <= self.analysis_params["low_volatility_atr"]:
            primary_regime = "low_volatility"
        else:
            primary_regime = "normal_volatility"
        
        # Determine if we're in a volatility transition
        if bb_width_trend == "expanding" and primary_regime != "high_volatility":
            volatility_transition = "increasing"
        elif bb_width_trend == "contracting" and primary_regime != "low_volatility":
            volatility_transition = "decreasing"
        else:
            volatility_transition = "stable"
        
        # Analyze volatility clustering
        recent_high_vol_days = sum(recent_short_df['atr_percent'] > self.analysis_params["high_volatility_atr"])
        volatility_clustering = recent_high_vol_days / len(recent_short_df) > 0.3
        
        # Determine expected volatility regime duration
        if volatility_transition != "stable":
            expected_duration = "transitioning"
        elif primary_regime == "high_volatility" and volatility_clustering:
            expected_duration = "persistent"
        elif primary_regime == "low_volatility" and bb_width_trend == "contracting":
            expected_duration = "persistent"
        else:
            expected_duration = "temporary"
        
        # Prepare trading implications
        if primary_regime == "high_volatility":
            implications = {
                "position_sizing": "Reduce position sizes by 30-50%",
                "stop_placement": "Wider stops necessary to accommodate swings",
                "strategy_types": ["volatility-based strategies", "options strategies", "short-term mean reversion"],
                "risk_management": "Crucial to implement strict risk controls"
            }
        elif primary_regime == "low_volatility":
            implications = {
                "position_sizing": "Can increase position sizes by 10-30%",
                "stop_placement": "Tighter stops possible with less noise",
                "strategy_types": ["breakout anticipation", "trend following after breakout", "carry strategies"],
                "risk_management": "Watch for sudden volatility expansion"
            }
        else:  # normal_volatility
            implications = {
                "position_sizing": "Standard position sizing appropriate",
                "stop_placement": "Normal stop distances based on support/resistance",
                "strategy_types": ["balanced approach", "trend following", "swing trading"],
                "risk_management": "Regular risk management protocols"
            }
        
        return {
            "primary_regime": primary_regime,
            "current_volatility": {
                "atr_percent": current_atr_percent,
                "historical_volatility": historical_volatility,
                "percentile": current_percentile
            },
            "volatility_characteristics": {
                "transition": volatility_transition,
                "clustering": volatility_clustering,
                "expected_duration": expected_duration,
                "bollinger_width_trend": bb_width_trend
            },
            "implications": implications
        }
    
    def _analyze_mean_reversion_vs_momentum(self, short_term_df: pd.DataFrame, 
                                         medium_term_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze whether mean reversion or momentum strategies are more likely to succeed.
        
        Args:
            short_term_df: Short-term market data
            medium_term_df: Medium-term market data
            
        Returns:
            Dictionary with mean reversion vs momentum analysis
        """
        # Extract relevant metrics
        recent_short_df = short_term_df.iloc[-20:]
        
        # ADX for trend strength
        adx = recent_short_df['adx'].mean()
        
        # RSI for overbought/oversold
        rsi = recent_short_df['rsi'].iloc[-1]
        
        # Distance from moving averages
        if 'sma20' in recent_short_df.columns and 'sma50' in recent_short_df.columns:
            price = recent_short_df['close'].iloc[-1]
            sma20 = recent_short_df['sma20'].iloc[-1]
            sma50 = recent_short_df['sma50'].iloc[-1]
            
            deviation_sma20 = (price / sma20 - 1) * 100
            deviation_sma50 = (price / sma50 - 1) * 100
        else:
            deviation_sma20 = 0
            deviation_sma50 = 0
        
        # Autocorrelation of returns (positive = momentum, negative = mean reversion)
        returns = recent_short_df['close'].pct_change().dropna()
        if len(returns) > 5:
            autocorrelation = returns.autocorr(lag=1)
        else:
            autocorrelation = 0
        
        # Hurst exponent (>0.5 = trend, <0.5 = mean reversion)
        hurst = self._calculate_hurst_exponent(recent_short_df['close'].values)
        
        # Determine regime dominance
        if hurst > 0.6 and adx > self.analysis_params["min_trend_adx"] and autocorrelation > 0.1:
            dominant_regime = "momentum"
            dominance_score = (hurst - 0.5) * 2 + (adx / self.analysis_params["min_trend_adx"] - 1) + autocorrelation
        elif hurst < 0.4 and adx < self.analysis_params["min_trend_adx"] and autocorrelation < -0.1:
            dominant_regime = "mean_reversion"
            dominance_score = (0.5 - hurst) * 2 + (1 - adx / self.analysis_params["min_trend_adx"]) - autocorrelation
        else:
            # Mixed regime - determine which has stronger signals
            momentum_score = (hurst - 0.5) * 2 + (adx / self.analysis_params["min_trend_adx"] - 1) + autocorrelation
            reversion_score = (0.5 - hurst) * 2 + (1 - adx / self.analysis_params["min_trend_adx"]) - autocorrelation
            
            if momentum_score > reversion_score:
                dominant_regime = "weak_momentum"
                dominance_score = momentum_score
            else:
                dominant_regime = "weak_mean_reversion"
                dominance_score = reversion_score
        
        # Normalize dominance score
        dominance_score = min(1.0, max(0.0, dominance_score / 3))
        
        # Identify specific opportunities
        opportunities = []
        
        if dominant_regime in ["momentum", "weak_momentum"]:
            # Look for momentum opportunities
            if (adx > self.analysis_params["min_trend_adx"] and 
                ((rsi > 60 and rsi < 80) or (rsi < 40 and rsi > 20))):
                opportunities.append("trend_following")
            
            if adx > self.analysis_params["min_trend_adx"] * 1.5:
                opportunities.append("breakout")
            
            if abs(deviation_sma20) > 5 and abs(deviation_sma50) > 8:
                opportunities.append("trend_continuation")
                
        elif dominant_regime in ["mean_reversion", "weak_mean_reversion"]:
            # Look for mean reversion opportunities
            if rsi > 70:
                opportunities.append("overbought_reversal")
            elif rsi < 30:
                opportunities.append("oversold_reversal")
            
            if abs(deviation_sma20) > 5:
                opportunities.append("return_to_mean")
            
            if 'bollinger_width' in recent_short_df.columns:
                bb_width = recent_short_df['bollinger_width'].iloc[-1]
                if bb_width > recent_short_df['bollinger_width'].mean() * 1.5:
                    opportunities.append("bollinger_band_bounce")
        
        # Prepare implications for trading
        if dominant_regime in ["momentum", "weak_momentum"]:
            implications = {
                "trade_setup": "Look for pullbacks in existing trends for entry",
                "exit_strategy": "Use trailing stops rather than profit targets",
                "timeframe": "Higher timeframes typically show stronger momentum effects",
                "indicators": "Focus on trend-following indicators like moving averages and MACD"
            }
        else:
            implications = {
                "trade_setup": "Look for extreme moves away from moving averages",
                "exit_strategy": "Use profit targets based on historical mean",
                "timeframe": "Lower timeframes typically show stronger mean reversion effects",
                "indicators": "Focus on oscillators like RSI, Stochastic, and Bollinger Bands"
            }
        
        return {
            "dominant_regime": dominant_regime,
            "dominance_score": dominance_score,
            "hurst_exponent": hurst,
            "autocorrelation": autocorrelation,
            "indicators": {
                "adx": adx,
                "rsi": rsi,
                "deviation_from_sma20": deviation_sma20,
                "deviation_from_sma50": deviation_sma50
            },
            "opportunities": opportunities,
            "implications": implications
        }
    
    def _calculate_hurst_exponent(self, price_array: np.ndarray, max_lag: int = 20) -> float:
        """
        Calculate the Hurst Exponent to determine if a time series is:
        - trending (H > 0.5)
        - mean reverting (H < 0.5)
        - random walk (H = 0.5)
        
        Args:
            price_array: Array of prices
            max_lag: Maximum lag to use
            
        Returns:
            Hurst exponent value
        """
        if len(price_array) < max_lag * 2:
            return 0.5  # Default to random walk if insufficient data
        
        # Returns
        returns = np.diff(np.log(price_array))
        if len(returns) < max_lag:
            return 0.5
        
        # Calculate range over time
        tau = np.arange(2, min(max_lag, len(returns)//2))
        rs = np.zeros(len(tau))
        
        for idx, lag in enumerate(tau):
            # Get chunks of data
            chunks = len(returns) // lag
            if chunks == 0:
                continue
                
            # Reshape returns into chunks
            reshaped_returns = returns[:chunks * lag].reshape((chunks, lag))
            
            # Calculate mean of each chunk
            means = np.mean(reshaped_returns, axis=1)
            
            # Calculate cumulative sum of deviations for each chunk
            adjusted_returns = reshaped_returns - means[:, np.newaxis]
            cumulative_sum = np.cumsum(adjusted_returns, axis=1)
            
            # Calculate range and standard deviation for each chunk
            ranges = np.max(cumulative_sum, axis=1) - np.min(cumulative_sum, axis=1)
            stds = np.std(reshaped_returns, axis=1)
            
            # Calculate rescaled range for non-zero standard deviations
            valid_stds = stds > 0
            if np.sum(valid_stds) > 0:
                rs[idx] = np.mean(ranges[valid_stds] / stds[valid_stds])
            else:
                rs[idx] = np.nan
        
        # Remove NaN values
        tau = tau[~np.isnan(rs)]
        rs = rs[~np.isnan(rs)]
        
        if len(tau) < 2 or len(rs) < 2:
            return 0.5
        
        # Perform linear regression on log-log data
        log_tau = np.log(tau)
        log_rs = np.log(rs)
        
        # Calculate Hurst exponent (slope of log-log plot)
        hurst = np.polyfit(log_tau, log_rs, 1)[0]
        
        return hurst
    
    def _detect_regime_transitions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect recent or impending market regime transitions.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with regime transition analysis
        """
        # Use sufficient data for transition detection
        if len(df) < 60:
            return {
                "recent_transition": "insufficient_data",
                "impending_transition": "insufficient_data",
                "transition_signals": []
            }
        
        # Extract key indicators for different timeframes
        recent_df = df.iloc[-20:]
        previous_df = df.iloc[-40:-20]
        
        # ADX trend strength
        recent_adx = recent_df['adx'].mean()
        previous_adx = previous_df['adx'].mean()
        adx_trend = recent_adx - previous_adx
        
        # Volatility
        recent_atr = recent_df['atr_percent'].mean()
        previous_atr = previous_df['atr_percent'].mean()
        volatility_trend = recent_atr - previous_atr
        
        # Trend direction
        recent_trend = recent_df['trend_direction'].mode()[0]
        previous_trend = previous_df['trend_direction'].mode()[0]
        
        # Bollinger Band width
        if 'bollinger_width' in df.columns:
            recent_bb_width = recent_df['bollinger_width'].mean()
            previous_bb_width = previous_df['bollinger_width'].mean()
            bb_width_trend = (recent_bb_width / previous_bb_width - 1) * 100
        else:
            bb_width_trend = 0
        
        # RSI
        recent_rsi = recent_df['rsi'].iloc[-1]
        rsi_trend = recent_df['rsi'].diff().mean()
        
        # Detect recent transitions
        recent_transition = "none"
        
        # Transition to trending regime
        if previous_adx < self.analysis_params["min_trend_adx"] and recent_adx >= self.analysis_params["min_trend_adx"]:
            if recent_trend == "bullish":
                recent_transition = "range_to_bull_trend"
            else:
                recent_transition = "range_to_bear_trend"
        
        # Transition to ranging regime
        elif previous_adx >= self.analysis_params["min_trend_adx"] and recent_adx < self.analysis_params["min_trend_adx"]:
            recent_transition = "trend_to_range"
        
        # Transition to high volatility
        elif previous_atr < self.analysis_params["high_volatility_atr"] and recent_atr >= self.analysis_params["high_volatility_atr"]:
            recent_transition = "normal_to_high_volatility"
        
        # Transition to low volatility
        elif previous_atr > self.analysis_params["low_volatility_atr"] * 1.5 and recent_atr <= self.analysis_params["low_volatility_atr"]:
            recent_transition = "normal_to_low_volatility"
        
        # Detect impending transitions
        impending_transition = "none"
        transition_signals = []
        
        # Potential transition from range to trend
        if recent_adx < self.analysis_params["min_trend_adx"] and adx_trend > 2:
            if recent_trend == "bullish":
                impending_transition = "potential_range_to_bull_trend"
                transition_signals.append({
                    "signal": "increasing_adx",
                    "value": recent_adx,
                    "trend": adx_trend
                })
            else:
                impending_transition = "potential_range_to_bear_trend"
                transition_signals.append({
                    "signal": "increasing_adx",
                    "value": recent_adx,
                    "trend": adx_trend
                })
        
        # Potential transition from trend to range
        elif recent_adx >= self.analysis_params["min_trend_adx"] and adx_trend < -2:
            impending_transition = "potential_trend_to_range"
            transition_signals.append({
                "signal": "decreasing_adx",
                "value": recent_adx,
                "trend": adx_trend
            })
        
        # Potential volatility expansion
        if bb_width_trend > 20 and volatility_trend > 0.5:
            if impending_transition == "none":
                impending_transition = "potential_volatility_expansion"
            
            transition_signals.append({
                "signal": "expanding_bollinger_bands",
                "value": bb_width_trend,
                "trend": volatility_trend
            })
        
        # Potential trend reversal signals
        if recent_trend == "bullish" and recent_rsi > 70 and rsi_trend < 0:
            if impending_transition == "none":
                impending_transition = "potential_bull_to_bear"
            
            transition_signals.append({
                "signal": "overbought_rsi_declining",
                "value": recent_rsi,
                "trend": rsi_trend
            })
        elif recent_trend == "bearish" and recent_rsi < 30 and rsi_trend > 0:
            if impending_transition == "none":
                impending_transition = "potential_bear_to_bull"
            
            transition_signals.append({
                "signal": "oversold_rsi_rising",
                "value": recent_rsi,
                "trend": rsi_trend
            })
        
        # Check for divergences
        if len(recent_df) >= 10:
            price_trend = recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]
            
            # Bullish divergence
            if price_trend < 0 and recent_df['rsi'].iloc[-1] > recent_df['rsi'].iloc[0]:
                transition_signals.append({
                    "signal": "bullish_divergence",
                    "price_change": price_trend,
                    "rsi_change": recent_df['rsi'].iloc[-1] - recent_df['rsi'].iloc[0]
                })
                
                if impending_transition == "none":
                    impending_transition = "potential_bear_to_bull"
            
            # Bearish divergence
            elif price_trend > 0 and recent_df['rsi'].iloc[-1] < recent_df['rsi'].iloc[0]:
                transition_signals.append({
                    "signal": "bearish_divergence",
                    "price_change": price_trend,
                    "rsi_change": recent_df['rsi'].iloc[-1] - recent_df['rsi'].iloc[0]
                })
                
                if impending_transition == "none":
                    impending_transition = "potential_bull_to_bear"
        
        return {
            "recent_transition": recent_transition,
            "impending_transition": impending_transition,
            "transition_signals": transition_signals
        }
    
    def _analyze_market_breadth(self, index_symbol: str, exchange: str, 
                             timeframe: str) -> Optional[Dict[str, Any]]:
        """
        Analyze market breadth metrics for an index.
        
        Args:
            index_symbol: Index symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            
        Returns:
            Dictionary with market breadth analysis or None
        """
        # Skip if not an index
        if not (index_symbol.endswith("NIFTY") or index_symbol.endswith("SENSEX")):
            return None
        
        try:
            # Try to get market breadth data from database
            recent_breadth = self.db.market_breadth_collection.find_one({
                "index_symbol": index_symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }, sort=[("timestamp", -1)])
            
            if not recent_breadth:
                return None
            
            # Extract key metrics
            advance_decline_ratio = recent_breadth.get("advance_decline_ratio", 1.0)
            new_highs_lows_ratio = recent_breadth.get("new_highs_lows_ratio", 1.0)
            percent_above_ma50 = recent_breadth.get("percent_above_ma50", 50)
            percent_above_ma200 = recent_breadth.get("percent_above_ma200", 50)
            mcclellan_oscillator = recent_breadth.get("mcclellan_oscillator", 0)
            
            # Get historical data for comparison
            historical_breadth = list(self.db.market_breadth_collection.find({
                "index_symbol": index_symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }).sort("timestamp", -1).limit(10))
            
            # Calculate trends
            if len(historical_breadth) >= 5:
                ad_trend = advance_decline_ratio - historical_breadth[4].get("advance_decline_ratio", 1.0)
                hl_trend = new_highs_lows_ratio - historical_breadth[4].get("new_highs_lows_ratio", 1.0)
                ma50_trend = percent_above_ma50 - historical_breadth[4].get("percent_above_ma50", 50)
                ma200_trend = percent_above_ma200 - historical_breadth[4].get("percent_above_ma200", 50)
            else:
                ad_trend = 0
                hl_trend = 0
                ma50_trend = 0
                ma200_trend = 0
            
            # Determine market breadth condition
            if (advance_decline_ratio > self.analysis_params["bullish_advance_decline"] and 
                percent_above_ma50 > 70 and 
                new_highs_lows_ratio > 2.0):
                breadth_condition = "strong_bullish"
            elif (advance_decline_ratio > 1.5 or
                 (percent_above_ma50 > 60 and ad_trend > 0)):
                breadth_condition = "bullish"
            elif (advance_decline_ratio < self.analysis_params["bearish_advance_decline"] and 
                 percent_above_ma50 < 30 and 
                 new_highs_lows_ratio < 0.5):
                breadth_condition = "strong_bearish"
            elif (advance_decline_ratio < 0.7 or
                 (percent_above_ma50 < 40 and ad_trend < 0)):
                breadth_condition = "bearish"
            else:
                breadth_condition = "neutral"
            
            # Check for divergences
            price_trend = recent_breadth.get("index_change_percent", 0)
            
            breadth_divergence = "none"
            if price_trend > 0 and ad_trend < 0 and ma50_trend < 0:
                breadth_divergence = "bearish"
            elif price_trend < 0 and ad_trend > 0 and ma50_trend > 0:
                breadth_divergence = "bullish"
            
            # Generate implications
            implications = {}
            
            if breadth_condition in ["strong_bullish", "bullish"]:
                implications = {
                    "market_health": "Healthy market with broad participation",
                    "trend_confirmation": "Confirms bullish trend in the index",
                    "stock_selection": "Focus on leading stocks in strong sectors",
                    "risk_level": "Lower risk for long positions"
                }
            elif breadth_condition in ["strong_bearish", "bearish"]:
                implications = {
                    "market_health": "Weak market with deteriorating participation",
                    "trend_confirmation": "Confirms bearish trend in the index",
                    "stock_selection": "Focus on defensive sectors or short opportunities",
                    "risk_level": "Higher risk for long positions"
                }
            else:
                implications = {
                    "market_health": "Mixed market health with selective participation",
                    "trend_confirmation": "Uncertain trend direction",
                    "stock_selection": "Be selective and focus on stock-specific opportunities",
                    "risk_level": "Moderate risk, maintain balanced exposure"
                }
            
            # Override implications if divergence exists
            if breadth_divergence == "bearish":
                implications["divergence_warning"] = "Bearish divergence suggests potential trend reversal or correction"
            elif breadth_divergence == "bullish":
                implications["divergence_warning"] = "Bullish divergence suggests potential trend reversal or rally"
            
            return {
                "timestamp": recent_breadth.get("timestamp"),
                "metrics": {
                    "advance_decline_ratio": advance_decline_ratio,
                    "new_highs_lows_ratio": new_highs_lows_ratio,
                    "percent_above_ma50": percent_above_ma50,
                    "percent_above_ma200": percent_above_ma200,
                    "mcclellan_oscillator": mcclellan_oscillator
                },
                "trends": {
                    "advance_decline_trend": ad_trend,
                    "highs_lows_trend": hl_trend,
                    "ma50_trend": ma50_trend,
                    "ma200_trend": ma200_trend
                },
                "breadth_condition": breadth_condition,
                "breadth_divergence": breadth_divergence,
                "implications": implications
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market breadth: {e}")
            return None
    
    def _recommend_trading_approaches(self, short_term_regime: Dict[str, Any], 
                                   medium_term_regime: Dict[str, Any], 
                                   long_term_regime: Dict[str, Any],
                                   volatility_regime: Dict[str, Any],
                                   mean_reversion_momentum: Dict[str, Any],
                                   market_cycle: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recommend suitable trading approaches based on market regimes.
        
        Args:
            short_term_regime: Short-term regime data
            medium_term_regime: Medium-term regime data
            long_term_regime: Long-term regime data
            volatility_regime: Volatility regime data
            mean_reversion_momentum: Mean reversion vs momentum analysis
            market_cycle: Market cycle data
            
        Returns:
            Dictionary with trading approach recommendations
        """
        # Extract key regime information
        short_regime = short_term_regime.get("primary_regime", "undefined")
        medium_regime = medium_term_regime.get("primary_regime", "undefined")
        volatility = volatility_regime.get("primary_regime", "normal_volatility")
        cycle_stage = market_cycle.get("cycle_stage", "undefined")
        dominant_effect = mean_reversion_momentum.get("dominant_regime", "undefined")
        
        # Initialize recommendation components
        suitable_strategies = []
        position_sizing = {}
        timeframes = []
        risk_management = []
        
        # Determine suitable strategies based on regimes
        if short_regime in ["bull_trending", "early_bull"]:
            suitable_strategies.extend(["trend_following", "momentum", "breakout"])
            if medium_regime in ["bull_trending", "early_bull"]:
                suitable_strategies.append("position_trading")
            
        elif short_regime in ["bear_trending", "early_bear"]:
            suitable_strategies.extend(["trend_following_short", "breakdown", "short_selling"])
            if medium_regime in ["bear_trending", "early_bear"]:
                suitable_strategies.append("position_short")
            
        elif short_regime in ["bull_ranging", "bear_ranging"]:
            suitable_strategies.extend(["range_trading", "support_resistance", "mean_reversion"])
            
        elif short_regime == "high_volatility":
            suitable_strategies.extend(["volatility_breakout", "options_strategies", "reduced_exposure"])
            
        elif short_regime == "low_volatility":
            suitable_strategies.extend(["breakout_anticipation", "accumulation", "income_strategies"])
        
        # Adjust based on market cycle
        if cycle_stage == "accumulation":
            suitable_strategies.extend(["bottom_fishing", "value_investing"])
            risk_management.append("Gradual position building recommended")
            
        elif cycle_stage == "distribution":
            suitable_strategies.extend(["profit_taking", "protective_puts"])
            risk_management.append("Begin reducing exposure in strongest positions")
            
        elif cycle_stage in ["early_markdown", "markdown"]:
            suitable_strategies.extend(["short_selling", "defensive_positioning"])
            risk_management.append("Tighten stops and reduce overall exposure")
            
        # Adjust based on mean reversion vs momentum
        if dominant_effect in ["momentum", "weak_momentum"] and "mean_reversion" in suitable_strategies:
            suitable_strategies.remove("mean_reversion")
        elif dominant_effect in ["mean_reversion", "weak_mean_reversion"] and "trend_following" in suitable_strategies:
            suitable_strategies.remove("trend_following")
        
        # Determine position sizing recommendations
        if volatility == "high_volatility":
            position_sizing = {
                "recommendation": "reduce",
                "adjustment": "-30% to -50% from standard size",
                "rationale": "High volatility requires smaller positions to manage risk"
            }
        elif volatility == "low_volatility":
            position_sizing = {
                "recommendation": "increase",
                "adjustment": "+10% to +30% from standard size",
                "rationale": "Low volatility allows larger positions with similar risk"
            }
        else:
            position_sizing = {
                "recommendation": "standard",
                "adjustment": "Use normal position sizing",
                "rationale": "Normal volatility conditions"
            }
        
        # Recommend suitable timeframes
        if short_regime in ["bull_trending", "bear_trending"] and medium_regime in ["bull_trending", "bear_trending"]:
            timeframes.extend(["daily", "4-hour"])
            if dominant_effect in ["momentum", "weak_momentum"]:
                timeframes.append("weekly")
                
        elif short_regime in ["bull_ranging", "bear_ranging"]:
            timeframes.extend(["4-hour", "1-hour"])
            if dominant_effect in ["mean_reversion", "weak_mean_reversion"]:
                timeframes.append("15-minute")
                
        elif short_regime == "high_volatility":
            timeframes.extend(["1-hour", "15-minute"])
            if dominant_effect == "mean_reversion":
                timeframes.append("5-minute")
                
        else:  # Default recommendation
            timeframes.extend(["daily", "4-hour", "1-hour"])
        
        # Risk management recommendations
        if short_regime != medium_regime:
            risk_management.append("Timeframe conflict - use smaller positions and tighter stops")
            
        if volatility == "high_volatility":
            risk_management.append("Use wider stops (1.5-2x normal) to accommodate increased volatility")
            
        if cycle_stage in ["late_markup", "distribution"]:
            risk_management.append("Consider partial profit taking and trailing stops")
            
        if short_regime == "high_volatility" and "high_volatility" not in medium_regime:
            risk_management.append("Short-term volatility spike - consider waiting for stabilization")
        
        # Combine recommendations
        return {
            "suitable_strategies": list(set(suitable_strategies)),  # Remove duplicates
            "position_sizing": position_sizing,
            "recommended_timeframes": timeframes,
            "risk_management": risk_management
        }
    
    def _generate_regime_summary(self, short_term_regime: Dict[str, Any], 
                              medium_term_regime: Dict[str, Any], 
                              long_term_regime: Dict[str, Any],
                              volatility_regime: Dict[str, Any],
                              market_cycle: Dict[str, Any],
                              regime_transitions: Dict[str, Any]) -> str:
        """
        Generate a concise summary of the market regime analysis.
        
        Args:
            short_term_regime: Short-term regime data
            medium_term_regime: Medium-term regime data
            long_term_regime: Long-term regime data
            volatility_regime: Volatility regime data
            market_cycle: Market cycle data
            regime_transitions: Regime transition data
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Short-term regime
        short_regime = short_term_regime.get("primary_regime", "undefined")
        regime_desc = self.regime_characteristics.get(short_regime, {}).get("description", "")
        
        if regime_desc:
            summary_parts.append(f"Short-term market regime: {short_regime.replace('_', ' ').title()} - {regime_desc}.")
        else:
            summary_parts.append(f"Short-term market regime: {short_regime.replace('_', ' ').title()}.")
        
        # Medium-term regime if different
        medium_regime = medium_term_regime.get("primary_regime", "undefined")
        if medium_regime != short_regime:
            summary_parts.append(f"Medium-term regime: {medium_regime.replace('_', ' ').title()}, indicating potential transition.")
        else:
            summary_parts.append(f"Medium-term regime confirms the short-term view.")
        
        # Volatility conditions
        vol_regime = volatility_regime.get("primary_regime", "normal_volatility")
        vol_transition = volatility_regime.get("volatility_characteristics", {}).get("transition", "stable")
        
        if vol_regime == "high_volatility":
            summary_parts.append(f"Market is experiencing high volatility that is {vol_transition}.")
        elif vol_regime == "low_volatility":
            summary_parts.append(f"Market is showing unusually low volatility that is {vol_transition}.")
        
        # Market cycle
        cycle_stage = market_cycle.get("cycle_stage", "undefined")
        if cycle_stage != "undefined":
            cycle_position = market_cycle.get("cycle_position_percent", 50)
            summary_parts.append(f"Currently in the {cycle_stage.replace('_', ' ').title()} phase ({cycle_position:.1f}% through the market cycle).")
        
        # Recent or impending transitions
        recent_transition = regime_transitions.get("recent_transition", "none")
        impending_transition = regime_transitions.get("impending_transition", "none")
        
        if recent_transition != "none" and recent_transition != "insufficient_data":
            summary_parts.append(f"Recently transitioned from {recent_transition.split('_to_')[0].replace('_', ' ')} to {recent_transition.split('_to_')[1].replace('_', ' ')}.")
        
        if impending_transition != "none" and impending_transition != "insufficient_data":
            summary_parts.append(f"Showing signs of potential transition to {impending_transition.split('_')[-1].replace('_', ' ')}.")
        
        # Trading approach summary
        if short_regime in ["bull_trending", "early_bull"]:
            summary_parts.append("Trending markets favor momentum strategies and trend following approaches.")
        elif short_regime in ["bear_trending", "early_bear"]:
            summary_parts.append("Bearish trend suggests defensive positioning and potential short opportunities.")
        elif short_regime in ["bull_ranging", "bear_ranging"]:
            summary_parts.append("Ranging market favors mean reversion strategies and range-bound trading approaches.")
        elif short_regime == "high_volatility":
            summary_parts.append("High volatility requires careful position sizing and volatility-based strategies.")
        elif short_regime == "low_volatility":
            summary_parts.append("Low volatility suggests preparing for eventual breakout and focusing on accumulation.")
        
        return " ".join(summary_parts)
    
    def _save_analysis(self, symbol: str, exchange: str, timeframe: str, result: Dict[str, Any]) -> bool:
        """
        Save market regime analysis results to database.
        
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
                "regimes": result.get("regimes"),
                "market_cycle": result.get("market_cycle"),
                "volatility_regime": result.get("volatility_regime"),
                "mean_reversion_momentum": result.get("mean_reversion_momentum"),
                "regime_transitions": result.get("regime_transitions"),
                "market_breadth": result.get("market_breadth"),
                "trading_approaches": result.get("trading_approaches"),
                "regime_summary": result.get("regime_summary")
            }
            
            # Insert into database
            self.db.market_regime_collection.insert_one(document)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving market regime analysis: {e}")
            return False
    
    def get_multi_asset_correlation(self, symbols: List[str], exchange: str = "NSE", 
                                  timeframe: str = "day", days: int = 60) -> Dict[str, Any]:
        """
        Calculate correlations between multiple assets.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            self.logger.info(f"Analyzing correlations between {len(symbols)} symbols")
            
            # Get price data for all symbols
            price_data = {}
            for symbol in symbols:
                data = self._get_market_data(symbol, exchange, timeframe, days)
                if not data:
                    continue
                
                df = pd.DataFrame(data)
                if "close" not in df.columns or "timestamp" not in df.columns:
                    continue
                
                # Extract closing prices with timestamps
                symbol_data = df[["timestamp", "close"]].sort_values("timestamp")
                price_data[symbol] = symbol_data
            
            if len(price_data) < 2:
                return {
                    "status": "error",
                    "error": "Insufficient data for correlation analysis"
                }
            
            # Create a combined dataframe with all closing prices
            combined_df = None
            
            for symbol, data in price_data.items():
                if combined_df is None:
                    combined_df = data.rename(columns={"close": symbol}).set_index("timestamp")
                else:
                    combined_df[symbol] = data.set_index("timestamp")["close"]
            
            # Fill missing values
            combined_df = combined_df.fillna(method="ffill").fillna(method="bfill")
            
            # Calculate returns
            returns_df = combined_df.pct_change().dropna()
            
            # Calculate correlation matrix
            correlation_matrix = returns_df.corr()
            
            # Find highly correlated and inversely correlated pairs
            high_correlation_pairs = []
            inverse_correlation_pairs = []
            
            symbols_list = list(correlation_matrix.columns)
            for i in range(len(symbols_list)):
                for j in range(i+1, len(symbols_list)):
                    symbol1 = symbols_list[i]
                    symbol2 = symbols_list[j]
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    
                    if corr >= self.analysis_params["high_correlation"]:
                        high_correlation_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": corr
                        })
                    elif corr <= self.analysis_params["inverse_correlation"]:
                        inverse_correlation_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "correlation": corr
                        })
            
            # Calculate average correlations for each symbol
            avg_correlations = {}
            for symbol in symbols_list:
                correlations = [correlation_matrix.loc[symbol, other] for other in symbols_list if other != symbol]
                avg_correlations[symbol] = sum(correlations) / len(correlations) if correlations else 0
            
            # Find potential diversifiers (low average correlation)
            potential_diversifiers = [
                {
                    "symbol": symbol,
                    "avg_correlation": corr
                }
                for symbol, corr in avg_correlations.items()
                if corr < 0.3
            ]
            
            # Sort by correlation
            potential_diversifiers.sort(key=lambda x: x["avg_correlation"])
            high_correlation_pairs.sort(key=lambda x: x["correlation"], reverse=True)
            inverse_correlation_pairs.sort(key=lambda x: x["correlation"])
            
            # Format correlation matrix for output
            formatted_matrix = []
            for symbol1 in symbols_list:
                row = {"symbol": symbol1}
                for symbol2 in symbols_list:
                    row[symbol2] = round(correlation_matrix.loc[symbol1, symbol2], 2)
                formatted_matrix.append(row)
            
            # Generate trading implications
            trading_implications = []
            
            if high_correlation_pairs:
                trading_implications.append(
                    f"Highly correlated pairs (e.g., {high_correlation_pairs[0]['symbol1']}/{high_correlation_pairs[0]['symbol2']}) "
                    "may present pair trading opportunities for mean reversion."
                )
            
            if inverse_correlation_pairs:
                trading_implications.append(
                    f"Inversely correlated pairs (e.g., {inverse_correlation_pairs[0]['symbol1']}/{inverse_correlation_pairs[0]['symbol2']}) "
                    "can be used for portfolio hedging and risk reduction."
                )
            
            if potential_diversifiers:
                trading_implications.append(
                    f"Consider {potential_diversifiers[0]['symbol']} for diversification as it has low correlation with the group."
                )
            
            avg_correlation = sum(sum(correlation_matrix.values)) / (len(symbols_list) ** 2)
            if avg_correlation > 0.7:
                trading_implications.append(
                    "Overall high correlation suggests market-wide factors dominating. Consider sector rotation or alternative assets."
                )
            
            return {
                "status": "success",
                "correlation_matrix": formatted_matrix,
                "high_correlation_pairs": high_correlation_pairs[:5],  # Top 5
                "inverse_correlation_pairs": inverse_correlation_pairs[:5],  # Top 5
                "potential_diversifiers": potential_diversifiers[:3],  # Top 3
                "average_correlation": avg_correlation,
                "trading_implications": trading_implications
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {
                "status": "error",
                "error": str(e)
            }