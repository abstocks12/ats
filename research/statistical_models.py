"""
Statistical Models for Market Analysis

This module provides statistical models for market prediction and analysis.
It includes mean reversion models, pair trading strategies, and statistical arbitrage techniques.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
import statsmodels.api as sm

class StatisticalModels:
    """
    Provides statistical models for market prediction and trading.
    Implements mean reversion, pair trading, and statistical arbitrage strategies.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the statistical models with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get query optimizer if available
        self.query_optimizer = getattr(self.db, 'get_query_optimizer', lambda: None)()
        
        # Define model parameters
        self.model_params = {
            # Mean reversion parameters
            "zscore_entry_threshold": 2.0,       # Z-score threshold for mean reversion entry
            "zscore_exit_threshold": 0.5,        # Z-score threshold for mean reversion exit
            "mean_reversion_window": 20,         # Rolling window for mean and std dev
            "mean_reversion_stop_loss": 2.5,     # Stop loss z-score multiple
            
            # Pair trading parameters
            "pair_correlation_threshold": 0.7,   # Minimum correlation for pair consideration
            "pair_cointegration_pvalue": 0.05,   # Maximum p-value for cointegration test
            "pair_entry_threshold": 2.0,         # Standard deviation entry threshold
            "pair_exit_threshold": 0.5,          # Standard deviation exit threshold
            
            # Time series forecast parameters
            "arima_max_order": (5, 1, 5),        # Maximum ARIMA model order to consider
            "forecast_horizon": 5,               # Number of periods to forecast
            
            # Volatility modeling parameters
            "garch_p": 1,                        # GARCH p parameter
            "garch_q": 1,                        # GARCH q parameter
            
            # Regime detection parameters
            "hmm_n_regimes": 3,                  # Number of regimes for HMM
            "hmm_n_iter": 100                    # Number of iterations for HMM training
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
    
    def analyze_mean_reversion(self, symbol: str, exchange: str = "NSE", 
                             timeframe: str = "day", days: int = 100) -> Dict[str, Any]:
        """
        Analyze mean reversion characteristics and signals for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with mean reversion analysis
        """
        try:
            self.logger.info(f"Analyzing mean reversion for {symbol} ({exchange}) on {timeframe} timeframe")
            
            # Get market data
            data = self._get_market_data(symbol, exchange, timeframe, days)
            
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
            if not all(col in df.columns for col in ["timestamp", "close"]):
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Incomplete data: missing required columns"
                }
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate mean reversion metrics
            window = self.model_params["mean_reversion_window"]
            
            # Calculate rolling mean and standard deviation
            df['rolling_mean'] = df['close'].rolling(window=window).mean()
            df['rolling_std'] = df['close'].rolling(window=window).std()
            
            # Calculate z-score (deviation from mean in standard deviation units)
            df['zscore'] = (df['close'] - df['rolling_mean']) / df['rolling_std']
            
            # Calculate rate of mean reversion (how quickly z-scores return to mean)
            df['zscore_change'] = df['zscore'].diff()
            df['zscore_lag'] = df['zscore'].shift(1)
            
            # Mean reversion coefficient (negative indicates mean reversion)
            # This is essentially the coefficient from regressing z-score change on z-score level
            valid_data = df.dropna()
            if len(valid_data) >= 30:  # Need sufficient data for regression
                X = sm.add_constant(valid_data['zscore_lag'])
                model = sm.OLS(valid_data['zscore_change'], X).fit()
                mean_reversion_coefficient = model.params[1]
                half_life = -math.log(2) / mean_reversion_coefficient if mean_reversion_coefficient < 0 else None
                regression_r2 = model.rsquared
            else:
                mean_reversion_coefficient = None
                half_life = None
                regression_r2 = None
            
            # Test for stationarity (required for mean reversion)
            adf_result = adfuller(df['close'].dropna())
            is_stationary = adf_result[1] < 0.05  # p-value < 0.05
            
            # Find potential mean reversion signals
            entry_signals = []
            exit_signals = []
            
            # Entry signals (extreme z-scores)
            high_zscore = df[df['zscore'] > self.model_params["zscore_entry_threshold"]].copy()
            low_zscore = df[df['zscore'] < -self.model_params["zscore_entry_threshold"]].copy()
            
            # Convert timestamps to string if they're datetime objects
            if isinstance(high_zscore['timestamp'].iloc[0], pd.Timestamp):
                high_zscore['timestamp'] = high_zscore['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if isinstance(low_zscore['timestamp'].iloc[0], pd.Timestamp):
                low_zscore['timestamp'] = low_zscore['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add signals
            for _, row in high_zscore.iterrows():
                entry_signals.append({
                    "timestamp": row['timestamp'],
                    "price": row['close'],
                    "zscore": row['zscore'],
                    "signal_type": "short",  # Short when price is above mean
                    "target_price": row['rolling_mean'],
                    "stop_loss": row['close'] * (1 + (row['zscore'] / self.model_params["zscore_entry_threshold"]) * 0.03)
                })
            
            for _, row in low_zscore.iterrows():
                entry_signals.append({
                    "timestamp": row['timestamp'],
                    "price": row['close'],
                    "zscore": row['zscore'],
                    "signal_type": "long",  # Long when price is below mean
                    "target_price": row['rolling_mean'],
                    "stop_loss": row['close'] * (1 - (abs(row['zscore']) / self.model_params["zscore_entry_threshold"]) * 0.03)
                })
            
            # Exit signals (z-scores returning to mean)
            df['prev_zscore'] = df['zscore'].shift(1)
            
            # Find crossovers of the exit threshold
            exits_from_high = df[(df['prev_zscore'] > self.model_params["zscore_exit_threshold"]) & 
                                (df['zscore'] <= self.model_params["zscore_exit_threshold"])].copy()
            
            exits_from_low = df[(df['prev_zscore'] < -self.model_params["zscore_exit_threshold"]) & 
                                (df['zscore'] >= -self.model_params["zscore_exit_threshold"])].copy()
            
            # Convert timestamps to string if they're datetime objects
            if len(exits_from_high) > 0 and isinstance(exits_from_high['timestamp'].iloc[0], pd.Timestamp):
                exits_from_high['timestamp'] = exits_from_high['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if len(exits_from_low) > 0 and isinstance(exits_from_low['timestamp'].iloc[0], pd.Timestamp):
                exits_from_low['timestamp'] = exits_from_low['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add signals
            for _, row in exits_from_high.iterrows():
                exit_signals.append({
                    "timestamp": row['timestamp'],
                    "price": row['close'],
                    "zscore": row['zscore'],
                    "signal_type": "exit_short",
                    "profit_potential": (row['prev_zscore'] - row['zscore']) / row['prev_zscore'] * 100 if row['prev_zscore'] != 0 else 0
                })
            
            for _, row in exits_from_low.iterrows():
                exit_signals.append({
                    "timestamp": row['timestamp'],
                    "price": row['close'],
                    "zscore": row['zscore'],
                    "signal_type": "exit_long",
                    "profit_potential": (row['zscore'] - row['prev_zscore']) / abs(row['prev_zscore']) * 100 if row['prev_zscore'] != 0 else 0
                })
            
            # Calculate mean reversion strength score
            mr_score = 0.0
            
            # Strong mean reversion has negative coefficient
            if mean_reversion_coefficient is not None:
                if mean_reversion_coefficient < -0.5:
                    mr_score += 40  # Very strong mean reversion
                elif mean_reversion_coefficient < -0.3:
                    mr_score += 30  # Strong mean reversion
                elif mean_reversion_coefficient < -0.1:
                    mr_score += 20  # Moderate mean reversion
                elif mean_reversion_coefficient < 0:
                    mr_score += 10  # Weak mean reversion
            
            # Stationarity is important for mean reversion
            if is_stationary:
                mr_score += 30
            
            # R-squared indicates how reliable the mean reversion is
            if regression_r2 is not None:
                mr_score += regression_r2 * 20
            
            # Reasonable half-life is good for mean reversion trading
            if half_life is not None and 1 <= half_life <= 10:
                mr_score += 10
            
            # Normalize to 0-100 scale
            mr_score = min(100, max(0, mr_score))
            
            # Determine if the asset is suitable for mean reversion strategies
            suitability = "poor"
            if mr_score >= 80:
                suitability = "excellent"
            elif mr_score >= 60:
                suitability = "good"
            elif mr_score >= 40:
                suitability = "moderate"
            
            # Format timestamp as string for proper JSON serialization
            if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate summary
            if suitability == "excellent":
                summary = f"{symbol} shows strong mean reversion characteristics with a half-life of approximately {half_life:.1f} periods. The asset is highly suitable for mean reversion trading strategies."
            elif suitability == "good":
                summary = f"{symbol} exhibits good mean reversion tendencies. Consider mean reversion strategies with appropriate risk management."
            elif suitability == "moderate":
                summary = f"{symbol} shows moderate mean reversion characteristics. It may be suitable for mean reversion trading in certain market conditions."
            else:
                summary = f"{symbol} does not show strong mean reversion characteristics. Other trading approaches may be more suitable."
            
            # Add recent z-score information
            recent_zscore = df['zscore'].iloc[-1] if not df.empty else None
            if recent_zscore is not None:
                if abs(recent_zscore) > self.model_params["zscore_entry_threshold"]:
                    summary += f" Current z-score of {recent_zscore:.2f} indicates a potential mean reversion opportunity."
                elif abs(recent_zscore) < self.model_params["zscore_exit_threshold"]:
                    summary += f" Current z-score of {recent_zscore:.2f} suggests the price is close to its mean."
                else:
                    summary += f" Current z-score is {recent_zscore:.2f}."
            
            # Assemble the analysis result
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "status": "success",
                "mean_reversion_metrics": {
                    "coefficient": mean_reversion_coefficient,
                    "half_life": half_life,
                    "r_squared": regression_r2,
                    "is_stationary": is_stationary,
                    "adf_pvalue": adf_result[1]
                },
                "mean_reversion_score": mr_score,
                "suitability": suitability,
                "recent_zscore": recent_zscore,
                "entry_signals": entry_signals[-5:] if entry_signals else [],  # Most recent 5 signals
                "exit_signals": exit_signals[-5:] if exit_signals else [],    # Most recent 5 signals
                "summary": summary,
                "recent_data": df.iloc[-20:][['timestamp', 'close', 'rolling_mean', 'zscore']].to_dict('records')
            }
            
            # Save analysis result to database
            self._save_mean_reversion_analysis(symbol, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing mean reversion for {symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e)
            }
    
    def analyze_pair_trading(self, symbol1: str, symbol2: str, exchange: str = "NSE", 
                           timeframe: str = "day", days: int = 100) -> Dict[str, Any]:
        """
        Analyze pair trading characteristics and signals.
        
        Args:
            symbol1: First stock symbol
            symbol2: Second stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with pair trading analysis
        """
        try:
            self.logger.info(f"Analyzing pair trading for {symbol1}-{symbol2} ({exchange}) on {timeframe} timeframe")
            
            # Get market data for both symbols
            data1 = self._get_market_data(symbol1, exchange, timeframe, days)
            data2 = self._get_market_data(symbol2, exchange, timeframe, days)
            
            if not data1 or not data2:
                return {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Data not found for one or both symbols"
                }
            
            # Convert to DataFrames
            df1 = pd.DataFrame(data1)
            df2 = pd.DataFrame(data2)
            
            # Make sure required columns are present
            required_columns = ["timestamp", "close"]
            if not all(col in df1.columns for col in required_columns) or not all(col in df2.columns for col in required_columns):
                return {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Incomplete data: missing required columns"
                }
            
            # Sort by timestamp
            df1 = df1.sort_values("timestamp")
            df2 = df2.sort_values("timestamp")
            
            # Align dates between the two DataFrames
            df1.set_index('timestamp', inplace=True)
            df2.set_index('timestamp', inplace=True)
            
            # Merge on matching dates
            merged = pd.merge(df1['close'], df2['close'], left_index=True, right_index=True, suffixes=('_1', '_2'))
            
            if len(merged) < 30:
                return {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Insufficient matching data points for pair analysis"
                }
            
            # Calculate correlation
            correlation = merged['close_1'].corr(merged['close_2'])
            
            # Test for cointegration
            coint_result = coint(merged['close_1'], merged['close_2'])
            coint_pvalue = coint_result[1]
            is_cointegrated = coint_pvalue < self.model_params["pair_cointegration_pvalue"]
            
            # If correlation is not strong enough or not cointegrated, pair trading may not be suitable
            if correlation < self.model_params["pair_correlation_threshold"] or not is_cointegrated:
                pair_score = correlation * 50 + (1 - coint_pvalue) * 50
                pair_score = min(100, max(0, pair_score))
                
                suitability = "poor"
                if pair_score >= 80:
                    suitability = "good"
                elif pair_score >= 60:
                    suitability = "moderate"
                
                return {
                    "symbol1": symbol1,
                    "symbol2": symbol2,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "success",
                    "correlation": correlation,
                    "cointegration_pvalue": coint_pvalue,
                    "is_cointegrated": is_cointegrated,
                    "pair_score": pair_score,
                    "suitability": suitability,
                    "summary": f"The pair {symbol1}-{symbol2} has correlation of {correlation:.2f} and cointegration p-value of {coint_pvalue:.4f}. It is {'suitable' if suitability != 'poor' else 'not well suited'} for pair trading."
                }
            
            # Calculate the spread
            # First, run OLS regression to find hedge ratio
            X = sm.add_constant(merged['close_2'])
            model = sm.OLS(merged['close_1'], X).fit()
            hedge_ratio = model.params[1]
            alpha = model.params[0]
            
            # Calculate the spread
            merged['spread'] = merged['close_1'] - (hedge_ratio * merged['close_2'] + alpha)
            
            # Calculate z-score of the spread
            window = self.model_params["mean_reversion_window"]
            merged['spread_mean'] = merged['spread'].rolling(window=window).mean()
            merged['spread_std'] = merged['spread'].rolling(window=window).std()
            merged['zscore'] = (merged['spread'] - merged['spread_mean']) / merged['spread_std']
            
            # Test spread for stationarity
            spread_adf_result = adfuller(merged['spread'].dropna())
            spread_is_stationary = spread_adf_result[1] < 0.05
            
            # Find potential pair trading signals
            entry_signals = []
            exit_signals = []
            
            # Entry signals (extreme z-scores)
            high_zscore = merged[merged['zscore'] > self.model_params["pair_entry_threshold"]].copy()
            low_zscore = merged[merged['zscore'] < -self.model_params["pair_entry_threshold"]].copy()
            
            # Reset index to get timestamp as a column
            high_zscore = high_zscore.reset_index()
            low_zscore = low_zscore.reset_index()
            
            # Convert timestamps to string if they're datetime objects
            if len(high_zscore) > 0 and isinstance(high_zscore['timestamp'].iloc[0], pd.Timestamp):
                high_zscore['timestamp'] = high_zscore['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if len(low_zscore) > 0 and isinstance(low_zscore['timestamp'].iloc[0], pd.Timestamp):
                low_zscore['timestamp'] = low_zscore['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add signals
            for _, row in high_zscore.iterrows():
                entry_signals.append({
                    "timestamp": row['timestamp'],
                    "signal_type": "pair_divergence",
                    "zscore": row['zscore'],
                    "action": f"Short {symbol1}, Long {symbol2}",
                    "ratio": f"1:{hedge_ratio:.2f}",
                    "expected_convergence": f"Spread expected to decrease by {abs(row['zscore'] - self.model_params['pair_exit_threshold']):.2f} standard deviations"
                })
            
            for _, row in low_zscore.iterrows():
                entry_signals.append({
                    "timestamp": row['timestamp'],
                    "signal_type": "pair_divergence",
                    "zscore": row['zscore'],
                    "action": f"Long {symbol1}, Short {symbol2}",
                    "ratio": f"1:{hedge_ratio:.2f}",
                    "expected_convergence": f"Spread expected to increase by {abs(row['zscore'] - (-self.model_params['pair_exit_threshold'])):.2f} standard deviations"
                })
            
            # Exit signals (z-scores returning to mean)
            merged['prev_zscore'] = merged['zscore'].shift(1)
            
            # Find crossovers of the exit threshold
            exits_from_high = merged[(merged['prev_zscore'] > self.model_params["pair_exit_threshold"]) & 
                                    (merged['zscore'] <= self.model_params["pair_exit_threshold"])].copy()
            
            exits_from_low = merged[(merged['prev_zscore'] < -self.model_params["pair_exit_threshold"]) & 
                                   (merged['zscore'] >= -self.model_params["pair_exit_threshold"])].copy()
            
            # Reset index to get timestamp as a column
            exits_from_high = exits_from_high.reset_index()
            exits_from_low = exits_from_low.reset_index()
            
            # Convert timestamps to string if they're datetime objects
            if len(exits_from_high) > 0 and isinstance(exits_from_high['timestamp'].iloc[0], pd.Timestamp):
                exits_from_high['timestamp'] = exits_from_high['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            if len(exits_from_low) > 0 and isinstance(exits_from_low['timestamp'].iloc[0], pd.Timestamp):
                exits_from_low['timestamp'] = exits_from_low['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Add signals
            for _, row in exits_from_high.iterrows():
                exit_signals.append({
                    "timestamp": row['timestamp'],
                    "signal_type": "pair_convergence",
                    "zscore": row['zscore'],
                    "action": f"Exit: Cover {symbol1}, Sell {symbol2}",
                    "profit_potential": (row['prev_zscore'] - row['zscore']) / row['prev_zscore'] * 100 if row['prev_zscore'] != 0 else 0
                })
            
            for _, row in exits_from_low.iterrows():
                exit_signals.append({
                    "timestamp": row['timestamp'],
                    "signal_type": "pair_convergence",
                    "zscore": row['zscore'],
                    "action": f"Exit: Sell {symbol1}, Cover {symbol2}",
                    "profit_potential": (row['zscore'] - row['prev_zscore']) / abs(row['prev_zscore']) * 100 if row['prev_zscore'] != 0 else 0
                })
            
            # Calculate success metrics for past trades
            if len(entry_signals) > 0 and len(exit_signals) > 0:
                # Simplistic backtesting - match entries with subsequent exits
                # This is a simplified approach - real backtesting would be more complex
                completed_trades = 0
                successful_trades = 0
                
                for entry in entry_signals:
                    entry_time = entry["timestamp"]
                    entry_zscore = entry["zscore"]
                    
                    # Find next exit after this entry
                    next_exits = [ex for ex in exit_signals if ex["timestamp"] > entry_time]
                    
                    if next_exits:
                        completed_trades += 1
                        next_exit = next_exits[0]
                        
                        # Check if the exit represents a move toward the mean
                        if (entry_zscore > 0 and next_exit["zscore"] < entry_zscore) or \
                           (entry_zscore < 0 and next_exit["zscore"] > entry_zscore):
                            successful_trades += 1
                
                success_rate = successful_trades / completed_trades if completed_trades > 0 else None
            else:
                success_rate = None
            
            # Calculate pair trading strength score
            pair_score = 0.0
            
            # Strong correlation is important
            pair_score += correlation * 30
            
            # Cointegration is crucial
            pair_score += (1 - coint_pvalue) * 30
            
            # Spread stationarity is important
            if spread_is_stationary:
                pair_score += 20
            
            # Success rate of past trades
            if success_rate:
                pair_score += success_rate * 20
            
            # Normalize to 0-100 scale
            pair_score = min(100, max(0, pair_score))
            
            # Determine if the pair is suitable for pair trading
            suitability = "poor"
            if pair_score >= 80:
                suitability = "excellent"
            elif pair_score >= 60:
                suitability = "good"
            elif pair_score >= 40:
                suitability = "moderate"
            
            # Generate summary
            if suitability == "excellent":
                summary = f"The pair {symbol1}-{symbol2} shows excellent characteristics for pair trading with a correlation of {correlation:.2f} and strong cointegration (p-value: {coint_pvalue:.4f})."
            elif suitability == "good":
                summary = f"The pair {symbol1}-{symbol2} exhibits good pair trading potential with a correlation of {correlation:.2f} and cointegration (p-value: {coint_pvalue:.4f})."
            elif suitability == "moderate":
                summary = f"The pair {symbol1}-{symbol2} shows moderate potential for pair trading. Consider additional filters or alternative pairs."
            else:
                summary = f"The pair {symbol1}-{symbol2} is not well suited for pair trading despite meeting minimum criteria."
            
            # Add recent z-score information
            recent_zscore = merged['zscore'].iloc[-1] if not merged.empty else None
            if recent_zscore is not None:
                if abs(recent_zscore) > self.model_params["pair_entry_threshold"]:
                    summary += f" Current z-score of {recent_zscore:.2f} indicates a potential pair trading opportunity."
                elif abs(recent_zscore) < self.model_params["pair_exit_threshold"]:
                    summary += f" Current z-score of {recent_zscore:.2f} suggests the spread is close to its mean."
                else:
                    summary += f" Current z-score is {recent_zscore:.2f}."
            
            # Reset index to get timestamp as a column for recent data
            merged_reset = merged.reset_index()
            
            # Convert timestamps to string for recent data
            if isinstance(merged_reset['timestamp'].iloc[0], pd.Timestamp):
                merged_reset['timestamp'] = merged_reset['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Assemble the analysis result
            result = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "status": "success",
                "pair_metrics": {
                    "correlation": correlation,
                    "hedge_ratio": hedge_ratio,
                    "alpha": alpha,
                    "cointegration_pvalue": coint_pvalue,
                    "is_cointegrated": is_cointegrated,
                    "spread_stationarity_pvalue": spread_adf_result[1],
                    "spread_is_stationary": spread_is_stationary,
                    "success_rate": success_rate
                },
                "pair_score": pair_score,
                "suitability": suitability,
                "recent_zscore": recent_zscore,
                "entry_signals": entry_signals[-5:] if entry_signals else [],  # Most recent 5 signals
                "exit_signals": exit_signals[-5:] if exit_signals else [],    # Most recent 5 signals
                "summary": summary,
                "recent_data": merged_reset.iloc[-20:][['timestamp', 'close_1', 'close_2', 'spread', 'zscore']].to_dict('records')
            }
            
            # Save analysis result to database
            self._save_pair_trading_analysis(symbol1, symbol2, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing pair trading for {symbol1}-{symbol2}: {e}")
            return {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e)
            }
    
    def analyze_time_series_forecast(self, symbol: str, exchange: str = "NSE", 
                                   timeframe: str = "day", days: int = 100) -> Dict[str, Any]:
        """
        Perform time series forecasting for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with time series forecast analysis
        """
        try:
            self.logger.info(f"Forecasting time series for {symbol} ({exchange}) on {timeframe} timeframe")
            
            # Get market data
            data = self._get_market_data(symbol, exchange, timeframe, days)
            
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
            if "close" not in df.columns:
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Incomplete data: missing close prices"
                }
            
            # Sort by timestamp
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp")
            
            # Get closing prices
            prices = df['close'].values
            
            # Calculate returns
            returns = np.diff(np.log(prices))
            
            # Try to fit ARIMA model
            try:
                import pmdarima as pm
                
                # Use auto_arima to find the best model
                model = pm.auto_arima(
                    returns,
                    start_p=0, start_q=0,
                    max_p=self.model_params["arima_max_order"][0],
                    max_d=self.model_params["arima_max_order"][1],
                    max_q=self.model_params["arima_max_order"][2],
                    seasonal=False,
                    trace=False,
                    error_action='ignore',
                    suppress_warnings=True,
                    stepwise=True
                )
                
                # Get model order
                order = model.order
                
                # Forecast returns
                forecast, conf_int = model.predict(
                    n_periods=self.model_params["forecast_horizon"],
                    return_conf_int=True
                )
                
                # Convert returns forecast to price forecast
                last_price = prices[-1]
                price_forecast = [last_price]
                
                for ret in forecast:
                    price_forecast.append(price_forecast[-1] * np.exp(ret))
                
                price_forecast = price_forecast[1:]  # Remove the initial price
                
                # Calculate confidence intervals for prices
                lower_bound = [last_price]
                upper_bound = [last_price]
                
                for i, ret in enumerate(forecast):
                    lower_bound.append(lower_bound[-1] * np.exp(conf_int[i, 0]))
                    upper_bound.append(upper_bound[-1] * np.exp(conf_int[i, 1]))
                
                lower_bound = lower_bound[1:]  # Remove the initial price
                upper_bound = upper_bound[1:]  # Remove the initial price
                
                # Get model metrics
                aic = model.aic()
                bic = model.bic()
                
                # Calculate model accuracy using backtest
                if len(returns) > 30:
                    train_size = len(returns) - 10
                    train_returns = returns[:train_size]
                    test_returns = returns[train_size:]
                    
                    backtest_model = pm.ARIMA(order=order)
                    backtest_model.fit(train_returns)
                    
                    predictions = backtest_model.predict(n_periods=len(test_returns))
                    
                    mse = np.mean((predictions - test_returns) ** 2)
                    mae = np.mean(np.abs(predictions - test_returns))
                    accuracy = 1 - mae / np.mean(np.abs(test_returns)) if np.mean(np.abs(test_returns)) > 0 else 0
                else:
                    mse = None
                    mae = None
                    accuracy = None
                
                # Determine forecast reliability
                reliability = "low"
                reliability_score = 0.0
                
                # Higher accuracy is better
                if accuracy is not None:
                    reliability_score += accuracy * 50
                    if accuracy > 0.7:
                        reliability = "high"
                    elif accuracy > 0.5:
                        reliability = "moderate"
                
                # Lower AIC/BIC is better
                if aic is not None and bic is not None:
                    # Normalize AIC and BIC to a 0-1 scale (lower is better)
                    # This is a very rough approximation
                    aic_normalized = max(0, min(1, 1 - (aic / 1000)))
                    bic_normalized = max(0, min(1, 1 - (bic / 1000)))
                    
                    reliability_score += (aic_normalized + bic_normalized) * 25
                
                # Tighter confidence intervals suggest more reliable forecasts
                avg_spread = np.mean((np.array(upper_bound) - np.array(lower_bound)) / last_price)
                if avg_spread < 0.05:
                    reliability_score += 25
                elif avg_spread < 0.1:
                    reliability_score += 15
                elif avg_spread < 0.2:
                    reliability_score += 5
                
                # Normalize to 0-100 scale
                reliability_score = min(100, max(0, reliability_score))
                
                if reliability_score >= 80:
                    reliability = "high"
                elif reliability_score >= 50:
                    reliability = "moderate"
                
                # Generate forecast summary
                forecast_direction = "upward" if price_forecast[-1] > last_price else "downward"
                forecast_change = (price_forecast[-1] / last_price - 1) * 100
                
                forecast_summary = f"The {order} ARIMA model forecasts a {forecast_direction} move of {abs(forecast_change):.2f}% over the next {self.model_params['forecast_horizon']} periods."
                
                if reliability == "high":
                    forecast_summary += " This forecast has high reliability based on model metrics and historical accuracy."
                elif reliability == "moderate":
                    forecast_summary += " This forecast has moderate reliability. Consider additional confirmation signals."
                else:
                    forecast_summary += " This forecast has low reliability. Use with caution and seek additional confirmation."
                
                # Trading implications
                if forecast_direction == "upward" and reliability in ["high", "moderate"]:
                    trading_implication = f"Consider bullish positions with a {self.model_params['forecast_horizon']}-period horizon. Target: {price_forecast[-1]:.2f}"
                elif forecast_direction == "downward" and reliability in ["high", "moderate"]:
                    trading_implication = f"Consider bearish positions with a {self.model_params['forecast_horizon']}-period horizon. Target: {price_forecast[-1]:.2f}"
                else:
                    trading_implication = "Insufficient forecast reliability. Consider alternative analysis methods."
                
                # Format the forecast data
                forecast_data = []
                for i in range(self.model_params["forecast_horizon"]):
                    forecast_data.append({
                        "period": i + 1,
                        "price": price_forecast[i],
                        "lower_bound": lower_bound[i],
                        "upper_bound": upper_bound[i]
                    })
                
                # Format timestamp as string for proper JSON serialization
                if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                    df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # Assemble the forecast result
                # Assemble the forecast result
                result = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "timestamp": datetime.now(),
                    "status": "success",
                    "model_details": {
                        "type": "ARIMA",
                        "order": order,
                        "aic": aic,
                        "bic": bic,
                        "mse": mse,
                        "mae": mae,
                        "accuracy": accuracy
                    },
                    "forecast": {
                        "horizon": self.model_params["forecast_horizon"],
                        "direction": forecast_direction,
                        "percent_change": forecast_change,
                        "reliability": reliability,
                        "reliability_score": reliability_score,
                        "data": forecast_data
                    },
                    "summary": forecast_summary,
                    "trading_implication": trading_implication,
                    "recent_data": df.iloc[-20:][['timestamp', 'close']].to_dict('records') if 'timestamp' in df.columns else df.iloc[-20:][['close']].to_dict('records')
                }
                
            except ImportError:
                # Fall back to simple moving average forecasting if pmdarima is not available
                result = self._simple_forecast(df, symbol, exchange, timeframe)
                
            # Save analysis result to database
            self._save_forecast_analysis(symbol, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error forecasting time series for {symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e)
            }
    
    def _simple_forecast(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """
        Perform a simple moving average forecast.
        
        Args:
            df: DataFrame with price data
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            
        Returns:
            Dictionary with simple forecast analysis
        """
        # Get closing prices
        prices = df['close'].values
        
        # Calculate simple moving averages
        window_sizes = [5, 10, 20]
        sma_values = {}
        
        for window in window_sizes:
            if len(prices) >= window:
                sma = np.convolve(prices, np.ones(window)/window, mode='valid')
                sma_values[window] = sma[-1]
        
        # Calculate average daily change over last 10 periods
        if len(prices) >= 11:
            daily_changes = [(prices[i] / prices[i-1]) - 1 for i in range(1, 11)]
            avg_daily_change = np.mean(daily_changes)
        else:
            avg_daily_change = 0
        
        # Generate a simple forecast
        last_price = prices[-1]
        forecast_data = []
        
        for i in range(1, self.model_params["forecast_horizon"] + 1):
            forecast_price = last_price * (1 + avg_daily_change) ** i
            lower_bound = forecast_price * 0.95  # 5% below forecast
            upper_bound = forecast_price * 1.05  # 5% above forecast
            
            forecast_data.append({
                "period": i,
                "price": forecast_price,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound
            })
        
        # Determine forecast direction and change
        forecast_direction = "upward" if forecast_data[-1]["price"] > last_price else "downward"
        forecast_change = (forecast_data[-1]["price"] / last_price - 1) * 100
        
        # Generate forecast summary
        forecast_summary = f"Simple forecast predicts a {forecast_direction} move of {abs(forecast_change):.2f}% over the next {self.model_params['forecast_horizon']} periods based on recent price action."
        forecast_summary += " This is a basic projection with low reliability. Consider more sophisticated analysis methods."
        
        # Trading implication
        trading_implication = "Simple forecasts have low reliability. Use only as a supplementary indicator."
        
        # Format timestamp as string for proper JSON serialization
        if 'timestamp' in df.columns and isinstance(df['timestamp'].iloc[0], pd.Timestamp):
            df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Assemble the forecast result
        return {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "timestamp": datetime.now(),
            "status": "success",
            "model_details": {
                "type": "Simple Moving Average",
                "sma_values": sma_values,
                "avg_daily_change": avg_daily_change
            },
            "forecast": {
                "horizon": self.model_params["forecast_horizon"],
                "direction": forecast_direction,
                "percent_change": forecast_change,
                "reliability": "low",
                "reliability_score": 30,
                "data": forecast_data
            },
            "summary": forecast_summary,
            "trading_implication": trading_implication,
            "recent_data": df.iloc[-20:][['timestamp', 'close']].to_dict('records') if 'timestamp' in df.columns else df.iloc[-20:][['close']].to_dict('records')
        }
    
    def find_pair_trading_opportunities(self, symbols: List[str], exchange: str = "NSE",
                                      timeframe: str = "day", days: int = 100) -> Dict[str, Any]:
        """
        Find pair trading opportunities among a list of symbols.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with pair trading opportunities
        """
        try:
            self.logger.info(f"Finding pair trading opportunities among {len(symbols)} symbols")
            
            if len(symbols) < 2:
                return {
                    "status": "error",
                    "error": "Need at least 2 symbols to find pairs"
                }
            
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
                    "error": "Insufficient data for pair analysis"
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
            
            # Calculate correlations
            correlation_matrix = combined_df.corr()
            
            # Find potential pairs based on correlation
            potential_pairs = []
            
            symbols_list = list(correlation_matrix.columns)
            for i in range(len(symbols_list)):
                for j in range(i+1, len(symbols_list)):
                    symbol1 = symbols_list[i]
                    symbol2 = symbols_list[j]
                    correlation = correlation_matrix.loc[symbol1, symbol2]
                    
                    if correlation >= self.model_params["pair_correlation_threshold"]:
                        # Test for cointegration
                        try:
                            coint_result = coint(combined_df[symbol1], combined_df[symbol2])
                            coint_pvalue = coint_result[1]
                            is_cointegrated = coint_pvalue < self.model_params["pair_cointegration_pvalue"]
                            
                            if is_cointegrated:
                                # Run OLS regression to find hedge ratio
                                X = sm.add_constant(combined_df[symbol2])
                                model = sm.OLS(combined_df[symbol1], X).fit()
                                hedge_ratio = model.params[1]
                                alpha = model.params[0]
                                
                                # Calculate spread and z-score
                                spread = combined_df[symbol1] - (hedge_ratio * combined_df[symbol2] + alpha)
                                spread_mean = spread.mean()
                                spread_std = spread.std()
                                current_zscore = (spread.iloc[-1] - spread_mean) / spread_std
                                
                                # Calculate pair score
                                pair_score = correlation * 50 + (1 - coint_pvalue) * 50
                                pair_score = min(100, max(0, pair_score))
                                
                                # Determine if there's a current trading opportunity
                                has_opportunity = abs(current_zscore) > self.model_params["pair_entry_threshold"]
                                opportunity_type = None
                                
                                if has_opportunity:
                                    if current_zscore > self.model_params["pair_entry_threshold"]:
                                        opportunity_type = f"Short {symbol1}, Long {symbol2}"
                                    else:
                                        opportunity_type = f"Long {symbol1}, Short {symbol2}"
                                
                                potential_pairs.append({
                                    "symbol1": symbol1,
                                    "symbol2": symbol2,
                                    "correlation": correlation,
                                    "cointegration_pvalue": coint_pvalue,
                                    "hedge_ratio": hedge_ratio,
                                    "current_zscore": current_zscore,
                                    "pair_score": pair_score,
                                    "has_opportunity": has_opportunity,
                                    "opportunity_type": opportunity_type
                                })
                        except:
                            # Skip pairs that fail cointegration test
                            continue
            
            # Sort pairs by score
            potential_pairs.sort(key=lambda x: x["pair_score"], reverse=True)
            
            # Find pairs with current opportunities
            current_opportunities = [pair for pair in potential_pairs if pair["has_opportunity"]]
            
            # Generate summary
            if potential_pairs:
                summary = f"Found {len(potential_pairs)} potential pairs for trading among {len(symbols)} symbols."
                if current_opportunities:
                    summary += f" {len(current_opportunities)} pairs currently present trading opportunities."
            else:
                summary = f"No suitable pairs found among {len(symbols)} symbols."
            
            return {
                "status": "success",
                "potential_pairs": potential_pairs,
                "current_opportunities": current_opportunities,
                "summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error finding pair trading opportunities: {e}")
            return {
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
                    with_indicators=False
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
    
    def _save_mean_reversion_analysis(self, symbol: str, exchange: str, timeframe: str, result: Dict[str, Any]) -> bool:
        """
        Save mean reversion analysis to database.
        
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
                "analysis_type": "mean_reversion",
                "mean_reversion_metrics": result.get("mean_reversion_metrics"),
                "mean_reversion_score": result.get("mean_reversion_score"),
                "suitability": result.get("suitability"),
                "recent_zscore": result.get("recent_zscore"),
                "entry_signals": result.get("entry_signals"),
                "exit_signals": result.get("exit_signals"),
                "summary": result.get("summary")
            }
            
            # Insert into database
            self.db.statistical_analysis_collection.insert_one(document)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving mean reversion analysis: {e}")
            return False
    
    def _save_pair_trading_analysis(self, symbol1: str, symbol2: str, exchange: str, 
                                 timeframe: str, result: Dict[str, Any]) -> bool:
        """
        Save pair trading analysis to database.
        
        Args:
            symbol1: First stock symbol
            symbol2: Second stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            result: Analysis result
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document for storage
            document = {
                "symbol1": symbol1,
                "symbol2": symbol2,
                "pair_id": f"{symbol1}-{symbol2}",
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "analysis_type": "pair_trading",
                "pair_metrics": result.get("pair_metrics"),
                "pair_score": result.get("pair_score"),
                "suitability": result.get("suitability"),
                "recent_zscore": result.get("recent_zscore"),
                "entry_signals": result.get("entry_signals"),
                "exit_signals": result.get("exit_signals"),
                "summary": result.get("summary")
            }
            
            # Insert into database
            self.db.statistical_analysis_collection.insert_one(document)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving pair trading analysis: {e}")
            return False
    
    def _save_forecast_analysis(self, symbol: str, exchange: str, timeframe: str, result: Dict[str, Any]) -> bool:
        """
        Save time series forecast analysis to database.
        
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
                "analysis_type": "time_series_forecast",
                "model_details": result.get("model_details"),
                "forecast": result.get("forecast"),
                "summary": result.get("summary"),
                "trading_implication": result.get("trading_implication")
            }
            
            # Insert into database
            self.db.statistical_analysis_collection.insert_one(document)
            
            return True
        except Exception as e:
            self.logger.error(f"Error saving forecast analysis: {e}")
            return False