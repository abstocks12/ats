"""
Volatility and Correlation Analysis

This module provides advanced volatility and correlation analysis capabilities for the automated trading system.
It analyzes volatility characteristics, regime shifts, and inter-asset correlations.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import math
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf
from arch import arch_model

class VolatilityAnalyzer:
    """
    Provides volatility and correlation analysis capabilities for trading decisions.
    Analyzes volatility patterns, regime shifts, and relationships between assets.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the volatility analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get query optimizer if available
        self.query_optimizer = getattr(self.db, 'get_query_optimizer', lambda: None)()
        
        # Define analysis parameters
        self.analysis_params = {
            # Volatility calculation
            "volatility_window": 21,             # Rolling window for historical volatility
            "parkinson_window": 21,              # Window for Parkinson volatility
            "garman_klass_window": 21,           # Window for Garman-Klass volatility
            
            # Volatility regime classification
            "high_volatility_percentile": 75,    # Percentile above which volatility is considered high
            "low_volatility_percentile": 25,     # Percentile below which volatility is considered low
            "volatility_history_days": 252,      # Days of history for percentile calculation
            
            # Correlation analysis
            "correlation_window": 63,            # Rolling window for correlation calculation (63 days ≈ 3 months)
            "strong_correlation": 0.7,           # Threshold for strong correlation
            "inverse_correlation": -0.7,         # Threshold for inverse correlation
            
            # Volatility clustering
            "high_vol_threshold": 1.5,           # Times median for high volatility day
            "autocorrelation_lags": 10,          # Lags for volatility autocorrelation
            
            # Regime detection
            "minimum_regime_duration": 21,       # Minimum days for a volatility regime
            "regime_detection_lookback": 252     # Lookback period for regime detection
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
    
    def analyze_volatility(self, symbol: str, exchange: str = "NSE", 
                          timeframe: str = "day", days: int = 252) -> Dict[str, Any]:
        """
        Perform comprehensive volatility analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary containing volatility analysis
        """
        try:
            self.logger.info(f"Analyzing volatility for {symbol} ({exchange}) on {timeframe} timeframe")
            
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
            required_columns = ["timestamp", "open", "high", "low", "close"]
            if not all(col in df.columns for col in required_columns):
                return {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "status": "error",
                    "error": "Incomplete data: missing required columns"
                }
            
            # Sort by timestamp
            df = df.sort_values("timestamp")
            
            # Calculate various volatility metrics
            df = self._calculate_volatility_metrics(df)
            
            # Detect volatility regimes
            regimes = self._detect_volatility_regimes(df)
            
            # Analyze volatility clustering
            clustering = self._analyze_volatility_clustering(df)
            
            # Calculate implied vs. realized volatility comparison if available
            implied_vs_realized = self._compare_implied_realized(symbol, exchange, df)
            
            # Analyze volume-volatility relationship
            volume_volatility = self._analyze_volume_volatility(df)
            
            # Try to predict next-period volatility
            volatility_forecast = self._forecast_volatility(df)
            
            # Generate volatility surface (term structure and skew)
            volatility_surface = self._generate_volatility_surface(symbol, exchange)
            
            # Format timestamp as string for proper JSON serialization
            if isinstance(df['timestamp'].iloc[0], pd.Timestamp):
                df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Generate summary
            volatility_summary = self._generate_volatility_summary(
                df, regimes, clustering, implied_vs_realized, volatility_forecast
            )
            
            # Assemble the analysis result
            result = {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "status": "success",
                "volatility_metrics": {
                    "current_historical_volatility": df['historical_volatility'].iloc[-1],
                    "current_parkinson_volatility": df['parkinson_volatility'].iloc[-1],
                    "current_garman_klass_volatility": df['garman_klass_volatility'].iloc[-1],
                    "annualized_volatility": df['annualized_volatility'].iloc[-1],
                    "volatility_percentile": df['volatility_percentile'].iloc[-1]
                },
                "volatility_regimes": regimes,
                "volatility_clustering": clustering,
                "implied_vs_realized": implied_vs_realized,
                "volume_volatility_relationship": volume_volatility,
                "volatility_forecast": volatility_forecast,
                "volatility_surface": volatility_surface,
                "volatility_summary": volatility_summary,
                "recent_data": df.iloc[-30:][['timestamp', 'close', 'historical_volatility', 'volatility_regime']].to_dict('records')
            }
            
            # Save analysis result to database
            self._save_volatility_analysis(symbol, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing volatility for {symbol}: {e}")
            return {
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "status": "error",
                "error": str(e)
            }
    
    def _calculate_volatility_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various volatility metrics.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            DataFrame with added volatility metrics
        """
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate returns
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # ==== 1. Historical Volatility (Close-to-Close) ====
        # Standard deviation of log returns
        window = self.analysis_params["volatility_window"]
        df['historical_volatility'] = df['log_return'].rolling(window=window).std() * 100
        
        # ==== 2. Parkinson Volatility ====
        # Uses high-low range, captures intraday volatility
        parkinson_window = self.analysis_params["parkinson_window"]
        df['parkinsons_range'] = np.log(df['high'] / df['low']) ** 2
        df['parkinson_volatility'] = np.sqrt(
            1 / (4 * np.log(2)) * 
            df['parkinsons_range'].rolling(window=parkinson_window).sum() / parkinson_window
        ) * 100
        
        # ==== 3. Garman-Klass Volatility ====
        # Incorporates open, high, low, and close
        gk_window = self.analysis_params["garman_klass_window"]
        
        # Calculate components
        df['hl_range'] = np.log(df['high'] / df['low']) ** 2
        df['co_range'] = np.log(df['close'] / df['open']) ** 2
        
        # Garman-Klass formula
        df['gk_component'] = 0.5 * df['hl_range'] - (2 * np.log(2) - 1) * df['co_range']
        df['garman_klass_volatility'] = np.sqrt(
            df['gk_component'].rolling(window=gk_window).sum() / gk_window
        ) * 100
        
        # ==== 4. Annualized Volatility ====
        # Convert to annualized volatility (assuming the timeframe is daily)
        trading_days_per_year = 252
        df['annualized_volatility'] = df['historical_volatility'] * np.sqrt(trading_days_per_year)
        
        # ==== 5. Volatility of Volatility ====
        # Measures the stability of volatility itself
        df['volatility_of_volatility'] = df['historical_volatility'].pct_change().rolling(window=window).std() * 100
        
        # ==== 6. Relative Volatility ====
        # Compare current volatility to its historical range
        if len(df) > self.analysis_params["volatility_history_days"]:
            lookback = min(len(df), self.analysis_params["volatility_history_days"])
            rolling_vol = df['historical_volatility'].rolling(window=lookback)
            df['volatility_percentile'] = df['historical_volatility'].rolling(window=1).apply(
                lambda x: stats.percentileofscore(
                    rolling_vol.dropna().values.flatten(), 
                    x[0]
                )
            )
        else:
            # If not enough history, use the available data
            rolling_vol = df['historical_volatility'].expanding()
            df['volatility_percentile'] = df['historical_volatility'].rolling(window=1).apply(
                lambda x: stats.percentileofscore(
                    rolling_vol.dropna().values.flatten(), 
                    x[0]
                ) if len(rolling_vol.dropna()) > 0 else 50
            )
        
        # ==== 7. Determine volatility regime ====
        # Classify as low, normal, or high volatility
        df['volatility_regime'] = 'normal'
        df.loc[df['volatility_percentile'] >= self.analysis_params["high_volatility_percentile"], 'volatility_regime'] = 'high'
        df.loc[df['volatility_percentile'] <= self.analysis_params["low_volatility_percentile"], 'volatility_regime'] = 'low'
        
        # ==== 8. Volatility trend ====
        # Calculate trend in volatility
        df['volatility_trend'] = np.nan
        
        # Calculate 5-day and 20-day moving averages of volatility
        df['vol_ma5'] = df['historical_volatility'].rolling(window=5).mean()
        df['vol_ma20'] = df['historical_volatility'].rolling(window=20).mean()
        
        # Determine trend based on moving average relationship
        df['volatility_trend'] = 'stable'
        df.loc[df['vol_ma5'] > df['vol_ma20'] * 1.1, 'volatility_trend'] = 'increasing'
        df.loc[df['vol_ma5'] < df['vol_ma20'] * 0.9, 'volatility_trend'] = 'decreasing'
        
        # Clean up NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return df
    
    def _detect_volatility_regimes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect volatility regimes using statistical methods.
        
        Args:
            df: DataFrame with volatility metrics
            
        Returns:
            Dictionary with volatility regime analysis
        """
        if len(df) < 60:
            return {
                "current_regime": "insufficient_data",
                "regimes": [],
                "transitions": []
            }
        
        # Get historical volatility
        volatility = df['historical_volatility'].values
        
        try:
            # Try to use change point detection to identify regime shifts
            from ruptures import Pelt
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Detect change points using Pelt algorithm
                model = "l2"  # L2 cost function
                algo = Pelt(model=model, min_size=self.analysis_params["minimum_regime_duration"]).fit(volatility.reshape(-1, 1))
                change_points = algo.predict(pen=10)
                
                # Convert change points to dates
                regime_changes = []
                for cp in change_points[:-1]:  # Last point is just the series length
                    if cp < len(df):
                        regime_changes.append({
                            "date": df['timestamp'].iloc[cp],
                            "volatility": df['historical_volatility'].iloc[cp],
                            "previous_regime": df['volatility_regime'].iloc[max(0, cp-1)],
                            "new_regime": df['volatility_regime'].iloc[min(cp, len(df)-1)]
                        })
                
                # Create regime periods
                regimes = []
                for i in range(len(change_points) - 1):
                    start = change_points[i]
                    end = change_points[i+1] - 1
                    
                    if start < len(df) and end < len(df):
                        # Calculate average volatility in this regime
                        avg_vol = df['historical_volatility'].iloc[start:end].mean()
                        
                        # Determine regime type
                        if avg_vol < df['historical_volatility'].quantile(0.33):
                            regime_type = "low"
                        elif avg_vol > df['historical_volatility'].quantile(0.67):
                            regime_type = "high"
                        else:
                            regime_type = "normal"
                        
                        regimes.append({
                            "start_date": df['timestamp'].iloc[start],
                            "end_date": df['timestamp'].iloc[end],
                            "duration_days": end - start + 1,
                            "avg_volatility": avg_vol,
                            "regime_type": regime_type
                        })
        except ImportError:
            # Fallback method if ruptures is not available
            # Use simple threshold crossings to identify regimes
            regimes = []
            regime_changes = []
            
            current_regime = df['volatility_regime'].iloc[0]
            regime_start = 0
            
            for i in range(1, len(df)):
                if df['volatility_regime'].iloc[i] != current_regime:
                    # Regime changed, check if it's long enough
                    if i - regime_start >= self.analysis_params["minimum_regime_duration"]:
                        # Record the completed regime
                        regimes.append({
                            "start_date": df['timestamp'].iloc[regime_start],
                            "end_date": df['timestamp'].iloc[i-1],
                            "duration_days": i - regime_start,
                            "avg_volatility": df['historical_volatility'].iloc[regime_start:i].mean(),
                            "regime_type": current_regime
                        })
                        
                        # Record the transition
                        regime_changes.append({
                            "date": df['timestamp'].iloc[i],
                            "volatility": df['historical_volatility'].iloc[i],
                            "previous_regime": current_regime,
                            "new_regime": df['volatility_regime'].iloc[i]
                        })
                        
                        # Start new regime
                        current_regime = df['volatility_regime'].iloc[i]
                        regime_start = i
                    else:
                        # Not a significant regime change, revert to previous regime
                        current_regime = df['volatility_regime'].iloc[i]
            
            # Add the final regime if it's long enough
            if len(df) - regime_start >= self.analysis_params["minimum_regime_duration"]:
                regimes.append({
                    "start_date": df['timestamp'].iloc[regime_start],
                    "end_date": df['timestamp'].iloc[-1],
                    "duration_days": len(df) - regime_start,
                    "avg_volatility": df['historical_volatility'].iloc[regime_start:].mean(),
                    "regime_type": current_regime
                })
        
        # Get current regime
        current_regime = df['volatility_regime'].iloc[-1]
        current_regime_duration = 0
        
        # Calculate how long we've been in the current regime
        for i in range(len(df) - 1, 0, -1):
            if df['volatility_regime'].iloc[i] != df['volatility_regime'].iloc[i-1]:
                current_regime_duration = len(df) - i
                break
        
        # If we never found a change, the entire series is one regime
        if current_regime_duration == 0:
            current_regime_duration = len(df)
        
        # Return the results
        return {
            "current_regime": current_regime,
            "current_regime_duration_days": current_regime_duration,
            "regimes": regimes[-5:] if len(regimes) > 5 else regimes,  # Return the most recent 5 regimes
            "transitions": regime_changes[-5:] if len(regime_changes) > 5 else regime_changes  # Return the most recent 5 transitions
        }
    
    def _analyze_volatility_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze volatility clustering characteristics.
        
        Args:
            df: DataFrame with volatility metrics
            
        Returns:
            Dictionary with volatility clustering analysis
        """
        if len(df) < 30:
            return {
                "has_clustering": "insufficient_data",
                "cluster_strength": None,
                "autocorrelation": None,
                "high_volatility_runs": []
            }
        
        # Get log returns and squared log returns (proxy for volatility)
        returns = df['log_return'].dropna().values
        squared_returns = returns ** 2
        
        # Calculate autocorrelation of squared returns
        lags = min(self.analysis_params["autocorrelation_lags"], len(squared_returns) // 4)
        autocorr = acf(squared_returns, nlags=lags)
        
        # Average autocorrelation (excluding lag 0 which is always 1)
        avg_autocorr = np.mean(autocorr[1:])
        
        # Determine if there is significant volatility clustering
        # Autocorrelation > 0.2 indicates clustering
        has_clustering = False
        cluster_strength = "none"
        
        if avg_autocorr > 0.3:
            has_clustering = True
            cluster_strength = "strong"
        elif avg_autocorr > 0.2:
            has_clustering = True
            cluster_strength = "moderate"
        elif avg_autocorr > 0.1:
            has_clustering = True
            cluster_strength = "weak"
        
        # Find runs of high volatility days
        median_vol = df['historical_volatility'].median()
        high_vol_threshold = median_vol * self.analysis_params["high_vol_threshold"]
        
        high_vol_days = df['historical_volatility'] > high_vol_threshold
        
        # Find runs of consecutive high volatility days
        high_vol_runs = []
        run_start = None
        
        for i in range(len(df)):
            if high_vol_days.iloc[i]:
                if run_start is None:
                    run_start = i
            else:
                if run_start is not None:
                    run_length = i - run_start
                    if run_length >= 3:  # Only count runs of at least 3 days
                        high_vol_runs.append({
                            "start_date": df['timestamp'].iloc[run_start],
                            "end_date": df['timestamp'].iloc[i-1],
                            "duration_days": run_length,
                            "avg_volatility": df['historical_volatility'].iloc[run_start:i].mean()
                        })
                    run_start = None
        
        # Check if there's an ongoing run at the end of the data
        if run_start is not None:
            run_length = len(df) - run_start
            if run_length >= 3:
                high_vol_runs.append({
                    "start_date": df['timestamp'].iloc[run_start],
                    "end_date": df['timestamp'].iloc[-1],
                    "duration_days": run_length,
                    "avg_volatility": df['historical_volatility'].iloc[run_start:].mean()
                })
        
        # Get most recent high volatility runs
        recent_runs = high_vol_runs[-3:] if high_vol_runs else []
        
        # Format autocorrelation for output
        autocorr_output = [{"lag": i, "autocorrelation": autocorr[i]} for i in range(1, len(autocorr))]
        
        return {
            "has_clustering": has_clustering,
            "cluster_strength": cluster_strength,
            "average_autocorrelation": avg_autocorr,
            "autocorrelation": autocorr_output,
            "high_volatility_runs": recent_runs
        }
    
    def _compare_implied_realized(self, symbol: str, exchange: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare implied and realized volatility if available.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            df: DataFrame with volatility metrics
            
        Returns:
            Dictionary with implied vs realized analysis
        """
        # Try to get implied volatility data from options collection
        try:
            # Find recent implied volatility data
            implied_vol_data = self.db.options_data_collection.find_one({
                "underlying_symbol": symbol,
                "exchange": exchange
            }, sort=[("timestamp", -1)])
            
            if implied_vol_data and "atm_implied_volatility" in implied_vol_data:
                # Get the at-the-money implied volatility
                atm_implied_vol = implied_vol_data["atm_implied_volatility"]
                iv_timestamp = implied_vol_data["timestamp"]
                
                # Get the realized volatility for comparison
                recent_realized_vol = df['annualized_volatility'].iloc[-1]
                
                # Calculate volatility risk premium (IV - RV)
                vol_risk_premium = atm_implied_vol - recent_realized_vol
                vol_premium_ratio = atm_implied_vol / recent_realized_vol if recent_realized_vol > 0 else None
                
                # Determine if options are relatively expensive or cheap
                if vol_premium_ratio is not None:
                    if vol_premium_ratio > 1.3:
                        options_valuation = "expensive"
                    elif vol_premium_ratio < 0.9:
                        options_valuation = "cheap"
                    else:
                        options_valuation = "fair"
                else:
                    options_valuation = "unknown"
                
                # Historical comparison
                historical_vols = list(self.db.options_data_collection.find({
                    "underlying_symbol": symbol,
                    "exchange": exchange
                }, {"timestamp": 1, "atm_implied_volatility": 1}).sort("timestamp", -1).limit(30))
                
                # Calculate average IV over recent history
                if historical_vols:
                    avg_iv = sum(doc.get("atm_implied_volatility", 0) for doc in historical_vols) / len(historical_vols)
                    iv_percentile = stats.percentileofscore(
                        [doc.get("atm_implied_volatility", 0) for doc in historical_vols], 
                        atm_implied_vol
                    )
                else:
                    avg_iv = None
                    iv_percentile = None
                
                return {
                    "implied_volatility": atm_implied_vol,
                    "implied_vol_date": iv_timestamp,
                    "realized_volatility": recent_realized_vol,
                    "volatility_risk_premium": vol_risk_premium,
                    "volatility_premium_ratio": vol_premium_ratio,
                    "options_valuation": options_valuation,
                    "avg_implied_volatility": avg_iv,
                    "implied_vol_percentile": iv_percentile
                }
            
        except Exception as e:
            self.logger.error(f"Error comparing implied/realized volatility: {e}")
        
        # Return empty result if no implied volatility data
        return {
            "implied_volatility": None,
            "realized_volatility": df['annualized_volatility'].iloc[-1] if len(df) > 0 else None,
            "volatility_risk_premium": None,
            "volatility_premium_ratio": None,
            "options_valuation": "no_data"
        }
    
    def _analyze_volume_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze relationship between volume and volatility.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Dictionary with volume-volatility analysis
        """
        if len(df) < 30 or 'volume' not in df.columns:
            return {
                "correlation": None,
                "relationship": "insufficient_data"
            }
        
        # Calculate normalized volume (as % of moving average)
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()
        df['normalized_volume'] = df['volume'] / df['volume_ma20']
        
        # Calculate correlation between volume and volatility
        vol_vol_corr = df['historical_volatility'].corr(df['volume'])
        norm_vol_vol_corr = df['historical_volatility'].corr(df['normalized_volume'])
        
        # Determine relationship type
        relationship = "no_clear_relationship"
        corr_to_use = norm_vol_vol_corr if not np.isnan(norm_vol_vol_corr) else vol_vol_corr
        
        if corr_to_use > 0.5:
            relationship = "strong_positive"
        elif corr_to_use > 0.3:
            relationship = "moderate_positive"
        elif corr_to_use > 0.1:
            relationship = "weak_positive"
        elif corr_to_use < -0.5:
            relationship = "strong_negative"
        elif corr_to_use < -0.3:
            relationship = "moderate_negative"
        elif corr_to_use < -0.1:
            relationship = "weak_negative"
        
        # Check if high volatility days coincide with high volume
        high_vol_days = df['historical_volatility'] > df['historical_volatility'].quantile(0.8)
        high_vol_avg_volume = df.loc[high_vol_days, 'normalized_volume'].mean()
        
        # Normal days average volume (for comparison)
        normal_vol_avg_volume = df.loc[~high_vol_days, 'normalized_volume'].mean()
        
        # Volume ratio during high volatility
        if normal_vol_avg_volume > 0:
            high_vol_volume_ratio = high_vol_avg_volume / normal_vol_avg_volume
        else:
            high_vol_volume_ratio = None
        
        # Analyze volume spikes vs volatility spikes
        volume_spikes = df['normalized_volume'] > 2.0
        vol_spikes = df['historical_volatility'] > df['historical_volatility'].rolling(window=20).mean() * 1.5
        
        # Count coincident spikes
        coincident_spikes = (volume_spikes & vol_spikes).sum()
        total_vol_spikes = vol_spikes.sum()
        
        # Calculate percentage of volatility spikes preceded by volume spikes
        if total_vol_spikes > 0:
            vol_spike_with_volume = coincident_spikes / total_vol_spikes * 100
        else:
            vol_spike_with_volume = None
        
        return {
            "correlation": corr_to_use,
            "relationship": relationship,
            "high_vol_volume_ratio": high_vol_volume_ratio,
            "volatility_spikes_with_volume_percent": vol_spike_with_volume,
            "interpretation": self._interpret_volume_vol_relationship(relationship, high_vol_volume_ratio)
        }
    
    def _interpret_volume_vol_relationship(self, relationship: str, ratio: Optional[float]) -> str:
        """
        Interpret the volume-volatility relationship.
        
        Args:
            relationship: Relationship type
            ratio: Volume ratio during high volatility
            
        Returns:
            Interpretation string
        """
        if relationship == "insufficient_data":
            return "Insufficient data to analyze volume-volatility relationship."
        
        interpretations = {
            "strong_positive": "Strong positive correlation between volume and volatility. " 
                             "Higher volume typically accompanies higher volatility.",
            "moderate_positive": "Moderate positive correlation between volume and volatility. "
                               "Higher volume often accompanies higher volatility.",
            "weak_positive": "Weak positive correlation between volume and volatility. "
                           "Some association between volume and volatility.",
            "no_clear_relationship": "No clear relationship between volume and volatility.",
            "weak_negative": "Weak negative correlation between volume and volatility. "
                           "Unusual pattern where higher volatility occurs with lower volume.",
            "moderate_negative": "Moderate negative correlation between volume and volatility. "
                               "Unusual pattern where higher volatility frequently occurs with lower volume.",
            "strong_negative": "Strong negative correlation between volume and volatility. "
                             "Unusual pattern where higher volatility consistently occurs with lower volume."
        }
        
        base_interpretation = interpretations.get(relationship, "Relationship is unclear.")
        
        # Add context about volume during high volatility periods
        if ratio is not None:
            if ratio > 1.5:
                return base_interpretation + f" During high volatility periods, volume is {ratio:.1f}x higher than normal."
            elif ratio > 1.1:
                return base_interpretation + f" During high volatility periods, volume is {ratio:.1f}x higher than normal."
            elif ratio < 0.9:
                return base_interpretation + f" During high volatility periods, volume is actually {1/ratio:.1f}x lower than normal, which is unusual."
            else:
                return base_interpretation + " Volume during high volatility periods is similar to normal periods."
        
        return base_interpretation
    
    def _forecast_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Forecast future volatility using GARCH model.
        
        Args:
            df: DataFrame with volatility metrics
            
        Returns:
            Dictionary with volatility forecast
        """
        if len(df) < 60:
            return {
                "status": "insufficient_data",
                "forecast": None,
                "model": None
            }
        
        try:
            # Try to fit GARCH model
            returns = df['log_return'].dropna() * 100  # Scale for numerical stability
            
            # Create GARCH model
            garch = arch_model(returns, vol='Garch', p=1, q=1)
            
            # Fit the model
            try:
                model_fit = garch.fit(disp='off')
                
                # Forecast volatility for the next 5 periods
                forecast = model_fit.forecast(horizon=5)
                forecast_variance = forecast.variance.iloc[-1].values
                
                # Convert variance to volatility (standard deviation)
                forecast_vol = np.sqrt(forecast_variance) / 100  # Scale back
                
                # Annualize the forecast
                trading_days = 252
                annualized_forecast = forecast_vol * np.sqrt(trading_days)
                
                # Create forecast data
                forecast_data = []
                for i in range(len(forecast_vol)):
                    forecast_data.append({
                        "horizon": i + 1,
                        "volatility": forecast_vol[i],
                        "annualized_volatility": annualized_forecast[i]
                    })
                
                # Determine forecast direction
                current_vol = df['historical_volatility'].iloc[-1]
                forecast_direction = "increasing" if forecast_vol[0] > current_vol else "decreasing"
                forecast_magnitude = abs(forecast_vol[0] - current_vol) / current_vol * 100
                
                return {
                    "status": "success",
                    "model": "GARCH(1,1)",
                    "current_volatility": current_vol,
                    "forecast_direction": forecast_direction,
                    "forecast_change_percent": forecast_magnitude,
                    "forecast_data": forecast_data
                }
                
            except:
                # If GARCH fails, try simple forecasting
                return self._simple_volatility_forecast(df)
                
        except Exception as e:
            self.logger.error(f"Error forecasting volatility: {e}")
            # Fall back to simple forecast
            return self._simple_volatility_forecast(df)
    
    def _simple_volatility_forecast(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Simple volatility forecast based on moving averages.
        
        Args:
            df: DataFrame with volatility metrics
            
        Returns:
            Dictionary with simple volatility forecast
        """
        # Get recent volatility values
        recent_vol = df['historical_volatility'].iloc[-10:].values
        
        # Calculate average volatility change over recent periods
        vol_changes = np.diff(recent_vol)
        avg_change = np.mean(vol_changes) if len(vol_changes) > 0 else 0
        
        # Current volatility
        current_vol = df['historical_volatility'].iloc[-1]
        
        # Project forward
        forecast_data = []
        for i in range(1, 6):
            forecast_vol = max(0, current_vol + avg_change * i)
            forecast_data.append({
                "horizon": i,
                "volatility": forecast_vol,
                "annualized_volatility": forecast_vol * np.sqrt(252)  # Assuming daily data
            })
        
        # Determine forecast direction
        forecast_direction = "increasing" if avg_change > 0 else "decreasing" if avg_change < 0 else "stable"
        
        # Forecast change magnitude
        forecast_magnitude = abs(avg_change * 5 / current_vol * 100) if current_vol > 0 else 0
        
        return {
            "status": "simple_forecast",
            "model": "Moving Average Projection",
            "current_volatility": current_vol,
            "forecast_direction": forecast_direction,
            "forecast_change_percent": forecast_magnitude,
            "forecast_data": forecast_data
        }
    
    def _generate_volatility_surface(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Generate volatility surface (term structure and skew) if options data is available.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with volatility surface information
        """
        try:
            # Find recent options data for the symbol
            options_data = self.db.options_data_collection.find_one({
                "underlying_symbol": symbol,
                "exchange": exchange
            }, sort=[("timestamp", -1)])
            
            if not options_data or "volatility_surface" not in options_data:
                return {
                    "available": False,
                    "reason": "No options data found"
                }
            
            # Get volatility surface data
            surface_data = options_data.get("volatility_surface", {})
            
            # Check if we have term structure and skew data
            if not surface_data or "term_structure" not in surface_data or "skew" not in surface_data:
                return {
                    "available": False,
                    "reason": "Incomplete volatility surface data"
                }
            
            term_structure = surface_data["term_structure"]
            volatility_skew = surface_data["skew"]
            
            # Analyze term structure shape
            if len(term_structure) >= 2:
                first_expiry_vol = term_structure[0]["implied_volatility"]
                last_expiry_vol = term_structure[-1]["implied_volatility"]
                
                if last_expiry_vol > first_expiry_vol * 1.1:
                    term_structure_shape = "upward_sloping"
                elif first_expiry_vol > last_expiry_vol * 1.1:
                    term_structure_shape = "downward_sloping"
                else:
                    term_structure_shape = "flat"
            else:
                term_structure_shape = "insufficient_data"
            
            # Analyze skew shape
            if len(volatility_skew) >= 3:
                otm_put_vol = volatility_skew[0]["implied_volatility"]
                atm_vol = volatility_skew[len(volatility_skew) // 2]["implied_volatility"]
                otm_call_vol = volatility_skew[-1]["implied_volatility"]
                
                if otm_put_vol > atm_vol * 1.1 and otm_call_vol > atm_vol * 1.05:
                    skew_shape = "smile"
                elif otm_put_vol > atm_vol * 1.1 and otm_call_vol <= atm_vol * 1.05:
                    skew_shape = "put_skew"
                elif otm_put_vol <= atm_vol * 1.05 and otm_call_vol > atm_vol * 1.1:
                    skew_shape = "call_skew"
                elif abs(otm_put_vol - atm_vol) / atm_vol < 0.05 and abs(otm_call_vol - atm_vol) / atm_vol < 0.05:
                    skew_shape = "flat"
                else:
                    skew_shape = "uneven"
            else:
                skew_shape = "insufficient_data"
            
            # Generate market interpretation
            interpretation = self._interpret_volatility_surface(term_structure_shape, skew_shape)
            
            return {
                "available": True,
                "timestamp": options_data.get("timestamp"),
                "term_structure": term_structure,
                "skew": volatility_skew,
                "term_structure_shape": term_structure_shape,
                "skew_shape": skew_shape,
                "market_interpretation": interpretation
            }
            
        except Exception as e:
            self.logger.error(f"Error generating volatility surface: {e}")
            return {
                "available": False,
                "reason": str(e)
            }
    
    def _interpret_volatility_surface(self, term_structure: str, skew_shape: str) -> str:
        """
        Interpret the volatility surface characteristics.
        
        Args:
            term_structure: Term structure shape
            skew_shape: Volatility skew shape
            
        Returns:
            Interpretation string
        """
        # Term structure interpretations
        term_interpretations = {
            "upward_sloping": "Upward sloping term structure indicates market expects volatility to increase in the longer term.",
            "downward_sloping": "Downward sloping term structure indicates market expects volatility to decrease in the longer term.",
            "flat": "Flat term structure indicates market expects similar volatility across different time horizons.",
            "insufficient_data": "Insufficient data to analyze term structure."
        }
        
        # Skew interpretations
        skew_interpretations = {
            "smile": "Volatility smile indicates market pricing in tail risks on both sides.",
            "put_skew": "Put skew indicates market concern about downside risks.",
            "call_skew": "Call skew indicates market anticipation of positive tail events.",
            "flat": "Flat skew suggests market sees similar probability of moves in either direction.",
            "uneven": "Uneven volatility skew without a clear pattern.",
            "insufficient_data": "Insufficient data to analyze volatility skew."
        }
        
        return f"{term_interpretations.get(term_structure, 'Term structure unclear.')} {skew_interpretations.get(skew_shape, 'Skew shape unclear.')}"
    
    def _generate_volatility_summary(self, df: pd.DataFrame, regimes: Dict[str, Any], 
                                  clustering: Dict[str, Any], 
                                  implied_vs_realized: Dict[str, Any],
                                  volatility_forecast: Dict[str, Any]) -> str:
        """
        Generate a concise summary of the volatility analysis.
        
        Args:
            df: DataFrame with volatility metrics
            regimes: Volatility regimes dictionary
            clustering: Volatility clustering dictionary
            implied_vs_realized: Implied vs realized volatility dictionary
            volatility_forecast: Volatility forecast dictionary
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Current volatility state
        current_vol = df['historical_volatility'].iloc[-1]
        current_percentile = df['volatility_percentile'].iloc[-1]
        current_regime = regimes.get("current_regime", "unknown")
        regime_duration = regimes.get("current_regime_duration_days", 0)
        
        # Basic volatility state
        if current_regime == "high":
            summary_parts.append(f"Currently in a high volatility regime (historical volatility: {current_vol:.2f}%, "
                               f"{current_percentile:.0f}th percentile) lasting {regime_duration} days.")
        elif current_regime == "low":
            summary_parts.append(f"Currently in a low volatility regime (historical volatility: {current_vol:.2f}%, "
                               f"{current_percentile:.0f}th percentile) lasting {regime_duration} days.")
        else:
            summary_parts.append(f"Currently in a normal volatility regime (historical volatility: {current_vol:.2f}%, "
                               f"{current_percentile:.0f}th percentile) lasting {regime_duration} days.")
        
        # Volatility trend
        volatility_trend = df['volatility_trend'].iloc[-1]
        if volatility_trend == "increasing":
            summary_parts.append("Volatility has been trending higher in recent periods.")
        elif volatility_trend == "decreasing":
            summary_parts.append("Volatility has been trending lower in recent periods.")
        else:
            summary_parts.append("Volatility has been relatively stable in recent periods.")
        
        # Volatility clustering
        has_clustering = clustering.get("has_clustering", False)
        if has_clustering:
            cluster_strength = clustering.get("cluster_strength", "unknown")
            if cluster_strength == "strong":
                summary_parts.append("Strong volatility clustering detected, suggesting that high volatility periods are likely to persist.")
            elif cluster_strength == "moderate":
                summary_parts.append("Moderate volatility clustering detected, suggesting some persistence in volatility.")
            else:
                summary_parts.append("Weak volatility clustering detected.")
        
        # Implied vs realized volatility
        iv = implied_vs_realized.get("implied_volatility")
        if iv is not None:
            rv = implied_vs_realized.get("realized_volatility")
            options_valuation = implied_vs_realized.get("options_valuation")
            
            if options_valuation == "expensive":
                summary_parts.append(f"Options implied volatility ({iv:.2f}%) is significantly higher than realized volatility ({rv:.2f}%), suggesting options may be expensive.")
            elif options_valuation == "cheap":
                summary_parts.append(f"Options implied volatility ({iv:.2f}%) is lower than realized volatility ({rv:.2f}%), suggesting options may be relatively cheap.")
            else:
                summary_parts.append(f"Options implied volatility ({iv:.2f}%) is in line with realized volatility ({rv:.2f}%).")
        
        # Volatility forecast
        forecast_direction = volatility_forecast.get("forecast_direction")
        forecast_change = volatility_forecast.get("forecast_change_percent")
        
        if forecast_direction and forecast_change is not None:
            if forecast_direction == "increasing":
                summary_parts.append(f"Volatility is forecast to increase by approximately {forecast_change:.1f}% in the near term.")
            elif forecast_direction == "decreasing":
                summary_parts.append(f"Volatility is forecast to decrease by approximately {forecast_change:.1f}% in the near term.")
            else:
                summary_parts.append("Volatility is forecast to remain relatively stable in the near term.")
        
        # Trading implications
        if current_regime == "high":
            if forecast_direction == "decreasing":
                summary_parts.append("Trading implication: Consider volatility mean-reversion strategies and gradually reducing hedges as volatility normalizes.")
            else:
                summary_parts.append("Trading implication: Maintain defensive positioning, consider volatility-based strategies, and use wider stops for directional trades.")
        elif current_regime == "low":
            if forecast_direction == "increasing":
                summary_parts.append("Trading implication: Consider adding portfolio protection as volatility is expected to rise from low levels.")
            else:
                summary_parts.append("Trading implication: Suitable for premium collection strategies and trend following, but be alert for volatility breakouts.")
        else:  # normal volatility
            if forecast_direction == "increasing":
                summary_parts.append("Trading implication: Consider adding hedges and reducing position sizes if volatility continues to increase.")
            else:
                summary_parts.append("Trading implication: Balanced approach to trading is appropriate under current volatility conditions.")
        
        return " ".join(summary_parts)
    
    def analyze_correlation_matrix(self, symbols: List[str], exchange: str = "NSE", 
                                 timeframe: str = "day", days: int = 63) -> Dict[str, Any]:
        """
        Analyze correlation matrix for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            self.logger.info(f"Analyzing correlation matrix for {len(symbols)} symbols")
            
            if len(symbols) < 2:
                return {
                    "status": "error",
                    "error": "Need at least 2 symbols for correlation analysis"
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
                    
                    if corr >= self.analysis_params["strong_correlation"]:
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
            
            # Calculate overall market correlation
            avg_correlation = sum(sum(correlation_matrix.values)) / (len(symbols_list) ** 2)
            
            # Identify correlation clusters
            clusters = self._identify_correlation_clusters(correlation_matrix)
            
            # Generate cluster summaries
            cluster_summaries = []
            for i, cluster in enumerate(clusters):
                avg_intra_cluster_corr = self._calculate_cluster_correlation(correlation_matrix, cluster)
                cluster_summaries.append({
                    "cluster_id": i + 1,
                    "symbols": cluster,
                    "avg_correlation": avg_intra_cluster_corr,
                    "description": f"Cluster {i+1}: {', '.join(cluster)}"
                })
            
            # Generate trading implications
            trading_implications = self._generate_correlation_implications(
                avg_correlation, high_correlation_pairs, inverse_correlation_pairs, potential_diversifiers, clusters
            )
            
            # Analyze for correlation stability
            correlation_stability = self._analyze_correlation_stability(symbols, exchange, timeframe)
            
            # Generate summary
            correlation_summary = self._generate_correlation_summary(
                avg_correlation, high_correlation_pairs, inverse_correlation_pairs, 
                potential_diversifiers, clusters, correlation_stability
            )
            
            # Assemble the analysis result
            result = {
                "timestamp": datetime.now(),
                "status": "success",
                "symbols": symbols_list,
                "correlation_matrix": formatted_matrix,
                "average_correlation": avg_correlation,
                "high_correlation_pairs": high_correlation_pairs[:10],  # Top 10
                "inverse_correlation_pairs": inverse_correlation_pairs[:10],  # Top 10
                "potential_diversifiers": potential_diversifiers[:5],   # Top 5
                "correlation_clusters": cluster_summaries,
                "correlation_stability": correlation_stability,
                "trading_implications": trading_implications,
                "correlation_summary": correlation_summary
            }
            
            # Save analysis result to database
            self._save_correlation_analysis(symbols, exchange, timeframe, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation matrix: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _identify_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> List[List[str]]:
        """
        Identify clusters of highly correlated symbols.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            List of symbol clusters
        """
        try:
            # Try to use hierarchical clustering
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            
            # Convert correlation matrix to distance matrix
            # Higher correlation = lower distance
            distance_matrix = 1 - np.abs(correlation_matrix.values)
            
            # Convert to condensed form
            condensed_distance = squareform(distance_matrix)
            
            # Perform hierarchical clustering
            z = linkage(condensed_distance, method='ward')
            
            # Form clusters
            max_clusters = min(5, len(correlation_matrix) // 2)  # At most 5 clusters or half the symbols
            max_clusters = max(2, max_clusters)  # At least 2 clusters
            
            clusters = fcluster(z, max_clusters, criterion='maxclust')
            
            # Group symbols by cluster
            symbols = correlation_matrix.columns
            cluster_groups = {}
            
            for i, cluster_id in enumerate(clusters):
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                
                cluster_groups[cluster_id].append(symbols[i])
            
            # Return clusters as list of lists
            return list(cluster_groups.values())
            
        except ImportError:
            # Fallback to simple clustering
            return self._simple_correlation_clustering(correlation_matrix)
    
    def _simple_correlation_clustering(self, correlation_matrix: pd.DataFrame) -> List[List[str]]:
        """
        Simple correlation clustering without scipy.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            List of symbol clusters
        """
        symbols = list(correlation_matrix.columns)
        correlation_threshold = self.analysis_params["strong_correlation"]
        
        # Initialize with each symbol in its own cluster
        clusters = [[s] for s in symbols]
        
        # Iteratively merge clusters
        merged = True
        while merged and len(clusters) > 1:
            merged = False
            
            for i in range(len(clusters)):
                if merged:
                    break
                    
                for j in range(i+1, len(clusters)):
                    # Calculate average correlation between clusters
                    correlations = []
                    
                    for s1 in clusters[i]:
                        for s2 in clusters[j]:
                            correlations.append(correlation_matrix.loc[s1, s2])
                    
                    avg_correlation = sum(correlations) / len(correlations) if correlations else 0
                    
                    # Merge if average correlation is above threshold
                    if avg_correlation >= correlation_threshold:
                        clusters[i].extend(clusters[j])
                        clusters.pop(j)
                        merged = True
                        break
        
        return clusters
    
    def _calculate_cluster_correlation(self, correlation_matrix: pd.DataFrame, cluster: List[str]) -> float:
        """
        Calculate average intra-cluster correlation.
        
        Args:
            correlation_matrix: Correlation matrix
            cluster: List of symbols in the cluster
            
        Returns:
            Average correlation within the cluster
        """
        if len(cluster) <= 1:
            return 1.0
        
        # Calculate all pairwise correlations within cluster
        correlations = []
        
        for i in range(len(cluster)):
            for j in range(i+1, len(cluster)):
                correlations.append(correlation_matrix.loc[cluster[i], cluster[j]])
        
        return sum(correlations) / len(correlations) if correlations else 0
    
    def _analyze_correlation_stability(self, symbols: List[str], exchange: str, 
                                     timeframe: str) -> Dict[str, Any]:
        """
        Analyze stability of correlations over time.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            
        Returns:
            Dictionary with correlation stability analysis
        """
        try:
            # Get historical correlation analyses
            cursor = self.db.correlation_analysis_collection.find({
                "symbols": {"$all": symbols},
                "exchange": exchange,
                "timeframe": timeframe
            }).sort("timestamp", -1).limit(10)
            
            historical_analyses = list(cursor)
            
            if len(historical_analyses) < 3:
                return {
                    "stability": "unknown",
                    "explanation": "Insufficient historical data to analyze correlation stability."
                }
            
            # Extract average correlations
            avg_correlations = [analysis.get("average_correlation", 0) for analysis in historical_analyses]
            
            # Calculate stability metrics
            correlation_std = np.std(avg_correlations)
            correlation_range = max(avg_correlations) - min(avg_correlations)
            
            # Determine stability
            stability = "stable"
            if correlation_std > 0.1:
                stability = "unstable"
            elif correlation_std > 0.05:
                stability = "moderately_stable"
            
            # Generate explanation
            if stability == "stable":
                explanation = f"Correlations have been stable over time with standard deviation of {correlation_std:.3f}."
            elif stability == "moderately_stable":
                explanation = f"Correlations have been moderately stable over time with standard deviation of {correlation_std:.3f}."
            else:
                explanation = f"Correlations have been unstable over time with standard deviation of {correlation_std:.3f} and range of {correlation_range:.3f}."
            
            return {
                "stability": stability,
                "standard_deviation": correlation_std,
                "range": correlation_range,
                "explanation": explanation
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation stability: {e}")
            return {
                "stability": "unknown",
                "explanation": f"Error analyzing correlation stability: {str(e)}"
            }
    
    def _generate_correlation_implications(self, avg_correlation: float, 
                                        high_pairs: List[Dict[str, Any]],
                                        inverse_pairs: List[Dict[str, Any]],
                                        diversifiers: List[Dict[str, Any]],
                                        clusters: List[List[str]]) -> List[str]:
        """
        Generate trading implications based on correlation analysis.
        
        Args:
            avg_correlation: Average correlation
            high_pairs: Highly correlated pairs
            inverse_pairs: Inversely correlated pairs
            diversifiers: Potential diversifiers
            clusters: Correlation clusters
            
        Returns:
            List of trading implication strings
        """
        implications = []
        
        # Overall market correlation
        if avg_correlation > 0.7:
            implications.append("High average correlation suggests limited diversification benefits within this universe. Consider expanding to other asset classes.")
        elif avg_correlation < 0.3:
            implications.append("Low average correlation provides good diversification opportunities within this universe.")
        
        # Pair trading opportunities
        if high_pairs:
            top_pair = high_pairs[0]
            implications.append(f"Strong correlation between {top_pair['symbol1']} and {top_pair['symbol2']} ({top_pair['correlation']:.2f}) suggests potential pair trading opportunities.")
        
        # Hedging opportunities
        if inverse_pairs:
            top_inverse = inverse_pairs[0]
            implications.append(f"Inverse correlation between {top_inverse['symbol1']} and {top_inverse['symbol2']} ({top_inverse['correlation']:.2f}) can be utilized for hedging purposes.")
        
        # Diversification recommendations
        if diversifiers:
            top_diversifier = diversifiers[0]
            implications.append(f"{top_diversifier['symbol']} shows low average correlation ({top_diversifier['avg_correlation']:.2f}) and may provide good diversification benefits.")
        
        # Cluster-based portfolio construction
        if len(clusters) > 1:
            implications.append(f"Consider selecting one representative security from each of the {len(clusters)} identified correlation clusters for optimal diversification.")
        
        # Risk management
        if avg_correlation > 0.5 and len(clusters) <= 2:
            implications.append("High correlations and limited clustering suggest the need for additional diversification outside this universe to manage portfolio risk.")
        
        # Sector rotation strategy
        if len(clusters) >= 3:
            implications.append("Distinct correlation clusters may enable sector rotation strategies, focusing on the strongest performing cluster while maintaining some exposure to others.")
        
        return implications
    
    def _generate_correlation_summary(self, avg_correlation: float, 
                                    high_pairs: List[Dict[str, Any]],
                                    inverse_pairs: List[Dict[str, Any]],
                                    diversifiers: List[Dict[str, Any]],
                                    clusters: List[List[str]],
                                    stability: Dict[str, Any]) -> str:
        """
        Generate a concise summary of the correlation analysis.
        
        Args:
            avg_correlation: Average correlation
            high_pairs: Highly correlated pairs
            inverse_pairs: Inversely correlated pairs
            diversifiers: Potential diversifiers
            clusters: Correlation clusters
            stability: Correlation stability analysis
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Overall correlation
        if avg_correlation > 0.7:
            summary_parts.append(f"The analyzed securities show high average correlation ({avg_correlation:.2f}), indicating strong co-movement.")
        elif avg_correlation > 0.4:
            summary_parts.append(f"The analyzed securities show moderate average correlation ({avg_correlation:.2f}).")
        else:
            summary_parts.append(f"The analyzed securities show low average correlation ({avg_correlation:.2f}), indicating good diversification potential.")
        
        # Correlation stability
        stability_level = stability.get("stability", "unknown")
        if stability_level == "stable":
            summary_parts.append("These correlation relationships have been stable over time.")
        elif stability_level == "moderately_stable":
            summary_parts.append("These correlation relationships have been moderately stable over time.")
        elif stability_level == "unstable":
            summary_parts.append("These correlation relationships have been unstable, suggesting caution when using them for long-term strategies.")
        
        # Highly correlated pairs
        if high_pairs:
            top_pair = high_pairs[0]
            summary_parts.append(f"The most highly correlated pair is {top_pair['symbol1']} and {top_pair['symbol2']} ({top_pair['correlation']:.2f}).")
        
        # Inversely correlated pairs
        if inverse_pairs:
            top_inverse = inverse_pairs[0]
            summary_parts.append(f"The most inversely correlated pair is {top_inverse['symbol1']} and {top_inverse['symbol2']} ({top_inverse['correlation']:.2f}).")
        
        # Diversifiers
        if diversifiers:
            top_div = diversifiers[0]
            summary_parts.append(f"{top_div['symbol']} shows the lowest average correlation ({top_div['avg_correlation']:.2f}) and may provide the best diversification benefits.")
        
        # Clusters
        if len(clusters) > 1:
            summary_parts.append(f"Analysis identified {len(clusters)} distinct correlation clusters that can be used for diversified portfolio construction.")
        
        # Trading implications
        if avg_correlation > 0.7:
            summary_parts.append("Consider pair trading strategies for highly correlated securities and seeking diversification outside this universe.")
        elif len(inverse_pairs) > 0:
            summary_parts.append("Opportunities exist for hedging using inversely correlated securities.")
        else:
            summary_parts.append("This universe offers reasonable diversification potential for portfolio construction.")
        
        return " ".join(summary_parts)
    
    def _get_market_data(self, symbol: str, exchange: str, timeframe: str, days: int) -> List[Dict[str, Any]]:
        """
        Get market data from the database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to retrieve
            
        Returns:
            List of market data documents
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Use query optimizer if available
            if self.query_optimizer:
                query = self.query_optimizer.optimize_market_data_query(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                # Default query
                query = {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe,
                    "timestamp": {
                        "$gte": start_date,
                        "$lte": end_date
                    }
                }
            
            # Get data from database
            cursor = self.db.market_data_collection.find(query).sort("timestamp", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return []
    
    def _save_volatility_analysis(self, symbol: str, exchange: str, timeframe: str, analysis: Dict[str, Any]) -> None:
        """
        Save volatility analysis result to database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            analysis: Analysis result dictionary
        """
        try:
            # Create document
            document = {
                "type": "volatility_analysis",
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "analysis": analysis
            }
            
            # Insert into database
            self.db.analysis_results_collection.insert_one(document)
            
        except Exception as e:
            self.logger.error(f"Error saving volatility analysis: {e}")
    
    def _save_correlation_analysis(self, symbols: List[str], exchange: str, timeframe: str, analysis: Dict[str, Any]) -> None:
        """
        Save correlation analysis result to database.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            analysis: Analysis result dictionary
        """
        try:
            # Create document
            document = {
                "type": "correlation_analysis",
                "symbols": symbols,
                "exchange": exchange,
                "timeframe": timeframe,
                "timestamp": datetime.now(),
                "analysis": analysis
            }
            
            # Insert into database
            self.db.correlation_analysis_collection.insert_one(document)
            
        except Exception as e:
            self.logger.error(f"Error saving correlation analysis: {e}")
    
    def get_recent_volatility_analysis(self, symbol: str, exchange: str = "NSE", 
                                    timeframe: str = "day") -> Optional[Dict[str, Any]]:
        """
        Get the most recent volatility analysis for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            
        Returns:
            Volatility analysis or None if not found
        """
        try:
            # Query the database for the most recent analysis
            document = self.db.analysis_results_collection.find_one(
                {
                    "type": "volatility_analysis",
                    "symbol": symbol,
                    "exchange": exchange,
                    "timeframe": timeframe
                },
                sort=[("timestamp", -1)]
            )
            
            if document:
                return document.get("analysis")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting recent volatility analysis: {e}")
            return None
    
    def get_recent_correlation_analysis(self, symbols: List[str], exchange: str = "NSE", 
                                      timeframe: str = "day") -> Optional[Dict[str, Any]]:
        """
        Get the most recent correlation analysis for a set of symbols.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            
        Returns:
            Correlation analysis or None if not found
        """
        try:
            # Query the database for the most recent analysis
            document = self.db.correlation_analysis_collection.find_one(
                {
                    "type": "correlation_analysis",
                    "symbols": {"$all": symbols},
                    "exchange": exchange,
                    "timeframe": timeframe
                },
                sort=[("timestamp", -1)]
            )
            
            if document:
                return document.get("analysis")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting recent correlation analysis: {e}")
            return None
    
    def analyze_sector_correlations(self, sector: str, exchange: str = "NSE", 
                                  timeframe: str = "day", days: int = 252) -> Dict[str, Any]:
        """
        Analyze correlations within a sector.
        
        Args:
            sector: Sector name
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with sector correlation analysis
        """
        try:
            # Get symbols in the sector
            symbols = self._get_sector_symbols(sector, exchange)
            
            if not symbols or len(symbols) < 2:
                return {
                    "status": "error",
                    "error": f"Insufficient symbols found for sector: {sector}"
                }
            
            # Perform correlation analysis
            result = self.analyze_correlation_matrix(symbols, exchange, timeframe, days)
            
            # Add sector information
            result["sector"] = sector
            result["sector_symbols"] = symbols
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing sector correlations: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _get_sector_symbols(self, sector: str, exchange: str) -> List[str]:
        """
        Get all symbols in a specific sector.
        
        Args:
            sector: Sector name
            exchange: Stock exchange
            
        Returns:
            List of symbols in the sector
        """
        try:
            # Query the portfolio collection for symbols in the sector
            cursor = self.db.portfolio_collection.find(
                {
                    "sector": sector,
                    "exchange": exchange,
                    "status": "active"
                }
            )
            
            return [doc["symbol"] for doc in cursor]
            
        except Exception as e:
            self.logger.error(f"Error getting sector symbols: {e}")
            return []
    
    def analyze_cross_asset_correlations(self, symbols: Dict[str, List[str]], 
                                       timeframe: str = "day", days: int = 252) -> Dict[str, Any]:
        """
        Analyze correlations across different asset classes.
        
        Args:
            symbols: Dictionary mapping asset class to list of symbols
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with cross-asset correlation analysis
        """
        try:
            # Flatten the symbols dictionary
            all_symbols = []
            symbol_classes = {}
            
            for asset_class, class_symbols in symbols.items():
                for symbol in class_symbols:
                    all_symbols.append(symbol)
                    symbol_classes[symbol] = asset_class
            
            if len(all_symbols) < 2:
                return {
                    "status": "error",
                    "error": "Need at least 2 symbols for correlation analysis"
                }
            
            # Get market data for all symbols
            price_data = {}
            for symbol in all_symbols:
                # Extract exchange if provided in format "SYMBOL:EXCHANGE"
                if ":" in symbol:
                    sym, exch = symbol.split(":")
                else:
                    sym = symbol
                    exch = "NSE"  # Default
                
                data = self._get_market_data(sym, exch, timeframe, days)
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
            
            # Calculate average correlations within and across asset classes
            intra_class_correlations = {}
            inter_class_correlations = {}
            
            for class1 in symbols.keys():
                # Symbols in this class that we have data for
                class1_symbols = [s for s in symbols.get(class1, []) if s in correlation_matrix.columns]
                
                if len(class1_symbols) < 2:
                    continue
                
                # Calculate average correlation within this class
                intra_corrs = []
                for i in range(len(class1_symbols)):
                    for j in range(i+1, len(class1_symbols)):
                        intra_corrs.append(correlation_matrix.loc[class1_symbols[i], class1_symbols[j]])
                
                intra_class_correlations[class1] = sum(intra_corrs) / len(intra_corrs) if intra_corrs else 0
                
                # Calculate average correlation with other classes
                for class2 in symbols.keys():
                    if class1 == class2:
                        continue
                    
                    class2_symbols = [s for s in symbols.get(class2, []) if s in correlation_matrix.columns]
                    
                    if not class2_symbols:
                        continue
                    
                    inter_corrs = []
                    for s1 in class1_symbols:
                        for s2 in class2_symbols:
                            inter_corrs.append(correlation_matrix.loc[s1, s2])
                    
                    if inter_corrs:
                        if class1 not in inter_class_correlations:
                            inter_class_correlations[class1] = {}
                        
                        inter_class_correlations[class1][class2] = sum(inter_corrs) / len(inter_corrs)
            
            # Format correlation matrix for output
            symbols_list = list(correlation_matrix.columns)
            formatted_matrix = []
            for symbol1 in symbols_list:
                row = {
                    "symbol": symbol1, 
                    "asset_class": symbol_classes.get(symbol1, "unknown")
                }
                for symbol2 in symbols_list:
                    row[symbol2] = round(correlation_matrix.loc[symbol1, symbol2], 2)
                formatted_matrix.append(row)
            
            # Find the most diversifying asset classes
            class_pairs = []
            for class1, correlations in inter_class_correlations.items():
                for class2, corr in correlations.items():
                    class_pairs.append({
                        "class1": class1,
                        "class2": class2,
                        "correlation": corr
                    })
            
            # Sort by correlation (lowest first for best diversification)
            class_pairs.sort(key=lambda x: x["correlation"])
            
            # Generate summary
            summary = self._generate_cross_asset_summary(
                intra_class_correlations, inter_class_correlations, class_pairs
            )
            
            # Assemble the analysis result
            result = {
                "timestamp": datetime.now(),
                "status": "success",
                "symbols_by_class": {k: [s for s in v if s in correlation_matrix.columns] for k, v in symbols.items()},
                "correlation_matrix": formatted_matrix,
                "intra_class_correlations": [{
                    "asset_class": k,
                    "avg_correlation": v
                } for k, v in intra_class_correlations.items()],
                "inter_class_correlations": [{
                    "class1": cp["class1"],
                    "class2": cp["class2"],
                    "correlation": cp["correlation"]
                } for cp in class_pairs],
                "best_diversification_pairs": class_pairs[:5],  # Top 5 lowest correlation pairs
                "cross_asset_summary": summary
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analyzing cross-asset correlations: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_cross_asset_summary(self, intra_correlations: Dict[str, float],
                                    inter_correlations: Dict[str, Dict[str, float]],
                                    class_pairs: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of cross-asset correlation analysis.
        
        Args:
            intra_correlations: Average correlation within each asset class
            inter_correlations: Average correlation between asset classes
            class_pairs: Sorted list of asset class pairs by correlation
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Highest internal correlation
        if intra_correlations:
            highest_intra = max(intra_correlations.items(), key=lambda x: x[1])
            lowest_intra = min(intra_correlations.items(), key=lambda x: x[1])
            
            summary_parts.append(f"{highest_intra[0]} shows the highest internal correlation ({highest_intra[1]:.2f}), indicating less diversification within this asset class.")
            summary_parts.append(f"{lowest_intra[0]} shows the lowest internal correlation ({lowest_intra[1]:.2f}), offering better intra-class diversification.")
        
        # Best diversification pairs
        if class_pairs:
            best_pair = class_pairs[0]
            summary_parts.append(f"The best diversification benefit is between {best_pair['class1']} and {best_pair['class2']} with correlation of {best_pair['correlation']:.2f}.")
            
            # If we have a high correlation pair, mention it too
            if len(class_pairs) > 1 and class_pairs[-1]['correlation'] > 0.6:
                worst_pair = class_pairs[-1]
                summary_parts.append(f"{worst_pair['class1']} and {worst_pair['class2']} show the highest cross-asset correlation ({worst_pair['correlation']:.2f}), suggesting limited diversification benefit.")
        
        # Portfolio construction advice
        if len(intra_correlations) >= 3 and class_pairs:
            low_corr_classes = [p['class1'] for p in class_pairs[:3]] + [p['class2'] for p in class_pairs[:3]]
            # Get unique classes that appear most frequently in low correlation pairs
            from collections import Counter
            top_classes = [item[0] for item in Counter(low_corr_classes).most_common(3)]
            
            if top_classes:
                summary_parts.append(f"For optimal diversification, consider allocating across these asset classes: {', '.join(top_classes)}.")
        
        # Overall diversification assessment
        avg_cross_asset = sum(p['correlation'] for p in class_pairs) / len(class_pairs) if class_pairs else 0
        
        if avg_cross_asset < 0.3:
            summary_parts.append(f"Overall, these asset classes provide excellent diversification with average cross-asset correlation of {avg_cross_asset:.2f}.")
        elif avg_cross_asset < 0.6:
            summary_parts.append(f"Overall, these asset classes provide moderate diversification with average cross-asset correlation of {avg_cross_asset:.2f}.")
        else:
            summary_parts.append(f"Overall, these asset classes show high correlation ({avg_cross_asset:.2f}) and may not provide optimal diversification benefits.")
        
        return " ".join(summary_parts)