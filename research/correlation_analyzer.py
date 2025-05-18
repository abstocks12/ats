"""
Correlation Analysis Module

This module provides specialized correlation analysis capabilities for the automated trading system.
It analyzes relationships between different assets, sectors, and market factors.
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
import networkx as nx

class CorrelationAnalyzer:
    """
    Provides specialized correlation analysis capabilities for the automated trading system.
    Focuses on relationships between different assets, sectors, and market factors.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the correlation analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Get query optimizer if available
        self.query_optimizer = getattr(self.db, 'get_query_optimizer', lambda: None)()
        
        # Define analysis parameters
        self.analysis_params = {
            # Correlation analysis
            "correlation_window": 63,            # Rolling window for correlation calculation (63 days ≈ 3 months)
            "strong_correlation": 0.7,           # Threshold for strong correlation
            "inverse_correlation": -0.7,         # Threshold for inverse correlation
            
            # Cointegration
            "coint_significance": 0.05,          # Significance level for cointegration test
            "min_history_days": 126,             # Minimum history for cointegration (126 days ≈ 6 months)
            
            # Correlation network
            "network_correlation_threshold": 0.5, # Minimum correlation to include in network
            
            # Correlation stability
            "stability_window": 21,              # Window for rolling correlation stability
            "lookback_periods": 4,               # Number of periods to analyze stability
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
    
    def analyze_correlation_matrix(self, symbols: List[str], exchange: str = "NSE", 
                                 timeframe: str = "day", days: int = 63) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis for a list of symbols.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary containing correlation analysis
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
            
            # Analyze correlation stability
            stability = self._analyze_correlation_stability(symbols, exchange, timeframe, returns_df)
            
            # Test for cointegrated pairs if enough history
            cointegrated_pairs = []
            if days >= self.analysis_params["min_history_days"]:
                cointegrated_pairs = self._find_cointegrated_pairs(combined_df)
            
            # Build correlation network
            correlation_network = self._build_correlation_network(correlation_matrix)
            
            # Generate trading implications
            trading_implications = self._generate_correlation_implications(
                avg_correlation, high_correlation_pairs, inverse_correlation_pairs, 
                potential_diversifiers, clusters, cointegrated_pairs
            )
            
            # Generate summary
            correlation_summary = self._generate_correlation_summary(
                avg_correlation, high_correlation_pairs, inverse_correlation_pairs, 
                potential_diversifiers, clusters, stability, cointegrated_pairs
            )
            
            # Analyze rolling correlations if enough data
            rolling_correlation = None
            if len(returns_df) >= 30:
                rolling_correlation = self._analyze_rolling_correlations(returns_df)
            
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
                "correlation_stability": stability,
                "cointegrated_pairs": cointegrated_pairs[:10] if cointegrated_pairs else [],  # Top 10
                "correlation_network": correlation_network,
                "rolling_correlation": rolling_correlation,
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
    
    def analyze_rolling_correlation(self, symbol1: str, symbol2: str, exchange: str = "NSE",
                                  timeframe: str = "day", days: int = 252, 
                                  window: int = 63) -> Dict[str, Any]:
        """
        Analyze the evolution of correlation between two symbols over time.
        
        Args:
            symbol1: First stock symbol
            symbol2: Second stock symbol
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            window: Rolling window size for correlation
            
        Returns:
            Dictionary with rolling correlation analysis
        """
        try:
            self.logger.info(f"Analyzing rolling correlation between {symbol1} and {symbol2}")
            
            # Get price data for both symbols
            data1 = self._get_market_data(symbol1, exchange, timeframe, days)
            data2 = self._get_market_data(symbol2, exchange, timeframe, days)
            
            if not data1 or not data2:
                return {
                    "status": "error",
                    "error": "Insufficient data for one or both symbols"
                }
            
            # Convert to DataFrames
            df1 = pd.DataFrame(data1)[["timestamp", "close"]].sort_values("timestamp")
            df2 = pd.DataFrame(data2)[["timestamp", "close"]].sort_values("timestamp")
            
            # Set timestamp as index
            df1 = df1.set_index("timestamp")
            df2 = df2.set_index("timestamp")
            
            # Combine into a single DataFrame
            combined_df = pd.DataFrame()
            combined_df[symbol1] = df1["close"]
            combined_df[symbol2] = df2["close"]
            
            # Fill missing values
            combined_df = combined_df.fillna(method="ffill").fillna(method="bfill")
            
            # Calculate returns
            returns_df = combined_df.pct_change().dropna()
            
            # Calculate rolling correlation
            rolling_corr = returns_df[symbol1].rolling(window=window).corr(returns_df[symbol2])
            
            # Format for output
            rolling_corr_data = []
            for idx, corr in zip(rolling_corr.index[window-1:], rolling_corr.values[window-1:]):
                if not np.isnan(corr):
                    rolling_corr_data.append({
                        "date": idx,
                        "correlation": round(corr, 3)
                    })
            
            # Calculate basic statistics
            valid_corrs = [x["correlation"] for x in rolling_corr_data]
            
            if not valid_corrs:
                return {
                    "status": "error",
                    "error": "Insufficient overlapping data for correlation calculation"
                }
            
            avg_corr = sum(valid_corrs) / len(valid_corrs)
            min_corr = min(valid_corrs)
            max_corr = max(valid_corrs)
            std_corr = np.std(valid_corrs)
            
            # Analyze stability
            corr_stability = "stable"
            if std_corr > 0.2:
                corr_stability = "unstable"
            elif std_corr > 0.1:
                corr_stability = "moderately_stable"
            
            # Analyze trend
            recent_corrs = valid_corrs[-10:]
            corr_trend = "stable"
            if len(recent_corrs) >= 10:
                recent_avg = sum(recent_corrs) / len(recent_corrs)
                overall_avg = avg_corr
                
                if recent_avg > overall_avg + 0.1:
                    corr_trend = "increasing"
                elif recent_avg < overall_avg - 0.1:
                    corr_trend = "decreasing"
            
            # Analyze correlation regimes using change point detection
            regimes = self._detect_correlation_regimes(valid_corrs)
            
            # Generate analysis summary
            summary = self._generate_rolling_correlation_summary(
                symbol1, symbol2, avg_corr, std_corr, min_corr, max_corr,
                corr_stability, corr_trend, regimes
            )
            
            return {
                "status": "success",
                "symbol1": symbol1,
                "symbol2": symbol2,
                "average_correlation": avg_corr,
                "min_correlation": min_corr,
                "max_correlation": max_corr,
                "correlation_std": std_corr,
                "correlation_stability": corr_stability,
                "correlation_trend": corr_trend,
                "correlation_regimes": regimes,
                "rolling_correlation_data": rolling_corr_data,
                "analysis_summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing rolling correlation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _detect_correlation_regimes(self, correlations: List[float]) -> List[Dict[str, Any]]:
        """
        Detect correlation regimes using change point detection.
        
        Args:
            correlations: List of correlation values
            
        Returns:
            List of correlation regimes
        """
        if len(correlations) < 30:
            return []
        
        try:
            # Try to use ruptures for change point detection
            import ruptures
            
            # Convert to numpy array
            corr_array = np.array(correlations)
            
            # Detect change points
            algo = ruptures.Pelt(model="l2").fit(corr_array.reshape(-1, 1))
            change_points = algo.predict(pen=0.5)
            
            # Create regimes
            regimes = []
            for i in range(len(change_points) - 1):
                start = change_points[i]
                end = change_points[i+1] - 1
                
                regime_corrs = correlations[start:end]
                if not regime_corrs:
                    continue
                
                avg_corr = sum(regime_corrs) / len(regime_corrs)
                
                # Categorize regime
                regime_type = "moderate_correlation"
                if avg_corr > 0.7:
                    regime_type = "high_correlation"
                elif avg_corr < 0.3:
                    regime_type = "low_correlation"
                elif avg_corr < 0:
                    regime_type = "inverse_correlation"
                
                regimes.append({
                    "start_index": start,
                    "end_index": end,
                    "duration": end - start + 1,
                    "avg_correlation": avg_corr,
                    "regime_type": regime_type
                })
            
            return regimes
            
        except ImportError:
            # Fallback to simple regime detection
            return self._simple_correlation_regimes(correlations)
    
    def _simple_correlation_regimes(self, correlations: List[float]) -> List[Dict[str, Any]]:
        """
        Simple correlation regime detection without ruptures.
        
        Args:
            correlations: List of correlation values
            
        Returns:
            List of correlation regimes
        """
        if len(correlations) < 30:
            return []
        
        regimes = []
        current_regime = "unknown"
        regime_start = 0
        min_regime_duration = 5
        
        for i in range(len(correlations)):
            corr = correlations[i]
            
            # Determine regime type
            regime_type = "moderate_correlation"
            if corr > 0.7:
                regime_type = "high_correlation"
            elif corr < 0.3:
                regime_type = "low_correlation"
            elif corr < 0:
                regime_type = "inverse_correlation"
            
            # Check if regime changed
            if regime_type != current_regime:
                # If we have a valid regime, record it
                if i - regime_start >= min_regime_duration and current_regime != "unknown":
                    regime_corrs = correlations[regime_start:i]
                    avg_corr = sum(regime_corrs) / len(regime_corrs)
                    
                    regimes.append({
                        "start_index": regime_start,
                        "end_index": i - 1,
                        "duration": i - regime_start,
                        "avg_correlation": avg_corr,
                        "regime_type": current_regime
                    })
                
                # Start new regime
                current_regime = regime_type
                regime_start = i
        
        # Add final regime if long enough
        if len(correlations) - regime_start >= min_regime_duration:
            regime_corrs = correlations[regime_start:]
            avg_corr = sum(regime_corrs) / len(regime_corrs)
            
            regimes.append({
                "start_index": regime_start,
                "end_index": len(correlations) - 1,
                "duration": len(correlations) - regime_start,
                "avg_correlation": avg_corr,
                "regime_type": current_regime
            })
        
        return regimes
    
    def _generate_rolling_correlation_summary(self, symbol1: str, symbol2: str,
                                           avg_corr: float, std_corr: float,
                                           min_corr: float, max_corr: float,
                                           stability: str, trend: str,
                                           regimes: List[Dict[str, Any]]) -> str:
        """
        Generate a summary of the rolling correlation analysis.
        
        Args:
            symbol1: First symbol
            symbol2: Second symbol
            avg_corr: Average correlation
            std_corr: Standard deviation of correlation
            min_corr: Minimum correlation
            max_corr: Maximum correlation
            stability: Correlation stability
            trend: Correlation trend
            regimes: Correlation regimes
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Basic correlation description
        if avg_corr > 0.7:
            summary_parts.append(f"{symbol1} and {symbol2} show strong average correlation ({avg_corr:.2f}).")
        elif avg_corr > 0.3:
            summary_parts.append(f"{symbol1} and {symbol2} show moderate average correlation ({avg_corr:.2f}).")
        elif avg_corr > 0:
            summary_parts.append(f"{symbol1} and {symbol2} show weak average correlation ({avg_corr:.2f}).")
        elif avg_corr > -0.3:
            summary_parts.append(f"{symbol1} and {symbol2} show weak inverse correlation ({avg_corr:.2f}).")
        elif avg_corr > -0.7:
            summary_parts.append(f"{symbol1} and {symbol2} show moderate inverse correlation ({avg_corr:.2f}).")
        else:
            summary_parts.append(f"{symbol1} and {symbol2} show strong inverse correlation ({avg_corr:.2f}).")
        
        # Range of correlations
        summary_parts.append(f"Correlation has ranged from {min_corr:.2f} to {max_corr:.2f}.")
        
        # Correlation stability
        if stability == "stable":
            summary_parts.append(f"The correlation has been stable with standard deviation of {std_corr:.2f}.")
        elif stability == "moderately_stable":
            summary_parts.append(f"The correlation has been moderately stable with standard deviation of {std_corr:.2f}.")
        else:
            summary_parts.append(f"The correlation has been unstable with high standard deviation of {std_corr:.2f}.")
        
        # Recent trend
        if trend == "increasing":
            summary_parts.append("The correlation has been trending higher in recent periods.")
        elif trend == "decreasing":
            summary_parts.append("The correlation has been trending lower in recent periods.")
        else:
            summary_parts.append("The correlation has been relatively stable in recent periods.")
        
        # Recent regime
        if regimes:
            recent_regime = regimes[-1]
            regime_type = recent_regime["regime_type"]
            
            if regime_type == "high_correlation":
                summary_parts.append(f"The most recent correlation regime shows high correlation ({recent_regime['avg_correlation']:.2f}).")
            elif regime_type == "moderate_correlation":
                summary_parts.append(f"The most recent correlation regime shows moderate correlation ({recent_regime['avg_correlation']:.2f}).")
            elif regime_type == "low_correlation":
                summary_parts.append(f"The most recent correlation regime shows low correlation ({recent_regime['avg_correlation']:.2f}).")
            elif regime_type == "inverse_correlation":
                summary_parts.append(f"The most recent correlation regime shows inverse correlation ({recent_regime['avg_correlation']:.2f}).")
        
        # Trading implications
        if avg_corr > 0.8 and stability == "stable":
            summary_parts.append("These assets show strong and stable correlation, suggesting limited diversification benefit but potential pair trading opportunities.")
        elif avg_corr < -0.7 and stability == "stable":
            summary_parts.append("These assets show strong and stable inverse correlation, suggesting excellent hedging potential.")
        elif std_corr > 0.2:
            summary_parts.append("The unstable correlation suggests caution when using these assets together in a portfolio strategy.")
        elif avg_corr > 0.3 and avg_corr < 0.7:
            summary_parts.append("The moderate correlation provides some diversification benefit while maintaining sector exposure.")
        
        return " ".join(summary_parts)
    
    def _analyze_rolling_correlations(self, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze rolling correlations for all pairs in the returns DataFrame.
        
        Args:
            returns_df: DataFrame with asset returns
            
        Returns:
            Dictionary with rolling correlation analysis
        """
        symbols = returns_df.columns
        window = min(30, len(returns_df) // 2)  # Use shorter window for limited data
        
        # Get the most recent correlation
        recent_corr = returns_df.iloc[-window:].corr()
        
        # Get correlation from the first window
        if len(returns_df) >= window * 2:
            early_corr = returns_df.iloc[:window].corr()
        else:
            early_corr = recent_corr
        
        # Find pairs with significant correlation changes
        significant_changes = []
        
        for i in range(len(symbols)):
            for j in range(i+1, len(symbols)):
                symbol1 = symbols[i]
                symbol2 = symbols[j]
                
                early = early_corr.loc[symbol1, symbol2]
                recent = recent_corr.loc[symbol1, symbol2]
                
                change = recent - early
                
                if abs(change) >= 0.3:  # Significant change threshold
                    significant_changes.append({
                        "symbol1": symbol1,
                        "symbol2": symbol2,
                        "early_correlation": early,
                        "recent_correlation": recent,
                        "change": change,
                        "change_type": "increasing" if change > 0 else "decreasing"
                    })
        
        # Sort by absolute change
        significant_changes.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        # Calculate average correlation change
        if significant_changes:
            avg_change = sum(abs(x["change"]) for x in significant_changes) / len(significant_changes)
        else:
            avg_change = 0
        
        # Determine correlation stability
        if avg_change > 0.4:
            stability = "unstable"
        elif avg_change > 0.2:
            stability = "moderately_stable"
        else:
            stability = "stable"
        
        return {
            "correlation_stability": stability,
            "average_change": avg_change,
            "significant_changes": significant_changes[:10],  # Top 10 changes
            "early_window_start": 0,
            "early_window_end": window - 1,
            "recent_window_start": len(returns_df) - window,
            "recent_window_end": len(returns_df) - 1
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
                                     timeframe: str, returns_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze stability of correlations over time.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            returns_df: DataFrame with asset returns
            
        Returns:
            Dictionary with correlation stability analysis
        """
        try:
            # Calculate stability using historical analysis if available
            cursor = self.db.correlation_analysis_collection.find({
                "symbols": {"$all": symbols},
                "exchange": exchange,
                "timeframe": timeframe
            }).sort("timestamp", -1).limit(10)
            
            historical_analyses = list(cursor)
            
            if len(historical_analyses) >= 3:
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
                # explanation = f"Correlations have been stable over time with standard deviation of {correlation_std:.3f}."
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
            
            # If not enough historical analyses, calculate from the returns data
            # Divide data into periods and compare correlations
            if len(returns_df) < 60:
                return {
                    "stability": "unknown",
                    "explanation": "Insufficient data to analyze correlation stability."
                }
            
            # Calculate correlations for different time periods
            period_length = len(returns_df) // self.analysis_params["lookback_periods"]
            period_correlations = []
            
            for i in range(self.analysis_params["lookback_periods"]):
                start_idx = i * period_length
                end_idx = (i + 1) * period_length if i < self.analysis_params["lookback_periods"] - 1 else len(returns_df)
                
                period_returns = returns_df.iloc[start_idx:end_idx]
                period_corr = period_returns.corr()
                
                # Calculate average correlation for this period
                symbols = period_corr.columns
                corr_sum = 0
                count = 0
                
                for i in range(len(symbols)):
                    for j in range(i+1, len(symbols)):
                        corr_sum += period_corr.loc[symbols[i], symbols[j]]
                        count += 1
                
                avg_corr = corr_sum / count if count > 0 else 0
                period_correlations.append(avg_corr)
            
            # Calculate stability metrics
            correlation_std = np.std(period_correlations)
            correlation_range = max(period_correlations) - min(period_correlations)
            
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
                "explanation": explanation,
                "period_correlations": period_correlations
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation stability: {e}")
            return {
                "stability": "unknown",
                "explanation": f"Error analyzing correlation stability: {str(e)}"
            }
    
    def _find_cointegrated_pairs(self, price_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Find cointegrated pairs of stocks (suitable for pair trading).
        
        Args:
            price_df: DataFrame with price data
            
        Returns:
            List of cointegrated pairs
        """
        try:
            from statsmodels.tsa.stattools import coint
            
            symbols = price_df.columns
            n = len(symbols)
            cointegrated_pairs = []
            
            # Check all pairs for cointegration
            for i in range(n):
                for j in range(i+1, n):
                    symbol1 = symbols[i]
                    symbol2 = symbols[j]
                    
                    # Get price series
                    series1 = price_df[symbol1].values
                    series2 = price_df[symbol2].values
                    
                    # Run cointegration test
                    score, pvalue, _ = coint(series1, series2)
                    
                    # Check if cointegrated at given significance level
                    if pvalue < self.analysis_params["coint_significance"]:
                        # Calculate hedge ratio using linear regression
                        model = sm.OLS(series1, sm.add_constant(series2)).fit()
                        hedge_ratio = model.params[1]
                        
                        cointegrated_pairs.append({
                            "symbol1": symbol1,
                            "symbol2": symbol2,
                            "p_value": pvalue,
                            "score": score,
                            "hedge_ratio": hedge_ratio
                        })
            
            # Sort by p-value (most significant first)
            cointegrated_pairs.sort(key=lambda x: x["p_value"])
            
            return cointegrated_pairs
            
        except Exception as e:
            self.logger.error(f"Error finding cointegrated pairs: {e}")
            return []
    
    def _build_correlation_network(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        Build a network representation of correlations for visualization.
        
        Args:
            correlation_matrix: Correlation matrix
            
        Returns:
            Dictionary with network representation
        """
        try:
            # Create network nodes (symbols)
            symbols = correlation_matrix.columns
            nodes = [{"id": symbol, "name": symbol} for symbol in symbols]
            
            # Create edges for correlations above threshold
            threshold = self.analysis_params["network_correlation_threshold"]
            
            edges = []
            for i in range(len(symbols)):
                for j in range(i+1, len(symbols)):
                    corr = correlation_matrix.loc[symbols[i], symbols[j]]
                    
                    if abs(corr) >= threshold:
                        edges.append({
                            "source": symbols[i],
                            "target": symbols[j],
                            "value": corr,
                            "type": "positive" if corr > 0 else "negative"
                        })
            
            # Try to calculate network metrics
            try:
                import networkx as nx
                
                # Create NetworkX graph
                G = nx.Graph()
                
                # Add nodes
                for node in nodes:
                    G.add_node(node["id"])
                
                # Add edges (use absolute correlation as weight)
                for edge in edges:
                    G.add_edge(edge["source"], edge["target"], weight=abs(edge["value"]))
                
                # Calculate centrality metrics
                degree_centrality = nx.degree_centrality(G)
                betweenness_centrality = nx.betweenness_centrality(G)
                
                # Update nodes with centrality
                for node in nodes:
                    node["degree_centrality"] = degree_centrality.get(node["id"], 0)
                    node["betweenness_centrality"] = betweenness_centrality.get(node["id"], 0)
                
                # Find communities (clusters)
                try:
                    from community import best_partition
                    communities = best_partition(G)
                    
                    # Add community to nodes
                    for node in nodes:
                        node["community"] = communities.get(node["id"], 0)
                    
                except ImportError:
                    # If community detection fails, use degree as community
                    for node in nodes:
                        node["community"] = 0
                
                # Find central nodes for each community
                community_centrals = {}
                for node in nodes:
                    comm = node["community"]
                    centrality = node["betweenness_centrality"]
                    
                    if comm not in community_centrals or centrality > community_centrals[comm]["centrality"]:
                        community_centrals[comm] = {
                            "symbol": node["id"],
                            "centrality": centrality
                        }
                
                # Format for output
                community_representatives = [
                    {"community": comm, "central_symbol": data["symbol"]}
                    for comm, data in community_centrals.items()
                ]
                
            except ImportError:
                # If NetworkX unavailable, skip network metrics
                for node in nodes:
                    node["degree_centrality"] = 0
                    node["betweenness_centrality"] = 0
                    node["community"] = 0
                
                community_representatives = []
            
            return {
                "nodes": nodes,
                "edges": edges,
                "community_representatives": community_representatives
            }
            
        except Exception as e:
            self.logger.error(f"Error building correlation network: {e}")
            return {
                "nodes": [],
                "edges": [],
                "community_representatives": []
            }
    
    def _generate_correlation_implications(self, avg_correlation: float, 
                                        high_pairs: List[Dict[str, Any]],
                                        inverse_pairs: List[Dict[str, Any]],
                                        diversifiers: List[Dict[str, Any]],
                                        clusters: List[List[str]],
                                        cointegrated_pairs: List[Dict[str, Any]]) -> List[str]:
        """
        Generate trading implications based on correlation analysis.
        
        Args:
            avg_correlation: Average correlation
            high_pairs: Highly correlated pairs
            inverse_pairs: Inversely correlated pairs
            diversifiers: Potential diversifiers
            clusters: Correlation clusters
            cointegrated_pairs: Cointegrated pairs
            
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
        if cointegrated_pairs:
            top_pair = cointegrated_pairs[0]
            implications.append(f"Cointegrated pair {top_pair['symbol1']} and {top_pair['symbol2']} (p-value: {top_pair['p_value']:.4f}) suggests potential statistical arbitrage opportunity.")
        elif high_pairs:
            top_pair = high_pairs[0]
            implications.append(f"Strong correlation between {top_pair['symbol1']} and {top_pair['symbol2']} ({top_pair['correlation']:.2f}) suggests potential pair trading opportunities, though cointegration testing is recommended.")
        
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
                                    stability: Dict[str, Any],
                                    cointegrated_pairs: List[Dict[str, Any]]) -> str:
        """
        Generate a concise summary of the correlation analysis.
        
        Args:
            avg_correlation: Average correlation
            high_pairs: Highly correlated pairs
            inverse_pairs: Inversely correlated pairs
            diversifiers: Potential diversifiers
            clusters: Correlation clusters
            stability: Correlation stability analysis
            cointegrated_pairs: Cointegrated pairs
            
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
        
        # Cointegrated pairs
        if cointegrated_pairs:
            top_coint = cointegrated_pairs[0]
            summary_parts.append(f"{top_coint['symbol1']} and {top_coint['symbol2']} show statistically significant cointegration (p-value: {top_coint['p_value']:.4f}), suggesting potential for statistical arbitrage.")
        
        # Inversely correlated pairs
        if inverse_pairs:
            top_inverse = inverse_pairs[0]
            summary_parts.append(f"The most inversely correlated pair is {top_inverse['symbol1']} and {top_inverse['symbol2']} ({top_inverse['correlation']:.2f}), potentially useful for hedging.")
        
        # Diversifiers
        if diversifiers:
            top_div = diversifiers[0]
            summary_parts.append(f"{top_div['symbol']} shows the lowest average correlation ({top_div['avg_correlation']:.2f}) and may provide the best diversification benefits.")
        
        # Clusters
        if len(clusters) > 1:
            summary_parts.append(f"Analysis identified {len(clusters)} distinct correlation clusters that can be used for diversified portfolio construction.")
        
        # Trading implications
        if cointegrated_pairs:
            summary_parts.append("Statistical arbitrage opportunities exist with the identified cointegrated pairs.")
        elif avg_correlation > 0.7:
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
    
    def analyze_beta(self, symbol: str, benchmark: str = "NIFTY", exchange: str = "NSE",
                   timeframe: str = "day", days: int = 252) -> Dict[str, Any]:
        """
        Analyze beta (market sensitivity) of a symbol relative to a benchmark.
        
        Args:
            symbol: Stock symbol
            benchmark: Benchmark symbol (default: NIFTY)
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with beta analysis
        """
        try:
            self.logger.info(f"Analyzing beta for {symbol} relative to {benchmark}")
            
            # Get price data for symbol and benchmark
            symbol_data = self._get_market_data(symbol, exchange, timeframe, days)
            benchmark_data = self._get_market_data(benchmark, exchange, timeframe, days)
            
            if not symbol_data or not benchmark_data:
                return {
                    "status": "error",
                    "error": "Insufficient data for symbol or benchmark"
                }
            
            # Convert to DataFrames
            symbol_df = pd.DataFrame(symbol_data)[["timestamp", "close"]].sort_values("timestamp")
            benchmark_df = pd.DataFrame(benchmark_data)[["timestamp", "close"]].sort_values("timestamp")
            
            # Set timestamp as index
            symbol_df = symbol_df.set_index("timestamp")
            benchmark_df = benchmark_df.set_index("timestamp")
            
            # Calculate returns
            symbol_returns = symbol_df["close"].pct_change().dropna()
            benchmark_returns = benchmark_df["close"].pct_change().dropna()
            
            # Align the series
            symbol_returns, benchmark_returns = symbol_returns.align(benchmark_returns, join='inner')
            
            if len(symbol_returns) < 30:
                return {
                    "status": "error",
                    "error": "Insufficient overlapping data for beta calculation"
                }
            
            # Calculate beta using regression
            model = sm.OLS(symbol_returns, sm.add_constant(benchmark_returns)).fit()
            beta = model.params[1]
            alpha = model.params[0]
            r_squared = model.rsquared
            
            # Calculate correlation
            correlation = symbol_returns.corr(benchmark_returns)
            
            # Calculate rolling beta (30-day window)
            rolling_window = min(30, len(symbol_returns) // 3)
            rolling_results = []
            
            # Create DataFrame with both returns
            combined_returns = pd.DataFrame({
                "symbol": symbol_returns,
                "benchmark": benchmark_returns
            })
            
            # Calculate rolling beta
            for i in range(rolling_window, len(combined_returns) + 1):
                window_data = combined_returns.iloc[i - rolling_window:i]
                
                if len(window_data) < rolling_window:
                    continue
                
                try:
                    window_model = sm.OLS(
                        window_data["symbol"],
                        sm.add_constant(window_data["benchmark"])
                    ).fit()
                    
                    window_beta = window_model.params[1]
                    window_alpha = window_model.params[0]
                    window_r_squared = window_model.rsquared
                    
                    rolling_results.append({
                        "date": combined_returns.index[i-1],
                        "beta": window_beta,
                        "alpha": window_alpha,
                        "r_squared": window_r_squared
                    })
                except:
                    continue
            
            # Calculate beta stability
            if rolling_results:
                beta_values = [result["beta"] for result in rolling_results]
                beta_std = np.std(beta_values)
                beta_range = max(beta_values) - min(beta_values)
                
                # Determine stability
                if beta_std > 0.3:
                    beta_stability = "unstable"
                elif beta_std > 0.15:
                    beta_stability = "moderately_stable"
                else:
                    beta_stability = "stable"
            else:
                beta_std = None
                beta_range = None
                beta_stability = "unknown"
            
            # Classify beta magnitude
            if beta > 1.5:
                beta_category = "very_high"
            elif beta > 1.2:
                beta_category = "high"
            elif beta > 0.8:
                beta_category = "moderate"
            elif beta > 0.5:
                beta_category = "low"
            else:
                beta_category = "very_low"
            
            # Generate summary
            summary = self._generate_beta_summary(
                symbol, benchmark, beta, alpha, r_squared,
                correlation, beta_stability, beta_std
            )
            
            return {
                "status": "success",
                "symbol": symbol,
                "benchmark": benchmark,
                "beta": beta,
                "alpha": alpha,
                "r_squared": r_squared,
                "correlation": correlation,
                "beta_category": beta_category,
                "beta_stability": beta_stability,
                "beta_std": beta_std,
                "beta_range": beta_range,
                "rolling_data": rolling_results,
                "analysis_summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing beta: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_beta_summary(self, symbol: str, benchmark: str, beta: float, 
                            alpha: float, r_squared: float, correlation: float,
                            stability: str, beta_std: Optional[float]) -> str:
        """
        Generate a summary of the beta analysis.
        
        Args:
            symbol: Stock symbol
            benchmark: Benchmark symbol
            beta: Beta value
            alpha: Alpha value
            r_squared: R-squared value
            correlation: Correlation value
            stability: Beta stability
            beta_std: Standard deviation of beta
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Basic beta description
        if beta > 1.5:
            summary_parts.append(f"{symbol} has a very high beta of {beta:.2f} relative to {benchmark}, indicating significantly higher volatility than the market.")
        elif beta > 1.2:
            summary_parts.append(f"{symbol} has a high beta of {beta:.2f} relative to {benchmark}, indicating higher volatility than the market.")
        elif beta > 0.8:
            summary_parts.append(f"{symbol} has a moderate beta of {beta:.2f} relative to {benchmark}, indicating similar volatility to the market.")
        elif beta > 0.5:
            summary_parts.append(f"{symbol} has a low beta of {beta:.2f} relative to {benchmark}, indicating lower volatility than the market.")
        else:
            summary_parts.append(f"{symbol} has a very low beta of {beta:.2f} relative to {benchmark}, indicating significantly lower volatility than the market.")
        
        # Alpha
        annualized_alpha = alpha * 252  # Assuming daily returns
        if annualized_alpha > 0.05:  # 5% annualized alpha
            summary_parts.append(f"The stock shows positive alpha of {annualized_alpha:.2%} (annualized), suggesting outperformance relative to its market risk.")
        elif annualized_alpha < -0.05:  # -5% annualized alpha
            summary_parts.append(f"The stock shows negative alpha of {annualized_alpha:.2%} (annualized), suggesting underperformance relative to its market risk.")
        else:
            summary_parts.append(f"The stock shows minimal alpha of {annualized_alpha:.2%} (annualized), suggesting performance in line with its market risk.")
        
        # R-squared
        if r_squared > 0.7:
            summary_parts.append(f"High R-squared of {r_squared:.2f} indicates that market movements explain a large portion of the stock's returns.")
        elif r_squared > 0.3:
            summary_parts.append(f"Moderate R-squared of {r_squared:.2f} indicates that market movements explain some of the stock's returns.")
        else:
            summary_parts.append(f"Low R-squared of {r_squared:.2f} indicates that market movements explain little of the stock's returns.")
        
        # Beta stability
        if stability == "stable":
            summary_parts.append(f"The beta has been stable over time with standard deviation of {beta_std:.2f}.")
        elif stability == "moderately_stable":
            summary_parts.append(f"The beta has been moderately stable over time with standard deviation of {beta_std:.2f}.")
        elif stability == "unstable":
            summary_parts.append(f"The beta has been unstable over time with standard deviation of {beta_std:.2f}.")
        
        # Trading implications
        if beta > 1.2 and stability != "unstable":
            summary_parts.append("Trading implications: Suitable for bullish market views, but consider position sizing due to higher volatility.")
        elif beta < 0.8 and stability != "unstable":
            summary_parts.append("Trading implications: Potential defensive play during market downturns, but may underperform in strong bull markets.")
        elif stability == "unstable":
            summary_parts.append("Trading implications: Caution advised due to unstable market relationship; monitor beta trends closely.")
        else:
            summary_parts.append("Trading implications: Market-like performance expected; focus on company-specific factors for alpha generation.")
        
        return " ".join(summary_parts)
    
    def analyze_sector_correlations(self, sector: str, exchange: str = "NSE", 
                                  timeframe: str = "day", days: int = 63) -> Dict[str, Any]:
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
    
    def analyze_multi_timeframe_correlation(self, symbol: str, benchmark: str = "NIFTY", 
                                         exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze correlation across multiple timeframes.
        
        Args:
            symbol: Stock symbol
            benchmark: Benchmark symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with multi-timeframe correlation analysis
        """
        try:
            self.logger.info(f"Analyzing multi-timeframe correlation between {symbol} and {benchmark}")
            
            # Define timeframes to analyze
            timeframes = [
                {"name": "intraday", "timeframe": "5min", "days": 5},
                {"name": "daily", "timeframe": "day", "days": 63},
                {"name": "weekly", "timeframe": "day", "days": 252},
                {"name": "monthly", "timeframe": "day", "days": 504}
            ]
            
            # Analyze correlation for each timeframe
            timeframe_results = []
            
            for tf in timeframes:
                # For weekly and monthly, we still use daily data but calculate weekly/monthly returns
                data1 = self._get_market_data(symbol, exchange, tf["timeframe"], tf["days"])
                data2 = self._get_market_data(benchmark, exchange, tf["timeframe"], tf["days"])
                
                if not data1 or not data2:
                    timeframe_results.append({
                        "timeframe": tf["name"],
                        "correlation": None,
                        "beta": None,
                        "error": "Insufficient data"
                    })
                    continue
                
                # Convert to DataFrames
                df1 = pd.DataFrame(data1)[["timestamp", "close"]].sort_values("timestamp")
                df2 = pd.DataFrame(data2)[["timestamp", "close"]].sort_values("timestamp")
                
                # Set timestamp as index
                df1 = df1.set_index("timestamp")
                df2 = df2.set_index("timestamp")
                
                # Calculate returns based on timeframe
                if tf["name"] == "intraday":
                    returns1 = df1["close"].pct_change().dropna()
                    returns2 = df2["close"].pct_change().dropna()
                elif tf["name"] == "daily":
                    returns1 = df1["close"].pct_change().dropna()
                    returns2 = df2["close"].pct_change().dropna()
                elif tf["name"] == "weekly":
                    # Resample to weekly returns
                    returns1 = df1["close"].resample('W').last().pct_change().dropna()
                    returns2 = df2["close"].resample('W').last().pct_change().dropna()
                else:  # monthly
                    # Resample to monthly returns
                    returns1 = df1["close"].resample('M').last().pct_change().dropna()
                    returns2 = df2["close"].resample('M').last().pct_change().dropna()
                
                # Align the series
                returns1, returns2 = returns1.align(returns2, join='inner')
                
                if len(returns1) < 10:  # Need at least 10 data points
                    timeframe_results.append({
                        "timeframe": tf["name"],
                        "correlation": None,
                        "beta": None,
                        "error": "Insufficient overlapping data"
                    })
                    continue
                
                # Calculate correlation
                correlation = returns1.corr(returns2)
                
                # Calculate beta
                try:
                    model = sm.OLS(returns1, sm.add_constant(returns2)).fit()
                    beta = model.params[1]
                    r_squared = model.rsquared
                except:
                    beta = None
                    r_squared = None
                
                timeframe_results.append({
                    "timeframe": tf["name"],
                    "correlation": correlation,
                    "beta": beta,
                    "r_squared": r_squared,
                    "data_points": len(returns1)
                })
            
            # Analyze consistency across timeframes
            correlations = [r["correlation"] for r in timeframe_results if r["correlation"] is not None]
            
            if len(correlations) >= 2:
                correlation_range = max(correlations) - min(correlations)
                correlation_std = np.std(correlations)
                
                # Determine consistency
                consistency = "consistent"
                if correlation_range > 0.3:
                    consistency = "inconsistent"
                elif correlation_range > 0.15:
                    consistency = "moderately_consistent"
            else:
                correlation_range = None
                correlation_std = None
                consistency = "unknown"
            
            # Generate summary
            summary = self._generate_multi_timeframe_summary(
                symbol, benchmark, timeframe_results, consistency
            )
            
            return {
                "status": "success",
                "symbol": symbol,
                "benchmark": benchmark,
                "timeframe_results": timeframe_results,
                "consistency": consistency,
                "correlation_range": correlation_range,
                "correlation_std": correlation_std,
                "analysis_summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing multi-timeframe correlation: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_multi_timeframe_summary(self, symbol: str, benchmark: str, 
                                       timeframe_results: List[Dict[str, Any]],
                                       consistency: str) -> str:
        """
        Generate a summary of multi-timeframe correlation analysis.
        
        Args:
            symbol: Stock symbol
            benchmark: Benchmark symbol
            timeframe_results: Results for each timeframe
            consistency: Consistency assessment
            
        Returns:
            Summary string
        """
        summary_parts = []
        
        # Start with overall assessment
        valid_results = [r for r in timeframe_results if r.get("correlation") is not None]
        
        if not valid_results:
            return "Insufficient data to analyze multi-timeframe correlation."
        
        # Find timeframes with highest and lowest correlations
        valid_results.sort(key=lambda x: x.get("correlation", -999))
        lowest_corr = valid_results[0]
        highest_corr = valid_results[-1]
        
        if consistency == "consistent":
            summary_parts.append(f"{symbol} shows consistent correlation with {benchmark} across different timeframes, suggesting a stable relationship.")
        elif consistency == "moderately_consistent":
            summary_parts.append(f"{symbol} shows moderately consistent correlation with {benchmark} across different timeframes.")
        else:
            summary_parts.append(f"{symbol} shows inconsistent correlation with {benchmark} across different timeframes, with strongest relationship in {highest_corr['timeframe']} timeframe ({highest_corr['correlation']:.2f}) and weakest in {lowest_corr['timeframe']} timeframe ({lowest_corr['correlation']:.2f}).")
        
        # Analyze each timeframe
        for result in timeframe_results:
            if result.get("correlation") is None:
                continue
                
            timeframe = result["timeframe"]
            correlation = result["correlation"]
            beta = result.get("beta")
            
            if timeframe == "intraday":
                if correlation > 0.7:
                    summary_parts.append(f"Strong intraday correlation ({correlation:.2f}) suggests the assets respond similarly to short-term market movements.")
                elif correlation < 0.3:
                    summary_parts.append(f"Weak intraday correlation ({correlation:.2f}) suggests limited short-term market relationship.")
            
            if timeframe == "daily":
                if beta is not None and beta > 1.2:
                    summary_parts.append(f"High daily beta ({beta:.2f}) indicates amplified response to daily market movements.")
                elif beta is not None and beta < 0.8:
                    summary_parts.append(f"Low daily beta ({beta:.2f}) suggests reduced sensitivity to daily market fluctuations.")
            
            if timeframe == "weekly" or timeframe == "monthly":
                if correlation > correlation_value_for_timeframe("daily", timeframe_results) + 0.2:
                    summary_parts.append(f"Correlation strengthens in {timeframe} timeframe ({correlation:.2f}), suggesting stronger long-term relationship than short-term.")
                elif correlation < correlation_value_for_timeframe("daily", timeframe_results) - 0.2:
                    summary_parts.append(f"Correlation weakens in {timeframe} timeframe ({correlation:.2f}), suggesting weaker long-term relationship than short-term.")
        
        # Trading implications
        if is_higher_correlation_in_longer_timeframes(timeframe_results):
            summary_parts.append("Trading implications: Consider longer holding periods to benefit from the stronger long-term correlation.")
        elif is_higher_correlation_in_shorter_timeframes(timeframe_results):
            summary_parts.append("Trading implications: Short-term trading strategies may be more appropriate given the stronger short-term correlation.")
        elif consistency == "inconsistent":
            summary_parts.append("Trading implications: Adapt strategies to the specific timeframe being traded, as correlation varies significantly across timeframes.")
        else:
            summary_parts.append("Trading implications: Consistent correlation across timeframes allows for flexible holding periods.")
        
        return " ".join(summary_parts)
    
    def analyze_correlation_index(self, symbols: List[str], exchange: str = "NSE",
                               timeframe: str = "day", days: int = 252) -> Dict[str, Any]:
        """
        Create a correlation index that measures overall correlation in a group of symbols.
        
        Args:
            symbols: List of stock symbols
            exchange: Stock exchange
            timeframe: Data timeframe
            days: Number of days to analyze
            
        Returns:
            Dictionary with correlation index analysis
        """
        try:
            self.logger.info(f"Analyzing correlation index for {len(symbols)} symbols")
            
            if len(symbols) < 5:
                return {
                    "status": "error",
                    "error": "Need at least 5 symbols for correlation index"
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
            
            valid_symbols = list(price_data.keys())
            if len(valid_symbols) < 5:
                return {
                    "status": "error",
                    "error": "Insufficient valid data for correlation index"
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
            
            # Calculate rolling average correlation
            window = min(30, len(returns_df) // 3)  # Reasonable window size
            
            # Initialize list to store rolling correlation index values
            rolling_corr_index = []
            dates = returns_df.index[window-1:]
            
            # Calculate rolling correlation index
            for i in range(window, len(returns_df) + 1):
                window_returns = returns_df.iloc[i-window:i]
                
                # Calculate correlation matrix
                corr_matrix = window_returns.corr()
                
                # Calculate average of off-diagonal elements (exclude self-correlations)
                corr_sum = 0
                count = 0
                
                for j in range(len(valid_symbols)):
                    for k in range(j+1, len(valid_symbols)):
                        corr_sum += corr_matrix.iloc[j, k]
                        count += 1
                
                avg_corr = corr_sum / count if count > 0 else 0
                
                rolling_corr_index.append({
                    "date": dates[i-window],
                    "correlation_index": avg_corr
                })
            
            # Calculate current correlation index (latest window)
            current_corr_index = rolling_corr_index[-1]["correlation_index"] if rolling_corr_index else None
            
            # Calculate historical statistics
            if rolling_corr_index:
                corr_values = [entry["correlation_index"] for entry in rolling_corr_index]
                avg_corr_index = sum(corr_values) / len(corr_values)
                std_corr_index = np.std(corr_values)
                min_corr_index = min(corr_values)
                max_corr_index = max(corr_values)
                
                # Calculate percentile of current value
                current_percentile = stats.percentileofscore(corr_values, current_corr_index)
                
                # Determine correlation regime
                if current_corr_index > np.percentile(corr_values, 80):
                    correlation_regime = "high_correlation"
                elif current_corr_index < np.percentile(corr_values, 20):
                    correlation_regime = "low_correlation"
                else:
                    correlation_regime = "normal_correlation"
            else:
                avg_corr_index = None
                std_corr_index = None
                min_corr_index = None
                max_corr_index = None
                current_percentile = None
                correlation_regime = "unknown"
            
            # Generate summary
            summary = self._generate_correlation_index_summary(
                current_corr_index, avg_corr_index, current_percentile,
                correlation_regime
            )
            
            return {
                "status": "success",
                "symbols_analyzed": valid_symbols,
                "current_correlation_index": current_corr_index,
                "average_correlation_index": avg_corr_index,
                "min_correlation_index": min_corr_index,
                "max_correlation_index": max_corr_index,
                "standard_deviation": std_corr_index,
                "current_percentile": current_percentile,
                "correlation_regime": correlation_regime,
                "rolling_correlation_index": rolling_corr_index,
                "analysis_summary": summary
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation index: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _generate_correlation_index_summary(self, current_index: float, 
                                         avg_index: float, percentile: float,
                                         regime: str) -> str:
        """
        Generate a summary of correlation index analysis.
        
        Args:
            current_index: Current correlation index
            avg_index: Average correlation index
            percentile: Current percentile
            regime: Correlation regime
            
        Returns:
            Summary string
        """
        if current_index is None:
            return "Insufficient data to generate correlation index summary."
            
        summary_parts = []
        
        # Current correlation state
        if regime == "high_correlation":
            summary_parts.append(f"The market is currently in a high correlation regime ({current_index:.2f}, {percentile:.0f}th percentile), indicating strong co-movement among assets.")
        elif regime == "low_correlation":
            summary_parts.append(f"The market is currently in a low correlation regime ({current_index:.2f}, {percentile:.0f}th percentile), indicating greater independence among assets.")
        else:
            summary_parts.append(f"The market is currently in a normal correlation regime ({current_index:.2f}, {percentile:.0f}th percentile).")
        
        # Compare to historical average
        if current_index > avg_index * 1.2:
            summary_parts.append(f"Current correlation is significantly higher than the historical average of {avg_index:.2f}.")
        elif current_index < avg_index * 0.8:
            summary_parts.append(f"Current correlation is significantly lower than the historical average of {avg_index:.2f}.")
        else:
            summary_parts.append(f"Current correlation is close to the historical average of {avg_index:.2f}.")
        
        # Trading implications
        if regime == "high_correlation":
            summary_parts.append("Trading implications: In high correlation environments, asset selection provides less diversification benefit. Focus on market direction and sector rotation strategies.")
        elif regime == "low_correlation":
            summary_parts.append("Trading implications: In low correlation environments, asset selection becomes more important. Focus on stock-specific factors and pair trading opportunities.")
        
        # Market interpretation
        if regime == "high_correlation":
            summary_parts.append("High correlation often occurs during market stress or strong trends, when macro factors dominate individual stock characteristics.")
        elif regime == "low_correlation":
            summary_parts.append("Low correlation often occurs in calm or range-bound markets, when company-specific factors have greater influence.")
        
        return " ".join(summary_parts)

# Helper functions for multi-timeframe analysis
def correlation_value_for_timeframe(timeframe: str, results: List[Dict[str, Any]]) -> float:
    """Get correlation value for a specific timeframe."""
    for result in results:
        if result.get("timeframe") == timeframe and result.get("correlation") is not None:
            return result["correlation"]
    return 0.0

def is_higher_correlation_in_longer_timeframes(results: List[Dict[str, Any]]) -> bool:
    """Check if correlation is higher in longer timeframes."""
    timeframe_order = {"intraday": 0, "daily": 1, "weekly": 2, "monthly": 3}
    valid_results = [(r["timeframe"], r["correlation"]) for r in results if r.get("correlation") is not None]
    
    if len(valid_results) < 2:
        return False
    
    # Sort by timeframe
    valid_results.sort(key=lambda x: timeframe_order.get(x[0], 0))
    
    # Check if correlation generally increases
    increasing = True
    for i in range(1, len(valid_results)):
        if valid_results[i][1] < valid_results[i-1][1]:
            increasing = False
            break
    
    return increasing

def is_higher_correlation_in_shorter_timeframes(results: List[Dict[str, Any]]) -> bool:
    """Check if correlation is higher in shorter timeframes."""
    timeframe_order = {"intraday": 0, "daily": 1, "weekly": 2, "monthly": 3}
    valid_results = [(r["timeframe"], r["correlation"]) for r in results if r.get("correlation") is not None]
    
    if len(valid_results) < 2:
        return False
    
    # Sort by timeframe
    valid_results.sort(key=lambda x: timeframe_order.get(x[0], 0))
    
    # Check if correlation generally decreases
    decreasing = True
    for i in range(1, len(valid_results)):
        if valid_results[i][1] > valid_results[i-1][1]:
            decreasing = False
            break
    
    return decreasing