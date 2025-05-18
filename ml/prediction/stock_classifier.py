# ml/prediction/stock_classifier.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pickle
import base64
import matplotlib.pyplot as plt
import io
import os

class StockClassifier:
    """
    Classify stocks based on various characteristics to determine optimal trading strategies.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the stock classifier.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'analysis_period': 252,  # 1 trading year
            'short_term_period': 20,  # 1 trading month
            'num_clusters': 5,  # Number of stock clusters
            'min_data_points': 60,  # Minimum data points required for classification
            'refresh_days': 30,  # Days after which to refresh classification
            'volatility_threshold': 0.02,  # 2% daily volatility threshold
            'stock_classes': {
                'trend_follower': 'Follows trends with significant momentum',
                'mean_reverter': 'Tends to revert to mean after deviations',
                'high_volatility': 'High volatility with significant price swings',
                'low_volatility': 'Low volatility with stable price movements',
                'swing_trader': 'Alternates between up and down moves in a range',
                'breakout_candidate': 'Forms consolidation patterns before breakouts',
                'range_bound': 'Trades within a defined price range',
                'high_volume': 'Trades with high liquidity and volume',
                'low_volume': 'Trades with low liquidity and thin volume',
                'news_driven': 'Price action highly correlated with news events'
            },
            'strategy_mapping': {
                'trend_follower': ['moving_average', 'macd', 'adx_trend'],
                'mean_reverter': ['rsi_reversal', 'bollinger_bands', 'stochastic_reversal'],
                'high_volatility': ['atr_trailing_stop', 'options_strategies', 'volatility_breakout'],
                'low_volatility': ['value_accumulation', 'dividend_capture', 'covered_call'],
                'swing_trader': ['swing_high_low', 'support_resistance', 'fibonacci_retracement'],
                'breakout_candidate': ['volume_breakout', 'triangle_breakout', 'resistance_breakthrough'],
                'range_bound': ['channel_trading', 'range_bounce', 'overbought_oversold'],
                'high_volume': ['liquidity_provision', 'volume_profile', 'market_depth_analysis'],
                'low_volume': ['cautious_entry', 'limit_orders', 'wider_stops'],
                'news_driven': ['event_trading', 'news_sentiment', 'earnings_plays']
            },
            'timeframe_mapping': {
                'trend_follower': 'daily',
                'mean_reverter': 'hourly',
                'high_volatility': 'intraday',
                'low_volatility': 'daily',
                'swing_trader': 'daily',
                'breakout_candidate': 'hourly',
                'range_bound': 'hourly',
                'high_volume': 'intraday',
                'low_volume': 'daily',
                'news_driven': 'hourly'
            }
        }
        
        # Models
        self.kmeans_model = None
        self.classifier_model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)  # For visualization
    
    def set_config(self, config):
        """
        Set classifier configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated stock classifier configuration: {self.config}")
    
    def get_stock_data(self, symbol, exchange, days=None):
        """
        Get historical stock data for classification.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            days (int): Number of days of data to retrieve
            
        Returns:
            DataFrame: Stock data
        """
        days = days or self.config['analysis_period']
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days * 1.5)  # Add buffer for weekends/holidays
        
        # Query database
        query = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': 'day',
            'timestamp': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        # Sort by timestamp
        cursor = self.db.market_data_collection.find(query).sort('timestamp', 1)
        
        # Convert to DataFrame
        market_data = pd.DataFrame(list(cursor))
        
        if len(market_data) < self.config['min_data_points']:
            self.logger.warning(f"Insufficient data for {symbol} {exchange}: {len(market_data)} < {self.config['min_data_points']}")
            return None
        
        # Take only the last 'days' days
        if len(market_data) > days:
            market_data = market_data.tail(days)
        
        self.logger.info(f"Retrieved {len(market_data)} days of data for {symbol} {exchange}")
        
        return market_data
    
    def extract_features(self, market_data):
        """
        Extract classification features from market data.
        
        Args:
            market_data (DataFrame): Market data
            
        Returns:
            dict: Feature dictionary
        """
        if market_data is None or len(market_data) < self.config['min_data_points']:
            return None
            
        # Extract basic data
        closes = market_data['close'].values
        highs = market_data['high'].values
        lows = market_data['low'].values
        volumes = market_data['volume'].values if 'volume' in market_data.columns else np.zeros_like(closes)
        
        # Calculate returns
        returns = np.diff(closes) / closes[:-1]
        
        # Create feature dictionary
        features = {}
        
        # Volatility features
        features['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        features['daily_volatility'] = np.std(returns)
        features['high_low_volatility'] = np.mean((highs - lows) / closes)
        
        # Trend features
        features['trend_strength'] = self._calculate_trend_strength(closes)
        features['momentum'] = self._calculate_momentum(returns)
        
        # Mean reversion features
        features['mean_reversion'] = self._calculate_mean_reversion(returns)
        features['autocorrelation'] = self._calculate_autocorrelation(returns)
        
        # Volume features
        if not np.all(volumes == 0):
            features['volume_volatility'] = np.std(volumes) / np.mean(volumes)
            features['volume_trend'] = self._calculate_trend_strength(volumes)
            features['price_volume_correlation'] = np.corrcoef(closes[:-1], volumes[:-1])[0, 1]
        else:
            features['volume_volatility'] = 0
            features['volume_trend'] = 0
            features['price_volume_correlation'] = 0
        
        # Range features
        features['range_bound'] = self._calculate_range_bound(closes)
        
        # Breakout features
        features['breakout_potential'] = self._calculate_breakout_potential(closes, volumes)
        
        # Short-term features
        short_term = self.config['short_term_period']
        if len(closes) > short_term:
            recent_returns = returns[-short_term:]
            features['recent_volatility'] = np.std(recent_returns) * np.sqrt(252)
            features['recent_trend'] = self._calculate_trend_strength(closes[-short_term:])
            features['volatility_change'] = features['recent_volatility'] / features['volatility'] - 1
        else:
            features['recent_volatility'] = features['volatility']
            features['recent_trend'] = features['trend_strength']
            features['volatility_change'] = 0
        
        # News sensitivity features
        features['news_sensitivity'] = self._calculate_news_sensitivity(market_data)
        
        return features
    
    def _calculate_trend_strength(self, data):
        """
        Calculate trend strength using linear regression.
        
        Args:
            data (array): Price or volume data
            
        Returns:
            float: Trend strength (-1 to 1)
        """
        try:
            x = np.arange(len(data))
            y = data
            
            if len(x) < 2:
                return 0
                
            # Calculate linear regression
            slope, _, r_value, _, _ = np.polyfit(x, y, 1, full=True)[0:5]
            
            # Normalize slope by average value
            norm_slope = slope * len(data) / np.mean(data)
            
            # Combine slope direction and r-squared for strength
            trend_strength = np.sign(norm_slope) * r_value ** 2
            
            return trend_strength
            
        except Exception as e:
            self.logger.warning(f"Error calculating trend strength: {e}")
            return 0
    
    def _calculate_momentum(self, returns):
        """
        Calculate momentum using return autocorrelation.
        
        Args:
            returns (array): Return data
            
        Returns:
            float: Momentum score (-1 to 1)
        """
        try:
            if len(returns) < 10:
                return 0
                
            # Calculate autocorrelation with lag 1
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Smooth with recent returns direction
            recent_return = np.mean(returns[-5:]) if len(returns) >= 5 else np.mean(returns)
            
            momentum = (autocorr + np.sign(recent_return)) / 2
            
            return momentum
            
        except Exception as e:
            self.logger.warning(f"Error calculating momentum: {e}")
            return 0
    
    def _calculate_mean_reversion(self, returns):
        """
        Calculate mean reversion tendency.
        
        Args:
            returns (array): Return data
            
        Returns:
            float: Mean reversion score (0 to 1)
        """
        try:
            if len(returns) < 10:
                return 0.5
                
            # Calculate autocorrelation with lag 1
            autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
            
            # Negative autocorrelation indicates mean reversion
            mean_reversion = 1 - (autocorr + 1) / 2
            
            return mean_reversion
            
        except Exception as e:
            self.logger.warning(f"Error calculating mean reversion: {e}")
            return 0.5
    
    def _calculate_autocorrelation(self, returns):
        """
        Calculate return autocorrelation at multiple lags.
        
        Args:
            returns (array): Return data
            
        Returns:
            float: Autocorrelation score
        """
        try:
            if len(returns) < 20:
                return 0
                
            # Calculate autocorrelation at different lags
            lags = [1, 2, 3, 5]
            autocorrs = []
            
            for lag in lags:
                if len(returns) <= lag:
                    continue
                    
                autocorr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                autocorrs.append(autocorr)
            
            if not autocorrs:
                return 0
                
            # Average autocorrelation
            avg_autocorr = np.mean(autocorrs)
            
            return avg_autocorr
            
        except Exception as e:
            self.logger.warning(f"Error calculating autocorrelation: {e}")
            return 0
    
    def _calculate_range_bound(self, closes):
        """
        Calculate how range-bound a stock is.
        
        Args:
            closes (array): Closing price data
            
        Returns:
            float: Range-bound score (0 to 1)
        """
        try:
            if len(closes) < 20:
                return 0.5
                
            # Calculate price range
            price_range = (np.max(closes) - np.min(closes)) / np.mean(closes)
            
            # Calculate trend strength
            trend_strength = abs(self._calculate_trend_strength(closes))
            
            # Range-bound score is inverse of normalized range and trend strength
            range_score = 1 - min(price_range / 0.3, 1)  # Normalize range to 0-1
            trend_score = 1 - trend_strength
            
            # Combine scores
            range_bound = (range_score + trend_score) / 2
            
            return range_bound
            
        except Exception as e:
            self.logger.warning(f"Error calculating range-bound: {e}")
            return 0.5
    
    def _calculate_breakout_potential(self, closes, volumes):
        """
        Calculate breakout potential based on price consolidation and volume patterns.
        
        Args:
            closes (array): Closing price data
            volumes (array): Volume data
            
        Returns:
            float: Breakout potential score (0 to 1)
        """
        try:
            if len(closes) < 20:
                return 0.5
                
            # Calculate recent volatility (last 10 days vs previous 10 days)
            if len(closes) >= 20:
                recent_vol = np.std(closes[-10:]) / np.mean(closes[-10:])
                prev_vol = np.std(closes[-20:-10]) / np.mean(closes[-20:-10])
                vol_ratio = recent_vol / prev_vol if prev_vol > 0 else 1
            else:
                vol_ratio = 1
            
            # Decreasing volatility indicates consolidation (potential breakout)
            volatility_score = 1 - min(vol_ratio, 1)
            
            # Check for volume pattern (declining volume during consolidation)
            if len(volumes) >= 10 and not np.all(volumes == 0):
                recent_vol_mean = np.mean(volumes[-10:])
                prev_vol_mean = np.mean(volumes[-20:-10]) if len(volumes) >= 20 else recent_vol_mean
                volume_declining = recent_vol_mean < prev_vol_mean
                volume_score = 0.7 if volume_declining else 0.3
            else:
                volume_score = 0.5
            
            # Check for price range narrowing
            if len(closes) >= 20:
                recent_range = (np.max(closes[-10:]) - np.min(closes[-10:])) / np.mean(closes[-10:])
                prev_range = (np.max(closes[-20:-10]) - np.min(closes[-20:-10])) / np.mean(closes[-20:-10])
                range_ratio = recent_range / prev_range if prev_range > 0 else 1
                range_score = 1 - min(range_ratio, 1)
            else:
                range_score = 0.5
            
            # Combine scores (weighted)
            breakout_potential = (volatility_score * 0.4 + volume_score * 0.3 + range_score * 0.3)
            
            return breakout_potential
            
        except Exception as e:
            self.logger.warning(f"Error calculating breakout potential: {e}")
            return 0.5
    
    def _calculate_news_sensitivity(self, market_data):
        """
        Calculate sensitivity to news events.
        
        Args:
            market_data (DataFrame): Market data
            
        Returns:
            float: News sensitivity score (0 to 1)
        """
        try:
            symbol = market_data['symbol'].iloc[0]
            exchange = market_data['exchange'].iloc[0]
            
            if 'timestamp' not in market_data.columns:
                return 0.5
                
            # Get date range from market data
            start_date = market_data['timestamp'].min()
            end_date = market_data['timestamp'].max()
            
            # Get news events for this symbol
            news_query = {
                'symbols': symbol,
                'published_date': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            news_count = self.db.news_collection.count_documents(news_query)
            
            if news_count == 0:
                return 0.5
                
            # Calculate large price moves
            closes = market_data['close'].values
            returns = np.diff(closes) / closes[:-1]
            
            # Define large move threshold (2 standard deviations)
            threshold = np.std(returns) * 2
            large_moves = np.sum(np.abs(returns) > threshold)
            
            # Calculate news-to-moves ratio
            trading_days = len(market_data)
            expected_large_moves = trading_days * 0.05  # Expect 5% of days to have large moves
            
            news_ratio = min(news_count / max(trading_days * 0.1, 1), 3)  # Cap at 3x expected
            move_ratio = min(large_moves / max(expected_large_moves, 1), 3)  # Cap at 3x expected
            
            # Calculate correlation between news days and large move days
            news_sensitivity = (news_ratio + move_ratio) / 6  # Normalize to 0-1
            
            return min(news_sensitivity, 1)
            
        except Exception as e:
            self.logger.warning(f"Error calculating news sensitivity: {e}")
            return 0.5
    
    def classify_stock(self, symbol, exchange, features=None, market_data=None):
        """
        Classify a stock based on its features.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            features (dict): Pre-computed features (optional)
            market_data (DataFrame): Market data (optional)
            
        Returns:
            dict: Classification results
        """
        self.logger.info(f"Classifying stock {symbol} {exchange}")
        
        # Check if classification exists and is recent
        existing = self._get_existing_classification(symbol, exchange)
        
        if existing and not self._should_refresh_classification(existing):
            self.logger.info(f"Using existing classification for {symbol} {exchange}")
            return existing
        
        # Get market data if not provided
        if market_data is None:
            market_data = self.get_stock_data(symbol, exchange)
            
        if market_data is None:
            self.logger.error(f"Insufficient data for {symbol} {exchange}")
            return None
        
        # Extract features if not provided
        if features is None:
            features = self.extract_features(market_data)
            
        if features is None:
            self.logger.error(f"Failed to extract features for {symbol} {exchange}")
            return None
        
        # Run classification
        classification = self._classify_by_rules(features)
        
        # Store additional metadata
        classification['symbol'] = symbol
        classification['exchange'] = exchange
        classification['timestamp'] = datetime.now()
        classification['features'] = features
        
        # Compute recommended strategies and timeframes
        classification['strategies'] = self._get_recommended_strategies(classification['classes'])
        classification['timeframe'] = self._get_recommended_timeframe(classification['classes'])
        
        # Add trading parameters
        classification['parameters'] = self._get_trading_parameters(classification)
        
        # Save classification
        self._save_classification(classification)
        
        self.logger.info(f"Classified {symbol} {exchange} as {classification['primary_class']}")
        
        return classification
    
    def _get_existing_classification(self, symbol, exchange):
        """
        Get existing classification from database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Existing classification or None
        """
        try:
            query = {
                'symbol': symbol,
                'exchange': exchange
            }
            
            classification = self.db.stock_classifications_collection.find_one(
                query, sort=[('timestamp', -1)]
            )
            
            return classification
            
        except Exception as e:
            self.logger.warning(f"Error retrieving classification: {e}")
            return None
    
    def _should_refresh_classification(self, classification):
        """
        Determine if classification should be refreshed.
        
        Args:
            classification (dict): Existing classification
            
        Returns:
            bool: True if should refresh
        """
        if 'timestamp' not in classification:
            return True
            
        timestamp = classification['timestamp']
        age_days = (datetime.now() - timestamp).days
        
        return age_days >= self.config['refresh_days']
    
    def _classify_by_rules(self, features):
        """
        Classify stock using rule-based approach.
        
        Args:
            features (dict): Stock features
            
        Returns:
            dict: Classification results
        """
        # Initialize class scores
        class_scores = {c: 0 for c in self.config['stock_classes'].keys()}
        
        # Rule-based classification
        
        # Trend follower
        if features['trend_strength'] > 0.6:
            class_scores['trend_follower'] += 1
        if features['momentum'] > 0.3:
            class_scores['trend_follower'] += 0.5
        
        # Mean reverter
        if features['mean_reversion'] > 0.6:
            class_scores['mean_reverter'] += 1
        if features['autocorrelation'] < -0.2:
            class_scores['mean_reverter'] += 0.5
        
        # High volatility
        if features['volatility'] > 0.4:  # 40% annualized volatility
            class_scores['high_volatility'] += 1
        if features['daily_volatility'] > self.config['volatility_threshold']:
            class_scores['high_volatility'] += 0.5
        
        # Low volatility
        if features['volatility'] < 0.2:  # 20% annualized volatility
            class_scores['low_volatility'] += 1
        if features['daily_volatility'] < self.config['volatility_threshold'] / 2:
            class_scores['low_volatility'] += 0.5
        
        # Swing trader
        if 0.4 < features['mean_reversion'] < 0.7 and 0.3 < features['range_bound'] < 0.7:
            class_scores['swing_trader'] += 1
        if -0.3 < features['autocorrelation'] < 0.3:
            class_scores['swing_trader'] += 0.5
        
        # Breakout candidate
        if features['breakout_potential'] > 0.7:
            class_scores['breakout_candidate'] += 1
        if features['volatility_change'] < -0.3:  # Decreasing volatility
            class_scores['breakout_candidate'] += 0.5
        
        # Range bound
        if features['range_bound'] > 0.7:
            class_scores['range_bound'] += 1
        if abs(features['trend_strength']) < 0.3:
            class_scores['range_bound'] += 0.5
        
        # High volume
        if features['volume_volatility'] < 0.5 and features['volume_trend'] > 0:
            class_scores['high_volume'] += 1
        
        # Low volume
        if features['volume_volatility'] > 1.0 or features['volume_trend'] < 0:
            class_scores['low_volume'] += 0.5
        
        # News driven
        if features['news_sensitivity'] > 0.7:
            class_scores['news_driven'] += 1
        
        # Normalize scores
        total_score = sum(class_scores.values())
        if total_score > 0:
            class_scores = {k: v / total_score for k, v in class_scores.items()}
        
        # Determine primary class (highest score)
        primary_class = max(class_scores.items(), key=lambda x: x[1])[0]
        
        # Find secondary classes (scores within 80% of highest)
        max_score = class_scores[primary_class]
        threshold = max_score * 0.8
        
        secondary_classes = [
            c for c, s in class_scores.items()
            if s >= threshold and c != primary_class
        ]
        
        # Combine results
        classification = {
            'primary_class': primary_class,
            'secondary_classes': secondary_classes,
            'class_scores': class_scores,
            'classes': [primary_class] + secondary_classes,
            'primary_description': self.config['stock_classes'][primary_class]
        }
        
        return classification
    
    def _get_recommended_strategies(self, classes):
        """
        Get recommended trading strategies based on classification.
        
        Args:
            classes (list): Primary and secondary classes
            
        Returns:
            list: Recommended strategies
        """
        strategies = []
        
        for cls in classes:
            if cls in self.config['strategy_mapping']:
                strategies.extend(self.config['strategy_mapping'][cls])
        
        # Remove duplicates while preserving order
        unique_strategies = []
        for s in strategies:
            if s not in unique_strategies:
                unique_strategies.append(s)
        
        return unique_strategies[:5]  # Return top 5 strategies
    
    def _get_recommended_timeframe(self, classes):
        """
        Get recommended trading timeframe based on classification.
        
        Args:
            classes (list): Primary and secondary classes
            
        Returns:
            str: Recommended timeframe
        """
        timeframes = {}
        
        for cls in classes:
            if cls in self.config['timeframe_mapping']:
                tf = self.config['timeframe_mapping'][cls]
                timeframes[tf] = timeframes.get(tf, 0) + 1
        
        if not timeframes:
            return 'daily'  # Default
            
        # Return most frequent timeframe
        return max(timeframes.items(), key=lambda x: x[1])[0]
    
    def _get_trading_parameters(self, classification):
        """
        Get recommended trading parameters based on classification.
        
        Args:
            classification (dict): Classification results
            
        Returns:
            dict: Trading parameters
        """
        features = classification.get('features', {})
        primary_class = classification.get('primary_class')
        
        # Default parameters
        parameters = {
            'position_size': 'normal',
            'stop_loss_percent': 2.0,
            'take_profit_percent': 4.0,
            'max_holding_period': 10,  # trading days
            'entry_timing': 'market',
            'exit_criteria': 'combined'
        }
        
        # Adjust based on volatility
        if 'daily_volatility' in features:
            volatility = features['daily_volatility']
            parameters['stop_loss_percent'] = max(1.0, min(5.0, volatility * 200))  # 2x daily volatility
            parameters['take_profit_percent'] = max(2.0, min(10.0, volatility * 400))  # 4x daily volatility
        
        # Adjust based on primary class
        if primary_class == 'trend_follower':
            parameters['position_size'] = 'larger'
            parameters['stop_loss_percent'] *= 1.2
            parameters['take_profit_percent'] *= 1.5
            parameters['max_holding_period'] = 20
            parameters['entry_timing'] = 'pullback'
            parameters['exit_criteria'] = 'trend_reversal'
            
        elif primary_class == 'mean_reverter':
            parameters['position_size'] = 'normal'
            parameters['stop_loss_percent'] *= 0.8
            parameters['take_profit_percent'] *= 0.8
            parameters['max_holding_period'] = 5
            parameters['entry_timing'] = 'extreme'
            parameters['exit_criteria'] = 'mean_touch'
            
        elif primary_class == 'high_volatility':
            parameters['position_size'] = 'smaller'
            parameters['stop_loss_percent'] *= 1.5
            parameters['take_profit_percent'] *= 1.5
            parameters['max_holding_period'] = 3
            parameters['entry_timing'] = 'confirmation'
            parameters['exit_criteria'] = 'trailing_stop'
            
        elif primary_class == 'low_volatility':
            parameters['position_size'] = 'larger'
            parameters['stop_loss_percent'] *= 0.7
            parameters['take_profit_percent'] *= 0.7
            parameters['max_holding_period'] = 30
            parameters['entry_timing'] = 'market'
            parameters['exit_criteria'] = 'time_based'
            
        elif primary_class == 'swing_trader':
            parameters['position_size'] = 'normal'
            parameters['stop_loss_percent'] *= 1.0
            parameters['take_profit_percent'] *= 1.0
            parameters['max_holding_period'] = 7
            parameters['entry_timing'] = 'swing_extreme'
            parameters['exit_criteria'] = 'opposite_extreme'
            
        elif primary_class == 'breakout_candidate':
            parameters['position_size'] = 'normal'
            parameters['stop_loss_percent'] *= 0.8
            parameters['take_profit_percent'] *= 1.3
            parameters['max_holding_period'] = 5
            parameters['entry_timing'] = 'breakout'
            parameters['exit_criteria'] = 'parabolic'
            
        elif primary_class == 'range_bound':
            parameters['position_size'] = 'normal'
            parameters['stop_loss_percent'] *= 0.7
            parameters['take_profit_percent'] *= 0.7
            parameters['max_holding_period'] = 5
            parameters['entry_timing'] = 'range_edge'
            parameters['exit_criteria'] = 'opposite_edge'
            
        elif primary_class == 'news_driven':
            parameters['position_size'] = 'smaller'
            parameters['stop_loss_percent'] *= 1.3
            parameters['take_profit_percent'] *= 1.2
            parameters['max_holding_period'] = 2
            parameters['entry_timing'] = 'news_reaction'
            parameters['exit_criteria'] = 'quick_profit'
        
        return parameters
    
    def _save_classification(self, classification):
        """
        Save classification to database.
        
        Args:
            classification (dict): Classification data
            
        Returns:
            str: Classification ID
        """
        try:
            # Insert into database
            result = self.db.stock_classifications_collection.insert_one(classification)
            classification_id = str(result.inserted_id)
            
            self.logger.info(f"Saved classification with ID: {classification_id}")
            
            return classification_id
            
        except Exception as e:
            self.logger.error(f"Error saving classification: {e}")
            return None
    
    def cluster_stocks(self, symbols_list=None, features_list=None):
        """
        Cluster stocks based on features using K-means.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples
            features_list (list): List of feature dictionaries (optional)
            
        Returns:
            dict: Clustering results
        """
        self.logger.info("Clustering stocks based on features")
        
        # Get features for each stock if not provided
        if features_list is None:
            features_list = []
            symbols_with_features = []
            
            if symbols_list is None:
                # Get all active symbols
                cursor = self.db.portfolio_collection.find({
                    'status': 'active'
                })
                
                symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
            
            for symbol, exchange in symbols_list:
                try:
                    # Get market data
                    market_data = self.get_stock_data(symbol, exchange)
                    
                    if market_data is None:
                        continue
                        
                    # Extract features
                    features = self.extract_features(market_data)
                    
                    if features is not None:
                        features_list.append(features)
                        symbols_with_features.append((symbol, exchange))
                        
                except Exception as e:
                    self.logger.warning(f"Error processing {symbol} {exchange}: {e}")
            
            symbols_list = symbols_with_features
        
        if not features_list or len(features_list) < 3:
            self.logger.error("Insufficient data for clustering")
            return None
        
        # Convert features to matrix
        feature_names = sorted(features_list[0].keys())
        X = np.zeros((len(features_list), len(feature_names)))
        
        for i, features in enumerate(features_list):
            for j, name in enumerate(feature_names):
                X[i, j] = features.get(name, 0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Run K-means clustering
        n_clusters = min(self.config['num_clusters'], len(X_scaled) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Store model
        self.kmeans_model = kmeans
        
        # Dimensionality reduction for visualization
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Analyze clusters
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            cluster_symbols = [symbols_list[i] for i in cluster_indices]
            
            # Calculate average features for this cluster
            cluster_features = X[cluster_indices]
            avg_features = {}
            
            for j, name in enumerate(feature_names):
                avg_features[name] = float(np.mean(cluster_features[:, j]))
            
            # Determine cluster characteristics
            primary_class = self._classify_by_rules(avg_features)['primary_class']
            
            cluster_analysis[cluster_id] = {
                'count': len(cluster_indices),
                'symbols': cluster_symbols,
                'avg_features': avg_features,
                'primary_class': primary_class,
                'visualization_coords': {
                    'x': float(np.mean(X_pca[cluster_indices, 0])),
                    'y': float(np.mean(X_pca[cluster_indices, 1]))
                }
            }
        
        # Generate visualization
        cluster_viz = self._visualize_clusters(X_pca, clusters, symbols_list)
        
        # Create result
        result = {
            'timestamp': datetime.now(),
            'total_stocks': len(symbols_list),
            'num_clusters': n_clusters,
            'feature_names': feature_names,
            'clusters': cluster_analysis,
            'visualization': cluster_viz
        }
        
        # Save clustering result
        self._save_clustering(result)
        
        return result
    
    def _visualize_clusters(self, X_pca, clusters, symbols_list):
        """
        Visualize clusters in 2D space.
        
        Args:
            X_pca (array): PCA-transformed features
            clusters (array): Cluster assignments
            symbols_list (list): List of (symbol, exchange) tuples
            
        Returns:
            str: Base64 encoded image
        """
        try:
            plt.figure(figsize=(12, 8))
            
            # Plot points colored by cluster
            unique_clusters = np.unique(clusters)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
            
            for cluster_id, color in zip(unique_clusters, colors):
                indices = np.where(clusters == cluster_id)[0]
                plt.scatter(X_pca[indices, 0], X_pca[indices, 1], c=[color], label=f'Cluster {cluster_id}')
            
            # Add labels for points
            for i, (symbol, _) in enumerate(symbols_list):
                plt.annotate(symbol, (X_pca[i, 0], X_pca[i, 1]), fontsize=8)
            
            plt.title('Stock Clusters')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.legend()
            plt.grid(True)
            
            # Save plot to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close()
            
            # Encode as base64
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode('utf-8')
            
            return img_str
            
        except Exception as e:
            self.logger.error(f"Error visualizing clusters: {e}")
            return None
    
    def _save_clustering(self, clustering):
        """
        Save clustering to database.
        
        Args:
            clustering (dict): Clustering data
            
        Returns:
            str: Clustering ID
        """
        try:
            # Insert into database
            result = self.db.stock_clustering_collection.insert_one(clustering)
            clustering_id = str(result.inserted_id)
            
            self.logger.info(f"Saved clustering with ID: {clustering_id}")
            
            return clustering_id
            
        except Exception as e:
            self.logger.error(f"Error saving clustering: {e}")
            return None
    
    def batch_classify(self, symbols_list=None):
        """
        Classify multiple stocks and analyze correlations.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples
            
        Returns:
            dict: Batch classification results
        """
        if symbols_list is None:
            # Get active symbols
            cursor = self.db.portfolio_collection.find({
                'status': 'active',
                'trading_config.enabled': True
            })
            
            symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
        
        self.logger.info(f"Batch classifying {len(symbols_list)} stocks")
        
        results = {}
        features_list = []
        symbols_with_features = []
        classifications_by_type = {}
        
        for symbol, exchange in symbols_list:
            try:
                # Get market data
                market_data = self.get_stock_data(symbol, exchange)
                
                if market_data is None:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Insufficient data'
                    }
                    continue
                    
                # Extract features
                features = self.extract_features(market_data)
                
                if features is None:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Failed to extract features'
                    }
                    continue
                
                # Classify stock
                classification = self.classify_stock(symbol, exchange, features, market_data)
                
                if classification:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'success',
                        'classification': classification
                    }
                    
                    # Store for clustering
                    features_list.append(features)
                    symbols_with_features.append((symbol, exchange))
                    
                    # Group by primary class
                    primary_class = classification['primary_class']
                    if primary_class not in classifications_by_type:
                        classifications_by_type[primary_class] = []
                        
                    classifications_by_type[primary_class].append({
                        'symbol': symbol,
                        'exchange': exchange,
                        'classification': classification
                    })
                else:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Classification failed'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error classifying {symbol} {exchange}: {e}")
                results[f"{symbol}_{exchange}"] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # Run clustering on collected features
        clustering = None
        if len(features_list) >= 3:
            clustering = self.cluster_stocks(symbols_with_features, features_list)
        
        # Generate summary
        summary = {
            'timestamp': datetime.now(),
            'total_stocks': len(symbols_list),
            'successful_classifications': len(symbols_with_features),
            'class_distribution': {
                cls: len(stocks) for cls, stocks in classifications_by_type.items()
            },
            'clustering_id': clustering.get('_id') if clustering else None
        }
        
        # Generate recommendations for each class
        class_recommendations = {}
        
        for cls, stocks in classifications_by_type.items():
            if len(stocks) == 0:
                continue
                
            # Get description and strategies
            description = self.config['stock_classes'].get(cls, '')
            strategies = self.config['strategy_mapping'].get(cls, [])
            timeframe = self.config['timeframe_mapping'].get(cls, 'daily')
            
            # Top stocks in this class (using their class score)
            top_stocks = sorted(
                stocks,
                key=lambda x: x['classification']['class_scores'].get(cls, 0),
                reverse=True
            )[:5]
            
            class_recommendations[cls] = {
                'count': len(stocks),
                'description': description,
                'recommended_strategies': strategies,
                'recommended_timeframe': timeframe,
                'top_stocks': [{'symbol': s['symbol'], 'exchange': s['exchange']} for s in top_stocks]
            }
        
        summary['class_recommendations'] = class_recommendations
        
        # Save summary
        try:
            result = self.db.stock_classification_summary_collection.insert_one(summary)
            summary_id = str(result.inserted_id)
            
            self.logger.info(f"Saved classification summary with ID: {summary_id}")
            summary['_id'] = summary_id
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
        
        return {
            'summary': summary,
            'results': results
        }
    
    def save_model(self):
        """
        Save the trained models to database.
        
        Returns:
            str: Model ID
        """
        if self.kmeans_model is None:
            self.logger.error("No models to save. Cluster stocks first.")
            return None
            
        try:
            # Serialize models
            import pickle
            
            kmeans_bytes = pickle.dumps(self.kmeans_model)
            kmeans_base64 = base64.b64encode(kmeans_bytes).decode('utf-8')
            
            scaler_bytes = pickle.dumps(self.scaler)
            scaler_base64 = base64.b64encode(scaler_bytes).decode('utf-8')
            
            pca_bytes = pickle.dumps(self.pca)
            pca_base64 = base64.b64encode(pca_bytes).decode('utf-8')
            
            # Create model document
            model_doc = {
                'model_name': 'stock_classifier',
                'model_type': 'clustering',
                'creation_date': datetime.now(),
                'kmeans_model': kmeans_base64,
                'scaler_model': scaler_base64,
                'pca_model': pca_base64,
                'config': self.config
            }
            
            # Insert into database
            result = self.db.ml_models_collection.insert_one(model_doc)
            model_id = str(result.inserted_id)
            
            self.logger.info(f"Saved stock classifier models with ID: {model_id}")
            
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
            return None
    
    def load_model(self, model_id=None):
        """
        Load models from database.
        
        Args:
            model_id (str): Model ID
            
        Returns:
            bool: Success/failure
        """
        try:
            # Query database
            query = {'model_name': 'stock_classifier'}
            
            if model_id:
                query['_id'] = ObjectId(model_id)
                
            model_doc = self.db.ml_models_collection.find_one(
                query, sort=[('creation_date', -1)]
            )
            
            if not model_doc:
                self.logger.warning(f"No stock classifier models found")
                return False
                
            # Deserialize models
            import pickle
            
            kmeans_base64 = model_doc['kmeans_model']
            kmeans_bytes = base64.b64decode(kmeans_base64)
            self.kmeans_model = pickle.loads(kmeans_bytes)
            
            scaler_base64 = model_doc['scaler_model']
            scaler_bytes = base64.b64decode(scaler_base64)
            self.scaler = pickle.loads(scaler_bytes)
            
            pca_base64 = model_doc['pca_model']
            pca_bytes = base64.b64decode(pca_base64)
            self.pca = pickle.loads(pca_bytes)
            
            # Load config
            if 'config' in model_doc:
                self.config.update(model_doc['config'])
                
            self.logger.info(f"Loaded stock classifier models successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            return False
    
    def get_trading_recommendations(self, symbol, exchange):
        """
        Get trading recommendations based on stock classification.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Trading recommendations
        """
        self.logger.info(f"Generating trading recommendations for {symbol} {exchange}")
        
        # Get classification
        classification = self._get_existing_classification(symbol, exchange)
        
        if not classification:
            # Generate new classification
            classification = self.classify_stock(symbol, exchange)
            
        if not classification:
            self.logger.error(f"Failed to classify {symbol} {exchange}")
            return None
        
        # Basic recommendations from classification
        recommendations = {
            'symbol': symbol,
            'exchange': exchange,
            'timestamp': datetime.now(),
            'classification': {
                'primary_class': classification['primary_class'],
                'secondary_classes': classification['secondary_classes'],
                'description': classification['primary_description']
            },
            'strategies': classification['strategies'],
            'timeframe': classification['timeframe'],
            'parameters': classification['parameters']
        }
        
        # Get current market data
        recent_data = self.get_stock_data(symbol, exchange, days=30)
        
        if recent_data is not None:
            # Current price
            current_price = float(recent_data['close'].iloc[-1])
            recommendations['current_price'] = current_price
            
            # Calculate support and resistance levels
            support, resistance = self._calculate_support_resistance(recent_data)
            
            recommendations['levels'] = {
                'support': [round(float(s), 2) for s in support[:3]],  # Top 3 support levels
                'resistance': [round(float(r), 2) for r in resistance[:3]],  # Top 3 resistance levels
                'stop_loss': round(float(current_price * (1 - classification['parameters']['stop_loss_percent'] / 100)), 2),
                'take_profit': round(float(current_price * (1 + classification['parameters']['take_profit_percent'] / 100)), 2)
            }
            
            # Entry and exit recommendations
            recommendations['entry'] = self._get_entry_recommendation(
                classification, recent_data, support, resistance
            )
            
            recommendations['exit'] = self._get_exit_recommendation(
                classification, recent_data, support, resistance
            )
        
        # Save recommendations
        self._save_recommendations(recommendations)
        
        return recommendations
    
    def _calculate_support_resistance(self, market_data):
        """
        Calculate support and resistance levels.
        
        Args:
            market_data (DataFrame): Market data
            
        Returns:
            tuple: (support levels, resistance levels)
        """
        try:
            if len(market_data) < 20:
                return [], []
                
            # Extract price data
            highs = market_data['high'].values
            lows = market_data['low'].values
            closes = market_data['close'].values
            
            # Current price
            current_price = closes[-1]
            
            # Find swing highs and lows
            swing_highs = []
            swing_lows = []
            
            for i in range(2, len(closes) - 2):
                # Swing high
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
                   highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    swing_highs.append(highs[i])
                
                # Swing low
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
                   lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    swing_lows.append(lows[i])
            
            # Add min/max if enough data
            if len(closes) > 10:
                swing_highs.append(np.max(highs))
                swing_lows.append(np.min(lows))
            
            # Filter relevant levels
            resistance = [r for r in swing_highs if r > current_price]
            support = [s for s in swing_lows if s < current_price]
            
            # Sort levels
            resistance.sort()
            support.sort(reverse=True)
            
            return support, resistance
            
        except Exception as e:
            self.logger.warning(f"Error calculating support/resistance: {e}")
            return [], []
    
    def _get_entry_recommendation(self, classification, market_data, support, resistance):
        """
        Generate entry recommendations.
        
        Args:
            classification (dict): Stock classification
            market_data (DataFrame): Market data
            support (list): Support levels
            resistance (list): Resistance levels
            
        Returns:
            dict: Entry recommendations
        """
        primary_class = classification['primary_class']
        parameters = classification['parameters']
        current_price = float(market_data['close'].iloc[-1])
        
        entry = {
            'conditions': [],
            'entry_style': parameters['entry_timing'],
            'position_size': parameters['position_size']
        }
        
        # Basic conditions based on classification
        if primary_class == 'trend_follower':
            entry['conditions'].append("Enter on pullbacks within the established trend")
            entry['conditions'].append("Confirm trend direction with moving averages")
            
        elif primary_class == 'mean_reverter':
            entry['conditions'].append("Enter when price reaches extreme overbought/oversold levels")
            entry['conditions'].append("Use RSI or Stochastic indicators for timing")
            
        elif primary_class == 'high_volatility':
            entry['conditions'].append("Enter after confirmation of direction to reduce false signals")
            entry['conditions'].append("Use smaller position size to manage risk")
            
        elif primary_class == 'low_volatility':
            entry['conditions'].append("Enter at market or with limit orders slightly below market")
            entry['conditions'].append("Can use larger position size due to lower volatility")
            
        elif primary_class == 'swing_trader':
            entry['conditions'].append("Enter at extremes of price swings")
            entry['conditions'].append("Wait for reversal confirmation before entry")
            
        elif primary_class == 'breakout_candidate':
            entry['conditions'].append("Enter on volume-confirmed breakouts above resistance")
            entry['conditions'].append("Avoid false breakouts by waiting for confirmation")
            
        elif primary_class == 'range_bound':
            entry['conditions'].append("Enter near range boundaries with expectation of reversal")
            entry['conditions'].append("Buy near support, sell near resistance")
            
        elif primary_class == 'news_driven':
            entry['conditions'].append("Enter after news-driven moves have stabilized")
            entry['conditions'].append("Be cautious of volatile price action after news events")
        
        # Add specific price levels
        if support:
            entry['support_levels'] = [float(s) for s in support[:2]]
        if resistance:
            entry['resistance_levels'] = [float(r) for r in resistance[:2]]
        
        return entry
    
    def _get_exit_recommendation(self, classification, market_data, support, resistance):
        """
        Generate exit recommendations.
        
        Args:
            classification (dict): Stock classification
            market_data (DataFrame): Market data
            support (list): Support levels
            resistance (list): Resistance levels
            
        Returns:
            dict: Exit recommendations
        """
        primary_class = classification['primary_class']
        parameters = classification['parameters']
        current_price = float(market_data['close'].iloc[-1])
        
        exit = {
            'conditions': [],
            'exit_style': parameters['exit_criteria'],
            'max_holding_period': parameters['max_holding_period']
        }
        
        # Calculate stop loss and take profit
        stop_loss = current_price * (1 - parameters['stop_loss_percent'] / 100)
        take_profit = current_price * (1 + parameters['take_profit_percent'] / 100)
        
        exit['stop_loss'] = float(stop_loss)
        exit['take_profit'] = float(take_profit)
        
        # Basic conditions based on classification
        if primary_class == 'trend_follower':
            exit['conditions'].append("Exit when trend shows signs of reversal")
            exit['conditions'].append("Use trailing stops to protect profits")
            
        elif primary_class == 'mean_reverter':
            exit['conditions'].append("Exit when price reverts to the mean")
            exit['conditions'].append("Take profits quickly as mean reversion is typically short-lived")
            
        elif primary_class == 'high_volatility':
            exit['conditions'].append("Use tight trailing stops to lock in profits")
            exit['conditions'].append("Exit quickly on trend reversal signals")
            
        elif primary_class == 'low_volatility':
            exit['conditions'].append("Allow more time for trades to develop")
            exit['conditions'].append("Use time-based exits if price action is too slow")
            
        elif primary_class == 'swing_trader':
            exit['conditions'].append("Exit at opposite extreme of price swing")
            exit['conditions'].append("Take partial profits at first targets")
            
        elif primary_class == 'breakout_candidate':
            exit['conditions'].append("Use parabolic exit when momentum slows")
            exit['conditions'].append("Exit quickly if breakout fails and price returns to range")
            
        elif primary_class == 'range_bound':
            exit['conditions'].append("Exit when price approaches opposite end of range")
            exit['conditions'].append("Exit immediately if range breaks with strong momentum")
            
        elif primary_class == 'news_driven':
            exit['conditions'].append("Take quick profits as news impact tends to fade")
            exit['conditions'].append("Exit if price fails to follow through after news reaction")
        
        return exit
    
    def _save_recommendations(self, recommendations):
        """
        Save recommendations to database.
        
        Args:
            recommendations (dict): Recommendations data
            
        Returns:
            str: Recommendations ID
        """
        try:
            # Insert into database
            result = self.db.trading_recommendations_collection.insert_one(recommendations)
            recommendations_id = str(result.inserted_id)
            
            self.logger.info(f"Saved recommendations with ID: {recommendations_id}")
            
            return recommendations_id
            
        except Exception as e:
            self.logger.error(f"Error saving recommendations: {e}")
            return None
    
    def batch_recommendations(self, symbols_list=None):
        """
        Generate trading recommendations for multiple symbols.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples
            
        Returns:
            dict: Batch recommendations
        """
        if symbols_list is None:
            # Get active symbols
            cursor = self.db.portfolio_collection.find({
                'status': 'active',
                'trading_config.enabled': True
            })
            
            symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
        
        self.logger.info(f"Generating trading recommendations for {len(symbols_list)} symbols")
        
        results = {}
        recommendations_by_class = {}
        
        for symbol, exchange in symbols_list:
            try:
                # Generate recommendations
                recommendation = self.get_trading_recommendations(symbol, exchange)
                
                if recommendation:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'success',
                        'recommendation': recommendation
                    }
                    
                    # Group by primary class
                    primary_class = recommendation['classification']['primary_class']
                    if primary_class not in recommendations_by_class:
                        recommendations_by_class[primary_class] = []
                        
                    recommendations_by_class[primary_class].append(recommendation)
                else:
                    results[f"{symbol}_{exchange}"] = {
                        'status': 'error',
                        'message': 'Failed to generate recommendations'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error generating recommendations for {symbol} {exchange}: {e}")
                results[f"{symbol}_{exchange}"] = {
                    'status': 'error',
                    'message': str(e)
                }
        
        # Generate summary
        summary = {
            'timestamp': datetime.now(),
            'total_symbols': len(symbols_list),
            'successful_recommendations': sum(1 for r in results.values() if r['status'] == 'success'),
            'class_distribution': {
                cls: len(recs) for cls, recs in recommendations_by_class.items()
            }
        }
        
        # Best opportunities by class
        opportunities = {}
        
        for cls, recs in recommendations_by_class.items():
            if not recs:
                continue
                
            # Sort by favorable technical setup
            # (distance to support/resistance, etc.)
            sorted_recs = self._sort_recommendations_by_opportunity(recs)
            
            opportunities[cls] = [{
                'symbol': r['symbol'],
                'exchange': r['exchange'],
                'current_price': r.get('current_price'),
                'strategies': r['strategies'][:1],  # Top strategy
                'timeframe': r['timeframe']
            } for r in sorted_recs[:3]]  # Top 3 opportunities
        
        summary['opportunities'] = opportunities
        
        # Save summary
        try:
            result = self.db.recommendation_summary_collection.insert_one(summary)
            summary_id = str(result.inserted_id)
            
            self.logger.info(f"Saved recommendation summary with ID: {summary_id}")
            summary['_id'] = summary_id
            
        except Exception as e:
            self.logger.error(f"Error saving summary: {e}")
        
        return {
            'summary': summary,
            'results': results
        }
    
    def _sort_recommendations_by_opportunity(self, recommendations):
        """
        Sort recommendations by opportunity quality.
        
        Args:
            recommendations (list): List of recommendations
            
        Returns:
            list: Sorted recommendations
        """
        # Score each recommendation for opportunity quality
        scored_recs = []
        
        for rec in recommendations:
            score = 0
            
            # Check if price is near support (for long opportunities)
            if 'levels' in rec and 'support' in rec['levels'] and 'current_price' in rec:
                current_price = rec['current_price']
                support = rec['levels']['support']
                
                if support:
                    nearest_support = support[0]
                    support_distance = (current_price - nearest_support) / current_price
                    
                    # Higher score for closer support
                    if 0 < support_distance < 0.05:  # Within 5%
                        score += 3
                    elif 0.05 <= support_distance < 0.1:  # Within 5-10%
                        score += 2
                    elif 0.1 <= support_distance < 0.15:  # Within 10-15%
                        score += 1
            
            # Add score based on classification
            primary_class = rec['classification']['primary_class']
            
            trend_classes = ['trend_follower', 'breakout_candidate']
            mean_reversion_classes = ['mean_reverter', 'swing_trader', 'range_bound']
            
            # Current market condition favors certain strategies
            market_regime = self._get_current_market_regime()
            
            if market_regime == 'trending' and primary_class in trend_classes:
                score += 2
            elif market_regime == 'ranging' and primary_class in mean_reversion_classes:
                score += 2
            
            # Score based on parameters
            params = rec.get('parameters', {})
            if params.get('position_size') == 'larger':
                score += 1
            
            # Add the recommendation with its score
            scored_recs.append((rec, score))
        
        # Sort by score (descending)
        sorted_recs = [r for r, s in sorted(scored_recs, key=lambda x: x[1], reverse=True)]
        
        return sorted_recs
    
    def _get_current_market_regime(self):
        """
        Get current market regime (trending or ranging).
        
        Returns:
            str: Market regime
        """
        try:
            # Get latest market analysis
            analysis = self.db.market_analysis_collection.find_one(
                {}, sort=[('date', -1)]
            )
            
            if analysis and 'regime' in analysis:
                regime = analysis['regime']
                
                if regime in ['bullish', 'bearish']:
                    return 'trending'
                else:
                    return 'ranging'
            
            return 'unknown'
            
        except Exception as e:
            self.logger.warning(f"Error getting market regime: {e}")
            return 'unknown'