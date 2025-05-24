#!/usr/bin/env python3
"""
Complete Model Training Script for Stock Market Prediction
Trains all types of models: Classifiers, Regressors, Ensemble, and Simple Reinforcement Learning
No TensorFlow dependency - uses simplified RL implementation
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection_manager import get_db
from utils.logging_utils import setup_logger
from portfolio.portfolio_manager import PortfolioManager

# Import model classes
from ml.models.classifier import MarketClassifier
from ml.models.regressor import MarketRegressor
from ml.models.ensemble_predictor import EnsemblePredictor
from ml.models.reinforcement import ReinforcementLearning  # Uses simplified version

logger = setup_logger(__name__)

def prepare_training_data(db, symbol, exchange, days_back=730):
    """
    Prepare comprehensive training data from your existing data sources.
    
    Args:
        db: Database connector
        symbol (str): Stock symbol
        exchange (str): Exchange
        days_back (int): Days of historical data to use
        
    Returns:
        tuple: (features_df, target_classification, target_regression, market_data)
    """
    logger.info(f"Preparing training data for {symbol} {exchange}")
    
    # Get market data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    market_data = db.get_market_data(
        symbol=symbol,
        exchange=exchange,
        timeframe='day',
        start_date=start_date,
        end_date=end_date
    )
    
    if not market_data:
        logger.error(f"No market data found for {symbol} {exchange}")
        return None, None, None, None
    
    # Convert to DataFrame
    df = pd.DataFrame(market_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    
    if len(df) < 100:
        logger.error(f"Insufficient data: {len(df)} samples")
        return None, None, None, None
    
    logger.info(f"Retrieved {len(df)} days of market data")
    
    # Generate comprehensive features
    features_df = generate_comprehensive_features(df, db, symbol, exchange)
    
    # Generate targets
    target_classification = generate_classification_targets(df)
    target_regression = generate_regression_targets(df)
    
    # Align all data
    common_index = features_df.index.intersection(target_classification.index)
    common_index = common_index.intersection(target_regression.index)
    
    if len(common_index) < 50:
        logger.error(f"Insufficient aligned data: {len(common_index)} samples")
        return None, None, None, None
    
    features_df = features_df.loc[common_index]
    target_classification = target_classification.loc[common_index]
    target_regression = target_regression.loc[common_index]
    market_data_aligned = df.loc[common_index]
    
    # Remove NaN values
    mask = ~(features_df.isnull().any(axis=1) | target_classification.isnull() | target_regression.isnull())
    
    features_df = features_df[mask]
    target_classification = target_classification[mask]
    target_regression = target_regression[mask]
    market_data_aligned = market_data_aligned[mask]
    
    logger.info(f"Final dataset: {len(features_df)} samples with {len(features_df.columns)} features")
    
    return features_df, target_classification, target_regression, market_data_aligned

def generate_comprehensive_features(df, db, symbol, exchange):
    """Generate comprehensive feature set from all available data sources."""
    logger.info("Generating comprehensive features")
    
    features = pd.DataFrame(index=df.index)
    
    # 1. PRICE-BASED FEATURES
    logger.info("Adding price-based features")
    features['price_change_1d'] = df['close'].pct_change()
    features['price_change_2d'] = df['close'].pct_change(2)
    features['price_change_5d'] = df['close'].pct_change(5)
    features['price_change_10d'] = df['close'].pct_change(10)
    features['price_change_20d'] = df['close'].pct_change(20)
    
    # 2. MOVING AVERAGES
    logger.info("Adding moving average features")
    for window in [5, 10, 20, 50]:
        features[f'sma_{window}'] = df['close'].rolling(window).mean()
        features[f'ema_{window}'] = df['close'].ewm(span=window).mean()
        features[f'price_to_sma_{window}'] = df['close'] / features[f'sma_{window}']
        features[f'price_to_ema_{window}'] = df['close'] / features[f'ema_{window}']
    
    # 3. VOLATILITY FEATURES
    logger.info("Adding volatility features")
    features['volatility_5d'] = df['close'].rolling(5).std()
    features['volatility_20d'] = df['close'].rolling(20).std()
    features['volatility_50d'] = df['close'].rolling(50).std()
    features['volatility_ratio_5_20'] = features['volatility_5d'] / features['volatility_20d']
    features['volatility_ratio_20_50'] = features['volatility_20d'] / features['volatility_50d']
    
    # 4. VOLUME FEATURES
    if 'volume' in df.columns:
        logger.info("Adding volume features")
        features['volume_sma_5'] = df['volume'].rolling(5).mean()
        features['volume_sma_20'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_sma_20']
        features['price_volume'] = df['close'] * df['volume']
        features['volume_price_trend'] = (df['volume'] * df['close']).rolling(5).mean()
    else:
        # Add zero volume features if volume data is missing
        features['volume_sma_5'] = 0
        features['volume_sma_20'] = 0
        features['volume_ratio'] = 1
        features['price_volume'] = df['close']
        features['volume_price_trend'] = df['close'].rolling(5).mean()
    
    # 5. HIGH-LOW FEATURES
    logger.info("Adding high-low features")
    features['high_low_ratio'] = df['high'] / df['low']
    features['close_to_high'] = df['close'] / df['high']
    features['close_to_low'] = df['close'] / df['low']
    features['daily_range'] = (df['high'] - df['low']) / df['close']
    features['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # 6. TECHNICAL INDICATORS
    logger.info("Adding technical indicators")
    # RSI
    features['rsi_14'] = calculate_rsi(df['close'], 14)
    features['rsi_5'] = calculate_rsi(df['close'], 5)
    features['rsi_30'] = calculate_rsi(df['close'], 30)
    
    # MACD
    macd_data = calculate_macd(df['close'])
    features['macd'] = macd_data['macd']
    features['macd_signal'] = macd_data['signal']
    features['macd_histogram'] = macd_data['histogram']
    features['macd_crossover'] = (features['macd'] > features['macd_signal']).astype(int)
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(df['close'], 20)
    features['bb_upper'] = bb_data['upper']
    features['bb_lower'] = bb_data['lower']
    features['bb_middle'] = bb_data['middle']
    features['bb_position'] = (df['close'] - bb_data['lower']) / (bb_data['upper'] - bb_data['lower'])
    features['bb_squeeze'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
    
    # Stochastic Oscillator
    stoch_data = calculate_stochastic(df, 14)
    features['stoch_k'] = stoch_data['%K']
    features['stoch_d'] = stoch_data['%D']
    
    # 7. MOMENTUM FEATURES
    logger.info("Adding momentum features")
    for period in [5, 10, 20]:
        features[f'momentum_{period}'] = df['close'] / df['close'].shift(period)
        features[f'roc_{period}'] = ((df['close'] - df['close'].shift(period)) / df['close'].shift(period)) * 100
    
    # 8. PATTERN FEATURES
    logger.info("Adding pattern features")
    features['doji'] = ((abs(df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8)) < 0.1).astype(int)
    features['hammer'] = (((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8) > 0.6) & 
                         ((df['open'] - df['low']) / (df['high'] - df['low'] + 1e-8) > 0.6)).astype(int)
    
    # 9. TIME-BASED FEATURES
    logger.info("Adding time-based features")
    features['day_of_week'] = df.index.dayofweek
    features['month'] = df.index.month
    features['quarter'] = df.index.quarter
    features['is_month_end'] = (df.index.day > 25).astype(int)
    features['is_quarter_end'] = ((df.index.month % 3 == 0) & (df.index.day > 25)).astype(int)
    
    # 10. FUNDAMENTAL FEATURES (from your financial data)
    logger.info("Adding fundamental features")
    try:
        financial_data = db.get_financial_data(symbol, exchange)
        if financial_data:
            # Get latest financial metrics and forward-fill
            latest_financial = financial_data[0] if financial_data else {}
            fin_data = latest_financial.get('data', {})
            
            # Add key financial ratios
            eps_value = fin_data.get('eps_in_rs', 0)
            revenue_value = fin_data.get('revenue +', 0)
            net_profit_value = fin_data.get('net_profit +', 0)
            gross_npa_value = fin_data.get('gross_npa_%', 0)
            net_npa_value = fin_data.get('net_npa_%', 0)
            
            # Create constant features (will be forward-filled)
            features['eps'] = eps_value
            features['revenue_growth'] = revenue_value
            features['net_profit'] = net_profit_value
            features['gross_npa'] = gross_npa_value
            features['net_npa'] = net_npa_value
            
            logger.info(f"Added fundamental features: EPS={eps_value}, Revenue={revenue_value}")
        else:
            # Add zero fundamental features if no financial data
            features['eps'] = 0
            features['revenue_growth'] = 0
            features['net_profit'] = 0
            features['gross_npa'] = 0
            features['net_npa'] = 0
    
    except Exception as e:
        logger.warning(f"Could not add fundamental features: {e}")
        # Add zero fundamental features
        features['eps'] = 0
        features['revenue_growth'] = 0
        features['net_profit'] = 0
        features['gross_npa'] = 0
        features['net_npa'] = 0
    
    # 11. NEWS SENTIMENT FEATURES (from your news data)
    logger.info("Adding news sentiment features")
    try:
        # Get recent news for sentiment analysis
        recent_news = db.get_news(symbol=symbol, limit=50)
        if recent_news:
            news_sentiment_score = calculate_news_sentiment(recent_news)
            news_count_recent = len([n for n in recent_news if n.get('published_date')])
            
            features['news_sentiment'] = news_sentiment_score
            features['news_count_recent'] = news_count_recent
            
            logger.info(f"Added news features: sentiment={news_sentiment_score:.2f}, count={news_count_recent}")
        else:
            features['news_sentiment'] = 0.0
            features['news_count_recent'] = 0
    
    except Exception as e:
        logger.warning(f"Could not add news features: {e}")
        features['news_sentiment'] = 0.0
        features['news_count_recent'] = 0
    
    # 12. LAGGED FEATURES
    logger.info("Adding lagged features")
    for lag in [1, 2, 3, 5]:
        features[f'close_lag_{lag}'] = df['close'].shift(lag)
        features[f'rsi_lag_{lag}'] = features['rsi_14'].shift(lag)
        features[f'volume_lag_{lag}'] = df['volume'].shift(lag) if 'volume' in df.columns else 0
    
    # Fill NaN values
    features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    logger.info(f"Generated {len(features.columns)} features")
    return features

def calculate_rsi(prices, window):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    """Calculate MACD indicator."""
    exp1 = prices.ewm(span=12).mean()
    exp2 = prices.ewm(span=26).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal
    
    return {
        'macd': macd,
        'signal': signal,
        'histogram': histogram
    }

def calculate_bollinger_bands(prices, window):
    """Calculate Bollinger Bands."""
    sma = prices.rolling(window).mean()
    std = prices.rolling(window).std()
    
    return {
        'upper': sma + (std * 2),
        'lower': sma - (std * 2),
        'middle': sma
    }

def calculate_stochastic(df, window):
    """Calculate Stochastic Oscillator."""
    lowest_low = df['low'].rolling(window).min()
    highest_high = df['high'].rolling(window).max()
    
    k_percent = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(3).mean()
    
    return {
        '%K': k_percent,
        '%D': d_percent
    }

def calculate_news_sentiment(news_list):
    """Calculate simple news sentiment score."""
    if not news_list:
        return 0.0
    
    # Simple keyword-based sentiment
    positive_words = ['profit', 'growth', 'gain', 'increase', 'rise', 'up', 'positive', 'strong', 'good']
    negative_words = ['loss', 'decline', 'fall', 'decrease', 'down', 'negative', 'weak', 'bad', 'drop']
    
    sentiment_score = 0
    for news in news_list:
        title = (news.get('title', '') + ' ' + news.get('description', '')).lower()
        
        positive_count = sum(word in title for word in positive_words)
        negative_count = sum(word in title for word in negative_words)
        
        sentiment_score += (positive_count - negative_count)
    
    # Normalize to [-1, 1] range
    return max(-1, min(1, sentiment_score / len(news_list)))

def generate_classification_targets(df, threshold=0.02):
    """Generate classification targets (UP/DOWN/NEUTRAL)."""
    # Calculate next day returns
    next_day_return = df['close'].shift(-1) / df['close'] - 1
    
    # Classify based on threshold
    labels = pd.Series(index=df.index, dtype='object')
    labels[next_day_return > threshold] = 'UP'
    labels[next_day_return < -threshold] = 'DOWN'
    labels[(next_day_return >= -threshold) & (next_day_return <= threshold)] = 'NEUTRAL'
    
    return labels

def generate_regression_targets(df):
    """Generate regression targets (next day percentage change)."""
    return df['close'].shift(-1) / df['close'] - 1

def train_classifier_models(db, symbol, exchange, features_df, target_classification):
    """Train all classifier models."""
    logger.info(f"Training classifier models for {symbol} {exchange}")
    
    # Define available model types (check if libraries are installed)
    model_types = ['random_forest', 'gradient_boosting']
    
    # Check for optional libraries
    try:
        import xgboost
        model_types.append('xgboost')
    except ImportError:
        logger.warning("XGBoost not available. Install with: pip install xgboost")
    
    try:
        import lightgbm
        model_types.append('lightgbm')
    except ImportError:
        logger.warning("LightGBM not available. Install with: pip install lightgbm")
    
    trained_models = {}
    
    for model_type in model_types:
        try:
            logger.info(f"Training {model_type} classifier")
            
            # Initialize classifier
            classifier = MarketClassifier(db)
            classifier.build_model(model_type)
            
            # Train model
            results = classifier.train(features_df, target_classification, time_series_split=True)
            
            if results:
                logger.info(f"{model_type} classifier - Accuracy: {results['accuracy']:.4f}")
                
                # Save model
                model_name = f"{symbol}_{exchange}_{model_type}_classifier"
                model_id = classifier.save_model(symbol, exchange, model_name)
                
                if model_id:
                    trained_models[model_type] = {
                        'model': classifier,
                        'results': results,
                        'model_id': model_id
                    }
                    logger.info(f"Saved {model_type} classifier with ID: {model_id}")
            
        except Exception as e:
            logger.error(f"Error training {model_type} classifier: {e}")
    
    return trained_models

def train_regressor_models(db, symbol, exchange, features_df, target_regression):
    """Train all regressor models."""
    logger.info(f"Training regressor models for {symbol} {exchange}")
    
    # Define available model types
    model_types = ['random_forest', 'gradient_boosting']
    
    # Check for optional libraries
    try:
        import xgboost
        model_types.append('xgboost')
    except ImportError:
        logger.warning("XGBoost not available. Install with: pip install xgboost")
    
    try:
        import lightgbm
        model_types.append('lightgbm')
    except ImportError:
        logger.warning("LightGBM not available. Install with: pip install lightgbm")
    
    trained_models = {}
    
    for model_type in model_types:
        try:
            logger.info(f"Training {model_type} regressor")
            
            # Initialize regressor
            regressor = MarketRegressor(db)
            regressor.build_model(model_type)
            
            # Train model
            results = regressor.train(features_df, target_regression, time_series_split=True)
            
            if results:
                logger.info(f"{model_type} regressor - RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}")
                
                # Save model
                model_name = f"{symbol}_{exchange}_{model_type}_regressor"
                model_id = regressor.save_model(symbol, exchange, model_name)
                
                if model_id:
                    trained_models[model_type] = {
                        'model': regressor,
                        'results': results,
                        'model_id': model_id
                    }
                    logger.info(f"Saved {model_type} regressor with ID: {model_id}")
            
        except Exception as e:
            logger.error(f"Error training {model_type} regressor: {e}")
    
    return trained_models

def train_ensemble_model(db, symbol, exchange, classifier_models, regressor_models, features_df, target_classification, target_regression):
    """Train ensemble model using trained individual models."""
    logger.info(f"Training ensemble model for {symbol} {exchange}")
    
    try:
        # Initialize ensemble
        ensemble = EnsemblePredictor(db)
        
        # Add classifiers to ensemble
        for model_type, model_data in classifier_models.items():
            ensemble.add_classifier(
                model_data['model'].model,
                name=f"{model_type}_classifier",
                weight=model_data['results']['accuracy']  # Weight by accuracy
            )
        
        # Add regressors to ensemble
        for model_type, model_data in regressor_models.items():
            ensemble.add_regressor(
                model_data['model'].model,
                name=f"{model_type}_regressor",
                weight=abs(model_data['results']['r2'])  # Weight by R²
            )
        
        # Build ensemble
        if ensemble.build_ensemble('both'):
            # Train ensemble
            classifier_results = ensemble.train_classifier(features_df, target_classification)
            regressor_results = ensemble.train_regressor(features_df, target_regression)
            
            if classifier_results and regressor_results:
                logger.info(f"Ensemble classifier - Accuracy: {classifier_results['accuracy']:.4f}")
                logger.info(f"Ensemble regressor - RMSE: {regressor_results['rmse']:.4f}")
                
                # Save ensemble
                ensemble_name = f"{symbol}_{exchange}_ensemble_classifier"
                ensemble_id = ensemble.save_models(symbol, exchange, ensemble_name)
                
                if ensemble_id:
                    logger.info(f"Saved ensemble model with ID: {ensemble_id}")
                    return {
                        'model': ensemble,
                        'classifier_results': classifier_results,
                        'regressor_results': regressor_results,
                        'ensemble_id': ensemble_id
                    }
        
    except Exception as e:
        logger.error(f"Error training ensemble model: {e}")
    
    return None

def train_reinforcement_model(db, symbol, exchange, market_data):
    """Train simple reinforcement learning model (no TensorFlow)."""
    logger.info(f"Training simple RL model for {symbol} {exchange}")
    
    try:
        # Initialize RL model
        rl_model = ReinforcementLearning(db)
        
        # Prepare data for RL training
        if 'volume' not in market_data.columns:
            market_data['volume'] = 1000000  # Add dummy volume if missing
        
        rl_train_data, rl_test_data = rl_model.prepare_data(
            market_data[['open', 'high', 'low', 'close', 'volume']],
            window_size=10,
            test_size=0.2
        )
        
        # Build agent
        agent, env = rl_model.build_agent(
            rl_train_data,
            window_size=10,
            learning_rate=0.1,
            epsilon=1.0,
            epsilon_decay=0.99
        )
        
        if agent and env:
            # Train the agent (reduced episodes for speed)
            training_results = rl_model.train(
                episodes=50,  # Reduced from 100
                max_steps=500  # Reduced from 1000
            )
            
            if training_results:
                # Evaluate the agent
                eval_results = rl_model.evaluate(rl_test_data, episodes=5)
                
                if eval_results:
                    logger.info(f"RL Model - Mean Return: {eval_results['mean_return']:.4f}")
                    logger.info(f"RL Model - Final Portfolio: {eval_results['mean_portfolio_value']:.2f}")
                    
                    # Save RL model
                    model_name = f"{symbol}_{exchange}_reinforcement_model"
                    model_id = rl_model.save_model(symbol, exchange, model_name)
                    
                    if model_id:
                        logger.info(f"Saved RL model with ID: {model_id}")
                        return {
                            'model': rl_model,
                            'training_results': training_results,
                            'eval_results': eval_results,
                            'model_id': model_id
                        }
        
    except Exception as e:
        logger.error(f"Error training reinforcement learning model: {e}")
    
    return None

def train_models_for_symbol(symbol, exchange, model_types='all', days_back=730):
    """Train all models for a single symbol."""
    logger.info(f"Starting model training for {symbol} {exchange}")
    
    # Initialize database
    db = get_db()
    
    # Prepare training data
    features_df, target_classification, target_regression, market_data = prepare_training_data(
        db, symbol, exchange, days_back
    )
    
    if features_df is None:
        logger.error(f"Failed to prepare training data for {symbol} {exchange}")
        return None
    
    results = {
        'symbol': symbol,
        'exchange': exchange,
        'training_date': datetime.now(),
        'data_samples': len(features_df),
        'features_count': len(features_df.columns)
    }
    
    # Train classifiers
    if model_types in ['all', 'classifier']:
        classifier_models = train_classifier_models(db, symbol, exchange, features_df, target_classification)
        results['classifiers'] = classifier_models
    else:
        classifier_models = {}
    
    # Train regressors
    if model_types in ['all', 'regressor']:
        regressor_models = train_regressor_models(db, symbol, exchange, features_df, target_regression)
        results['regressors'] = regressor_models
    else:
        regressor_models = {}
    
    # Train ensemble
    if model_types in ['all', 'ensemble'] and classifier_models and regressor_models:
        ensemble_results = train_ensemble_model(
            db, symbol, exchange, classifier_models, regressor_models, 
            features_df, target_classification, target_regression
        )
        results['ensemble'] = ensemble_results
    
    # Train reinforcement learning
    if model_types in ['all', 'reinforcement']:
        rl_results = train_reinforcement_model(db, symbol, exchange, market_data)
        results['reinforcement'] = rl_results
    
    logger.info(f"Completed model training for {symbol} {exchange}")
    return results

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train ML models for stock prediction (No TensorFlow)')
    
    parser.add_argument('--symbol', help='Stock symbol to train models for')
    parser.add_argument('--exchange', help='Exchange (e.g., NSE, BSE)')
    parser.add_argument('--all', action='store_true', help='Train models for all active instruments')
    parser.add_argument('--model-types', 
                       choices=['all', 'classifier', 'regressor', 'ensemble', 'reinforcement'],
                       default='all',
                       help='Types of models to train')
    parser.add_argument('--days-back', type=int, default=730, 
                       help='Days of historical data to use for training')
    
    args = parser.parse_args()
    
    # Get symbols to train
    symbols_to_train = []
    
    if args.all:
        # Get all active instruments
        db = get_db()
        portfolio_manager = PortfolioManager(db)
        instruments = portfolio_manager.get_active_instruments()
        symbols_to_train = [(inst['symbol'], inst['exchange']) for inst in instruments]
        logger.info(f"Training models for {len(symbols_to_train)} instruments")
    
    elif args.symbol and args.exchange:
        symbols_to_train = [(args.symbol, args.exchange)]
        logger.info(f"Training models for {args.symbol} {args.exchange}")
    
    else:
        logger.error("Please specify --symbol and --exchange, or use --all")
        return
    
    # Train models
    all_results = []
    
    for symbol, exchange in symbols_to_train:
        try:
            results = train_models_for_symbol(
                symbol, exchange, 
                model_types=args.model_types,
                days_back=args.days_back
            )
            
            if results:
                all_results.append(results)
                logger.info(f"✅ Successfully trained models for {symbol} {exchange}")
            else:
                logger.error(f"❌ Failed to train models for {symbol} {exchange}")
        
        except Exception as e:
            logger.error(f"❌ Error training models for {symbol} {exchange}: {e}")
    
    # Summary
    logger.info("="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total symbols processed: {len(symbols_to_train)}")
    logger.info(f"Successfully trained: {len(all_results)}")
    logger.info(f"Failed: {len(symbols_to_train) - len(all_results)}")
    
    for result in all_results:
        symbol = result['symbol']
        exchange = result['exchange']
        samples = result['data_samples']
        features = result['features_count']
        
        logger.info(f"{symbol} {exchange}: {samples} samples, {features} features")
        
        if 'classifiers' in result:
            logger.info(f"  - Classifiers trained: {len(result['classifiers'])}")
        if 'regressors' in result:
            logger.info(f"  - Regressors trained: {len(result['regressors'])}")
        if 'ensemble' in result and result['ensemble']:
            logger.info(f"  - Ensemble model trained ✅")
        if 'reinforcement' in result and result['reinforcement']:
            logger.info(f"  - RL model trained ✅")
    
    logger.info("Training completed!")
    
    if all_results:
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("1. Generate predictions: python3 scripts/generate_predictions.py --all")
        logger.info("2. Check predictions: python3 scripts/check_predictions.py")
        logger.info("3. View model performance in your database")

if __name__ == '__main__':
    main()