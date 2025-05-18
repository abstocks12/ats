# ml/training/pipeline.py
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import os
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from ml.features.technical_features import TechnicalFeatureGenerator
from ml.features.fundamental_features import FundamentalFeatureGenerator
from ml.features.global_features import GlobalFeatureGenerator
from ml.features.sentiment_features import SentimentFeatureGenerator
from ml.models.classifier import MarketClassifier
from ml.models.regressor import MarketRegressor
from ml.models.ensemble_predictor import EnsemblePredictor

class ModelTrainingPipeline:
    """Model training pipeline for market prediction."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the training pipeline.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize feature generators
        self.technical_features = TechnicalFeatureGenerator(db_connector)
        self.fundamental_features = FundamentalFeatureGenerator(db_connector)
        self.global_features = GlobalFeatureGenerator(db_connector)
        self.sentiment_features = SentimentFeatureGenerator(db_connector)
        
        # Initialize models
        self.classifiers = {}
        self.regressors = {}
        self.ensemble = None
        
        # Training configuration
        self.config = {
            'target_type': 'classification',  # 'classification' or 'regression'
            'prediction_horizon': 1,  # Days ahead to predict
            'training_period': 365,  # Days of data to use
            'test_size': 0.2,
            'use_time_series_split': True,
            'feature_groups': ['technical', 'fundamental', 'global', 'sentiment'],
            'model_types': ['random_forest', 'gradient_boosting', 'xgboost'],
            'use_ensemble': True
        }
        
        self.training_status = {
            'started': None,
            'completed': None,
            'status': 'not_started',
            'error': None,
            'features_generated': False,
            'models_trained': False,
            'ensemble_trained': False
        }
    
    def set_config(self, config):
        """
        Set training configuration.
        
        Args:
            config (dict): Configuration parameters
        """
        self.config.update(config)
        self.logger.info(f"Updated training configuration: {self.config}")
    
    def get_market_data(self, symbol, exchange, start_date, end_date=None):
        """
        Get market data for training.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date
            end_date (datetime): End date (default: today)
            
        Returns:
            DataFrame: Market data
        """
        end_date = end_date or datetime.now()
        
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
        
        if len(market_data) == 0:
            self.logger.error(f"No market data found for {symbol} {exchange} from {start_date} to {end_date}")
            return None
        
        # Set timestamp as index
        market_data.set_index('timestamp', inplace=True)
        
        self.logger.info(f"Retrieved {len(market_data)} days of market data for {symbol} {exchange}")
        
        return market_data
    
    def generate_features(self, symbol, exchange, start_date=None, end_date=None):
        """
        Generate features for training.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            DataFrame: Feature data
        """
        self.logger.info(f"Generating features for {symbol} {exchange}")
        
        # Set dates
        end_date = end_date or datetime.now()
        
        if start_date is None:
            # Calculate start date based on config
            training_days = self.config['training_period']
            start_date = end_date - timedelta(days=training_days + 100)  # Add buffer for feature calculation
        
        # Get market data
        market_data = self.get_market_data(symbol, exchange, start_date, end_date)
        
        if market_data is None or len(market_data) == 0:
            self.logger.error(f"Failed to retrieve market data for {symbol} {exchange}")
            return None
        
        features_data = pd.DataFrame(index=market_data.index)
        
        # Generate features based on configuration
        feature_groups = self.config['feature_groups']
        
        try:
            # Technical features
            if 'technical' in feature_groups:
                tech_features = self.technical_features.generate_all_features(
                    symbol, exchange, market_data
                )
                if tech_features is not None:
                    features_data = features_data.join(tech_features)
            
            # Fundamental features
            if 'fundamental' in feature_groups:
                fund_features = self.fundamental_features.generate_features(
                    symbol, exchange, start_date, end_date
                )
                if fund_features is not None:
                    # Join on date
                    features_data = features_data.join(fund_features)
            
            # Global features
            if 'global' in feature_groups:
                global_features = self.global_features.generate_features(
                    start_date, end_date
                )
                if global_features is not None:
                    # Join on date
                    features_data = features_data.join(global_features)
            
            # Sentiment features
            if 'sentiment' in feature_groups:
                sentiment_features = self.sentiment_features.generate_features(
                    symbol, exchange, start_date, end_date
                )
                if sentiment_features is not None:
                    # Join on date
                    features_data = features_data.join(sentiment_features)
            
            # Generate target variable
            prediction_horizon = self.config['prediction_horizon']
            
            if self.config['target_type'] == 'classification':
                # Classification target: 1 for up, 0 for down
                features_data['target'] = market_data['close'].pct_change(prediction_horizon).shift(-prediction_horizon) > 0
                features_data['target'] = features_data['target'].astype(int)
            else:
                # Regression target: percent change
                features_data['target'] = market_data['close'].pct_change(prediction_horizon).shift(-prediction_horizon)
            
            # Remove rows with NaN values
            features_data = features_data.dropna()
            
            self.logger.info(f"Generated {len(features_data.columns) - 1} features for {symbol} {exchange}")
            
            # Update status
            self.training_status['features_generated'] = True
            
            return features_data
            
        except Exception as e:
            self.logger.error(f"Error generating features: {e}")
            self.training_status['error'] = str(e)
            return None
    
    def train_models(self, features_data, symbol, exchange):
        """
        Train models using generated features.
        
        Args:
            features_data (DataFrame): Feature data with target
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            dict: Training results
        """
        if features_data is None or len(features_data) == 0:
            self.logger.error("No feature data provided for training")
            return None
        
        self.logger.info(f"Training models for {symbol} {exchange}")
        
        results = {
            'classifiers': {},
            'regressors': {},
            'ensemble': None
        }
        
        try:
            # Split data
            X = features_data.drop('target', axis=1)
            y = features_data['target']
            
            # Train models based on configuration
            model_types = self.config['model_types']
            target_type = self.config['target_type']
            
            if target_type == 'classification':
                # Train classification models
                for model_type in model_types:
                    self.logger.info(f"Training {model_type} classifier")
                    
                    classifier = MarketClassifier(self.db)
                    classifier.build_model(model_type=model_type)
                    
                    # Train
                    train_results = classifier.train(
                        X, y,
                        test_size=self.config['test_size'],
                        time_series_split=self.config['use_time_series_split']
                    )
                    
                    # Store model
                    self.classifiers[model_type] = classifier
                    
                    # Store results
                    results['classifiers'][model_type] = {
                        'accuracy': train_results['accuracy'],
                        'precision': train_results['precision'],
                        'recall': train_results['recall'],
                        'f1': train_results['f1']
                    }
                    
                    # Save model
                    model_name = f"{symbol}_{exchange}_{model_type}_classifier"
                    model_id = classifier.save_model(
                        symbol, exchange, model_name,
                        description=f"{model_type} classifier for {symbol} {exchange}"
                    )
                    
                    results['classifiers'][model_type]['model_id'] = model_id
                
                # Train ensemble if configured
                if self.config['use_ensemble'] and len(self.classifiers) >= 2:
                    self.logger.info("Training classifier ensemble")
                    
                    self.ensemble = EnsemblePredictor(self.db)
                    
                    # Add models to ensemble
                    for model_type, classifier in self.classifiers.items():
                        self.ensemble.add_classifier(classifier.model, name=model_type)
                    
                    # Build ensemble
                    self.ensemble.build_ensemble('classifier')
                    
                    # Train ensemble
                    ensemble_results = self.ensemble.train_classifier(
                        X, y,
                        test_size=self.config['test_size'],
                        time_series_split=self.config['use_time_series_split']
                    )
                    
                    # Save ensemble
                    ensemble_name = f"{symbol}_{exchange}_ensemble_classifier"
                    ensemble_id = self.ensemble.save_models(
                        symbol, exchange, ensemble_name,
                        description=f"Ensemble classifier for {symbol} {exchange}"
                    )
                    
                    results['ensemble'] = {
                        'type': 'classifier',
                        'accuracy': ensemble_results['accuracy'],
                        'precision': ensemble_results['precision'],
                        'recall': ensemble_results['recall'],
                        'f1': ensemble_results['f1'],
                        'ensemble_id': ensemble_id
                    }
            
            else:
                # Train regression models
                for model_type in model_types:
                    self.logger.info(f"Training {model_type} regressor")
                    
                    regressor = MarketRegressor(self.db)
                    regressor.build_model(model_type=model_type)
                    
                    # Train
                    train_results = regressor.train(
                        X, y,
                        test_size=self.config['test_size'],
                        time_series_split=self.config['use_time_series_split']
                    )
                    
                    # Store model
                    self.regressors[model_type] = regressor
                    
                    # Store results
                    results['regressors'][model_type] = {
                        'mse': train_results['mse'],
                        'rmse': train_results['rmse'],
                        'mae': train_results['mae'],
                        'r2': train_results['r2']
                    }
                    
                    # Save model
                    model_name = f"{symbol}_{exchange}_{model_type}_regressor"
                    model_id = regressor.save_model(
                        symbol, exchange, model_name,
                        description=f"{model_type} regressor for {symbol} {exchange}"
                    )
                    
                    results['regressors'][model_type]['model_id'] = model_id
                
                # Train ensemble if configured
                if self.config['use_ensemble'] and len(self.regressors) >= 2:
                    self.logger.info("Training regressor ensemble")
                    
                    self.ensemble = EnsemblePredictor(self.db)
                    
                    # Add models to ensemble
                    for model_type, regressor in self.regressors.items():
                        self.ensemble.add_regressor(regressor.model, name=model_type)
                    
                    # Build ensemble
                    self.ensemble.build_ensemble('regressor')
                    
                    # Train ensemble
                    ensemble_results = self.ensemble.train_regressor(
                        X, y,
                        test_size=self.config['test_size'],
                        time_series_split=self.config['use_time_series_split']
                    )
                    
                    # Save ensemble
                    ensemble_name = f"{symbol}_{exchange}_ensemble_regressor"
                    ensemble_id = self.ensemble.save_models(
                        symbol, exchange, ensemble_name,
                        description=f"Ensemble regressor for {symbol} {exchange}"
                    )
                    
                    results['ensemble'] = {
                        'type': 'regressor',
                        'mse': ensemble_results['mse'],
                        'rmse': ensemble_results['rmse'],
                        'mae': ensemble_results['mae'],
                        'r2': ensemble_results['r2'],
                        'directional_accuracy': ensemble_results['directional_accuracy'],
                        'ensemble_id': ensemble_id
                    }
            
            # Update status
            self.training_status['models_trained'] = True
            self.training_status['ensemble_trained'] = self.config['use_ensemble']
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error training models: {e}")
            self.training_status['error'] = str(e)
            return None
    
    def run_pipeline(self, symbol, exchange, start_date=None, end_date=None):
        """
        Run the full training pipeline.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            start_date (datetime): Start date
            end_date (datetime): End date
            
        Returns:
            dict: Pipeline results
        """
        self.logger.info(f"Starting training pipeline for {symbol} {exchange}")
        
        # Update status
        self.training_status['started'] = datetime.now()
        self.training_status['status'] = 'running'
        
        # Generate features
        features_data = self.generate_features(symbol, exchange, start_date, end_date)
        
        if features_data is None:
            self.training_status['status'] = 'failed'
            self.logger.error("Feature generation failed")
            return None
        
        # Train models
        results = self.train_models(features_data, symbol, exchange)
        
        if results is None:
            self.training_status['status'] = 'failed'
            self.logger.error("Model training failed")
            return None
        
        # Update status
        self.training_status['completed'] = datetime.now()
        self.training_status['status'] = 'completed'
        
        # Save pipeline run details
        pipeline_run = {
            'symbol': symbol,
            'exchange': exchange,
            'start_date': start_date,
            'end_date': end_date,
            'config': self.config,
            'status': self.training_status,
            'results': results,
            'feature_count': len(features_data.columns) - 1,  # Exclude target
            'sample_count': len(features_data),
            'run_date': datetime.now()
        }
        
        self.db.training_runs_collection.insert_one(pipeline_run)
        
        self.logger.info(f"Training pipeline completed for {symbol} {exchange}")
        
        return {
            'status': self.training_status,
            'results': results
        }
    
    def generate_daily_predictions(self, symbols_list=None):
        """
        Generate predictions for a list of symbols using trained models.
        
        Args:
            symbols_list (list): List of (symbol, exchange) tuples, or None for all active symbols
            
        Returns:
            dict: Prediction results
        """
        if symbols_list is None:
            # Get all active symbols from portfolio
            cursor = self.db.portfolio_collection.find({
                'status': 'active',
                'trading_config.enabled': True
            })
            
            symbols_list = [(doc['symbol'], doc['exchange']) for doc in cursor]
        
        self.logger.info(f"Generating daily predictions for {len(symbols_list)} symbols")
        
        results = {}
        
        for symbol, exchange in symbols_list:
            try:
                self.logger.info(f"Generating prediction for {symbol} {exchange}")
                
                # Get latest data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=100)  # Need enough data for feature generation
                
                # Generate features
                features_data = self.generate_features(symbol, exchange, start_date, end_date)
                
                if features_data is None or len(features_data) == 0:
                    self.logger.error(f"Failed to generate features for {symbol} {exchange}")
                    continue
                
                # Get the latest data point for prediction
                latest_features = features_data.iloc[-1:].drop('target', axis=1)
                
                # Load the latest ensemble model
                ensemble = EnsemblePredictor(self.db)
                
                target_type = self.config['target_type']
                ensemble_name = f"{symbol}_{exchange}_ensemble_{target_type}r"
                
                if ensemble.load_models(symbol=symbol, exchange=exchange, ensemble_name=ensemble_name):
                    # Generate prediction
                    prediction = ensemble.generate_market_prediction(
                        symbol, exchange, latest_features, save_prediction=True
                    )
                    
                    results[f"{symbol}_{exchange}"] = prediction
                else:
                    self.logger.warning(f"No ensemble model found for {symbol} {exchange}, trying individual models")
                    
                    # Try to load individual models
                    if target_type == 'classification':
                        model = MarketClassifier(self.db)
                        model_name = f"{symbol}_{exchange}_random_forest_classifier"
                        
                        if model.load_model(symbol=symbol, exchange=exchange, model_name=model_name):
                            prediction = model.generate_market_prediction(
                                symbol, exchange, latest_features, save_prediction=True
                            )
                            
                            results[f"{symbol}_{exchange}"] = prediction
                        else:
                            self.logger.error(f"No models found for {symbol} {exchange}")
                    else:
                        model = MarketRegressor(self.db)
                        model_name = f"{symbol}_{exchange}_random_forest_regressor"
                        
                        if model.load_model(symbol=symbol, exchange=exchange, model_name=model_name):
                            prediction = model.generate_price_prediction(
                                symbol, exchange, latest_features, save_prediction=True
                            )
                            
                            results[f"{symbol}_{exchange}"] = prediction
                        else:
                            self.logger.error(f"No models found for {symbol} {exchange}")
            
            except Exception as e:
                self.logger.error(f"Error generating prediction for {symbol} {exchange}: {e}")
        
        self.logger.info(f"Generated predictions for {len(results)} symbols")
        
        return results