# ml/models/regressor.py
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class MarketRegressor:
    """Regression models for market prediction tasks."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the regressor.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.model_params = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        
    def build_model(self, model_type='random_forest', params=None):
        """
        Build a regression model.
        
        Args:
            model_type (str): Type of model to build. Options:
                'random_forest', 'gradient_boosting', 'linear', 'ridge', 'lasso', 
                'elastic_net', 'svr', 'neural_network', 'xgboost', 'lightgbm'
            params (dict): Model hyperparameters
            
        Returns:
            The model instance
        """
        self.logger.info(f"Building {model_type} regression model")
        
        default_params = self._get_default_params(model_type)
        params = params or default_params
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(**params)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(**params)
        elif model_type == 'linear':
            self.model = LinearRegression(**params)
        elif model_type == 'ridge':
            self.model = Ridge(**params)
        elif model_type == 'lasso':
            self.model = Lasso(**params)
        elif model_type == 'elastic_net':
            self.model = ElasticNet(**params)
        elif model_type == 'svr':
            self.model = SVR(**params)
        elif model_type == 'neural_network':
            self.model = MLPRegressor(**params)
        elif model_type == 'xgboost':
            self.model = XGBRegressor(**params)
        elif model_type == 'lightgbm':
            self.model = LGBMRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model_params = {
            'type': model_type,
            'params': params
        }
        
        return self.model
    
    def _get_default_params(self, model_type):
        """
        Get default hyperparameters for each model type.
        
        Args:
            model_type (str): Type of model
            
        Returns:
            dict: Default parameters
        """
        if model_type == 'random_forest':
            return {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            }
        elif model_type == 'gradient_boosting':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            }
        elif model_type == 'linear':
            return {
                'fit_intercept': True,
                'n_jobs': -1
            }
        elif model_type == 'ridge':
            return {
                'alpha': 1.0,
                'random_state': 42
            }
        elif model_type == 'lasso':
            return {
                'alpha': 1.0,
                'random_state': 42
            }
        elif model_type == 'elastic_net':
            return {
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'random_state': 42
            }
        elif model_type == 'svr':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'epsilon': 0.1
            }
        elif model_type == 'neural_network':
            return {
                'hidden_layer_sizes': (100,),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.0001,
                'random_state': 42
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'objective': 'reg:squarederror',
                'random_state': 42
            }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'objective': 'regression',
                'random_state': 42
            }
        else:
            return {}
    
    def train(self, X, y, test_size=0.2, time_series_split=True, shuffle=False):
        """
        Train the regression model.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            test_size (float): Proportion of data to use for testing
            time_series_split (bool): Whether to use time series split (vs random)
            shuffle (bool): Whether to shuffle data (ignored if time_series_split is True)
            
        Returns:
            dict: Training results including metrics
        """
        if self.model is None:
            self.logger.error("Model not built. Call build_model() first.")
            return None
            
        self.logger.info(f"Training {self.model_params['type']} regression model")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        if time_series_split:
            # Use the last test_size percent for testing
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=test_size, shuffle=shuffle, random_state=42
            )
            
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store feature importance if available
        self._compute_feature_importance(X)
        
        self.logger.info(f"Model training complete. RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'test_size': test_size,
            'feature_importance': self.feature_importance
        }
    
    def _compute_feature_importance(self, X):
        """
        Compute and store feature importance if the model supports it.
        
        Args:
            X (DataFrame): Feature matrix with column names
        """
        importance_models = ['random_forest', 'gradient_boosting', 'xgboost', 'lightgbm']
        coefficient_models = ['linear', 'ridge', 'lasso', 'elastic_net']
        
        if self.model_params['type'] in importance_models and hasattr(self.model, 'feature_importances_'):
            feature_names = X.columns.tolist()
            importances = self.model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            self.feature_importance = [{
                'feature': feature_names[i],
                'importance': float(importances[i])
            } for i in indices]
            
        elif self.model_params['type'] in coefficient_models and hasattr(self.model, 'coef_'):
            feature_names = X.columns.tolist()
            coefficients = self.model.coef_
            
            # Get absolute coefficient values and sort
            abs_coefficients = np.abs(coefficients)
            indices = np.argsort(abs_coefficients)[::-1]
            
            self.feature_importance = [{
                'feature': feature_names[i],
                'importance': float(abs_coefficients[i]),
                'coefficient': float(coefficients[i])
            } for i in indices]
        else:
            self.feature_importance = None
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5, scoring='neg_mean_squared_error', time_series_cv=True):
        """
        Perform hyperparameter tuning using grid search.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            param_grid (dict): Parameter grid to search
            cv (int): Number of cross-validation folds
            scoring (str): Scoring metric
            time_series_cv (bool): Whether to use time series cross-validation
            
        Returns:
            dict: Best parameters and scores
        """
        if self.model is None:
            self.logger.error("Model not built. Call build_model() first.")
            return None
            
        if param_grid is None:
            param_grid = self._default_param_grid()
            
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Set up cross-validation
        if time_series_cv:
            cv = TimeSeriesSplit(n_splits=cv)
            
        # Grid search
        grid_search = GridSearchCV(
            self.model, param_grid, cv=cv, scoring=scoring, n_jobs=-1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.model_params['params'] = grid_search.best_params_
        
        self.logger.info(f"Hyperparameter tuning complete. Best score: {grid_search.best_score_:.4f}")
        
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def _default_param_grid(self):
        """
        Get default parameter grid for hyperparameter tuning based on model type.
        
        Returns:
            dict: Parameter grid
        """
        model_type = self.model_params['type']
        
        if model_type == 'random_forest':
            return {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            }
        elif model_type == 'gradient_boosting':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        elif model_type == 'linear':
            return {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
        elif model_type == 'ridge':
            return {
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
            }
        elif model_type == 'lasso':
            return {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
                'selection': ['cyclic', 'random']
            }
        elif model_type == 'elastic_net':
            return {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            }
        elif model_type == 'svr':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'epsilon': [0.01, 0.1, 0.5]
            }
        elif model_type == 'neural_network':
            return {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        elif model_type == 'xgboost':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'num_leaves': [31, 63, 127]
            }
        else:
            return {}
    
    def predict(self, X):
        """
        Generate predictions for new data.
        
        Args:
            X (DataFrame): Feature matrix
            
        Returns:
            array: Predictions
        """
        if self.model is None:
            self.logger.error("Model not trained. Train the model first.")
            return None
            
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model on new data.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): True values
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            self.logger.error("Model not trained. Train the model first.")
            return None
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Calculate directional accuracy (% of times direction is correct)
        y_diff = np.diff(np.append([0], y))
        pred_diff = np.diff(np.append([0], y_pred))
        directional_accuracy = np.mean((y_diff > 0) == (pred_diff > 0))
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def save_model(self, symbol, exchange, model_name, description=None):
        """
        Save the trained model to the database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            model_name (str): Name of the model
            description (str): Model description
            
        Returns:
            str: Model ID
        """
        if self.model is None:
            self.logger.error("No model to save. Train the model first.")
            return None
            
        import pickle
        import base64
        from datetime import datetime
        
        # Serialize the model
        model_bytes = pickle.dumps(self.model)
        model_base64 = base64.b64encode(model_bytes).decode('utf-8')
        
        # Serialize the scaler
        scaler_bytes = pickle.dumps(self.scaler)
        scaler_base64 = base64.b64encode(scaler_bytes).decode('utf-8')
        
        # Create model document
        model_doc = {
            'symbol': symbol,
            'exchange': exchange,
            'model_name': model_name,
            'model_type': 'regressor',
            'algorithm': self.model_params['type'],
            'parameters': self.model_params['params'],
            'model_data': model_base64,
            'scaler_data': scaler_base64,
            'feature_importance': self.feature_importance,
            'created_date': datetime.now(),
            'description': description or f"Regression model for {symbol} {exchange}"
        }
        
        # Insert into database
        result = self.db.models_collection.insert_one(model_doc)
        model_id = str(result.inserted_id)
        
        self.logger.info(f"Model saved to database with ID: {model_id}")
        
        return model_id
    
    def load_model(self, model_id=None, symbol=None, exchange=None, model_name=None):
        """
        Load a model from the database.
        
        Args:
            model_id (str): Model ID
            symbol (str): Trading symbol
            exchange (str): Exchange
            model_name (str): Name of the model
            
        Returns:
            bool: Success/failure
        """
        import pickle
        import base64
        
        # Query database
        query = {}
        if model_id:
            from bson.objectid import ObjectId
            query['_id'] = ObjectId(model_id)
        else:
            if symbol:
                query['symbol'] = symbol
            if exchange:
                query['exchange'] = exchange
            if model_name:
                query['model_name'] = model_name
            query['model_type'] = 'regressor'
        
        # Find model
        model_doc = self.db.models_collection.find_one(query, sort=[('created_date', -1)])
        
        if not model_doc:
            self.logger.error(f"Model not found: {query}")
            return False
            
        try:
            # Deserialize model
            model_base64 = model_doc['model_data']
            model_bytes = base64.b64decode(model_base64)
            self.model = pickle.loads(model_bytes)
            
            # Deserialize scaler
            scaler_base64 = model_doc['scaler_data']
            scaler_bytes = base64.b64decode(scaler_base64)
            self.scaler = pickle.loads(scaler_bytes)
            
            # Load model params
            self.model_params = {
                'type': model_doc['algorithm'],
                'params': model_doc['parameters']
            }
            
            # Load feature importance
            self.feature_importance = model_doc.get('feature_importance')
            
            self.logger.info(f"Model loaded: {model_doc['model_name']} ({model_doc['algorithm']})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def generate_price_prediction(self, symbol, exchange, features_data, save_prediction=True):
        """
        Generate a price prediction using the trained model.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            features_data (DataFrame): Feature data for prediction
            save_prediction (bool): Whether to save prediction to database
            
        Returns:
            dict: Prediction details
        """
        if self.model is None:
            self.logger.error("Model not trained. Train the model first.")
            return None
            
        try:
            # Scale features
            features_scaled = self.scaler.transform(features_data)
            
            # Generate prediction
            price_change_pred = float(self.model.predict(features_scaled)[0])
            
            # Convert to price movement direction
            prediction_value = "up" if price_change_pred > 0 else "down"
            if abs(price_change_pred) < 0.001:  # Very small change
                prediction_value = "neutral"
                
            # Calculate confidence based on the magnitude of the prediction
            # Normalize to a reasonable range (0.5-1.0)
            confidence = min(0.5 + abs(price_change_pred) * 5, 1.0)
            
            # Create prediction document
            from datetime import datetime
            
            prediction_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'date': datetime.now(),
                'for_date': datetime.now(),  # Next day prediction
                'prediction_type': 'price_change',
                'prediction': prediction_value,
                'expected_change_percent': price_change_pred * 100,  # Convert to percentage
                'confidence': confidence,
                'model_id': self.model_params['type'],
                'timeframe': 'daily',
                'supporting_factors': []
            }
            
            # Add supporting factors based on feature importance
            if self.feature_importance:
                # For regression, need to consider direction of impact
                top_features = sorted(
                    self.feature_importance[:10],
                    key=lambda x: abs(x.get('coefficient', x['importance'])) * (-1 if prediction_value == 'down' else 1),
                    reverse=True
                )[:5]  # Top 5 features
                
                prediction_doc['supporting_factors'] = [{
                    'factor': feature['feature'],
                    'weight': feature.get('coefficient', feature['importance'])
                } for feature in top_features]
            
            # Add target price if current price available
            current_price = self._get_current_price(symbol, exchange)
            if current_price:
                prediction_doc['current_price'] = current_price
                prediction_doc['target_price'] = current_price * (1 + price_change_pred)
                
                # Add stop loss and target levels
                if prediction_value == 'up':
                    prediction_doc['stop_loss'] = current_price * (1 - abs(price_change_pred) * 0.5)
                    prediction_doc['take_profit'] = current_price * (1 + abs(price_change_pred) * 1.5)
                else:
                    prediction_doc['stop_loss'] = current_price * (1 + abs(price_change_pred) * 0.5)
                    prediction_doc['take_profit'] = current_price * (1 - abs(price_change_pred) * 1.5)
            
            # Save prediction to database
            if save_prediction:
                self.db.predictions_collection.insert_one(prediction_doc)
                self.logger.info(f"Prediction saved for {symbol} {exchange}")
            
            return prediction_doc
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return None
    
    def _get_current_price(self, symbol, exchange):
        """
        Get the most recent price for a symbol.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            
        Returns:
            float: Current price or None if not available
        """
        try:
            # Get the most recent 1-minute candle
            latest_data = self.db.market_data_collection.find_one(
                {
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': '1min'
                },
                sort=[('timestamp', -1)]
            )
            
            if latest_data and 'close' in latest_data:
                return latest_data['close']
                
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current price: {e}")
            return None
    
    def walk_forward_validation(self, X, y, window_size=252, step_size=22, min_train_size=504):
        """
        Perform walk-forward validation for time series data.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            window_size (int): Size of each validation window
            step_size (int): Number of steps to move window forward
            min_train_size (int): Minimum size of training set
            
        Returns:
            dict: Validation results
        """
        if X.shape[0] < min_train_size + window_size:
            self.logger.error(f"Not enough data for walk-forward validation. Need at least {min_train_size + window_size} samples.")
            return None
            
        self.logger.info("Starting walk-forward validation")
        
        # Initialize results storage
        results = {
            'mse': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'directional_accuracy': []
        }
        
        # Scale the entire dataset
        X_scaled = self.scaler.fit_transform(X)
        
        # Determine number of validation windows
        n_splits = (X.shape[0] - min_train_size) // step_size
        
        for i in range(n_splits):
            # Calculate indices
            train_end = min_train_size + i * step_size
            val_start = train_end
            val_end = min(val_start + window_size, X.shape[0])
            
            # Skip if validation window is too small
            if val_end - val_start < window_size / 2:
                continue
                
            # Split data
            X_train, X_val = X_scaled[:train_end], X_scaled[val_start:val_end]
            y_train, y_val = y.iloc[:train_end], y.iloc[val_start:val_end]
            
            # Train model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            
            # Calculate metrics
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            # Calculate directional accuracy
            y_diff = np.diff(np.append([y_val.iloc[0]], y_val))
            pred_diff = np.diff(np.append([y_pred[0]], y_pred))
            directional_accuracy = np.mean((y_diff > 0) == (pred_diff > 0))
            
            # Store results
            results['mse'].append(mse)
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['r2'].append(r2)
            results['directional_accuracy'].append(directional_accuracy)
            
            self.logger.info(f"Window {i+1}/{n_splits}: RMSE={rmse:.4f}, R²={r2:.4f}, Dir Acc={directional_accuracy:.4f}")
        
        # Calculate average metrics
        avg_results = {
            'avg_mse': np.mean(results['mse']),
            'avg_rmse': np.mean(results['rmse']),
            'avg_mae': np.mean(results['mae']),
            'avg_r2': np.mean(results['r2']),
            'avg_directional_accuracy': np.mean(results['directional_accuracy']),
            'window_metrics': results
        }
        
        self.logger.info(f"Walk-forward validation complete. Average RMSE: {avg_results['avg_rmse']:.4f}")
        
        return avg_results