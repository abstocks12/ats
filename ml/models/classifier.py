# ml/models/classifier.py
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

class MarketClassifier:
    """Classification models for market prediction tasks."""
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the classifier.
        
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
        Build a classification model.
        
        Args:
            model_type (str): Type of model to build. Options:
                'random_forest', 'gradient_boosting', 'logistic', 'svm', 'neural_network',
                'xgboost', 'lightgbm'
            params (dict): Model hyperparameters
            
        Returns:
            The model instance
        """
        self.logger.info(f"Building {model_type} classification model")
        
        default_params = self._get_default_params(model_type)
        params = params or default_params
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(**params)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(**params)
        elif model_type == 'logistic':
            self.model = LogisticRegression(**params)
        elif model_type == 'svm':
            self.model = SVC(**params)
        elif model_type == 'neural_network':
            self.model = MLPClassifier(**params)
        elif model_type == 'xgboost':
            self.model = XGBClassifier(**params)
        elif model_type == 'lightgbm':
            self.model = LGBMClassifier(**params)
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
        elif model_type == 'logistic':
            return {
                'C': 1.0,
                'penalty': 'l2',
                'solver': 'liblinear',
                'random_state': 42
            }
        elif model_type == 'svm':
            return {
                'C': 1.0,
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
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
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'random_state': 42
            }
        elif model_type == 'lightgbm':
            return {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'objective': 'binary',
                'random_state': 42
            }
        else:
            return {}
    
    def train(self, X, y, test_size=0.2, time_series_split=True, shuffle=False):
        """
        Train the classification model.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            test_size (float): Proportion of data to use for testing
            time_series_split (bool): Whether to use time series split (vs random)
            shuffle (bool): Whether to shuffle data (ignored if time_series_split is True)
            
        Returns:
            dict: Training results including accuracy and other metrics
        """
        if self.model is None:
            self.logger.error("Model not built. Call build_model() first.")
            return None
            
        self.logger.info(f"Training {self.model_params['type']} classification model")
        
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
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Store feature importance if available
        self._compute_feature_importance(X)
        
        self.logger.info(f"Model training complete. Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
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
        
        if self.model_params['type'] in importance_models and hasattr(self.model, 'feature_importances_'):
            feature_names = X.columns.tolist()
            importances = self.model.feature_importances_
            
            # Sort features by importance
            indices = np.argsort(importances)[::-1]
            
            self.feature_importance = [{
                'feature': feature_names[i],
                'importance': float(importances[i])
            } for i in indices]
        else:
            self.feature_importance = None
    
    def hyperparameter_tuning(self, X, y, param_grid=None, cv=5, scoring='accuracy', time_series_cv=True):
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
        elif model_type == 'logistic':
            return {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            }
        elif model_type == 'svm':
            return {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.1, 0.01]
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
    
    def predict(self, X, probabilities=False):
        """
        Generate predictions for new data.
        
        Args:
            X (DataFrame): Feature matrix
            probabilities (bool): Whether to return class probabilities
            
        Returns:
            array: Predictions or probabilities
        """
        if self.model is None:
            self.logger.error("Model not trained. Train the model first.")
            return None
            
        X_scaled = self.scaler.transform(X)
        
        if probabilities and hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_scaled)
        else:
            return self.model.predict(X_scaled)
    
    def evaluate(self, X, y):
        """
        Evaluate model on new data.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            self.logger.error("Model not trained. Train the model first.")
            return None
            
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted')
        recall = recall_score(y, y_pred, average='weighted')
        f1 = f1_score(y, y_pred, average='weighted')
        
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'classification_report': report
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
            'model_type': 'classifier',
            'algorithm': self.model_params['type'],
            'parameters': self.model_params['params'],
            'model_data': model_base64,
            'scaler_data': scaler_base64,
            'feature_importance': self.feature_importance,
            'created_date': datetime.now(),
            'description': description or f"Classification model for {symbol} {exchange}"
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
            query['model_type'] = 'classifier'
        
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
    
    def generate_market_prediction(self, symbol, exchange, features_data, save_prediction=True):
        """
        Generate a market prediction using the trained model.
        
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
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                prediction_class = self.model.predict(features_scaled)[0]
                confidence = float(max(probabilities))
            else:
                prediction_class = self.model.predict(features_scaled)[0]
                confidence = 0.7  # Default confidence if probabilities not available
            
            # Map prediction to up/down/neutral
            prediction_value = "up" if prediction_class == 1 else "down"
            if hasattr(self.model, 'classes_') and len(self.model.classes_) > 2:
                if prediction_class == 0:
                    prediction_value = "down"
                elif prediction_class == 1:
                    prediction_value = "neutral"
                else:
                    prediction_value = "up"
            
            # Create prediction document
            from datetime import datetime
            
            prediction_doc = {
                'symbol': symbol,
                'exchange': exchange,
                'date': datetime.now(),
                'for_date': datetime.now(),  # Next day prediction
                'prediction_type': 'price_movement',
                'prediction': prediction_value,
                'confidence': confidence,
                'model_id': self.model_params['type'],
                'timeframe': 'daily',
                'supporting_factors': []
            }
            
            # Add supporting factors based on feature importance
            if self.feature_importance:
                top_features = self.feature_importance[:5]  # Top 5 features
                prediction_doc['supporting_factors'] = [{
                    'factor': feature['feature'],
                    'weight': feature['importance']
                } for feature in top_features]
            
            # Save prediction to database
            if save_prediction:
                self.db.predictions_collection.insert_one(prediction_doc)
                self.logger.info(f"Prediction saved for {symbol} {exchange}")
            
            return prediction_doc
            
        except Exception as e:
            self.logger.error(f"Error generating prediction: {e}")
            return None