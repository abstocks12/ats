# ml/models/ensemble_predictor.py
import numpy as np
import pandas as pd
import logging
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
from datetime import datetime

class EnsemblePredictor:
    """
    Ensemble predictor combining multiple classification and regression models.
    """
    
    def __init__(self, db_connector, logger=None):
        """
        Initialize the ensemble predictor.
        
        Args:
            db_connector: MongoDB connector
            logger: Logger instance
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        self.classifier = None
        self.regressor = None
        self.classifier_models = []
        self.regressor_models = []
        self.classifier_scaler = StandardScaler()
        self.regressor_scaler = StandardScaler()
        
    def add_classifier(self, model, name=None, weight=1):
        """
        Add a classifier to the ensemble.
        
        Args:
            model: Classification model instance
            name (str): Model name
            weight (float): Voting weight
            
        Returns:
            bool: Success/failure
        """
        try:
            if name is None:
                name = f"classifier_{len(self.classifier_models)}"
                
            self.classifier_models.append((name, model, weight))
            self.logger.info(f"Added classifier {name} to ensemble with weight {weight}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding classifier: {e}")
            return False
    
    def add_regressor(self, model, name=None, weight=1):
        """
        Add a regressor to the ensemble.
        
        Args:
            model: Regression model instance
            name (str): Model name
            weight (float): Voting weight
            
        Returns:
            bool: Success/failure
        """
        try:
            if name is None:
                name = f"regressor_{len(self.regressor_models)}"
                
            self.regressor_models.append((name, model, weight))
            self.logger.info(f"Added regressor {name} to ensemble with weight {weight}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding regressor: {e}")
            return False
    
    def build_ensemble(self, predictor_type='both'):
        """
        Build ensemble models based on added models.
        
        Args:
            predictor_type (str): Type of ensemble to build 
                ('classifier', 'regressor', or 'both')
                
        Returns:
            bool: Success/failure
        """
        success = True
        
        if predictor_type in ['classifier', 'both']:
            if len(self.classifier_models) < 2:
                self.logger.error("Not enough classifiers. Add at least 2 classifiers.")
                success = False
            else:
                try:
                    self.classifier = VotingClassifier(
                        estimators=[(name, model) for name, model, _ in self.classifier_models],
                        weights=[weight for _, _, weight in self.classifier_models],
                        voting='soft'
                    )
                    self.logger.info(f"Built classifier ensemble with {len(self.classifier_models)} models")
                except Exception as e:
                    self.logger.error(f"Error building classifier ensemble: {e}")
                    success = False
        
        if predictor_type in ['regressor', 'both']:
            if len(self.regressor_models) < 2:
                self.logger.error("Not enough regressors. Add at least 2 regressors.")
                success = False
            else:
                try:
                    self.regressor = VotingRegressor(
                        estimators=[(name, model) for name, model, _ in self.regressor_models],
                        weights=[weight for _, _, weight in self.regressor_models]
                    )
                    self.logger.info(f"Built regressor ensemble with {len(self.regressor_models)} models")
                except Exception as e:
                    self.logger.error(f"Error building regressor ensemble: {e}")
                    success = False
        
        return success
    
    def train_classifier(self, X, y, test_size=0.2, time_series_split=True, shuffle=False):
        """
        Train the classification ensemble.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            test_size (float): Proportion of data to use for testing
            time_series_split (bool): Whether to use time series split (vs random)
            shuffle (bool): Whether to shuffle data (ignored if time_series_split is True)
            
        Returns:
            dict: Training results including metrics
        """
        if self.classifier is None:
            self.logger.error("Classifier ensemble not built. Call build_ensemble() first.")
            return None
            
        self.logger.info("Training classifier ensemble")
        
        # Scale features
        X_scaled = self.classifier_scaler.fit_transform(X)
        
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
            
        # Train the ensemble
        self.classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        self.logger.info(f"Classifier ensemble training complete. Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'test_size': test_size
        }
    
    def train_regressor(self, X, y, test_size=0.2, time_series_split=True, shuffle=False):
        """
        Train the regression ensemble.
        
        Args:
            X (DataFrame): Feature matrix
            y (Series): Target variable
            test_size (float): Proportion of data to use for testing
            time_series_split (bool): Whether to use time series split (vs random)
            shuffle (bool): Whether to shuffle data (ignored if time_series_split is True)
            
        Returns:
            dict: Training results including metrics
        """
        if self.regressor is None:
            self.logger.error("Regressor ensemble not built. Call build_ensemble() first.")
            return None
            
        self.logger.info("Training regressor ensemble")
        
        # Scale features
        X_scaled = self.regressor_scaler.fit_transform(X)
        
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
            
        # Train the ensemble
        self.regressor.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.regressor.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate directional accuracy
        y_diff = np.diff(np.append([0], y_test))
        pred_diff = np.diff(np.append([0], y_pred))
        directional_accuracy = np.mean((y_diff > 0) == (pred_diff > 0))
        
        self.logger.info(f"Regressor ensemble training complete. RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'directional_accuracy': directional_accuracy,
            'test_size': test_size
        }
    
    def predict_classification(self, X, probabilities=False):
        """
        Generate classification predictions for new data.
        
        Args:
            X (DataFrame): Feature matrix
            probabilities (bool): Whether to return class probabilities
            
        Returns:
            array: Predictions or probabilities
        """
        if self.classifier is None:
            self.logger.error("Classifier ensemble not trained. Train the model first.")
            return None
            
        X_scaled = self.classifier_scaler.transform(X)
        
        if probabilities and hasattr(self.classifier, 'predict_proba'):
            return self.classifier.predict_proba(X_scaled)
        else:
            return self.classifier.predict(X_scaled)
    
    def predict_regression(self, X):
        """
        Generate regression predictions for new data.
        
        Args:
            X (DataFrame): Feature matrix
            
        Returns:
            array: Predictions
        """
        if self.regressor is None:
            self.logger.error("Regressor ensemble not trained. Train the model first.")
            return None
            
        X_scaled = self.regressor_scaler.transform(X)
        return self.regressor.predict(X_scaled)
    
    def generate_market_prediction(self, symbol, exchange, features_data, save_prediction=True):
        """
        Generate a market prediction using both classification and regression ensembles.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            features_data (DataFrame): Feature data for prediction
            save_prediction (bool): Whether to save prediction to database
            
        Returns:
            dict: Prediction details
        """
        prediction_doc = {
            'symbol': symbol,
            'exchange': exchange,
            'date': datetime.now(),
            'for_date': datetime.now(),  # Next day prediction
            'prediction_type': 'ensemble',
            'timeframe': 'daily'
        }
        
        # Classification prediction
        if self.classifier is not None:
            try:
                X_class_scaled = self.classifier_scaler.transform(features_data)
                class_pred = self.classifier.predict(X_class_scaled)[0]
                
                if hasattr(self.classifier, 'predict_proba'):
                    proba = self.classifier.predict_proba(X_class_scaled)[0]
                    class_confidence = float(max(proba))
                else:
                    class_confidence = 0.7  # Default if no probabilities
                
                # Map prediction
                class_prediction = "up" if class_pred == 1 else "down"
                if hasattr(self.classifier, 'classes_') and len(self.classifier.classes_) > 2:
                    if class_pred == 0:
                        class_prediction = "down"
                    elif class_pred == 1:
                        class_prediction = "neutral"
                    else:
                        class_prediction = "up"
                
                prediction_doc['classification_prediction'] = class_prediction
                prediction_doc['classification_confidence'] = class_confidence
                
            except Exception as e:
                self.logger.error(f"Error generating classification prediction: {e}")
        
        # Regression prediction
        if self.regressor is not None:
            try:
                X_reg_scaled = self.regressor_scaler.transform(features_data)
                reg_pred = float(self.regressor.predict(X_reg_scaled)[0])
                
                # Convert to movement direction
                reg_prediction = "up" if reg_pred > 0 else "down"
                if abs(reg_pred) < 0.001:  # Very small change
                    reg_prediction = "neutral"
                
                # Calculate confidence based on magnitude
                reg_confidence = min(0.5 + abs(reg_pred) * 5, 1.0)
                
                prediction_doc['regression_prediction'] = reg_prediction
                prediction_doc['regression_confidence'] = reg_confidence
                prediction_doc['expected_change_percent'] = reg_pred * 100  # Convert to percentage
                
                # Get current price if available
                current_price = self._get_current_price(symbol, exchange)
                if current_price:
                    prediction_doc['current_price'] = current_price
                    prediction_doc['target_price'] = current_price * (1 + reg_pred)
                
            except Exception as e:
                self.logger.error(f"Error generating regression prediction: {e}")
        
        # Combine predictions if both are available
        if 'classification_prediction' in prediction_doc and 'regression_prediction' in prediction_doc:
            class_pred = prediction_doc['classification_prediction']
            reg_pred = prediction_doc['regression_prediction']
            class_conf = prediction_doc['classification_confidence']
            reg_conf = prediction_doc['regression_confidence']
            
            # Simple weighted average based on confidence
            total_weight = class_conf + reg_conf
            
            if class_pred == reg_pred:
                # Both models agree
                final_pred = class_pred
                final_conf = (class_conf + reg_conf) / 2  # Average confidence
            else:
                # Models disagree, use the more confident one
                if class_conf > reg_conf:
                    final_pred = class_pred
                    final_conf = class_conf * 0.9  # Reduce confidence slightly due to disagreement
                else:
                    final_pred = reg_pred
                    final_conf = reg_conf * 0.9  # Reduce confidence slightly due to disagreement
            
            prediction_doc['prediction'] = final_pred
            prediction_doc['confidence'] = final_conf
        elif 'classification_prediction' in prediction_doc:
            prediction_doc['prediction'] = prediction_doc['classification_prediction']
            prediction_doc['confidence'] = prediction_doc['classification_confidence']
        elif 'regression_prediction' in prediction_doc:
            prediction_doc['prediction'] = prediction_doc['regression_prediction']
            prediction_doc['confidence'] = prediction_doc['regression_confidence']
        else:
            self.logger.error("No predictions generated")
            return None
        
        # Add supporting factors
        prediction_doc['supporting_factors'] = []
        
        # Save prediction to database
        if save_prediction:
            self.db.predictions_collection.insert_one(prediction_doc)
            self.logger.info(f"Ensemble prediction saved for {symbol} {exchange}")
        
        return prediction_doc
    
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
    
    def save_models(self, symbol, exchange, ensemble_name, description=None):
        """
        Save the ensemble models to the database.
        
        Args:
            symbol (str): Trading symbol
            exchange (str): Exchange
            ensemble_name (str): Name of the ensemble
            description (str): Model description
            
        Returns:
            str: Ensemble ID
        """
        import pickle
        import base64
        from datetime import datetime
        
        # Check if we have models to save
        if self.classifier is None and self.regressor is None:
            self.logger.error("No models to save. Train the ensembles first.")
            return None
        
        # Create ensemble document
        ensemble_doc = {
            'symbol': symbol,
            'exchange': exchange,
            'ensemble_name': ensemble_name,
            'created_date': datetime.now(),
            'description': description or f"Ensemble model for {symbol} {exchange}",
            'has_classifier': self.classifier is not None,
            'has_regressor': self.regressor is not None,
            'classifier_models': [name for name, _, _ in self.classifier_models],
            'regressor_models': [name for name, _, _ in self.regressor_models]
        }
        
        # Serialize models if available
        if self.classifier:
            classifier_bytes = pickle.dumps(self.classifier)
            ensemble_doc['classifier_data'] = base64.b64encode(classifier_bytes).decode('utf-8')
            
            # Serialize classifier scaler
            classifier_scaler_bytes = pickle.dumps(self.classifier_scaler)
            ensemble_doc['classifier_scaler_data'] = base64.b64encode(classifier_scaler_bytes).decode('utf-8')
        
        if self.regressor:
            regressor_bytes = pickle.dumps(self.regressor)
            ensemble_doc['regressor_data'] = base64.b64encode(regressor_bytes).decode('utf-8')
            
            # Serialize regressor scaler
            regressor_scaler_bytes = pickle.dumps(self.regressor_scaler)
            ensemble_doc['regressor_scaler_data'] = base64.b64encode(regressor_scaler_bytes).decode('utf-8')
        
        # Insert into database
        result = self.db.ensemble_models_collection.insert_one(ensemble_doc)
        ensemble_id = str(result.inserted_id)
        
        self.logger.info(f"Ensemble saved to database with ID: {ensemble_id}")
        
        return ensemble_id
    
    def load_models(self, ensemble_id=None, symbol=None, exchange=None, ensemble_name=None):
        """
        Load ensemble models from the database.
        
        Args:
            ensemble_id (str): Ensemble ID
            symbol (str): Trading symbol
            exchange (str): Exchange
            ensemble_name (str): Name of the ensemble
            
        Returns:
            bool: Success/failure
        """
        import pickle
        import base64
        
        # Query database
        query = {}
        if ensemble_id:
            from bson.objectid import ObjectId
            query['_id'] = ObjectId(ensemble_id)
        else:
            if symbol:
                query['symbol'] = symbol
            if exchange:
                query['exchange'] = exchange
            if ensemble_name:
                query['ensemble_name'] = ensemble_name
        
        # Find ensemble
        ensemble_doc = self.db.ensemble_models_collection.find_one(query, sort=[('created_date', -1)])
        
        if not ensemble_doc:
            self.logger.error(f"Ensemble not found: {query}")
            return False
            
        try:
            # Deserialize classifier if available
            if ensemble_doc.get('has_classifier') and 'classifier_data' in ensemble_doc:
                classifier_base64 = ensemble_doc['classifier_data']
                classifier_bytes = base64.b64decode(classifier_base64)
                self.classifier = pickle.loads(classifier_bytes)
                
                # Deserialize classifier scaler
                classifier_scaler_base64 = ensemble_doc['classifier_scaler_data']
                classifier_scaler_bytes = base64.b64decode(classifier_scaler_base64)
                self.classifier_scaler = pickle.loads(classifier_scaler_bytes)
                
                self.logger.info(f"Loaded classifier ensemble from {ensemble_doc['ensemble_name']}")
            
            # Deserialize regressor if available
            if ensemble_doc.get('has_regressor') and 'regressor_data' in ensemble_doc:
                regressor_base64 = ensemble_doc['regressor_data']
                regressor_bytes = base64.b64decode(regressor_base64)
                self.regressor = pickle.loads(regressor_bytes)
                
                # Deserialize regressor scaler
                regressor_scaler_base64 = ensemble_doc['regressor_scaler_data']
                regressor_scaler_bytes = base64.b64decode(regressor_scaler_base64)
                self.regressor_scaler = pickle.loads(regressor_scaler_bytes)
                
                self.logger.info(f"Loaded regressor ensemble from {ensemble_doc['ensemble_name']}")
            
            # Update model lists
            self.classifier_models = [(name, None, 1.0) for name in ensemble_doc.get('classifier_models', [])]
            self.regressor_models = [(name, None, 1.0) for name in ensemble_doc.get('regressor_models', [])]
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble: {e}")
            return False