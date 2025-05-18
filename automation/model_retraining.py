# automation/model_retraining.py
import logging
from datetime import datetime, timedelta
import os
import sys

class ModelRetraining:
    """
    Automated model retraining system.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the model retraining system.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'retraining_interval_days': 30,  # Retrain models every 30 days
            'min_samples': 100,  # Minimum samples for training
            'validation_split': 0.2,  # Validation data split
            'training_epochs': 100,  # Training epochs
            'early_stopping_patience': 10,  # Early stopping patience
            'batch_size': 32,  # Training batch size
            'use_class_weights': True,  # Use class weights for imbalanced data
            'save_history': True,  # Save training history
            'learning_rate': 0.001,  # Learning rate
            'max_training_time': 3600,  # Maximum training time in seconds
            'max_memory_usage': 1024  # Maximum memory usage in MB
        }
        
        self.logger.info("Model retraining system initialized")
        
    def retrain_all_models(self):
        """
        Retrain all models that are due for retraining.
        
        Returns:
            dict: Retraining results
        """
        try:
            self.logger.info("Starting model retraining")
            
            # Get models due for retraining
            due_models = self._get_models_due_for_retraining()
            
            if not due_models:
                self.logger.info("No models due for retraining")
                return {"status": "success", "models_retrained": 0, "message": "No models due for retraining"}
                
            self.logger.info(f"Found {len(due_models)} models due for retraining")
            
            # Retrain each model
            results = []
            
            for model in due_models:
                try:
                    self.logger.info(f"Retraining model: {model['model_id']}")
                    
                    # Retrain model
                    result = self.retrain_model(
                        model['symbol'],
                        model['exchange'],
                        model['model_type']
                    )
                    
                    # Add to results
                    results.append({
                        "model_id": model['model_id'],
                        "status": result.get('status', 'failed'),
                        "message": result.get('message', ''),
                        "accuracy": result.get('accuracy', 0)
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error retraining model {model['model_id']}: {e}")
                    
                    # Add to results
                    results.append({
                        "model_id": model['model_id'],
                        "status": "failed",
                        "message": str(e),
                        "accuracy": 0
                    })
            
            # Compile results
            retraining_results = {
                "status": "success",
                "models_retrained": len(results),
                "successful": sum(1 for r in results if r['status'] == 'success'),
                "failed": sum(1 for r in results if r['status'] == 'failed'),
                "results": results
            }
            
            self.logger.info(f"Model retraining completed: {retraining_results['successful']} successful, {retraining_results['failed']} failed")
            
            return retraining_results
            
        except Exception as e:
            self.logger.error(f"Error in model retraining: {e}")
            return {"status": "failed", "error": str(e)}
            
    def retrain_model(self, symbol, exchange, model_type):
        """
        Retrain a specific model.
        
        Args:
            symbol (str): Symbol name
            exchange (str): Exchange name
            model_type (str): Model type (daily_predictor, overnight_gap, etc.)
            
        Returns:
            dict: Retraining result
        """
        try:
            self.logger.info(f"Retraining {model_type} model for {symbol}/{exchange}")
            
            # Get model data
            model_id = f"{symbol}_{exchange}_{model_type}"
            
            # Get training data
            training_data = self._get_training_data(symbol, exchange, model_type)
            
            if not training_data or len(training_data) < self.config['min_samples']:
                message = f"Insufficient training data: {len(training_data) if training_data else 0} samples"
                self.logger.warning(message)
                return {"status": "failed", "message": message}
                
            # Initialize appropriate trainer
            trainer = self._get_model_trainer(model_type)
            
            if not trainer:
                message = f"Unsupported model type: {model_type}"
                self.logger.error(message)
                return {"status": "failed", "message": message}
                
            # Retrain model
            result = trainer.train(
                symbol=symbol,
                exchange=exchange,
                training_data=training_data,
                config=self.config
            )
            
            # Update model metadata
            if result.get('status') == 'success':
                self._update_model_metadata(model_id, result)
                
            # Log result
            if result.get('status') == 'success':
                self.logger.info(f"Model retraining successful: {model_id}, accuracy: {result.get('accuracy', 0):.4f}")
            else:
                self.logger.error(f"Model retraining failed: {model_id}, error: {result.get('message', 'Unknown error')}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error retraining {model_type} model for {symbol}/{exchange}: {e}")
            return {"status": "failed", "message": str(e)}
            
    def _get_models_due_for_retraining(self):
        """
        Get models that are due for retraining.
        
        Returns:
            list: List of models due for retraining
        """
        try:
            if not self.db:
                return []
                
            # Calculate cutoff date
            cutoff_date = datetime.now() - timedelta(days=self.config['retraining_interval_days'])
            
            # Get models from database
            cursor = self.db.ml_models.find({
                "$or": [
                    {"last_trained": {"$lt": cutoff_date}},
                    {"last_trained": {"$exists": False}}
                ]
            })
            
            # Convert to list
            models = list(cursor)
            
            # Check for models with poor performance
            performance_cursor = self.db.model_evaluations.find({
                "accuracy": {"$lt": 0.55},  # Models with less than 55% accuracy
                "evaluated_at": {"$gt": datetime.now() - timedelta(days=7)}  # Evaluated recently
            })
            
            # Get model IDs with poor performance
            poor_performance_ids = [p['model_id'] for p in performance_cursor]
            
            # Add models with poor performance that aren't already in the list
            if poor_performance_ids:
                additional_models = list(self.db.ml_models.find({
                    "model_id": {"$in": poor_performance_ids},
                    "_id": {"$nin": [m['_id'] for m in models]}
                }))
                
                models.extend(additional_models)
                
            return models
            
        except Exception as e:
            self.logger.error(f"Error getting models due for retraining: {e}")
            return []
            
    def _get_training_data(self, symbol, exchange, model_type):
        """
        Get training data for a model.
        
        Args:
            symbol (str): Symbol name
            exchange (str): Exchange name
            model_type (str): Model type
            
        Returns:
            list: Training data
        """
        try:
            if not self.db:
                return []
                
            # Get appropriate feature generator based on model type
            feature_generator = self._get_feature_generator(model_type)
            
            if not feature_generator:
                self.logger.error(f"Unsupported model type for training data: {model_type}")
                return []
                
            # Get training period
            if model_type == 'daily_predictor':
                # Use 2 years of data for daily predictions
                days = 2 * 365
            elif model_type == 'overnight_gap':
                # Use 1 year of data for overnight gap predictions
                days = 365
            elif model_type == 'stock_classifier':
                # Use 3 years of data for stock classification
                days = 3 * 365
            else:
                # Default to 1 year
                days = 365
                
            # Get data start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Generate training data
            training_data = feature_generator.generate_training_data(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=datetime.now()
            )
            
            return training_data
            
        except Exception as e:
            self.logger.error(f"Error getting training data for {model_type} model ({symbol}/{exchange}): {e}")
            return []
            
    def _get_model_trainer(self, model_type):
        """
        Get appropriate model trainer.
        
        Args:
            model_type (str): Model type
            
        Returns:
            object: Model trainer
        """
        try:
            # Import appropriate trainer based on model type
            if model_type == 'daily_predictor':
                from ml.training.daily_predictor_trainer import DailyPredictorTrainer
                return DailyPredictorTrainer(self.db)
            elif model_type == 'overnight_gap':
                from ml.training.overnight_gap_trainer import OvernightGapTrainer
                return OvernightGapTrainer(self.db)
            elif model_type == 'stock_classifier':
                from ml.training.stock_classifier_trainer import StockClassifierTrainer
                return StockClassifierTrainer(self.db)
            else:
                self.logger.error(f"Unsupported model type: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting model trainer for {model_type}: {e}")
            return None
            
    def _get_feature_generator(self, model_type):
        """
        Get appropriate feature generator.
        
        Args:
            model_type (str): Model type
            
        Returns:
            object: Feature generator
        """
        try:
            # Import appropriate feature generator based on model type
            if model_type == 'daily_predictor':
                from ml.features.technical_features import TechnicalFeatureGenerator
                return TechnicalFeatureGenerator(self.db)
            elif model_type == 'overnight_gap':
                from ml.features.technical_features import TechnicalFeatureGenerator
                return TechnicalFeatureGenerator(self.db)
            elif model_type == 'stock_classifier':
                from ml.features.technical_features import TechnicalFeatureGenerator
                return TechnicalFeatureGenerator(self.db)
            else:
                self.logger.error(f"Unsupported model type for feature generator: {model_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting feature generator for {model_type}: {e}")
            return None
            
    def _update_model_metadata(self, model_id, training_result):
        """
        Update model metadata after retraining.
        
        Args:
            model_id (str): Model ID
            training_result (dict): Training result
        """
        try:
            if not self.db:
                return
                
            # Update model metadata
            self.db.ml_models.update_one(
                {"model_id": model_id},
                {"$set": {
                    "last_trained": datetime.now(),
                    "training_accuracy": training_result.get('accuracy', 0),
                    "validation_accuracy": training_result.get('validation_accuracy', 0),
                    "training_loss": training_result.get('loss', 0),
                    "validation_loss": training_result.get('validation_loss', 0),
                    "training_samples": training_result.get('training_samples', 0),
                    "validation_samples": training_result.get('validation_samples', 0),
                    "training_time": training_result.get('training_time', 0),
                    "model_version": training_result.get('model_version', 1)
                }},
                upsert=True
            )
            
            # Store training history if available
            if 'history' in training_result and self.config['save_history']:
                history_id = f"{model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                self.db.training_history.insert_one({
                    "history_id": history_id,
                    "model_id": model_id,
                    "timestamp": datetime.now(),
                    "history": training_result['history']
                })
                
        except Exception as e:
            self.logger.error(f"Error updating model metadata for {model_id}: {e}")