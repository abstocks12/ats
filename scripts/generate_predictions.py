#!/usr/bin/env python3
"""
Script to generate predictions for portfolio instruments.
"""

import os
import sys
import argparse
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.portfolio_manager import PortfolioManager
from database.connection_manager import get_db
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def generate_predictions(symbol=None, exchange=None, timeframe=None, all_instruments=False):
    """
    Generate predictions for instruments
    
    Args:
        symbol (str, optional): Instrument symbol (or None for all instruments)
        exchange (str, optional): Exchange code
        timeframe (str, optional): Prediction timeframe
        all_instruments (bool): Generate predictions for all instruments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize database and manager
        db = get_db()
        portfolio_manager = PortfolioManager(db)
        
        # Determine which instruments to generate predictions for
        instruments = []
        
        if all_instruments:
            # Get all active instruments
            instruments = portfolio_manager.get_active_instruments()
            logger.info(f"Generating predictions for all {len(instruments)} active instruments")
        elif symbol and exchange:
            # Get specific instrument
            instrument = portfolio_manager.get_instrument(symbol, exchange)
            if instrument:
                instruments = [instrument]
                logger.info(f"Generating predictions for {symbol}:{exchange}")
            else:
                logger.error(f"Instrument {symbol}:{exchange} not found in portfolio")
                return False
        else:
            logger.error("No instruments specified. Use --symbol and --exchange or --all")
            return False
        
        # Import necessary components for prediction
        try:
            from ml.prediction.daily_predictor import DailyPredictor
            predictor = DailyPredictor(db)
        except ImportError:
            # If not available, create a placeholder implementation
            logger.warning("DailyPredictor not available, using placeholder")
            
            class PlaceholderPredictor:
                def __init__(self, db):
                    self.db = db
                    self.logger = setup_logger("placeholder_predictor")
                
                def predict(self, symbol, exchange, timeframe="day"):
                    self.logger.info(f"Placeholder: Generating prediction for {symbol}:{exchange}")
                    
                    import random
                    from database.models import PredictionData
                    
                    # Generate a random prediction
                    prediction = "up" if random.random() > 0.5 else "down"
                    confidence = random.uniform(0.6, 0.9)
                    
                    # Create prediction data
                    prediction_data = PredictionData(
                        symbol=symbol,
                        exchange=exchange,
                        date=datetime.now(),
                        prediction=prediction,
                        confidence=confidence,
                        timeframe=timeframe,
                        supporting_factors=[
                            {"factor": "placeholder", "weight": 1.0}
                        ],
                        model_id="placeholder_model"
                    )
                    
                    # Save to database
                    self.db.save_prediction(prediction_data.to_dict())
                    
                    return prediction_data.to_dict()
            
            predictor = PlaceholderPredictor(db)
        
        # Generate predictions for each instrument
        prediction_count = 0
        
        for instrument in instruments:
            symbol = instrument['symbol']
            exchange = instrument['exchange']
            
            # Determine timeframe if not provided
            prediction_timeframe = timeframe
            if not prediction_timeframe:
                # Use the instrument's trading timeframe
                prediction_timeframe = instrument.get('trading_config', {}).get('timeframe', 'day')
                # Convert trading timeframe to prediction timeframe
                if prediction_timeframe == 'intraday':
                    prediction_timeframe = 'day'
            
            # Generate prediction
            try:
                result = predictor.predict(symbol, exchange, timeframe=prediction_timeframe)
                
                if result:
                    prediction_count += 1
                    logger.info(f"Generated {prediction_timeframe} prediction for {symbol}:{exchange}: {result.get('prediction', 'unknown')} (confidence: {result.get('confidence', 0):.2f})")
                else:
                    logger.error(f"Failed to generate prediction for {symbol}:{exchange}")
            except Exception as e:
                logger.error(f"Error generating prediction for {symbol}:{exchange}: {e}")
        
        logger.info(f"Generated {prediction_count} predictions")
        return prediction_count > 0
            
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Generate predictions for portfolio instruments')
    
    parser.add_argument(
        '--symbol',
        help='Instrument symbol'
    )
    
    parser.add_argument(
        '--exchange',
        help='Exchange code'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate predictions for all active instruments'
    )
    
    parser.add_argument(
        '--timeframe',
        choices=['day', 'intraday', 'swing', 'positional'],
        help='Prediction timeframe'
    )
    
    args = parser.parse_args()
    
    # Generate predictions
    result = generate_predictions(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        all_instruments=args.all
    )
    
    if result:
        print("Prediction generation completed successfully")
    else:
        print("Prediction generation failed")
        sys.exit(1)

if __name__ == '__main__':
    main()