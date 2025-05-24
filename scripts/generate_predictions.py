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
        
        # Import prediction components
        try:
            from ml.prediction.daily_predictor import DailyPredictor
            predictor = DailyPredictor(db, logger)
            logger.info("DailyPredictor loaded successfully")
        except ImportError as e:
            logger.error(f"Failed to import DailyPredictor: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing DailyPredictor: {e}")
            return False
        
        # Generate predictions for each instrument
        prediction_count = 0
        
        for instrument in instruments:
            symbol = instrument['symbol']
            exchange = instrument['exchange']
            
            # Generate prediction
            try:
                logger.info(f"Starting prediction generation for {symbol}:{exchange}")
                
                result = predictor.generate_prediction(symbol, exchange)
                
                if result:
                    prediction_count += 1
                    pred_direction = result.get('prediction', 'unknown')
                    confidence = result.get('confidence', 0)
                    logger.info(f"Generated prediction for {symbol}:{exchange}: {pred_direction} (confidence: {confidence:.2f})")
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