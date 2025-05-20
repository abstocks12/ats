#!/usr/bin/env python3
"""
Minimal test script for zerodha data collection
"""

import os
import sys
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database connection
from database.connection_manager import get_db
from data.market.zerodha_connector import ZerodhaConnector

def test_minimal(symbol, exchange, timeframe="day"):
    """Minimal test for ZerodhaConnector"""
    try:
        # Get database connection
        db = get_db()
        logger.info("Database connection established")
        
        # Create zerodha connector
        zerodha = ZerodhaConnector()
        logger.info("ZerodhaConnector created")
        
        # Set date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get data
        logger.info(f"Getting {timeframe} data for {symbol}@{exchange} from {start_date} to {end_date}")
        
        try:
            data = zerodha.get_historical_data(
                symbol, exchange, timeframe, start_date, end_date
            )
            
            logger.info(f"Got {len(data) if data else 0} records")
            
            # Print first record if available
            if data and len(data) > 0:
                logger.info(f"First record: {data[0]}")
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python test_zerodha_data.py SYMBOL EXCHANGE [TIMEFRAME]")
        sys.exit(1)
    
    symbol = sys.argv[1]
    exchange = sys.argv[2]
    timeframe = sys.argv[3] if len(sys.argv) > 3 else "day"
    
    test_minimal(symbol, exchange, timeframe)