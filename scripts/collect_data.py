#!/usr/bin/env python3
"""
Script to manually collect data for portfolio instruments.
"""

import os
import sys
import argparse
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.portfolio_manager import PortfolioManager
from portfolio.data_pipeline_trigger import DataPipelineTrigger
from database.connection_manager import get_db
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def collect_data(symbol=None, exchange=None, data_types=None, all_instruments=False):
    """
    Collect data for instruments
    
    Args:
        symbol (str, optional): Instrument symbol (or None for all instruments)
        exchange (str, optional): Exchange code
        data_types (list, optional): List of data types to collect
        all_instruments (bool): Collect data for all instruments
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize database and managers
        db = get_db()
        portfolio_manager = PortfolioManager(db)
        data_pipeline = DataPipelineTrigger(db)
        
        # Determine which instruments to collect data for
        instruments = []
        
        if all_instruments:
            # Get all active instruments
            instruments = portfolio_manager.get_active_instruments()
            logger.info(f"Collecting data for all {len(instruments)} active instruments")
        elif symbol and exchange:
            # Get specific instrument
            instrument = portfolio_manager.get_instrument(symbol, exchange)
            if instrument:
                instruments = [instrument]
                logger.info(f"Collecting data for {symbol}:{exchange}")
            else:
                logger.error(f"Instrument {symbol}:{exchange} not found in portfolio")
                return False
        else:
            logger.error("No instruments specified. Use --symbol and --exchange or --all")
            return False
        
        # Determine which data types to collect
        if not data_types:
            data_types = ["historical", "financial", "news", "global"]
        
        # Start data collection for each instrument
        for instrument in instruments:
            symbol = instrument['symbol']
            exchange = instrument['exchange']
            instrument_type = instrument.get('instrument_type', 'equity')
            
            logger.info(f"Starting data collection for {symbol}:{exchange}")
            
            # Reset data collection status if collecting all types
            if set(data_types) == set(["historical", "financial", "news", "global"]):
                portfolio_manager.update_data_collection_status(symbol, exchange, "all", False)
            
            # Manually run the data collection steps
            for data_type in data_types:
                if data_type == "historical":
                    logger.info(f"Collecting historical data for {symbol}:{exchange}")
                    data_pipeline._collect_historical_data(symbol, exchange, {})
                elif data_type == "financial":
                    logger.info(f"Collecting financial data for {symbol}:{exchange}")
                    data_pipeline._collect_financial_data(symbol, exchange, {})
                elif data_type == "news":
                    logger.info(f"Collecting news data for {symbol}:{exchange}")
                    data_pipeline._collect_news_data(symbol, exchange, {})
                elif data_type == "global":
                    logger.info(f"Collecting global data for {symbol}:{exchange}")
                    data_pipeline._collect_global_data(symbol, exchange, {'sector': instrument.get('sector')})
            
            # Run initial analysis
            if "analysis" in data_types:
                logger.info(f"Running analysis for {symbol}:{exchange}")
                data_pipeline._run_initial_analysis(symbol, exchange, {})
        
        return True
            
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Collect data for portfolio instruments')
    
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
        help='Collect data for all active instruments'
    )
    
    parser.add_argument(
        '--types',
        nargs='+',
        choices=['historical', 'financial', 'news', 'global', 'analysis', 'all'],
        help='Data types to collect'
    )
    
    args = parser.parse_args()
    
    # Expand 'all' to all data types
    data_types = args.types
    if data_types and 'all' in data_types:
        data_types = ['historical', 'financial', 'news', 'global', 'analysis']
    
    # Collect data
    result = collect_data(
        symbol=args.symbol,
        exchange=args.exchange,
        data_types=data_types,
        all_instruments=args.all
    )
    
    if result:
        print("Data collection completed successfully")
    else:
        print("Data collection failed")
        sys.exit(1)

if __name__ == '__main__':
    main()