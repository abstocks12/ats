#!/usr/bin/env python3
"""
Script to add an instrument to the portfolio.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from portfolio.portfolio_manager import PortfolioManager
from database.connection_manager import get_db
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def add_instrument(symbol, exchange, instrument_type=None, sector=None, **kwargs):
    """
    Add an instrument to the portfolio
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        instrument_type (str, optional): Instrument type
        sector (str, optional): Sector of the instrument
        **kwargs: Additional parameters
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get database connection
        db = get_db()
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(db)
        
        # Add instrument
        instrument_id = portfolio_manager.add_instrument(
            symbol=symbol,
            exchange=exchange,
            instrument_type=instrument_type,
            sector=sector,
            **kwargs
        )
        
        if instrument_id:
            logger.info(f"Added instrument {symbol}:{exchange} to portfolio (ID: {instrument_id})")
            return True
        else:
            logger.error(f"Failed to add instrument {symbol}:{exchange} to portfolio")
            return False
            
    except Exception as e:
        logger.error(f"Error adding instrument: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Add an instrument to the portfolio')
    
    parser.add_argument(
        '--symbol',
        required=True,
        help='Instrument symbol'
    )
    
    parser.add_argument(
        '--exchange',
        required=True,
        help='Exchange code (NSE, BSE)'
    )
    
    parser.add_argument(
        '--type',
        choices=['equity', 'futures', 'options'],
        help='Instrument type'
    )
    
    parser.add_argument(
        '--sector',
        help='Sector of the instrument'
    )
    
    parser.add_argument(
        '--industry',
        help='Industry of the instrument'
    )
    
    parser.add_argument(
        '--timeframe',
        choices=['intraday', 'swing', 'positional', 'long_term'],
        help='Trading timeframe'
    )
    
    parser.add_argument(
        '--position-size',
        type=float,
        help='Position size percentage'
    )
    
    parser.add_argument(
        '--max-risk',
        type=float,
        help='Maximum risk percentage'
    )
    
    args = parser.parse_args()
    
    # Convert arguments to kwargs
    kwargs = {}
    
    if args.industry:
        kwargs['industry'] = args.industry
    
    if args.timeframe:
        kwargs['timeframe'] = args.timeframe
    
    if args.position_size:
        kwargs['position_size_percent'] = args.position_size
    
    if args.max_risk:
        kwargs['max_risk_percent'] = args.max_risk
    
    # Add instrument
    result = add_instrument(
        symbol=args.symbol,
        exchange=args.exchange,
        instrument_type=args.type,
        sector=args.sector,
        **kwargs
    )
    
    if result:
        print(f"Successfully added {args.symbol}:{args.exchange} to portfolio")
    else:
        print(f"Failed to add {args.symbol}:{args.exchange} to portfolio")
        sys.exit(1)

if __name__ == '__main__':
    main()