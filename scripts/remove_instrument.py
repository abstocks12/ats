#!/usr/bin/env python3
"""
Script to remove an instrument from the portfolio.
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

def remove_instrument(symbol, exchange, force=False):
    """
    Remove an instrument from the portfolio
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        force (bool): Force removal even with open positions
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get database connection
        db = get_db()
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(db)
        
        # Remove instrument
        result = portfolio_manager.remove_instrument(
            symbol=symbol,
            exchange=exchange,
            force=force
        )
        
        if result:
            logger.info(f"Removed instrument {symbol}:{exchange} from portfolio")
            return True
        else:
            logger.error(f"Failed to remove instrument {symbol}:{exchange} from portfolio")
            return False
            
    except Exception as e:
        logger.error(f"Error removing instrument: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Remove an instrument from the portfolio')
    
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
        '--force',
        action='store_true',
        help='Force removal even with open positions'
    )
    
    args = parser.parse_args()
    
    # Remove instrument
    result = remove_instrument(
        symbol=args.symbol,
        exchange=args.exchange,
        force=args.force
    )
    
    if result:
        print(f"Successfully removed {args.symbol}:{args.exchange} from portfolio")
    else:
        print(f"Failed to remove {args.symbol}:{args.exchange} from portfolio")
        sys.exit(1)

if __name__ == '__main__':
    main()