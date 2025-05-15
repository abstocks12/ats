#!/usr/bin/env python3
"""
Script to list all instruments in the portfolio.
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

def list_instruments(instrument_type=None, sector=None, format='text'):
    """
    List all instruments in the portfolio
    
    Args:
        instrument_type (str, optional): Filter by instrument type
        sector (str, optional): Filter by sector
        format (str): Output format ('text', 'json')
        
    Returns:
        list: List of instruments
    """
    try:
        # Get database connection
        db = get_db()
        
        # Create portfolio manager
        portfolio_manager = PortfolioManager(db)
        
        # Get active instruments
        instruments = portfolio_manager.get_active_instruments(
            instrument_type=instrument_type,
            sector=sector
        )
        
        if format == 'json':
            import json
            # Convert ObjectId to string for JSON serialization
            for instrument in instruments:
                if '_id' in instrument:
                    instrument['_id'] = str(instrument['_id'])
            
            return json.dumps(instruments, indent=2, default=str)
        else:
            # Format as text
            if not instruments:
                return "No instruments found in portfolio."
            
            lines = []
            lines.append("-" * 80)
            lines.append(f"{'Symbol':<10} {'Exchange':<10} {'Type':<10} {'Sector':<15} {'Enabled':<10} {'Added Date':<20}")
            lines.append("-" * 80)
            
            for instrument in instruments:
                symbol = instrument.get('symbol', '')
                exchange = instrument.get('exchange', '')
                instrument_type = instrument.get('instrument_type', '')
                sector = instrument.get('sector', '')
                enabled = 'Yes' if instrument.get('trading_config', {}).get('enabled', False) else 'No'
                added_date = instrument.get('added_date', '')
                
                if added_date:
                    if isinstance(added_date, datetime):
                        added_date = added_date.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        added_date = str(added_date)
                
                lines.append(f"{symbol:<10} {exchange:<10} {instrument_type:<10} {sector:<15} {enabled:<10} {added_date:<20}")
            
            lines.append("-" * 80)
            lines.append(f"Total: {len(instruments)} instruments")
            
            return "\n".join(lines)
            
    except Exception as e:
        logger.error(f"Error listing instruments: {e}")
        return "Error listing instruments."

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='List all instruments in the portfolio')
    
    parser.add_argument(
        '--type',
        choices=['equity', 'futures', 'options'],
        help='Filter by instrument type'
    )
    
    parser.add_argument(
        '--sector',
        help='Filter by sector'
    )
    
    parser.add_argument(
        '--format',
        choices=['text', 'json'],
        default='text',
        help='Output format'
    )
    
    args = parser.parse_args()
    
    # List instruments
    result = list_instruments(
        instrument_type=args.type,
        sector=args.sector,
        format=args.format
    )
    
    print(result)

if __name__ == '__main__':
    main()