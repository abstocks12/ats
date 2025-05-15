#!/usr/bin/env python3
"""
Script to stop the trading system.
"""

import os
import sys
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trading.trading_controller import TradingController
from database.connection_manager import get_db
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def stop_trading(close_positions=False, emergency=False):
    """
    Stop the trading system
    
    Args:
        close_positions (bool): Close all open positions
        emergency (bool): Emergency stop
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Stopping trading system")
        
        # Initialize trading controller
        db = get_db()
        controller = TradingController(db)
        
        if emergency:
            # Emergency stop
            result = controller.emergency_stop()
            
            if result:
                logger.info("Emergency stop executed successfully")
                return True
            else:
                logger.error("Failed to execute emergency stop")
                return False
        else:
            # Normal stop
            result = controller.stop_trading(close_positions=close_positions)
            
            if result:
                logger.info("Trading stopped successfully")
                if close_positions:
                    logger.info("All positions closed")
                return True
            else:
                logger.error("Failed to stop trading")
                return False
                
    except Exception as e:
        logger.error(f"Error stopping trading: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stop the trading system')
    
    parser.add_argument(
        '--close-positions',
        action='store_true',
        help='Close all open positions'
    )
    
    parser.add_argument(
        '--emergency',
        action='store_true',
        help='Emergency stop'
    )
    
    args = parser.parse_args()
    
    # Stop trading
    stop_trading(
        close_positions=args.close_positions,
        emergency=args.emergency
    )

if __name__ == '__main__':
    main()