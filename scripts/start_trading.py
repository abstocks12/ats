#!/usr/bin/env python3
"""
Script to start the trading system.
"""

import os
import sys
import argparse
import signal
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from trading.trading_controller import TradingController
from database.connection_manager import get_db
from utils.logging_utils import setup_logger
from utils.helper_functions import is_market_open

logger = setup_logger(__name__)

def start_trading(mode=None, instruments=None, capital=None, wait_for_market=False):
    """
    Start the trading system
    
    Args:
        mode (str, optional): Trading mode (live, paper)
        instruments (list, optional): List of instruments to trade
        capital (float, optional): Trading capital
        wait_for_market (bool): Wait for market to open
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get mode from settings if not provided
        if not mode:
            mode = settings.TRADING_MODE
        
        logger.info(f"Starting trading system in {mode} mode")
        
        # Initialize trading controller
        db = get_db()
        controller = TradingController(db, mode=mode, capital=capital)
        
        # Check if market is open
        if not is_market_open() and wait_for_market:
            logger.info("Market is closed. Waiting for market to open...")
            
            # Calculate time until market open
            from utils.helper_functions import get_trading_sessions
            sessions = get_trading_sessions()
            market_open = sessions['market']['start']
            now = datetime.now()
            
            if now.date() == market_open.date() and now < market_open:
                # Market opens today
                wait_seconds = (market_open - now).total_seconds()
                wait_minutes = int(wait_seconds / 60)
                
                logger.info(f"Market opens in {wait_minutes} minutes. Waiting...")
                
                # Wait for market to open (with check every minute)
                while not is_market_open():
                    time.sleep(60)  # Check every minute
            else:
                # Market doesn't open today
                logger.info("Market doesn't open today. Exiting.")
                return False
        
        # Start trading
        result = controller.start_trading(instruments=instruments)
        
        if result:
            logger.info(f"Trading started successfully in {mode} mode")
            
            # Set up signal handler for graceful shutdown
            def signal_handler(sig, frame):
                logger.info("Stopping trading due to signal...")
                controller.stop_trading()
                sys.exit(0)
            
            # Register signal handlers
            signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
            signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
            
            # Keep the script running
            logger.info("Trading system is running. Press Ctrl+C to stop.")
            while True:
                time.sleep(60)  # Sleep to avoid high CPU usage
                
                # Check if trading is still active
                if not controller.is_trading_active():
                    logger.info("Trading has stopped. Exiting.")
                    break
                
                # Check if market is closed
                if not is_market_open() and controller.is_trading_active():
                    logger.info("Market has closed. Stopping trading.")
                    controller.stop_trading()
                    break
            
            return True
        else:
            logger.error("Failed to start trading")
            return False
            
    except Exception as e:
        logger.error(f"Error starting trading: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Start the trading system')
    
    parser.add_argument(
        '--mode',
        choices=['live', 'paper'],
        help='Trading mode (live, paper)'
    )
    
    parser.add_argument(
        '--instruments',
        nargs='+',
        help='List of instruments to trade (format: SYMBOL:EXCHANGE)'
    )
    
    parser.add_argument(
        '--capital',
        type=float,
        help='Trading capital'
    )
    
    parser.add_argument(
        '--wait-for-market',
        action='store_true',
        help='Wait for market to open'
    )
    
    args = parser.parse_args()
    
    # Parse instruments if provided
    instruments = None
    if args.instruments:
        instruments = []
        for instrument_str in args.instruments:
            parts = instrument_str.split(':')
            if len(parts) >= 2:
                symbol = parts[0]
                exchange = parts[1]
                instruments.append({'symbol': symbol, 'exchange': exchange})
    
    # Start trading
    start_trading(
        mode=args.mode,
        instruments=instruments,
        capital=args.capital,
        wait_for_market=args.wait_for_market
    )

if __name__ == '__main__':
    main()