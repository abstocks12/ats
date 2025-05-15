"""
Main entry point for the Automated Trading System.
Initializes the system and provides a command-line interface.
"""

import os
import sys
import argparse
import signal
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import settings
from utils.logging_utils import setup_logger, log_error

logger = setup_logger('main')

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Automated Trading System')
    
    parser.add_argument(
        '--config', 
        help='Path to custom configuration file'
    )
    
    parser.add_argument(
        '--mode',
        choices=['live', 'paper'],
        help='Trading mode (live or paper)'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    parser.add_argument(
        '--version',
        action='store_true',
        help='Show version information'
    )
    
    parser.add_argument(
        '--init-db',
        action='store_true',
        help='Initialize the database'
    )
    
    parser.add_argument(
        '--add-instrument',
        help='Add an instrument to the portfolio (format: SYMBOL:EXCHANGE:TYPE)'
    )
    
    parser.add_argument(
        '--remove-instrument',
        help='Remove an instrument from the portfolio (format: SYMBOL:EXCHANGE)'
    )
    
    parser.add_argument(
        '--collect-data',
        action='store_true',
        help='Run data collection for portfolio instruments'
    )
    
    parser.add_argument(
        '--start-trading',
        action='store_true',
        help='Start the trading system'
    )
    
    parser.add_argument(
        '--stop-trading',
        action='store_true',
        help='Stop the trading system'
    )
    
    return parser.parse_args()

def handle_init_db():
    """Initialize the database"""
    try:
        # Importing here to avoid circular imports
        from database.mongodb_connector import MongoDBConnector
        
        db = MongoDBConnector()
        db.initialize_database()
        
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        log_error(e, context={"action": "initialize_database"})
        return False

def handle_add_instrument(instrument_arg):
    """Add an instrument to the portfolio"""
    try:
        # Parse the instrument argument (SYMBOL:EXCHANGE:TYPE)
        parts = instrument_arg.split(':')
        if len(parts) < 2:
            logger.error("Invalid instrument format. Use SYMBOL:EXCHANGE[:TYPE]")
            return False
        
        symbol = parts[0].upper()
        exchange = parts[1].upper()
        
        # Type is optional, defaults to 'equity'
        instrument_type = parts[2].lower() if len(parts) > 2 else 'equity'
        
        # Importing here to avoid circular imports
        from database.mongodb_connector import MongoDBConnector
        from portfolio.portfolio_manager import PortfolioManager
        
        db = MongoDBConnector()
        portfolio_manager = PortfolioManager(db)
        
        result = portfolio_manager.add_instrument(
            symbol=symbol,
            exchange=exchange,
            instrument_type=instrument_type
        )
        
        if result:
            logger.info(f"Instrument {symbol} added to portfolio")
            return True
        else:
            logger.error(f"Failed to add instrument {symbol}")
            return False
    except Exception as e:
        log_error(e, context={"action": "add_instrument", "instrument": instrument_arg})
        return False

def handle_remove_instrument(instrument_arg):
    """Remove an instrument from the portfolio"""
    try:
        # Parse the instrument argument (SYMBOL:EXCHANGE)
        parts = instrument_arg.split(':')
        if len(parts) < 2:
            logger.error("Invalid instrument format. Use SYMBOL:EXCHANGE")
            return False
        
        symbol = parts[0].upper()
        exchange = parts[1].upper()
        
        # Importing here to avoid circular imports
        from database.mongodb_connector import MongoDBConnector
        from portfolio.portfolio_manager import PortfolioManager
        
        db = MongoDBConnector()
        portfolio_manager = PortfolioManager(db)
        
        result = portfolio_manager.remove_instrument(
            symbol=symbol,
            exchange=exchange
        )
        
        if result:
            logger.info(f"Instrument {symbol} removed from portfolio")
            return True
        else:
            logger.error(f"Failed to remove instrument {symbol}")
            return False
    except Exception as e:
        log_error(e, context={"action": "remove_instrument", "instrument": instrument_arg})
        return False

def handle_collect_data():
    """Run data collection for portfolio instruments"""
    try:
        # Importing here to avoid circular imports
        from database.mongodb_connector import MongoDBConnector
        from portfolio.portfolio_manager import PortfolioManager
        from data.orchestrator import DataOrchestrator
        
        db = MongoDBConnector()
        portfolio_manager = PortfolioManager(db)
        data_orchestrator = DataOrchestrator(db)
        
        # Get all active instruments
        instruments = portfolio_manager.get_active_instruments()
        
        if not instruments:
            logger.warning("No active instruments found in portfolio")
            return False
        
        # Collect data for each instrument
        for instrument in instruments:
            logger.info(f"Collecting data for {instrument['symbol']}")
            data_orchestrator.collect_all_data(
                symbol=instrument['symbol'],
                exchange=instrument['exchange'],
                instrument_type=instrument.get('instrument_type', 'equity')
            )
        
        logger.info("Data collection completed")
        return True
    except Exception as e:
        log_error(e, context={"action": "collect_data"})
        return False

def handle_start_trading():
    """Start the trading system"""
    try:
        # Importing here to avoid circular imports
        from database.mongodb_connector import MongoDBConnector
        from trading.trading_controller import TradingController
        
        db = MongoDBConnector()
        trading_controller = TradingController(db, mode=settings.TRADING_MODE)
        
        result = trading_controller.start_trading()
        
        if result:
            logger.info(f"Trading started in {settings.TRADING_MODE} mode")
            
            # Keep running until interrupted
            def signal_handler(sig, frame):
                logger.info("Stopping trading...")
                trading_controller.stop_trading()
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            logger.info("Press Ctrl+C to stop trading")
            
            # Keep the main thread alive
            while True:
                signal.pause()
        else:
            logger.error("Failed to start trading")
            return False
            
        return True
    except Exception as e:
        log_error(e, context={"action": "start_trading"})
        return False

def handle_stop_trading():
    """Stop the trading system"""
    try:
        # Importing here to avoid circular imports
        from database.mongodb_connector import MongoDBConnector
        from trading.trading_controller import TradingController
        
        db = MongoDBConnector()
        trading_controller = TradingController(db)
        
        result = trading_controller.stop_trading()
        
        if result:
            logger.info("Trading stopped")
            return True
        else:
            logger.error("Failed to stop trading")
            return False
    except Exception as e:
        log_error(e, context={"action": "stop_trading"})
        return False

def print_banner():
    """Print a welcome banner"""
    version_info = settings.get_version_info()
    
    print("\n" + "=" * 60)
    print(f"  AUTOMATED TRADING SYSTEM v{version_info['version']}")
    print(f"  Mode: {version_info['trading_mode']}  |  Environment: {version_info['environment']}")
    print(f"  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60 + "\n")

def main():
    """Main function"""
    args = parse_arguments()
    
    # Load custom configuration if provided
    if args.config:
        settings.load_custom_config(args.config)
    
    # Override settings with command line arguments
    if args.mode:
        settings.TRADING_MODE = args.mode
    
    if args.debug:
        settings.DEBUG = True
    
    # Show version information
    if args.version:
        version_info = settings.get_version_info()
        print(f"Automated Trading System v{version_info['version']}")
        print(f"Environment: {version_info['environment']}")
        print(f"Trading Mode: {version_info['trading_mode']}")
        return
    
    # Print welcome banner
    print_banner()
    
    # Initialize the database
    if args.init_db:
        if handle_init_db():
            print("Database initialized successfully")
        else:
            print("Failed to initialize database")
        return
    
    # Add an instrument to the portfolio
    if args.add_instrument:
        if handle_add_instrument(args.add_instrument):
            print(f"Instrument {args.add_instrument} added to portfolio")
        else:
            print(f"Failed to add instrument {args.add_instrument}")
        return
    
    # Remove an instrument from the portfolio
    if args.remove_instrument:
        if handle_remove_instrument(args.remove_instrument):
            print(f"Instrument {args.remove_instrument} removed from portfolio")
        else:
            print(f"Failed to remove instrument {args.remove_instrument}")
        return
    
    # Collect data for portfolio instruments
    if args.collect_data:
        if handle_collect_data():
            print("Data collection completed")
        else:
            print("Data collection failed")
        return
    
    # Start the trading system
    if args.start_trading:
        handle_start_trading()
        return
    
    # Stop the trading system
    if args.stop_trading:
        if handle_stop_trading():
            print("Trading stopped")
        else:
            print("Failed to stop trading")
        return
    
    # If no specific action is requested, print help
    print("No action specified. Use --help to see available options.")

if __name__ == '__main__':
    main()