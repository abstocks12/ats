from portfolio.instrument_setup import InstrumentSetup
from portfolio.data_pipeline_trigger import DataPipelineTrigger
from database.connection_manager import get_db
from utils.logging_utils import setup_logger
from utils.helper_functions import normalize_symbol

logger = setup_logger("test_portfolio")

def test_portfolio_manager():
   """Test portfolio manager functionality"""
   try:
       # Get database connection
       db = get_db()
       
       # Create portfolio manager
       portfolio_manager = PortfolioManager(db)
       
       # Test symbol
       symbol = "TATASTEEL"
       exchange = "NSE"
       
       # Check if instrument exists and remove if it does
       existing = portfolio_manager.get_instrument(symbol, exchange)
       if existing:
           portfolio_manager.remove_instrument(symbol, exchange, force=True)
           logger.info(f"Removed existing instrument {symbol}:{exchange}")
       
       # Add instrument
       instrument_id = portfolio_manager.add_instrument(
           symbol=symbol,
           exchange=exchange,
           instrument_type="equity",
           sector="metal"
       )
       
       logger.info(f"Added instrument {symbol}:{exchange} with ID: {instrument_id}")
       
       # Get the instrument
       instrument = portfolio_manager.get_instrument(symbol, exchange)
       
       if not instrument:
           logger.error(f"Failed to get instrument {symbol}:{exchange}")
           return False
       
       logger.info(f"Retrieved instrument: {instrument['symbol']}:{instrument['exchange']}")
       
       # Update trading configuration
       result = portfolio_manager.update_instrument_config(
           symbol=symbol,
           exchange=exchange,
           config_updates={
               "position_size_percent": 3.0,
               "max_risk_percent": 0.5
           }
       )
       
       if not result:
           logger.error(f"Failed to update trading configuration for {symbol}:{exchange}")
           return False
       
       logger.info(f"Updated trading configuration for {symbol}:{exchange}")
       
       # Get all active instruments
       active_instruments = portfolio_manager.get_active_instruments()
       
       logger.info(f"Found {len(active_instruments)} active instruments")
       
       # Test enable/disable trading
       result = portfolio_manager.disable_trading(symbol, exchange)
       
       if not result:
           logger.error(f"Failed to disable trading for {symbol}:{exchange}")
           return False
       
       logger.info(f"Disabled trading for {symbol}:{exchange}")
       
       result = portfolio_manager.enable_trading(symbol, exchange)
       
       if not result:
           logger.error(f"Failed to enable trading for {symbol}:{exchange}")
           return False
       
       logger.info(f"Enabled trading for {symbol}:{exchange}")
       
       # Test updating data collection status
       result = portfolio_manager.update_data_collection_status(
           symbol=symbol,
           exchange=exchange,
           data_type="historical",
           status=True
       )
       
       if not result:
           logger.error(f"Failed to update data collection status for {symbol}:{exchange}")
           return False
       
       logger.info(f"Updated data collection status for {symbol}:{exchange}")
       
       # Get trading parameters
       params = portfolio_manager.get_trading_parameters(symbol, exchange)
       
       logger.info(f"Trading parameters for {symbol}:{exchange}: {params}")
       
       # Test check position limit
       result = portfolio_manager.check_position_limit(symbol, exchange)
       
       logger.info(f"Position limit check for {symbol}:{exchange}: {result}")
       
       # Remove the instrument
       result = portfolio_manager.remove_instrument(symbol, exchange, force=True)
       
       if not result:
           logger.error(f"Failed to remove instrument {symbol}:{exchange}")
           return False
       
       logger.info(f"Removed instrument {symbol}:{exchange}")
       
       # Test portfolio exposure (returns empty since we removed the instrument)
       exposure = portfolio_manager.get_portfolio_exposure()
       
       logger.info(f"Portfolio exposure: {exposure}")
       
       logger.info("Portfolio manager test completed successfully")
       return True
   except Exception as e:
       logger.error(f"Error in portfolio manager test: {e}")
       return False

def test_instrument_setup():
   """Test instrument setup functionality"""
   try:
       # Get database connection
       db = get_db()
       
       # Create instrument setup
       instrument_setup = InstrumentSetup(db)
       
       # Test symbol
       symbol = "TATASTEEL"
       exchange = "NSE"
       
       # Set up instrument
       config = instrument_setup.setup_instrument(
           symbol=symbol,
           exchange=exchange,
           instrument_type="equity",
           sector="metal"
       )
       
       logger.info(f"Set up instrument {symbol}:{exchange} with configuration")
       logger.info(f"Timeframes: {config.get('timeframes', [])}")
       logger.info(f"Strategies: {config.get('strategies', [])}")
       logger.info(f"Data sources: {config.get('data_sources', [])}")
       
       # Get strategy parameters
       strategy = "technical"
       params = instrument_setup.get_strategy_parameters(strategy)
       
       logger.info(f"Strategy parameters for {strategy}: {params}")
       
       # Configure from database
       # First add the instrument to the database
       portfolio_manager = PortfolioManager(db)
       instrument_id = portfolio_manager.add_instrument(
           symbol=symbol,
           exchange=exchange,
           instrument_type="equity",
           sector="metal"
       )
       
       # Configure from database
       config = instrument_setup.configure_from_database(symbol, exchange)
       
       logger.info(f"Configured from database: {symbol}:{exchange}")
       
       # Clean up
       portfolio_manager.remove_instrument(symbol, exchange, force=True)
       
       logger.info("Instrument setup test completed successfully")
       return True
   except Exception as e:
       logger.error(f"Error in instrument setup test: {e}")
       return False

def test_data_pipeline_trigger():
   """Test data pipeline trigger functionality"""
   try:
       # Get database connection
       db = get_db()
       
       # Create data pipeline trigger
       pipeline = DataPipelineTrigger(db)
       
       # Test symbol
       symbol = "TATASTEEL"
       exchange = "NSE"
       
       # First add the instrument to the database
       portfolio_manager = PortfolioManager(db)
       instrument_id = portfolio_manager.add_instrument(
           symbol=symbol,
           exchange=exchange,
           instrument_type="equity",
           sector="metal"
       )
       
       # Trigger data collection
       pipeline.trigger_data_collection(symbol, exchange, "equity")
       
       logger.info(f"Triggered data collection for {symbol}:{exchange}")
       
       # Wait a bit for data collection to start
       time.sleep(2)
       
       # Clean up
       portfolio_manager.remove_instrument(symbol, exchange, force=True)
       
       logger.info("Data pipeline trigger test completed successfully")
       return True
   except Exception as e:
       logger.error(f"Error in data pipeline trigger test: {e}")
       return False

def main():
   """Main function"""
   print("Testing portfolio management system...")
   
   # Test portfolio manager
   print("\n=== Testing Portfolio Manager ===")
   if test_portfolio_manager():
       print("Portfolio manager test passed")
   else:
       print("Portfolio manager test failed")
       sys.exit(1)
   
   # Test instrument setup
   print("\n=== Testing Instrument Setup ===")
   if test_instrument_setup():
       print("Instrument setup test passed")
   else:
       print("Instrument setup test failed")
       sys.exit(1)
   
   # Test data pipeline trigger
   print("\n=== Testing Data Pipeline Trigger ===")
   if test_data_pipeline_trigger():
       print("Data pipeline trigger test passed")
   else:
       print("Data pipeline trigger test failed")
       sys.exit(1)
   
   print("\nAll tests passed successfully")

if __name__ == '__main__':
   main()