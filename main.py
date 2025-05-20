#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automated Trading System - Main Application Entry Point
------------------------------------------------------
This script serves as the main entry point for the Automated Trading System.
It initializes all components and handles the system lifecycle.

Author: Ashokstocks
Date: May 2025
"""
import os
import sys
import time
import signal
import logging
import argparse
from datetime import datetime, timedelta
import pytz
from dotenv import load_dotenv
import threading
from trading.zerodha_manager import ensure_zerodha_connection

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import system components
from config import settings
from database.mongodb_connector import MongoDBConnector
from trading.market_hours import MarketHours
from trading.trading_controller import TradingController
from portfolio.portfolio_manager import PortfolioManager
from data.orchestrator import DataOrchestrator
from automation.scheduler import Scheduler
from automation.daily_workflow import DailyWorkflow
from automation.weekly_workflow import WeeklyWorkflow
from automation.monthly_workflow import MonthlyWorkflow
from automation.model_retraining import ModelRetraining
from research.market_analysis import MarketAnalyzer
from communication.notification_manager import NotificationManager
from utils.logging_utils import setup_logger

# Global flags
running = True
system_components = {}

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Automated Trading System')
    parser.add_argument('--mode', choices=['live', 'paper', 'backtest'], 
                      default=os.getenv('TRADING_MODE', 'paper'),
                      help='Trading mode: live, paper, or backtest')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      default=os.getenv('LOG_LEVEL', 'INFO'),
                      help='Set the logging level')
    parser.add_argument('--config', type=str, default='default',
                      help='Configuration profile to use')
    parser.add_argument('--instruments', type=str, 
                      help='Comma-separated list of instruments to trade')
    parser.add_argument('--backtest-start', type=str, 
                      help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--backtest-end', type=str, 
                      help='Backtest end date (YYYY-MM-DD)')
    return parser.parse_args()

def signal_handler(sig, frame):
    """Handle termination signals."""
    global running
    logger.info("Shutdown signal received, stopping system...")
    running = False

def initialize_system(args):
    """Initialize all system components."""
    logger.info("Initializing Automated Trading System...")
    
    # Check Zerodha connection if trading mode is live
    mode = getattr(args, 'mode', 'paper')
    if mode == 'live':
        zerodha_status = ensure_zerodha_connection()
        if not zerodha_status:
            logger.warning("Zerodha not connected. System will operate with limited capabilities.")
            logger.info("Run scripts/zerodha_login.py to authenticate with Zerodha.")
            
            # Optionally ask to continue
            if not getattr(args, 'force', False):
                response = input("Continue without Zerodha connection? (y/n): ")
                if response.lower() != 'y':
                    logger.info("System initialization aborted by user.")
                    sys.exit(0)
    # Define system_components dictionary
    # system_components = {}
    # Initialize Trading Controller
    # Initialize MongoDB connection
    db_connector = MongoDBConnector(
        uri=os.getenv('MONGODB_URI', 'mongodb://localhost:27017/'),
        db_name=os.getenv('DB_NAME', 'automated_trading')
    )
    logger.info("Connected to MongoDB")
    try:
        
        from trading.trading_controller import TradingController
        
        trading_controller = TradingController(
            db_connector=db_connector,
            mode=mode
        )
        logger.info(f"Trading Controller initialized in {mode} mode")
        system_components['trading_controller'] = trading_controller
    except ImportError as e:
        logger.error(f"Trading Controller not available: {e}")
        trading_controller = None
    # Load settings
    from config import settings
    # Only try to load custom config if it's not the default value
    if args.config and args.config != 'default':
        config = settings.load_custom_config(args.config)
    logger.info(f"Loaded configuration profile: {args.config or 'default'}")
    
    
    
    # Initialize the scheduler
    
    scheduler = None
    try:
        from automation.scheduler import Scheduler
        scheduler = Scheduler(db_connector)
        try:
            cleared_count = scheduler.clear_tasks()
            logger.info(f"Cleared {cleared_count} existing scheduled tasks")
        except Exception as e:
            logger.warning(f"Could not clear scheduled tasks: {e}")
        logger.info("Scheduler initialized")
        
    except ImportError as e:
        logger.warning(f"Scheduler not available: {e}")
        # Create a simple placeholder scheduler class to avoid KeyError
        class PlaceholderScheduler:
            def __init__(self):
                pass
            def schedule_daily(self, **kwargs):
                logger.warning("Placeholder scheduler - schedule_daily called but not implemented")
            def start(self):
                logger.warning("Placeholder scheduler - start called but not implemented")
            def stop(self):
                logger.warning("Placeholder scheduler - stop called but not implemented")
        
        scheduler = PlaceholderScheduler()
    # Clear existing scheduled tasks

    # Always add scheduler to system_components, even if it's a placeholder
    system_components['scheduler'] = scheduler
    
    # Initialize Portfolio Manager
    from portfolio.portfolio_manager import PortfolioManager
    portfolio_manager = PortfolioManager(db_connector)
    if hasattr(args, 'instruments') and args.instruments:
        instruments = args.instruments.split(',')
        for instrument in instruments:
            parts = instrument.strip().split(':')
            if len(parts) >= 2:
                symbol = parts[0]
                exchange = parts[1]
                instrument_type = parts[2] if len(parts) > 2 else "equity"
                portfolio_manager.add_instrument(symbol, exchange, instrument_type=instrument_type)
            else:
                logger.warning(f"Invalid instrument format: {instrument}. Use symbol:exchange:type")
    logger.info("Portfolio Manager initialized")
    
    # Initialize Data Orchestrator with a modified constructor to handle missing methods
    from data.orchestrator import DataOrchestrator
    
    # Create a wrapper for the DataOrchestrator class to handle missing methods
    class SafeDataOrchestrator(DataOrchestrator):
        def __init__(self, db):
            self.logger = setup_logger(__name__)
            self.db = db
            
            # Safely try to access optimizer and partitioner
            try:
                self.db_optimizer = db.get_optimizer()
            except AttributeError:
                self.logger.warning("Database optimizer not available")
                self.db_optimizer = None
                
            try:
                self.time_partitioner = db.get_partitioner()
            except AttributeError:
                self.logger.warning("Time partitioner not available")
                self.time_partitioner = None
            
            # Skip partitioning if not available
            if self.time_partitioner:
                self._setup_partitioning()
            else:
                self.logger.warning("Skipping collection partitioning setup")
            
            # Track ongoing collection tasks
            self.active_collections = {}
            self.collection_locks = {}
            
            # Create locks for different data types
            for data_type in ["market", "financial", "news", "global", "alternative"]:
                self.collection_locks[data_type] = threading.Lock()
    
    data_orchestrator = SafeDataOrchestrator(db_connector)
    logger.info("Data Orchestrator initialized")
    
    # Initialize Market Analyzer
    market_analyzer = None
    try:
        from research.market_analysis import MarketAnalyzer
        market_analyzer = MarketAnalyzer(db_connector)
        logger.info("Market Analyzer initialized")
    except ImportError as e:
        logger.warning(f"Market Analyzer not available: {e}")
    
    system_components['market_analyzer'] = market_analyzer
    
    # Initialize Notification Manager
    notification_manager = None
    try:
        from communication.notification_manager import NotificationManager
        notification_manager = NotificationManager(db_connector)
        logger.info("Notification Manager initialized")
    except ImportError as e:
        logger.warning(f"Notification Manager not available: {e}")
    
    system_components['notification_manager'] = notification_manager
    
    # Initialize Market Hours
    from trading.market_hours import MarketHours
    market_hours = MarketHours()
    logger.info("Market Hours initialized")
    
    # Initialize components based on mode
    mode = getattr(args, 'mode', 'paper')  # Default to paper if not specified
    
    if mode == 'backtest':
        # Initialize Backtest Engine
        try:
            from backtesting.engine import BacktestEngine
            
            backtest_start = getattr(args, 'backtest_start', None)
            backtest_end = getattr(args, 'backtest_end', None)
            
            backtest_engine = BacktestEngine(
                db_connector=db_connector, 
                portfolio_manager=portfolio_manager,
                start_date=backtest_start,
                end_date=backtest_end
            )
            logger.info("Backtest Engine initialized")
            system_components['backtest_engine'] = backtest_engine
        except ImportError as e:
            logger.warning(f"Backtest Engine not available: {e}")
    else:
        # Initialize Trading Controller
        try:
            from trading.trading_controller import TradingController
            
            trading_controller = TradingController(
                db_connector=db_connector,
                mode=mode
            )
            logger.info(f"Trading Controller initialized in {mode} mode")
            system_components['trading_controller'] = trading_controller
        except ImportError as e:
            logger.error(f"Trading Controller not available: {e}")
            trading_controller = None
        
        # Initialize Workflow components
        try:
            from automation.daily_workflow import DailyWorkflow
            from automation.weekly_workflow import WeeklyWorkflow 
            from automation.monthly_workflow import MonthlyWorkflow
            from automation.model_retraining import ModelRetraining
            
            daily_workflow = DailyWorkflow(db_connector)
            weekly_workflow = WeeklyWorkflow(db_connector)
            monthly_workflow = MonthlyWorkflow(db_connector)
            model_retraining = ModelRetraining(db_connector)
            
            # Register workflow tasks with scheduler
            daily_workflow.register_tasks(scheduler)
            weekly_workflow.register_tasks(scheduler)
            monthly_workflow.register_tasks(scheduler)
            
            # Schedule model retraining during off-hours
            scheduler.schedule_daily(
                func=model_retraining.retrain_all_models,
                time_str="01:00",  # 1 AM
                name="Daily Model Retraining"
            )
            
            # Store components in global dict
            system_components.update({
                'daily_workflow': daily_workflow,
                'weekly_workflow': weekly_workflow,
                'monthly_workflow': monthly_workflow,
                'model_retraining': model_retraining
            })
        except ImportError as e:
            logger.warning(f"Workflow components not available: {e}")
    
    # Common components for all modes
    system_components.update({
        'settings': settings,
        'db_connector': db_connector,
        'portfolio_manager': portfolio_manager,
        'data_orchestrator': data_orchestrator,
        'market_hours': market_hours
    })
    
    logger.info("System initialization complete")
    return system_components

def run_backtest():
    """Run system in backtest mode."""
    logger.info("Starting backtest...")
    backtest_engine = system_components['backtest_engine']
    result = backtest_engine.run()
    logger.info("Backtest completed")
    
    # Generate backtest report
    backtest_engine.generate_report(result)
    
    logger.info("Backtest report generated")
    return result

def run_live_system():
    """Run the live trading system."""
    global running
    
    logger.info("Starting live system...")
    
    # Get components
    scheduler = system_components['scheduler']
    trading_controller = system_components['trading_controller']
    notification_manager = system_components['notification_manager']
    market_hours = system_components['market_hours']
    scheduler.clear_tasks()
    
    
    # Start the scheduler
    scheduler.start()
    
    # Send startup notification
    notification_manager.send_system_alert("Automated Trading System started", level="info")
    
    # Track market status
    is_market_open_prev = False
    
    # Main system loop
    while running:
        # Check if market is open
        is_market_open_current = market_hours.is_market_open()
        
        # Detect market hours transitions
        if is_market_open_current and not is_market_open_prev:
            logger.info("Market opened, activating trading system")
            notification_manager.send_system_alert("Market opened. Trading system activated.", level="info")
            trading_controller.start_trading()
        elif not is_market_open_current and is_market_open_prev:
            logger.info("Market closed, deactivating trading system")
            notification_manager.send_system_alert("Market closed. Trading system deactivated.", level="info")
            trading_controller.stop_trading()
        
        # Update previous market status
        is_market_open_prev = is_market_open_current
        
        # Sleep to prevent high CPU usage
        time.sleep(5)
    
    # Graceful shutdown
    logger.info("Shutting down live system...")
    
    # Stop trading if active
    if is_market_open_prev:
        trading_controller.stop_trading()
    
    # Stop scheduler
    scheduler.stop()
    
    # Send shutdown notification
    notification_manager.send_system_alert("Automated Trading System stopped", level="info")
    
    logger.info("System shutdown complete")

def main():
    """Main entry point for the Automated Trading System."""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logger
    global logger
    logger = setup_logger('main', level=args.log_level)
    
    try:
        # Initialize the system
        initialize_system(args)
        
        # Run the system based on mode
        if args.mode == 'backtest':
            run_backtest()
        else:
            run_live_system()
    
    except Exception as e:
        logger.critical(f"System failed: {str(e)}", exc_info=True)
        if 'notification_manager' in system_components:
            system_components['notification_manager'].send_system_alert(
                f"CRITICAL ERROR: System failed: {str(e)}",
                level="critical"
            )
        sys.exit(1)

if __name__ == "__main__":
    main()