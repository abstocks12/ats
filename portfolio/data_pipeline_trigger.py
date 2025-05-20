"""
Data Pipeline Trigger for the Automated Trading System.
Manages the data collection pipeline for instruments.
"""

import os
import sys
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database.connection_manager import get_db
from utils.logging_utils import setup_logger, log_error
from utils.helper_functions import normalize_symbol
from research.fundamental_analyzer import FundamentalAnalyzer
from data.news.sentiment_analyzer import SentimentAnalyzer
from data.market.historical_data import HistoricalDataCollector
from portfolio.instrument_setup import InstrumentSetup
from portfolio.portfolio_manager import PortfolioManager


from data.global_markets.indices_collector import IndicesCollector

from data.financial.financial_scraper import FinancialScraper
from research.technical_analyzer import TechnicalAnalyzer


from ml.prediction.daily_predictor import DailyPredictor
import random
from database.models import PredictionData
from data.news.news_aggregator import NewsAggregator





class DataPipelineTrigger:
    """
    Data Pipeline Trigger for managing data collection
    """
    
    def __init__(self, db=None):
        """
        Initialize Data Pipeline Trigger
        
        Args:
            db: Database connector (optional, will use global connection if not provided)
        """
        self.logger = setup_logger(__name__)
        self.db = db or get_db()
    
    def trigger_data_collection(self, symbol: str, exchange: str, 
                               instrument_type: Optional[str] = None) -> None:
        """
        Trigger data collection for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str, optional): Instrument type
        """
        # Normalize symbol and exchange
        symbol = normalize_symbol(symbol)
        exchange = exchange.upper()
        
        # Create a thread for data collection
        thread = threading.Thread(
            target=self._collect_data_thread,
            args=(symbol, exchange, instrument_type)
        )
        
        # Set thread as daemon so it doesn't block program exit
        thread.daemon = True
        
        # Start the thread
        thread.start()
        
        self.logger.info(f"Triggered data collection for {symbol}:{exchange}")
    
    def _collect_data_thread(self, symbol: str, exchange: str, 
                            instrument_type: Optional[str] = None) -> None:
        """
        Thread function for data collection
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str, optional): Instrument type
        """
        try:
            # Set up instrument configuration
            
            setup = InstrumentSetup(self.db)
            config = setup.setup_instrument(symbol, exchange, instrument_type)
            
            # Collect historical market data
            self._collect_historical_data(symbol, exchange, config)
            
            # Collect financial data (if applicable)
            if "financial" in config.get("data_sources", []):
                self._collect_financial_data(symbol, exchange, config)
            
            # Collect news data
            self._collect_news_data(symbol, exchange, config)
            
            # Collect global market data (if applicable)
            if "global" in config.get("data_sources", []):
                self._collect_global_data(symbol, exchange, config)
            
            # Run initial analysis
            self._run_initial_analysis(symbol, exchange, config)
            
            # Update portfolio status
            
            portfolio_manager = PortfolioManager(self.db)
            portfolio_manager.update_data_collection_status(symbol, exchange, "all", True)
            
            self.logger.info(f"Completed data collection for {symbol}:{exchange}")
            
        except Exception as e:
            log_error(e, context={"action": "collect_data_thread", "symbol": symbol, "exchange": exchange})
        # Re-raise to see full traceback
        raise
    
    def _collect_historical_data(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
       """
       Collect historical market data
       
       Args:
           symbol (str): Instrument symbol
           exchange (str): Exchange code
           config (dict): Instrument configuration
       """
       try:
           self.logger.info(f"Collecting historical data for {symbol}:{exchange}")
           
           # Import necessary components
           # We import here to avoid circular imports
           try:
               # First try to import the real implementation
               collector = HistoricalDataCollector(self.db)
           except ImportError:
               # If not available, create a placeholder implementation
               self.logger.warning("HistoricalDataCollector not available, using placeholder")
               
               class PlaceholderCollector:
                   def __init__(self, db):
                       self.db = db
                       self.logger = setup_logger("placeholder_collector")
                   
                   def collect_data(self, symbol, exchange, timeframe, days):
                       self.logger.info(f"Placeholder: Collecting {timeframe} data for {symbol}:{exchange}")
                       
                       # Update data collection status
                       
                       portfolio_manager = PortfolioManager(self.db)
                       portfolio_manager.update_data_collection_status(symbol, exchange, "historical", True)
                       
                       return True
               
               collector = PlaceholderCollector(self.db)
           
           # Collect data for each timeframe
           for timeframe in config.get("timeframes", ["day", "60min", "5min"]):
                try:
                   # Convert timeframe to collector format if needed
                    collector_timeframe = timeframe
                    if timeframe == "60min":
                        collector_timeframe = "hour"
                    elif timeframe == "1min":
                        collector_timeframe = "minute"
                    
                    # Get historical days based on timeframe
                    if timeframe == "day":
                        days = settings.HISTORICAL_DAYS_DEFAULT
                    elif timeframe == "60min" or timeframe == "hour":
                        days = 90  # 3 months
                    elif timeframe == "15min":
                        days = 30  # 1 month
                    elif timeframe == "5min":
                        days = 15  # 15 days
                    elif timeframe == "1min" or timeframe == "minute":
                        days = 7  # 1 week
                    else:
                        days = 30  # Default
                    try:
                        # Collect data
                        collector.collect_data(symbol, exchange, collector_timeframe, days)
                    except ZeroDivisionError as zde:
                        self.logger.error(f"Division by zero error in historical data collection: {zde}")
                        self.logger.error(f"This likely indicates missing or invalid data for {symbol}:{exchange}")
                        # Continue with next timeframe instead of failing
                        continue
                        
                except Exception as e:
                    self.logger.error(f"Error collecting data for timeframe {timeframe}: {e}")
                    continue
            
           # Update data collection status
          
           portfolio_manager = PortfolioManager(self.db)
           portfolio_manager.update_data_collection_status(symbol, exchange, "historical", True)
           
           self.logger.info(f"Completed historical data collection for {symbol}:{exchange}")
           
       except Exception as e:
           log_error(e, context={"action": "collect_historical_data", "symbol": symbol, "exchange": exchange})
           
           # Update data collection status with failure
           try:
               
               portfolio_manager = PortfolioManager(self.db)
               portfolio_manager.update_data_collection_status(symbol, exchange, "historical", False)
           except:
               pass
   
    def _collect_financial_data(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Collect financial data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            self.logger.info(f"Collecting financial data for {symbol}:{exchange}")
            
            # Import necessary components
            try:
                # First try to import the real implementation
                
                scraper = FinancialScraper(symbol, exchange, self.db)
            except ImportError:
                # If not available, create a placeholder implementation
                self.logger.warning("FinancialScraper not available, using placeholder")
                
                class PlaceholderScraper:
                    def __init__(self, symbol, exchange, db):
                        self.symbol = symbol
                        self.exchange = exchange
                        self.db = db
                        self.logger = setup_logger("placeholder_scraper")
                    
                    def run(self):
                        self.logger.info(f"Placeholder: Collecting financial data for {self.symbol}:{self.exchange}")
                        
                        # Update data collection status
                        
                        portfolio_manager = PortfolioManager(self.db)
                        portfolio_manager.update_data_collection_status(self.symbol, self.exchange, "financial", True)
                        
                        return {"status": "success"}
                
                scraper = PlaceholderScraper(symbol, exchange, self.db)
            
            # Run the scraper
            result = scraper.run()
            
            # Check result
            if result:
                # Update data collection status
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "financial", True)
                
                self.logger.info(f"Completed financial data collection for {symbol}:{exchange}")
            else:
                self.logger.error(f"Failed to collect financial data for {symbol}:{exchange}")
                
                # Update data collection status with failure
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "financial", False)
            
        except Exception as e:
            log_error(e, context={"action": "collect_financial_data", "symbol": symbol, "exchange": exchange})
            
            # Update data collection status with failure
            try:
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "financial", False)
            except:
                pass
    
    def _collect_news_data(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Collect news data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            self.logger.info(f"Collecting news data for {symbol}:{exchange}")
            
            # Import necessary components
            try:
                # First try to import the real implementation
               
                aggregator = NewsAggregator(self.db)
            except ImportError:
                # If not available, create a placeholder implementation
                self.logger.warning("NewsAggregator not available, using placeholder")
                
                class PlaceholderAggregator:
                    def __init__(self, db):
                        self.db = db
                        self.logger = setup_logger("placeholder_aggregator")
                    
                    def collect_news(self, symbol, exchange, days=30, limit=100):
                        self.logger.info(f"Placeholder: Collecting news for {symbol}:{exchange}")
                        
                        # Update data collection status
                         
                        portfolio_manager = PortfolioManager(self.db)
                        portfolio_manager.update_data_collection_status(symbol, exchange, "news", True)
                        
                        return []
                
                aggregator = PlaceholderAggregator(self.db)
            
            # Collect news for the specified days
            days = settings.NEWS_DAYS_DEFAULT
            
            news_items = aggregator.collect_news(symbol, exchange, days=days)
            
            # Check result
            if news_items is not None:
                # Update data collection status
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "news", True)
                
                self.logger.info(f"Collected {len(news_items)} news items for {symbol}:{exchange}")
            else:
                self.logger.error(f"Failed to collect news data for {symbol}:{exchange}")
                
                # Update data collection status with failure
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "news", False)
            
        except Exception as e:
            log_error(e, context={"action": "collect_news_data", "symbol": symbol, "exchange": exchange})
            
            # Update data collection status with failure
            try:
               
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "news", False)
            except:
                pass
    
    def _collect_global_data(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Collect global market data
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            sector = config.get("sector")
            
            if not sector:
                self.logger.warning(f"No sector specified for {symbol}:{exchange}, skipping global data collection")
                
                # Update data collection status (marked as complete since we're skipping)
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "global", True)
                
                return
            
            self.logger.info(f"Collecting global data for {symbol}:{exchange} (sector: {sector})")
            
            # Import necessary components
            try:
                # First try to import the real implementation
                collector = IndicesCollector(self.db)
            except ImportError:
                # If not available, create a placeholder implementation
                self.logger.warning("IndicesCollector not available, using placeholder")
                
                class PlaceholderCollector:
                    def __init__(self, db):
                        self.db = db
                        self.logger = setup_logger("placeholder_collector")
                    
                    def collect_data(self, sector, days=30):
                        self.logger.info(f"Placeholder: Collecting global data for sector: {sector}")
                        return []
                
                collector = PlaceholderCollector(self.db)
            
            # Collect global data for the specified sector
            days = 30  # 1 month
            
            global_data = collector.collect_data(sector, days=days)
            
            # Update data collection status
            
            portfolio_manager = PortfolioManager(self.db)
            portfolio_manager.update_data_collection_status(symbol, exchange, "global", True)
            
            self.logger.info(f"Completed global data collection for {symbol}:{exchange} (sector: {sector})")
            
        except Exception as e:
            log_error(e, context={"action": "collect_global_data", "symbol": symbol, "exchange": exchange})
            
            # Update data collection status with failure
            try:
                
                portfolio_manager = PortfolioManager(self.db)
                portfolio_manager.update_data_collection_status(symbol, exchange, "global", False)
            except:
                pass
    
    def _run_initial_analysis(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Run initial analysis for the instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            self.logger.info(f"Running initial analysis for {symbol}:{exchange}")
            
            # Run technical analysis
            self._run_technical_analysis(symbol, exchange, config)
            
            # Run fundamental analysis if applicable
            if "financial" in config.get("data_sources", []):
                self._run_fundamental_analysis(symbol, exchange, config)
            
            # Run sentiment analysis
            self._run_sentiment_analysis(symbol, exchange, config)
            
            # Generate initial predictions
            self._generate_initial_prediction(symbol, exchange, config)
            
            self.logger.info(f"Completed initial analysis for {symbol}:{exchange}")
            
        except Exception as e:
            log_error(e, context={"action": "run_initial_analysis", "symbol": symbol, "exchange": exchange})
    
    def _run_technical_analysis(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Run technical analysis
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            self.logger.info(f"Running technical analysis for {symbol}:{exchange}")
            
            # Import necessary components
            try:
                # First try to import the real implementation
                
                analyzer = TechnicalAnalyzer(self.db)
            except ImportError:
                # If not available, create a placeholder implementation
                self.logger.warning("TechnicalAnalyzer not available, using placeholder")
                
                class PlaceholderAnalyzer:
                    def __init__(self, db):
                        self.db = db
                        self.logger = setup_logger("placeholder_analyzer")
                    
                    def analyze(self, symbol, exchange, timeframe=None):
                        timeframe = timeframe or ["day", "60min", "5min"]
                        self.logger.info(f"Placeholder: Running technical analysis for {symbol}:{exchange}")
                        return {"status": "success"}
                
                analyzer = PlaceholderAnalyzer(self.db)
            
            # Get timeframes from config
            timeframes = config.get("timeframes", ["day", "60min", "5min"])
            
            # Run technical analysis
            result = analyzer.analyze(symbol, exchange, timeframes=timeframes)
            
            if result:
                self.logger.info(f"Completed technical analysis for {symbol}:{exchange}")
            else:
                self.logger.error(f"Failed to run technical analysis for {symbol}:{exchange}")
            
        except Exception as e:
            log_error(e, context={"action": "run_technical_analysis", "symbol": symbol, "exchange": exchange})
    
    def _run_fundamental_analysis(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Run fundamental analysis
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            self.logger.info(f"Running fundamental analysis for {symbol}:{exchange}")
            
            # Import necessary components
            try:
                # First try to import the real implementation
                
                analyzer = FundamentalAnalyzer(self.db)
            except ImportError:
                # If not available, create a placeholder implementation
                self.logger.warning("FundamentalAnalyzer not available, using placeholder")
                
                class PlaceholderAnalyzer:
                    def __init__(self, db):
                        self.db = db
                        self.logger = setup_logger("placeholder_analyzer")
                    
                    def analyze(self, symbol, exchange):
                        self.logger.info(f"Placeholder: Running fundamental analysis for {symbol}:{exchange}")
                        return {"status": "success"}
                
                analyzer = PlaceholderAnalyzer(self.db)
            
            # Run fundamental analysis
            result = analyzer.analyze(symbol, exchange)
            
            if result:
                self.logger.info(f"Completed fundamental analysis for {symbol}:{exchange}")
            else:
                self.logger.error(f"Failed to run fundamental analysis for {symbol}:{exchange}")
            
        except Exception as e:
            log_error(e, context={"action": "run_fundamental_analysis", "symbol": symbol, "exchange": exchange})
    
    def _run_sentiment_analysis(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
        """
        Run sentiment analysis
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            config (dict): Instrument configuration
        """
        try:
            self.logger.info(f"Running sentiment analysis for {symbol}:{exchange}")
            
            # Import necessary components
            try:
                # First try to import the real implementation
                
                analyzer = SentimentAnalyzer(self.db)
            except ImportError:
                # If not available, create a placeholder implementation
                self.logger.warning("SentimentAnalyzer not available, using placeholder")
                
                class PlaceholderAnalyzer:
                    def __init__(self, db):
                        self.db = db
                        self.logger = setup_logger("placeholder_analyzer")
                    
                    def analyze(self, symbol, exchange, days=30):
                        self.logger.info(f"Placeholder: Running sentiment analysis for {symbol}:{exchange}")
                        return {"status": "success"}
                
                analyzer = PlaceholderAnalyzer(self.db)
            
            # Run sentiment analysis for the last 30 days
            days = 30
            
            result = analyzer.analyze(symbol, exchange, days=days)
            
            if result:
                self.logger.info(f"Completed sentiment analysis for {symbol}:{exchange}")
            else:
                self.logger.error(f"Failed to run sentiment analysis for {symbol}:{exchange}")
            
        except Exception as e:
            log_error(e, context={"action": "run_sentiment_analysis", "symbol": symbol, "exchange": exchange})
    
    def _generate_initial_prediction(self, symbol: str, exchange: str, config: Dict[str, Any]) -> None:
       """
       Generate initial prediction
       
       Args:
           symbol (str): Instrument symbol
           exchange (str): Exchange code
           config (dict): Instrument configuration
       """
       try:
           self.logger.info(f"Generating initial prediction for {symbol}:{exchange}")
           
           # Import necessary components
           try:
               # First try to import the real implementation
               
               predictor = DailyPredictor(self.db)
           except ImportError:
               # If not available, create a placeholder implementation
               self.logger.warning("DailyPredictor not available, using placeholder")
               
               class PlaceholderPredictor:
                   def __init__(self, db):
                       self.db = db
                       self.logger = setup_logger("placeholder_predictor")
                   
                   def predict(self, symbol, exchange, timeframe="day"):
                       self.logger.info(f"Placeholder: Generating prediction for {symbol}:{exchange}")
                      
                       
                       # Generate a random prediction
                       prediction = "up" if random.random() > 0.5 else "down"
                       confidence = random.uniform(0.6, 0.9)
                       
                       # Create prediction data
                       prediction_data = PredictionData(
                           symbol=symbol,
                           exchange=exchange,
                           date=datetime.now(),
                           prediction=prediction,
                           confidence=confidence,
                           timeframe="intraday",
                           supporting_factors=[
                               {"factor": "placeholder", "weight": 1.0}
                           ],
                           model_id="placeholder_model"
                       )
                       
                       # Save to database
                       self.db.save_prediction(prediction_data.to_dict())
                       
                       return prediction_data.to_dict()
               
               predictor = PlaceholderPredictor(self.db)
           
           # Generate predictions for different timeframes
           timeframes = ["day"]
           if "intraday" in config.get("trading_timeframe", ""):
               timeframes.append("intraday")
           
           for timeframe in timeframes:
               result = predictor.predict(symbol, exchange, timeframe=timeframe)
               
               if result:
                   self.logger.info(f"Generated {timeframe} prediction for {symbol}:{exchange}")
               else:
                   self.logger.error(f"Failed to generate {timeframe} prediction for {symbol}:{exchange}")
           
       except Exception as e:
           log_error(e, context={"action": "generate_initial_prediction", "symbol": symbol, "exchange": exchange})