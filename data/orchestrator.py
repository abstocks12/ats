"""
Data Orchestration Module for the Automated Trading System.
Coordinates and manages data collection across different sources.
"""

import os
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from database.connection_manager import get_db
from utils.logging_utils import setup_logger, log_error, log_execution_time
from utils.helper_functions import retry_function, get_date_range
 

class DataOrchestrator:
    """
    Coordinates and manages data collection across different data sources.
    Provides a central interface for requesting and scheduling data collection.
    """
    
    def __init__(self, db=None):
        """
        Initialize the data orchestrator
        
        Args:
            db: Database connector (optional, will use global connection if not provided)
        """
        self.logger = setup_logger(__name__)
        self.db = db or get_db()
        
        self.db_optimizer = db.get_optimizer()
        self.time_partitioner = db.get_partitioner()
        
        # Set up partitioning for key collections
        self._setup_partitioning()

        # Track ongoing collection tasks
        self.active_collections = {}
        self.collection_locks = {}
        
        # Create locks for different data types
        for data_type in ["market", "financial", "news", "global", "alternative"]:
            self.collection_locks[data_type] = threading.Lock()
    
    @log_execution_time(setup_logger("timing_orchestrator"))
    def collect_all_data(self, symbol: str, exchange: str, instrument_type: Optional[str] = None,
                        days: int = None) -> Dict[str, bool]:
        """
        Collect all types of data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str, optional): Instrument type
            days (int, optional): Number of days to collect (default: from settings)
            
        Returns:
            dict: Results of collection operations
        """
        self.logger.info(f"Starting complete data collection for {symbol}:{exchange}")
        
        # Get instrument details
        instrument = self._get_instrument_details(symbol, exchange, instrument_type)
        
        # Determine collection days
        if days is None:
            days = settings.HISTORICAL_DAYS_DEFAULT
        
        # Initialize results dictionary
        results = {
            "market": False,
            "financial": False, 
            "news": False,
            "global": False,
            "alternative": False
        }
        
        # Collect data in separate threads
        threads = []
        
        # Market data collection
        market_thread = threading.Thread(
            target=self._collect_market_data_thread,
            args=(symbol, exchange, instrument, days, results)
        )
        market_thread.daemon = True
        threads.append(market_thread)
        
        # Financial data collection (if applicable)
        if instrument_type != "index":
            financial_thread = threading.Thread(
                target=self._collect_financial_data_thread,
                args=(symbol, exchange, instrument, results)
            )
            financial_thread.daemon = True
            threads.append(financial_thread)
        
        # News data collection
        news_thread = threading.Thread(
            target=self._collect_news_data_thread,
            args=(symbol, exchange, instrument, days, results)
        )
        news_thread.daemon = True
        threads.append(news_thread)
        
        # Global market data (if applicable for the symbol)
        if instrument.get("sector"):
            global_thread = threading.Thread(
                target=self._collect_global_data_thread,
                args=(symbol, exchange, instrument, days, results)
            )
            global_thread.daemon = True
            threads.append(global_thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete (with timeout)
        timeout_per_thread = 300  # 5 minutes per thread
        for thread in threads:
            thread.join(timeout=timeout_per_thread)
        
        # Update instrument data collection status
        self._update_collection_status(symbol, exchange, results)
        
        self.logger.info(f"Completed data collection for {symbol}:{exchange}")
        return results
    
    def _setup_partitioning(self):
        """Set up time-based partitioning for important collections."""
        collections = ["market_data_collection", "trades_collection", "news_collection"]
        for collection in collections:
            self.time_partitioner.setup_partitioning(collection)
    
    def schedule_optimization_tasks(self):
        """Schedule regular database optimization tasks."""
        # This would be called during system initialization
        # And would integrate with your task scheduler
        
        # Run optimization daily at midnight
        from automation.scheduler import Scheduler
        scheduler = Scheduler()
        scheduler.schedule_daily(
            name="database_optimization",
            time="00:00",
            task=self.db_optimizer.optimize_database
        )
        
        # Run partition cleanup weekly
        scheduler.schedule_weekly(
            name="partition_cleanup",
            day=0,  # Sunday
            time="01:00",
            task=self.time_partitioner.cleanup_partitions
        )

    def collect_market_data(self, symbol: str, exchange: str, timeframes: List[str] = None,
                           days: int = None) -> bool:
        """
        Collect market data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            timeframes (list, optional): List of timeframes to collect
            days (int, optional): Number of days to collect
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Acquire lock for market data collection
        if not self.collection_locks["market"].acquire(blocking=False):
            self.logger.warning(f"Market data collection already in progress for {symbol}:{exchange}")
            return False
        
        try:
            # Set collection as active
            collection_key = f"market_{symbol}_{exchange}"
            self.active_collections[collection_key] = datetime.now()
            
            self.logger.info(f"Starting market data collection for {symbol}:{exchange}")
            
            # Default timeframes if not provided
            if timeframes is None:
                timeframes = ["day", "60min", "15min", "5min", "1min"]
            
            # Default days if not provided
            if days is None:
                days = settings.HISTORICAL_DAYS_DEFAULT
            
            try:
                # Import historical data collector
                from data.market.historical_data import HistoricalDataCollector
                
                # Create collector instance
                collector = HistoricalDataCollector(self.db)
                
                # Collect data for each timeframe
                success_count = 0
                for timeframe in timeframes:
                    try:
                        result = collector.collect_data(
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=timeframe,
                            days=days
                        )
                        
                        if result:
                            success_count += 1
                            self.logger.info(f"Collected {timeframe} data for {symbol}:{exchange}")
                        else:
                            self.logger.warning(f"Failed to collect {timeframe} data for {symbol}:{exchange}")
                    except Exception as e:
                        log_error(e, context={"action": "collect_market_data", "timeframe": timeframe})
                
                # Update collection status
                if success_count > 0:
                    self._update_single_collection_status(symbol, exchange, "historical", True)
                    return True
                else:
                    self._update_single_collection_status(symbol, exchange, "historical", False)
                    return False
                
            except ImportError:
                self.logger.error("HistoricalDataCollector not available")
                return False
            except Exception as e:
                log_error(e, context={"action": "collect_market_data"})
                return False
                
        finally:
            # Remove from active collections
            collection_key = f"market_{symbol}_{exchange}"
            if collection_key in self.active_collections:
                del self.active_collections[collection_key]
            
            # Release the lock
            self.collection_locks["market"].release()
    
    def collect_financial_data(self, symbol: str, exchange: str) -> bool:
        """
        Collect financial data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Acquire lock for financial data collection
        if not self.collection_locks["financial"].acquire(blocking=False):
            self.logger.warning(f"Financial data collection already in progress for {symbol}:{exchange}")
            return False
        
        try:
            # Set collection as active
            collection_key = f"financial_{symbol}_{exchange}"
            self.active_collections[collection_key] = datetime.now()
            
            self.logger.info(f"Starting financial data collection for {symbol}:{exchange}")
            
            try:
                # Import financial scraper
                try:
                    from data.financial.financial_scraper import FinancialScraper
                    
                    # Create scraper instance
                    scraper = FinancialScraper(symbol, exchange, self.db)
                    
                    # Run scraper
                    result = scraper.run()
                    
                    if result:
                        self.logger.info(f"Collected financial data for {symbol}:{exchange}")
                        self._update_single_collection_status(symbol, exchange, "financial", True)
                        return True
                    else:
                        self.logger.warning(f"Failed to collect financial data for {symbol}:{exchange}")
                        self._update_single_collection_status(symbol, exchange, "financial", False)
                        return False
                        
                except ImportError:
                    self.logger.error("FinancialScraper not available")
                    return False
                
            except Exception as e:
                log_error(e, context={"action": "collect_financial_data"})
                self._update_single_collection_status(symbol, exchange, "financial", False)
                return False
                
        finally:
            # Remove from active collections
            collection_key = f"financial_{symbol}_{exchange}"
            if collection_key in self.active_collections:
                del self.active_collections[collection_key]
            
            # Release the lock
            self.collection_locks["financial"].release()
    
    def collect_news_data(self, symbol: str, exchange: str, days: int = None) -> bool:
        """
        Collect news data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            days (int, optional): Number of days of news to collect
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Acquire lock for news data collection
        if not self.collection_locks["news"].acquire(blocking=False):
            self.logger.warning(f"News data collection already in progress for {symbol}:{exchange}")
            return False
        
        try:
            # Set collection as active
            collection_key = f"news_{symbol}_{exchange}"
            self.active_collections[collection_key] = datetime.now()
            
            self.logger.info(f"Starting news data collection for {symbol}:{exchange}")
            
            # Default days if not provided
            if days is None:
                days = settings.NEWS_DAYS_DEFAULT
            
            try:
                # Import news aggregator
                try:
                    from data.news.news_aggregator import NewsAggregator
                    
                    # Create aggregator instance
                    aggregator = NewsAggregator(self.db)
                    
                    # Collect news
                    news_items = aggregator.collect_news(
                        symbol=symbol,
                        exchange=exchange,
                        days=days
                    )
                    
                    if news_items is not None:
                        self.logger.info(f"Collected {len(news_items)} news items for {symbol}:{exchange}")
                        self._update_single_collection_status(symbol, exchange, "news", True)
                        return True
                    else:
                        self.logger.warning(f"Failed to collect news for {symbol}:{exchange}")
                        self._update_single_collection_status(symbol, exchange, "news", False)
                        return False
                        
                except ImportError:
                    self.logger.error("NewsAggregator not available")
                    return False
                
            except Exception as e:
                log_error(e, context={"action": "collect_news_data"})
                self._update_single_collection_status(symbol, exchange, "news", False)
                return False
                
        finally:
            # Remove from active collections
            collection_key = f"news_{symbol}_{exchange}"
            if collection_key in self.active_collections:
                del self.active_collections[collection_key]
            
            # Release the lock
            self.collection_locks["news"].release()
    
    def collect_global_data(self, sector: str, days: int = None) -> bool:
        """
        Collect global market data for a sector
        
        Args:
            sector (str): Market sector
            days (int, optional): Number of days to collect
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Acquire lock for global data collection
        if not self.collection_locks["global"].acquire(blocking=False):
            self.logger.warning(f"Global data collection already in progress for sector {sector}")
            return False
        
        try:
            # Set collection as active
            collection_key = f"global_{sector}"
            self.active_collections[collection_key] = datetime.now()
            
            self.logger.info(f"Starting global data collection for sector {sector}")
            
            # Default days if not provided
            if days is None:
                days = 30  # Default to 30 days for global data
            
            try:
                # Import global indices collector
                try:
                    from data.global_markets.indices_collector import GlobalIndicesCollector
                    
                    # Create collector instance
                    collector = GlobalIndicesCollector(self.db)
                    
                    # Collect data
                    result = collector.collect_data(
                        sector=sector,
                        days=days
                    )
                    
                    if result:
                        self.logger.info(f"Collected global data for sector {sector}")
                        return True
                    else:
                        self.logger.warning(f"Failed to collect global data for sector {sector}")
                        return False
                        
                except ImportError:
                    self.logger.error("GlobalIndicesCollector not available")
                    return False
                
            except Exception as e:
                log_error(e, context={"action": "collect_global_data"})
                return False
                
        finally:
            # Remove from active collections
            collection_key = f"global_{sector}"
            if collection_key in self.active_collections:
                del self.active_collections[collection_key]
            
            # Release the lock
            self.collection_locks["global"].release()
    
    def collect_alternative_data(self, symbol: str, exchange: str, data_type: str = "all") -> bool:
        """
        Collect alternative data for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            data_type (str): Type of alternative data ('social', 'trends', or 'all')
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Acquire lock for alternative data collection
        if not self.collection_locks["alternative"].acquire(blocking=False):
            self.logger.warning(f"Alternative data collection already in progress for {symbol}:{exchange}")
            return False
        
        try:
            # Set collection as active
            collection_key = f"alternative_{symbol}_{exchange}"
            self.active_collections[collection_key] = datetime.now()
            
            self.logger.info(f"Starting alternative data collection for {symbol}:{exchange}")
            
            results = {}
            
            # Collect social sentiment data
            if data_type in ["social", "all"]:
                try:
                    from data.alternative.social_sentiment import SocialSentimentCollector
                    
                    # Create collector instance
                    social_collector = SocialSentimentCollector(self.db)
                    
                    # Collect data
                    social_result = social_collector.collect_data(
                        symbol=symbol,
                        exchange=exchange
                    )
                    
                    results["social"] = social_result
                    
                except ImportError:
                    self.logger.error("SocialSentimentCollector not available")
                    results["social"] = False
                except Exception as e:
                    log_error(e, context={"action": "collect_social_sentiment"})
                    results["social"] = False
            
            # Collect Google Trends data
            if data_type in ["trends", "all"]:
                try:
                    from data.alternative.google_trends import GoogleTrendsCollector
                    
                    # Create collector instance
                    trends_collector = GoogleTrendsCollector(self.db)
                    
                    # Collect data
                    trends_result = trends_collector.collect_data(
                        symbol=symbol,
                        exchange=exchange
                    )
                    
                    results["trends"] = trends_result
                    
                except ImportError:
                    self.logger.error("GoogleTrendsCollector not available")
                    results["trends"] = False
                except Exception as e:
                    log_error(e, context={"action": "collect_google_trends"})
                    results["trends"] = False
            
            # Check results
            if any(results.values()):
                self.logger.info(f"Collected alternative data for {symbol}:{exchange}")
                return True
            else:
                self.logger.warning(f"Failed to collect alternative data for {symbol}:{exchange}")
                return False
                
        finally:
            # Remove from active collections
            collection_key = f"alternative_{symbol}_{exchange}"
            if collection_key in self.active_collections:
                del self.active_collections[collection_key]
            
            # Release the lock
            self.collection_locks["alternative"].release()
    
    def get_collection_status(self, symbol: str = None, exchange: str = None) -> Dict[str, Any]:
        """
        Get status of ongoing and completed data collections
        
        Args:
            symbol (str, optional): Filter by instrument symbol
            exchange (str, optional): Filter by exchange
            
        Returns:
            dict: Collection status information
        """
        # Get active collections
        active = {}
        for key, start_time in self.active_collections.items():
            # Filter by symbol and exchange if provided
            if symbol and exchange:
                if not (symbol in key and exchange in key):
                    continue
            elif symbol:
                if symbol not in key:
                    continue
            elif exchange:
                if exchange not in key:
                    continue
            
            # Calculate duration
            duration = datetime.now() - start_time
            
            # Add to active collections
            parts = key.split('_')
            data_type = parts[0]
            
            if len(parts) > 2:
                item_symbol = parts[1]
                item_exchange = parts[2]
                
                active[key] = {
                    "data_type": data_type,
                    "symbol": item_symbol,
                    "exchange": item_exchange,
                    "start_time": start_time.isoformat(),
                    "duration_seconds": duration.total_seconds()
                }
            else:
                # For global data which doesn't have symbol/exchange
                item_sector = parts[1]
                
                active[key] = {
                    "data_type": data_type,
                    "sector": item_sector,
                    "start_time": start_time.isoformat(),
                    "duration_seconds": duration.total_seconds()
                }
        
        # Get completed collections (from portfolio status)
        completed = {}
        
        if symbol and exchange:
            # Get specific instrument
            instrument = self._get_instrument_details(symbol, exchange)
            
            if instrument and "data_collection_status" in instrument:
                completed[f"{symbol}_{exchange}"] = instrument["data_collection_status"]
        else:
            # Get all instruments
            query = {"status": "active"}
            
            if symbol:
                query["symbol"] = symbol
            if exchange:
                query["exchange"] = exchange
            
            instruments = self.db.portfolio_collection.find(query)
            
            for instrument in instruments:
                if "data_collection_status" in instrument:
                    key = f"{instrument['symbol']}_{instrument['exchange']}"
                    completed[key] = instrument["data_collection_status"]
        
        return {
            "active_collections": active,
            "completed_collections": completed
        }
    
    def update_real_time_data(self, symbols: List[str], exchanges: List[str]) -> Dict[str, bool]:
        """
        Update real-time data for a list of instruments
        
        Args:
            symbols (list): List of instrument symbols
            exchanges (list): List of exchanges
            
        Returns:
            dict: Results of update operations
        """
        if len(symbols) != len(exchanges):
            self.logger.error("Number of symbols and exchanges must match")
            return {}
        
        self.logger.info(f"Updating real-time data for {len(symbols)} instruments")
        
        results = {}
        
        try:
            # Import real-time data collector
            from data.market.real_time import RealTimeDataCollector
            
            # Create collector instance
            collector = RealTimeDataCollector(self.db)
            
            # Start the collector if needed
            if not collector.is_connected:
                collector.start()
            
            # Batch collect data
            batch_results = collector.collect_batch(symbols, exchanges)
            
            # Process results
            for i, symbol in enumerate(symbols):
                exchange = exchanges[i]
                key = f"{symbol}@{exchange}"
                
                if key in batch_results:
                    results[key] = True
                else:
                    results[key] = False
            
            self.logger.info(f"Updated real-time data for {sum(results.values())} out of {len(symbols)} instruments")
            
        except ImportError:
            self.logger.error("RealTimeDataCollector not available")
        except Exception as e:
            log_error(e, context={"action": "update_real_time_data"})
        
        return results
    
    def _collect_market_data_thread(self, symbol: str, exchange: str, instrument: Dict[str, Any],
                                   days: int, results: Dict[str, bool]) -> None:
        """
        Thread function for market data collection
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument (dict): Instrument details
            days (int): Number of days to collect
            results (dict): Results dictionary to update
        """
        try:
            # Get timeframes from instrument configuration
            timeframes = instrument.get("timeframes", ["day", "60min", "15min", "5min", "1min"])
            
            # Collect market data
            market_result = self.collect_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframes=timeframes,
                days=days
            )
            
            # Update results
            results["market"] = market_result
            
        except Exception as e:
            log_error(e, context={"action": "collect_market_data_thread"})
            results["market"] = False
    
    def _collect_financial_data_thread(self, symbol: str, exchange: str, instrument: Dict[str, Any],
                                      results: Dict[str, bool]) -> None:
        """
        Thread function for financial data collection
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument (dict): Instrument details
            results (dict): Results dictionary to update
        """
        try:
            # Collect financial data
            financial_result = self.collect_financial_data(
                symbol=symbol,
                exchange=exchange
            )
            
            # Update results
            results["financial"] = financial_result
            
        except Exception as e:
            log_error(e, context={"action": "collect_financial_data_thread"})
            results["financial"] = False
    
    def _collect_news_data_thread(self, symbol: str, exchange: str, instrument: Dict[str, Any],
                                 days: int, results: Dict[str, bool]) -> None:
        """
        Thread function for news data collection
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument (dict): Instrument details
            days (int): Number of days to collect
            results (dict): Results dictionary to update
        """
        try:
            # Collect news data
            news_result = self.collect_news_data(
                symbol=symbol,
                exchange=exchange,
                days=days
            )
            
            # Update results
            results["news"] = news_result
            
        except Exception as e:
            log_error(e, context={"action": "collect_news_data_thread"})
            results["news"] = False
    
    def _collect_global_data_thread(self, symbol: str, exchange: str, instrument: Dict[str, Any],
                                   days: int, results: Dict[str, bool]) -> None:
        """
        Thread function for global data collection
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument (dict): Instrument details
            days (int): Number of days to collect
            results (dict): Results dictionary to update
        """
        try:
            # Get sector
            sector = instrument.get("sector")
            
            if not sector:
                self.logger.warning(f"No sector specified for {symbol}:{exchange}, skipping global data collection")
                results["global"] = False
                return
            
            # Collect global data
            global_result = self.collect_global_data(
                sector=sector,
                days=days
            )
            
            # Update results
            results["global"] = global_result
            
            # Update instrument collection status
            if global_result:
                self._update_single_collection_status(symbol, exchange, "global", True)
            else:
                self._update_single_collection_status(symbol, exchange, "global", False)
            
        except Exception as e:
            log_error(e, context={"action": "collect_global_data_thread"})
            results["global"] = False
    
    def _get_instrument_details(self, symbol: str, exchange: str, instrument_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get instrument details from portfolio collection
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            instrument_type (str, optional): Instrument type
            
        Returns:
            dict: Instrument details
        """
        # Check if instrument exists in portfolio
        instrument = self.db.portfolio_collection.find_one({
            "symbol": symbol,
            "exchange": exchange,
            "status": "active"
        })
        
        if instrument:
            return instrument
        
        # If not in portfolio, create a minimal instrument details dict
        return {
            "symbol": symbol,
            "exchange": exchange,
            "instrument_type": instrument_type or "equity",
            "timeframes": ["day", "60min", "15min", "5min", "1min"]
        }
    
    def _update_collection_status(self, symbol: str, exchange: str, results: Dict[str, bool]) -> None:
        """
        Update data collection status for an instrument
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            results (dict): Collection results
        """
        try:
            # Check if instrument exists in portfolio
            instrument = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "status": "active"
            })
            
            if not instrument:
                return
            
            # Update status for each data type
            for data_type, success in results.items():
                if data_type == "market":
                    key = "historical"  # Map market to historical in status
                else:
                    key = data_type
                
                self.db.portfolio_collection.update_one(
                    {"_id": instrument["_id"]},
                    {"$set": {f"data_collection_status.{key}": success}}
                )
            
            # Get updated instrument status
            updated = self.db.portfolio_collection.find_one({"_id": instrument["_id"]})
            
            # Check if all collections are done and successful
            if updated and "data_collection_status" in updated:
                if all(updated["data_collection_status"].values()):
                    # Enable trading for this instrument
                    self.db.portfolio_collection.update_one(
                        {"_id": instrument["_id"]},
                        {"$set": {"trading_config.enabled": True}}
                    )
                    
                    self.logger.info(f"All data collection completed for {symbol}:{exchange}, trading enabled")
            
        except Exception as e:
            log_error(e, context={"action": "update_collection_status"})
    
    def _update_single_collection_status(self, symbol: str, exchange: str, data_type: str, success: bool) -> None:
        """
        Update a single data collection status
        
        Args:
            symbol (str): Instrument symbol
            exchange (str): Exchange code
            data_type (str): Data type
            success (bool): Whether collection was successful
        """
        try:
            # Check if instrument exists in portfolio
            instrument = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "status": "active"
            })
            
            if not instrument:
                return
            
            # Update status
            self.db.portfolio_collection.update_one(
                {"_id": instrument["_id"]},
                {"$set": {f"data_collection_status.{data_type}": success}}
            )
            
            # Get updated instrument status
            updated = self.db.portfolio_collection.find_one({"_id": instrument["_id"]})
            
            # Check if all collections are done and successful
            if updated and "data_collection_status" in updated:
                if all(updated["data_collection_status"].values()):
                    # Enable trading for this instrument
                    self.db.portfolio_collection.update_one(
                        {"_id": instrument["_id"]},
                        {"$set": {"trading_config.enabled": True}}
                    )
                    
                    self.logger.info(f"All data collection completed for {symbol}:{exchange}, trading enabled")
            
        except Exception as e:
            log_error(e, context={"action": "update_single_collection_status"})