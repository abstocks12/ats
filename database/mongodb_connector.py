"""
MongoDB connector for the Automated Trading System.
Handles database connections, collections, and basic CRUD operations.
"""

import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from database.database_optimizer import DatabaseOptimizer
from database.query_optimizer import QueryOptimizer
from database.time_series_partitioner import TimeSeriesPartitioner

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pymongo
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.database import Database
    from pymongo.errors import ConnectionFailure, OperationFailure, ServerSelectionTimeoutError
except ImportError:
    print("Error: pymongo not installed. Run 'pip install pymongo'")
    sys.exit(1)

from config import settings
from utils.logging_utils import setup_logger, log_error

class MongoDBConnector:
    """MongoDB connector for the Automated Trading System"""
    
    def __init__(self, uri=None, db_name=None, username=None, password=None):
        """
        Initialize MongoDB connector
        
        Args:
            uri (str, optional): MongoDB connection URI
            db_name (str, optional): Database name
            username (str, optional): MongoDB username
            password (str, optional): MongoDB password
        """
        self.logger = setup_logger(__name__)
        
        # Use provided parameters or fall back to settings
        self.uri = uri or settings.MONGO_URI
        self.db_name = db_name or settings.MONGO_DB_NAME
        self.username = username or settings.MONGO_USERNAME
        self.password = password or settings.MONGO_PASSWORD
        
        # Connection and database objects
        self.client = None
        self.db = None
        
        # Try to connect to MongoDB
        self._connect()
    
    def get_optimizer(self):
        """Get a database optimizer instance."""
        return DatabaseOptimizer(self)
    
    def get_query_optimizer(self):
        """Get a query optimizer instance."""
        return QueryOptimizer(self)
    
    def get_partitioner(self):
        """Get a time series partitioner instance."""
        return TimeSeriesPartitioner(self)
    
    def optimize_database(self):
        """Run database optimization."""
        optimizer = self.get_optimizer()
        return optimizer.optimize_database()

    def _connect(self):
        """Connect to MongoDB database"""
        try:
            # Setup connection options
            connect_options = {
                'serverSelectionTimeoutMS': 5000,  # 5 seconds timeout
                'connectTimeoutMS': 10000,
                'socketTimeoutMS': 45000,
                'maxPoolSize': 100,
                'minPoolSize': 10,
                'maxIdleTimeMS': 30000,
                'waitQueueTimeoutMS': 10000
            }
            
            # Add authentication if provided
            if self.username and self.password:
                connect_options['username'] = self.username
                connect_options['password'] = self.password
            
            # Create client
            self.client = MongoClient(self.uri, **connect_options)
            
            # Test connection
            self.client.admin.command('ping')
            
            # Access database
            self.db = self.client[self.db_name]
            
            # Setup instance variables for collections
            self._setup_collections()
            
            self.logger.info(f"Connected to MongoDB database: {self.db_name}")
            
        except (ConnectionFailure, ServerSelectionTimeoutError) as e:
            self.logger.error(f"Failed to connect to MongoDB: {e}")
            self.client = None
            self.db = None
            # Re-raise the error to be handled by the caller
            raise
    
    def _setup_collections(self):
        """Setup instance variables for collections"""
        # Map collection names to instance variables
        # Map collection names to instance variables
        for name, collection in settings.MONGODB_COLLECTIONS.items():
            setattr(self, f"{name}_collection", self.db[collection])
        
        # Add tasks collection if not already included
        if not hasattr(self, "tasks_collection"):
            self.tasks_collection = self.db["tasks"]
    
    def list_collection_names(self):
        """
        Get list of collection names in the database
        
        Returns:
            list: List of collection names
        """
        self._ensure_connected()
        return self.db.list_collection_names()

    def initialize_database(self):
        """Initialize database with collections and indexes"""
        """Initialize database with collections and indexes"""
        if self.db is None:
            self.logger.error("Database connection not established")
            return False
        
        try:
            # Create collections if they don't exist
            collection_names = self.db.list_collection_names()
            for name, collection_name in settings.MONGODB_COLLECTIONS.items():
                if collection_name not in collection_names:
                    self.db.create_collection(collection_name)
                    self.logger.info(f"Created collection: {collection_name}")
            
            # Create indexes for portfolio collection
            self.portfolio_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("exchange", pymongo.ASCENDING),
                ("status", pymongo.ASCENDING)
            ], unique=True, name="portfolio_symbol_exchange_status_index")
            
            # Create indexes for market data collection
            self.market_data_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("exchange", pymongo.ASCENDING),
                ("timeframe", pymongo.ASCENDING),
                ("timestamp", pymongo.ASCENDING)
            ], unique=True, name="market_data_symbol_exchange_timeframe_timestamp_index")
            
            # Create indexes for news collection
            self.news_collection.create_index([
                ("title", pymongo.ASCENDING)
            ], unique=True, name="news_title_index")
            
            self.news_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("published_date", pymongo.DESCENDING)
            ], name="news_symbol_published_date_index")
            
            # Create indexes for financial collection
            self.financial_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("exchange", pymongo.ASCENDING),
                ("report_type", pymongo.ASCENDING),
                ("period", pymongo.ASCENDING)
            ], unique=True, name="financial_symbol_exchange_report_period_index")
            
            # Create indexes for predictions collection
            self.predictions_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("exchange", pymongo.ASCENDING),
                ("date", pymongo.DESCENDING)
            ], name="predictions_symbol_exchange_date_index")
            
            # Create indexes for trades collection
            self.trades_collection.create_index([
                ("symbol", pymongo.ASCENDING),
                ("exchange", pymongo.ASCENDING),
                ("entry_time", pymongo.DESCENDING)
            ], name="trades_symbol_exchange_entry_time_index")
            
            # Create indexes for performance collection
            self.performance_collection.create_index([
                ("date", pymongo.DESCENDING)
            ], name="performance_date_index")
            
            # Create indexes for system logs collection
            self.system_logs_collection.create_index([
                ("timestamp", pymongo.DESCENDING)
            ], name="system_logs_timestamp_index")
            
            self.logger.info("Database initialized with collections and indexes")
            return True
            
        except Exception as e:
            log_error(e, context={"action": "initialize_database"})
            return False
    
    def reconnect(self):
        """Reconnect to MongoDB if connection is lost"""
        if self.client is not None:
            self.client.close()
        
        self._connect()
    
    def _ensure_connected(self):
        """Ensure connection to MongoDB is established"""
        if self.client is None or self.db is None:
            self.reconnect()

    def rollback(self):
        """
        Handle rollback requests (MongoDB standalone doesn't support transactions)
        This is a no-op method to maintain API compatibility
        """
        self.logger.warning("Rollback called, but MongoDB standalone doesn't support transactions")
        # In a production system, you might implement a custom rollback mechanism
        # using backup collections or journaling
        pass

    def get_collection(self, collection_name):
        """
        Get MongoDB collection
        
        Args:
            collection_name (str): Name of the collection
            
        Returns:
            Collection: MongoDB collection
        """
        self._ensure_connected()
        
        if collection_name in settings.MONGODB_COLLECTIONS.values():
            return self.db[collection_name]
        else:
            raise ValueError(f"Unknown collection: {collection_name}")
    
    def insert_one(self, collection_name, document, bypass_validation=False):
        """
        Insert a single document into a collection
        
        Args:
            collection_name (str): Name of the collection
            document (dict): Document to insert
            bypass_validation (bool): Whether to bypass document validation
            
        Returns:
            str: ID of the inserted document
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        # Add timestamps if not present
        if 'created_at' not in document:
            document['created_at'] = datetime.now()
        
        if 'updated_at' not in document:
            document['updated_at'] = document['created_at']
        
        try:
            result = collection.insert_one(document, bypass_document_validation=bypass_validation)
            return result.inserted_id
        except Exception as e:
            log_error(e, context={"action": "insert_one", "collection": collection_name})
            return None
    
    def insert_many(self, collection_name, documents, ordered=True, bypass_validation=False):
        """
        Insert multiple documents into a collection
        
        Args:
            collection_name (str): Name of the collection
            documents (list): List of documents to insert
            ordered (bool): Whether to insert documents in order
            bypass_validation (bool): Whether to bypass document validation
            
        Returns:
            list: IDs of inserted documents
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        # Add timestamps if not present
        now = datetime.now()
        for doc in documents:
            if 'created_at' not in doc:
                doc['created_at'] = now
            
            if 'updated_at' not in doc:
                doc['updated_at'] = doc['created_at']
        
        try:
            result = collection.insert_many(
                documents, 
                ordered=ordered, 
                bypass_document_validation=bypass_validation
            )
            return result.inserted_ids
        except Exception as e:
            log_error(e, context={"action": "insert_many", "collection": collection_name})
            return []
    
    def find_one(self, collection_name, query=None, projection=None, sort=None):
        """
        Find a single document in a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict, optional): Query filter
            projection (dict, optional): Projection
            sort (list, optional): Sort specification
            
        Returns:
            dict: Found document or None
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            # Build find operation
            find_op = collection.find(query or {}, projection or {})
            
            # Apply sort if provided
            if sort:
                find_op = find_op.sort(sort)
            
            # Get first document
            return find_op.limit(1).next()
        except StopIteration:
            # No documents found
            return None
        except Exception as e:
            log_error(e, context={"action": "find_one", "collection": collection_name})
            return None
    
    def find(self, collection_name, query=None, projection=None, sort=None, limit=0, skip=0):
        """
        Find documents in a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict, optional): Query filter
            projection (dict, optional): Projection
            sort (list, optional): Sort specification
            limit (int, optional): Maximum number of documents to return
            skip (int, optional): Number of documents to skip
            
        Returns:
            list: List of found documents
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            # Build find operation
            find_op = collection.find(query or {}, projection or {})
            
            # Apply sort, skip, and limit if provided
            if sort:
                find_op = find_op.sort(sort)
            
            if skip:
                find_op = find_op.skip(skip)
            
            if limit:
                find_op = find_op.limit(limit)
            
            # Convert cursor to list
            return list(find_op)
        except Exception as e:
            log_error(e, context={"action": "find", "collection": collection_name})
            return []
    
    def update_one(self, collection_name, query, update, upsert=False):
        """
        Update a single document in a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict): Query filter
            update (dict): Update operations
            upsert (bool): Whether to insert if document doesn't exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            # Add updated_at timestamp if not in $set
            if '$set' in update:
                if 'updated_at' not in update['$set']:
                    update['$set']['updated_at'] = datetime.now()
            else:
                update['$set'] = {'updated_at': datetime.now()}
            
            result = collection.update_one(query, update, upsert=upsert)
            return result.modified_count > 0 or result.upserted_id is not None
        except Exception as e:
            log_error(e, context={"action": "update_one", "collection": collection_name})
            return False
    
    def update_many(self, collection_name, query, update, upsert=False):
        """
        Update multiple documents in a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict): Query filter
            update (dict): Update operations
            upsert (bool): Whether to insert if documents don't exist
            
        Returns:
            int: Number of modified documents
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            # Add updated_at timestamp if not in $set
            if '$set' in update:
                if 'updated_at' not in update['$set']:
                    update['$set']['updated_at'] = datetime.now()
            else:
                update['$set'] = {'updated_at': datetime.now()}
            
            result = collection.update_many(query, update, upsert=upsert)
            return result.modified_count
        except Exception as e:
            log_error(e, context={"action": "update_many", "collection": collection_name})
            return 0
    
    def delete_one(self, collection_name, query):
        """
        Delete a single document from a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict): Query filter
            
        Returns:
            bool: True if successful, False otherwise
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            result = collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            log_error(e, context={"action": "delete_one", "collection": collection_name})
            return False
    
    def delete_many(self, collection_name, query):
        """
        Delete multiple documents from a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict): Query filter
            
        Returns:
            int: Number of deleted documents
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            result = collection.delete_many(query)
            return result.deleted_count
        except Exception as e:
            log_error(e, context={"action": "delete_many", "collection": collection_name})
            return 0
    
    def count_documents(self, collection_name, query=None):
        """
        Count documents in a collection
        
        Args:
            collection_name (str): Name of the collection
            query (dict, optional): Query filter
            
        Returns:
            int: Number of documents
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            return collection.count_documents(query or {})
        except Exception as e:
            log_error(e, context={"action": "count_documents", "collection": collection_name})
            return 0
    
    def aggregate(self, collection_name, pipeline):
        """
        Perform an aggregation on a collection
        
        Args:
            collection_name (str): Name of the collection
            pipeline (list): Aggregation pipeline
            
        Returns:
            list: Aggregation results
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            return list(collection.aggregate(pipeline))
        except Exception as e:
            log_error(e, context={"action": "aggregate", "collection": collection_name})
            return []
    
    def distinct(self, collection_name, field, query=None):
        """
        Get distinct values for a field in a collection
        
        Args:
            collection_name (str): Name of the collection
            field (str): Field name
            query (dict, optional): Query filter
            
        Returns:
            list: Distinct values
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            return collection.distinct(field, query or {})
        except Exception as e:
            log_error(e, context={"action": "distinct", "collection": collection_name})
            return []
    
    def create_index(self, collection_name, keys, **kwargs):
        """
        Create an index for a collection
        
        Args:
            collection_name (str): Name of the collection
            keys (list): List of (key, direction) pairs
            **kwargs: Additional index options
            
        Returns:
            str: Index name
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            return collection.create_index(keys, **kwargs)
        except Exception as e:
            log_error(e, context={"action": "create_index", "collection": collection_name})
            return None
    
    def bulk_write(self, collection_name, operations, ordered=True):
        """
        Perform a bulk write operation
        
        Args:
            collection_name (str): Name of the collection
            operations (list): List of write operations
            ordered (bool): Whether operations should be executed in order
            
        Returns:
            BulkWriteResult: Result of the bulk operation
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            return collection.bulk_write(operations, ordered=ordered)
        except Exception as e:
            log_error(e, context={"action": "bulk_write", "collection": collection_name})
            return None
    
    def find_by_id(self, collection_name, document_id):
        """
        Find a document by its ID
        
        Args:
            collection_name (str): Name of the collection
            document_id: Document ID
            
        Returns:
            dict: Found document or None
        """
        self._ensure_connected()
        
        collection = self.get_collection(collection_name)
        
        try:
            return collection.find_one({"_id": document_id})
        except Exception as e:
            log_error(e, context={"action": "find_by_id", "collection": collection_name})
            return None
    
    def close(self):
        """Close the MongoDB connection"""
        if self.client is not None:
            try:
                self.client.close()
            except Exception as e:
                self.logger.error(f"Error closing MongoDB connection: {e}")
            self.client = None
            self.db = None
            self.logger.info("MongoDB connection closed")
    
    def __del__(self):
        """Destructor to ensure connection is closed"""
        try:
            self.close()
        except (ImportError, AttributeError, TypeError) as e:
            # Ignore errors during interpreter shutdown
            pass


# Helper methods for specific collections

def get_market_data(self, symbol, exchange, timeframe, start_date=None, end_date=None):
    """
    Get market data for a specific symbol, exchange, and timeframe
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        timeframe (str): Timeframe (e.g. '1min', '5min', 'day')
        start_date (datetime, optional): Start date
        end_date (datetime, optional): End date
        
    Returns:
        list: List of market data documents
    """
    query = {
        "symbol": symbol,
        "exchange": exchange,
        "timeframe": timeframe
    }
    
    # Add date range if provided
    if start_date or end_date:
        query["timestamp"] = {}
        
        if start_date:
            query["timestamp"]["$gte"] = start_date
        
        if end_date:
            query["timestamp"]["$lte"] = end_date
    
    return self.find(
        collection_name="market_data",
        query=query,
        sort=[("timestamp", pymongo.ASCENDING)]
    )

def save_market_data(self, market_data):
    """
    Save market data to the database
    
    Args:
        market_data (list): List of market data documents
        
    Returns:
        int: Number of documents inserted
    """
    if not market_data:
        return 0
    
    # Create bulk operations
    operations = []
    collection = self.get_collection("market_data")
    
    for data in market_data:
        # Ensure required fields
        if not all(field in data for field in ["symbol", "exchange", "timeframe", "timestamp"]):
            continue
        
        # Create update operation
        operations.append(
            pymongo.UpdateOne(
                {
                    "symbol": data["symbol"],
                    "exchange": data["exchange"],
                    "timeframe": data["timeframe"],
                    "timestamp": data["timestamp"]
                },
                {"$set": data},
                upsert=True
            )
        )
    
    # Execute bulk operation
    if operations:
        try:
            result = collection.bulk_write(operations)
            return result.upserted_count + result.modified_count
        except Exception as e:
            log_error(e, context={"action": "save_market_data"})
            return 0
    
    return 0

def get_latest_price(self, symbol, exchange):
    """
    Get the latest price for a symbol
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        
    Returns:
        float: Latest price or None
    """
    result = self.find_one(
        collection_name="market_data",
        query={"symbol": symbol, "exchange": exchange},
        sort=[("timestamp", pymongo.DESCENDING)]
    )
    
    if result and "close" in result:
        return result["close"]
    
    return None

def save_financial_data(self, data):
    """
    Save financial data to the database
    
    Args:
        data (dict): Financial data document
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not data:
        return False
    
    # Ensure required fields
    if not all(field in data for field in ["symbol", "exchange", "report_type", "period"]):
        return False
    
    try:
        # Use upsert to insert or update
        result = self.update_one(
            collection_name="financial",
            query={
                "symbol": data["symbol"],
                "exchange": data["exchange"],
                "report_type": data["report_type"],
                "period": data["period"]
            },
            update={"$set": data},
            upsert=True
        )
        
        return result
    except Exception as e:
        log_error(e, context={"action": "save_financial_data"})
        return False

def get_financial_data(self, symbol, exchange, report_type=None):
    """
    Get financial data for a symbol
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        report_type (str, optional): Report type (e.g. 'quarterly', 'annual')
        
    Returns:
        list: List of financial data documents
    """
    query = {
        "symbol": symbol,
        "exchange": exchange
    }
    
    if report_type:
        query["report_type"] = report_type
    
    return self.find(
        collection_name="financial",
        query=query,
        sort=[("period", pymongo.DESCENDING)]
    )

def save_news(self, news_items):
    """
    Save news items to the database
    
    Args:
        news_items (list): List of news item documents
        
    Returns:
        int: Number of news items saved
    """
    if not news_items:
        return 0
    
    # Create bulk operations
    operations = []
    collection = self.get_collection("news")
    
    for news in news_items:
        # Skip if no title
        if "title" not in news:
            continue
        
        # Create update operation
        operations.append(
            pymongo.UpdateOne(
                {"title": news["title"]},
                {"$set": news},
                upsert=True
            )
        )
    
    # Execute bulk operation
    if operations:
        try:
            result = collection.bulk_write(operations)
            return result.upserted_count + result.modified_count
        except Exception as e:
            log_error(e, context={"action": "save_news"})
            return 0
    
    return 0

def get_news(self, symbol=None, start_date=None, end_date=None, limit=20):
    """
    Get news items
    
    Args:
        symbol (str, optional): Instrument symbol
        start_date (datetime, optional): Start date
        end_date (datetime, optional): End date
        limit (int, optional): Maximum number of items to return
        
    Returns:
        list: List of news items
    """
    query = {}
    
    if symbol:
        query["$or"] = [
            {"symbol": symbol},
            {"related_symbols": symbol}
        ]
    
    # Add date range if provided
    if start_date or end_date:
        query["published_date"] = {}
        
        if start_date:
            query["published_date"]["$gte"] = start_date
        
        if end_date:
            query["published_date"]["$lte"] = end_date
    
    return self.find(
        collection_name="news",
        query=query,
        sort=[("published_date", pymongo.DESCENDING)],
        limit=limit
    )

def save_prediction(self, prediction):
    """
    Save a prediction to the database
    
    Args:
        prediction (dict): Prediction document
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not prediction:
        return False
    
    # Ensure required fields
    if not all(field in prediction for field in ["symbol", "exchange", "date", "prediction"]):
        return False
    
    try:
        result = self.insert_one(
            collection_name="predictions",
            document=prediction
        )
        
        return result is not None
    except Exception as e:
        log_error(e, context={"action": "save_prediction"})
        return False

def get_predictions(self, symbol, exchange, start_date=None, end_date=None, limit=10):
    """
    Get predictions for a symbol
    
    Args:
        symbol (str): Instrument symbol
        exchange (str): Exchange code
        start_date (datetime, optional): Start date
        end_date (datetime, optional): End date
        limit (int, optional): Maximum number of items to return
        
    Returns:
        list: List of predictions
    """
    query = {
        "symbol": symbol,
        "exchange": exchange
    }
    
    # Add date range if provided
    if start_date or end_date:
        query["date"] = {}
        
        if start_date:
            query["date"]["$gte"] = start_date
        
        if end_date:
            query["date"]["$lte"] = end_date
    
    return self.find(
        collection_name="predictions",
        query=query,
        sort=[("date", pymongo.DESCENDING)],
        limit=limit
    )

def save_trade(self, trade):
    """
    Save a trade to the database
    
    Args:
        trade (dict): Trade document
        
    Returns:
        str: Trade ID if successful, None otherwise
    """
    if not trade:
        return None
    
    # Ensure required fields
    if not all(field in trade for field in ["symbol", "exchange", "trade_type", "entry_price", "entry_time", "quantity"]):
        return None
    
    try:
        result = self.insert_one(
            collection_name="trades",
            document=trade
        )
        
        return result
    except Exception as e:
        log_error(e, context={"action": "save_trade"})
        return None

def update_trade(self, trade_id, updates):
    """
    Update a trade
    
    Args:
        trade_id: Trade ID
        updates (dict): Updates to apply
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not trade_id or not updates:
        return False
    
    try:
        result = self.update_one(
            collection_name="trades",
            query={"_id": trade_id},
            update={"$set": updates}
        )
        
        return result
    except Exception as e:
        log_error(e, context={"action": "update_trade"})
        return False

def get_trades(self, symbol=None, start_date=None, end_date=None, status=None, limit=100):
    """
    Get trades
    
    Args:
        symbol (str, optional): Instrument symbol
        start_date (datetime, optional): Start date
        end_date (datetime, optional): End date
        status (str, optional): Trade status (open, closed)
        limit (int, optional): Maximum number of items to return
        
    Returns:
        list: List of trades
    """
    query = {}
    
    if symbol:
        query["symbol"] = symbol
    
    if status:
        query["status"] = status
    
    # Add date range if provided
    if start_date or end_date:
        query["entry_time"] = {}
        
        if start_date:
            query["entry_time"]["$gte"] = start_date
        
        if end_date:
            query["entry_time"]["$lte"] = end_date
    
    return self.find(
        collection_name="trades",
        query=query,
        sort=[("entry_time", pymongo.DESCENDING)],
        limit=limit
    )

def save_performance(self, performance):
    """
    Save performance metrics
    
    Args:
        performance (dict): Performance document
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not performance:
        return False
    
    # Ensure required fields
    if "date" not in performance:
        performance["date"] = datetime.now()
    
    try:
        result = self.insert_one(
            collection_name="performance",
            document=performance
        )
        
        return result is not None
    except Exception as e:
        log_error(e, context={"action": "save_performance"})
        return False

def get_performance(self, start_date=None, end_date=None):
    """
    Get performance metrics
    
    Args:
        start_date (datetime, optional): Start date
        end_date (datetime, optional): End date
        
    Returns:
        list: List of performance documents
    """
    query = {}
    
    # Add date range if provided
    if start_date or end_date:
        query["date"] = {}
        
        if start_date:
            query["date"]["$gte"] = start_date
        
        if end_date:
            query["date"]["$lte"] = end_date
    
    return self.find(
        collection_name="performance",
        query=query,
        sort=[("date", pymongo.ASCENDING)]
    )

def log_system_event(self, event_type, details=None):
   """
   Log a system event
   
   Args:
       event_type (str): Type of event
       details (dict, optional): Event details
       
   Returns:
       bool: True if successful, False otherwise
   """
   log_entry = {
       "timestamp": datetime.now(),
       "event_type": event_type,
       "details": details or {}
   }
   
   try:
       result = self.insert_one(
           collection_name="system_logs",
           document=log_entry
       )
       
       return result is not None
   except Exception as e:
       log_error(e, context={"action": "log_system_event"})
       return False

def get_system_logs(self, event_type=None, start_date=None, end_date=None, limit=100):
   """
   Get system logs
   
   Args:
       event_type (str, optional): Type of event
       start_date (datetime, optional): Start date
       end_date (datetime, optional): End date
       limit (int, optional): Maximum number of items to return
       
   Returns:
       list: List of log entries
   """
   query = {}
   
   if event_type:
       query["event_type"] = event_type
   
   # Add date range if provided
   if start_date or end_date:
       query["timestamp"] = {}
       
       if start_date:
           query["timestamp"]["$gte"] = start_date
       
       if end_date:
           query["timestamp"]["$lte"] = end_date
   
   return self.find(
       collection_name="system_logs",
       query=query,
       sort=[("timestamp", pymongo.DESCENDING)],
       limit=limit
   )