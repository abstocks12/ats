#!/usr/bin/env python3
"""
Test script for MongoDB connectivity.
"""

import os
import sys
import time
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb_connector import MongoDBConnector
from database.connection_manager import get_db
from database.models import MarketData, NewsItem, FinancialData, PredictionData, TradeData, PerformanceData
from utils.logging_utils import setup_logger

logger = setup_logger("test_mongodb")

def test_connection():
    """Test MongoDB connection"""
    try:
        # Get database connection
        db = get_db()
        
        # Test connection
        db.db.command('ping')
        
        logger.info("MongoDB connection successful")
        return True
    except Exception as e:
        logger.error(f"MongoDB connection failed: {e}")
        return False

def test_crud_operations():
    """Test CRUD operations"""
    try:
        # Get database connection
        db = get_db()
        
        # Test collection name
        collection_name = "test_collection"
        
        # Create test collection if it doesn't exist
        if collection_name not in db.db.list_collection_names():
            db.db.create_collection(collection_name)
        
        # Test insert
        test_doc = {
            "test_id": "test1",
            "name": "Test Document",
            "value": 42,
            "timestamp": datetime.now()
        }
        
        insert_result = db.insert_one(collection_name, test_doc)
        logger.info(f"Insert result: {insert_result}")
        
        # Test find
        find_result = db.find_one(collection_name, {"test_id": "test1"})
        logger.info(f"Find result: {find_result}")
        
        # Test update
        update_result = db.update_one(
            collection_name,
            {"test_id": "test1"},
            {"$set": {"value": 99}}
        )
        logger.info(f"Update result: {update_result}")
        
        # Test find after update
        find_after_update = db.find_one(collection_name, {"test_id": "test1"})
        logger.info(f"Find after update: {find_after_update}")
        
        # Test delete
        delete_result = db.delete_one(collection_name, {"test_id": "test1"})
        logger.info(f"Delete result: {delete_result}")
        
        # Test find after delete
        find_after_delete = db.find_one(collection_name, {"test_id": "test1"})
        logger.info(f"Find after delete: {find_after_delete}")
        
        # Drop test collection
        db.db.drop_collection(collection_name)
        
        logger.info("CRUD operations test successful")
        return True
    except Exception as e:
        logger.error(f"CRUD operations test failed: {e}")
        return False

def test_models():
    """Test database models"""
    try:
        # Get database connection
        db = get_db()
        
        # Test MarketData model
        market_data = MarketData(
            symbol="TATASTEEL",
            exchange="NSE",
            timeframe="1min",
            timestamp=datetime.now(),
            open_price=100.0,
            high_price=101.0,
            low_price=99.0,
            close_price=100.5,
            volume=1000,
            indicators={
                "sma_20": 98.5,
                "rsi_14": 55.0
            }
        )
        
        logger.info(f"MarketData model: {market_data}")
        logger.info(f"MarketData dict: {market_data.to_dict()}")
        
        # Test NewsItem model
        news_item = NewsItem(
            title="Test News",
            description="This is a test news item",
            source="Test Source",
            url="https://example.com/news/1",
            published_date=datetime.now(),
            sentiment="positive",
            sentiment_score=0.8,
            categories=["test", "news"],
            symbols=["TATASTEEL"]
        )
        
        logger.info(f"NewsItem model: {news_item}")
        logger.info(f"NewsItem dict: {news_item.to_dict()}")
        
        # Test FinancialData model
        financial_data = FinancialData(
            symbol="TATASTEEL",
            exchange="NSE",
            report_type="quarterly",
            period="Q1-2023",
            report_date=datetime.now(),
            data={
                "sales": 1000000,
                "profit": 100000,
                "eps": 10.5
            }
        )
        
        logger.info(f"FinancialData model: {financial_data}")
        logger.info(f"FinancialData dict: {financial_data.to_dict()}")
        
        # Test PredictionData model
        prediction_data = PredictionData(
            symbol="TATASTEEL",
            exchange="NSE",
            date=datetime.now(),
            prediction="up",
            confidence=0.75,
            timeframe="intraday",
            supporting_factors=[
                {"factor": "technical", "weight": 0.6},
                {"factor": "sentiment", "weight": 0.4}
            ],
            target_price=105.0,
            stop_loss=98.0,
            model_id="test_model"
        )
        
        logger.info(f"PredictionData model: {prediction_data}")
        logger.info(f"PredictionData dict: {prediction_data.to_dict()}")
        
        # Test TradeData model
        trade_data = TradeData(
            symbol="TATASTEEL",
            exchange="NSE",
            instrument_type="equity",
            trade_type="buy",
            entry_price=100.0,
            entry_time=datetime.now(),
            quantity=10,
            strategy="test_strategy",
            timeframe="intraday",
            initial_stop_loss=98.0,
            target_price=105.0
        )
        
        logger.info(f"TradeData model: {trade_data}")
        logger.info(f"TradeData dict: {trade_data.to_dict()}")
        
        # Close trade
        trade_data.close_trade(
            exit_price=104.0,
            exit_time=datetime.now()
        )
        
        logger.info(f"Closed TradeData model: {trade_data}")
        logger.info(f"Closed TradeData dict: {trade_data.to_dict()}")
        
        # Test PerformanceData model
        performance_data = PerformanceData(
            date=datetime.now(),
            portfolio_value=1100000.0,
            cash_balance=1100000.0,
            daily_pnl=10000.0,
            daily_pnl_percent=1.0,
            total_trades=10,
            winning_trades=7,
            metrics={
                "sharpe_ratio": 1.5,
                "max_drawdown": 5.0
            },
            positions=[
                {
                    "symbol": "TATASTEEL",
                    "quantity": 10,
                    "entry_price": 100.0,
                    "current_price": 104.0,
                    "position_value": 1040.0,
                    "position_pnl": 40.0,
                    "position_pnl_percent": 4.0
                }
            ]
        )
        
        logger.info(f"PerformanceData model: {performance_data}")
        logger.info(f"PerformanceData dict: {performance_data.to_dict()}")
        
        logger.info("Models test successful")
        return True
    except Exception as e:
        logger.error(f"Models test failed: {e}")
        return False

def main():
    """Main function"""
    print("Testing MongoDB connectivity...")
    
    # Test connection
    if not test_connection():
        print("Connection test failed")
        sys.exit(1)
    
    print("Connection test passed")
    
    # Test CRUD operations
    if not test_crud_operations():
        print("CRUD operations test failed")
        sys.exit(1)
    
    print("CRUD operations test passed")
    
    # Test models
    if not test_models():
        print("Models test failed")
        sys.exit(1)
    
    print("Models test passed")
    
    print("All tests passed successfully")

if __name__ == '__main__':
    main()