#!/usr/bin/env python3
"""
Data Check and Setup Script
Verifies your data is ready for model training
"""

import os
import sys
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.connection_manager import get_db
from utils.logging_utils import setup_logger
from portfolio.portfolio_manager import PortfolioManager

logger = setup_logger(__name__)

def check_data_availability():
    """Check what data is available for training."""
    logger.info("Checking data availability...")
    
    # Initialize database
    db = get_db()
    portfolio_manager = PortfolioManager(db)
    
    # Get active instruments
    instruments = portfolio_manager.get_active_instruments()
    logger.info(f"Found {len(instruments)} active instruments")
    
    data_summary = {}
    
    for instrument in instruments:
        symbol = instrument['symbol']
        exchange = instrument['exchange']
        
        summary = {
            'symbol': symbol,
            'exchange': exchange,
            'market_data': 0,
            'financial_data': 0,
            'news_data': 0,
            'date_range': None,
            'ready_for_training': False
        }
        
        # Check market data
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            market_data = db.get_market_data(
                symbol=symbol,
                exchange=exchange,
                timeframe='day',
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data:
                summary['market_data'] = len(market_data)
                dates = [d['timestamp'] for d in market_data if 'timestamp' in d]
                if dates:
                    summary['date_range'] = f"{min(dates)} to {max(dates)}"
        
        except Exception as e:
            logger.warning(f"Error checking market data for {symbol}: {e}")
        
        # Check financial data
        try:
            financial_data = db.get_financial_data(symbol, exchange)
            if financial_data:
                summary['financial_data'] = len(financial_data)
        
        except Exception as e:
            logger.warning(f"Error checking financial data for {symbol}: {e}")
        
        # Check news data
        try:
            news_data = db.get_news(symbol=symbol, limit=100)
            if news_data:
                summary['news_data'] = len(news_data)
        
        except Exception as e:
            logger.warning(f"Error checking news data for {symbol}: {e}")
        
        # Determine if ready for training
        summary['ready_for_training'] = summary['market_data'] >= 100
        
        data_summary[f"{symbol}_{exchange}"] = summary
    
    return data_summary

def print_data_summary(data_summary):
    """Print formatted data summary."""
    print("\n" + "="*80)
    print("DATA AVAILABILITY SUMMARY")
    print("="*80)
    
    ready_count = 0
    total_count = len(data_summary)
    
    for key, summary in data_summary.items():
        symbol = summary['symbol']
        exchange = summary['exchange']
        market_data = summary['market_data']
        financial_data = summary['financial_data']
        news_data = summary['news_data']
        ready = summary['ready_for_training']
        
        status = "✅ READY" if ready else "❌ NOT READY"
        if ready:
            ready_count += 1
        
        print(f"\n{symbol} {exchange} - {status}")
        print(f"  Market Data: {market_data} days")
        print(f"  Financial Data: {financial_data} reports")
        print(f"  News Data: {news_data} articles")
        
        if summary['date_range']:
            print(f"  Date Range: {summary['date_range']}")
        
        if not ready:
            print(f"  ⚠️  Need at least 100 days of market data (currently: {market_data})")
    
    print(f"\n📊 SUMMARY: {ready_count}/{total_count} instruments ready for training")
    
    return ready_count > 0

def setup_missing_collections():
    """Setup any missing database collections."""
    logger.info("Setting up missing database collections...")
    
    db = get_db()
    
    # Check and create collections if they don't exist
    collections_needed = [
        'models',
        'ensemble_models', 
        'predictions',
        'features',
        'performance_reports',
        'prediction_reports'
    ]
    
    existing_collections = db.list_collection_names()
    
    for collection_name in collections_needed:
        if collection_name not in existing_collections:
            try:
                db.db.create_collection(collection_name)
                logger.info(f"Created collection: {collection_name}")
            except Exception as e:
                logger.warning(f"Could not create collection {collection_name}: {e}")
    
    logger.info("Database setup completed")

def main():
    """Main function."""
    print("🔍 Checking your data for model training...")
    
    # Setup database collections
    setup_missing_collections()
    
    # Check data availability
    data_summary = check_data_availability()
    
    # Print summary
    has_ready_data = print_data_summary(data_summary)
    
    if has_ready_data:
        print("\n🎉 GOOD NEWS! You have data ready for training.")
        print("\n📋 NEXT STEPS:")
        print("1. Run quick training:")
        print("   python3 scripts/quick_train.py --symbol SBIN --exchange NSE")
        print("\n2. Or train comprehensive models:")
        print("   python3 scripts/train_models.py --symbol SBIN --exchange NSE")
        print("\n3. Or train for all ready instruments:")
        print("   python3 scripts/train_models.py --all")
        print("\n4. After training, generate predictions:")
        print("   python3 scripts/generate_predictions.py --all")
    
    else:
        print("\n⚠️  You need more data before training models.")
        print("\n📋 RECOMMENDATIONS:")
        print("1. Ensure you have at least 100 days of market data per symbol")
        print("2. Run your data collection scripts to gather more historical data")
        print("3. Check your database connections and data pipeline")

if __name__ == '__main__':
    main()