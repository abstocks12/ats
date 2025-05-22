#!/usr/bin/env python3
"""
Debug Stock Finder - Check what's in the portfolio collection
"""

import pymongo
import argparse

def debug_database(db_name="automated_trading"):
    """Debug the database to see what stocks are available"""
    
    try:
        client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        portfolio_collection = db.portfolio_collection
        
        print(f"Connected to database: {db_name}")
        
        # Check if collection exists
        collections = db.list_collection_names()
        print(f"Available collections: {collections}")
        
        if 'portfolio_collection' not in collections:
            print("ERROR: portfolio_collection not found!")
            return
        
        # Count total documents
        total_count = portfolio_collection.count_documents({})
        print(f"Total documents in portfolio_collection: {total_count}")
        
        if total_count == 0:
            print("ERROR: Portfolio collection is empty!")
            return
        
        # Show all symbols
        print("\nAll stocks in portfolio collection:")
        cursor = portfolio_collection.find({}, {"symbol": 1, "exchange": 1, "company_name": 1, "news_urls": 1})
        
        for doc in cursor:
            symbol = doc.get('symbol', 'N/A')
            exchange = doc.get('exchange', 'N/A')
            company_name = doc.get('company_name', 'N/A')
            has_news_urls = 'news_urls' in doc
            
            print(f"  - {symbol}:{exchange} ({company_name}) - News URLs: {has_news_urls}")
            
            if has_news_urls:
                print(f"    URLs: {doc['news_urls']}")
        
        client.close()
        
    except Exception as e:
        print(f"Database error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Debug Database Contents')
    parser.add_argument('--db', '-d', default='automated_trading', help='Database name')
    
    args = parser.parse_args()
    debug_database(args.db)

if __name__ == "__main__":
    main()