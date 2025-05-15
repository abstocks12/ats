#!/usr/bin/env python3
"""
Script to initialize the MongoDB database with collections and indexes.
"""

import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.mongodb_connector import MongoDBConnector
from database.connection_manager import get_db
from utils.logging_utils import setup_logger

logger = setup_logger("init_db")

def init_database(drop_existing=False):
    """
    Initialize the database
    
    Args:
        drop_existing (bool): Whether to drop existing collections
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get database connection
        db = get_db()
        
        # Drop existing collections if requested
        if drop_existing:
            logger.info("Dropping existing collections...")
            
            from config import settings
            
            for _, collection_name in settings.MONGODB_COLLECTIONS.items():
                db.db.drop_collection(collection_name)
                logger.info(f"Dropped collection: {collection_name}")
        
        # Initialize database
        logger.info("Initializing database...")
        result = db.initialize_database()
        
        if result:
            logger.info("Database initialized successfully")
            return True
        else:
            logger.error("Failed to initialize database")
            return False
            
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Initialize MongoDB database')
    
    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop existing collections'
    )
    
    args = parser.parse_args()
    
    # Initialize database
    result = init_database(drop_existing=args.drop)
    
    if result:
        print("Database initialized successfully")
    else:
        print("Failed to initialize database")
        sys.exit(1)

if __name__ == '__main__':
    main()