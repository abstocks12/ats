# Create file: database/connection_manager.py
"""
Database Connection Manager for the Automated Trading System.
Provides a single point of access to the database connection.
"""

import os
from database.mongodb_connector import MongoDBConnector
from utils.logging_utils import setup_logger
from config import settings

# Global database connection
_db_connection = None

def get_db():
    """
    Get the database connection
    
    Returns:
        MongoDBConnector: Database connection
    """
    global _db_connection
    
    # Create connection if it doesn't exist
    if _db_connection is None:
        logger = setup_logger("database.connection_manager")
        
        try:
            _db_connection = MongoDBConnector(
                uri=os.getenv('MONGODB_URI', settings.MONGO_URI),
                db_name=os.getenv('DB_NAME', settings.MONGO_DB_NAME),
                username=os.getenv('MONGO_USERNAME', settings.MONGO_USERNAME),
                password=os.getenv('MONGO_PASSWORD', settings.MONGO_PASSWORD)
            )
        except Exception as e:
            logger.error(f"Failed to create database connection: {e}")
            raise
    
    return _db_connection

def close_db_connection():
    """Close the database connection"""
    global _db_connection
    
    if _db_connection is not None:
        _db_connection.close()
        _db_connection = None