"""
Database connection manager for the Automated Trading System.
Ensures a singleton connection to MongoDB.
"""

import os
import sys
from typing import Optional
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .mongodb_connector import MongoDBConnector
from utils.logging_utils import setup_logger

class ConnectionManager:
    """
    MongoDB connection manager that ensures a singleton connection.
    Thread-safe implementation using a lock.
    """
    
    _instance: Optional[MongoDBConnector] = None
    _lock = threading.Lock()
    
    @classmethod
    def get_connection(cls) -> MongoDBConnector:
        """
        Get MongoDB connection
        
        Returns:
            MongoDBConnector: MongoDB connector instance
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check lock pattern
                if cls._instance is None:
                    cls._instance = MongoDBConnector()
        
        return cls._instance
    
    @classmethod
    def close_connection(cls) -> None:
        """Close MongoDB connection"""
        if cls._instance is not None:
            with cls._lock:
                if cls._instance is not None:
                    cls._instance.close()
                    cls._instance = None
    
    @classmethod
    def refresh_connection(cls) -> MongoDBConnector:
        """
        Refresh MongoDB connection
        
        Returns:
            MongoDBConnector: MongoDB connector instance
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance.close()
            
            cls._instance = MongoDBConnector()
        
        return cls._instance


# Create a global function for easy access
def get_db() -> MongoDBConnector:
    """
    Get MongoDB connection
    
    Returns:
        MongoDBConnector: MongoDB connector instance
    """
    return ConnectionManager.get_connection()