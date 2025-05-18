"""
Query Optimizer

This module provides optimized query patterns for common trading system operations.
It includes best practices for efficient MongoDB queries and aggregations.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta

class QueryOptimizer:
    """
    Provides optimized query patterns for common trading operations.
    Includes methods for market data retrieval, trading analytics, and time-series analysis.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the query optimizer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """Set up a logger for this module."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def get_market_data(self, symbol: str, timeframe: str = "day", days: int = 30, 
                      with_indicators: bool = True) -> Dict[str, Any]:
        """
        Optimized query for retrieving market data.
        
        Args:
            symbol: Stock symbol
            timeframe: Data timeframe
            days: Number of days to retrieve
            with_indicators: Whether to include technical indicators
            
        Returns:
            Dictionary with query results
        """
        try:
            # Calculate start date
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
            
            # Determine which collection to query (base or partitioned)
            # This is a simplified approach - production would use more sophisticated routing
            collection = self.db.market_data_collection
            
            # Prepare query
            query = {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": {"$gte": start_date}
            }
            
            # Prepare projection
            projection = {
                "_id": 0,
                "symbol": 1,
                "timestamp": 1,
                "open": 1, 
                "high": 1, 
                "low": 1, 
                "close": 1, 
                "volume": 1
            }
            
            # Add indicators if requested
            if with_indicators:
                projection["indicators"] = 1
            
            # Execute query with proper index hint
            cursor = collection.find(
                query,
                projection
            ).sort("timestamp", 1).hint([("symbol", 1), ("timeframe", 1), ("timestamp", -1)])
            
            # Convert to list
            data = list(cursor)
            
            return {
                "status": "success",
                "count": len(data),
                "data": data
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving market data: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_ohlcv_aggregated(self, symbol: str, timeframe: str = "day", 
                           to_timeframe: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """
        Get OHLCV data aggregated to a different timeframe.
        
        Args:
            symbol: Stock symbol
            timeframe: Source data timeframe
            to_timeframe: Target aggregation timeframe (None for no aggregation)
            days: Number of days to retrieve
            
        Returns:
            Dictionary with aggregated data
        """
        try:
            # Calculate start date
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
            
            # If no aggregation needed, use standard query
            if not to_timeframe:
                return self.get_market_data(symbol, timeframe, days, False)
            
            # Determine aggregation timeframe parameters
            group_by_format = None
            if to_timeframe == "hour" and timeframe in ["1min", "5min", "15min"]:
                group_by_format = "%Y-%m-%d %H:00:00"
            elif to_timeframe == "day" and timeframe in ["1min", "5min", "15min", "hour"]:
                group_by_format = "%Y-%m-%d"
            elif to_timeframe == "week" and timeframe in ["day", "hour"]:
                group_by_format = "%Y-%U"  # ISO week number
            elif to_timeframe == "month" and timeframe in ["day", "week"]:
                group_by_format = "%Y-%m"
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported aggregation from {timeframe} to {to_timeframe}"
                }
            
            # Define the aggregation pipeline
            pipeline = [
                # Match stage - filter by symbol, timeframe and date range
                {
                    "$match": {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "timestamp": {"$gte": start_date}
                    }
                },
                # Add formatted date field for grouping
                {
                    "$addFields": {
                        "date_group": {
                            "$dateToString": {
                                "format": group_by_format, 
                                "date": "$timestamp"
                            }
                        }
                    }
                },
                # Group by the formatted date
                {
                    "$group": {
                        "_id": "$date_group",
                        "timestamp_first": {"$first": "$timestamp"},
                        "open": {"$first": "$open"},
                        "high": {"$max": "$high"},
                        "low": {"$min": "$low"},
                        "close": {"$last": "$close"},
                        "volume": {"$sum": "$volume"}
                    }
                },
                # Rename fields
                {
                    "$project": {
                        "_id": 0,
                        "timestamp": "$timestamp_first",
                        "open": 1,
                        "high": 1,
                        "low": 1,
                        "close": 1,
                        "volume": 1
                    }
                },
                # Sort by timestamp
                {
                    "$sort": {"timestamp": 1}
                }
            ]
            
            # Execute the aggregation
            results = list(self.db.market_data_collection.aggregate(pipeline))
            
            return {
                "status": "success",
                "source_timeframe": timeframe,
                "target_timeframe": to_timeframe,
                "count": len(results),
                "data": results
            }
            
        except Exception as e:
            self.logger.error(f"Error aggregating OHLCV data: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_performance_stats(self, days: int = 30, by_strategy: bool = False) -> Dict[str, Any]:
        """
        Get trading performance statistics with optimized queries.
        
        Args:
            days: Number of days to analyze
            by_strategy: Whether to group results by strategy
            
        Returns:
            Dictionary with performance statistics
        """
        try:
            # Calculate start date
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
            
            # Base match condition
            match_stage = {
                "$match": {
                    "entry_time": {"$gte": start_date},
                    "exit_time": {"$ne": None}  # Only consider closed trades
                }
            }
            
            # Define the aggregation pipeline
            pipeline = [match_stage]
            
            # Group stage depends on whether to group by strategy
            if by_strategy:
                group_stage = {
                    "$group": {
                        "_id": "$strategy",
                        "total_trades": {"$sum": 1},
                        "winning_trades": {
                            "$sum": {"$cond": [{"$gt": ["$profit_loss", 0]}, 1, 0]}
                        },
                        "total_profit_loss": {"$sum": "$profit_loss"},
                        "avg_profit_loss_percent": {"$avg": "$profit_loss_percent"},
                        "max_profit_percent": {"$max": "$profit_loss_percent"},
                        "max_loss_percent": {"$min": "$profit_loss_percent"}
                    }
                }
            else:
                group_stage = {
                    "$group": {
                        "_id": None,
                        "total_trades": {"$sum": 1},
                        "winning_trades": {
                            "$sum": {"$cond": [{"$gt": ["$profit_loss", 0]}, 1, 0]}
                        },
                        "total_profit_loss": {"$sum": "$profit_loss"},
                        "avg_profit_loss_percent": {"$avg": "$profit_loss_percent"},
                        "max_profit_percent": {"$max": "$profit_loss_percent"},
                        "max_loss_percent": {"$min": "$profit_loss_percent"}
                    }
                }
            
            pipeline.append(group_stage)
            
            # Add projection stage
            projection_stage = {
                "$project": {
                    "_id": 0,
                    "strategy": "$_id",
                    "total_trades": 1,
                    "winning_trades": 1,
                    "losing_trades": {"$subtract": ["$total_trades", "$winning_trades"]},
                    "win_rate": {"$divide": ["$winning_trades", "$total_trades"]},
                    "total_profit_loss": 1,
                    "avg_profit_loss_percent": 1,
                    "max_profit_percent": 1,
                    "max_loss_percent": 1
                }
            }
            
            pipeline.append(projection_stage)
            
            # Sort by total profit if grouping by strategy
            if by_strategy:
                pipeline.append({"$sort": {"total_profit_loss": -1}})
            
            # Execute the aggregation
            results = list(self.db.trades_collection.aggregate(pipeline))
            
            # For overall stats, unwrap the single result
            if not by_strategy and results:
                results = results[0]
                
            return {
                "status": "success",
                "period_days": days,
                "by_strategy": by_strategy,
                "stats": results
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving performance stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_correlated_symbols(self, symbol: str, timeframe: str = "day", 
                           days: int = 90, min_correlation: float = 0.7) -> Dict[str, Any]:
        """
        Find symbols with price correlation to the target symbol.
        
        Args:
            symbol: Target stock symbol
            timeframe: Data timeframe
            days: Number of days for correlation calculation
            min_correlation: Minimum correlation coefficient
            
        Returns:
            Dictionary with correlated symbols
        """
        try:
            # Get target symbol data first (this is more efficient than joining in MongoDB)
            target_data = self.get_market_data(symbol, timeframe, days, False)
            
            if target_data["status"] != "success":
                return target_data
            
            # Convert to a more usable format
            import pandas as pd
            target_df = pd.DataFrame(target_data["data"])
            
            if len(target_df) < 20:  # Need enough data points for meaningful correlation
                return {
                    "status": "error",
                    "error": f"Insufficient data points for {symbol}: {len(target_df)} found, minimum 20 required"
                }
            
            # Get list of all symbols in portfolio
            portfolio_symbols = list(self.db.portfolio_collection.find(
                {"status": "active"},
                {"_id": 0, "symbol": 1, "exchange": 1}
            ))
            
            # Calculate correlations
            correlations = []
            
            for portfolio_item in portfolio_symbols:
                portfolio_symbol = portfolio_item["symbol"]
                
                # Skip the target symbol itself
                if portfolio_symbol == symbol:
                    continue
                
                # Get data for this symbol
                symbol_data = self.get_market_data(portfolio_symbol, timeframe, days, False)
                
                if symbol_data["status"] != "success" or len(symbol_data["data"]) < 20:
                    continue
                
                # Convert to DataFrame
                symbol_df = pd.DataFrame(symbol_data["data"])
                
                # Align dates
                merged = pd.merge(
                    target_df[["timestamp", "close"]], 
                    symbol_df[["timestamp", "close"]], 
                    on="timestamp", 
                    suffixes=("_target", "_symbol")
                )
                
                if len(merged) < 20:
                    continue
                
                # Calculate correlation
                correlation = merged["close_target"].corr(merged["close_symbol"])
                
                if abs(correlation) >= min_correlation:
                    correlations.append({
                        "symbol": portfolio_symbol,
                        "exchange": portfolio_item["exchange"],
                        "correlation": correlation,
                        "data_points": len(merged)
                    })
            
            # Sort by absolute correlation (highest first)
            correlations = sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)
            
            return {
                "status": "success",
                "target_symbol": symbol,
                "timeframe": timeframe,
                "period_days": days,
                "min_correlation": min_correlation,
                "correlations": correlations
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_sentiment_timeline(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get sentiment analysis timeline for a symbol with efficient querying.
        
        Args:
            symbol: Stock symbol
            days: Number of days to analyze
            
        Returns:
            Dictionary with sentiment timeline
        """
        try:
            # Calculate start date
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
            
            # Define the aggregation pipeline
            pipeline = [
                # Match stage - filter by symbol and date range
                {
                    "$match": {
                        "entities": symbol,  # This assumes news is tagged with symbol names
                        "published_date": {"$gte": start_date}
                    }
                },
                # Group by day
                {
                    "$group": {
                        "_id": {
                            "$dateToString": {
                                "format": "%Y-%m-%d", 
                                "date": "$published_date"
                            }
                        },
                        "avg_sentiment": {"$avg": "$sentiment_score"},
                        "news_count": {"$sum": 1},
                        "positive_count": {
                            "$sum": {"$cond": [{"$gte": ["$sentiment_score", 0.6]}, 1, 0]}
                        },
                        "negative_count": {
                            "$sum": {"$cond": [{"$lte": ["$sentiment_score", 0.4]}, 1, 0]}
                        },
                        "neutral_count": {
                            "$sum": {
                                "$cond": [
                                    {"$and": [
                                        {"$gt": ["$sentiment_score", 0.4]},
                                        {"$lt": ["$sentiment_score", 0.6]}
                                    ]}, 
                                    1, 
                                    0
                                ]
                            }
                        },
                        "news_ids": {"$push": "$_id"}  # References to original news items
                    }
                },
                # Rename and format fields
                {
                    "$project": {
                        "_id": 0,
                        "date": "$_id",
                        "avg_sentiment": 1,
                        "news_count": 1,
                        "positive_count": 1,
                        "negative_count": 1,
                        "neutral_count": 1,
                        "sentiment_ratio": {
                            "$cond": [
                                {"$eq": ["$negative_count", 0]},
                                "$positive_count",  # Avoid division by zero
                                {"$divide": ["$positive_count", "$negative_count"]}
                            ]
                        }
                    }
                },
                # Sort by date
                {
                    "$sort": {"date": 1}
                }
            ]
            
            # Execute the aggregation
            results = list(self.db.news_collection.aggregate(pipeline))
            
            return {
                "status": "success",
                "symbol": symbol,
                "period_days": days,
                "days_with_news": len(results),
                "timeline": results
            }
            
        except Exception as e:
            self.logger.error(f"Error retrieving sentiment timeline: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_prediction_accuracy(self, days: int = 30, by_symbol: bool = False) -> Dict[str, Any]:
        """
        Calculate prediction accuracy with efficient querying.
        
        Args:
            days: Number of days to analyze
            by_symbol: Whether to group results by symbol
            
        Returns:
            Dictionary with prediction accuracy metrics
        """
        try:
            # Calculate start date
            start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
            
            # Get predictions
            pipeline = [
                # Match stage - filter by date range and only include predictions with validation
                {
                    "$match": {
                        "date": {"$gte": start_date},
                        "validated": True  # Only include predictions that have been validated
                    }
                }
            ]
            
            # Group stage depends on whether to group by symbol
            if by_symbol:
                group_stage = {
                    "$group": {
                        "_id": "$symbol",
                        "total_predictions": {"$sum": 1},
                        "correct_predictions": {
                            "$sum": {"$cond": [{"$eq": ["$actual_outcome", "$prediction"]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$confidence"},
                        "avg_expected_change": {"$avg": "$expected_change_percent"},
                        "avg_actual_change": {"$avg": "$actual_change_percent"}
                    }
                }
            else:
                group_stage = {
                    "$group": {
                        "_id": None,
                        "total_predictions": {"$sum": 1},
                        "correct_predictions": {
                            "$sum": {"$cond": [{"$eq": ["$actual_outcome", "$prediction"]}, 1, 0]}
                        },
                        "avg_confidence": {"$avg": "$confidence"},
                        "avg_expected_change": {"$avg": "$expected_change_percent"},
                        "avg_actual_change": {"$avg": "$actual_change_percent"}
                    }
                }
            
            pipeline.append(group_stage)
            
            # Add projection stage
            projection_stage = {
                "$project": {
                    "_id": 0,
                    "symbol": "$_id",
                    "total_predictions": 1,
                    "correct_predictions": 1,
                    "accuracy": {"$divide": ["$correct_predictions", "$total_predictions"]},
                    "avg_confidence": 1,
                    "confidence_accuracy_ratio": {
                        "$divide": [
                            {"$divide": ["$correct_predictions", "$total_predictions"]},
                            "$avg_confidence"
                        ]
                    },
                    "avg_expected_change": 1,
                    "avg_actual_change": 1,
                    "change_prediction_ratio": {
                        "$cond": [
                            {"$eq": ["$avg_expected_change", 0]},
                            0,
                            {"$divide": ["$avg_actual_change", "$avg_expected_change"]}
                        ]
                    }
                }
            }
            
            pipeline.append(projection_stage)
            
            # Sort by accuracy if grouping by symbol
            if by_symbol:
                pipeline.append({"$sort": {"accuracy": -1}})
            
            # Execute the aggregation
            results = list(self.db.predictions_collection.aggregate(pipeline))
            
            # For overall stats, unwrap the single result
            if not by_symbol and results:
                results = results[0]
                
            return {
                "status": "success",
                "period_days": days,
                "by_symbol": by_symbol,
                "accuracy_stats": results
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction accuracy: {e}")
            return {
                "status": "error",
                "error": str(e)
            }