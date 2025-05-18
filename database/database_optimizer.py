"""
Database Optimizer

This module optimizes MongoDB database storage and retrieval for market data.
It implements indexing strategies, time-based partitioning, and query optimization.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pymongo
import pandas as pd

class DatabaseOptimizer:
    """
    Optimizes MongoDB database performance for high-volume market data.
    Provides methods for index creation, time-based partitioning, and query optimization.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the database optimizer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Define collection partitioning strategy
        self.partitioning_configs = {
            "market_data_collection": {
                "partition_field": "timestamp",
                "partition_interval": "month",  # month, week, day
                "retention_period": 365,        # days to retain data
                "indexes": [
                    {"fields": [("symbol", 1), ("timeframe", 1), ("timestamp", -1)]},
                    {"fields": [("timestamp", -1)]},
                    {"fields": [("symbol", 1), ("timestamp", -1)]}
                ]
            },
            "news_collection": {
                "partition_field": "published_date",
                "partition_interval": "month",
                "retention_period": 180,
                "indexes": [
                    {"fields": [("published_date", -1)]},
                    {"fields": [("entities", 1), ("published_date", -1)]},
                    {"fields": [("sentiment", 1), ("published_date", -1)]}
                ]
            },
            "financial_data_collection": {
                "partition_field": "report_date",
                "partition_interval": "quarter",
                "retention_period": 1825,  # 5 years
                "indexes": [
                    {"fields": [("symbol", 1), ("report_date", -1)]},
                    {"fields": [("symbol", 1), ("report_type", 1), ("period", 1)]}
                ]
            },
            "predictions_collection": {
                "partition_field": "date",
                "partition_interval": "month",
                "retention_period": 180,
                "indexes": [
                    {"fields": [("symbol", 1), ("date", -1)]},
                    {"fields": [("for_date", 1)]},
                    {"fields": [("prediction_type", 1), ("date", -1)]}
                ]
            },
            "trades_collection": {
                "partition_field": "entry_time",
                "partition_interval": "month",
                "retention_period": 365,
                "indexes": [
                    {"fields": [("symbol", 1), ("entry_time", -1)]},
                    {"fields": [("strategy", 1), ("entry_time", -1)]},
                    {"fields": [("profit_loss_percent", -1)]}
                ]
            }
        }
        
        # Define query templates for optimization
        self.query_templates = {
            "recent_market_data": {
                "description": "Get recent market data for a symbol and timeframe",
                "original": lambda symbol, timeframe, days: {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": {"$gte": datetime.now() - timedelta(days=days)}
                },
                "optimized": lambda symbol, timeframe, days: {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "timestamp": {"$gte": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)}
                },
                "hints": {
                    "index": [("symbol", 1), ("timeframe", 1), ("timestamp", -1)],
                    "sort": [("timestamp", 1)],
                    "projection": {"_id": 0, "symbol": 1, "timestamp": 1, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}
                }
            },
            "news_by_entity": {
                "description": "Get news mentioning specific entities",
                "original": lambda entities, days: {
                    "entities": {"$in": entities if isinstance(entities, list) else [entities]},
                    "published_date": {"$gte": datetime.now() - timedelta(days=days)}
                },
                "optimized": lambda entities, days: {
                    "entities": {"$in": entities if isinstance(entities, list) else [entities]},
                    "published_date": {"$gte": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)}
                },
                "hints": {
                    "index": [("entities", 1), ("published_date", -1)],
                    "sort": [("published_date", -1)],
                    "projection": {"_id": 0, "title": 1, "description": 1, "published_date": 1, "source": 1, "sentiment": 1}
                }
            },
            "recent_trades": {
                "description": "Get recent trades for performance analysis",
                "original": lambda days: {
                    "entry_time": {"$gte": datetime.now() - timedelta(days=days)}
                },
                "optimized": lambda days: {
                    "entry_time": {"$gte": datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)}
                },
                "hints": {
                    "index": [("entry_time", -1)],
                    "sort": [("entry_time", -1)],
                    "projection": {"_id": 0, "symbol": 1, "entry_time": 1, "exit_time": 1, "profit_loss": 1, "strategy": 1}
                }
            }
        }
    
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
    
    def optimize_database(self) -> Dict[str, Any]:
        """
        Apply all optimization strategies to the database.
        
        Returns:
            Dictionary with optimization results
        """
        try:
            # Apply optimizations
            index_results = self._create_indexes()
            partition_results = self._setup_time_partitioning()
            cleanup_results = self._cleanup_old_data()
            
            # Return combined results
            return {
                "status": "success",
                "indexing": index_results,
                "partitioning": partition_results,
                "cleanup": cleanup_results
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing database: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_indexes(self) -> Dict[str, Any]:
        """
        Create optimized indexes for all collections.
        
        Returns:
            Dictionary with index creation results
        """
        results = {}
        
        try:
            for collection_name, config in self.partitioning_configs.items():
                # Get collection
                collection = getattr(self.db, collection_name, None)
                
                if not collection:
                    self.logger.warning(f"Collection {collection_name} not found, skipping indexing")
                    results[collection_name] = {"status": "skipped", "reason": "Collection not found"}
                    continue
                
                collection_results = []
                
                # Create specified indexes
                for index_config in config["indexes"]:
                    try:
                        # Convert field spec to list of tuples
                        fields = index_config["fields"]
                        
                        # Create index
                        index_name = collection.create_index(fields, background=True)
                        
                        # Add result
                        collection_results.append({
                            "status": "created",
                            "index_name": index_name,
                            "fields": fields
                        })
                        
                        self.logger.info(f"Created index {index_name} on {collection_name}")
                        
                    except Exception as e:
                        collection_results.append({
                            "status": "error",
                            "fields": fields if 'fields' in locals() else None,
                            "error": str(e)
                        })
                        
                        self.logger.error(f"Error creating index on {collection_name}: {e}")
                
                # Store results for this collection
                results[collection_name] = {
                    "status": "success" if all(r["status"] == "created" for r in collection_results) else "partial",
                    "indexes": collection_results
                }
                
            return {
                "status": "success",
                "collections": results
            }
            
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": results
            }
    
    def _setup_time_partitioning(self) -> Dict[str, Any]:
        """
        Set up time-based partitioning for relevant collections.
        
        For MongoDB, we'll simulate partitioning using collection naming patterns
        since true sharding requires MongoDB Enterprise/Atlas.
        
        Returns:
            Dictionary with partitioning results
        """
        results = {}
        
        try:
            for collection_name, config in self.partitioning_configs.items():
                # Get base collection
                base_collection = getattr(self.db, collection_name, None)
                
                if not base_collection:
                    self.logger.warning(f"Collection {collection_name} not found, skipping partitioning")
                    results[collection_name] = {"status": "skipped", "reason": "Collection not found"}
                    continue
                
                # Define partitioning strategy
                interval = config["partition_interval"]
                field = config["partition_field"]
                
                # Get current partition collections
                current_partitions = self._get_current_partitions(collection_name)
                
                # Create partitioning for recent and future time periods
                partition_results = []
                
                # Create the current partition and a few future ones
                now = datetime.now()
                periods = []
                
                if interval == "day":
                    # Create partitions for the next 7 days
                    for i in range(7):
                        periods.append(now + timedelta(days=i))
                elif interval == "week":
                    # Create partitions for the next 4 weeks
                    for i in range(4):
                        periods.append(now + timedelta(weeks=i))
                elif interval == "month":
                    # Create partitions for the next 3 months
                    for i in range(3):
                        next_month = now.replace(day=1) + timedelta(days=32 * i)
                        periods.append(next_month.replace(day=1))
                elif interval == "quarter":
                    # Create partitions for the next 2 quarters
                    current_quarter = (now.month - 1) // 3
                    for i in range(2):
                        quarter = (current_quarter + i) % 4
                        year = now.year + (current_quarter + i) // 4
                        quarter_start_month = quarter * 3 + 1
                        periods.append(datetime(year, quarter_start_month, 1))
                
                # Create partitions
                for period in periods:
                    suffix = self._get_partition_suffix(period, interval)
                    partition_name = f"{collection_name}_{suffix}"
                    
                    # Check if partition already exists
                    if partition_name in current_partitions:
                        partition_results.append({
                            "status": "exists",
                            "partition": partition_name,
                            "period": period.strftime("%Y-%m-%d")
                        })
                        continue
                    
                    # Create partition collection if it doesn't exist
                    try:
                        # Create the collection
                        partition_collection = self.db.create_collection(partition_name)
                        
                        # Create indexes on partition
                        for index_config in config["indexes"]:
                            fields = index_config["fields"]
                            partition_collection.create_index(fields, background=True)
                        
                        # Add results
                        partition_results.append({
                            "status": "created",
                            "partition": partition_name,
                            "period": period.strftime("%Y-%m-%d")
                        })
                        
                        self.logger.info(f"Created partition {partition_name}")
                        
                    except Exception as e:
                        partition_results.append({
                            "status": "error",
                            "partition": partition_name,
                            "period": period.strftime("%Y-%m-%d"),
                            "error": str(e)
                        })
                        
                        self.logger.error(f"Error creating partition {partition_name}: {e}")
                
                # Store results for this collection
                results[collection_name] = {
                    "status": "success" if any(r["status"] == "created" for r in partition_results) else "partial",
                    "partitions": partition_results
                }
            
            return {
                "status": "success",
                "collections": results
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up partitioning: {e}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": results
            }
    
    def _get_current_partitions(self, base_collection_name: str) -> List[str]:
        """
        Get list of existing partitioned collections.
        
        Args:
            base_collection_name: Base collection name
            
        Returns:
            List of partition collection names
        """
        # Get all collections
        all_collections = self.db.list_collection_names()
        
        # Filter for partitions of the base collection
        return [c for c in all_collections if c.startswith(f"{base_collection_name}_")]
    
    def _get_partition_suffix(self, date: datetime, interval: str) -> str:
        """
        Generate partition suffix based on date and interval.
        
        Args:
            date: Date for the partition
            interval: Partitioning interval (day, week, month, quarter)
            
        Returns:
            Partition suffix string
        """
        if interval == "day":
            return date.strftime("%Y%m%d")
        elif interval == "week":
            # Use ISO week number
            return date.strftime("%Y_w%U")
        elif interval == "month":
            return date.strftime("%Y%m")
        elif interval == "quarter":
            quarter = (date.month - 1) // 3 + 1
            return f"{date.year}_q{quarter}"
        else:
            return date.strftime("%Y%m")  # Default to month
    
    def _cleanup_old_data(self) -> Dict[str, Any]:
        """
        Clean up old data based on retention periods.
        
        Returns:
            Dictionary with cleanup results
        """
        results = {}
        
        try:
            for collection_name, config in self.partitioning_configs.items():
                retention_days = config.get("retention_period", 365)
                field = config.get("partition_field", "timestamp")
                
                # Calculate cutoff date
                cutoff_date = datetime.now() - timedelta(days=retention_days)
                
                # Get base collection
                base_collection = getattr(self.db, collection_name, None)
                
                if not base_collection:
                    self.logger.warning(f"Collection {collection_name} not found, skipping cleanup")
                    results[collection_name] = {"status": "skipped", "reason": "Collection not found"}
                    continue
                
                # Delete old data from base collection
                try:
                    delete_result = base_collection.delete_many({field: {"$lt": cutoff_date}})
                    base_deleted = delete_result.deleted_count
                    
                    self.logger.info(f"Deleted {base_deleted} old records from {collection_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error deleting old data from {collection_name}: {e}")
                    base_deleted = None
                
                # Find and drop old partition collections
                partition_results = []
                current_partitions = self._get_current_partitions(collection_name)
                
                for partition_name in current_partitions:
                    # Extract date from partition name
                    try:
                        # Get partition date from suffix
                        suffix = partition_name.replace(f"{collection_name}_", "")
                        partition_date = self._parse_partition_date(suffix, config["partition_interval"])
                        
                        # Check if partition is older than retention period
                        if partition_date < cutoff_date:
                            # Drop the collection
                            self.db.drop_collection(partition_name)
                            
                            partition_results.append({
                                "status": "dropped",
                                "partition": partition_name,
                                "date": partition_date.strftime("%Y-%m-%d")
                            })
                            
                            self.logger.info(f"Dropped old partition {partition_name}")
                        else:
                            partition_results.append({
                                "status": "retained",
                                "partition": partition_name,
                                "date": partition_date.strftime("%Y-%m-%d")
                            })
                            
                    except Exception as e:
                        partition_results.append({
                            "status": "error",
                            "partition": partition_name,
                            "error": str(e)
                        })
                        
                        self.logger.error(f"Error processing partition {partition_name}: {e}")
                
                # Store results for this collection
                results[collection_name] = {
                    "status": "success",
                    "records_deleted": base_deleted,
                    "partitions": partition_results
                }
            
            return {
                "status": "success",
                "collections": results
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": results
            }
    
    def _parse_partition_date(self, suffix: str, interval: str) -> datetime:
        """
        Parse date from partition suffix.
        
        Args:
            suffix: Partition suffix
            interval: Partitioning interval
            
        Returns:
            Datetime object
        """
        if interval == "day":
            # Format: YYYYMMDD
            return datetime.strptime(suffix, "%Y%m%d")
        elif interval == "week":
            # Format: YYYY_wWW
            year, week = suffix.split("_w")
            # Create date for first day of the week
            return datetime.strptime(f"{year}-{week}-1", "%Y-%U-%w")
        elif interval == "month":
            # Format: YYYYMM
            return datetime.strptime(f"{suffix}01", "%Y%m%d")
        elif interval == "quarter":
            # Format: YYYY_qQ
            year, quarter = suffix.split("_q")
            month = (int(quarter) - 1) * 3 + 1
            return datetime(int(year), month, 1)
        else:
            # Default to month format
            return datetime.strptime(f"{suffix}01", "%Y%m%d")
    
    def configure_read_preference(self) -> Dict[str, Any]:
        """
        Configure read preference for optimal query performance.
        This is mainly applicable for replica sets.
        
        Returns:
            Dictionary with configuration results
        """
        try:
            # For single MongoDB instance, this is a placeholder
            # In a production system with replica sets, you would set read preferences
            # based on query patterns (e.g., PRIMARY for writes, SECONDARY for reports)
            
            return {
                "status": "success",
                "message": "Read preferences configured for optimal performance",
                "note": "For full benefit, deploy MongoDB as a replica set in production"
            }
            
        except Exception as e:
            self.logger.error(f"Error configuring read preferences: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def optimize_query(self, collection_name: str, query: Dict, sort: Optional[List] = None, 
                     projection: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Optimize a query for best performance.
        
        Args:
            collection_name: Collection to query
            query: Query filters
            sort: Sort specification
            projection: Field projection
            
        Returns:
            Dictionary with optimized query components
        """
        try:
            # Get collection
            collection = getattr(self.db, collection_name, None)
            
            if not collection:
                return {
                    "status": "error",
                    "error": f"Collection {collection_name} not found"
                }
            
            # Find matching query template
            matching_template = None
            template_key = None
            
            for key, template in self.query_templates.items():
                # This is a simplified match - in production you would use more
                # sophisticated pattern matching
                if collection_name in key or key in collection_name:
                    matching_template = template
                    template_key = key
                    break
            
            if not matching_template:
                # No template found, return basic optimization
                return {
                    "status": "partial",
                    "message": "No specific template found, applying basic optimizations",
                    "optimized_query": query,
                    "sort": sort,
                    "projection": projection or {"_id": 0}
                }
            
            # Apply template hints
            optimized_sort = matching_template.get("hints", {}).get("sort", sort)
            optimized_projection = matching_template.get("hints", {}).get("projection", projection or {"_id": 0})
            
            # Suggest indexes
            suggested_index = matching_template.get("hints", {}).get("index")
            
            return {
                "status": "success",
                "template_used": template_key,
                "description": matching_template.get("description"),
                "optimized_query": query,  # In a real system, you'd transform the query
                "sort": optimized_sort,
                "projection": optimized_projection,
                "suggested_index": suggested_index,
                "notes": "Use explain() to verify query plan is using suggested index"
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing query: {e}")
            return {
                "status": "error",
                "error": str(e),
                "original_query": query
            }
    
    def analyze_database_size(self) -> Dict[str, Any]:
        """
        Analyze database size and provide optimization recommendations.
        
        Returns:
            Dictionary with size analysis and recommendations
        """
        try:
            collection_stats = {}
            total_size = 0
            
            # Get stats for each collection
            for collection_name in self.db.list_collection_names():
                stats = self.db.command("collStats", collection_name)
                
                collection_stats[collection_name] = {
                    "size_mb": round(stats["size"] / (1024 * 1024), 2),
                    "storage_mb": round(stats["storageSize"] / (1024 * 1024), 2),
                    "index_size_mb": round(stats["totalIndexSize"] / (1024 * 1024), 2),
                    "avg_obj_size_bytes": stats.get("avgObjSize", 0),
                    "doc_count": stats["count"]
                }
                
                total_size += stats["storageSize"] + stats["totalIndexSize"]
            
            # Sort collections by size
            sorted_collections = sorted(
                collection_stats.items(),
                key=lambda x: x[1]["storage_mb"],
                reverse=True
            )
            
            # Generate recommendations
            recommendations = []
            
            # Check for large collections
            for name, stats in sorted_collections[:3]:  # Top 3 largest collections
                if stats["storage_mb"] > 1000:  # More than 1GB
                    recommendations.append({
                        "collection": name,
                        "issue": "Large collection size",
                        "recommendation": "Consider time-based partitioning or archiving older data"
                    })
            
            # Check for index size
            for name, stats in collection_stats.items():
                if stats["index_size_mb"] > stats["storage_mb"] * 0.5:  # Indexes more than 50% of data size
                    recommendations.append({
                        "collection": name,
                        "issue": "Large index size relative to data",
                        "recommendation": "Review indexes for potential redundancy or consolidation"
                    })
            
            # Check for small avg object size (potential for schema optimization)
            for name, stats in collection_stats.items():
                if stats["avg_obj_size_bytes"] < 100 and stats["doc_count"] > 10000:
                    recommendations.append({
                        "collection": name,
                        "issue": "Small average object size with many documents",
                        "recommendation": "Consider schema redesign to group related data"
                    })
            
            return {
                "status": "success",
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "collections": sorted_collections,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing database size: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def generate_query_reports(self) -> Dict[str, Any]:
        """
        Generate reports on slow queries and suggest improvements.
        
        Returns:
            Dictionary with query analysis and suggestions
        """
        try:
            # This requires MongoDB profiling to be enabled
            # In a production system, you would analyze the system.profile collection
            
            # For now, provide example implementation guidance
            return {
                "status": "info",
                "message": "Query profiling analysis",
                "setup_instructions": [
                    "Enable profiling: db.setProfilingLevel(1, { slowms: 100 })",
                    "Analyze slow queries: db.system.profile.find().sort({millis:-1}).limit(10)"
                ],
                "common_optimizations": [
                    "Ensure queries use indexes by checking explain() plans",
                    "Use covered queries where possible (queries satisfied entirely by indexes)",
                    "Add compound indexes for common query patterns",
                    "Use projection to limit fields returned",
                    "Consider time-based filters for large collections"
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Error generating query reports: {e}")
            return {
                "status": "error",
                "error": str(e)
            }