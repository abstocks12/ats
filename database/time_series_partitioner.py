"""
Time Series Partitioner

This module manages time-based partitioning and data management for time series data.
It optimizes storage and retrieval for high-volume market and trading data.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import pymongo

class TimeSeriesPartitioner:
    """
    Manages time-based partitioning for market data collections.
    Provides methods for partitioning, data migration, and query routing.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the time series partitioner with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Configure partition settings
        self.partition_configs = {
            "market_data_collection": {
                "partition_field": "timestamp",
                "partition_interval": "month",  # month, week, day
                "partitions_to_keep": 3,        # Keep 3 recent partitions in hot storage
                "archive_older": True           # Archive older partitions
            },
            "trades_collection": {
                "partition_field": "entry_time",
                "partition_interval": "month",
                "partitions_to_keep": 6,
                "archive_older": True
            },
            "news_collection": {
                "partition_field": "published_date",
                "partition_interval": "month",
                "partitions_to_keep": 3,
                "archive_older": True
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
    
    def setup_partitioning(self, collection_name: str) -> Dict[str, Any]:
        """
        Set up time-based partitioning for a collection.
        
        Args:
            collection_name: Name of the collection to partition
            
        Returns:
            Dictionary with setup results
        """
        try:
            if collection_name not in self.partition_configs:
                return {
                    "status": "error",
                    "error": f"No partition configuration found for {collection_name}"
                }
            
            config = self.partition_configs[collection_name]
            
            # Create partitions for current period and recent periods
            current_partition = self._create_current_partition(collection_name, config)
            recent_partitions = self._create_recent_partitions(collection_name, config)
            
            # Create indexes on the partitions
            self._create_partition_indexes(collection_name, current_partition)
            for partition in recent_partitions:
                self._create_partition_indexes(collection_name, partition)
            
            # Set up TTL index on base collection if archiving is enabled
            if config["archive_older"]:
                self._setup_ttl_index(collection_name, config)
            
            return {
                "status": "success",
                "collection": collection_name,
                "current_partition": current_partition,
                "recent_partitions": recent_partitions,
                "partitioning_active": True
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up partitioning for {collection_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _create_current_partition(self, collection_name: str, config: Dict) -> str:
        """
        Create partition collection for the current time period.
        
        Args:
            collection_name: Base collection name
            config: Partitioning configuration
            
        Returns:
            Name of the current partition collection
        """
        # Get current date
        now = datetime.now()
        
        # Generate partition name based on time interval
        interval = config["partition_interval"]
        if interval == "day":
            suffix = now.strftime("%Y%m%d")
        elif interval == "week":
            suffix = now.strftime("%Y_w%U")  # ISO week number
        elif interval == "month":
            suffix = now.strftime("%Y%m")
        elif interval == "quarter":
            quarter = (now.month - 1) // 3 + 1
            suffix = f"{now.year}_q{quarter}"
        else:
            suffix = now.strftime("%Y%m")  # Default to month
        
        partition_name = f"{collection_name}_{suffix}"
        
        # Check if partition already exists
        if partition_name in self.db.list_collection_names():
            self.logger.info(f"Partition {partition_name} already exists")
            return partition_name
        
        # Create the partition collection
        self.db.create_collection(partition_name)
        self.logger.info(f"Created partition {partition_name}")
        
        return partition_name
    
    def _create_recent_partitions(self, collection_name: str, config: Dict) -> List[str]:
        """
        Create partition collections for recent time periods.
        
        Args:
            collection_name: Base collection name
            config: Partitioning configuration
            
        Returns:
            List of recent partition collection names
        """
        # Get current date
        now = datetime.now()
        
        # Determine how many previous periods to create
        interval = config["partition_interval"]
        partitions_to_keep = config.get("partitions_to_keep", 3)
        
        # Generate dates for recent periods
        recent_dates = []
        for i in range(1, partitions_to_keep):
            if interval == "day":
                recent_dates.append(now - timedelta(days=i))
            elif interval == "week":
                recent_dates.append(now - timedelta(weeks=i))
            elif interval == "month":
                # Go back i months
                month = now.month - i
                year = now.year
                while month <= 0:
                    month += 12
                    year -= 1
                recent_dates.append(datetime(year, month, 1))
            elif interval == "quarter":
                # Go back i quarters
                quarter = (now.month - 1) // 3 + 1 - i
                year = now.year
                while quarter <= 0:
                    quarter += 4
                    year -= 1
                month = (quarter - 1) * 3 + 1
                recent_dates.append(datetime(year, month, 1))
        
        # Create partitions for each recent date
        recent_partitions = []
        for date in recent_dates:
            if interval == "day":
                suffix = date.strftime("%Y%m%d")
            elif interval == "week":
                suffix = date.strftime("%Y_w%U")
            elif interval == "month":
                suffix = date.strftime("%Y%m")
            elif interval == "quarter":
                quarter = (date.month - 1) // 3 + 1
                suffix = f"{date.year}_q{quarter}"
            else:
                suffix = date.strftime("%Y%m")
            
            partition_name = f"{collection_name}_{suffix}"
            
            # Check if partition already exists
            if partition_name in self.db.list_collection_names():
                self.logger.info(f"Partition {partition_name} already exists")
            else:
                # Create the partition collection
                self.db.create_collection(partition_name)
                self.logger.info(f"Created partition {partition_name}")
            
            recent_partitions.append(partition_name)
        
        return recent_partitions
    
    def _create_partition_indexes(self, collection_name: str, partition_name: str) -> None:
        """
        Create indexes on a partition collection.
        
        Args:
            collection_name: Base collection name
            partition_name: Partition collection name
        """
        # Define standard indexes based on collection type
        if collection_name == "market_data_collection":
            # Indexes for market data
            indexes = [
                [("symbol", 1), ("timestamp", -1)],
                [("timestamp", -1)],
                [("symbol", 1), ("timeframe", 1), ("timestamp", -1)]
            ]
        elif collection_name == "trades_collection":
            # Indexes for trades
            indexes = [
                [("symbol", 1), ("entry_time", -1)],
                [("entry_time", -1)],
                [("strategy", 1), ("entry_time", -1)]
            ]
        elif collection_name == "news_collection":
            # Indexes for news
            indexes = [
                [("entities", 1), ("published_date", -1)],
                [("published_date", -1)],
                [("sentiment", 1), ("published_date", -1)]
            ]
        else:
            # Default indexes
            indexes = [
                [("timestamp", -1)]
            ]
        
        # Create each index
        for index in indexes:
            try:
                self.db[partition_name].create_index(index, background=True)
                self.logger.info(f"Created index {index} on {partition_name}")
            except Exception as e:
                self.logger.error(f"Error creating index {index} on {partition_name}: {e}")
    
    def _setup_ttl_index(self, collection_name: str, config: Dict) -> None:
        """
        Set up TTL index for data expiration on base collection.
        
        Args:
            collection_name: Base collection name
            config: Partitioning configuration
        """
        # Calculate TTL based on partitioning configuration
        interval = config["partition_interval"]
        partitions_to_keep = config.get("partitions_to_keep", 3)
        
        # Set TTL based on interval
        if interval == "day":
            ttl_days = partitions_to_keep
        elif interval == "week":
            ttl_days = partitions_to_keep * 7
        elif interval == "month":
            ttl_days = partitions_to_keep * 30  # Approximate
        elif interval == "quarter":
            ttl_days = partitions_to_keep * 90  # Approximate
        else:
            ttl_days = 30  # Default
        
        # Create TTL index
        field = config["partition_field"]
        try:
            self.db[collection_name].create_index(
                [(field, -1)],
                expireAfterSeconds=ttl_days * 24 * 60 * 60,
                background=True
            )
            self.logger.info(f"Created TTL index on {collection_name}.{field} with {ttl_days} days expiration")
        except Exception as e:
            self.logger.error(f"Error creating TTL index on {collection_name}: {e}")
    
    def migrate_data_to_partitions(self, collection_name: str) -> Dict[str, Any]:
        """
        Migrate existing data from base collection to time-based partitions.
        
        Args:
            collection_name: Base collection name
            
        Returns:
            Dictionary with migration results
        """
        try:
            if collection_name not in self.partition_configs:
                return {
                    "status": "error",
                    "error": f"No partition configuration found for {collection_name}"
                }
            
            config = self.partition_configs[collection_name]
            field = config["partition_field"]
            interval = config["partition_interval"]
            
            # Get existing data from base collection
            base_collection = getattr(self.db, collection_name)
            
            # Set up aggregation to group by time period
            if interval == "day":
                group_format = "%Y%m%d"
            elif interval == "week":
                group_format = "%Y_w%U"
            elif interval == "month":
                group_format = "%Y%m"
            elif interval == "quarter":
                group_format = {
                    "$concat": [
                        {"$toString": {"$year": f"${field}"}},
                        "_q",
                        {"$toString": {"$ceil": {"$divide": [{"$month": f"${field}"}, 3]}}}
                    ]
                }
            else:
                group_format = "%Y%m"  # Default to month
            
            # For quarter, we need a different approach due to complex grouping
            if interval == "quarter":
                # Process in smaller batches for quarter partitioning
                cursor = base_collection.find({})
                partitioned_counts = {}
                
                for doc in cursor:
                    # Calculate quarter
                    doc_date = doc.get(field)
                    if not doc_date:
                        continue
                    
                    quarter = (doc_date.month - 1) // 3 + 1
                    suffix = f"{doc_date.year}_q{quarter}"
                    partition_name = f"{collection_name}_{suffix}"
                    
                    # Insert into partition
                    try:
                        self.db[partition_name].insert_one(doc)
                        partitioned_counts[partition_name] = partitioned_counts.get(partition_name, 0) + 1
                    except Exception as e:
                        self.logger.error(f"Error inserting document to {partition_name}: {e}")
            else:
                # For day, week, month, use aggregation pipeline
                # For day, week, month, use aggregation pipeline
                pipeline = [
                    {
                        "$group": {
                            "_id": {
                                "$dateToString": {
                                    "format": group_format, 
                                    "date": f"${field}"
                                }
                            },
                            "docs": {"$push": "$$ROOT"}
                        }
                    }
                ]
                
                results = base_collection.aggregate(pipeline)
                partitioned_counts = {}
                
                for result in results:
                    suffix = result["_id"]
                    if not suffix:
                        continue
                    
                    partition_name = f"{collection_name}_{suffix}"
                    docs = result["docs"]
                    
                    # Insert documents into partition collection
                    try:
                        if len(docs) > 0:
                            self.db[partition_name].insert_many(docs, ordered=False)
                            partitioned_counts[partition_name] = len(docs)
                            self.logger.info(f"Migrated {len(docs)} documents to {partition_name}")
                    except Exception as e:
                        self.logger.error(f"Error inserting documents to {partition_name}: {e}")
            
            # Log and return results
            total_migrated = sum(partitioned_counts.values())
            self.logger.info(f"Completed migration for {collection_name}: {total_migrated} documents migrated")
            
            return {
                "status": "success",
                "collection": collection_name,
                "total_migrated": total_migrated,
                "partition_counts": partitioned_counts
            }
            
        except Exception as e:
            self.logger.error(f"Error migrating data for {collection_name}: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def get_partition_for_date(self, collection_name: str, date: datetime) -> str:
        """
        Get the appropriate partition collection name for a specific date.
        
        Args:
            collection_name: Base collection name
            date: Date to determine partition for
            
        Returns:
            Partition collection name
        """
        if collection_name not in self.partition_configs:
            return collection_name  # Fall back to base collection
        
        config = self.partition_configs[collection_name]
        interval = config["partition_interval"]
        
        # Generate suffix based on interval
        if interval == "day":
            suffix = date.strftime("%Y%m%d")
        elif interval == "week":
            suffix = date.strftime("%Y_w%U")
        elif interval == "month":
            suffix = date.strftime("%Y%m")
        elif interval == "quarter":
            quarter = (date.month - 1) // 3 + 1
            suffix = f"{date.year}_q{quarter}"
        else:
            suffix = date.strftime("%Y%m")  # Default to month
        
        return f"{collection_name}_{suffix}"
    
    def route_query(self, collection_name: str, query: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """
        Route a query to the appropriate partition based on date filters.
        
        Args:
            collection_name: Base collection name
            query: Query dictionary
            
        Returns:
            Tuple of (routed_collection_name, modified_query)
        """
        if collection_name not in self.partition_configs:
            return collection_name, query
        
        config = self.partition_configs[collection_name]
        field = config["partition_field"]
        
        # Check for date field in query
        date_filter = None
        
        # Extract date from query
        if field in query:
            date_filter = query[field]
        elif f"{field}.$gte" in query:
            date_filter = query[f"{field}.$gte"]
        elif field in query and "$gte" in query[field]:
            date_filter = query[field]["$gte"]
        
        # If no date filter or it's a complex query, use base collection
        if not date_filter or not isinstance(date_filter, datetime):
            return collection_name, query
        
        # Get partition name for this date
        partition_name = self.get_partition_for_date(collection_name, date_filter)
        
        # Check if partition exists
        if partition_name not in self.db.list_collection_names():
            return collection_name, query
        
        return partition_name, query
    
    def multi_partition_query(self, collection_name: str, query: Dict[str, Any], 
                           projection: Optional[Dict] = None, sort: Optional[List] = None) -> List[Dict]:
        """
        Query across multiple partitions and merge results.
        
        Args:
            collection_name: Base collection name
            query: Query dictionary
            projection: Fields to include in results
            sort: Sort specification
            
        Returns:
            List of documents from all matching partitions
        """
        if collection_name not in self.partition_configs:
            # Fall back to simple query
            return list(self.db[collection_name].find(query, projection).sort(sort))
        
        config = self.partition_configs[collection_name]
        field = config["partition_field"]
        
        # Check for date range query
        start_date = None
        end_date = None
        
        if field in query and isinstance(query[field], dict):
            if "$gte" in query[field]:
                start_date = query[field]["$gte"]
            if "$lte" in query[field]:
                end_date = query[field]["$lte"]
        
        # If no date range, use simple query
        if not start_date:
            return list(self.db[collection_name].find(query, projection).sort(sort))
        
        # Set end date if not specified
        if not end_date:
            end_date = datetime.now()
        
        # Get all partitions in the date range
        target_partitions = []
        current_date = start_date
        
        while current_date <= end_date:
            partition_name = self.get_partition_for_date(collection_name, current_date)
            if partition_name not in target_partitions and partition_name in self.db.list_collection_names():
                target_partitions.append(partition_name)
            
            # Increment date based on partition interval
            interval = config["partition_interval"]
            if interval == "day":
                current_date += timedelta(days=1)
            elif interval == "week":
                current_date += timedelta(weeks=1)
            elif interval == "month":
                # Move to next month
                month = current_date.month + 1
                year = current_date.year
                if month > 12:
                    month = 1
                    year += 1
                current_date = datetime(year, month, 1)
            elif interval == "quarter":
                # Move to next quarter
                quarter = (current_date.month - 1) // 3 + 1
                quarter += 1
                year = current_date.year
                if quarter > 4:
                    quarter = 1
                    year += 1
                month = (quarter - 1) * 3 + 1
                current_date = datetime(year, month, 1)
            else:
                current_date += timedelta(days=30)  # Default increment
        
        # Also check base collection
        target_partitions.append(collection_name)
        
        # Query all partitions and combine results
        all_results = []
        
        for partition_name in target_partitions:
            try:
                results = list(self.db[partition_name].find(query, projection))
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Error querying partition {partition_name}: {e}")
        
        # Sort combined results if needed
        if sort:
            from operator import itemgetter
            
            # Extract sort field and direction
            sort_field = sort[0][0]
            sort_direction = sort[0][1]
            
            # Sort results
            all_results.sort(
                key=itemgetter(sort_field),
                reverse=(sort_direction == -1)
            )
        
        return all_results
    
    def cleanup_partitions(self) -> Dict[str, Any]:
        """
        Clean up old partitions based on retention policy.
        
        Returns:
            Dictionary with cleanup results
        """
        try:
            results = {}
            
            for collection_name, config in self.partition_configs.items():
                # Skip if archiving is not enabled
                if not config.get("archive_older", True):
                    continue
                
                # Get all partitions for this collection
                partitions = [name for name in self.db.list_collection_names() 
                             if name.startswith(f"{collection_name}_")]
                
                # Calculate cutoff date based on partitions to keep
                now = datetime.now()
                partitions_to_keep = config.get("partitions_to_keep", 3)
                interval = config["partition_interval"]
                
                if interval == "day":
                    cutoff_date = now - timedelta(days=partitions_to_keep)
                elif interval == "week":
                    cutoff_date = now - timedelta(weeks=partitions_to_keep)
                elif interval == "month":
                    # Go back partitions_to_keep months
                    month = now.month - partitions_to_keep
                    year = now.year
                    while month <= 0:
                        month += 12
                        year -= 1
                    cutoff_date = datetime(year, month, 1)
                elif interval == "quarter":
                    # Go back partitions_to_keep quarters
                    quarter = (now.month - 1) // 3 + 1 - partitions_to_keep
                    year = now.year
                    while quarter <= 0:
                        quarter += 4
                        year -= 1
                    month = (quarter - 1) * 3 + 1
                    cutoff_date = datetime(year, month, 1)
                else:
                    cutoff_date = now - timedelta(days=90)  # Default: 3 months
                
                deleted_partitions = []
                
                # Process each partition
                for partition_name in partitions:
                    try:
                        # Extract date from partition name
                        suffix = partition_name.replace(f"{collection_name}_", "")
                        
                        # Parse date from suffix
                        if interval == "day":
                            partition_date = datetime.strptime(suffix, "%Y%m%d")
                        elif interval == "week":
                            year, week = suffix.split("_w")
                            # Create date for first day of the week
                            partition_date = datetime.strptime(f"{year}-{week}-1", "%Y-%U-%w")
                        elif interval == "month":
                            partition_date = datetime.strptime(f"{suffix}01", "%Y%m%d")
                        elif interval == "quarter":
                            year, quarter = suffix.split("_q")
                            month = (int(quarter) - 1) * 3 + 1
                            partition_date = datetime(int(year), month, 1)
                        else:
                            # Default format
                            partition_date = datetime.strptime(f"{suffix}01", "%Y%m%d")
                        
                        # Check if partition is older than cutoff
                        if partition_date < cutoff_date:
                            # Archive collection (backup or move to cold storage would be here)
                            # For now, just drop it
                            self.db.drop_collection(partition_name)
                            deleted_partitions.append(partition_name)
                            self.logger.info(f"Dropped old partition {partition_name}")
                            
                    except Exception as e:
                        self.logger.error(f"Error processing partition {partition_name}: {e}")
                
                # Store results
                results[collection_name] = {
                    "partitions_deleted": deleted_partitions,
                    "cutoff_date": cutoff_date.strftime("%Y-%m-%d")
                }
            
            return {
                "status": "success",
                "cleanup_results": results
            }
            
        except Exception as e:
            self.logger.error(f"Error cleaning up partitions: {e}")
            return {
                "status": "error",
                "error": str(e)
            }