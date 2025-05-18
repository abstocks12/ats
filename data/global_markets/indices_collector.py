"""
Global Market Indices Collector

This module collects data on major global market indices for analysis and correlation with Indian markets.
It provides data on major US, European, and Asian market indices.
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

class IndicesCollector:
    """
    Collects data on global market indices from various sources.
    Tracks major US, European, and Asian indices for global market analysis.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the indices collector with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Define major indices to track
        self.indices = {
            # US Indices
            "US": {
                "SPX": {"name": "S&P 500", "symbol": "^GSPC"},
                "DJI": {"name": "Dow Jones Industrial Average", "symbol": "^DJI"},
                "IXIC": {"name": "NASDAQ Composite", "symbol": "^IXIC"},
                "RUT": {"name": "Russell 2000", "symbol": "^RUT"},
                "VIX": {"name": "CBOE Volatility Index", "symbol": "^VIX"}
            },
            # European Indices
            "Europe": {
                "FTSE": {"name": "FTSE 100", "symbol": "^FTSE"},
                "GDAXI": {"name": "DAX", "symbol": "^GDAXI"},
                "FCHI": {"name": "CAC 40", "symbol": "^FCHI"},
                "STOXX50E": {"name": "EURO STOXX 50", "symbol": "^STOXX50E"}
            },
            # Asian Indices
            "Asia": {
                "N225": {"name": "Nikkei 225", "symbol": "^N225"},
                "HSI": {"name": "Hang Seng Index", "symbol": "^HSI"},
                "SSEC": {"name": "Shanghai Composite", "symbol": "^SSEC"},
                "STI": {"name": "Straits Times Index", "symbol": "^STI"},
                "KOSPI": {"name": "KOSPI Composite Index", "symbol": "^KS11"}
            },
            # Indian Indices (for reference)
            "India": {
                "NSEI": {"name": "NIFTY 50", "symbol": "^NSEI"},
                "BSESN": {"name": "BSE SENSEX", "symbol": "^BSESN"}
            }
        }
        
        # API configuration
        self.api_keys = self._load_api_keys()
        self.base_url = "https://api.marketdata.com/v1/indices"  # Replace with actual API URL
        self.alternative_url = "https://www.alphavantage.co/query"  # Backup API
        
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
    
    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables or configuration."""
        try:
            # Try to load from database
            api_config = self.db.system_config_collection.find_one({"config_type": "api_keys"})
            
            if api_config and "global_markets" in api_config:
                return api_config["global_markets"]
            
            # Fallback to default keys (should be replaced in production)
            return {
                "primary_api": "your_primary_api_key",
                "alpha_vantage": "your_alpha_vantage_key"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")
            return {}
    
    def collect_current_data(self) -> Dict[str, Any]:
        """
        Collect current data for all tracked global indices.
        
        Returns:
            Dictionary containing current data for all indices
        """
        self.logger.info("Collecting current data for global indices")
        
        results = {
            "timestamp": datetime.now(),
            "indices": {}
        }
        
        # Process each region
        for region, indices in self.indices.items():
            results["indices"][region] = {}
            
            for index_code, index_info in indices.items():
                try:
                    # Get current data for this index
                    index_data = self._fetch_current_data(index_info["symbol"])
                    
                    if index_data:
                        # Add metadata
                        index_data["name"] = index_info["name"]
                        index_data["code"] = index_code
                        index_data["region"] = region
                        
                        # Add to results
                        results["indices"][region][index_code] = index_data
                        
                        # Save to database
                        self._save_current_data(index_code, index_data)
                        
                        self.logger.info(f"Collected data for {index_info['name']} ({index_code})")
                    else:
                        self.logger.warning(f"No data returned for {index_info['name']} ({index_code})")
                
                except Exception as e:
                    self.logger.error(f"Error collecting data for {index_code}: {e}")
                
                # Respect API rate limits
                time.sleep(1)
        
        return results
    
    def _fetch_current_data(self, symbol: str) -> Dict[str, Any]:
        """
        Fetch current data for a specific index.
        
        Args:
            symbol: Index symbol
            
        Returns:
            Dictionary with current index data
        """
        try:
            # Try primary API first
            if "primary_api" in self.api_keys:
                response = requests.get(
                    self.base_url,
                    params={
                        "symbol": symbol,
                        "apikey": self.api_keys["primary_api"]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and "error" not in data:
                        return self._parse_primary_api_response(data)
            
            # Fall back to Alpha Vantage
            if "alpha_vantage" in self.api_keys:
                response = requests.get(
                    self.alternative_url,
                    params={
                        "function": "GLOBAL_QUOTE",
                        "symbol": symbol,
                        "apikey": self.api_keys["alpha_vantage"]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and "Global Quote" in data:
                        return self._parse_alpha_vantage_response(data)
            
            self.logger.warning(f"Could not fetch data for {symbol} from any API")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching current data for {symbol}: {e}")
            return {}
    
    def _parse_primary_api_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response from primary API."""
        try:
            # Adjust this based on the actual API response structure
            quote = data.get("quote", {})
            
            return {
                "price": float(quote.get("price", 0)),
                "change": float(quote.get("change", 0)),
                "change_percent": float(quote.get("changePercent", 0)),
                "open": float(quote.get("open", 0)),
                "high": float(quote.get("high", 0)),
                "low": float(quote.get("low", 0)),
                "prev_close": float(quote.get("previousClose", 0)),
                "volume": int(quote.get("volume", 0)),
                "timestamp": quote.get("timestamp", datetime.now().isoformat())
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing primary API response: {e}")
            return {}
    
    def _parse_alpha_vantage_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response from Alpha Vantage API."""
        try:
            quote = data.get("Global Quote", {})
            
            return {
                "price": float(quote.get("05. price", 0)),
                "change": float(quote.get("09. change", 0)),
                "change_percent": float(quote.get("10. change percent", "0%").rstrip("%")),
                "open": float(quote.get("02. open", 0)),
                "high": float(quote.get("03. high", 0)),
                "low": float(quote.get("04. low", 0)),
                "prev_close": float(quote.get("08. previous close", 0)),
                "volume": int(quote.get("06. volume", 0)),
                "timestamp": quote.get("07. latest trading day", datetime.now().strftime("%Y-%m-%d"))
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpha Vantage response: {e}")
            return {}
    
    def _save_current_data(self, index_code: str, index_data: Dict[str, Any]) -> bool:
        """
        Save current index data to database.
        
        Args:
            index_code: Index code
            index_data: Index data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document
            document = {
                "index_code": index_code,
                "name": index_data.get("name", ""),
                "region": index_data.get("region", ""),
                "price": index_data.get("price", 0),
                "change": index_data.get("change", 0),
                "change_percent": index_data.get("change_percent", 0),
                "open": index_data.get("open", 0),
                "high": index_data.get("high", 0),
                "low": index_data.get("low", 0),
                "prev_close": index_data.get("prev_close", 0),
                "volume": index_data.get("volume", 0),
                "timestamp": datetime.now(),
                "trading_date": datetime.now().strftime("%Y-%m-%d")
            }
            
            # Insert into database
            self.db.global_indices_collection.insert_one(document)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving current data for {index_code}: {e}")
            return False
    
    def collect_historical_data(self, days: int = 365) -> Dict[str, Any]:
        """
        Collect historical data for all tracked global indices.
        
        Args:
            days: Number of days of historical data to collect
            
        Returns:
            Dictionary containing historical data for all indices
        """
        self.logger.info(f"Collecting {days} days of historical data for global indices")
        
        results = {
            "timestamp": datetime.now(),
            "days": days,
            "indices": {}
        }
        
        # Process each region
        for region, indices in self.indices.items():
            results["indices"][region] = {}
            
            for index_code, index_info in indices.items():
                try:
                    # Get historical data for this index
                    historical_data = self._fetch_historical_data(index_info["symbol"], days)
                    
                    if historical_data:
                        # Add to results
                        results["indices"][region][index_code] = {
                            "name": index_info["name"],
                            "code": index_code,
                            "region": region,
                            "data_points": len(historical_data),
                            "start_date": historical_data[0].get("date") if historical_data else None,
                            "end_date": historical_data[-1].get("date") if historical_data else None
                        }
                        
                        # Save to database
                        saved = self._save_historical_data(index_code, index_info["name"], region, historical_data)
                        
                        if saved:
                            self.logger.info(f"Collected historical data for {index_info['name']} ({index_code})")
                        else:
                            self.logger.warning(f"Failed to save historical data for {index_info['name']} ({index_code})")
                    else:
                        self.logger.warning(f"No historical data returned for {index_info['name']} ({index_code})")
                
                except Exception as e:
                    self.logger.error(f"Error collecting historical data for {index_code}: {e}")
                
                # Respect API rate limits
                time.sleep(1)
        
        return results
    
    def _fetch_historical_data(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """
        Fetch historical data for a specific index.
        
        Args:
            symbol: Index symbol
            days: Number of days of historical data
            
        Returns:
            List of dictionaries with historical index data
        """
        try:
            # Try primary API first
            if "primary_api" in self.api_keys:
                response = requests.get(
                    f"{self.base_url}/historical",
                    params={
                        "symbol": symbol,
                        "period": f"{days}d",
                        "apikey": self.api_keys["primary_api"]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and "error" not in data:
                        return self._parse_primary_historical_response(data)
            
            # Fall back to Alpha Vantage
            if "alpha_vantage" in self.api_keys:
                response = requests.get(
                    self.alternative_url,
                    params={
                        "function": "TIME_SERIES_DAILY",
                        "symbol": symbol,
                        "outputsize": "full" if days > 100 else "compact",
                        "apikey": self.api_keys["alpha_vantage"]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and "Time Series (Daily)" in data:
                        return self._parse_alpha_vantage_historical(data, days)
            
            self.logger.warning(f"Could not fetch historical data for {symbol} from any API")
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return []
    
    def _parse_primary_historical_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Parse response from primary API for historical data."""
        try:
            historical_data = []
            
            # Adjust based on actual API response structure
            for item in data.get("historical", []):
                historical_data.append({
                    "date": item.get("date", ""),
                    "open": float(item.get("open", 0)),
                    "high": float(item.get("high", 0)),
                    "low": float(item.get("low", 0)),
                    "close": float(item.get("close", 0)),
                    "volume": int(item.get("volume", 0))
                })
            
            # Sort by date (oldest first)
            historical_data.sort(key=lambda x: x["date"])
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error parsing primary historical API response: {e}")
            return []
    
    def _parse_alpha_vantage_historical(self, data: Dict[str, Any], days: int) -> List[Dict[str, Any]]:
        """Parse response from Alpha Vantage API for historical data."""
        try:
            time_series = data.get("Time Series (Daily)", {})
            historical_data = []
            
            for date, values in time_series.items():
                historical_data.append({
                    "date": date,
                    "open": float(values.get("1. open", 0)),
                    "high": float(values.get("2. high", 0)),
                    "low": float(values.get("3. low", 0)),
                    "close": float(values.get("4. close", 0)),
                    "volume": int(values.get("5. volume", 0))
                })
            
            # Sort by date (oldest first)
            historical_data.sort(key=lambda x: x["date"])
            
            # Limit to requested number of days
            return historical_data[-days:] if len(historical_data) > days else historical_data
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpha Vantage historical response: {e}")
            return []
    
    def _save_historical_data(self, index_code: str, name: str, region: str, 
                             historical_data: List[Dict[str, Any]]) -> bool:
        """
        Save historical index data to database.
        
        Args:
            index_code: Index code
            name: Index name
            region: Geographic region
            historical_data: List of historical data points
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare documents for bulk insert
            documents = []
            
            for data_point in historical_data:
                document = {
                    "index_code": index_code,
                    "name": name,
                    "region": region,
                    "date": data_point["date"],
                    "open": data_point["open"],
                    "high": data_point["high"],
                    "low": data_point["low"],
                    "close": data_point["close"],
                    "volume": data_point["volume"],
                    "timestamp": datetime.now()
                }
                
                documents.append(document)
            
            if documents:
                # Check if we need to update or insert
                existing_dates = set()
                
                existing = self.db.global_indices_historical_collection.find(
                    {"index_code": index_code},
                    {"date": 1}
                )
                
                for doc in existing:
                    existing_dates.add(doc.get("date"))
                
                # Filter out documents with dates that already exist
                new_documents = [doc for doc in documents if doc["date"] not in existing_dates]
                
                if new_documents:
                    self.db.global_indices_historical_collection.insert_many(new_documents)
                    self.logger.info(f"Inserted {len(new_documents)} new historical data points for {index_code}")
                else:
                    self.logger.info(f"No new historical data points for {index_code}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving historical data for {index_code}: {e}")
            return False
    
    def get_latest_indices_data(self) -> Dict[str, Any]:
        """
        Get the latest data for all tracked indices.
        
        Returns:
            Dictionary containing the latest data for all indices
        """
        try:
            results = {
                "timestamp": datetime.now(),
                "indices": {}
            }
            
            # Get latest data for each region/index
            for region, indices in self.indices.items():
                results["indices"][region] = {}
                
                for index_code in indices.keys():
                    # Get latest data from database
                    latest = self.db.global_indices_collection.find_one(
                        {"index_code": index_code},
                        sort=[("timestamp", -1)]
                    )
                    
                    if latest:
                        results["indices"][region][index_code] = {
                            "name": latest.get("name", ""),
                            "price": latest.get("price", 0),
                            "change": latest.get("change", 0),
                            "change_percent": latest.get("change_percent", 0),
                            "timestamp": latest.get("timestamp")
                        }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting latest indices data: {e}")
            return {"error": str(e)}
    
    def calculate_correlations(self, target_index: str = "NSEI", days: int = 90) -> Dict[str, Any]:
        """
        Calculate correlations between target index and other global indices.
        
        Args:
            target_index: Target index code to compare (default: NIFTY 50)
            days: Number of days for correlation calculation
            
        Returns:
            Dictionary containing correlation data
        """
        try:
            # Get historical data for target index
            target_data = list(self.db.global_indices_historical_collection.find(
                {"index_code": target_index},
                sort=[("date", -1)]
            ).limit(days))
            
            if not target_data:
                return {"error": f"No historical data found for {target_index}"}
            
            # Create DataFrame for target index
            target_df = pd.DataFrame(target_data)
            
            # Dictionary to store correlations
            correlations = {
                "target_index": target_index,
                "period_days": days,
                "timestamp": datetime.now(),
                "correlations": {}
            }
            
            # Process each region
            for region, indices in self.indices.items():
                correlations["correlations"][region] = {}
                
                for index_code in indices.keys():
                    # Skip the target index
                    if index_code == target_index:
                        continue
                    
                    # Get historical data for this index
                    index_data = list(self.db.global_indices_historical_collection.find(
                        {"index_code": index_code},
                        sort=[("date", -1)]
                    ).limit(days))
                    
                    if index_data:
                        # Create DataFrame
                        index_df = pd.DataFrame(index_data)
                        
                        # Ensure we have matching dates
                        merged_df = pd.merge(
                            target_df[["date", "close"]],
                            index_df[["date", "close"]],
                            on="date",
                            how="inner",
                            suffixes=("_target", "_index")
                        )
                        
                        if len(merged_df) > 10:  # Need enough data points
                            # Calculate correlation
                            corr = merged_df["close_target"].corr(merged_df["close_index"])
                            
                            # Calculate 1-day lagged correlation (if index moves ahead of target)
                            merged_df["close_index_lag1"] = merged_df["close_index"].shift(-1)
                            lag_corr = merged_df["close_target"].corr(merged_df["close_index_lag1"])
                            
                            # Calculate returns correlation
                            merged_df["target_return"] = merged_df["close_target"].pct_change()
                            merged_df["index_return"] = merged_df["close_index"].pct_change()
                            returns_corr = merged_df["target_return"].corr(merged_df["index_return"])
                            
                            correlations["correlations"][region][index_code] = {
                                "name": indices[index_code]["name"],
                                "price_correlation": round(corr, 4),
                                "lagged_correlation": round(lag_corr, 4),
                                "returns_correlation": round(returns_corr, 4),
                                "data_points": len(merged_df)
                            }
            
            # Save correlations to database
            self._save_correlations(correlations)
            
            return correlations
            
        except Exception as e:
            self.logger.error(f"Error calculating correlations: {e}")
            return {"error": str(e)}
    
    def _save_correlations(self, correlations: Dict[str, Any]) -> bool:
        """
        Save correlation data to database.
        
        Args:
            correlations: Correlation data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.db.indices_correlations_collection.insert_one(correlations)
            self.logger.info("Saved index correlations to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving correlations: {e}")
            return False
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Get a summary of global market performance.
        
        Returns:
            Dictionary containing market summary
        """
        try:
            # Get latest data for all indices
            latest_data = self.get_latest_indices_data()
            
            if "error" in latest_data:
                return latest_data
            
            summary = {
                "timestamp": datetime.now(),
                "regions": {}
            }
            
            # Calculate performance metrics for each region
            for region, indices in latest_data["indices"].items():
                up_count = 0
                down_count = 0
                unchanged_count = 0
                avg_change_pct = 0
                
                for index_code, index_data in indices.items():
                    change_pct = index_data.get("change_percent", 0)
                    avg_change_pct += change_pct
                    
                    if change_pct > 0:
                        up_count += 1
                    elif change_pct < 0:
                        down_count += 1
                    else:
                        unchanged_count += 1
                
                total_indices = len(indices)
                if total_indices > 0:
                    avg_change_pct /= total_indices
                
                # Market breadth
                breadth = (up_count - down_count) / total_indices if total_indices > 0 else 0
                
                # Determine market sentiment
                if breadth > 0.6:
                    sentiment = "Strongly Bullish"
                elif breadth > 0.2:
                    sentiment = "Bullish"
                elif breadth > -0.2:
                    sentiment = "Neutral"
                elif breadth > -0.6:
                    sentiment = "Bearish"
                else:
                    sentiment = "Strongly Bearish"
                
                summary["regions"][region] = {
                    "total_indices": total_indices,
                    "up_count": up_count,
                    "down_count": down_count,
                    "unchanged_count": unchanged_count,
                    "breadth": round(breadth, 2),
                    "avg_change_percent": round(avg_change_pct, 2),
                    "sentiment": sentiment,
                    "indices": indices
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting market summary: {e}")
            return {"error": str(e)}
    
    def run_daily_collection(self) -> Dict[str, Any]:
        """
        Run daily collection of global indices data.
        
        Returns:
            Dictionary containing collection results
        """
        self.logger.info("Running daily collection of global indices data")
        
        results = {
            "timestamp": datetime.now(),
            "current_data": None,
            "historical_update": None,
            "correlations": None
        }
        
        try:
            # 1. Collect current data
            current_data = self.collect_current_data()
            results["current_data"] = {
                "status": "success" if current_data else "failed",
                "count": sum(len(indices) for region, indices in current_data.get("indices", {}).items())
            }
            
            # 2. Update historical data (just the latest day)
            historical_update = self.collect_historical_data(days=7)  # Get recent days to ensure we have latest
            results["historical_update"] = {
                "status": "success" if historical_update else "failed",
                "count": sum(indices.get("data_points", 0) for region, indices in historical_update.get("indices", {}).items())
            }
            
            # 3. Calculate correlations
            if current_data:
                correlations = self.calculate_correlations()
                results["correlations"] = {
                    "status": "success" if "error" not in correlations else "failed",
                    "target_index": correlations.get("target_index")
                }
            
            self.logger.info("Daily collection of global indices data completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in daily collection: {e}")
            results["error"] = str(e)
        
        return results


# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize collector
    indices_collector = IndicesCollector(db)
    
    # Run daily collection
    results = indices_collector.run_daily_collection()
    print(json.dumps(results, default=str, indent=2))