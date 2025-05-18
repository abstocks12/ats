"""
Forex Data Collector

This module collects foreign exchange rates data for major global currencies.
It provides current and historical exchange rates relevant to the Indian market and global trade.
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json

class ForexCollector:
    """
    Collects data on major foreign exchange rates.
    Tracks INR pairs and major global currency pairs for forex analysis.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the forex collector with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Define currency pairs to track
        self.currency_pairs = {
            # INR Pairs (Indian Rupee)
            "INR": {
                "USDINR": {"base": "USD", "quote": "INR", "description": "US Dollar to Indian Rupee"},
                "EURINR": {"base": "EUR", "quote": "INR", "description": "Euro to Indian Rupee"},
                "GBPINR": {"base": "GBP", "quote": "INR", "description": "British Pound to Indian Rupee"},
                "JPYINR": {"base": "JPY", "quote": "INR", "description": "Japanese Yen to Indian Rupee"},
                "SGDINR": {"base": "SGD", "quote": "INR", "description": "Singapore Dollar to Indian Rupee"}
            },
            # Major Pairs
            "Major": {
                "EURUSD": {"base": "EUR", "quote": "USD", "description": "Euro to US Dollar"},
                "GBPUSD": {"base": "GBP", "quote": "USD", "description": "British Pound to US Dollar"},
                "USDJPY": {"base": "USD", "quote": "JPY", "description": "US Dollar to Japanese Yen"},
                "USDCHF": {"base": "USD", "quote": "CHF", "description": "US Dollar to Swiss Franc"},
                "AUDUSD": {"base": "AUD", "quote": "USD", "description": "Australian Dollar to US Dollar"},
                "USDCAD": {"base": "USD", "quote": "CAD", "description": "US Dollar to Canadian Dollar"}
            },
            # Asian Pairs
            "Asian": {
                "USDCNY": {"base": "USD", "quote": "CNY", "description": "US Dollar to Chinese Yuan"},
                "USDHKD": {"base": "USD", "quote": "HKD", "description": "US Dollar to Hong Kong Dollar"},
                "USDSGD": {"base": "USD", "quote": "SGD", "description": "US Dollar to Singapore Dollar"},
                "USDKRW": {"base": "USD", "quote": "KRW", "description": "US Dollar to South Korean Won"}
            },
            # Commodity Currencies
            "Commodity": {
                "AUDUSD": {"base": "AUD", "quote": "USD", "description": "Australian Dollar to US Dollar"},
                "NZDUSD": {"base": "NZD", "quote": "USD", "description": "New Zealand Dollar to US Dollar"},
                "USDCAD": {"base": "USD", "quote": "CAD", "description": "US Dollar to Canadian Dollar"},
                "USDBRL": {"base": "USD", "quote": "BRL", "description": "US Dollar to Brazilian Real"},
                "USDRUB": {"base": "USD", "quote": "RUB", "description": "US Dollar to Russian Ruble"}
            }
        }
        
        # API configuration
        self.api_keys = self._load_api_keys()
        self.base_url = "https://api.exchangerate.host/latest"  # Free forex API
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
            
            if api_config and "forex" in api_config:
                return api_config["forex"]
            
            # Fallback to default keys (should be replaced in production)
            return {
                "primary_api": "your_exchangerate_api_key",
                "alpha_vantage": "your_alpha_vantage_key"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")
            return {}
    
    def collect_current_rates(self) -> Dict[str, Any]:
        """
        Collect current exchange rates for all tracked currency pairs.
        
        Returns:
            Dictionary containing current exchange rates
        """
        self.logger.info("Collecting current exchange rates")
        
        results = {
            "timestamp": datetime.now(),
            "pairs": {}
        }
        
        # Process each category
        for category, pairs in self.currency_pairs.items():
            results["pairs"][category] = {}
            
            for pair_code, pair_info in pairs.items():
                try:
                    # Get current rate for this pair
                    rate_data = self._fetch_current_rate(pair_info["base"], pair_info["quote"])
                    
                    if rate_data:
                        # Add metadata
                        rate_data["pair"] = pair_code
                        rate_data["description"] = pair_info["description"]
                        rate_data["category"] = category
                        
                        # Add to results
                        results["pairs"][category][pair_code] = rate_data
                        
                        # Save to database
                        self._save_current_rate(pair_code, rate_data)
                        
                        self.logger.info(f"Collected rate for {pair_code}: {rate_data['rate']}")
                    else:
                        self.logger.warning(f"No data returned for {pair_code}")
                
                except Exception as e:
                    self.logger.error(f"Error collecting rate for {pair_code}: {e}")
                
                # Respect API rate limits
                time.sleep(0.5)
        
        return results
    
    def _fetch_current_rate(self, base_currency: str, quote_currency: str) -> Dict[str, Any]:
        """
        Fetch current exchange rate for a specific currency pair.
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
            
        Returns:
            Dictionary with current exchange rate data
        """
        try:
            # Try primary API first (exchangerate.host)
            response = requests.get(
                self.base_url,
                params={
                    "base": base_currency,
                    "symbols": quote_currency
                },
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data and "rates" in data and quote_currency in data["rates"]:
                    rate = data["rates"][quote_currency]
                    
                    return {
                        "base": base_currency,
                        "quote": quote_currency,
                        "rate": float(rate),
                        "timestamp": datetime.now().isoformat(),
                        "source": "exchangerate.host"
                    }
            
            # Fall back to Alpha Vantage
            if "alpha_vantage" in self.api_keys:
                response = requests.get(
                    self.alternative_url,
                    params={
                        "function": "CURRENCY_EXCHANGE_RATE",
                        "from_currency": base_currency,
                        "to_currency": quote_currency,
                        "apikey": self.api_keys["alpha_vantage"]
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and "Realtime Currency Exchange Rate" in data:
                        return self._parse_alpha_vantage_response(data)
            
            self.logger.warning(f"Could not fetch rate for {base_currency}/{quote_currency} from any API")
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching current rate for {base_currency}/{quote_currency}: {e}")
            return {}
    
    def _parse_alpha_vantage_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse response from Alpha Vantage API."""
        try:
            exchange_rate = data.get("Realtime Currency Exchange Rate", {})
            
            return {
                "base": exchange_rate.get("1. From_Currency Code", ""),
                "quote": exchange_rate.get("3. To_Currency Code", ""),
                "rate": float(exchange_rate.get("5. Exchange Rate", 0)),
                "timestamp": exchange_rate.get("6. Last Refreshed", datetime.now().isoformat()),
                "source": "Alpha Vantage"
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpha Vantage response: {e}")
            return {}
    
    def _save_current_rate(self, pair_code: str, rate_data: Dict[str, Any]) -> bool:
        """
        Save current exchange rate to database.
        
        Args:
            pair_code: Currency pair code
            rate_data: Exchange rate data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare document
            document = {
                "pair_code": pair_code,
                "base_currency": rate_data.get("base", ""),
                "quote_currency": rate_data.get("quote", ""),
                "description": rate_data.get("description", ""),
                "category": rate_data.get("category", ""),
                "rate": rate_data.get("rate", 0),
                "collection_timestamp": datetime.now(),
                "rate_timestamp": rate_data.get("timestamp", datetime.now().isoformat()),
                "source": rate_data.get("source", "")
            }
            
            # Insert into database
            self.db.forex_rates_collection.insert_one(document)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving current rate for {pair_code}: {e}")
            return False
    
    def collect_historical_rates(self, days: int = 365) -> Dict[str, Any]:
        """
        Collect historical exchange rates for all tracked currency pairs.
        
        Args:
            days: Number of days of historical data to collect
            
        Returns:
            Dictionary containing historical exchange rates
        """
        self.logger.info(f"Collecting {days} days of historical exchange rates")
        
        results = {
            "timestamp": datetime.now(),
            "days": days,
            "pairs": {}
        }
        
        # Calculate start date
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Process each category
        for category, pairs in self.currency_pairs.items():
            results["pairs"][category] = {}
            
            for pair_code, pair_info in pairs.items():
                try:
                    # Get historical rates for this pair
                    historical_data = self._fetch_historical_rates(
                        pair_info["base"], 
                        pair_info["quote"],
                        start_date,
                        end_date
                    )
                    
                    if historical_data:
                        # Add to results
                        results["pairs"][category][pair_code] = {
                            "pair": pair_code,
                            "description": pair_info["description"],
                            "category": category,
                            "data_points": len(historical_data),
                            "start_date": historical_data[0].get("date") if historical_data else None,
                            "end_date": historical_data[-1].get("date") if historical_data else None
                        }
                        
                        # Save to database
                        saved = self._save_historical_rates(
                            pair_code, 
                            pair_info["description"], 
                            category,
                            pair_info["base"],
                            pair_info["quote"],
                            historical_data
                        )
                        
                        if saved:
                            self.logger.info(f"Collected historical rates for {pair_code}")
                        else:
                            self.logger.warning(f"Failed to save historical rates for {pair_code}")
                    else:
                        self.logger.warning(f"No historical rates returned for {pair_code}")
                
                except Exception as e:
                    self.logger.error(f"Error collecting historical rates for {pair_code}: {e}")
                
                # Respect API rate limits
                time.sleep(1)
        
        return results
    
    def _fetch_historical_rates(self, base_currency: str, quote_currency: str, 
                               start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        """
        Fetch historical exchange rates for a specific currency pair.
        
        Args:
            base_currency: Base currency code
            quote_currency: Quote currency code
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            List of dictionaries with historical exchange rate data
        """
        try:
            # Format dates
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")
            
            # Try exchangerate.host API (supports historical timeseries)
            response = requests.get(
                "https://api.exchangerate.host/timeseries",
                params={
                    "base": base_currency,
                    "symbols": quote_currency,
                    "start_date": start_str,
                    "end_date": end_str
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if data and "rates" in data:
                    return self._parse_exchangerate_historical(data, quote_currency)
            
            # Fall back to Alpha Vantage (limited to daily data for FX)
            if "alpha_vantage" in self.api_keys:
                response = requests.get(
                    self.alternative_url,
                    params={
                        "function": "FX_DAILY",
                        "from_symbol": base_currency,
                        "to_symbol": quote_currency,
                        "outputsize": "full",
                        "apikey": self.api_keys["alpha_vantage"]
                    },
                    timeout=15
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data and "Time Series FX (Daily)" in data:
                        return self._parse_alpha_vantage_historical(data, start_date)
            
            self.logger.warning(f"Could not fetch historical rates for {base_currency}/{quote_currency} from any API")
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching historical rates for {base_currency}/{quote_currency}: {e}")
            return []
    
    def _parse_exchangerate_historical(self, data: Dict[str, Any], quote_currency: str) -> List[Dict[str, Any]]:
        """Parse response from exchangerate.host API for historical data."""
        try:
            historical_data = []
            
            for date, rates in data.get("rates", {}).items():
                if quote_currency in rates:
                    historical_data.append({
                        "date": date,
                        "rate": float(rates[quote_currency])
                    })
            
            # Sort by date (oldest first)
            historical_data.sort(key=lambda x: x["date"])
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error parsing exchangerate historical API response: {e}")
            return []
    
    def _parse_alpha_vantage_historical(self, data: Dict[str, Any], start_date: datetime) -> List[Dict[str, Any]]:
        """Parse response from Alpha Vantage API for historical data."""
        try:
            time_series = data.get("Time Series FX (Daily)", {})
            historical_data = []
            
            for date, values in time_series.items():
                # Check if date is after start_date
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                if date_obj >= start_date:
                    historical_data.append({
                        "date": date,
                        "rate": float(values.get("4. close", 0))
                    })
            
            # Sort by date (oldest first)
            historical_data.sort(key=lambda x: x["date"])
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error parsing Alpha Vantage historical response: {e}")
            return []
    
    def _save_historical_rates(self, pair_code: str, description: str, category: str,
                              base_currency: str, quote_currency: str,
                              historical_data: List[Dict[str, Any]]) -> bool:
        """
        Save historical exchange rates to database.
        
        Args:
            pair_code: Currency pair code
            description: Description of the currency pair
            category: Category of the currency pair
            base_currency: Base currency code
            quote_currency: Quote currency code
            historical_data: List of historical exchange rate data points
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare documents for bulk insert
            documents = []
            
            for data_point in historical_data:
                document = {
                    "pair_code": pair_code,
                    "description": description,
                    "category": category,
                    "base_currency": base_currency,
                    "quote_currency": quote_currency,
                    "date": data_point["date"],
                    "rate": data_point["rate"],
                    "collection_timestamp": datetime.now()
                }
                
                documents.append(document)
            
            if documents:
                # Check if we need to update or insert
                existing_dates = set()
                
                existing = self.db.forex_historical_collection.find(
                    {"pair_code": pair_code},
                    {"date": 1}
                )
                
                for doc in existing:
                    existing_dates.add(doc.get("date"))
                
                # Filter out documents with dates that already exist
                new_documents = [doc for doc in documents if doc["date"] not in existing_dates]
                
                if new_documents:
                    self.db.forex_historical_collection.insert_many(new_documents)
                    self.logger.info(f"Inserted {len(new_documents)} new historical rates for {pair_code}")
                else:
                    self.logger.info(f"No new historical rates for {pair_code}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving historical rates for {pair_code}: {e}")
            return False
    
    def get_latest_rates(self) -> Dict[str, Any]:
        """
        Get the latest exchange rates for all tracked currency pairs.
        
        Returns:
            Dictionary containing the latest exchange rates
        """
        try:
            results = {
                "timestamp": datetime.now(),
                "pairs": {}
            }
            
            # Get latest rates for each category/pair
            for category, pairs in self.currency_pairs.items():
                results["pairs"][category] = {}
                
                for pair_code in pairs.keys():
                    # Get latest rate from database
                    latest = self.db.forex_rates_collection.find_one(
                        {"pair_code": pair_code},
                        sort=[("collection_timestamp", -1)]
                    )
                    
                    if latest:
                        results["pairs"][category][pair_code] = {
                            "description": latest.get("description", ""),
                            "base": latest.get("base_currency", ""),
                            "quote": latest.get("quote_currency", ""),
                            "rate": latest.get("rate", 0),
                            "timestamp": latest.get("collection_timestamp")
                        }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting latest exchange rates: {e}")
            return {"error": str(e)}
    
    def calculate_volatility(self, days: int = 30) -> Dict[str, Any]:
        """
        Calculate exchange rate volatility for all currency pairs.
        
        Args:
            days: Number of days for volatility calculation
            
        Returns:
            Dictionary containing volatility data
        """
        try:
            results = {
                "timestamp": datetime.now(),
                "period_days": days,
                "pairs": {}
            }
            
            # Get volatility for each category/pair
            for category, pairs in self.currency_pairs.items():
                results["pairs"][category] = {}
                
                for pair_code, pair_info in pairs.items():
                    # Get historical rates from database
                    historical = list(self.db.forex_historical_collection.find(
                        {"pair_code": pair_code},
                        sort=[("date", -1)]
                    ).limit(days + 1))  # Need n+1 points to get n returns
                    
                    if len(historical) > 10:  # Need enough data points
                        # Calculate daily returns
                        rates = [h["rate"] for h in reversed(historical)]  # Oldest first
                        returns = []
                        
                        for i in range(1, len(rates)):
                            daily_return = (rates[i] / rates[i-1]) - 1
                            returns.append(daily_return)
                        
                        # Calculate volatility (annualized)
                        volatility = np.std(returns) * np.sqrt(252)  # Annualized
                        
                        # Calculate other metrics
                        max_return = max(returns)
                        min_return = min(returns)
                        range_pct = max(rates) / min(rates) - 1
                        
                        results["pairs"][category][pair_code] = {
                            "description": pair_info["description"],
                            "volatility_annualized": round(volatility * 100, 2),  # As percentage
                            "max_daily_change_pct": round(max_return * 100, 2),
                            "min_daily_change_pct": round(min_return * 100, 2),
                            "range_pct": round(range_pct * 100, 2),
                            "latest_rate": rates[-1],
                            "data_points": len(returns)
                        }
            
            # Save volatility data to database
            self._save_volatility(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility: {e}")
            return {"error": str(e)}
    
    def _save_volatility(self, volatility_data: Dict[str, Any]) -> bool:
        """
        Save volatility data to database.
        
        Args:
            volatility_data: Volatility data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.db.forex_volatility_collection.insert_one(volatility_data)
            self.logger.info("Saved forex volatility data to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving volatility data: {e}")
            return False
    
    def calculate_correlations(self, base_pair: str = "USDINR", days: int = 60) -> Dict[str, Any]:
        """
        Calculate correlations between a base currency pair and other pairs.
        
        Args:
            base_pair: Base currency pair code to compare
            days: Number of days for correlation calculation
            
        Returns:
            Dictionary containing correlation data
        """
        try:
            # Get historical data for base pair
            base_data = list(self.db.forex_historical_collection.find(
                {"pair_code": base_pair},
                sort=[("date", -1)]
            ).limit(days))
            
            if not base_data:
                return {"error": f"No historical data found for {base_pair}"}
            
            # Create DataFrame for base pair
            base_df = pd.DataFrame(base_data)
            
            # Dictionary to store correlations
            correlations = {
                "base_pair": base_pair,
                "period_days": days,
                "timestamp": datetime.now(),
                "correlations": {}
            }
            
            # Process each category
            for category, pairs in self.currency_pairs.items():
                correlations["correlations"][category] = {}
                
                for pair_code in pairs.keys():
                    # Skip the base pair
                    if pair_code == base_pair:
                        continue
                    
                    # Get historical data for this pair
                    pair_data = list(self.db.forex_historical_collection.find(
                        {"pair_code": pair_code},
                        sort=[("date", -1)]
                    ).limit(days))
                    
                    if pair_data:
                        # Create DataFrame
                        pair_df = pd.DataFrame(pair_data)
                        
                        # Ensure we have matching dates
                        merged_df = pd.merge(
                            base_df[["date", "rate"]],
                            pair_df[["date", "rate"]],
                            on="date",
                            how="inner",
                            suffixes=("_base", "_pair")
                        )
                        
                        if len(merged_df) > 10:  # Need enough data points
                            # Calculate correlation of rates
                            rate_corr = merged_df["rate_base"].corr(merged_df["rate_pair"])
                            
                            # Calculate returns
                            merged_df["return_base"] = merged_df["rate_base"].pct_change()
                            merged_df["return_pair"] = merged_df["rate_pair"].pct_change()
                            
                            # Calculate correlation of returns
                            return_corr = merged_df["return_base"].corr(merged_df["return_pair"])
                            
                            correlations["correlations"][category][pair_code] = {
                                "description": pairs[pair_code]["description"],
                                "rate_correlation": round(rate_corr, 4),
                                "return_correlation": round(return_corr, 4),
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
            self.db.forex_correlations_collection.insert_one(correlations)
            self.logger.info("Saved forex correlations to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving correlations: {e}")
            return False
    
    def get_currency_strength_index(self, days: int = 14) -> Dict[str, Any]:
        """
        Calculate a Currency Strength Index for major currencies.
        
        The index measures the relative strength of each currency against a basket
        of other major currencies based on recent performance.
        
        Args:
            days: Number of days for strength calculation
            
        Returns:
            Dictionary containing currency strength data
        """
        try:
            # Get unique currencies from all pairs
            currencies = set()
            for category in self.currency_pairs.values():
                for pair_info in category.values():
                    currencies.add(pair_info["base"])
                    currencies.add(pair_info["quote"])
            
            # Calculate strength for each currency
            strength_data = {
                "timestamp": datetime.now(),
                "period_days": days,
                "currencies": {}
            }
            
            for currency in currencies:
                # Get all pairs involving this currency
                base_pairs = []  # Currency as base (e.g., USD in USDINR)
                quote_pairs = []  # Currency as quote (e.g., INR in USDINR)
                
                for category, pairs in self.currency_pairs.items():
                    for pair_code, pair_info in pairs.items():
                        if pair_info["base"] == currency:
                            base_pairs.append(pair_code)
                        elif pair_info["quote"] == currency:
                            quote_pairs.append(pair_code)
                
                # Calculate performance in each pair
                performance = []
                
                # For base pairs, higher rate means stronger base currency
                for pair in base_pairs:
                    historical = list(self.db.forex_historical_collection.find(
                        {"pair_code": pair},
                        sort=[("date", -1)]
                    ).limit(days + 1))
                    
                    if len(historical) > 5:  # Need enough data points
                        oldest = historical[-1]["rate"]
                        latest = historical[0]["rate"]
                        percent_change = (latest / oldest - 1) * 100
                        performance.append(percent_change)
                
                # For quote pairs, lower rate means stronger quote currency
                for pair in quote_pairs:
                    historical = list(self.db.forex_historical_collection.find(
                        {"pair_code": pair},
                        sort=[("date", -1)]
                    ).limit(days + 1))
                    
                    if len(historical) > 5:  # Need enough data points
                        oldest = historical[-1]["rate"]
                        latest = historical[0]["rate"]
                        percent_change = (1 - latest / oldest) * 100  # Inverse for quote currency
                        performance.append(percent_change)
                
                # Calculate average performance (strength index)
                if performance:
                    average_performance = sum(performance) / len(performance)
                    
                    strength_data["currencies"][currency] = {
                        "strength_index": round(average_performance, 2),
                        "pairs_count": len(performance)
                    }
            
            # Normalize strength values to a 0-100 scale
            if strength_data["currencies"]:
                values = [data["strength_index"] for data in strength_data["currencies"].values()]
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val
                
                if range_val > 0:
                    for currency in strength_data["currencies"]:
                        original = strength_data["currencies"][currency]["strength_index"]
                        normalized = 100 * (original - min_val) / range_val
                        strength_data["currencies"][currency]["normalized_index"] = round(normalized, 2)
            
            # Save strength data to database
            # Save strength data to database
            self.db.forex_strength_collection.insert_one(strength_data)
            
            return strength_data
            
        except Exception as e:
            self.logger.error(f"Error calculating currency strength index: {e}")
            return {"error": str(e)}
    
    def run_daily_collection(self) -> Dict[str, Any]:
        """
        Run daily collection of forex data.
        
        Returns:
            Dictionary containing collection results
        """
        self.logger.info("Running daily collection of forex data")
        
        results = {
            "timestamp": datetime.now(),
            "current_rates": None,
            "historical_update": None,
            "volatility": None,
            "correlations": None,
            "strength_index": None
        }
        
        try:
            # 1. Collect current rates
            current_rates = self.collect_current_rates()
            results["current_rates"] = {
                "status": "success" if current_rates else "failed",
                "count": sum(len(pairs) for category, pairs in current_rates.get("pairs", {}).items())
            }
            
            # 2. Update historical rates (just the latest day)
            historical_update = self.collect_historical_rates(days=7)  # Get recent days to ensure we have latest
            results["historical_update"] = {
                "status": "success" if historical_update else "failed",
                "count": sum(pairs.get(pair_code, {}).get("data_points", 0) 
                           for category, pairs in historical_update.get("pairs", {}).items() 
                           for pair_code in pairs)
            }
            
            # 3. Calculate volatility
            if current_rates:
                volatility = self.calculate_volatility()
                results["volatility"] = {
                    "status": "success" if "error" not in volatility else "failed",
                    "period_days": volatility.get("period_days")
                }
            
            # 4. Calculate correlations (with USDINR as base)
            if current_rates:
                correlations = self.calculate_correlations()
                results["correlations"] = {
                    "status": "success" if "error" not in correlations else "failed",
                    "base_pair": correlations.get("base_pair")
                }
            
            # 5. Calculate currency strength index
            if current_rates:
                strength_index = self.get_currency_strength_index()
                results["strength_index"] = {
                    "status": "success" if "error" not in strength_index else "failed",
                    "currency_count": len(strength_index.get("currencies", {}))
                }
            
            self.logger.info("Daily collection of forex data completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in daily collection: {e}")
            results["error"] = str(e)
        
        return results

    def generate_forex_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive forex market report.
        
        Returns:
            Dictionary containing forex market report
        """
        try:
            report = {
                "timestamp": datetime.now(),
                "title": "Forex Market Daily Report",
                "sections": {}
            }
            
            # 1. INR Performance
            inr_performance = {}
            
            # Get latest rates for INR pairs
            latest_rates = self.get_latest_rates()
            if "pairs" in latest_rates and "INR" in latest_rates["pairs"]:
                inr_pairs = latest_rates["pairs"]["INR"]
                
                # Get rate changes (1-day)
                for pair_code, pair_data in inr_pairs.items():
                    # Get yesterday's rate
                    yesterday_rate = self.db.forex_rates_collection.find_one(
                        {
                            "pair_code": pair_code,
                            "collection_timestamp": {"$lt": datetime.now() - timedelta(hours=12)}
                        },
                        sort=[("collection_timestamp", -1)]
                    )
                    
                    if yesterday_rate:
                        current_rate = pair_data["rate"]
                        prev_rate = yesterday_rate["rate"]
                        
                        # Calculate change
                        change_pct = (current_rate / prev_rate - 1) * 100
                        
                        inr_performance[pair_code] = {
                            "description": pair_data["description"],
                            "current_rate": current_rate,
                            "previous_rate": prev_rate,
                            "change_percent": round(change_pct, 2)
                        }
            
            report["sections"]["inr_performance"] = {
                "title": "INR Performance",
                "data": inr_performance
            }
            
            # 2. Major Currency Movements
            major_movements = {}
            
            # Get latest rates for major pairs
            if "pairs" in latest_rates and "Major" in latest_rates["pairs"]:
                major_pairs = latest_rates["pairs"]["Major"]
                
                # Get rate changes (1-day)
                for pair_code, pair_data in major_pairs.items():
                    # Get yesterday's rate
                    yesterday_rate = self.db.forex_rates_collection.find_one(
                        {
                            "pair_code": pair_code,
                            "collection_timestamp": {"$lt": datetime.now() - timedelta(hours=12)}
                        },
                        sort=[("collection_timestamp", -1)]
                    )
                    
                    if yesterday_rate:
                        current_rate = pair_data["rate"]
                        prev_rate = yesterday_rate["rate"]
                        
                        # Calculate change
                        change_pct = (current_rate / prev_rate - 1) * 100
                        
                        major_movements[pair_code] = {
                            "description": pair_data["description"],
                            "current_rate": current_rate,
                            "previous_rate": prev_rate,
                            "change_percent": round(change_pct, 2)
                        }
            
            report["sections"]["major_movements"] = {
                "title": "Major Currency Movements",
                "data": major_movements
            }
            
            # 3. Currency Strength
            strength_data = self.db.forex_strength_collection.find_one(
                sort=[("timestamp", -1)]
            )
            
            if strength_data and "currencies" in strength_data:
                # Sort currencies by strength (strongest first)
                sorted_currencies = sorted(
                    [(currency, data) for currency, data in strength_data["currencies"].items()],
                    key=lambda x: x[1].get("normalized_index", 0),
                    reverse=True
                )
                
                currency_strength = {
                    currency: {
                        "normalized_index": data.get("normalized_index", 0),
                        "strength_index": data.get("strength_index", 0)
                    }
                    for currency, data in sorted_currencies
                }
                
                report["sections"]["currency_strength"] = {
                    "title": "Currency Strength Index",
                    "period_days": strength_data.get("period_days", 14),
                    "data": currency_strength
                }
            
            # 4. INR Correlations
            correlation_data = self.db.forex_correlations_collection.find_one(
                {"base_pair": "USDINR"},
                sort=[("timestamp", -1)]
            )
            
            if correlation_data and "correlations" in correlation_data:
                inr_correlations = {}
                
                # Extract correlations with USDINR
                for category, pairs in correlation_data["correlations"].items():
                    for pair_code, pair_data in pairs.items():
                        inr_correlations[pair_code] = {
                            "description": pair_data.get("description", ""),
                            "rate_correlation": pair_data.get("rate_correlation", 0),
                            "return_correlation": pair_data.get("return_correlation", 0)
                        }
                
                # Sort by absolute correlation (highest first)
                sorted_correlations = {
                    pair: data
                    for pair, data in sorted(
                        inr_correlations.items(),
                        key=lambda x: abs(x[1]["return_correlation"]),
                        reverse=True
                    )
                }
                
                report["sections"]["inr_correlations"] = {
                    "title": "INR Correlations",
                    "base_pair": "USDINR",
                    "period_days": correlation_data.get("period_days", 60),
                    "data": sorted_correlations
                }
            
            # 5. Market Volatility
            volatility_data = self.db.forex_volatility_collection.find_one(
                sort=[("timestamp", -1)]
            )
            
            if volatility_data and "pairs" in volatility_data:
                # Combine all pairs and sort by volatility
                all_pairs_volatility = {}
                
                for category, pairs in volatility_data["pairs"].items():
                    for pair_code, pair_data in pairs.items():
                        all_pairs_volatility[pair_code] = {
                            "description": pair_data.get("description", ""),
                            "volatility_annualized": pair_data.get("volatility_annualized", 0),
                            "range_pct": pair_data.get("range_pct", 0)
                        }
                
                # Sort by volatility (highest first)
                sorted_volatility = {
                    pair: data
                    for pair, data in sorted(
                        all_pairs_volatility.items(),
                        key=lambda x: x[1]["volatility_annualized"],
                        reverse=True
                    )
                }
                
                report["sections"]["market_volatility"] = {
                    "title": "Market Volatility",
                    "period_days": volatility_data.get("period_days", 30),
                    "data": sorted_volatility
                }
            
            # Save report to database
            self.db.forex_reports_collection.insert_one(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating forex report: {e}")
            return {"error": str(e)}


# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize collector
    forex_collector = ForexCollector(db)
    
    # Run daily collection
    results = forex_collector.run_daily_collection()
    print(json.dumps(results, default=str, indent=2))
    
    # Generate forex report
    report = forex_collector.generate_forex_report()
    print(json.dumps(report, default=str, indent=2))