"""
Google Trends Analyzer

This module analyzes Google search trends for stocks and market-related terms.
It provides insights into search interest that can indicate market movements.
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from pytrends.request import TrendReq
import requests
from pymongo import UpdateOne
import re

class GoogleTrends:
    """
    Analyzes Google search trends for stocks and market-related terms.
    Provides insights into public interest and potential market movements.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the Google Trends analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Initialize PyTrends client
        self.pytrends = self._init_pytrends()
        
        # Default timeframes for different analyses
        self.timeframes = {
            "short_term": "now 7-d",     # Last 7 days (hourly data)
            "medium_term": "today 3-m",  # Last 3 months (daily data)
            "long_term": "today 12-m"    # Last 12 months (weekly data)
        }
        
        # Default regions
        self.regions = {
            "india": "IN",               # India
            "global": ""                 # Global search interest
        }
        
        # Related search terms for context
        self.related_market_terms = {
            "stock": ["stock market", "investing", "share market", "trading"],
            "finance": ["finance", "stock price", "dividend", "earnings"],
            "economy": ["economy", "gdp", "recession", "inflation"],
            "bear_market": ["bear market", "market crash", "stock market crash", "sell stocks"],
            "bull_market": ["bull market", "market rally", "buy stocks", "stock market high"]
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
    
    def _init_pytrends(self) -> TrendReq:
        """Initialize PyTrends client."""
        try:
            # Initialize with default parameters
            client = TrendReq(hl='en-US', tz=330)  # India timezone (UTC+5:30)
            return client
            
        except Exception as e:
            self.logger.error(f"Error initializing PyTrends client: {e}")
            return None
    
    def analyze_stock_interest(self, symbol: str, company_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze Google search interest for a specific stock.
        
        Args:
            symbol: Stock symbol
            company_name: Full company name (optional)
            
        Returns:
            Dictionary containing search interest analysis
        """
        self.logger.info(f"Analyzing Google Trends for {symbol}")
        
        # If company name not provided, try to get it from database
        if not company_name:
            stock_info = self.db.portfolio_collection.find_one({"symbol": symbol})
            if stock_info and "company_name" in stock_info:
                company_name = stock_info["company_name"]
        
        # Clean symbol (remove exchange suffixes like .NS)
        clean_symbol = symbol.split('.')[0]
        
        # Prepare search keywords
        search_terms = []
        if clean_symbol:
            search_terms.append(clean_symbol)
        if company_name:
            search_terms.append(company_name)
        
        if not search_terms:
            return {"error": "No valid search terms provided"}
        
        # Analyze search trends for different timeframes
        results = {
            "symbol": symbol,
            "company_name": company_name,
            "timestamp": datetime.now(),
            "search_terms": search_terms,
            "trends": {}
        }
        
        # Analyze each search term separately
        for term in search_terms:
            term_results = {}
            
            # Analyze for different timeframes
            for timeframe_name, timeframe in self.timeframes.items():
                # Analyze for India
                india_data = self._get_interest_over_time(term, timeframe, self.regions["india"])
                
                if india_data.get("error"):
                    self.logger.warning(f"Error getting India trends for {term} ({timeframe_name}): {india_data['error']}")
                else:
                    term_results[f"{timeframe_name}_india"] = india_data
                
                # Rate limiting to prevent API blocks
                time.sleep(1)
                
                # Analyze for global
                global_data = self._get_interest_over_time(term, timeframe, self.regions["global"])
                
                if global_data.get("error"):
                    self.logger.warning(f"Error getting global trends for {term} ({timeframe_name}): {global_data['error']}")
                else:
                    term_results[f"{timeframe_name}_global"] = global_data
                
                # Rate limiting to prevent API blocks
                time.sleep(1)
            
            # Get related queries for the term
            related_queries = self._get_related_queries(term, self.timeframes["medium_term"], self.regions["india"])
            if not related_queries.get("error"):
                term_results["related_queries"] = related_queries
            
            # Rate limiting to prevent API blocks
            time.sleep(1)
            
            # Add term results to overall results
            results["trends"][term] = term_results
        
        # Calculate overall trend metrics
        results["trend_metrics"] = self._calculate_trend_metrics(results["trends"])
        
        # Save to database
        self._save_trends_data(results)
        
        return results
    
    def _get_interest_over_time(self, keyword: str, timeframe: str, geo: str) -> Dict[str, Any]:
        """
        Get interest over time for a keyword.
        
        Args:
            keyword: Search keyword
            timeframe: Time range (e.g., 'now 7-d', 'today 3-m')
            geo: Geographic region (e.g., 'IN' for India, '' for global)
            
        Returns:
            Dictionary containing interest over time data
        """
        try:
            if not self.pytrends:
                return {"error": "PyTrends client not initialized"}
            
            # Build payload
            self.pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo)
            
            # Get interest over time
            interest_over_time_df = self.pytrends.interest_over_time()
            
            # If no data returned
            if interest_over_time_df.empty:
                return {"error": "No data returned for this query"}
            
            # Extract data
            dates = interest_over_time_df.index.strftime("%Y-%m-%d").tolist()
            values = interest_over_time_df[keyword].tolist()
            
            # Calculate metrics
            avg_interest = np.mean(values)
            max_interest = np.max(values)
            min_interest = np.min(values)
            latest_interest = values[-1] if values else 0
            
            # Calculate trend direction
            if len(values) >= 2:
                # Linear regression to determine trend
                x = np.array(range(len(values)))
                slope, _ = np.polyfit(x, values, 1)
                
                # Determine direction based on slope
                if slope > 0.1:
                    trend_direction = "increasing"
                elif slope < -0.1:
                    trend_direction = "decreasing"
                else:
                    trend_direction = "stable"
                
                # Calculate percentage change
                first_val = values[0]
                last_val = values[-1]
                
                if first_val > 0:
                    pct_change = ((last_val - first_val) / first_val) * 100
                else:
                    pct_change = 0
            else:
                trend_direction = "unknown"
                pct_change = 0
                slope = 0
            
            return {
                "dates": dates,
                "values": values,
                "metrics": {
                    "average": avg_interest,
                    "maximum": max_interest,
                    "minimum": min_interest,
                    "latest": latest_interest,
                    "trend_direction": trend_direction,
                    "percent_change": pct_change,
                    "slope": slope
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting interest over time for {keyword}: {e}")
            return {"error": str(e)}
    
    def _get_related_queries(self, keyword: str, timeframe: str, geo: str) -> Dict[str, Any]:
        """
        Get related queries for a keyword.
        
        Args:
            keyword: Search keyword
            timeframe: Time range (e.g., 'now 7-d', 'today 3-m')
            geo: Geographic region (e.g., 'IN' for India, '' for global)
            
        Returns:
            Dictionary containing related queries
        """
        try:
            if not self.pytrends:
                return {"error": "PyTrends client not initialized"}
            
            # Build payload
            self.pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo=geo)
            
            # Get related queries
            related_queries = self.pytrends.related_queries()
            
            if keyword not in related_queries or not related_queries[keyword]:
                return {"error": "No related queries returned"}
            
            top_queries = []
            rising_queries = []
            
            # Extract top queries
            if 'top' in related_queries[keyword] and not related_queries[keyword]['top'].empty:
                top_df = related_queries[keyword]['top']
                for _, row in top_df.iterrows():
                    top_queries.append({
                        "query": row['query'],
                        "value": row['value']
                    })
            
            # Extract rising queries
            if 'rising' in related_queries[keyword] and not related_queries[keyword]['rising'].empty:
                rising_df = related_queries[keyword]['rising']
                for _, row in rising_df.iterrows():
                    rising_queries.append({
                        "query": row['query'],
                        "value": row['value']
                    })
            
            return {
                "top": top_queries,
                "rising": rising_queries
            }
            
        except Exception as e:
            self.logger.error(f"Error getting related queries for {keyword}: {e}")
            return {"error": str(e)}
    
    def _calculate_trend_metrics(self, trends_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall trend metrics from collected data.
        
        Args:
            trends_data: Trends data by search term
            
        Returns:
            Dictionary containing trend metrics
        """
        # Extract short-term and medium-term metrics
        short_term_metrics = []
        medium_term_metrics = []
        
        for term, term_data in trends_data.items():
            # Short-term India trends
            if "short_term_india" in term_data and "metrics" in term_data["short_term_india"]:
                metrics = term_data["short_term_india"]["metrics"]
                short_term_metrics.append(metrics)
            
            # Medium-term India trends
            if "medium_term_india" in term_data and "metrics" in term_data["medium_term_india"]:
                metrics = term_data["medium_term_india"]["metrics"]
                medium_term_metrics.append(metrics)
        
        # Calculate averages if data exists
        result = {}
        
        if short_term_metrics:
            # Average latest interest and trend direction
            avg_latest = np.mean([m["latest"] for m in short_term_metrics])
            
            # Count trend directions
            directions = [m["trend_direction"] for m in short_term_metrics]
            direction_counts = {d: directions.count(d) for d in set(directions)}
            
            # Get most common direction
            most_common_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
            
            result["short_term"] = {
                "average_latest_interest": avg_latest,
                "trend_direction": most_common_direction,
                "direction_confidence": direction_counts[most_common_direction] / len(directions)
            }
        
        if medium_term_metrics:
            # Average latest interest and trend direction
            avg_latest = np.mean([m["latest"] for m in medium_term_metrics])
            
            # Count trend directions
            directions = [m["trend_direction"] for m in medium_term_metrics]
            direction_counts = {d: directions.count(d) for d in set(directions)}
            
            # Get most common direction
            most_common_direction = max(direction_counts.items(), key=lambda x: x[1])[0]
            
            result["medium_term"] = {
                "average_latest_interest": avg_latest,
                "trend_direction": most_common_direction,
                "direction_confidence": direction_counts[most_common_direction] / len(directions)
            }
        
        # Calculate overall trend status
        if "short_term" in result and "medium_term" in result:
            short_direction = result["short_term"]["trend_direction"]
            medium_direction = result["medium_term"]["trend_direction"]
            
            if short_direction == "increasing" and medium_direction == "increasing":
                overall_status = "strong_uptrend"
            elif short_direction == "decreasing" and medium_direction == "decreasing":
                overall_status = "strong_downtrend"
            elif short_direction == "increasing" and medium_direction != "increasing":
                overall_status = "recent_uptrend"
            elif short_direction == "decreasing" and medium_direction != "decreasing":
                overall_status = "recent_downtrend"
            else:
                overall_status = "mixed"
            
            result["overall_status"] = overall_status
        
        return result
    
    def _save_trends_data(self, trends_data: Dict[str, Any]) -> bool:
        """
        Save trends data to database.
        
        Args:
            trends_data: Trends data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.db.google_trends_collection.insert_one(trends_data)
            self.logger.info(f"Saved Google Trends data for {trends_data['symbol']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving trends data: {e}")
            return False
    
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """
        Analyze Google search trends for market sentiment indicators.
        
        Returns:
            Dictionary containing market sentiment analysis
        """
        self.logger.info("Analyzing market sentiment from Google Trends")
        
        results = {
            "timestamp": datetime.now(),
            "categories": {},
            "sentiment_indicators": {}
        }
        
        # Analyze each category of related terms
        for category, terms in self.related_market_terms.items():
            category_results = {}
            
            for term in terms:
                # Analyze for medium-term India trends
                term_data = self._get_interest_over_time(term, self.timeframes["medium_term"], self.regions["india"])
                
                if not term_data.get("error"):
                    category_results[term] = term_data
                
                # Rate limiting to prevent API blocks
                time.sleep(1)
            
            if category_results:
                results["categories"][category] = category_results
        
        # Calculate sentiment indicators
        if results["categories"]:
            # Bull/Bear Ratio (compare bull_market vs bear_market terms)
            if "bull_market" in results["categories"] and "bear_market" in results["categories"]:
                bull_values = []
                bear_values = []
                
                # Collect values for bull market terms
                for term, data in results["categories"]["bull_market"].items():
                    if "metrics" in data and "latest" in data["metrics"]:
                        bull_values.append(data["metrics"]["latest"])
                
                # Collect values for bear market terms
                for term, data in results["categories"]["bear_market"].items():
                    if "metrics" in data and "latest" in data["metrics"]:
                        bear_values.append(data["metrics"]["latest"])
                
                # Calculate ratio
                if bull_values and bear_values:
                    avg_bull = np.mean(bull_values)
                    avg_bear = np.mean(bear_values)
                    
                    if avg_bear > 0:
                        bull_bear_ratio = avg_bull / avg_bear
                    else:
                        bull_bear_ratio = float('inf') if avg_bull > 0 else 1.0
                    
                    results["sentiment_indicators"]["bull_bear_ratio"] = bull_bear_ratio
                    
                    # Interpret ratio
                    if bull_bear_ratio > 1.5:
                        results["sentiment_indicators"]["market_sentiment"] = "bullish"
                    elif bull_bear_ratio < 0.7:
                        results["sentiment_indicators"]["market_sentiment"] = "bearish"
                    else:
                        results["sentiment_indicators"]["market_sentiment"] = "neutral"
            
            # Trend analysis for "stock market" term
            stock_market_trend = self._get_interest_over_time("stock market", self.timeframes["medium_term"], self.regions["india"])
            
            if not stock_market_trend.get("error"):
                results["sentiment_indicators"]["stock_market_interest"] = stock_market_trend
            
            # Compare with other economic indicators
            recession_trend = self._get_interest_over_time("recession", self.timeframes["medium_term"], self.regions["india"])
            if not recession_trend.get("error"):
                results["sentiment_indicators"]["recession_interest"] = recession_trend["metrics"] if "metrics" in recession_trend else {}
            
            # Anxiety index (terms like "market crash", "bear market", "recession")
            anxiety_terms = ["market crash", "bear market", "recession"]
            anxiety_values = []
            
            for term in anxiety_terms:
                term_data = self._get_interest_over_time(term, self.timeframes["short_term"], self.regions["india"])
                
                if not term_data.get("error") and "metrics" in term_data and "latest" in term_data["metrics"]:
                    anxiety_values.append(term_data["metrics"]["latest"])
            
            if anxiety_values:
                anxiety_index = np.mean(anxiety_values)
                results["sentiment_indicators"]["anxiety_index"] = anxiety_index
                
                # Interpret anxiety index
                if anxiety_index > 75:
                    results["sentiment_indicators"]["anxiety_level"] = "extreme"
                elif anxiety_index > 50:
                    results["sentiment_indicators"]["anxiety_level"] = "high"
                elif anxiety_index > 25:
                    results["sentiment_indicators"]["anxiety_level"] = "moderate"
                else:
                    results["sentiment_indicators"]["anxiety_level"] = "low"
        
        # Save to database
        try:
            self.db.google_trends_market_sentiment_collection.insert_one(results)
            self.logger.info("Saved market sentiment analysis")
        except Exception as e:
            self.logger.error(f"Error saving market sentiment analysis: {e}")
        
        return results
    
    def get_trends_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical Google Trends data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
            
        Returns:
            Dictionary containing historical trends data
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Query database for historical trends
            historical_data = list(self.db.google_trends_collection.find(
                {
                    "symbol": symbol,
                    "timestamp": {"$gte": start_date}
                },
                {"_id": 0}
            ).sort("timestamp", 1))
            
            # Process data
            if historical_data:
                # Extract dates and trend metrics
                dates = [d["timestamp"].strftime("%Y-%m-%d") for d in historical_data]
                
                # Extract trend metrics if available
                interest_values = []
                trend_status = []
                
                for record in historical_data:
                    if "trend_metrics" in record and "short_term" in record["trend_metrics"]:
                        interest_values.append(record["trend_metrics"]["short_term"].get("average_latest_interest", 0))
                        trend_status.append(record["trend_metrics"].get("overall_status", "unknown"))
                
                # Determine overall trend
                if trend_status:
                    trend_counts = {s: trend_status.count(s) for s in set(trend_status)}
                    most_common_trend = max(trend_counts.items(), key=lambda x: x[1])[0]
                else:
                    most_common_trend = "unknown"
                
                return {
                    "symbol": symbol,
                    "days": days,
                    "data_points": len(historical_data),
                    "trend_status": most_common_trend,
                    "latest_data": historical_data[-1] if historical_data else {},
                    "historical_summary": {
                        "dates": dates,
                        "interest_values": interest_values,
                        "trend_status": trend_status
                    }
                }
            
            return {"symbol": symbol, "days": days, "data_points": 0}
            
        except Exception as e:
            self.logger.error(f"Error getting trends history for {symbol}: {e}")
            return {"error": str(e)}
    
    def run_daily_collection(self) -> Dict[str, Any]:
        """
        Run daily collection of Google Trends data for all portfolio symbols.
        
        Returns:
            Dictionary containing collection results
        """
        self.logger.info("Running daily collection of Google Trends data")
        
        results = {
            "timestamp": datetime.now(),
            "symbols_analyzed": 0,
            "successful": 0,
            "failed": 0,
            "details": {},
            "market_sentiment": None
        }
        
        try:
            # Get all active symbols from portfolio
            portfolio_symbols = list(self.db.portfolio_collection.find(
                {"status": "active"},
                {"symbol": 1, "company_name": 1}
            ))
            
            results["symbols_analyzed"] = len(portfolio_symbols)
            
            for stock in portfolio_symbols:
                symbol = stock.get("symbol", "")
                company_name = stock.get("company_name", "")
                
                if symbol:
                    try:
                        # Analyze trends
                        trends_result = self.analyze_stock_interest(symbol, company_name)
                        
                        if "error" not in trends_result:
                            results["successful"] += 1
                            results["details"][symbol] = "success"
                        else:
                            results["failed"] += 1
                            results["details"][symbol] = trends_result["error"]
                        
                        # Respect rate limits
                        time.sleep(5)  # Longer delay for Google Trends API
                        
                    except Exception as e:
                        results["failed"] += 1
                        results["details"][symbol] = str(e)
                        self.logger.error(f"Error analyzing trends for {symbol}: {e}")
            
            # Analyze market sentiment
            try:
                market_sentiment = self.analyze_market_sentiment()
                if "error" not in market_sentiment:
                    results["market_sentiment"] = "success"
                else:
                    results["market_sentiment"] = market_sentiment["error"]
            except Exception as e:
                results["market_sentiment"] = str(e)
                self.logger.error(f"Error analyzing market sentiment: {e}")
            
            self.logger.info(f"Daily trends collection completed: {results['successful']} successful, {results['failed']} failed")
            
        except Exception as e:
            self.logger.error(f"Error in daily trends collection: {e}")
            results["error"] = str(e)
        
        return results
    
    def compare_search_to_price(self, symbol: str, days: int = 90) -> Dict[str, Any]:
        """
        Compare Google search trends to stock price movements.
        
        Args:
            symbol: Stock symbol
            days: Number of days for comparison
            
        Returns:
            Dictionary containing comparison analysis
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Get search trends data
            trends_data = list(self.db.google_trends_collection.find(
                {
                    "symbol": symbol,
                    "timestamp": {"$gte": start_date}
                }
            ).sort("timestamp", 1))
            
            # Get price data
            price_data = list(self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "timeframe": "day",
                    "timestamp": {"$gte": start_date}
                }
            ).sort("timestamp", 1))
            
            if not trends_data or not price_data:
                return {
                    "symbol": symbol,
                    "days": days,
                    "error": "Insufficient data for comparison"
                }
            
            # Extract all the search terms that have data
            search_terms_data = {}
            
            for record in trends_data:
                if "trends" in record:
                    for term, term_data in record["trends"].items():
                        # Check if we have medium term India data
                        if "medium_term_india" in term_data and "values" in term_data["medium_term_india"]:
                            if term not in search_terms_data:
                                search_terms_data[term] = {
                                    "dates": term_data["medium_term_india"]["dates"],
                                    "values": term_data["medium_term_india"]["values"]
                                }
            
            # Extract price data
            price_dates = [d["timestamp"].strftime("%Y-%m-%d") for d in price_data]
            closing_prices = [d["close"] for d in price_data]
            
            # Analyze correlations for each search term
            correlations = {}
            
            for term, data in search_terms_data.items():
                try:
                    # Create DataFrames
                    search_df = pd.DataFrame({
                        "date": pd.to_datetime(data["dates"]),
                        "interest": data["values"]
                    })
                    
                    price_df = pd.DataFrame({
                        "date": pd.to_datetime(price_dates),
                        "close": closing_prices
                    })
                    
                    # Merge data on date
                    merged_df = pd.merge(search_df, price_df, on="date", how="inner")
                    
                    if len(merged_df) < 5:  # Need at least 5 points for meaningful correlation
                        continue
                    
                    # Calculate correlations
                    # Same-day correlation
                    same_day_corr = merged_df["interest"].corr(merged_df["close"])
                    
                    # Next-day price correlation (search leading price)
                    merged_df["next_day_close"] = merged_df["close"].shift(-1)
                    search_lead_corr = merged_df["interest"].corr(merged_df["next_day_close"])
                    
                    # Previous-day search correlation (price leading search)
                    merged_df["prev_day_interest"] = merged_df["interest"].shift(1)
                    price_lead_corr = merged_df["close"].corr(merged_df["prev_day_interest"])
                    
                    # Calculate price changes
                    merged_df["price_change"] = merged_df["close"].pct_change()
                    
                    # Correlation between search and price changes
                    search_price_change_corr = merged_df["interest"].corr(merged_df["price_change"])
                    
                    # Next-day price change correlation
                    merged_df["next_day_change"] = merged_df["price_change"].shift(-1)
                    search_next_day_change_corr = merged_df["interest"].corr(merged_df["next_day_change"])
                    
                    # Determine if search is a leading indicator
                    is_leading_indicator = abs(search_next_day_change_corr) > abs(same_day_corr)
                    
                    # Determine correlation strength
                    def get_correlation_strength(corr):
                        abs_corr = abs(corr)
                        if abs_corr > 0.7:
                            return "strong"
                        elif abs_corr > 0.4:
                            return "moderate"
                        elif abs_corr > 0.2:
                            return "weak"
                        else:
                            return "no correlation"
                    
                    correlations[term] = {
                        "data_points": len(merged_df),
                        "same_day_correlation": {
                            "coefficient": same_day_corr,
                            "strength": get_correlation_strength(same_day_corr)
                        },
                        "search_leading_price": {
                            "coefficient": search_lead_corr,
                            "strength": get_correlation_strength(search_lead_corr)
                        },
                        "price_leading_search": {
                            "coefficient": price_lead_corr,
                            "strength": get_correlation_strength(price_lead_corr)
                        },
                        "search_price_change_correlation": {
                            "coefficient": search_price_change_corr,
                            "strength": get_correlation_strength(search_price_change_corr)
                        },
                        "search_next_day_change_correlation": {
                            "coefficient": search_next_day_change_corr,
                            "strength": get_correlation_strength(search_next_day_change_corr)
                        },
                        "is_leading_indicator": is_leading_indicator
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing correlation for term '{term}': {e}")
            
            # Determine best predictive term
            if correlations:
                best_term = max(correlations.items(), 
                                key=lambda x: abs(x[1]["search_next_day_change_correlation"]["coefficient"]))
                
                predictive_value = "high" if abs(best_term[1]["search_next_day_change_correlation"]["coefficient"]) > 0.4 else \
                                  "moderate" if abs(best_term[1]["search_next_day_change_correlation"]["coefficient"]) > 0.2 else \
                                  "low"
            else:
                best_term = (None, None)
                predictive_value = "unknown"
            
            return {
                "symbol": symbol,
                "days_analyzed": days,
                "search_terms_analyzed": len(correlations),
                "term_correlations": correlations,
                "best_predictive_term": best_term[0] if best_term[0] else None,
                "predictive_value": predictive_value
            }
            
        except Exception as e:
            self.logger.error(f"Error comparing search to price for {symbol}: {e}")
            return {"error": str(e)}
    
    def generate_trends_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive Google Trends report.
        
        Returns:
            Dictionary containing trends report
        """
        try:
            # Get market sentiment analysis
            market_sentiment = self.db.google_trends_market_sentiment_collection.find_one(
                sort=[("timestamp", -1)]
            )
            
            # Get all active symbols from portfolio
            portfolio_symbols = [doc["symbol"] for doc in self.db.portfolio_collection.find(
                {"status": "active"},
                {"symbol": 1}
            )]
            
            # Get latest trends data for each symbol
            symbols_data = {}
            for symbol in portfolio_symbols:
                latest_data = self.db.google_trends_collection.find_one(
                    {"symbol": symbol},
                    sort=[("timestamp", -1)]
                )
                
                if latest_data:
                    trend_metrics = latest_data.get("trend_metrics", {})
                    symbols_data[symbol] = {
                        "overall_status": trend_metrics.get("overall_status", "unknown"),
                        "short_term": trend_metrics.get("short_term", {}),
                        "medium_term": trend_metrics.get("medium_term", {})
                    }
            
            # Find symbols with strongest trends
            def get_trend_strength(data):
                if "short_term" in data and "average_latest_interest" in data["short_term"]:
                    return data["short_term"]["average_latest_interest"]
                return 0
            
            sorted_symbols = sorted(
                [(symbol, data) for symbol, data in symbols_data.items()],
                key=lambda x: get_trend_strength(x[1]),
                reverse=True
            )
            
            top_symbols = [s[0] for s in sorted_symbols[:5]]
            
            # Get correlation analysis for top symbols
            correlations = {}
            for symbol in top_symbols:
                correlation_data = self.compare_search_to_price(symbol)
                if "error" not in correlation_data:
                    correlations[symbol] = correlation_data
            
            # Create report
            report = {
                "timestamp": datetime.now(),
                "title": "Google Trends Market Report",
                "market_sentiment": market_sentiment["sentiment_indicators"] if market_sentiment else {},
                "top_interest_symbols": [
                    {
                        "symbol": symbol,
                        "trend_data": symbols_data[symbol]
                    }
                    for symbol in top_symbols if symbol in symbols_data
                ],
                "search_price_correlations": correlations,
                "trending_status": {
                    symbol: data["overall_status"]
                    for symbol, data in symbols_data.items()
                    if "overall_status" in data
                }
            }
            
            # Save report to database
            self.db.google_trends_reports_collection.insert_one(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating trends report: {e}")
            return {"error": str(e)}

# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize analyzer
    trends_analyzer = GoogleTrends(db)
    
    # Analyze trends for a symbol
    result = trends_analyzer.analyze_stock_interest("HDFCBANK", "HDFC Bank Ltd")
    print(json.dumps(result, default=str, indent=2))
    
    # Analyze market sentiment
    # sentiment = trends_analyzer.analyze_market_sentiment()
    # print(json.dumps(sentiment, default=str, indent=2))
    
    # Run daily collection
    # collection_results = trends_analyzer.run_daily_collection()
    # print(json.dumps(collection_results, default=str, indent=2))