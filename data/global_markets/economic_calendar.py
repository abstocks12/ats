"""
Economic Calendar Collector

This module collects economic events and indicators from around the world.
It provides data on important economic releases, central bank meetings, and other market-moving events.
"""

import requests
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import json
from bs4 import BeautifulSoup

class EconomicCalendar:
    """
    Collects and processes economic events and indicators from global markets.
    Tracks important events that may impact trading decisions.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the economic calendar collector with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Define importance levels
        self.importance_levels = {
            "low": 1,      # Low impact events
            "medium": 2,   # Medium impact events
            "high": 3      # High impact events
        }
        
        # Define event categories
        self.event_categories = {
            "economic_indicator": ["GDP", "CPI", "PMI", "Unemployment", "Retail Sales", "Trade Balance"],
            "central_bank": ["Rate Decision", "Minutes", "Speech", "Testimony"],
            "government": ["Elections", "Budget", "Policy Announcement"],
            "earnings": ["Earnings Report", "Corporate Announcement"]
        }
        
        # Tracked countries (focus on major economies and India)
        self.tracked_countries = [
            "India", "US", "Eurozone", "UK", "Japan", "China", "Germany", 
            "France", "Australia", "Canada", "Brazil", "South Korea"
        ]
        
        # API configuration
        self.api_keys = self._load_api_keys()
        self.base_url = "https://api.economicdata.com/calendar"  # Replace with actual API URL
        self.alternative_url = "https://www.investing.com/economic-calendar/"  # Backup source for scraping
        
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
            
            if api_config and "economic_calendar" in api_config:
                return api_config["economic_calendar"]
            
            # Fallback to default keys (should be replaced in production)
            return {
                "primary_api": "your_economic_calendar_api_key"
            }
            
        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")
            return {}
    
    def collect_upcoming_events(self, days: int = 7) -> Dict[str, Any]:
        """
        Collect upcoming economic events for the specified number of days.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            Dictionary containing upcoming economic events
        """
        self.logger.info(f"Collecting economic events for the next {days} days")
        
        results = {
            "timestamp": datetime.now(),
            "days": days,
            "events": []
        }
        
        try:
            # Calculate date range
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Try primary API first
            if "primary_api" in self.api_keys:
                try:
                    api_events = self._fetch_events_from_api(start_date, end_date)
                    if api_events:
                        results["events"] = api_events
                        results["source"] = "primary_api"
                except Exception as e:
                    self.logger.error(f"Error fetching events from primary API: {e}")
            
            # If no results or error, try alternative source (web scraping)
            if not results["events"]:
                try:
                    scraped_events = self._scrape_economic_calendar(days)
                    if scraped_events:
                        results["events"] = scraped_events
                        results["source"] = "web_scraping"
                except Exception as e:
                    self.logger.error(f"Error scraping economic calendar: {e}")
            
            # Save events to database
            if results["events"]:
                self._save_events(results["events"])
                self.logger.info(f"Collected {len(results['events'])} economic events")
            else:
                self.logger.warning("No economic events collected")
        
        except Exception as e:
            self.logger.error(f"Error collecting upcoming events: {e}")
            results["error"] = str(e)
        
        return results
    
    def _fetch_events_from_api(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Fetch economic events from API.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of economic events
        """
        try:
            response = requests.get(
                self.base_url,
                params={
                    "start_date": start_date,
                    "end_date": end_date,
                    "countries": ",".join(self.tracked_countries),
                    "apikey": self.api_keys["primary_api"]
                },
                timeout=15
            )
            
            if response.status_code == 200:
                data = response.json()
                
                if "events" in data:
                    events = []
                    
                    for event in data["events"]:
                        processed_event = {
                            "event_id": event.get("id", ""),
                            "title": event.get("title", ""),
                            "country": event.get("country", ""),
                            "date": event.get("date", ""),
                            "time": event.get("time", ""),
                            "importance": event.get("importance", "medium"),
                            "previous": event.get("previous", ""),
                            "forecast": event.get("forecast", ""),
                            "actual": event.get("actual", ""),
                            "category": self._determine_category(event.get("title", "")),
                            "source": "API"
                        }
                        
                        events.append(processed_event)
                    
                    return events
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching events from API: {e}")
            return []
    
    def _scrape_economic_calendar(self, days: int) -> List[Dict[str, Any]]:
        """
        Scrape economic calendar from alternative source.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of economic events
        """
        try:
            events = []
            
            # Loop through each day
            for day in range(days):
                date = datetime.now() + timedelta(days=day)
                date_str = date.strftime("%Y-%m-%d")
                
                # Construct URL
                url = f"{self.alternative_url}?day={date_str}"
                
                # Fetch page
                response = requests.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                })
                
                if response.status_code == 200:
                    # Parse HTML
                    soup = BeautifulSoup(response.text, "html.parser")
                    
                    # Find event table
                    event_table = soup.find("table", class_="economicCalendarTable")
                    
                    if event_table:
                        # Process table rows
                        rows = event_table.find_all("tr", class_="js-event-item")
                        
                        for row in rows:
                            try:
                                # Extract country
                                country_cell = row.find("td", class_="flagCur")
                                country = country_cell.find("span", class_="ceFlags").get("title") if country_cell else ""
                                
                                # Only process if country is in our tracked list
                                if country in self.tracked_countries:
                                    # Extract event details
                                    time = row.find("td", class_="time").text.strip() if row.find("td", class_="time") else ""
                                    title = row.find("td", class_="event").text.strip() if row.find("td", class_="event") else ""
                                    
                                    # Extract importance
                                    importance_cell = row.find("td", class_="sentiment")
                                    importance = "medium"  # Default
                                    
                                    if importance_cell:
                                        bull_spans = importance_cell.find_all("i", class_="grayFullBullishIcon")
                                        if bull_spans:
                                            if len(bull_spans) >= 3:
                                                importance = "high"
                                            elif len(bull_spans) >= 2:
                                                importance = "medium"
                                            else:
                                                importance = "low"
                                    
                                    # Extract forecast, previous, actual
                                    forecast = row.find("td", class_="forecast").text.strip() if row.find("td", class_="forecast") else ""
                                    previous = row.find("td", class_="prev").text.strip() if row.find("td", class_="prev") else ""
                                    actual = row.find("td", class_="actual").text.strip() if row.find("td", class_="actual") else ""
                                    
                                    # Create event object
                                    event = {
                                        "event_id": f"scraped-{date_str}-{country}-{time}",
                                        "title": title,
                                        "country": country,
                                        "date": date_str,
                                        "time": time,
                                        "importance": importance,
                                        "previous": previous,
                                        "forecast": forecast,
                                        "actual": actual,
                                        "category": self._determine_category(title),
                                        "source": "Web"
                                    }
                                    
                                    events.append(event)
                            
                            except Exception as e:
                                self.logger.error(f"Error processing event row: {e}")
                
                # Respect website's crawling policy
                time.sleep(2)
            
            return events
            
        except Exception as e:
            self.logger.error(f"Error scraping economic calendar: {e}")
            return []
    
    def _determine_category(self, title: str) -> str:
        """
        Determine the category of an economic event based on its title.
        
        Args:
            title: Event title
            
        Returns:
            Category name
        """
        title_lower = title.lower()
        
        for category, keywords in self.event_categories.items():
            for keyword in keywords:
                if keyword.lower() in title_lower:
                    return category
        
        # Default category if no match
        return "other"
    
    def _save_events(self, events: List[Dict[str, Any]]) -> bool:
        """
        Save economic events to database.
        
        Args:
            events: List of economic events
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for existing events to avoid duplicates
            for event in events:
                existing = self.db.economic_events_collection.find_one({
                    "event_id": event["event_id"]
                })
                
                if existing:
                    # Update existing event (actual results might have been updated)
                    self.db.economic_events_collection.update_one(
                        {"_id": existing["_id"]},
                        {"$set": event}
                    )
                else:
                    # Insert new event
                    self.db.economic_events_collection.insert_one(event)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving events: {e}")
            return False
    
    def get_today_events(self) -> Dict[str, Any]:
        """
        Get economic events for today.
        
        Returns:
            Dictionary containing today's economic events
        """
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Query database for today's events
            events = list(self.db.economic_events_collection.find({
                "date": today
            }))
            
            # Sort by importance (high to low) and time
            sorted_events = sorted(
                events,
                key=lambda x: (
                    -self.importance_levels.get(x.get("importance", "medium"), 2),  # Negative for descending order
                    x.get("time", "")
                )
            )
            
            # Group by country
            grouped_events = {}
            for event in sorted_events:
                country = event.get("country", "Other")
                
                if country not in grouped_events:
                    grouped_events[country] = []
                
                grouped_events[country].append({
                    "title": event.get("title", ""),
                    "time": event.get("time", ""),
                    "importance": event.get("importance", "medium"),
                    "previous": event.get("previous", ""),
                    "forecast": event.get("forecast", ""),
                    "actual": event.get("actual", ""),
                    "category": event.get("category", "other")
                })
            
            return {
                "date": today,
                "total_events": len(sorted_events),
                "countries": len(grouped_events),
                "events_by_country": grouped_events
            }
            
        except Exception as e:
            self.logger.error(f"Error getting today's events: {e}")
            return {"error": str(e)}
    
    def get_high_impact_events(self, days: int = 7) -> Dict[str, Any]:
        """
        Get high-impact economic events for the specified number of days.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            Dictionary containing high-impact economic events
        """
        try:
            # Calculate date range
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Query database for high-impact events
            events = list(self.db.economic_events_collection.find({
                "date": {"$gte": start_date, "$lte": end_date},
                "importance": "high"
            }))
            
            # Sort by date and time
            sorted_events = sorted(
                events,
                key=lambda x: (x.get("date", ""), x.get("time", ""))
            )
            
            # Group by date
            grouped_events = {}
            for event in sorted_events:
                date = event.get("date", "Unknown")
                
                if date not in grouped_events:
                    grouped_events[date] = []
                
                grouped_events[date].append({
                    "title": event.get("title", ""),
                    "country": event.get("country", ""),
                    "time": event.get("time", ""),
                    "previous": event.get("previous", ""),
                    "forecast": event.get("forecast", ""),
                    "category": event.get("category", "other")
                })
            
            return {
                "date_range": f"{start_date} to {end_date}",
                "total_events": len(sorted_events),
                "events_by_date": grouped_events
            }
            
        except Exception as e:
            self.logger.error(f"Error getting high-impact events: {e}")
            return {"error": str(e)}
    
    def get_country_events(self, country: str, days: int = 7) -> Dict[str, Any]:
        """
        Get economic events for a specific country.
        
        Args:
            country: Country name
            days: Number of days to look ahead
            
        Returns:
            Dictionary containing economic events for the specified country
        """
        try:
            # Calculate date range
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Query database for country events
            events = list(self.db.economic_events_collection.find({
                "date": {"$gte": start_date, "$lte": end_date},
                "country": country
            }))
            
            # Sort by importance, date, and time
            sorted_events = sorted(
                events,
                key=lambda x: (
                    x.get("date", ""),
                    -self.importance_levels.get(x.get("importance", "medium"), 2),  # Negative for descending order
                    x.get("time", "")
                )
            )
            
            # Group by date
            grouped_events = {}
            for event in sorted_events:
                date = event.get("date", "Unknown")
                
                if date not in grouped_events:
                    grouped_events[date] = []
                
                grouped_events[date].append({
                    "title": event.get("title", ""),
                    "time": event.get("time", ""),
                    "importance": event.get("importance", "medium"),
                    "previous": event.get("previous", ""),
                    "forecast": event.get("forecast", ""),
                    "actual": event.get("actual", ""),
                    "category": event.get("category", "other")
                })
            
            return {
                "country": country,
                "date_range": f"{start_date} to {end_date}",
                "total_events": len(sorted_events),
                "events_by_date": grouped_events
            }
            
        except Exception as e:
            self.logger.error(f"Error getting country events: {e}")
            return {"error": str(e)}
    
    def generate_market_impact_report(self) -> Dict[str, Any]:
        """
        Generate a report on potential market impact of upcoming economic events.
        
        Returns:
            Dictionary containing market impact report
        """
        try:
            # Get upcoming high-impact events
            high_impact_events = self.get_high_impact_events(days=7)
            
            # Get India events
            india_events = self.get_country_events("India", days=7)
            
            # Create market impact report
            report = {
                "timestamp": datetime.now(),
                "title": "Economic Calendar Market Impact Report",
                "sections": {
                    "today_events": self.get_today_events(),
                    "high_impact_events": high_impact_events,
                    "india_events": india_events,
                    "market_impact_analysis": {}
                }
            }
            
            # Analyze potential market impact
            impact_analysis = {}
            
            # Process all high-impact events
            if "events_by_date" in high_impact_events:
                for date, events in high_impact_events["events_by_date"].items():
                    for event in events:
                        event_title = event.get("title", "")
                        country = event.get("country", "")
                        
                        # Determine potential market impact
                        impact = self._analyze_event_impact(event, country)
                        
                        if impact:
                            key = f"{date} - {country} - {event_title}"
                            impact_analysis[key] = impact
            
            report["sections"]["market_impact_analysis"] = impact_analysis
            
            # Save report to database
            self.db.economic_reports_collection.insert_one(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating market impact report: {e}")
            return {"error": str(e)}
    
    def _analyze_event_impact(self, event: Dict[str, Any], country: str) -> Dict[str, Any]:
        """
        Analyze potential market impact of an economic event.
        
        Args:
            event: Economic event data
            country: Country name
            
        Returns:
            Dictionary containing impact analysis
        """
        try:
            title = event.get("title", "")
            category = event.get("category", "other")
            previous = event.get("previous", "")
            forecast = event.get("forecast", "")
            
            impact = {
                "markets_affected": [],
                "potential_impact": "",
                "analysis": ""
            }
            
            # Determine markets affected based on country and event type
            if country == "India":
                impact["markets_affected"] = ["NIFTY", "SENSEX", "USDINR"]
            elif country == "US":
                impact["markets_affected"] = ["Global Equities", "USD", "US Treasuries"]
                if "Fed" in title or "Interest Rate" in title:
                    impact["markets_affected"].append("Global Risk Assets")
            elif country == "Eurozone":
                impact["markets_affected"] = ["EUR", "European Equities"]
            elif country == "China":
                impact["markets_affected"] = ["CNY", "Asian Equities", "Commodities"]
            else:
                impact["markets_affected"] = [f"{country} Assets"]
            
            # Analyze potential impact based on event category
            if category == "central_bank":
                if "Rate Decision" in title:
                    impact["potential_impact"] = "High"
                    impact["analysis"] = "Central bank rate decisions can significantly impact currency values and equity markets. Watch for any deviation from expectations."
                elif "Minutes" in title:
                    impact["potential_impact"] = "Medium"
                    impact["analysis"] = "Central bank minutes provide insights into future policy direction, which can cause market volatility."
            elif category == "economic_indicator":
                if "GDP" in title:
                    impact["potential_impact"] = "High"
                    impact["analysis"] = "GDP releases are key indicators of economic health and can drive significant market movements."
                elif "CPI" in title or "Inflation" in title:
                    impact["potential_impact"] = "High"
                    impact["analysis"] = "Inflation data impacts interest rate expectations and can cause volatility in currency and bond markets."
                elif "Employment" in title or "Unemployment" in title:
                    impact["potential_impact"] = "High"
                    impact["analysis"] = "Employment data is a leading indicator for economic health and monetary policy decisions."
                elif "PMI" in title or "Manufacturing" in title:
                    impact["potential_impact"] = "Medium"
                    impact["analysis"] = "PMI data provides insights into economic activity and can impact expectations for future growth."
            
            # Add forecast vs previous analysis if available
            if forecast and previous:
                try:
                    # Clean values and convert to float
                    forecast_val = float(forecast.replace("%", "").replace("K", "000").replace("M", "000000").strip())
                    previous_val = float(previous.replace("%", "").replace("K", "000").replace("M", "000000").strip())
                    
                    # Calculate percent change
                    if previous_val != 0:
                        percent_change = ((forecast_val - previous_val) / abs(previous_val)) * 100
                        
                        if percent_change > 5:
                            impact["analysis"] += f" Forecast shows significant improvement ({percent_change:.1f}% higher) from previous reading."
                        elif percent_change < -5:
                            impact["analysis"] += f" Forecast shows significant deterioration ({abs(percent_change):.1f}% lower) from previous reading."
                except:
                    # If conversion fails, just skip this analysis
                    pass
            
            return impact
            
        except Exception as e:
            self.logger.error(f"Error analyzing event impact: {e}")
            return {}
    
    def run_daily_collection(self) -> Dict[str, Any]:
        """
        Run daily collection of economic calendar data.
        
        Returns:
            Dictionary containing collection results
        """
        self.logger.info("Running daily collection of economic calendar data")
        
        results = {
            "timestamp": datetime.now(),
            "upcoming_events": None,
            "today_events": None,
            "market_impact_report": None
        }
        
        try:
            # 1. Collect upcoming events
            upcoming_events = self.collect_upcoming_events()
            results["upcoming_events"] = {
                "status": "success" if "error" not in upcoming_events else "failed",
                "count": len(upcoming_events.get("events", []))
            }
            
            # 2. Get today's events
            today_events = self.get_today_events()
            results["today_events"] = {
                "status": "success" if "error" not in today_events else "failed",
                "count": today_events.get("total_events", 0)
            }
            
            # 3. Generate market impact report
            if upcoming_events and "error" not in upcoming_events:
                market_impact = self.generate_market_impact_report()
                results["market_impact_report"] = {
                    "status": "success" if "error" not in market_impact else "failed"
                }
            
            self.logger.info("Daily collection of economic calendar data completed successfully")
            
        except Exception as e:
            self.logger.error(f"Error in daily collection: {e}")
            results["error"] = str(e)
        
        return results
    
    def get_relevant_events_for_symbol(self, symbol: str, days: int = 7) -> Dict[str, Any]:
        """
        Get economic events relevant to a specific trading symbol.
        
        Args:
            symbol: Trading symbol
            days: Number of days to look ahead
            
        Returns:
            Dictionary containing relevant economic events
        """
        try:
            # Determine relevant countries based on symbol
            relevant_countries = []
            
            # Indian stocks and indices
            if symbol in ["NIFTY", "SENSEX"] or symbol.endswith(".NS"):
                relevant_countries = ["India", "US", "China"]  # India and major global influences
            
            # Currency pairs
            elif "INR" in symbol:
                if "USD" in symbol:
                    relevant_countries = ["India", "US"]
                elif "EUR" in symbol:
                    relevant_countries = ["India", "Eurozone"]
                elif "GBP" in symbol:
                    relevant_countries = ["India", "UK"]
                else:
                    relevant_countries = ["India"]
            
            # US stocks and indices
            elif symbol in ["SPY", "QQQ", "DIA"] or symbol.endswith(".US"):
                relevant_countries = ["US", "China"]
            
            # Default to major economies if symbol type can't be determined
            else:
                relevant_countries = ["India", "US", "Eurozone", "China"]
            
            # Calculate date range
            start_date = datetime.now().strftime("%Y-%m-%d")
            end_date = (datetime.now() + timedelta(days=days)).strftime("%Y-%m-%d")
            
            # Query database for relevant events
            events = list(self.db.economic_events_collection.find({
                "date": {"$gte": start_date, "$lte": end_date},
                "country": {"$in": relevant_countries}
            }))
            
            # Filter to medium and high importance events only
            filtered_events = [event for event in events if event.get("importance") in ["medium", "high"]]
            
            # Sort by date, importance, and time
            sorted_events = sorted(
                filtered_events,
                key=lambda x: (
                    x.get("date", ""),
                    -self.importance_levels.get(x.get("importance", "medium"), 2),  # Negative for descending order
                    x.get("time", "")
                )
            )
            
            # Group by date
            grouped_events = {}
            for event in sorted_events:
                date = event.get("date", "Unknown")
                
                if date not in grouped_events:
                    grouped_events[date] = []
                
                grouped_events[date].append({
                    "title": event.get("title", ""),
                    "country": event.get("country", ""),
                    "time": event.get("time", ""),
                    "importance": event.get("importance", "medium"),
                    "previous": event.get("previous", ""),
                    "forecast": event.get("forecast", ""),
                    "category": event.get("category", "other")
                })
            
            return {
                "symbol": symbol,
                "relevant_countries": relevant_countries,
                "date_range": f"{start_date} to {end_date}",
                "total_events": len(sorted_events),
                "events_by_date": grouped_events
            }
            
        except Exception as e:
            self.logger.error(f"Error getting relevant events for symbol {symbol}: {e}")
            return {"error": str(e)}


# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize collector
    calendar = EconomicCalendar(db)
    
    # Run daily collection
    results = calendar.run_daily_collection()
    print(json.dumps(results, default=str, indent=2))
    
    # Get relevant events for NIFTY
    nifty_events = calendar.get_relevant_events_for_symbol("NIFTY")
    print(json.dumps(nifty_events, default=str, indent=2))