# reports/templates/morning_report.py (Session 47: Report Templates & Formatters)

import logging
from datetime import datetime, timedelta

class MorningReport:
    """
    Morning report template for trading day preparation.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the morning report generator.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info("Morning report template initialized")
    
    def generate_report(self):
        """
        Generate morning report.
        
        Returns:
            dict: Report data
        """
        try:
            self.logger.info("Generating morning report")
            
            # Get today's date
            today = datetime.now().date()
            today_str = today.strftime("%A, %B %d, %Y")
            
            # Check if market is open today
            from trading.market_hours import MarketHours
            market_hours = MarketHours()
            
            is_trading_day = market_hours.is_trading_day()
            
            if not is_trading_day:
                # Generate non-trading day report
                report_data = {
                    "date": today,
                    "is_trading_day": False,
                    "message": f"Today ({today_str}) is not a trading day. Markets are closed."
                }
                
                return report_data
            
            # Get global market data
            global_markets = self._get_global_markets()
            
            # Get economic events
            economic_events = self._get_economic_events()
            
            # Get market outlook
            market_outlook = self._generate_market_outlook(global_markets, economic_events)
            
            # Get today's predictions
            predictions = self._get_predictions()
            
            # Get top opportunities
            opportunities = self._generate_top_opportunities(predictions)
            
            # Get important news
            news = self._get_important_news()
            
            # Create report data
            report_data = {
                "date": today,
                "is_trading_day": True,
                "title": f"Morning Report - {today_str}",
                "market_outlook": market_outlook,
                "global_markets": global_markets,
                "economic_events": economic_events,
                "predictions": predictions,
                "opportunities": opportunities,
                "news": news
            }
            
            # Save report to database
            if self.db:
                report_id = f"morning_report_{today.strftime('%Y%m%d')}"
                
                self.db.reports.update_one(
                    {"report_id": report_id},
                    {"$set": {
                        "report_id": report_id,
                        "type": "morning_report",
                        "date": datetime.now(),
                        "data": report_data
                    }},
                    upsert=True
                )
            
            return report_data
            
        except Exception as e:
            self.logger.error(f"Error generating morning report: {e}")
            return {"error": str(e)}
    
    def _get_global_markets(self):
        """
        Get global market data.
        
        Returns:
            dict: Global market data
        """
        try:
            if not self.db:
                return {}
            
            # Import required modules
            from data.global_markets.indices_collector import IndicesCollector
            
            # Initialize collector
            collector = IndicesCollector(self.db)
            
            # Get global indices data
            global_indices = collector.get_latest_data()
            
            # Get forex data
            from data.global_markets.forex_collector import ForexCollector
            forex_collector = ForexCollector(self.db)
            
            forex_data = forex_collector.get_latest_data()
            
            # Process data
            us_markets = {}
            asian_markets = {}
            european_markets = {}
            forex_pairs = {}
            
            # Categorize indices
            for index_name, data in global_indices.items():
                # US markets
                if any(us_market in index_name for us_market in ["SPX", "NDX", "DJIA", "RUT"]):
                    us_markets[index_name] = data
                # Asian markets
                elif any(asian_market in index_name for asian_market in ["NIKKEI", "HANG SENG", "SHANGHAI", "KOSPI"]):
                    asian_markets[index_name] = data
                # European markets
                elif any(european_market in index_name for european_market in ["FTSE", "DAX", "CAC", "STOXX"]):
                    european_markets[index_name] = data
            
            # Key forex pairs
            key_pairs = ["USDINR", "EURUSD", "GBPUSD", "USDJPY"]
            for pair in key_pairs:
                if pair in forex_data:
                    forex_pairs[pair] = forex_data[pair]
            
            # Calculate regional sentiment
            def calculate_sentiment(markets_data):
                changes = [data.get('percent_change', 0) for data in markets_data.values()]
                avg_change = sum(changes) / len(changes) if changes else 0
                
                if avg_change > 0.5:
                    return "strongly positive"
                elif avg_change > 0:
                    return "positive"
                elif avg_change < -0.5:
                    return "strongly negative"
                elif avg_change < 0:
                    return "negative"
                else:
                    return "neutral"
            
            us_sentiment = calculate_sentiment(us_markets)
            asian_sentiment = calculate_sentiment(asian_markets)
            european_sentiment = calculate_sentiment(european_markets)
            
            # Create global markets data
            global_markets = {
                "us_markets": us_markets,
                "asian_markets": asian_markets,
                "european_markets": european_markets,
                "forex_pairs": forex_pairs,
                "us_sentiment": us_sentiment,
                "asian_sentiment": asian_sentiment,
                "european_sentiment": european_sentiment,
                "global_sentiment": self._determine_global_sentiment(us_sentiment, asian_sentiment, european_sentiment)
            }
            
            return global_markets
            
        except Exception as e:
            self.logger.error(f"Error getting global market data: {e}")
            return {}
    
    def _determine_global_sentiment(self, us_sentiment, asian_sentiment, european_sentiment):
        """
        Determine overall global sentiment.
        
        Args:
            us_sentiment (str): US market sentiment
            asian_sentiment (str): Asian market sentiment
            european_sentiment (str): European market sentiment
            
        Returns:
            str: Overall global sentiment
        """
        # Convert sentiment to numeric score
        sentiment_scores = {
            "strongly positive": 2,
            "positive": 1,
            "neutral": 0,
            "negative": -1,
            "strongly negative": -2
        }
        
        us_score = sentiment_scores.get(us_sentiment, 0)
        asian_score = sentiment_scores.get(asian_sentiment, 0)
        european_score = sentiment_scores.get(european_sentiment, 0)
        
        # Calculate weighted average (US has higher weight)
        avg_score = (us_score * 2 + asian_score + european_score) / 4
        
        # Convert back to sentiment
        if avg_score > 1.5:
            return "strongly positive"
        elif avg_score > 0.5:
            return "positive"
        elif avg_score < -1.5:
            return "strongly negative"
        elif avg_score < -0.5:
            return "negative"
        else:
            return "neutral"
    
    def _get_economic_events(self):
        """
        Get economic calendar events.
        
        Returns:
            dict: Economic events data
        """
        try:
            if not self.db:
                return {}
            
            # Import economic calendar
            from data.global_markets.economic_calendar import EconomicCalendar
            calendar = EconomicCalendar(self.db)
            
            # Get today's events
            today = datetime.now().date()
            events = calendar.get_events_for_date(today)
            
            # Categorize by importance
            high_importance = [e for e in events if e.get('importance') == 'high']
            medium_importance = [e for e in events if e.get('importance') == 'medium']
            
            # Sort by time
            high_importance.sort(key=lambda x: x.get('time', '00:00'))
            medium_importance.sort(key=lambda x: x.get('time', '00:00'))
            
            # Create economic events data
            economic_events = {
                "date": today,
                "high_importance": high_importance,
                "medium_importance": medium_importance,
                "all_events": events,
                "has_high_importance": len(high_importance) > 0
            }
            
            return economic_events
            
        except Exception as e:
            self.logger.error(f"Error getting economic events: {e}")
            return {}
    
    def _generate_market_outlook(self, global_markets, economic_events):
        """
        Generate market outlook.
        
        Args:
            global_markets (dict): Global market data
            economic_events (dict): Economic events data
            
        Returns:
            dict: Market outlook
        """
        try:
            # Initialize factors
            factors = []
            
            # Global markets factor
            global_sentiment = global_markets.get('global_sentiment', 'neutral')
            
            if global_sentiment in ["strongly positive", "positive"]:
                factors.append("Positive global market sentiment")
            elif global_sentiment in ["strongly negative", "negative"]:
                factors.append("Negative global market sentiment")
            
            # US market factor
            us_sentiment = global_markets.get('us_sentiment', 'neutral')
            
            if us_sentiment in ["strongly positive", "positive"]:
                factors.append("Positive US market performance")
            elif us_sentiment in ["strongly negative", "negative"]:
                factors.append("Negative US market performance")
            
            # Economic events factor
            high_importance = economic_events.get('high_importance', [])
            
            if high_importance:
                factors.append(f"{len(high_importance)} high-importance economic events today")
            
            # Forex factor - USDINR
            forex_pairs = global_markets.get('forex_pairs', {})
            
            if 'USDINR' in forex_pairs:
                usdinr_change = forex_pairs['USDINR'].get('percent_change', 0)
                
                if usdinr_change > 0.3:
                    factors.append("Significant INR weakening against USD")
                elif usdinr_change < -0.3:
                    factors.append("Significant INR strengthening against USD")
            
            # Determine overall outlook
            if global_sentiment in ["strongly positive", "positive"] and us_sentiment in ["strongly positive", "positive"]:
                outlook = "positive"
                description = "Global markets are showing positive sentiment, which could support Indian markets today."
            elif global_sentiment in ["strongly negative", "negative"] and us_sentiment in ["strongly negative", "negative"]:
                outlook = "negative"
                description = "Global markets are under pressure, which could weigh on Indian markets today."
            else:
                outlook = "mixed"
                description = "Mixed signals from global markets suggest cautious trading today."
            
            # Create market outlook
            market_outlook = {
                "outlook": outlook,
                "description": description,
                "factors": factors,
                "global_sentiment": global_sentiment
            }
            
            return market_outlook
            
        except Exception as e:
            self.logger.error(f"Error generating market outlook: {e}")
            return {"outlook": "neutral", "description": "Unable to determine market outlook", "factors": []}
    
    def _get_predictions(self):
        """
        Get today's trading predictions.
        
        Returns:
            dict: Prediction data
        """
        try:
            if not self.db:
                return {}
            
            # Import predictor
            from ml.prediction.daily_predictor import DailyPredictor
            predictor = DailyPredictor(self.db)
            
            # Get all predictions
            all_predictions = predictor.get_all_predictions()
            
            # Filter for high confidence
            high_confidence = [p for p in all_predictions if p.get('confidence', 0) >= 0.7]
            
            # Sort by confidence
            high_confidence.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Limit to top 10
            top_predictions = high_confidence[:10]
            
            # Separate by direction
            bullish = [p for p in top_predictions if p.get('prediction') == 'up']
            bearish = [p for p in top_predictions if p.get('prediction') == 'down']
            
            # Create prediction data
            prediction_data = {
                "all": all_predictions,
                "high_confidence": high_confidence,
                "top": top_predictions,
                "bullish": bullish,
                "bearish": bearish,
                "bias": "bullish" if len(bullish) > len(bearish) else "bearish" if len(bearish) > len(bullish) else "neutral"
            }
            
            return prediction_data
            
        except Exception as e:
            self.logger.error(f"Error getting predictions: {e}")
            return {}
    
    def _generate_top_opportunities(self, predictions):
        """
        Generate top trading opportunities.
        
        Args:
            predictions (dict): Prediction data
            
        Returns:
            dict: Trading opportunities
        """
        try:
            # Top 3 bullish opportunities
            bullish = predictions.get('bullish', [])[:3]
            
            # Top 3 bearish opportunities
            bearish = predictions.get('bearish', [])[:3]
            
            # Format opportunities
            bullish_opportunities = []
            
            for pred in bullish:
                opportunity = {
                    "symbol": pred.get('symbol', 'Unknown'),
                    "exchange": pred.get('exchange', 'Unknown'),
                    "confidence": pred.get('confidence', 0),
                    "target_price": pred.get('target_price', 0),
                    "stop_loss": pred.get('stop_loss', 0),
                    "supporting_factors": pred.get('supporting_factors', [])
                }
                
                bullish_opportunities.append(opportunity)
            
            bearish_opportunities = []
            
            for pred in bearish:
                opportunity = {
                    "symbol": pred.get('symbol', 'Unknown'),
                    "exchange": pred.get('exchange', 'Unknown'),
                    "confidence": pred.get('confidence', 0),
                    "target_price": pred.get('target_price', 0),
                    "stop_loss": pred.get('stop_loss', 0),
                    "supporting_factors": pred.get('supporting_factors', [])
                }
                
                bearish_opportunities.append(opportunity)
            
            # Create opportunities data
            opportunities = {
                "bullish": bullish_opportunities,
                "bearish": bearish_opportunities
            }
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error generating top opportunities: {e}")
            return {"bullish": [], "bearish": []}
    
    def _get_important_news(self):
        """
        Get important market news.
        
        Returns:
            list: News items
        """
        try:
            if not self.db:
                return []
            
            # Get recent news
            yesterday = datetime.now() - timedelta(days=1)
            
            cursor = self.db.news_data.find({
                "published_date": {"$gt": yesterday}
            }).sort("published_date", -1)
            
            news = list(cursor)
            
            # Filter to important news (high sentiment score or from reputable sources)
            important_news = []
            
            reputable_sources = ["Bloomberg", "Reuters", "Economic Times", "Financial Times"]
            
            for item in news:
                sentiment_score = item.get('sentiment_score', 0)
                source = item.get('source', '')
                
                # Include if high sentiment (positive or negative) or from reputable source
                if abs(sentiment_score) > 0.6 or any(src in source for src in reputable_sources):
                    important_news.append(item)
            
            # Sort by absolute sentiment (most impactful first)
            important_news.sort(key=lambda x: abs(x.get('sentiment_score', 0)), reverse=True)
            
            # Limit to top 5
            return important_news[:5]
            
        except Exception as e:
            self.logger.error(f"Error getting important news: {e}")
            return []