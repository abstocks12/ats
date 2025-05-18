# automation/morning_briefing.py
import logging
from datetime import datetime, timedelta
import os
import json

class MorningBriefing:
    """
    Generates morning briefing reports for the trading day.
    """
    
    def __init__(self, db_connector=None, logger=None):
        """
        Initialize the morning briefing generator.
        
        Args:
            db_connector: MongoDB connector (optional)
            logger: Logger instance (optional)
        """
        self.db = db_connector
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.config = {
            'include_global_markets': True,      # Include global market data
            'include_sector_analysis': True,     # Include sector analysis
            'include_watchlist_analysis': True,  # Include watchlist analysis
            'include_economic_events': True,     # Include economic calendar events
            'include_earnings': True,            # Include earnings announcements
            'include_predictions': True,         # Include trading predictions
            'max_predictions': 5,                # Maximum predictions to include
            'min_confidence': 0.65,              # Minimum confidence for predictions
            'include_news': True,                # Include market news
            'max_news_items': 5,                 # Maximum news items to include
            'format': 'markdown',                # Output format (markdown, html)
            'delivery_methods': ['whatsapp', 'email'],  # Delivery methods
            'whatsapp_number': None,             # WhatsApp number (overrides default)
            'email_recipients': None,            # Email recipients (overrides default)
            'save_to_database': True             # Save report to database
        }
        
        self.logger.info("Morning briefing generator initialized")
        
    def generate_briefing(self):
        """
        Generate morning briefing report.
        
        Returns:
            dict: Briefing data and formatted report
        """
        try:
            self.logger.info("Generating morning briefing")
            
            # Get today's date
            today = datetime.now().date()
            today_str = today.strftime("%A, %B %d, %Y")
            
            # Check if market is open today
            from trading.market_hours import MarketHours
            market_hours = MarketHours()
            
            is_trading_day = market_hours.is_trading_day()
            
            if not is_trading_day:
                self.logger.info(f"Today ({today_str}) is not a trading day")
                
                # Generate simple non-trading day report
                report_data = {
                    "date": today,
                    "is_trading_day": False,
                    "message": f"Today ({today_str}) is not a trading day. Markets are closed."
                }
                
                # Format report
                formatted_report = self._format_non_trading_day_report(report_data)
                
                # Store report
                if self.config['save_to_database'] and self.db:
                    self._save_report_to_database(report_data, formatted_report)
                    
                # Return result
                return {
                    "report_data": report_data,
                    "formatted_report": formatted_report,
                    "is_trading_day": False
                }
            
            # Collect briefing data
            global_markets = self._get_global_markets() if self.config['include_global_markets'] else None
            sector_analysis = self._get_sector_analysis() if self.config['include_sector_analysis'] else None
            watchlist = self._get_watchlist_analysis() if self.config['include_watchlist_analysis'] else None
            economic_events = self._get_economic_events() if self.config['include_economic_events'] else None
            earnings = self._get_earnings_announcements() if self.config['include_earnings'] else None
            predictions = self._get_trading_predictions() if self.config['include_predictions'] else None
            news = self._get_market_news() if self.config['include_news'] else None
            
            # Generate market outlook
            market_outlook = self._generate_market_outlook(
                global_markets, 
                sector_analysis,
                economic_events,
                news
            )
            
            # Generate trading plan
            trading_plan = self._generate_trading_plan(
                predictions,
                watchlist,
                earnings
            )
            
            # Create report data
            report_data = {
                "date": today,
                "is_trading_day": True,
                "market_outlook": market_outlook,
                "trading_plan": trading_plan,
                "global_markets": global_markets,
                "sector_analysis": sector_analysis,
                "watchlist": watchlist,
                "economic_events": economic_events,
                "earnings": earnings,
                "predictions": predictions,
                "news": news
            }
            
            # Format report
            formatted_report = self._format_report(report_data)
            
            # Store report
            if self.config['save_to_database'] and self.db:
                self._save_report_to_database(report_data, formatted_report)
                
            # Deliver report
            self._deliver_report(formatted_report)
            
            self.logger.info("Morning briefing generated and delivered")
            
            # Return result
            return {
                "report_data": report_data,
                "formatted_report": formatted_report,
                "is_trading_day": True
            }
            
        except Exception as e:
            self.logger.error(f"Error generating morning briefing: {e}")
            return {"error": str(e)}
            
    def _get_global_markets(self):
        """
        Get global market data.
        
        Returns:
            dict: Global market data
        """
        try:
            self.logger.info("Getting global market data")
            
            # Import required modules
            from data.global_markets.indices_collector import IndicesCollector
            
            # Initialize collector
            collector = IndicesCollector(self.db)
            
            # Get latest global indices data
            global_indices = collector.get_latest_data()
            
            # Process data for report
            us_markets = {}
            asian_markets = {}
            european_markets = {}
            
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
                    
            # Calculate average performance by region
            def calculate_avg_change(markets_data):
                changes = [data.get('percent_change', 0) for data in markets_data.values()]
                return sum(changes) / len(changes) if changes else 0
                
            us_avg_change = calculate_avg_change(us_markets)
            asian_avg_change = calculate_avg_change(asian_markets)
            european_avg_change = calculate_avg_change(european_markets)
            
            # Determine overall sentiment
            if us_avg_change > 0.5 and asian_avg_change > 0.5 and european_avg_change > 0.5:
                sentiment = "strongly bullish"
            elif us_avg_change > 0 and asian_avg_change > 0 and european_avg_change > 0:
                sentiment = "bullish"
            elif us_avg_change < -0.5 and asian_avg_change < -0.5 and european_avg_change < -0.5:
                sentiment = "strongly bearish"
            elif us_avg_change < 0 and asian_avg_change < 0 and european_avg_change < 0:
                sentiment = "bearish"
            else:
                sentiment = "mixed"
                
            # Create global markets data
            global_markets = {
                "us_markets": us_markets,
                "asian_markets": asian_markets,
                "european_markets": european_markets,
                "us_avg_change": us_avg_change,
                "asian_avg_change": asian_avg_change,
                "european_avg_change": european_avg_change,
                "sentiment": sentiment
            }
            
            return global_markets
            
        except Exception as e:
            self.logger.error(f"Error getting global market data: {e}")
            return None
            
    def _get_sector_analysis(self):
        """
        Get sector analysis data.
        
        Returns:
            dict: Sector analysis data
        """
        try:
            self.logger.info("Getting sector analysis")
            
            # Import required modules
            from ml.prediction.sector_rotation_analyzer import SectorRotationAnalyzer
            
            # Initialize analyzer
            analyzer = SectorRotationAnalyzer(self.db)
            
            # Get sector analysis
            sector_data = analyzer.analyze_daily_rotation()
            
            # Get recent sector performance
            if self.db:
                # Get sector performance for the last 5 days
                yesterday = (datetime.now() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                five_days_ago = (datetime.now() - timedelta(days=5)).replace(hour=0, minute=0, second=0, microsecond=0)
                
                # Get sector data
                cursor = self.db.sector_performance.find({
                    "date": {"$gte": five_days_ago, "$lte": yesterday}
                }).sort("date", 1)
                
                sector_history = list(cursor)
                
                # Add sector history
                sector_data["history"] = sector_history
            
            return sector_data
            
        except Exception as e:
            self.logger.error(f"Error getting sector analysis: {e}")
            return None
            
    def _get_watchlist_analysis(self):
        """
        Get watchlist analysis data.
        
        Returns:
            dict: Watchlist analysis data
        """
        try:
            self.logger.info("Getting watchlist analysis")
            
            if not self.db:
                return None
                
            # Get active stocks from portfolio
            stocks = list(self.db.portfolio_collection.find({
                "status": "active",
                "instrument_type": "equity"
            }))
            
            if not stocks:
                return None
                
            # Analyze each stock
            watchlist_data = []
            
            for stock in stocks:
                try:
                    symbol = stock.get('symbol')
                    exchange = stock.get('exchange')
                    
                    # Get technical analysis
                    from research.technical_analyzer import TechnicalAnalyzer
                    tech_analyzer = TechnicalAnalyzer(self.db)
                    
                    technical_data = tech_analyzer.get_latest_analysis(symbol, exchange)
                    
                    # Get prediction
                    from ml.prediction.daily_predictor import DailyPredictor
                    predictor = DailyPredictor(self.db)
                    
                    prediction = predictor.get_latest_prediction(symbol, exchange)
                    
                    # Create stock data
                    stock_data = {
                        "symbol": symbol,
                        "exchange": exchange,
                        "name": stock.get('name', symbol),
                        "sector": stock.get('sector', 'Unknown'),
                        "technical": {
                            "trend": technical_data.get('trend', 'neutral'),
                            "rsi": technical_data.get('rsi', 50),
                            "support": technical_data.get('support', 0),
                            "resistance": technical_data.get('resistance', 0)
                        },
                        "prediction": {
                            "direction": prediction.get('prediction', 'unknown'),
                            "confidence": prediction.get('confidence', 0),
                            "price_target": prediction.get('target_price', 0)
                        } if prediction else None
                    }
                    
                    watchlist_data.append(stock_data)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing stock {stock.get('symbol', 'unknown')}: {e}")
            
            # Sort by prediction confidence (highest first)
            watchlist_data.sort(key=lambda x: (x.get('prediction', {}).get('confidence', 0) if x.get('prediction') else 0), reverse=True)
            
            return watchlist_data
            
        except Exception as e:
            self.logger.error(f"Error getting watchlist analysis: {e}")
            return None
            
    def _get_economic_events(self):
        """
        Get economic calendar events.
        
        Returns:
            dict: Economic events data
        """
        try:
            self.logger.info("Getting economic events")
            
            # Import required modules
            from data.global_markets.economic_calendar import EconomicCalendar
            
            # Initialize calendar
            calendar = EconomicCalendar(self.db)
            
            # Get today's events
            today = datetime.now().date()
            
            events = calendar.get_events_for_date(today)
            
            # Categorize events by importance
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
                "all_events": events
            }
            
            return economic_events
            
        except Exception as e:
            self.logger.error(f"Error getting economic events: {e}")
            return None
            
    def _get_earnings_announcements(self):
        """
        Get earnings announcements.
        
        Returns:
            dict: Earnings announcements data
        """
        try:
            self.logger.info("Getting earnings announcements")
            
            if not self.db:
                return None
                
            # Get today's date
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            # Get earnings announcements
            today_earnings = list(self.db.earnings_calendar.find({
                "date": today
            }))
            
            tomorrow_earnings = list(self.db.earnings_calendar.find({
                "date": tomorrow
            }))
            
            # Filter for portfolio stocks
            portfolio_symbols = set()
            
            portfolio_stocks = list(self.db.portfolio_collection.find({
                "status": "active",
                "instrument_type": "equity"
            }))
            
            for stock in portfolio_stocks:
                portfolio_symbols.add(stock.get('symbol'))
                
            # Filter earnings
            portfolio_today = [e for e in today_earnings if e.get('symbol') in portfolio_symbols]
            portfolio_tomorrow = [e for e in tomorrow_earnings if e.get('symbol') in portfolio_symbols]
            
            # Create earnings data
            earnings_data = {
                "today": {
                    "all": today_earnings,
                    "portfolio": portfolio_today
                },
                "tomorrow": {
                    "all": tomorrow_earnings,
                    "portfolio": portfolio_tomorrow
                }
            }
            
            return earnings_data
            
        except Exception as e:
            self.logger.error(f"Error getting earnings announcements: {e}")
            return None
            
    def _get_trading_predictions(self):
        """
        Get trading predictions.
        
        Returns:
            dict: Trading predictions data
        """
        try:
            self.logger.info("Getting trading predictions")
            
            if not self.db:
                return None
                
            # Import required modules
            from ml.prediction.daily_predictor import DailyPredictor
            
            # Initialize predictor
            predictor = DailyPredictor(self.db)
            
            # Get latest predictions
            predictions = predictor.get_all_predictions()
            
            # Filter by confidence
            min_confidence = self.config['min_confidence']
            high_confidence = [p for p in predictions if p.get('confidence', 0) >= min_confidence]
            
            # Sort by confidence (highest first)
            high_confidence.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Limit to max predictions
            top_predictions = high_confidence[:self.config['max_predictions']]
            
            # Categorize predictions
            bullish = [p for p in top_predictions if p.get('prediction') == 'up']
            bearish = [p for p in top_predictions if p.get('prediction') == 'down']
            
            # Create trading predictions data
            predictions_data = {
                "all": predictions,
                "high_confidence": high_confidence,
                "top": top_predictions,
                "bullish": bullish,
                "bearish": bearish
            }
            
            return predictions_data
            
        except Exception as e:
            self.logger.error(f"Error getting trading predictions: {e}")
            return None
            
    def _get_market_news(self):
        """
        Get market news.
        
        Returns:
            dict: Market news data
        """
        try:
            self.logger.info("Getting market news")
            
            if not self.db:
                return None
                
            # Get today's date
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday = today - timedelta(days=1)
            
            # Get recent news
            cursor = self.db.news_data.find({
                "published_date": {"$gte": yesterday, "$lt": today + timedelta(days=1)}
            }).sort("published_date", -1)
            
            recent_news = list(cursor)
            
            # Get top news by sentiment
            positive_news = sorted(
                [n for n in recent_news if n.get('sentiment_score', 0) > 0.5],
                key=lambda x: x.get('sentiment_score', 0),
                reverse=True
            )
            
            negative_news = sorted(
                [n for n in recent_news if n.get('sentiment_score', 0) < -0.5],
                key=lambda x: x.get('sentiment_score', 0)
            )
            
            # Limit to max items
            top_news = (positive_news + negative_news)[:self.config['max_news_items']]
            
            # Sort by time (newest first)
            top_news.sort(key=lambda x: x.get('published_date', datetime.now()), reverse=True)
            
            # Create news data
            news_data = {
                "recent": recent_news,
                "top": top_news,
                "positive": positive_news[:5],
                "negative": negative_news[:5]
            }
            
            return news_data
            
        except Exception as e:
            self.logger.error(f"Error getting market news: {e}")
            return None
            
    def _generate_market_outlook(self, global_markets, sector_analysis, economic_events, news):
        """
        Generate market outlook analysis.
        
        Args:
            global_markets (dict): Global market data
            sector_analysis (dict): Sector analysis data
            economic_events (dict): Economic events data
            news (dict): Market news data
            
        Returns:
            dict: Market outlook data
        """
        try:
            self.logger.info("Generating market outlook")
            
            # Initialize factors
            factors = []
            sentiment_scores = []
            
            # Global markets factor
            if global_markets:
                sentiment = global_markets.get('sentiment', 'neutral')
                
                if sentiment == 'strongly bullish':
                    factors.append("Global markets showing strong bullish sentiment")
                    sentiment_scores.append(1.0)
                elif sentiment == 'bullish':
                    factors.append("Global markets showing bullish sentiment")
                    sentiment_scores.append(0.7)
                elif sentiment == 'strongly bearish':
                    factors.append("Global markets showing strong bearish sentiment")
                    sentiment_scores.append(-1.0)
                elif sentiment == 'bearish':
                    factors.append("Global markets showing bearish sentiment")
                    sentiment_scores.append(-0.7)
                else:
                    factors.append("Global markets showing mixed sentiment")
                    sentiment_scores.append(0)
                    
            # Sector rotation factor
            if sector_analysis:
                rotation_pattern = sector_analysis.get('rotation_pattern', 'neutral')
                
                if rotation_pattern == 'risk-on':
                    factors.append("Sector rotation showing risk-on behavior")
                    sentiment_scores.append(0.8)
                elif rotation_pattern == 'risk-off':
                    factors.append("Sector rotation showing risk-off behavior")
                    sentiment_scores.append(-0.8)
                elif rotation_pattern == 'broad advance':
                    factors.append("Sectors showing broad advance")
                    sentiment_scores.append(0.9)
                elif rotation_pattern == 'broad decline':
                    factors.append("Sectors showing broad decline")
                    sentiment_scores.append(-0.9)
                else:
                    factors.append(f"Sector rotation pattern: {rotation_pattern}")
                    sentiment_scores.append(0)
                    
            # Economic events factor
            if economic_events:
                high_importance = economic_events.get('high_importance', [])
                
                if high_importance:
                    factors.append(f"{len(high_importance)} high-importance economic events today")
                    sentiment_scores.append(0)  # Neutral as impact depends on the results
                    
            # News sentiment factor
            if news:
                positive_news = news.get('positive', [])
                negative_news = news.get('negative', [])
                
                if len(positive_news) > len(negative_news) and positive_news:
                    factors.append("News sentiment primarily positive")
                    sentiment_scores.append(0.6)
                elif len(negative_news) > len(positive_news) and negative_news:
                    factors.append("News sentiment primarily negative")
                    sentiment_scores.append(-0.6)
                elif positive_news or negative_news:
                    factors.append("Mixed news sentiment")
                    sentiment_scores.append(0)
                    
            # Determine overall outlook
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                if avg_sentiment > 0.7:
                    outlook = "strongly bullish"
                elif avg_sentiment > 0.3:
                    outlook = "bullish"
                elif avg_sentiment < -0.7:
                    outlook = "strongly bearish"
                elif avg_sentiment < -0.3:
                    outlook = "bearish"
                else:
                    outlook = "neutral"
            else:
                outlook = "neutral"
                
            # Create outlook data
            outlook_data = {
                "outlook": outlook,
                "factors": factors,
                "sentiment_score": avg_sentiment if sentiment_scores else 0
            }
            
            return outlook_data
            
        except Exception as e:
            self.logger.error(f"Error generating market outlook: {e}")
            return {"outlook": "neutral", "factors": [], "sentiment_score": 0}
            
    def _generate_trading_plan(self, predictions, watchlist, earnings):
        """
        Generate trading plan recommendations.
        
        Args:
            predictions (dict): Trading predictions data
            watchlist (list): Watchlist analysis data
            earnings (dict): Earnings announcements data
            
        Returns:
            dict: Trading plan data
        """
        try:
            self.logger.info("Generating trading plan")
            
            # Initialize recommendations
            long_ideas = []
            short_ideas = []
            watch_ideas = []
            risk_warnings = []
            
            # Add prediction-based ideas
            if predictions:
                bullish = predictions.get('bullish', [])
                bearish = predictions.get('bearish', [])
                
                # Add top bullish ideas
                for pred in bullish[:3]:
                    symbol = pred.get('symbol')
                    confidence = pred.get('confidence', 0)
                    target = pred.get('target_price', 0)
                    
                    if confidence >= 0.7:
                        long_ideas.append({
                            "symbol": symbol,
                            "reason": f"High confidence bullish prediction ({confidence:.1%})",
                            "target": target,
                            "confidence": confidence
                        })
                    elif confidence >= 0.6:
                        watch_ideas.append({
                            "symbol": symbol,
                            "reason": f"Moderate confidence bullish prediction ({confidence:.1%})",
                            "watch_for": "Confirmation of upward movement",
                            "confidence": confidence
                        })
                        
                # Add top bearish ideas
                for pred in bearish[:3]:
                    symbol = pred.get('symbol')
                    confidence = pred.get('confidence', 0)
                    target = pred.get('target_price', 0)
                    
                    if confidence >= 0.7:
                        short_ideas.append({
                            "symbol": symbol,
                            "reason": f"High confidence bearish prediction ({confidence:.1%})",
                            "target": target,
                            "confidence": confidence
                        })
                    elif confidence >= 0.6:
                        watch_ideas.append({
                            "symbol": symbol,
                            "reason": f"Moderate confidence bearish prediction ({confidence:.1%})",
                            "watch_for": "Confirmation of downward movement",
                            "confidence": confidence
                        })
                        
            # Add earnings-based ideas
            if earnings:
                portfolio_today = earnings.get('today', {}).get('portfolio', [])
                portfolio_tomorrow = earnings.get('tomorrow', {}).get('portfolio', [])
                
                # Add earnings risk warnings
                for earning in portfolio_today:
                    symbol = earning.get('symbol')
                    time = earning.get('time', 'Unknown')
                    
                    risk_warnings.append({
                        "symbol": symbol,
                        "warning": f"Earnings announcement today ({time})",
                        "action": "Consider reducing position size or hedging"
                    })
                    
                # Add watch ideas for tomorrow's earnings
                for earning in portfolio_tomorrow:
                    symbol = earning.get('symbol')
                    
                    watch_ideas.append({
                        "symbol": symbol,
                        "reason": "Earnings announcement tomorrow",
                        "watch_for": "Pre-earnings movement",
                        "confidence": 0.5
                    })
                    
            # Add watchlist-based ideas
            if watchlist:
                for stock in watchlist:
                    symbol = stock.get('symbol')
                    technical = stock.get('technical', {})
                    prediction = stock.get('prediction', {})
                    
                    # Check for technical setups
                    trend = technical.get('trend', 'neutral')
                    rsi = technical.get('rsi', 50)
                    
                    if trend == 'strong_uptrend' and prediction and prediction.get('direction') == 'up':
                        if symbol not in [idea.get('symbol') for idea in long_ideas]:
                            long_ideas.append({
                                "symbol": symbol,
                                "reason": "Strong uptrend with bullish prediction",
                                "target": prediction.get('price_target', 0),
                                "confidence": prediction.get('confidence', 0.6)
                            })
                    elif trend == 'strong_downtrend' and prediction and prediction.get('direction') == 'down':
                        if symbol not in [idea.get('symbol') for idea in short_ideas]:
                            short_ideas.append({
                                "symbol": symbol,
                                "reason": "Strong downtrend with bearish prediction",
                                "target": prediction.get('price_target', 0),
                                "confidence": prediction.get('confidence', 0.6)
                            })
                    elif rsi > 70 and prediction and prediction.get('direction') == 'down':
                        if symbol not in [idea.get('symbol') for idea in short_ideas]:
                            short_ideas.append({
                                "symbol": symbol,
                                "reason": f"Overbought (RSI: {rsi:.1f}) with bearish prediction",
                                "target": prediction.get('price_target', 0),
                                "confidence": prediction.get('confidence', 0.6)
                            })
                    elif rsi < 30 and prediction and prediction.get('direction') == 'up':
                        if symbol not in [idea.get('symbol') for idea in long_ideas]:
                            long_ideas.append({
                                "symbol": symbol,
                                "reason": f"Oversold (RSI: {rsi:.1f}) with bullish prediction",
                                "target": prediction.get('price_target', 0),
                                "confidence": prediction.get('confidence', 0.6)
                            })
            
            # Sort ideas by confidence
            long_ideas.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            short_ideas.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            watch_ideas.sort(key=lambda x: x.get('confidence', 0), reverse=True)
            
            # Create trading plan
            trading_plan = {
                "long_ideas": long_ideas,
                "short_ideas": short_ideas,
                "watch_ideas": watch_ideas,
                "risk_warnings": risk_warnings
            }
            
            return trading_plan
            
        except Exception as e:
            self.logger.error(f"Error generating trading plan: {e}")
            return {
                "long_ideas": [],
                "short_ideas": [],
                "watch_ideas": [],
                "risk_warnings": []
            }
            
    def _format_report(self, report_data):
        """
        Format report for delivery.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Formatted report
        """
        try:
            format_type = self.config['format']
            
            if format_type == 'markdown':
                return self._format_markdown_report(report_data)
            elif format_type == 'html':
                return self._format_html_report(report_data)
            else:
                return self._format_markdown_report(report_data)
                
        except Exception as e:
            self.logger.error(f"Error formatting report: {e}")
            return f"Error formatting report: {e}"
            
    def _format_markdown_report(self, report_data):
        """
        Format report in Markdown.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Markdown formatted report
        """
        try:
            today_str = report_data['date'].strftime("%A, %B %d, %Y")
            
            # Start with header
            md = f"# Morning Briefing - {today_str}\n\n"
            
            # Market outlook
            outlook = report_data.get('market_outlook', {})
            outlook_sentiment = outlook.get('outlook', 'neutral').title()
            
            md += f"## Market Outlook: {outlook_sentiment}\n\n"
            
            factors = outlook.get('factors', [])
            if factors:
                md += "Key factors influencing today's market:\n"
                for factor in factors:
                    md += f"* {factor}\n"
                md += "\n"
                
            # Global markets
            # Global markets
            global_markets = report_data.get('global_markets')
            if global_markets and self.config['include_global_markets']:
                md += "## Global Markets\n\n"
                
                # US markets
                us_markets = global_markets.get('us_markets', {})
                if us_markets:
                    md += "### US Markets\n"
                    for name, data in us_markets.items():
                        change = data.get('percent_change', 0)
                        direction = "▲" if change > 0 else "▼" if change < 0 else "■"
                        md += f"* {name}: {direction} {abs(change):.2f}%\n"
                    md += "\n"
                
                # Asian markets
                asian_markets = global_markets.get('asian_markets', {})
                if asian_markets:
                    md += "### Asian Markets\n"
                    for name, data in asian_markets.items():
                        change = data.get('percent_change', 0)
                        direction = "▲" if change > 0 else "▼" if change < 0 else "■"
                        md += f"* {name}: {direction} {abs(change):.2f}%\n"
                    md += "\n"
                
                # European markets
                european_markets = global_markets.get('european_markets', {})
                if european_markets:
                    md += "### European Markets\n"
                    for name, data in european_markets.items():
                        change = data.get('percent_change', 0)
                        direction = "▲" if change > 0 else "▼" if change < 0 else "■"
                        md += f"* {name}: {direction} {abs(change):.2f}%\n"
                    md += "\n"
            
            # Sector analysis
            sector_analysis = report_data.get('sector_analysis')
            if sector_analysis and self.config['include_sector_analysis']:
                md += "## Sector Analysis\n\n"
                
                # Sector rotation
                rotation = sector_analysis.get('rotation_pattern', 'neutral')
                md += f"Current rotation pattern: **{rotation}**\n\n"
                
                # Top performing sectors
                top_sectors = sector_analysis.get('top_sectors', [])
                if top_sectors:
                    md += "### Top Performing Sectors\n"
                    for sector in top_sectors:
                        sector_name = sector.get('name', 'Unknown')
                        change = sector.get('change', 0)
                        md += f"* {sector_name}: +{change:.2f}%\n"
                    md += "\n"
                
                # Bottom performing sectors
                bottom_sectors = sector_analysis.get('bottom_sectors', [])
                if bottom_sectors:
                    md += "### Underperforming Sectors\n"
                    for sector in bottom_sectors:
                        sector_name = sector.get('name', 'Unknown')
                        change = sector.get('change', 0)
                        md += f"* {sector_name}: {change:.2f}%\n"
                    md += "\n"
            
            # Economic events
            economic_events = report_data.get('economic_events')
            if economic_events and self.config['include_economic_events']:
                md += "## Economic Calendar\n\n"
                
                # High importance events
                high_importance = economic_events.get('high_importance', [])
                if high_importance:
                    md += "### High Importance Events\n"
                    for event in high_importance:
                        time = event.get('time', 'TBA')
                        title = event.get('title', 'Unknown')
                        country = event.get('country', '')
                        md += f"* {time} - {country} - {title}\n"
                    md += "\n"
                
                # Medium importance events
                medium_importance = economic_events.get('medium_importance', [])
                if medium_importance and len(medium_importance) <= 5:  # Only show if not too many
                    md += "### Medium Importance Events\n"
                    for event in medium_importance:
                        time = event.get('time', 'TBA')
                        title = event.get('title', 'Unknown')
                        country = event.get('country', '')
                        md += f"* {time} - {country} - {title}\n"
                    md += "\n"
            
            # Earnings announcements
            earnings = report_data.get('earnings')
            if earnings and self.config['include_earnings']:
                md += "## Earnings Announcements\n\n"
                
                # Portfolio stocks with earnings today
                portfolio_today = earnings.get('today', {}).get('portfolio', [])
                if portfolio_today:
                    md += "### Portfolio Stocks Reporting Today ⚠️\n"
                    for earning in portfolio_today:
                        symbol = earning.get('symbol', 'Unknown')
                        time = earning.get('time', 'Unknown')
                        md += f"* {symbol} - {time}\n"
                    md += "\n"
                
                # Notable earnings today
                all_today = earnings.get('today', {}).get('all', [])
                if all_today and len(all_today) <= 10:  # Only show if not too many
                    md += "### Notable Earnings Today\n"
                    for earning in all_today[:5]:  # Show top 5
                        symbol = earning.get('symbol', 'Unknown')
                        time = earning.get('time', 'Unknown')
                        md += f"* {symbol} - {time}\n"
                    md += "\n"
                
                # Portfolio stocks with earnings tomorrow
                portfolio_tomorrow = earnings.get('tomorrow', {}).get('portfolio', [])
                if portfolio_tomorrow:
                    md += "### Portfolio Stocks Reporting Tomorrow\n"
                    for earning in portfolio_tomorrow:
                        symbol = earning.get('symbol', 'Unknown')
                        md += f"* {symbol}\n"
                    md += "\n"
            
            # Trading plan
            trading_plan = report_data.get('trading_plan', {})
            md += "## Trading Plan\n\n"
            
            # Long ideas
            long_ideas = trading_plan.get('long_ideas', [])
            if long_ideas:
                md += "### Long Ideas\n"
                for idea in long_ideas:
                    symbol = idea.get('symbol', 'Unknown')
                    reason = idea.get('reason', 'N/A')
                    target = idea.get('target', 0)
                    
                    md += f"* **{symbol}**: {reason}"
                    if target > 0:
                        md += f" (Target: ₹{target:.2f})"
                    md += "\n"
                md += "\n"
            
            # Short ideas
            short_ideas = trading_plan.get('short_ideas', [])
            if short_ideas:
                md += "### Short Ideas\n"
                for idea in short_ideas:
                    symbol = idea.get('symbol', 'Unknown')
                    reason = idea.get('reason', 'N/A')
                    target = idea.get('target', 0)
                    
                    md += f"* **{symbol}**: {reason}"
                    if target > 0:
                        md += f" (Target: ₹{target:.2f})"
                    md += "\n"
                md += "\n"
            
            # Watch ideas
            watch_ideas = trading_plan.get('watch_ideas', [])
            if watch_ideas:
                md += "### Watch List\n"
                for idea in watch_ideas:
                    symbol = idea.get('symbol', 'Unknown')
                    reason = idea.get('reason', 'N/A')
                    watch_for = idea.get('watch_for', '')
                    
                    md += f"* **{symbol}**: {reason}"
                    if watch_for:
                        md += f" (Watch for: {watch_for})"
                    md += "\n"
                md += "\n"
            
            # Risk warnings
            risk_warnings = trading_plan.get('risk_warnings', [])
            if risk_warnings:
                md += "### Risk Warnings ⚠️\n"
                for warning in risk_warnings:
                    symbol = warning.get('symbol', 'Unknown')
                    warning_text = warning.get('warning', 'N/A')
                    action = warning.get('action', '')
                    
                    md += f"* **{symbol}**: {warning_text}"
                    if action:
                        md += f" - {action}"
                    md += "\n"
                md += "\n"
            
            # News highlights
            news = report_data.get('news')
            if news and self.config['include_news']:
                md += "## Market News\n\n"
                
                top_news = news.get('top', [])
                if top_news:
                    for item in top_news[:5]:  # Show top 5
                        title = item.get('title', 'Unknown')
                        source = item.get('source', 'Unknown')
                        sentiment = item.get('sentiment_score', 0)
                        
                        if sentiment > 0.3:
                            sentiment_marker = "📈"
                        elif sentiment < -0.3:
                            sentiment_marker = "📉"
                        else:
                            sentiment_marker = "📊"
                        
                        md += f"* {sentiment_marker} **{title}** - *{source}*\n"
                    md += "\n"
            
            # Footer
            md += "---\n"
            md += "*This briefing was automatically generated. Data may be delayed.*\n"
            
            return md
            
        except Exception as e:
            self.logger.error(f"Error formatting markdown report: {e}")
            return f"Error formatting report: {e}"
            
    def _format_html_report(self, report_data):
        """
        Format report in HTML.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: HTML formatted report
        """
        try:
            today_str = report_data['date'].strftime("%A, %B %d, %Y")
            
            # Start with header
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Morning Briefing - {today_str}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333366; }}
                    h2 {{ color: #335588; margin-top: 20px; border-bottom: 1px solid #ccc; }}
                    h3 {{ color: #337799; }}
                    .up {{ color: green; }}
                    .down {{ color: red; }}
                    .neutral {{ color: gray; }}
                    .warning {{ color: orange; }}
                    .section {{ margin-bottom: 20px; }}
                    .footer {{ margin-top: 40px; font-size: 12px; color: #777; }}
                </style>
            </head>
            <body>
                <h1>Morning Briefing - {today_str}</h1>
            """
            
            # Market outlook
            outlook = report_data.get('market_outlook', {})
            outlook_sentiment = outlook.get('outlook', 'neutral').title()
            
            sentiment_class = ""
            if "bullish" in outlook_sentiment.lower():
                sentiment_class = "up"
            elif "bearish" in outlook_sentiment.lower():
                sentiment_class = "down"
            else:
                sentiment_class = "neutral"
            
            html += f"""
                <div class="section">
                    <h2>Market Outlook: <span class="{sentiment_class}">{outlook_sentiment}</span></h2>
            """
            
            factors = outlook.get('factors', [])
            if factors:
                html += "<p>Key factors influencing today's market:</p><ul>"
                for factor in factors:
                    html += f"<li>{factor}</li>"
                html += "</ul>"
            
            html += "</div>"
            
            # Global markets
            global_markets = report_data.get('global_markets')
            if global_markets and self.config['include_global_markets']:
                html += """
                <div class="section">
                    <h2>Global Markets</h2>
                """
                
                # US markets
                us_markets = global_markets.get('us_markets', {})
                if us_markets:
                    html += "<h3>US Markets</h3><ul>"
                    for name, data in us_markets.items():
                        change = data.get('percent_change', 0)
                        class_name = "up" if change > 0 else "down" if change < 0 else "neutral"
                        direction = "▲" if change > 0 else "▼" if change < 0 else "■"
                        html += f'<li>{name}: <span class="{class_name}">{direction} {abs(change):.2f}%</span></li>'
                    html += "</ul>"
                
                # Asian markets
                asian_markets = global_markets.get('asian_markets', {})
                if asian_markets:
                    html += "<h3>Asian Markets</h3><ul>"
                    for name, data in asian_markets.items():
                        change = data.get('percent_change', 0)
                        class_name = "up" if change > 0 else "down" if change < 0 else "neutral"
                        direction = "▲" if change > 0 else "▼" if change < 0 else "■"
                        html += f'<li>{name}: <span class="{class_name}">{direction} {abs(change):.2f}%</span></li>'
                    html += "</ul>"
                
                # European markets
                european_markets = global_markets.get('european_markets', {})
                if european_markets:
                    html += "<h3>European Markets</h3><ul>"
                    for name, data in european_markets.items():
                        change = data.get('percent_change', 0)
                        class_name = "up" if change > 0 else "down" if change < 0 else "neutral"
                        direction = "▲" if change > 0 else "▼" if change < 0 else "■"
                        html += f'<li>{name}: <span class="{class_name}">{direction} {abs(change):.2f}%</span></li>'
                    html += "</ul>"
                
                html += "</div>"
            
            # Sector analysis
            sector_analysis = report_data.get('sector_analysis')
            if sector_analysis and self.config['include_sector_analysis']:
                html += """
                <div class="section">
                    <h2>Sector Analysis</h2>
                """
                
                # Sector rotation
                rotation = sector_analysis.get('rotation_pattern', 'neutral')
                html += f"<p>Current rotation pattern: <strong>{rotation}</strong></p>"
                
                # Top performing sectors
                top_sectors = sector_analysis.get('top_sectors', [])
                if top_sectors:
                    html += "<h3>Top Performing Sectors</h3><ul>"
                    for sector in top_sectors:
                        sector_name = sector.get('name', 'Unknown')
                        change = sector.get('change', 0)
                        html += f'<li>{sector_name}: <span class="up">+{change:.2f}%</span></li>'
                    html += "</ul>"
                
                # Bottom performing sectors
                bottom_sectors = sector_analysis.get('bottom_sectors', [])
                if bottom_sectors:
                    html += "<h3>Underperforming Sectors</h3><ul>"
                    for sector in bottom_sectors:
                        sector_name = sector.get('name', 'Unknown')
                        change = sector.get('change', 0)
                        html += f'<li>{sector_name}: <span class="down">{change:.2f}%</span></li>'
                    html += "</ul>"
                
                html += "</div>"
            
            # Economic events
            economic_events = report_data.get('economic_events')
            if economic_events and self.config['include_economic_events']:
                html += """
                <div class="section">
                    <h2>Economic Calendar</h2>
                """
                
                # High importance events
                high_importance = economic_events.get('high_importance', [])
                if high_importance:
                    html += "<h3>High Importance Events</h3><ul>"
                    for event in high_importance:
                        time = event.get('time', 'TBA')
                        title = event.get('title', 'Unknown')
                        country = event.get('country', '')
                        html += f"<li><strong>{time}</strong> - {country} - {title}</li>"
                    html += "</ul>"
                
                # Medium importance events
                medium_importance = economic_events.get('medium_importance', [])
                if medium_importance and len(medium_importance) <= 5:  # Only show if not too many
                    html += "<h3>Medium Importance Events</h3><ul>"
                    for event in medium_importance:
                        time = event.get('time', 'TBA')
                        title = event.get('title', 'Unknown')
                        country = event.get('country', '')
                        html += f"<li><strong>{time}</strong> - {country} - {title}</li>"
                    html += "</ul>"
                
                html += "</div>"
            
            # Earnings announcements
            earnings = report_data.get('earnings')
            if earnings and self.config['include_earnings']:
                html += """
                <div class="section">
                    <h2>Earnings Announcements</h2>
                """
                
                # Portfolio stocks with earnings today
                portfolio_today = earnings.get('today', {}).get('portfolio', [])
                if portfolio_today:
                    html += '<h3 class="warning">Portfolio Stocks Reporting Today ⚠️</h3><ul>'
                    for earning in portfolio_today:
                        symbol = earning.get('symbol', 'Unknown')
                        time = earning.get('time', 'Unknown')
                        html += f"<li><strong>{symbol}</strong> - {time}</li>"
                    html += "</ul>"
                
                # Notable earnings today
                all_today = earnings.get('today', {}).get('all', [])
                if all_today and len(all_today) <= 10:  # Only show if not too many
                    html += "<h3>Notable Earnings Today</h3><ul>"
                    for earning in all_today[:5]:  # Show top 5
                        symbol = earning.get('symbol', 'Unknown')
                        time = earning.get('time', 'Unknown')
                        html += f"<li><strong>{symbol}</strong> - {time}</li>"
                    html += "</ul>"
                
                # Portfolio stocks with earnings tomorrow
                portfolio_tomorrow = earnings.get('tomorrow', {}).get('portfolio', [])
                if portfolio_tomorrow:
                    html += "<h3>Portfolio Stocks Reporting Tomorrow</h3><ul>"
                    for earning in portfolio_tomorrow:
                        symbol = earning.get('symbol', 'Unknown')
                        html += f"<li><strong>{symbol}</strong></li>"
                    html += "</ul>"
                
                html += "</div>"
            
            # Trading plan
            trading_plan = report_data.get('trading_plan', {})
            html += """
                <div class="section">
                    <h2>Trading Plan</h2>
            """
            
            # Long ideas
            long_ideas = trading_plan.get('long_ideas', [])
            if long_ideas:
                html += "<h3>Long Ideas</h3><ul>"
                for idea in long_ideas:
                    symbol = idea.get('symbol', 'Unknown')
                    reason = idea.get('reason', 'N/A')
                    target = idea.get('target', 0)
                    
                    html += f"<li><strong>{symbol}</strong>: {reason}"
                    if target > 0:
                        html += f" (Target: ₹{target:.2f})"
                    html += "</li>"
                html += "</ul>"
            
            # Short ideas
            short_ideas = trading_plan.get('short_ideas', [])
            if short_ideas:
                html += "<h3>Short Ideas</h3><ul>"
                for idea in short_ideas:
                    symbol = idea.get('symbol', 'Unknown')
                    reason = idea.get('reason', 'N/A')
                    target = idea.get('target', 0)
                    
                    html += f"<li><strong>{symbol}</strong>: {reason}"
                    if target > 0:
                        html += f" (Target: ₹{target:.2f})"
                    html += "</li>"
                html += "</ul>"
            
            # Watch ideas
            watch_ideas = trading_plan.get('watch_ideas', [])
            if watch_ideas:
                html += "<h3>Watch List</h3><ul>"
                for idea in watch_ideas:
                    symbol = idea.get('symbol', 'Unknown')
                    reason = idea.get('reason', 'N/A')
                    watch_for = idea.get('watch_for', '')
                    
                    html += f"<li><strong>{symbol}</strong>: {reason}"
                    if watch_for:
                        html += f" (Watch for: {watch_for})"
                    html += "</li>"
                html += "</ul>"
            
            # Risk warnings
            risk_warnings = trading_plan.get('risk_warnings', [])
            if risk_warnings:
                html += '<h3 class="warning">Risk Warnings ⚠️</h3><ul>'
                for warning in risk_warnings:
                    symbol = warning.get('symbol', 'Unknown')
                    warning_text = warning.get('warning', 'N/A')
                    action = warning.get('action', '')
                    
                    html += f"<li><strong>{symbol}</strong>: {warning_text}"
                    if action:
                        html += f" - {action}"
                    html += "</li>"
                html += "</ul>"
            
            html += "</div>"
            
            # News highlights
            news = report_data.get('news')
            if news and self.config['include_news']:
                html += """
                <div class="section">
                    <h2>Market News</h2>
                    <ul>
                """
                
                top_news = news.get('top', [])
                if top_news:
                    for item in top_news[:5]:  # Show top 5
                        title = item.get('title', 'Unknown')
                        source = item.get('source', 'Unknown')
                        sentiment = item.get('sentiment_score', 0)
                        
                        if sentiment > 0.3:
                            sentiment_marker = "📈"
                            sentiment_class = "up"
                        elif sentiment < -0.3:
                            sentiment_marker = "📉"
                            sentiment_class = "down"
                        else:
                            sentiment_marker = "📊"
                            sentiment_class = "neutral"
                        
                        html += f'<li>{sentiment_marker} <strong>{title}</strong> - <em>{source}</em></li>'
                
                html += """
                    </ul>
                </div>
                """
            
            # Footer
            html += """
                <div class="footer">
                    <hr>
                    <p>This briefing was automatically generated. Data may be delayed.</p>
                </div>
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error formatting HTML report: {e}")
            return f"<html><body><h1>Error</h1><p>{e}</p></body></html>"
            
    def _format_non_trading_day_report(self, report_data):
        """
        Format non-trading day report.
        
        Args:
            report_data (dict): Report data
            
        Returns:
            str: Formatted report
        """
        try:
            format_type = self.config['format']
            
            if format_type == 'markdown':
                today_str = report_data['date'].strftime("%A, %B %d, %Y")
                
                md = f"# Morning Briefing - {today_str}\n\n"
                md += "## Market Status: CLOSED\n\n"
                md += f"{report_data.get('message', 'Markets are closed today.')}\n\n"
                
                return md
            elif format_type == 'html':
                today_str = report_data['date'].strftime("%A, %B %d, %Y")
                
                html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Morning Briefing - {today_str}</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1 {{ color: #333366; }}
                        h2 {{ color: #335588; margin-top: 20px; border-bottom: 1px solid #ccc; }}
                        .closed {{ color: orange; }}
                        .footer {{ margin-top: 40px; font-size: 12px; color: #777; }}
                    </style>
                </head>
                <body>
                    <h1>Morning Briefing - {today_str}</h1>
                    <h2>Market Status: <span class="closed">CLOSED</span></h2>
                    <p>{report_data.get('message', 'Markets are closed today.')}</p>
                    <div class="footer">
                        <hr>
                        <p>This briefing was automatically generated.</p>
                    </div>
                </body>
                </html>
                """
                
                return html
            else:
                return report_data.get('message', 'Markets are closed today.')
                
        except Exception as e:
            self.logger.error(f"Error formatting non-trading day report: {e}")
            return f"Error formatting report: {e}"
            
    def _save_report_to_database(self, report_data, formatted_report):
        """
        Save report to database.
        
        Args:
            report_data (dict): Report data
            formatted_report (str): Formatted report
        """
        try:
            if not self.db:
                return
                
            # Create report document
            report_doc = {
                "type": "morning_briefing",
                "date": datetime.now(),
                "is_trading_day": report_data.get('is_trading_day', False),
                "report_data": report_data,
                "formatted_report": formatted_report,
                "format": self.config['format']
            }
            
            # Store in database
            self.db.reports.insert_one(report_doc)
            
            self.logger.info("Morning briefing saved to database")
            
        except Exception as e:
            self.logger.error(f"Error saving report to database: {e}")
            
    def _deliver_report(self, formatted_report):
        """
        Deliver report via configured methods.
        
        Args:
            formatted_report (str): Formatted report
        """
        try:
            # Check delivery methods
            delivery_methods = self.config['delivery_methods']
            
            if not delivery_methods:
                self.logger.warning("No delivery methods configured")
                return
                
            # Import report distributor
            from communication.report_distributor import ReportDistributor
            
            # Initialize distributor
            distributor = ReportDistributor(self.db)
            
            # WhatsApp delivery
            if 'whatsapp' in delivery_methods:
                try:
                    # Get WhatsApp number
                    whatsapp_number = self.config['whatsapp_number']
                    
                    # Send via WhatsApp
                    distributor.distribute_via_whatsapp(
                        formatted_report,
                        recipient=whatsapp_number
                    )
                    
                    self.logger.info("Morning briefing delivered via WhatsApp")
                    
                except Exception as e:
                    self.logger.error(f"Error delivering via WhatsApp: {e}")
            
            # Email delivery
            if 'email' in delivery_methods:
                try:
                    # Get email recipients
                    recipients = self.config['email_recipients']
                    
                    # Format subject
                    today_str = datetime.now().strftime("%d-%b-%Y")
                    subject = f"Morning Trading Briefing - {today_str}"
                    
                    # Send via email
                    distributor.distribute_via_email(
                        content=formatted_report,
                        subject=subject,
                        recipients=recipients,
                        is_html=(self.config['format'] == 'html')
                    )
                    
                    self.logger.info("Morning briefing delivered via email")
                    
                except Exception as e:
                    self.logger.error(f"Error delivering via email: {e}")
            
        except Exception as e:
            self.logger.error(f"Error delivering report: {e}")