"""
Event-Driven Strategy Module

This module implements event-driven trading strategies based on:
- Earnings announcements
- Corporate actions (dividends, splits, buybacks)
- News sentiment analysis
- Economic events and data releases
- Regulatory events
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import re
from nltk.sentiment import SentimentIntensityAnalyzer

class EventDrivenStrategy:
    """
    Implements event-driven trading strategies based on various market events.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the strategy with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Strategy parameters (configurable)
        self.params = {
            # Earnings parameters
            "pre_earnings_days": 5,  # Days before earnings to consider
            "post_earnings_days": 3,  # Days after earnings to monitor
            "earnings_volatility_threshold": 0.2,  # Minimum historical earnings volatility
            
            # News parameters
            "news_sentiment_period": 7,  # Days to analyze news sentiment
            "sentiment_threshold": 0.3,  # Sentiment score threshold
            "min_news_count": 3,  # Minimum number of news items required
            
            # Corporate action parameters
            "dividend_lookback": 14,  # Days before ex-dividend to enter
            "split_lookback": 10,  # Days before split to enter
            
            # Economic event parameters
            "economic_event_impact_threshold": "medium",  # Minimum impact level
            "pre_event_entry_days": 2,  # Days before event to enter
            
            # Risk management
            "max_position_size": 0.05,  # Maximum position size as fraction of portfolio
            "earnings_stop_loss": 0.05,  # 5% stop loss for earnings plays
            "news_stop_loss": 0.03,  # 3% stop loss for news-based trades
        }
        
        # Initialize sentiment analyzer if nltk is available
        self.sia = None
        try:
            self.sia = SentimentIntensityAnalyzer()
        except:
            self.logger.warning("NLTK SentimentIntensityAnalyzer not available, using simplified sentiment analysis")
    
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
    
    def analyze_earnings_events(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze upcoming and recent earnings events.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with earnings event analysis
        """
        try:
            # Get earnings calendar data
            earnings_data = self._get_earnings_data(symbol, exchange)
            
            if not earnings_data:
                return {"status": "no_earnings_data"}
            
            # Get the most recent and next earnings dates
            upcoming_earnings = None
            previous_earnings = None
            
            # Sort earnings dates
            earnings_dates = sorted(earnings_data, key=lambda x: x.get("date", datetime.now()))
            
            # Find the next and previous earnings dates
            today = datetime.now()
            for event in earnings_dates:
                event_date = event.get("date")
                if event_date > today:
                    if upcoming_earnings is None or event_date < upcoming_earnings.get("date"):
                        upcoming_earnings = event
                else:
                    if previous_earnings is None or event_date > previous_earnings.get("date"):
                        previous_earnings = event
            
            # Check if we're in the pre-earnings period
            in_pre_earnings = False
            days_to_earnings = None
            
            if upcoming_earnings:
                earnings_date = upcoming_earnings.get("date")
                days_to_earnings = (earnings_date - today).days
                
                if 0 <= days_to_earnings <= self.params["pre_earnings_days"]:
                    in_pre_earnings = True
            
            # Check if we're in the post-earnings period
            in_post_earnings = False
            days_since_earnings = None
            
            if previous_earnings:
                earnings_date = previous_earnings.get("date")
                days_since_earnings = (today - earnings_date).days
                
                if 0 <= days_since_earnings <= self.params["post_earnings_days"]:
                    in_post_earnings = True
            
            # Analyze historical earnings volatility
            historical_reactions = self._analyze_historical_earnings_reactions(symbol, exchange)
            
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            # Determine expected move
            expected_move = None
            expected_move_percent = None
            
            if current_price and historical_reactions and upcoming_earnings:
                avg_reaction = historical_reactions.get("avg_abs_move_percent", 0)
                expected_move_percent = avg_reaction
                expected_move = current_price * avg_reaction / 100
            
            # Generate trading signals
            signals = []
            
            # Pre-earnings signal
            if in_pre_earnings and historical_reactions:
                avg_reaction = historical_reactions.get("avg_abs_move_percent", 0)
                
                # Only generate signal if historical volatility is high enough
                if avg_reaction >= self.params["earnings_volatility_threshold"] * 100:
                    # Determine direction bias based on historical patterns
                    direction = historical_reactions.get("direction_bias")
                    
                    # Generate pre-earnings signal
                    signals.append({
                        "type": "pre_earnings",
                        "direction": direction,
                        "days_to_earnings": days_to_earnings,
                        "expected_move_percent": expected_move_percent,
                        "historical_volatility": avg_reaction,
                        "confidence": min(avg_reaction / 20, 0.8)  # Scale based on historical volatility
                    })
            
            # Post-earnings signal
            if in_post_earnings and previous_earnings:
                actual_eps = previous_earnings.get("actual_eps")
                expected_eps = previous_earnings.get("expected_eps")
                
                if actual_eps is not None and expected_eps is not None:
                    surprise_percent = (actual_eps - expected_eps) / abs(expected_eps) * 100 if expected_eps != 0 else 0
                    
                    # Get post-earnings price reaction
                    price_reaction = self._get_post_earnings_reaction(symbol, exchange, previous_earnings.get("date"))
                    
                    if price_reaction:
                        # Generate post-earnings signal
                        direction = "bullish" if price_reaction > 0 else "bearish"
                        
                        signals.append({
                            "type": "post_earnings",
                            "direction": direction,
                            "days_since_earnings": days_since_earnings,
                            "surprise_percent": surprise_percent,
                            "price_reaction_percent": price_reaction,
                            "actual_eps": actual_eps,
                            "expected_eps": expected_eps,
                            "confidence": min(abs(surprise_percent) / 20, 0.8)  # Scale based on surprise magnitude
                        })
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "upcoming_earnings": upcoming_earnings,
                "previous_earnings": previous_earnings,
                "in_pre_earnings": in_pre_earnings,
                "in_post_earnings": in_post_earnings,
                "days_to_earnings": days_to_earnings,
                "days_since_earnings": days_since_earnings,
                "historical_reactions": historical_reactions,
                "expected_move": expected_move,
                "expected_move_percent": expected_move_percent,
                "signals": signals,
                "current_price": current_price,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing earnings events for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_news_sentiment(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze news sentiment for trading opportunities.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with news sentiment analysis
        """
        try:
            # Get recent news data
            news_data = self._get_recent_news(
                symbol, exchange, days=self.params["news_sentiment_period"]
            )
            
            if not news_data or len(news_data) < self.params["min_news_count"]:
                return {"status": "insufficient_news"}
            
            # Calculate sentiment scores
            sentiment_scores = []
            total_sentiment = 0
            processed_count = 0
            
            for news in news_data:
                title = news.get("title", "")
                description = news.get("description", "")
                
                # Combine title and description for analysis
                content = f"{title}. {description}"
                
                # Calculate sentiment score
                sentiment = self._calculate_sentiment(content)
                total_sentiment += sentiment
                processed_count += 1
                
                news_item = {
                    "title": title,
                    "date": news.get("published_date", news.get("scraped_date")),
                    "source": news.get("source", "Unknown"),
                    "sentiment": sentiment,
                    "url": news.get("url", "")
                }
                
                sentiment_scores.append(news_item)
            
            # Calculate average sentiment
            avg_sentiment = total_sentiment / processed_count if processed_count > 0 else 0
            
            # Determine sentiment trend
            sentiment_trend = "neutral"
            if avg_sentiment > self.params["sentiment_threshold"]:
                sentiment_trend = "positive"
            elif avg_sentiment < -self.params["sentiment_threshold"]:
                sentiment_trend = "negative"
            
            # Count positive, negative, and neutral news
            positive_count = sum(1 for item in sentiment_scores if item["sentiment"] > self.params["sentiment_threshold"])
            negative_count = sum(1 for item in sentiment_scores if item["sentiment"] < -self.params["sentiment_threshold"])
            neutral_count = processed_count - positive_count - negative_count
            
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            # Generate trading signal if sentiment is strong enough
            signal = None
            
            if processed_count >= self.params["min_news_count"]:
                if sentiment_trend == "positive" and positive_count > negative_count:
                    # Check sentiment strength and consistency
                    confidence = min(avg_sentiment * 2, 0.8)  # Scale confidence based on sentiment strength
                    
                    signal = {
                        "type": "news_sentiment",
                        "direction": "bullish",
                        "sentiment_score": avg_sentiment,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "neutral_count": neutral_count,
                        "confidence": confidence
                    }
                    
                elif sentiment_trend == "negative" and negative_count > positive_count:
                    # Check sentiment strength and consistency
                    confidence = min(abs(avg_sentiment) * 2, 0.8)  # Scale confidence based on sentiment strength
                    
                    signal = {
                        "type": "news_sentiment",
                        "direction": "bearish",
                        "sentiment_score": avg_sentiment,
                        "positive_count": positive_count,
                        "negative_count": negative_count,
                        "neutral_count": neutral_count,
                        "confidence": confidence
                    }
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "news_count": processed_count,
                "avg_sentiment": avg_sentiment,
                "sentiment_trend": sentiment_trend,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "recent_news": sentiment_scores[:10],  # Include the 10 most recent news items
                "signal": signal,
                "current_price": current_price,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_corporate_actions(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze corporate actions for trading opportunities.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with corporate action analysis
        """
        try:
            # Get corporate actions data
            corporate_actions = self._get_corporate_actions(symbol, exchange)
            
            if not corporate_actions:
                return {"status": "no_corporate_actions"}
            
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            today = datetime.now()
            
            # Find upcoming dividends
            upcoming_dividends = []
            for action in corporate_actions:
                if action.get("type") == "dividend":
                    ex_date = action.get("ex_date")
                    if ex_date and ex_date >= today:
                        days_to_ex_date = (ex_date - today).days
                        
                        upcoming_dividends.append({
                            "amount": action.get("amount"),
                            "ex_date": ex_date,
                            "days_to_ex_date": days_to_ex_date,
                            "dividend_yield": (action.get("amount", 0) / current_price * 100) if current_price else None
                        })
            
            # Sort upcoming dividends by ex-date
            upcoming_dividends.sort(key=lambda x: x.get("ex_date", today))
            
            # Find upcoming stock splits
            upcoming_splits = []
            for action in corporate_actions:
                if action.get("type") == "split":
                    split_date = action.get("date")
                    if split_date and split_date >= today:
                        days_to_split_date = (split_date - today).days
                        
                        upcoming_splits.append({
                            "ratio": action.get("ratio"),
                            "date": split_date,
                            "days_to_split_date": days_to_split_date
                        })
            
            # Sort upcoming splits by date
            upcoming_splits.sort(key=lambda x: x.get("date", today))
            
            # Find upcoming buybacks
            upcoming_buybacks = []
            for action in corporate_actions:
                if action.get("type") == "buyback":
                    start_date = action.get("start_date")
                    if start_date and start_date >= today:
                        days_to_start_date = (start_date - today).days
                        
                        upcoming_buybacks.append({
                            "amount": action.get("amount"),
                            "price": action.get("price"),
                            "start_date": start_date,
                            "end_date": action.get("end_date"),
                            "days_to_start_date": days_to_start_date
                        })
            
            # Sort upcoming buybacks by start date
            upcoming_buybacks.sort(key=lambda x: x.get("start_date", today))
            
            # Generate trading signals
            signals = []
            
            # Dividend trading signals
            for dividend in upcoming_dividends:
                ex_date = dividend.get("ex_date")
                days_to_ex_date = dividend.get("days_to_ex_date")
                
                if days_to_ex_date <= self.params["dividend_lookback"]:
                    dividend_yield = dividend.get("dividend_yield")
                    
                    if dividend_yield and dividend_yield > 1.0:  # Only consider dividends with yield > 1%
                        confidence = min(dividend_yield / 5, 0.8)  # Scale confidence based on yield
                        
                        signals.append({
                            "type": "dividend",
                            "direction": "bullish",
                            "ex_date": ex_date,
                            "days_to_ex_date": days_to_ex_date,
                            "dividend_amount": dividend.get("amount"),
                            "dividend_yield": dividend_yield,
                            "confidence": confidence
                        })
            
            # Split trading signals
            for split in upcoming_splits:
                split_date = split.get("date")
                days_to_split_date = split.get("days_to_split_date")
                split_ratio = split.get("ratio")
                
                if days_to_split_date <= self.params["split_lookback"]:
                    # Parse split ratio
                    if isinstance(split_ratio, str) and ":" in split_ratio:
                        try:
                            before, after = split_ratio.split(":")
                            before = float(before.strip())
                            after = float(after.strip())
                            
                            if after > before:  # Stock split (e.g., 1:5)
                                signals.append({
                                    "type": "split",
                                    "direction": "bullish",
                                    "split_date": split_date,
                                    "days_to_split_date": days_to_split_date,
                                    "split_ratio": split_ratio,
                                    "confidence": 0.6  # Fixed confidence for splits
                                })
                            elif before > after:  # Reverse split (e.g., 5:1)
                                signals.append({
                                    "type": "reverse_split",
                                    "direction": "bearish",
                                    "split_date": split_date,
                                    "days_to_split_date": days_to_split_date,
                                    "split_ratio": split_ratio,
                                    "confidence": 0.6  # Fixed confidence for splits
                                })
                        except:
                            pass
                    elif isinstance(split_ratio, (int, float)) and split_ratio > 1:
                        signals.append({
                            "type": "split",
                            "direction": "bullish",
                            "split_date": split_date,
                            "days_to_split_date": days_to_split_date,
                            "split_ratio": split_ratio,
                            "confidence": 0.6  # Fixed confidence for splits
                        })
            
            # Buyback trading signals
            for buyback in upcoming_buybacks:
                start_date = buyback.get("start_date")
                days_to_start_date = buyback.get("days_to_start_date")
                buyback_price = buyback.get("price")
                
                if buyback_price and current_price:
                    premium = (buyback_price - current_price) / current_price * 100
                    
                    if premium > 5:  # Only consider buybacks with premium > 5%
                        confidence = min(premium / 20, 0.8)  # Scale confidence based on premium
                        
                        signals.append({
                            "type": "buyback",
                            "direction": "bullish",
                            "start_date": start_date,
                            "days_to_start_date": days_to_start_date,
                            "buyback_price": buyback_price,
                            "premium": premium,
                            "confidence": confidence
                        })
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "current_price": current_price,
                "upcoming_dividends": upcoming_dividends,
                "upcoming_splits": upcoming_splits,
                "upcoming_buybacks": upcoming_buybacks,
                "signals": signals,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing corporate actions for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def analyze_economic_events(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Analyze economic events impact on trading opportunities.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with economic event analysis
        """
        try:
            # Get upcoming economic events
            events = self._get_upcoming_economic_events(days=7)
            
            if not events:
                return {"status": "no_economic_events"}
            
            # Get stock sector
            stock_info = self._get_stock_info(symbol, exchange)
            sector = stock_info.get("sector") if stock_info else None
            
            # Filter events by impact level and relevance to the stock's sector
            relevant_events = []
            
            for event in events:
                impact = event.get("impact", "low")
                event_sectors = event.get("affected_sectors", [])
                
                # Check if impact meets threshold
                impact_meets_threshold = (
                    (impact == "high") or
                    (impact == "medium" and self.params["economic_event_impact_threshold"] in ["medium", "low"]) or
                    (impact == "low" and self.params["economic_event_impact_threshold"] == "low")
                )
                
                # Check sector relevance
                sector_relevant = (
                    not event_sectors or  # If no specific sectors listed, consider relevant
                    not sector or  # If stock sector unknown, include event
                    sector in event_sectors  # Stock sector is in affected sectors
                )
                
                if impact_meets_threshold and sector_relevant:
                    # Calculate days until event
                    event_date = event.get("date")
                    days_to_event = (event_date - datetime.now()).days if event_date else None
                    
                    # Add days to event
                    event_copy = event.copy()
                    event_copy["days_to_event"] = days_to_event
                    
                    relevant_events.append(event_copy)
            
            # Sort events by date
            relevant_events.sort(key=lambda x: x.get("date", datetime.now()))
            
            # Generate trading signals
            signals = []
            
            for event in relevant_events:
                event_name = event.get("name", "Unknown event")
                event_date = event.get("date")
                days_to_event = event.get("days_to_event")
                impact = event.get("impact", "low")
                expected_direction = event.get("expected_direction", "neutral")
                
                # Only generate signals for high-impact events close to release
                if impact == "high" and 0 <= days_to_event <= self.params["pre_event_entry_days"]:
                    # Set confidence based on impact
                    confidence = 0.7 if impact == "high" else 0.5
                    
                    if expected_direction in ["bullish", "bearish"]:
                        signals.append({
                            "type": "economic_event",
                            "direction": expected_direction,
                            "event_name": event_name,
                            "event_date": event_date,
                            "days_to_event": days_to_event,
                            "impact": impact,
                            "confidence": confidence
                        })
            
            return {
                "symbol": symbol,
                "exchange": exchange,
                "sector": sector,
                "relevant_events": relevant_events,
                "signals": signals,
                "analysis_date": datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing economic events for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def generate_event_driven_signals(self, symbol: str, exchange: str = "NSE") -> Dict[str, Any]:
        """
        Generate comprehensive event-driven trading signals.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with trading signals
        """
        try:
            # Get latest price data
            latest_price_data = self._get_latest_price(symbol, exchange)
            current_price = latest_price_data.get("price") if latest_price_data else None
            
            if not current_price:
                return {"status": "error", "message": "Could not determine current price"}
            
            # Collect all event-driven analyses
            earnings_analysis = self.analyze_earnings_events(symbol, exchange)
            news_analysis = self.analyze_news_sentiment(symbol, exchange)
            corporate_analysis = self.analyze_corporate_actions(symbol, exchange)
            economic_analysis = self.analyze_economic_events(symbol, exchange)
            
            # Aggregate signals
            signals = []
            
            # 1. Earnings signals
            if earnings_analysis and "signals" in earnings_analysis:
                for signal in earnings_analysis["signals"]:
                    signal_type = signal.get("type")
                    direction = signal.get("direction")
                    confidence = signal.get("confidence", 0.5)
                    
                    if direction in ["bullish", "bearish"]:
                        signals.append({
                            "category": "earnings",
                            "type": signal_type,
                            "direction": direction,
                            "confidence": confidence,
                            "details": signal
                        })
            
            # 2. News sentiment signals
            if news_analysis and "signal" in news_analysis and news_analysis["signal"]:
                signal = news_analysis["signal"]
                direction = signal.get("direction")
                confidence = signal.get("confidence", 0.5)
                
                if direction in ["bullish", "bearish"]:
                    signals.append({
                        "category": "news_sentiment",
                        "type": "sentiment",
                        "direction": direction,
                        "confidence": confidence,
                        "details": signal
                    })
            
            # 3. Corporate action signals
            if corporate_analysis and "signals" in corporate_analysis:
                for signal in corporate_analysis["signals"]:
                    signal_type = signal.get("type")
                    direction = signal.get("direction")
                    confidence = signal.get("confidence", 0.5)
                    
                    if direction in ["bullish", "bearish"]:
                        signals.append({
                            "category": "corporate_action",
                            "type": signal_type,
                            "direction": direction,
                            "confidence": confidence,
                            "details": signal
                        })
            
            # 4. Economic event signals
            if economic_analysis and "signals" in economic_analysis:
                for signal in economic_analysis["signals"]:
                    event_name = signal.get("event_name", "Unknown event")
                    direction = signal.get("direction")
                    confidence = signal.get("confidence", 0.5)
                    
                    if direction in ["bullish", "bearish"]:
                        signals.append({
                            "category": "economic_event",
                            "type": "event",
                            "event_name": event_name,
                            "direction": direction,
                            "confidence": confidence,
                            "details": signal
                        })
            
            # Determine overall signal
            bullish_signals = [s for s in signals if s["direction"] == "bullish"]
            bearish_signals = [s for s in signals if s["direction"] == "bearish"]
            
            # Weight signals by confidence
            bullish_score = sum(s["confidence"] for s in bullish_signals)
            bearish_score = sum(s["confidence"] for s in bearish_signals)
            
            overall_signal = "neutral"
            if bullish_score > bearish_score + 0.3:  # Require a meaningful difference
                overall_signal = "bullish"
            elif bearish_score > bullish_score + 0.3:
                overall_signal = "bearish"
            
            # Calculate risk management parameters
            stop_loss = None
            target_price = None
            
            if overall_signal == "bullish":
                # Use different stop loss based on the type of signal
                if any(s["category"] == "earnings" for s in signals):
                    # Wider stop for earnings
                    stop_loss = current_price * (1 - self.params["earnings_stop_loss"])
                else:
                    # Tighter stop for other events
                    stop_loss = current_price * (1 - self.params["news_stop_loss"])
                
                # Set target based on expected move
                expected_move = 0.0
                if "expected_move_percent" in earnings_analysis:
                    expected_move = earnings_analysis["expected_move_percent"]
                
                if expected_move:
                    target_price = current_price * (1 + expected_move/100)
                else:
                    # Default target - 2:1 risk-reward
                    risk = current_price - stop_loss
                    target_price = current_price + 2 * risk
            
            elif overall_signal == "bearish":
                # Use different stop loss based on the type of signal
                if any(s["category"] == "earnings" for s in signals):
                    # Wider stop for earnings
                    stop_loss = current_price * (1 + self.params["earnings_stop_loss"])
                else:
                    # Tighter stop for other events
                    stop_loss = current_price * (1 + self.params["news_stop_loss"])
                
                # Set target based on expected move
                expected_move = 0.0
                if "expected_move_percent" in earnings_analysis:
                    expected_move = earnings_analysis["expected_move_percent"]
                
                if expected_move:
                    target_price = current_price * (1 - expected_move/100)
                else:
                    # Default target - 2:1 risk-reward
                    risk = stop_loss - current_price
                    target_price = current_price - 2 * risk
            
            # Calculate risk-reward ratio
            risk_reward_ratio = None
            if stop_loss and target_price:
                risk = abs(current_price - stop_loss)
                reward = abs(target_price - current_price)
                
                if risk > 0:
                    risk_reward_ratio = reward / risk
            
            # Calculate overall confidence
            overall_confidence = 0.0
            if overall_signal == "bullish":
                overall_confidence = bullish_score / len(bullish_signals) if bullish_signals else 0.0
            elif overall_signal == "bearish":
                overall_confidence = bearish_score / len(bearish_signals) if bearish_signals else 0.0
            
            # Generate final signal
            trading_signal = {
                "strategy": "event_driven",
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": datetime.now(),
                "current_price": current_price,
                "signal": overall_signal,
                "stop_loss": stop_loss,
                "target_price": target_price,
                "risk_reward_ratio": risk_reward_ratio,
                "signals": signals,
                "bullish_score": bullish_score,
                "bearish_score": bearish_score,
                "overall_confidence": overall_confidence
            }
            
            return trading_signal
            
        except Exception as e:
            self.logger.error(f"Error generating event-driven signals for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def scan_for_event_opportunities(self, symbols: List[str], exchange: str = "NSE") -> Dict[str, List[Dict[str, Any]]]:
        """
        Scan for event-driven trading opportunities across multiple symbols.
        
        Args:
            symbols: List of stock symbols to analyze
            exchange: Stock exchange
            
        Returns:
            Dictionary with opportunities for each event type
        """
        results = {
            "earnings": [],
            "news_sentiment": [],
            "corporate_actions": [],
            "economic_events": []
        }
        
        for symbol in symbols:
            try:
                # Get comprehensive event-driven signals
                signals = self.generate_event_driven_signals(symbol, exchange)
                
                if not signals or "status" in signals and signals["status"] == "error":
                    continue
                
                # Check if there's a valid trading signal
                if signals.get("signal") == "neutral" or not signals.get("risk_reward_ratio", 0) >= 1.5:
                    continue
                
                # Sort into appropriate categories based on signal type
                detailed_signals = signals.get("signals", [])
                
                for signal in detailed_signals:
                    category = signal.get("category")
                    
                    if category == "earnings":
                        if symbol not in [s.get("symbol") for s in results["earnings"]]:
                            results["earnings"].append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "signal": signals.get("signal"),
                                "current_price": signals.get("current_price"),
                                "stop_loss": signals.get("stop_loss"),
                                "target_price": signals.get("target_price"),
                                "risk_reward_ratio": signals.get("risk_reward_ratio"),
                                "confidence": signal.get("confidence"),
                                "details": signal.get("details")
                            })
                    
                    elif category == "news_sentiment":
                        if symbol not in [s.get("symbol") for s in results["news_sentiment"]]:
                            results["news_sentiment"].append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "signal": signals.get("signal"),
                                "current_price": signals.get("current_price"),
                                "stop_loss": signals.get("stop_loss"),
                                "target_price": signals.get("target_price"),
                                "risk_reward_ratio": signals.get("risk_reward_ratio"),
                                "confidence": signal.get("confidence"),
                                "details": signal.get("details")
                            })
                    
                    elif category == "corporate_action":
                        if symbol not in [s.get("symbol") for s in results["corporate_actions"]]:
                            results["corporate_actions"].append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "signal": signals.get("signal"),
                                "current_price": signals.get("current_price"),
                                "stop_loss": signals.get("stop_loss"),
                                "target_price": signals.get("target_price"),
                                "risk_reward_ratio": signals.get("risk_reward_ratio"),
                                "confidence": signal.get("confidence"),
                                "details": signal.get("details")
                            })
                    
                    elif category == "economic_event":
                        if symbol not in [s.get("symbol") for s in results["economic_events"]]:
                            results["economic_events"].append({
                                "symbol": symbol,
                                "exchange": exchange,
                                "signal": signals.get("signal"),
                                "current_price": signals.get("current_price"),
                                "stop_loss": signals.get("stop_loss"),
                                "target_price": signals.get("target_price"),
                                "risk_reward_ratio": signals.get("risk_reward_ratio"),
                                "confidence": signal.get("confidence"),
                                "details": signal.get("details")
                            })
            
            except Exception as e:
                self.logger.error(f"Error scanning event opportunities for {symbol}: {e}")
        
        return results

    def _get_earnings_data(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Get earnings data from database.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            List of earnings events
        """
        try:
            # Query the earnings collection
            cursor = self.db.earnings_collection.find({
                "symbol": symbol,
                "exchange": exchange
            }).sort("date", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting earnings data for {symbol}: {e}")
            return []
    
    def _analyze_historical_earnings_reactions(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Analyze historical price reactions to earnings.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with earnings reaction statistics
        """
        try:
            # Get earnings data
            earnings_data = self._get_earnings_data(symbol, exchange)
            
            if not earnings_data:
                return {}
            
            # Track price moves after earnings
            price_moves = []
            eps_surprises = []
            
            for earnings in earnings_data:
                earnings_date = earnings.get("date")
                
                if not earnings_date:
                    continue
                
                # Get price action around earnings
                pre_earnings_price = self._get_price_at_date(symbol, exchange, earnings_date - timedelta(days=1))
                post_earnings_price = self._get_price_at_date(symbol, exchange, earnings_date + timedelta(days=1))
                
                if pre_earnings_price and post_earnings_price:
                    # Calculate percentage move
                    price_move = (post_earnings_price - pre_earnings_price) / pre_earnings_price * 100
                    price_moves.append(price_move)
                
                # Track earnings surprises if available
                actual_eps = earnings.get("actual_eps")
                expected_eps = earnings.get("expected_eps")
                
                if actual_eps is not None and expected_eps is not None and expected_eps != 0:
                    surprise_percent = (actual_eps - expected_eps) / abs(expected_eps) * 100
                    eps_surprises.append({
                        "surprise_percent": surprise_percent,
                        "price_move": price_move if pre_earnings_price and post_earnings_price else None
                    })
            
            # Calculate statistics
            if price_moves:
                avg_move = sum(price_moves) / len(price_moves)
                avg_abs_move = sum(abs(move) for move in price_moves) / len(price_moves)
                max_up_move = max(price_moves) if price_moves else 0
                max_down_move = min(price_moves) if price_moves else 0
                
                # Determine direction bias
                positive_moves = sum(1 for move in price_moves if move > 0)
                negative_moves = sum(1 for move in price_moves if move < 0)
                
                direction_bias = "neutral"
                if positive_moves > negative_moves * 1.5:
                    direction_bias = "bullish"
                elif negative_moves > positive_moves * 1.5:
                    direction_bias = "bearish"
                
                return {
                    "avg_move_percent": avg_move,
                    "avg_abs_move_percent": avg_abs_move,
                    "max_up_move_percent": max_up_move,
                    "max_down_move_percent": max_down_move,
                    "positive_reactions": positive_moves,
                    "negative_reactions": negative_moves,
                    "direction_bias": direction_bias,
                    "eps_surprises": eps_surprises
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error analyzing historical earnings reactions for {symbol}: {e}")
            return {}
    
    def _get_post_earnings_reaction(self, symbol: str, exchange: str, earnings_date: datetime) -> Optional[float]:
        """
        Get the price reaction after earnings.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            earnings_date: Earnings announcement date
            
        Returns:
            Percentage price change after earnings
        """
        try:
            # Get price before earnings
            pre_earnings_price = self._get_price_at_date(symbol, exchange, earnings_date - timedelta(days=1))
            
            # Get price after earnings
            post_earnings_price = self._get_price_at_date(symbol, exchange, earnings_date + timedelta(days=1))
            
            if pre_earnings_price and post_earnings_price:
                # Calculate percentage move
                price_move = (post_earnings_price - pre_earnings_price) / pre_earnings_price * 100
                return price_move
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting post-earnings reaction for {symbol}: {e}")
            return None
    
    def _get_recent_news(self, symbol: str, exchange: str, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get recent news for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            days: Number of days to look back
            
        Returns:
            List of news items
        """
        try:
            # Calculate the start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Query the news collection
            cursor = self.db.news_collection.find({
                "$or": [
                    {"entities": symbol},  # Symbol mentioned in entities
                    {"related_symbols": symbol},  # Symbol in related symbols
                    {"title": {"$regex": symbol, "$options": "i"}}  # Symbol in title
                ]
            }).sort("published_date", -1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting recent news for {symbol}: {e}")
            return []
    
    def _calculate_sentiment(self, text: str) -> float:
        """
        Calculate sentiment score for text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Sentiment score (-1 to 1, negative to positive)
        """
        try:
            if self.sia:
                # Use NLTK's SentimentIntensityAnalyzer if available
                sentiment = self.sia.polarity_scores(text)
                return sentiment["compound"]
            else:
                # Simple keyword-based approach as fallback
                positive_words = [
                    "up", "higher", "rise", "gain", "profit", "grow", "positive", "beat", 
                    "exceed", "increase", "strong", "good", "boost", "jump", "rally", 
                    "improve", "advantage", "opportunity", "bullish", "outperform"
                ]
                negative_words = [
                    "down", "lower", "fall", "drop", "loss", "decline", "negative", "miss", 
                    "below", "decrease", "weak", "bad", "risk", "concern", "worry", 
                    "poor", "trouble", "problem", "bearish", "underperform"
                ]
                
                # Convert to lowercase and split into words
                words = text.lower().split()
                
                # Count positive and negative words
                positive_count = sum(1 for word in words if word in positive_words)
                negative_count = sum(1 for word in words if word in negative_words)
                
                # Calculate sentiment score
                total_count = positive_count + negative_count
                if total_count > 0:
                    return (positive_count - negative_count) / total_count
                
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {e}")
            return 0.0
    
    def _get_corporate_actions(self, symbol: str, exchange: str) -> List[Dict[str, Any]]:
        """
        Get corporate actions for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            List of corporate actions
        """
        try:
            # Query the corporate actions collection
            cursor = self.db.corporate_actions_collection.find({
                "symbol": symbol,
                "exchange": exchange
            }).sort("date", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting corporate actions for {symbol}: {e}")
            return []
    
    def _get_upcoming_economic_events(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming economic events.
        
        Args:
            days: Number of days to look ahead
            
        Returns:
            List of economic events
        """
        try:
            # Calculate the end date
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days)
            
            # Query the economic events collection
            cursor = self.db.economic_events_collection.find({
                "date": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }).sort("date", 1)
            
            return list(cursor)
            
        except Exception as e:
            self.logger.error(f"Error getting upcoming economic events: {e}")
            return []
    
    def _get_stock_info(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get basic stock information.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with stock information
        """
        try:
            # Query the portfolio collection for stock info
            stock_info = self.db.portfolio_collection.find_one({
                "symbol": symbol,
                "exchange": exchange
            })
            
            return stock_info or {}
            
        except Exception as e:
            self.logger.error(f"Error getting stock info for {symbol}: {e}")
            return {}
    
    def _get_latest_price(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get latest price data for a symbol.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            
        Returns:
            Dictionary with latest price data
        """
        try:
            # Query the market data collection
            latest_data = self.db.market_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day"
            }, sort=[("timestamp", -1)])
            
            if latest_data:
                return {
                    "price": latest_data.get("close"),
                    "timestamp": latest_data.get("timestamp"),
                    "open": latest_data.get("open"),
                    "high": latest_data.get("high"),
                    "low": latest_data.get("low"),
                    "volume": latest_data.get("volume")
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error getting latest price for {symbol}: {e}")
            return {}
    
    def _get_price_at_date(self, symbol: str, exchange: str, date: datetime) -> Optional[float]:
        """
        Get closing price at a specific date.
        
        Args:
            symbol: Stock symbol
            exchange: Stock exchange
            date: Date to get price for
            
        Returns:
            Closing price or None if not available
        """
        try:
            # Set date range (+/- 1 day to handle weekends and holidays)
            start_date = date - timedelta(days=1)
            end_date = date + timedelta(days=1)
            
            # Query the market data collection
            data = self.db.market_data_collection.find_one({
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": "day",
                "timestamp": {
                    "$gte": start_date,
                    "$lte": end_date
                }
            }, sort=[("timestamp", -1)])
            
            if data and "close" in data:
                return data["close"]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting price at date for {symbol}: {e}")
            return None