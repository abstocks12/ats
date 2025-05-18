"""
Social Media Sentiment Analyzer

This module collects and analyzes sentiment from social media platforms about stocks and markets.
It provides sentiment scores and trend analysis for trading decisions.
"""

import requests
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any, Optional, Tuple
import json
from textblob import TextBlob
import tweepy
from pymongo import UpdateOne

class SocialSentiment:
    """
    Collects and analyzes sentiment from social media platforms about stocks and markets.
    """
    
    def __init__(self, db_connector):
        """
        Initialize the social sentiment analyzer with database connection.
        
        Args:
            db_connector: MongoDB database connector
        """
        self.db = db_connector
        self.logger = self._setup_logger()
        
        # Load API keys
        self.api_keys = self._load_api_keys()
        
        # Initialize API clients if keys are available
        self.twitter_api = self._init_twitter_api() if "twitter" in self.api_keys else None
        self.reddit_api_headers = self._get_reddit_headers() if "reddit" in self.api_keys else None
        self.stocktwits_url = "https://api.stocktwits.com/api/2"
        
        # Platform weights for combined sentiment (adjustable based on reliability)
        self.platform_weights = {
            "twitter": 0.4,
            "reddit": 0.3,
            "stocktwits": 0.3
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            "very_negative": -0.6,
            "negative": -0.2,
            "neutral": 0.2,
            "positive": 0.6,
            "very_positive": 1.0
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
    
    def _load_api_keys(self) -> Dict[str, Dict[str, str]]:
        """Load API keys from database or environment."""
        try:
            # Try to load from database
            api_config = self.db.system_config_collection.find_one({"config_type": "api_keys"})
            
            if api_config and "social_media" in api_config:
                return api_config["social_media"]
            
            # Fallback to default keys (should be replaced in production)
            self.logger.warning("Using default API keys - replace in production")
            return {
                "twitter": {
                    "consumer_key": "your_consumer_key",
                    "consumer_secret": "your_consumer_secret",
                    "access_token": "your_access_token",
                    "access_token_secret": "your_access_token_secret"
                },
                "reddit": {
                    "client_id": "your_client_id",
                    "client_secret": "your_client_secret",
                    "user_agent": "your_user_agent"
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error loading API keys: {e}")
            return {}
    
    def _init_twitter_api(self) -> Optional[tweepy.API]:
        """Initialize Twitter API client."""
        try:
            if "twitter" in self.api_keys:
                auth = tweepy.OAuthHandler(
                    self.api_keys["twitter"]["consumer_key"],
                    self.api_keys["twitter"]["consumer_secret"]
                )
                auth.set_access_token(
                    self.api_keys["twitter"]["access_token"],
                    self.api_keys["twitter"]["access_token_secret"]
                )
                api = tweepy.API(auth, wait_on_rate_limit=True)
                
                # Test the connection
                api.verify_credentials()
                self.logger.info("Twitter API initialized successfully")
                return api
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error initializing Twitter API: {e}")
            return None
    
    def _get_reddit_headers(self) -> Optional[Dict[str, str]]:
        """Get headers for Reddit API requests."""
        try:
            if "reddit" in self.api_keys:
                # Get OAuth token
                auth = requests.auth.HTTPBasicAuth(
                    self.api_keys["reddit"]["client_id"],
                    self.api_keys["reddit"]["client_secret"]
                )
                
                data = {
                    'grant_type': 'password',
                    'username': self.api_keys["reddit"].get("username", ""),
                    'password': self.api_keys["reddit"].get("password", "")
                }
                
                headers = {'User-Agent': self.api_keys["reddit"]["user_agent"]}
                
                response = requests.post(
                    'https://www.reddit.com/api/v1/access_token',
                    auth=auth,
                    data=data,
                    headers=headers
                )
                
                if response.status_code == 200:
                    token = response.json().get('access_token')
                    
                    if token:
                        headers = {
                            'Authorization': f'bearer {token}',
                            'User-Agent': self.api_keys["reddit"]["user_agent"]
                        }
                        self.logger.info("Reddit API headers initialized successfully")
                        return headers
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting Reddit headers: {e}")
            return None
    
    def analyze_symbol_sentiment(self, symbol: str, full_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze social media sentiment for a specific stock symbol.
        
        Args:
            symbol: Stock symbol
            full_name: Full company name (optional)
            
        Returns:
            Dictionary containing sentiment analysis results
        """
        self.logger.info(f"Analyzing sentiment for {symbol}")
        
        # If full name not provided, try to get it from database
        if not full_name:
            stock_info = self.db.portfolio_collection.find_one({"symbol": symbol})
            if stock_info and "company_name" in stock_info:
                full_name = stock_info["company_name"]
        
        # Clean symbol (remove exchange suffixes like .NS)
        clean_symbol = symbol.split('.')[0]
        
        # Search terms
        search_terms = [f"${clean_symbol}", clean_symbol]
        if full_name:
            search_terms.append(full_name)
        
        # Collect sentiment from different platforms
        twitter_sentiment = self._analyze_twitter_sentiment(search_terms) if self.twitter_api else {}
        reddit_sentiment = self._analyze_reddit_sentiment(search_terms) if self.reddit_api_headers else {}
        stocktwits_sentiment = self._analyze_stocktwits_sentiment(clean_symbol)
        
        # Combine sentiment
        combined_sentiment = self._combine_sentiment(twitter_sentiment, reddit_sentiment, stocktwits_sentiment)
        
        # Create result object
        result = {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "search_terms": search_terms,
            "sentiment": {
                "combined": combined_sentiment,
                "twitter": twitter_sentiment,
                "reddit": reddit_sentiment,
                "stocktwits": stocktwits_sentiment
            }
        }
        
        # Save to database
        self._save_sentiment(result)
        
        return result
    
    def _analyze_twitter_sentiment(self, search_terms: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment from Twitter posts.
        
        Args:
            search_terms: List of search terms
            
        Returns:
            Dictionary containing Twitter sentiment analysis
        """
        try:
            if not self.twitter_api:
                return {"error": "Twitter API not initialized"}
            
            all_tweets = []
            sentiment_scores = []
            
            # Search tweets for each term
            for term in search_terms:
                try:
                    # Get tweets (adjust count as needed)
                    tweets = self.twitter_api.search_tweets(
                        q=term,
                        lang="en",
                        count=100,
                        result_type="recent",
                        tweet_mode="extended"
                    )
                    
                    for tweet in tweets:
                        # Skip retweets to avoid duplicates
                        if hasattr(tweet, "retweeted_status"):
                            continue
                        
                        # Get full text
                        if hasattr(tweet, "full_text"):
                            text = tweet.full_text
                        else:
                            text = tweet.text
                        
                        # Clean text
                        clean_text = self._clean_text(text)
                        
                        # Analyze sentiment
                        blob = TextBlob(clean_text)
                        sentiment_score = blob.sentiment.polarity
                        
                        # Add to results
                        all_tweets.append({
                            "id": tweet.id_str,
                            "text": clean_text,
                            "created_at": tweet.created_at,
                            "likes": tweet.favorite_count,
                            "retweets": tweet.retweet_count,
                            "sentiment_score": sentiment_score
                        })
                        
                        sentiment_scores.append(sentiment_score)
                    
                    # Respect rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error searching Twitter for term '{term}': {e}")
            
            # Calculate sentiment metrics
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                weighted_sentiment = sum(tweet["sentiment_score"] * (tweet["likes"] + tweet["retweets"] + 1) 
                                       for tweet in all_tweets) / sum(tweet["likes"] + tweet["retweets"] + 1 
                                                                    for tweet in all_tweets)
                
                sentiment_category = self._get_sentiment_category(avg_sentiment)
                weighted_sentiment_category = self._get_sentiment_category(weighted_sentiment)
                
                return {
                    "posts_analyzed": len(all_tweets),
                    "avg_sentiment_score": avg_sentiment,
                    "weighted_sentiment_score": weighted_sentiment,
                    "sentiment_category": sentiment_category,
                    "weighted_sentiment_category": weighted_sentiment_category,
                    "sentiment_distribution": self._get_sentiment_distribution(sentiment_scores),
                    "sample_posts": all_tweets[:5]  # Include sample posts for reference
                }
            
            return {"posts_analyzed": 0}
            
        except Exception as e:
            self.logger.error(f"Error analyzing Twitter sentiment: {e}")
            return {"error": str(e)}
    
    def _analyze_reddit_sentiment(self, search_terms: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment from Reddit posts.
        
        Args:
            search_terms: List of search terms
            
        Returns:
            Dictionary containing Reddit sentiment analysis
        """
        try:
            if not self.reddit_api_headers:
                return {"error": "Reddit API not initialized"}
            
            all_posts = []
            sentiment_scores = []
            
            # List of finance-related subreddits
            subreddits = ["investing", "stocks", "wallstreetbets", "StockMarket", "options", "IndianStreetBets"]
            
            for subreddit in subreddits:
                for term in search_terms:
                    try:
                        # Search posts
                        response = requests.get(
                            f"https://oauth.reddit.com/r/{subreddit}/search",
                            headers=self.reddit_api_headers,
                            params={
                                "q": term,
                                "sort": "new",
                                "t": "week",
                                "limit": 25
                            }
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            
                            for post in data.get("data", {}).get("children", []):
                                post_data = post.get("data", {})
                                
                                # Get text content
                                title = post_data.get("title", "")
                                selftext = post_data.get("selftext", "")
                                text = f"{title} {selftext}".strip()
                                
                                # Skip if no text
                                if not text:
                                    continue
                                
                                # Clean text
                                clean_text = self._clean_text(text)
                                
                                # Analyze sentiment
                                blob = TextBlob(clean_text)
                                sentiment_score = blob.sentiment.polarity
                                
                                # Add to results
                                all_posts.append({
                                    "id": post_data.get("id", ""),
                                    "title": title,
                                    "subreddit": subreddit,
                                    "created_at": datetime.fromtimestamp(post_data.get("created_utc", 0)),
                                    "score": post_data.get("score", 0),
                                    "num_comments": post_data.get("num_comments", 0),
                                    "sentiment_score": sentiment_score
                                })
                                
                                sentiment_scores.append(sentiment_score)
                                
                                # Get comments
                                post_id = post_data.get("id", "")
                                if post_id:
                                    comment_response = requests.get(
                                        f"https://oauth.reddit.com/r/{subreddit}/comments/{post_id}",
                                        headers=self.reddit_api_headers,
                                        params={"limit": 10, "sort": "top"}
                                    )
                                    
                                    if comment_response.status_code == 200:
                                        comment_data = comment_response.json()
                                        
                                        # Process comments
                                        if len(comment_data) > 1:  # First element is the post, second contains comments
                                            for comment in comment_data[1].get("data", {}).get("children", []):
                                                comment_body = comment.get("data", {}).get("body", "")
                                                
                                                if comment_body:
                                                    # Clean text
                                                    clean_comment = self._clean_text(comment_body)
                                                    
                                                    # Analyze sentiment
                                                    comment_blob = TextBlob(clean_comment)
                                                    comment_sentiment = comment_blob.sentiment.polarity
                                                    
                                                    sentiment_scores.append(comment_sentiment)
                        
                        # Respect rate limits
                        time.sleep(2)
                        
                    except Exception as e:
                        self.logger.error(f"Error searching Reddit for term '{term}' in r/{subreddit}: {e}")
            
            # Calculate sentiment metrics
            if sentiment_scores:
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                weighted_sentiment = sum(post["sentiment_score"] * (post["score"] + post["num_comments"] + 1) 
                                      for post in all_posts) / sum(post["score"] + post["num_comments"] + 1 
                                                                 for post in all_posts)
                
                sentiment_category = self._get_sentiment_category(avg_sentiment)
                weighted_sentiment_category = self._get_sentiment_category(weighted_sentiment)
                
                return {
                    "posts_analyzed": len(all_posts),
                    "comments_analyzed": len(sentiment_scores) - len(all_posts),
                    "avg_sentiment_score": avg_sentiment,
                    "weighted_sentiment_score": weighted_sentiment,
                    "sentiment_category": sentiment_category,
                    "weighted_sentiment_category": weighted_sentiment_category,
                    "sentiment_distribution": self._get_sentiment_distribution(sentiment_scores),
                    "sample_posts": all_posts[:5]  # Include sample posts for reference
                }
            
            return {"posts_analyzed": 0, "comments_analyzed": 0}
            
        except Exception as e:
            self.logger.error(f"Error analyzing Reddit sentiment: {e}")
            return {"error": str(e)}
    
    def _analyze_stocktwits_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze sentiment from StockTwits posts.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary containing StockTwits sentiment analysis
        """
        try:
            # StockTwits API doesn't require authentication for symbol streams
            url = f"{self.stocktwits_url}/streams/symbol/{symbol}.json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                if "messages" in data:
                    messages = data["messages"]
                    
                    # Extract sentiment data
                    sentiment_scores = []
                    bullish_count = 0
                    bearish_count = 0
                    
                    for message in messages:
                        # Get message text
                        text = message.get("body", "")
                        clean_text = self._clean_text(text)
                        
                        # Get StockTwits sentiment if available
                        entities = message.get("entities", {})
                        sentiment = entities.get("sentiment", {})
                        
                        if sentiment and "basic" in sentiment:
                            if sentiment["basic"] == "Bullish":
                                sentiment_score = 0.5  # Assign positive sentiment score
                                bullish_count += 1
                            elif sentiment["basic"] == "Bearish":
                                sentiment_score = -0.5  # Assign negative sentiment score
                                bearish_count += 1
                            else:
                                # Analyze text if no explicit sentiment
                                blob = TextBlob(clean_text)
                                sentiment_score = blob.sentiment.polarity
                        else:
                            # Analyze text if no explicit sentiment
                            blob = TextBlob(clean_text)
                            sentiment_score = blob.sentiment.polarity
                        
                        sentiment_scores.append(sentiment_score)
                    
                    # Calculate sentiment metrics
                    if sentiment_scores:
                        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                        sentiment_category = self._get_sentiment_category(avg_sentiment)
                        
                        # Calculate bull/bear ratio
                        bull_bear_ratio = bullish_count / bearish_count if bearish_count > 0 else float('inf')
                        
                        return {
                            "messages_analyzed": len(messages),
                            "avg_sentiment_score": avg_sentiment,
                            "sentiment_category": sentiment_category,
                            "bullish_count": bullish_count,
                            "bearish_count": bearish_count,
                            "bull_bear_ratio": bull_bear_ratio,
                            "sentiment_distribution": self._get_sentiment_distribution(sentiment_scores)
                        }
            
            return {"messages_analyzed": 0}
            
        except Exception as e:
            self.logger.error(f"Error analyzing StockTwits sentiment for {symbol}: {e}")
            return {"error": str(e)}
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text for sentiment analysis.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove mentions (@user)
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove cashtags ($SYMBOL)
        text = re.sub(r'\$\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _get_sentiment_category(self, sentiment_score: float) -> str:
        """
        Get sentiment category based on score.
        
        Args:
            sentiment_score: Sentiment score (-1 to 1)
            
        Returns:
            Sentiment category
        """
        if sentiment_score <= self.sentiment_thresholds["very_negative"]:
            return "very_negative"
        elif sentiment_score <= self.sentiment_thresholds["negative"]:
            return "negative"
        elif sentiment_score <= self.sentiment_thresholds["neutral"]:
            return "neutral"
        elif sentiment_score <= self.sentiment_thresholds["positive"]:
            return "positive"
        else:
            return "very_positive"
    
    def _get_sentiment_distribution(self, sentiment_scores: List[float]) -> Dict[str, int]:
        """
        Get distribution of sentiment categories.
        
        Args:
            sentiment_scores: List of sentiment scores
            
        Returns:
            Dictionary with count of posts in each category
        """
        distribution = {
            "very_negative": 0,
            "negative": 0,
            "neutral": 0,
            "positive": 0,
            "very_positive": 0
        }
        
        for score in sentiment_scores:
            category = self._get_sentiment_category(score)
            distribution[category] += 1
        
        return distribution
    
    def _combine_sentiment(self, twitter_sentiment: Dict[str, Any], 
                          reddit_sentiment: Dict[str, Any],
                          stocktwits_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Combine sentiment from different platforms.
        
        Args:
            twitter_sentiment: Twitter sentiment data
            reddit_sentiment: Reddit sentiment data
            stocktwits_sentiment: StockTwits sentiment data
            
        Returns:
            Combined sentiment data
        """
        platforms = []
        weighted_scores = []
        total_weight = 0
        
        # Twitter
        if twitter_sentiment and "avg_sentiment_score" in twitter_sentiment:
            platforms.append("twitter")
            score = twitter_sentiment["avg_sentiment_score"]
            weight = self.platform_weights["twitter"]
            weighted_scores.append(score * weight)
            total_weight += weight
        
        # Reddit
        if reddit_sentiment and "avg_sentiment_score" in reddit_sentiment:
            platforms.append("reddit")
            score = reddit_sentiment["avg_sentiment_score"]
            weight = self.platform_weights["reddit"]
            weighted_scores.append(score * weight)
            total_weight += weight
        
        # StockTwits
        if stocktwits_sentiment and "avg_sentiment_score" in stocktwits_sentiment:
            platforms.append("stocktwits")
            score = stocktwits_sentiment["avg_sentiment_score"]
            weight = self.platform_weights["stocktwits"]
            weighted_scores.append(score * weight)
            total_weight += weight
        
        # Calculate combined score
        if weighted_scores and total_weight > 0:
            combined_score = sum(weighted_scores) / total_weight
            combined_category = self._get_sentiment_category(combined_score)
            
            return {
                "platforms_included": platforms,
                "score": combined_score,
                "category": combined_category
            }
        
        return {"platforms_included": []}
    
    def _save_sentiment(self, sentiment_data: Dict[str, Any]) -> bool:
        """
        Save sentiment data to database.
        
        Args:
            sentiment_data: Sentiment data to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.db.social_sentiment_collection.insert_one(sentiment_data)
            self.logger.info(f"Saved sentiment data for {sentiment_data['symbol']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving sentiment data: {e}")
            return False
    
    def get_sentiment_history(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Get historical sentiment data for a symbol.
        
        Args:
            symbol: Stock symbol
            days: Number of days of history
            
        Returns:
            Dictionary containing historical sentiment data
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Query database for historical sentiment
            historical_data = list(self.db.social_sentiment_collection.find(
                {
                    "symbol": symbol,
                    "timestamp": {"$gte": start_date}
                },
                {"_id": 0}
            ).sort("timestamp", 1))
            
            # Process data
            if historical_data:
                # Extract dates and scores
                dates = [d["timestamp"].strftime("%Y-%m-%d") for d in historical_data]
                scores = [d["sentiment"]["combined"].get("score", 0) if "combined" in d["sentiment"] else 0 
                         for d in historical_data]
                
                # Calculate trend
                if len(scores) >= 2:
                    slope = (scores[-1] - scores[0]) / len(scores)
                    trend = "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable"
                else:
                    trend = "neutral"
                
                return {
                    "symbol": symbol,
                    "days": days,
                    "data_points": len(historical_data),
                    "sentiment_trend": trend,
                    "latest_sentiment": historical_data[-1]["sentiment"]["combined"] if historical_data else {},
                    "historical_data": {
                        "dates": dates,
                        "scores": scores
                    }
                }
            
            return {"symbol": symbol, "days": days, "data_points": 0}
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment history for {symbol}: {e}")
            return {"error": str(e)}
    
    def analyze_portfolio_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Analyze social sentiment for a portfolio of symbols.
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary containing portfolio sentiment analysis
        """
        try:
            results = {
                "timestamp": datetime.now(),
                "total_symbols": len(symbols),
                "symbols_analyzed": 0,
                "portfolio_sentiment": 0,
                "sentiment_by_symbol": {}
            }
            
            successful_symbols = 0
            total_sentiment = 0
            
            for symbol in symbols:
                try:
                    # Check if recent sentiment exists in database
                    recent_sentiment = self.db.social_sentiment_collection.find_one(
                        {
                            "symbol": symbol,
                            "timestamp": {"$gte": datetime.now() - timedelta(hours=24)}
                        },
                        sort=[("timestamp", -1)]
                    )
                    
                    if recent_sentiment:
                        sentiment_data = recent_sentiment["sentiment"]["combined"]
                    else:
                        # Analyze sentiment
                        sentiment_result = self.analyze_symbol_sentiment(symbol)
                        sentiment_data = sentiment_result["sentiment"]["combined"]
                    
                    # Add to results
                    if "score" in sentiment_data:
                        results["sentiment_by_symbol"][symbol] = {
                            "score": sentiment_data["score"],
                            "category": sentiment_data["category"]
                        }
                        
                        total_sentiment += sentiment_data["score"]
                        successful_symbols += 1
                    
                    # Respect rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
                    results["sentiment_by_symbol"][symbol] = {"error": str(e)}
            
            # Calculate portfolio sentiment
            if successful_symbols > 0:
                results["symbols_analyzed"] = successful_symbols
                results["portfolio_sentiment"] = total_sentiment / successful_symbols
                
                # Sort symbols by sentiment (most positive first)
                results["sentiment_by_symbol"] = {
                    k: v for k, v in sorted(
                        results["sentiment_by_symbol"].items(),
                        key=lambda item: item[1].get("score", -999),
                        reverse=True
                    )
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing portfolio sentiment: {e}")
            return {"error": str(e)}
    
    def run_daily_collection(self) -> Dict[str, Any]:
        """
        Run daily collection of social sentiment data for all portfolio symbols.
        
        Returns:
            Dictionary containing collection results
        """
        self.logger.info("Running daily collection of social sentiment data")
        
        results = {
            "timestamp": datetime.now(),
            "symbols_analyzed": 0,
            "successful": 0,
            "failed": 0,
            "details": {}
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
                        # Analyze sentiment
                        sentiment_result = self.analyze_symbol_sentiment(symbol, company_name)
                        
                        if "error" not in sentiment_result:
                            results["successful"] += 1
                            results["details"][symbol] = "success"
                        else:
                            results["failed"] += 1
                            results["details"][symbol] = sentiment_result["error"]
                        
                        # Respect rate limits
                        time.sleep(2)
                        
                    except Exception as e:
                        results["failed"] += 1
                        results["details"][symbol] = str(e)
                        self.logger.error(f"Error analyzing sentiment for {symbol}: {e}")
            
            self.logger.info(f"Daily sentiment collection completed: {results['successful']} successful, {results['failed']} failed")
            
        except Exception as e:
            self.logger.error(f"Error in daily sentiment collection: {e}")
            results["error"] = str(e)
        
        return results
    
    def generate_sentiment_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive sentiment report for all portfolio symbols.
        
        Returns:
            Dictionary containing sentiment report
        """
        try:
            # Get all active symbols from portfolio
            portfolio_symbols = [doc["symbol"] for doc in self.db.portfolio_collection.find(
                {"status": "active"},
                {"symbol": 1}
            )]
            
            # Analyze portfolio sentiment
            portfolio_sentiment = self.analyze_portfolio_sentiment(portfolio_symbols)
            
            # Get sentiment history for top and bottom symbols
            sentiment_by_symbol = portfolio_sentiment.get("sentiment_by_symbol", {})
            
            # Get top 5 and bottom 5 symbols by sentiment
            # Get top 5 and bottom 5 symbols by sentiment
            sorted_symbols = [(k, v.get("score", 0)) for k, v in sentiment_by_symbol.items() 
                             if "score" in v and not isinstance(v, str)]
            sorted_symbols.sort(key=lambda x: x[1], reverse=True)
            
            top_symbols = [s[0] for s in sorted_symbols[:5]]
            bottom_symbols = [s[0] for s in sorted_symbols[-5:]]
            
            # Get historical sentiment for these symbols
            historical_data = {}
            
            for symbol in top_symbols + bottom_symbols:
                history = self.get_sentiment_history(symbol, days=14)
                if "error" not in history:
                    historical_data[symbol] = history
            
            # Create report
            report = {
                "timestamp": datetime.now(),
                "title": "Social Media Sentiment Report",
                "portfolio_sentiment": {
                    "score": portfolio_sentiment.get("portfolio_sentiment", 0),
                    "symbols_analyzed": portfolio_sentiment.get("symbols_analyzed", 0)
                },
                "most_positive": [
                    {
                        "symbol": symbol,
                        "score": sentiment_by_symbol[symbol].get("score", 0),
                        "category": sentiment_by_symbol[symbol].get("category", "")
                    }
                    for symbol in top_symbols if symbol in sentiment_by_symbol
                ],
                "most_negative": [
                    {
                        "symbol": symbol,
                        "score": sentiment_by_symbol[symbol].get("score", 0),
                        "category": sentiment_by_symbol[symbol].get("category", "")
                    }
                    for symbol in bottom_symbols if symbol in sentiment_by_symbol
                ],
                "historical_trends": historical_data,
                "sentiment_distribution": self._get_portfolio_sentiment_distribution(sentiment_by_symbol)
            }
            
            # Save report to database
            self.db.sentiment_reports_collection.insert_one(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment report: {e}")
            return {"error": str(e)}
    
    def _get_portfolio_sentiment_distribution(self, sentiment_by_symbol: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of sentiment categories across portfolio.
        
        Args:
            sentiment_by_symbol: Sentiment data by symbol
            
        Returns:
            Dictionary with count of symbols in each category
        """
        distribution = {
            "very_negative": 0,
            "negative": 0,
            "neutral": 0,
            "positive": 0,
            "very_positive": 0
        }
        
        for symbol, data in sentiment_by_symbol.items():
            if isinstance(data, dict) and "category" in data:
                category = data["category"]
                if category in distribution:
                    distribution[category] += 1
        
        return distribution
    
    def get_correlations_with_price(self, symbol: str, days: int = 30) -> Dict[str, Any]:
        """
        Analyze correlation between sentiment and price movements.
        
        Args:
            symbol: Stock symbol
            days: Number of days for correlation analysis
            
        Returns:
            Dictionary containing correlation analysis
        """
        try:
            # Calculate start date
            start_date = datetime.now() - timedelta(days=days)
            
            # Get sentiment history
            sentiment_data = list(self.db.social_sentiment_collection.find(
                {
                    "symbol": symbol,
                    "timestamp": {"$gte": start_date}
                }
            ).sort("timestamp", 1))
            
            # Get price history
            price_data = list(self.db.market_data_collection.find(
                {
                    "symbol": symbol,
                    "timeframe": "day",
                    "timestamp": {"$gte": start_date}
                }
            ).sort("timestamp", 1))
            
            if not sentiment_data or not price_data:
                return {
                    "symbol": symbol,
                    "days": days,
                    "error": "Insufficient data for correlation analysis"
                }
            
            # Extract dates, sentiment scores, and prices
            sentiment_dates = [d["timestamp"].strftime("%Y-%m-%d") for d in sentiment_data]
            sentiment_scores = [d["sentiment"]["combined"].get("score", 0) if "combined" in d["sentiment"] else 0 
                               for d in sentiment_data]
            
            price_dates = [d["timestamp"].strftime("%Y-%m-%d") for d in price_data]
            closing_prices = [d["close"] for d in price_data]
            
            # Create DataFrames
            sentiment_df = pd.DataFrame({
                "date": pd.to_datetime(sentiment_dates),
                "sentiment": sentiment_scores
            })
            
            price_df = pd.DataFrame({
                "date": pd.to_datetime(price_dates),
                "close": closing_prices
            })
            
            # Merge data on date
            merged_df = pd.merge(sentiment_df, price_df, on="date", how="inner")
            
            if len(merged_df) < 5:  # Need at least 5 points for meaningful correlation
                return {
                    "symbol": symbol,
                    "days": days,
                    "error": "Insufficient matching data points for correlation analysis"
                }
            
            # Calculate correlations
            # Same-day correlation
            same_day_corr = merged_df["sentiment"].corr(merged_df["close"])
            
            # Next-day price correlation (sentiment leading price)
            merged_df["next_day_close"] = merged_df["close"].shift(-1)
            sentiment_lead_corr = merged_df["sentiment"].corr(merged_df["next_day_close"])
            
            # Previous-day sentiment correlation (price leading sentiment)
            merged_df["prev_day_sentiment"] = merged_df["sentiment"].shift(1)
            price_lead_corr = merged_df["close"].corr(merged_df["prev_day_sentiment"])
            
            # Calculate price changes
            merged_df["price_change"] = merged_df["close"].pct_change()
            
            # Correlation between sentiment and price changes
            sentiment_price_change_corr = merged_df["sentiment"].corr(merged_df["price_change"])
            
            # Next-day price change correlation
            merged_df["next_day_change"] = merged_df["price_change"].shift(-1)
            sentiment_next_day_change_corr = merged_df["sentiment"].corr(merged_df["next_day_change"])
            
            # Determine if sentiment is a leading indicator
            is_leading_indicator = abs(sentiment_next_day_change_corr) > abs(same_day_corr)
            
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
            
            return {
                "symbol": symbol,
                "days_analyzed": days,
                "data_points": len(merged_df),
                "same_day_correlation": {
                    "coefficient": same_day_corr,
                    "strength": get_correlation_strength(same_day_corr)
                },
                "sentiment_leading_price": {
                    "coefficient": sentiment_lead_corr,
                    "strength": get_correlation_strength(sentiment_lead_corr)
                },
                "price_leading_sentiment": {
                    "coefficient": price_lead_corr,
                    "strength": get_correlation_strength(price_lead_corr)
                },
                "sentiment_price_change_correlation": {
                    "coefficient": sentiment_price_change_corr,
                    "strength": get_correlation_strength(sentiment_price_change_corr)
                },
                "sentiment_next_day_change_correlation": {
                    "coefficient": sentiment_next_day_change_corr,
                    "strength": get_correlation_strength(sentiment_next_day_change_corr)
                },
                "is_leading_indicator": is_leading_indicator
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing correlation for {symbol}: {e}")
            return {"error": str(e)}

# Usage example
if __name__ == "__main__":
    # This would be used for testing only
    from pymongo import MongoClient
    
    # Example connection to MongoDB
    client = MongoClient("mongodb://localhost:27017/")
    db = client["automated_trading"]
    
    # Initialize analyzer
    sentiment_analyzer = SocialSentiment(db)
    
    # Analyze sentiment for a symbol
    result = sentiment_analyzer.analyze_symbol_sentiment("HDFCBANK", "HDFC Bank Ltd")
    print(json.dumps(result, default=str, indent=2))
    
    # Run daily collection
    # collection_results = sentiment_analyzer.run_daily_collection()
    # print(json.dumps(collection_results, default=str, indent=2))
    
    # Generate sentiment report
    # report = sentiment_analyzer.generate_sentiment_report()
    # print(json.dumps(report, default=str, indent=2))