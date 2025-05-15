"""
News data models for the Automated Trading System.
Defines the structure for news data collections.
"""

from datetime import datetime
from typing import List, Dict, Any, Optional, Union
import hashlib

class NewsItem:
    """News item model for storing news articles"""
    
    def __init__(self, title: str, description: Optional[str] = None, source: Optional[str] = None,
                 url: Optional[str] = None, published_date: Optional[datetime] = None,
                 sentiment: Optional[str] = None, sentiment_score: Optional[float] = None,
                 categories: Optional[List[str]] = None, symbols: Optional[List[str]] = None):
        """
        Initialize news item model
        
        Args:
            title (str): News title
            description (str, optional): News description or content
            source (str, optional): News source
            url (str, optional): URL to the news article
            published_date (datetime, optional): Publication date
            sentiment (str, optional): Sentiment classification ('positive', 'negative', 'neutral')
            sentiment_score (float, optional): Sentiment score (-1.0 to 1.0)
            categories (list, optional): List of categories
            symbols (list, optional): List of related stock symbols
        """
        self.title = title
        self.description = description or ""
        self.source = source or "Unknown"
        self.url = url or ""
        self.published_date = published_date or datetime.now()
        self.sentiment = sentiment or "neutral"
        self.sentiment_score = sentiment_score or 0.0
        self.categories = categories or []
        self.symbols = symbols or []
        self.scraped_at = datetime.now()
        self.hash = self._generate_hash()
        self.created_at = datetime.now()
        self.updated_at = self.created_at
    
    def _generate_hash(self) -> str:
        """Generate a unique hash for the news item"""
        content = f"{self.title}|{self.source}|{self.published_date.isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert news item to dictionary"""
        return {
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "url": self.url,
            "published_date": self.published_date,
            "sentiment": self.sentiment,
            "sentiment_score": self.sentiment_score,
            "categories": self.categories,
            "symbols": self.symbols,
            "scraped_at": self.scraped_at,
            "hash": self.hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsItem':
        """Create news item from dictionary"""
        return cls(
            title=data["title"],
            description=data.get("description", ""),
            source=data.get("source", "Unknown"),
            url=data.get("url", ""),
            published_date=data.get("published_date", datetime.now()),
            sentiment=data.get("sentiment", "neutral"),
            sentiment_score=data.get("sentiment_score", 0.0),
            categories=data.get("categories", []),
            symbols=data.get("symbols", [])
        )
    
    @classmethod
    def from_pulse_scraper(cls, pulse_data: Dict[str, Any]) -> 'NewsItem':
        """Create news item from Pulse scraper output"""
        # Extract publication date
        published_date = None
        if "timestamp" in pulse_data:
            try:
                published_date = datetime.fromisoformat(pulse_data["timestamp"])
            except (ValueError, TypeError):
                try:
                    # Try to parse with different format
                    published_date = datetime.strptime(pulse_data["timestamp"], "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    published_date = datetime.now()
        
        # Extract sentiment
        sentiment = pulse_data.get("sentiment", "neutral")
        sentiment_score = 0.0
        
        if sentiment == "positive":
            sentiment_score = 0.7
        elif sentiment == "negative":
            sentiment_score = -0.7
        
        # Extract categories and symbols
        categories = pulse_data.get("categories", [])
        
        # Extract symbols from categories (if any category is a stock symbol)
        symbols = []
        for category in categories:
            if category.isupper() and len(category) <= 10:
                symbols.append(category)
        
        return cls(
            title=pulse_data["title"],
            description=pulse_data.get("description", ""),
            source=pulse_data.get("source", "Unknown"),
            url=pulse_data.get("link", ""),
            published_date=published_date,
            sentiment=sentiment,
            sentiment_score=sentiment_score,
            categories=categories,
            symbols=symbols
        )
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.title} - {self.source} ({self.published_date.strftime('%Y-%m-%d')})"