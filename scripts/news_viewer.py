# news_viewer.py
import pymongo
from datetime import datetime, timedelta
from tabulate import tabulate
import pandas as pd

class NewsViewer:
    def __init__(self, mongodb_uri="mongodb://localhost:27017/", db_name="trading_system"):
        self.client = pymongo.MongoClient(mongodb_uri)
        self.db = self.client[db_name]
        self.raw_news = self.db.raw_news_collection
        self.processed_news = self.db.processed_news_collection
        self.news_alerts = self.db.news_alerts_collection
        
    def show_latest_news(self, limit=10):
        """Show latest news items"""
        print("\n=== LATEST NEWS ===")
        
        pipeline = [
            {"$sort": {"scraped_date": -1}},
            {"$limit": limit},
            {"$lookup": {
                "from": "processed_news_collection",
                "localField": "_id",
                "foreignField": "raw_news_id",
                "as": "processed"
            }}
        ]
        
        news_items = list(self.raw_news.aggregate(pipeline))
        
        data = []
        for item in news_items:
            processed = item.get('processed', [{}])[0]
            sentiment = processed.get('sentiment', {}).get('overall', 0)
            relevance = processed.get('trading_relevance', {}).get('score', 0)
            
            data.append([
                item['scraped_date'].strftime('%H:%M:%S'),
                item['source'][:10],
                item['title'][:60] + '...',
                f"{sentiment:.2f}",
                f"{relevance:.2f}",
                item['priority']
            ])
            
        print(tabulate(data, 
                      headers=['Time', 'Source', 'Title', 'Sentiment', 'Relevance', 'Priority'],
                      tablefmt='grid'))
        
    def show_high_impact_news(self, hours=1):
        """Show high impact news from last N hours"""
        print(f"\n=== HIGH IMPACT NEWS (Last {hours} hours) ===")
        
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        
        pipeline = [
            {"$match": {"processed_date": {"$gte": cutoff}}},
            {"$match": {"trading_relevance.score": {"$gte": 0.7}}},
            {"$lookup": {
                "from": "raw_news_collection",
                "localField": "raw_news_id",
                "foreignField": "_id",
                "as": "raw"
            }},
            {"$unwind": "$raw"},
            {"$sort": {"trading_relevance.score": -1}}
        ]
        
        high_impact = list(self.processed_news.aggregate(pipeline))
        
        for item in high_impact[:10]:
            print(f"\n📰 {item['raw']['title']}")
            print(f"   Source: {item['raw']['source']}")
            print(f"   Sentiment: {item['sentiment']['overall']:.2f}")
            print(f"   Relevance: {item['trading_relevance']['score']:.2f}")
            print(f"   Impact: {item['trading_relevance']['expected_impact']}")
            if item['trading_relevance']['affected_symbols']:
                print(f"   Affected: {', '.join(item['trading_relevance']['affected_symbols'])}")
            print(f"   URL: {item['raw']['url']}")
            
    def show_alerts(self, limit=10):
        """Show recent alerts"""
        print(f"\n=== RECENT ALERTS ===")
        
        alerts = list(self.news_alerts.find().sort("created_date", -1).limit(limit))
        
        for alert in alerts:
            print(f"\n🚨 {alert['created_date'].strftime('%H:%M:%S')}")
            print(f"   {alert['message'][:200]}...")
            
    def show_market_sentiment(self):
        """Show overall market sentiment"""
        print("\n=== MARKET SENTIMENT ===")
        
        # Last 1 hour
        for hours in [1, 4, 24]:
            cutoff = datetime.utcnow() - timedelta(hours=hours)
            
            pipeline = [
                {"$match": {"processed_date": {"$gte": cutoff}}},
                {"$group": {
                    "_id": None,
                    "avg_sentiment": {"$avg": "$sentiment.overall"},
                    "total": {"$sum": 1},
                    "positive": {"$sum": {"$cond": [{"$gt": ["$sentiment.overall", 0.2]}, 1, 0]}},
                    "negative": {"$sum": {"$cond": [{"$lt": ["$sentiment.overall", -0.2]}, 1, 0]}},
                    "high_impact": {"$sum": {"$cond": [{"$gt": ["$trading_relevance.score", 0.7]}, 1, 0]}}
                }}
            ]
            
            result = list(self.processed_news.aggregate(pipeline))
            
            if result:
                data = result[0]
                sentiment_label = (
                    "🟢 Bullish" if data['avg_sentiment'] > 0.1
                    else "🔴 Bearish" if data['avg_sentiment'] < -0.1
                    else "🟡 Neutral"
                )
                
                print(f"\nLast {hours} hour(s):")
                print(f"  Sentiment: {sentiment_label} ({data['avg_sentiment']:.3f})")
                print(f"  Total News: {data['total']}")
                print(f"  Positive: {data['positive']} | Negative: {data['negative']}")
                print(f"  High Impact: {data['high_impact']}")
                
    def show_symbol_news(self, symbol):
        """Show news for a specific symbol"""
        print(f"\n=== NEWS FOR {symbol} ===")
        
        pipeline = [
            {"$match": {"trading_relevance.affected_symbols": symbol}},
            {"$lookup": {
                "from": "raw_news_collection",
                "localField": "raw_news_id",
                "foreignField": "_id",
                "as": "raw"
            }},
            {"$unwind": "$raw"},
            {"$sort": {"processed_date": -1}},
            {"$limit": 10}
        ]
        
        news_items = list(self.processed_news.aggregate(pipeline))
        
        if not news_items:
            print(f"No news found for {symbol}")
            return
            
        for item in news_items:
            print(f"\n📰 {item['raw']['title']}")
            print(f"   Time: {item['processed_date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Sentiment: {item['sentiment']['overall']:.2f}")
            print(f"   Type: {item['classification']['news_type']}")
            
    def show_statistics(self):
        """Show collection statistics"""
        print("\n=== COLLECTION STATISTICS ===")
        
        total_raw = self.raw_news.count_documents({})
        total_processed = self.processed_news.count_documents({})
        total_alerts = self.news_alerts.count_documents({})
        
        print(f"\nTotal News Collected: {total_raw}")
        print(f"Total Processed: {total_processed}")
        print(f"Total Alerts Generated: {total_alerts}")
        
        # News by source
        print("\n📊 News by Source:")
        pipeline = [
            {"$group": {"_id": "$source", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        
        sources = list(self.raw_news.aggregate(pipeline))
        for source in sources:
            print(f"   {source['_id']}: {source['count']}")
            
        # News by priority
        print("\n📊 News by Priority:")
        pipeline = [
            {"$group": {"_id": "$priority", "count": {"$sum": 1}}},
            {"$sort": {"_id": 1}}
        ]
        
        priorities = list(self.raw_news.aggregate(pipeline))
        for priority in priorities:
            print(f"   {priority['_id']}: {priority['count']}")
            
    def search_news(self, keyword):
        """Search news by keyword"""
        print(f"\n=== SEARCH RESULTS FOR '{keyword}' ===")
        
        regex_pattern = {"$regex": keyword, "$options": "i"}
        
        results = list(self.raw_news.find({
            "$or": [
                {"title": regex_pattern},
                {"description": regex_pattern}
            ]
        }).sort("scraped_date", -1).limit(10))
        
        if not results:
            print(f"No news found containing '{keyword}'")
            return
            
        for item in results:
            print(f"\n📰 {item['title']}")
            print(f"   Source: {item['source']}")
            print(f"   Time: {item['scraped_date'].strftime('%Y-%m-%d %H:%M')}")
            print(f"   Priority: {item['priority']}")


if __name__ == "__main__":
    viewer = NewsViewer()
    
    while True:
        print("\n" + "="*50)
        print("NEWS VIEWER MENU")
        print("="*50)
        print("1. Show Latest News")
        print("2. Show High Impact News")
        print("3. Show Recent Alerts")
        print("4. Show Market Sentiment")
        print("5. Show News for Symbol")
        print("6. Show Statistics")
        print("7. Search News")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ")
        
        if choice == '1':
            viewer.show_latest_news(20)
        elif choice == '2':
            hours = int(input("Enter hours to look back (default 1): ") or "1")
            viewer.show_high_impact_news(hours)
        elif choice == '3':
            viewer.show_alerts(10)
        elif choice == '4':
            viewer.show_market_sentiment()
        elif choice == '5':
            symbol = input("Enter symbol (e.g., TATASTEEL): ").upper()
            viewer.show_symbol_news(symbol)
        elif choice == '6':
            viewer.show_statistics()
        elif choice == '7':
            keyword = input("Enter search keyword: ")
            viewer.search_news(keyword)
        elif choice == '8':
            print("Exiting...")
            break
        else:
            print("Invalid choice! Please try again.")