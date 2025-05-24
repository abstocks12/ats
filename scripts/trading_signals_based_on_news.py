# news_trading_signals.py
import pymongo
from datetime import datetime, timedelta

class NewsTradingSignals:
    def __init__(self):
        self.client = pymongo.MongoClient("mongodb://localhost:27017/")
        self.db = self.client['trading_system']
        
    def get_trading_signals(self, minutes=15):
        """Get trading signals from recent news"""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        
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
            {"$match": {"trading_relevance.affected_symbols": {"$exists": True, "$ne": []}}},
            {"$sort": {"trading_relevance.score": -1}}
        ]
        
        signals = []
        for item in self.db.processed_news_collection.aggregate(pipeline):
            for symbol in item['trading_relevance']['affected_symbols']:
                signal = {
                    'symbol': symbol,
                    'action': 'BUY' if item['sentiment']['overall'] > 0.3 else 'SELL' if item['sentiment']['overall'] < -0.3 else 'HOLD',
                    'confidence': item['trading_relevance']['score'],
                    'reason': item['raw']['title'],
                    'news_type': item['classification']['news_type'],
                    'time': item['processed_date']
                }
                signals.append(signal)
                
        return signals

# Usage
signal_gen = NewsTradingSignals()
signals = signal_gen.get_trading_signals(30)  # Last 30 minutes

for signal in signals:
    print(f"\n🎯 {signal['symbol']}: {signal['action']} (Confidence: {signal['confidence']:.2f})")
    print(f"   Reason: {signal['reason'][:80]}...")