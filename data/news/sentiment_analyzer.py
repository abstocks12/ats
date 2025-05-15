"""
Sentiment Analyzer Module for the Automated Trading System.
Analyzes sentiment of news articles and other text.
"""

import os
import sys
import re
from typing import List, Dict, Any, Optional, Union, Tuple
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import settings
from utils.logging_utils import setup_logger, log_error

class SentimentAnalyzer:
    """
    Analyzes sentiment of text using NLTK's VADER sentiment analyzer.
    """
    
    def __init__(self):
        """Initialize sentiment analyzer"""
        self.logger = setup_logger(__name__)
        
        # Set up NLTK
        self._setup_nltk()
        
        # Initialize VADER sentiment analyzer
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Load financial domain-specific lexicon
        self.financial_words = self._load_financial_lexicon()
    
    def _setup_nltk(self) -> None:
        """Set up NLTK and download required resources"""
        try:
            # Check if VADER lexicon exists
            try:
                nltk.data.find('sentiment/vader_lexicon.zip')
            except LookupError:
                nltk.download('vader_lexicon')
            
            # Check if punkt tokenizer exists
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
        except Exception as e:
            log_error(e, context={"action": "setup_nltk"})
            self.logger.warning("Failed to download NLTK resources. Sentiment analysis may be less accurate.")
    
    def _load_financial_lexicon(self) -> Dict[str, float]:
        """
        Load financial domain-specific lexicon
        
        Returns:
            dict: Dictionary of words and their sentiment scores
        """
        # Default financial words with sentiment scores
        financial_words = {
            # Positive financial terms
            'bull': 0.6,
            'bullish': 0.7,
            'outperform': 0.6,
            'upgrade': 0.6,
            'profit': 0.7,
            'growth': 0.6,
            'up': 0.3,
            'rises': 0.5,
            'gain': 0.5,
            'beat': 0.6,
            'exceeded': 0.6,
            'dividend': 0.4,
            'rally': 0.7,
            'expand': 0.4,
            'improvement': 0.5,
            'recovery': 0.4,
            'momentum': 0.3,
            'surge': 0.6,
            'upside': 0.5,
            'breakthrough': 0.7,
            
            # Negative financial terms
            'bear': -0.6,
            'bearish': -0.7,
            'underperform': -0.6,
            'downgrade': -0.6,
            'loss': -0.7,
            'decline': -0.5,
            'down': -0.3,
            'falls': -0.5,
            'miss': -0.6,
            'missed': -0.6,
            'debt': -0.3,
            'correction': -0.4,
            'crash': -0.9,
            'default': -0.8,
            'recession': -0.8,
            'bankrupt': -0.9,
            'bankruptcy': -0.9,
            'downside': -0.5,
            'risk': -0.4,
            'volatile': -0.4,
            'volatility': -0.4,
            'margin': 0.1,  # Neutral or slightly positive
            'warning': -0.6,
            'investigation': -0.6,
            'sue': -0.7,
            'sued': -0.7,
            'lawsuit': -0.7,
            'settlement': -0.3,
            'penalty': -0.6,
            'fine': -0.6,
            'probe': -0.5,
            
            # Banking specific terms
            'npa': -0.7,
            'restructure': -0.4,
            'deposit': 0.3,
            'deposits': 0.3,
            'loan': 0.1,
            'provisions': -0.5,
            'write-off': -0.7,
            'capital': 0.4,
            'liquidity': 0.4,
            'credit growth': 0.5,
            
            # Market specific terms
            'inflation': -0.3,
            'rate hike': -0.5,
            'rate cut': 0.5,
            'fed': 0.0,  # Neutral
            'fomc': 0.0,  # Neutral
            'rbi': 0.0,  # Neutral
            'policy': 0.0,  # Neutral
            'monetary policy': 0.0,  # Neutral
            'stimulus': 0.6,
            'slowdown': -0.5,
            'slowing': -0.4,
            'contraction': -0.6,
            'recession fears': -0.7,
            'outlook': 0.0,  # Neutral
            'guidance': 0.0,  # Neutral
            'forecast': 0.0,  # Neutral
            'target price': 0.3,
            'target raised': 0.6,
            'target lowered': -0.6,
            'overweight': 0.6,
            'underweight': -0.6,
            'neutral': 0.0,  # Neutral
            'hold': 0.0,  # Neutral
            'buy': 0.7,
            'sell': -0.7,
            'strong buy': 0.9,
            'strong sell': -0.9
        }
        
        # Try to load custom lexicon from file if available
        try:
            financial_lexicon_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'data',
                'financial_lexicon.csv'
            )
            
            if os.path.exists(financial_lexicon_path):
                df = pd.read_csv(financial_lexicon_path)
                
                for _, row in df.iterrows():
                    word = row.get('word', '')
                    score = row.get('score', 0.0)
                    
                    if word and isinstance(score, (int, float)):
                        financial_words[word] = score
        
        except Exception as e:
            log_error(e, context={"action": "load_financial_lexicon_from_file"})
        
        return financial_words
    
    def analyze_sentiment(self, title: str, content: str = "") -> Tuple[str, float]:
        """
        Analyze sentiment of a news article
        
        Args:
            title (str): Article title
            content (str, optional): Article content
            
        Returns:
            tuple: (sentiment_label, sentiment_score)
        """
        # Combine title and content with title having higher weight
        text = title + " " + title + " " + content
        
        try:
            # Get VADER sentiment scores
            scores = self.analyzer.polarity_scores(text)
            
            # Apply financial domain adjustments
            scores = self._apply_financial_adjustments(text, scores)
            
            # Get compound score
            compound_score = scores['compound']
            
            # Determine sentiment label
            if compound_score >= 0.25:
                sentiment_label = "positive"
            elif compound_score <= -0.25:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            # Normalize score to 0-1 range for storage
            normalized_score = (compound_score + 1) / 2
            
            return sentiment_label, normalized_score
            
        except Exception as e:
            log_error(e, context={"action": "analyze_sentiment", "text": title})
            
            # Return neutral sentiment as fallback
            return "neutral", 0.5
    
    def _apply_financial_adjustments(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """
        Apply financial domain-specific adjustments to sentiment scores
        
        Args:
            text (str): Text being analyzed
            scores (dict): VADER sentiment scores
            
        Returns:
            dict: Adjusted sentiment scores
        """
        # Lowercase text for comparison
        text_lower = text.lower()
        
        # Tokenize text
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Check for financial domain terms
        for word in words:
            if word in self.financial_words:
                adjustment = self.financial_words[word]
                
                # Add to appropriate score component
                if adjustment > 0:
                    scores['pos'] += adjustment
                elif adjustment < 0:
                    scores['neg'] -= adjustment
        
        # Re-calculate compound score
        pos = scores['pos']
        neg = scores['neg']
        neu = scores['neu']
        
        # Normalize to range [-1, 1]
        total = pos + neg + neu
        if total > 0:
            normalized_pos = pos / total
            normalized_neg = neg / total
            
            # Adjust compound score (VADER algorithm approximation)
            compound = normalized_pos - normalized_neg
            
            # Apply nonlinear scaling (similar to VADER's normalization)
            alpha = 15  # VADER's normalization parameter
            compound = compound / (compound**2 + alpha)
            
            scores['compound'] = compound
        
        # Ensure compound is in range [-1, 1]
        scores['compound'] = max(-1.0, min(1.0, scores['compound']))
        
        return scores
    
    def analyze_sector_sentiment(self, news_items: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Analyze sentiment for news items by sector
        
        Args:
            news_items (list): List of news items
            
        Returns:
            dict: Sector sentiment scores
        """
        sector_sentiments = {}
        sector_counts = {}
        
        for item in news_items:
            # Skip items without categories
            if 'categories' not in item:
                continue
            
            categories = item.get('categories', [])
            sentiment_score = item.get('sentiment_score', 0.5)
            
            # Normalize sentiment score to -1 to +1 range
            normalized_score = (sentiment_score * 2) - 1
            
            for category in categories:
                if category not in sector_sentiments:
                    sector_sentiments[category] = 0.0
                    sector_counts[category] = 0
                
                sector_sentiments[category] += normalized_score
                sector_counts[category] += 1
        
        # Calculate average sentiment for each sector
        for sector, total in sector_sentiments.items():
            count = sector_counts.get(sector, 1)  # Avoid division by zero
            sector_sentiments[sector] = total / count
        
        return sector_sentiments
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extract named entities from text
        
        Args:
            text (str): Text to analyze
            
        Returns:
            list: List of extracted entities
        """
        entities = []
        
        try:
            # Check if required NLTK packages are available
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('taggers/averaged_perceptron_tagger')
                nltk.data.find('chunkers/maxent_ne_chunker')
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('punkt')
                nltk.download('averaged_perceptron_tagger')
                nltk.download('maxent_ne_chunker')
                nltk.download('words')
            
            # Tokenize text
            tokens = nltk.word_tokenize(text)
            
            # Part-of-speech tagging
            tagged = nltk.pos_tag(tokens)
            
            # Named entity recognition
            ne_tree = nltk.ne_chunk(tagged)
            
            # Extract entities
            for subtree in ne_tree:
                if isinstance(subtree, nltk.Tree):
                    entity_type = subtree.label()
                    entity_text = " ".join([word for word, tag in subtree.leaves()])
                    
                    if entity_type in ['ORGANIZATION', 'PERSON', 'GPE']:
                        entities.append(entity_text)
            
        except Exception as e:
            log_error(e, context={"action": "extract_entities", "text": text[:100]})
        
        return list(set(entities))  # Remove duplicates
    
    def extract_keywords(self, text: str, num_keywords: int = 5) -> List[str]:
        """
        Extract keywords from text
        
        Args:
            text (str): Text to analyze
            num_keywords (int): Number of keywords to extract
            
        Returns:
            list: List of keywords
        """
        keywords = []
        
        try:
            # Check if required NLTK packages are available
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            from nltk.corpus import stopwords
            from nltk.tokenize import word_tokenize
            
            # Get stopwords
            stop_words = set(stopwords.words('english'))
            
            # Add custom stopwords for financial news
            custom_stopwords = {
                'said', 'says', 'according', 'reported', 'company', 'companies',
                'market', 'markets', 'share', 'shares', 'stock', 'stocks',
                'year', 'month', 'day', 'percent', 'reuters', 'news'
            }
            
            stop_words.update(custom_stopwords)
            
            # Tokenize text
            tokens = word_tokenize(text.lower())
            
            # Remove stopwords and non-alphabetic tokens
            filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
            
            # Count word frequencies
            from collections import Counter
            word_counts = Counter(filtered_tokens)
            
            # Get most common words
            keywords = [word for word, count in word_counts.most_common(num_keywords)]
            
        except Exception as e:
            log_error(e, context={"action": "extract_keywords", "text": text[:100]})
        
        return keywords
    
    def create_entity_graph(self, news_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create entity relationship graph from news items
        
        Args:
            news_items (list): List of news items
            
        Returns:
            dict: Entity graph
        """
        entity_connections = {}
        entity_sentiments = {}
        entity_counts = {}
        
        for item in news_items:
            # Skip items without title
            if 'title' not in item:
                continue
            
            text = item['title']
            if 'description' in item and item['description']:
                text += " " + item['description']
            
            # Extract entities
            entities = self.extract_entities(text)
            
            # Skip if no entities found
            if not entities:
                continue
            
            # Update entity counts
            for entity in entities:
                if entity not in entity_counts:
                    entity_counts[entity] = 0
                entity_counts[entity] += 1
            
            # Connect co-occurring entities
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity1 = entities[i]
                    entity2 = entities[j]
                    
                    # Create connection if it doesn't exist
                    if entity1 not in entity_connections:
                        entity_connections[entity1] = {}
                    
                    if entity2 not in entity_connections[entity1]:
                        entity_connections[entity1][entity2] = 0
                    
                    # Increment connection count
                    entity_connections[entity1][entity2] += 1
                    
                    # Mirror connection
                    if entity2 not in entity_connections:
                        entity_connections[entity2] = {}
                    
                    if entity1 not in entity_connections[entity2]:
                        entity_connections[entity2][entity1] = 0
                    
                    entity_connections[entity2][entity1] += 1
            
            # Add sentiment
            sentiment_score = item.get('sentiment_score', 0.5)
            
            # Normalize to -1 to +1 range
            normalized_score = (sentiment_score * 2) - 1
            
            for entity in entities:
                if entity not in entity_sentiments:
                    entity_sentiments[entity] = 0.0
                
                entity_sentiments[entity] += normalized_score
        
        # Calculate average sentiment for each entity
        for entity, total in entity_sentiments.items():
            count = entity_counts.get(entity, 1)  # Avoid division by zero
            entity_sentiments[entity] = total / count
        
        # Construct graph data
        nodes = []
        links = []
        
        for entity, connections in entity_connections.items():
            # Only include entities that appear multiple times
            if entity_counts.get(entity, 0) < 2:
                continue
            
            # Add node
            nodes.append({
                'id': entity,
                'name': entity,
                'count': entity_counts.get(entity, 0),
                'sentiment': entity_sentiments.get(entity, 0)
            })
            
            # Add links
            for target, weight in connections.items():
                # Only include entities that appear multiple times
                if entity_counts.get(target, 0) < 2:
                    continue
                
                links.append({
                    'source': entity,
                    'target': target,
                    'value': weight
                })
        
        return {
            'nodes': nodes,
            'links': links
        }