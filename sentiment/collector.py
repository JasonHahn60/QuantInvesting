"""
Sentiment data collector for news articles and social media.
"""
import logging
from typing import Dict, List
from datetime import datetime
from textblob import TextBlob
import pandas as pd

logger = logging.getLogger(__name__)

class SentimentCollector:
    """Collects and analyzes sentiment data."""
    
    def __init__(self):
        """Initialize the sentiment collector."""
        pass
    
    def analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a text."""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            return {'polarity': 0, 'subjectivity': 0, 'timestamp': datetime.now()}
    
    def analyze_articles(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment of multiple articles."""
        results = []
        for article in articles:
            sentiment = self.analyze_text(article.get('content', ''))
            results.append({
                'title': article.get('title'),
                'source': article.get('source'),
                'date': article.get('date'),
                'sentiment': sentiment
            })
        return results
    
    def analyze_tweets(self, tweets: List[Dict]) -> List[Dict]:
        """Analyze sentiment of multiple tweets."""
        results = []
        for tweet in tweets:
            sentiment = self.analyze_text(tweet.get('text', ''))
            results.append({
                'id': tweet.get('id'),
                'author': tweet.get('author'),
                'date': tweet.get('date'),
                'sentiment': sentiment
            })
        return results 